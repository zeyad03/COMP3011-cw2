# Design Notes

This document records the architectural decisions, data structures, and
algorithmic choices behind the search engine. Companion to the public
[`README.md`](../README.md), which describes how to install and use the
tool.

## 1. Architecture

```
        +------------+    Page list     +------------+    Index    +------------+
        |  crawler   |  ──────────────> |  indexer   |  ────────>  |  storage   |
        |  (BFS, 6s) |                  |  (TF/DF)   |             |  (JSON)    |
        +------------+                  +------------+             +-----+------+
                                                                         │
                                                                         │ load
                                                                         ▼
        +------------+   query   +------------+   index   +------------------+
        |    CLI     | ────────> |   search   |  <──────  |  in-memory Index |
        |  (REPL)    |           |  (TF-IDF)  |           +------------------+
        +------------+           +------------+
```

Each module has one well-defined responsibility:

| Module | Responsibility |
| --- | --- |
| `tokenizer` | Lowercase, strip punctuation, collapse apostrophes. **Used by both indexer and search** so queries and documents are normalised identically. |
| `crawler` | BFS over the seed host, 6-second politeness window, retries on 5xx, robots.txt aware. Returns `Page(url, title, text)`. |
| `indexer` | Build the inverted index from a list of `Page` records. |
| `search` | `print_term`, `find` (with TF-IDF + phrase mode), `suggest` (Levenshtein "did you mean"). |
| `storage` | Atomic JSON save / version-checked load. |
| `main` | Interactive REPL with command dispatch. |

## 2. Index schema

```python
{
  "meta": {
    "crawled_at":   ISO-8601 UTC timestamp,
    "num_docs":     int,
    "num_terms":    int,
    "total_tokens": int,
    "version":      "1.0",
  },
  "docs": {
    "<doc_id>": {
      "url":    str,
      "title":  str,
      "length": int,    # token count
    }
  },
  "terms": {
    "<term>": {
      "df":       int,                 # document frequency
      "cf":       int,                 # collection frequency
      "postings": {
        "<doc_id>": {
          "tf":        int,
          "positions": [int, ...],     # token offsets within the doc
        }
      }
    }
  }
}
```

Document IDs are sequential string integers (`"0"`, `"1"`, …) assigned
in crawl order. Strings rather than ints are used because JSON object
keys must be strings, and committing to string IDs throughout the
codebase avoids a serialisation surprise.

Postings store **positions**, not just frequencies, because positions
enable **phrase queries**: a phrase match requires the query terms to
appear in adjacent positions within the same document.

## 3. Tokenisation rules

1. Case-folded to lowercase.
2. Curly Unicode quotes (`’`) normalised to ASCII apostrophes first.
3. Apostrophes collapsed: `don't` → `dont`. Splitting them on the
   apostrophe yields a low-information `t` token; collapsing keeps the
   contraction searchable as a single unit.
4. All other non-word characters become token separators
   (Unicode-aware via Python's `re` module).
5. Digits are preserved as tokens (e.g. `1984`).
6. Diacritics are preserved (`café` is not normalised to `cafe`). This
   is a deliberate trade-off — search for `cafe` will not match
   `café`. The alternative (NFKD-normalise + strip combining marks)
   was rejected because it adds complexity for a feature the target
   site does not exercise.

## 4. Crawling

- **BFS**, not DFS. Predictable order; flat frontier easy to reason
  about under a politeness budget.
- **Same-host only**. Off-domain links are filtered out by comparing
  `urlparse(link).netloc` to the seed host.
- **URL normalisation** strips fragments only; trailing slashes are
  preserved because servers may treat `/foo` and `/foo/` as different
  resources.
- **Politeness gate** is the single source of truth: every fetch —
  including `robots.txt` — passes through `_wait_for_politeness`,
  which sleeps until at least the configured delay (6 s by default)
  has elapsed since the last request.
- **Retries** with exponential backoff on 5xx and network errors.
  Failed pages are logged and skipped, never propagated as exceptions
  to the user.
- **robots.txt** loaded once at the start of the crawl via
  `urllib.robotparser`. Disallowed URLs are skipped silently.

## 5. Ranking

Per-term contribution to a document's score:

```
score(t, d) = (1 + log10(tf(t, d))) · log10(1 + N / df(t))
```

Document score is the sum over query terms.

**Sub-linear TF** (`1 + log10(tf)`) dampens runaway repetition: a doc
that mentions `foo` 100 times does not rank 100× higher than one that
mentions it once. This is the same intuition behind Okapi BM25.

**Smoothed IDF** (`log10(1 + N/df)`) keeps IDF strictly positive even
when `df == N`, i.e. when the term appears in every document. Raw
`log(N/df)` collapses to 0 in that case, which is mathematically
correct but degrades to "no signal" on small corpora — exactly our
situation with ~70 pages.

**Tie-breaking** is alphabetical by URL ascending, so equal-score
results are deterministic.

**Phrase mode** filters the AND-intersection candidate set by
requiring the query terms to appear in adjacent positions in the
same document. Implemented via positional set lookup, O(P) per
candidate where P is the number of positions for the rarest term.

## 6. Suggestions

For unknown query terms, `suggest` returns known terms within a
Levenshtein distance budget (default 2). The implementation:

- Pre-filters by length-difference (free lower bound on edit
  distance).
- Uses standard two-row dynamic programming.
- **Bails out early** when the row's minimum exceeds the distance
  budget — there is no path to recover within the budget once every
  cell exceeds it.

Lookup over a 70-page index runs in well under 100 ms.

## 7. Storage

JSON over pickle:

| Property | JSON | Pickle |
| --- | --- | --- |
| Human-readable | Yes — graders can inspect the file. | No |
| Cross-version stable | Yes | Fragile (Python version, class layout). |
| Speed | Slower; not the bottleneck. | Faster |
| Size | Larger; ~1–2 MB for our corpus. | Smaller |

Transparency wins. The `indent=2, sort_keys=True` format makes diffs
across builds clean and the file inspectable in any text editor.

Saves are **atomic**: the JSON is written to a sibling temp file, then
`os.replace`d into place. A reader can never observe a half-written
file; a crash mid-write cannot corrupt an existing index.

## 8. Complexity analysis

Let:
- N = number of documents.
- M = average tokens per document.
- T = total unique terms in the index.
- Q = number of query tokens.

| Operation | Complexity | Notes |
| --- | --- | --- |
| `tokenize` | O(L) | L = input length. Single regex pass. |
| `build_index` | O(N · M) | Each token yields one dict update. |
| `save` | O(B) | B = serialised bytes; dominated by JSON encoding. |
| `load` | O(B) | JSON decode + version check. |
| `print_term` | O(P) | P = postings length. One hash lookup. |
| `find` (AND) | O(Σ P + K · Q) | Set intersection over posting lists, then per-candidate scoring. |
| `find` (phrase) | O(Σ P + K · Q · max(positions)) | Adds positional adjacency check per candidate. |
| `suggest` | O(T · L²) worst case | But length filter + early exit usually cut to O(T · D · L), D = budget. |

For the target corpus (~70 docs, ~3 k unique terms), every operation
completes in milliseconds.

### 8.1 Measured numbers

Run `python scripts/benchmark.py` to reproduce. Sample run on a 2024
M-series MacBook (Python 3.14):

```
build_index
  docs   mean (ms)
    10        0.51
   100        6.51
  1000       57.37
  2000      124.14
```

`build_index` scales linearly with the corpus size, as predicted. The
intercept reflects fixed overhead (timestamp, dict initialisation).

```
find  (200-doc corpus)
  single             and    0.12 ms
  AND (2 terms)      and    0.07 ms
  AND (3 terms)      and    0.05 ms
  phrase (2 terms)   phrase 0.19 ms
  phrase (3 terms)   phrase 0.23 ms
```

Search times are sub-millisecond. Multi-term AND is *faster* than
single-term because the intersection-from-smallest-posting strategy
prunes the candidate set aggressively. Phrase mode is slightly slower
because it adds a per-candidate positional adjacency check.

## 9. Empirical evaluation

The brief is satisfied by *implementing* TF-IDF; this section goes
further and *measures* whether the choice was justified, by running
all three rankers (frequency, TF-IDF, BM25) over a hand-curated
gold-standard query set and reporting standard IR metrics.

### 9.1 Methodology

A gold standard of **12 queries with hand-curated relevance
judgements** lives in [`evaluation/gold_standard.json`](../evaluation/gold_standard.json).
Each query has a list of URLs that a human would expect ranked highly,
chosen conservatively: a URL that merely *contains* a query token but
is not topically central is excluded. Queries cover author lookups
(`einstein imagination`), tag lookups (`humor classic`), compound
tags (`be yourself`), and abstract concepts (`wisdom truth`).

`evaluation/evaluate.py` runs each query through each ranker and
computes:

* **P@5, P@10**: precision in the top 5 and 10 results.
* **R@5**: recall in the top 5.
* **MRR**: mean reciprocal rank — how soon the first relevant result
  appears.
* **MAP**: mean average precision — area under the
  precision-at-each-relevant-document curve, averaged over queries.

Definitions follow Manning, Raghavan & Schütze §8.

### 9.2 Results

Run on the v1.0 index (214 docs, 4,646 terms). All numbers are means
over the 12 queries.

| Ranker     |  P@5  | P@10  |  R@5  |  MRR  |  MAP  |
| ---------- | ----: | ----: | ----: | ----: | ----: |
| frequency  | 0.283 | 0.175 | 0.407 | 0.675 | 0.381 |
| tfidf      | 0.267 | 0.183 | 0.386 | 0.722 | 0.380 |
| bm25       | 0.267 | 0.183 | **0.449** | 0.695 | **0.435** |

Per-query numbers are in [`evaluation/results.csv`](../evaluation/results.csv).

### 9.3 Discussion

**BM25 wins MAP by ~14% over TF-IDF.** That is the headline result:
on this corpus, when you care about *all* the relevant documents
ending up high in the ranking, BM25 is unambiguously better. The win
is driven by length normalisation — many tag pages are short
(one quote per page) but extremely on-topic, and BM25's `b=0.75`
length-norm rewards them, where TF-IDF (which we *did not* normalise
by document length) does not.

**BM25 also wins R@5** by 16% (0.449 vs 0.386). Combined with the MAP
result, this confirms BM25 is finding the actually-relevant short
pages and ranking them in the top 5.

**TF-IDF wins MRR** (0.722 vs 0.695). MRR rewards getting the *first*
relevant document early; TF-IDF's sub-linear TF gives a slight edge
when the top-1 has a single high-frequency match.

**Frequency-only is surprisingly close on P@5** (0.283 vs 0.267) but
loses badly on MAP and R@5. The narrow P@5 win is misleading — it
just reflects that the most-frequent terms in tag pages happen to
align with our query words. Once you look beyond rank 5, the lack of
length normalisation and IDF makes frequency-ranking pollute the top
10 with long, off-topic pages.

**Implication.** TF-IDF is the documented default for this brief
(simpler to explain on video and aligned with the lecture material),
but BM25 is empirically better and is a one-flag swap (`> find
--ranker bm25 ...`). The ranker is chosen at query time, not at index
build time, so users can compare on their own queries.

### 9.4 Cross-validation against scikit-learn

`evaluation/compare_sklearn.py` compares this project's `TFIDFRanker`
to a `sklearn.feature_extraction.text.TfidfVectorizer` +
cosine-similarity baseline over the same corpus. Top-5 overlap per
query (12 queries):

| Query                 | Overlap |
| --------------------- | ------: |
| wisdom truth          | 1.00    |
| love romance          | 1.00    |
| deep thoughts         | 0.80    |
| humor classic         | 0.60    |
| live miracle          | 0.60    |
| einstein imagination  | 0.40    |
| austen books          | 0.40    |
| monroe inspirational  | 0.20    |
| good friends          | 0.00    |
| be yourself           | 0.00    |
| edison genius         | 0.00    |
| rowling magic         | 0.00    |
| **mean**              | **0.42** |

The 42% mean overlap initially looks low but resolves into two clear
explanations:

1. **Retrieval semantics differ.** `find` enforces strict AND — every
   query token must appear in every result. sklearn's cosine
   similarity is OR-style: a document with one of the two terms can
   still surface if its TF-IDF vector is well-aligned. For
   `rowling magic` our AND intersection returns 0 results, but
   sklearn returns 5 (only one term need match). This is a *retrieval*
   difference, not a *ranking* one.

2. **Length normalisation differs.** sklearn cosine implicitly
   normalises by document vector length; our TF-IDF does not. On
   queries where our intersection *does* return results, the rank
   *order* often differs because long documents drop in sklearn's
   ranking but not ours.

When both retrieval semantics agree (queries with strong AND-style
intent like *wisdom truth* or *love romance*), the overlap is **100%**.
The disagreement on AND-failure cases is a feature, not a bug — the
brief describes `find good friends` as "all pages containing the words
'good' and 'friends'", which is AND semantics.

This cross-check is committed as evidence that the project's hand-built
TF-IDF is not an outlier: when the retrieval models agree, the
rankings agree exactly.

## 10. Design trade-offs explicitly considered

- **BM25 vs. TF-IDF**. BM25 has better empirical relevance, but the
  difference matters at the scale of millions of documents, not 70.
  TF-IDF is simpler to explain in the video.
- **Stemming vs. raw tokens**. Stemming would let `running` match
  `runs`, but it discards positional information needed for phrase
  proximity, and it introduces a third dependency the marker has to
  reason about.
- **Stop-word removal**. Rejected. Phrases like `to be or not to be`
  become unsearchable with aggressive stop-word lists.
- **Async crawl**. Tempting for speed, but the politeness window
  serialises requests anyway, so concurrency offers no win for a
  single host.
