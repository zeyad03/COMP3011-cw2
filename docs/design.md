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

## 9. Design trade-offs explicitly considered

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
