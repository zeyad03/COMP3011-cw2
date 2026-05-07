# Search Engine Tool

A Python command-line search engine that crawls
[quotes.toscrape.com](https://quotes.toscrape.com), builds a
**positional inverted index** of every word it sees, and answers
single-, multi-word, **boolean (`AND`/`OR`/`NOT`)**, and **phrase**
queries with five interchangeable rankers (frequency, TF-IDF, BM25,
BM25F, and an optional dense / hybrid / learning-to-rank pipeline).

> Coursework 2 for **COMP3011 — Web Services and Web Data**, University of Leeds.

**Current release:** [`v2.0.0`](https://github.com/zeyad03/COMP3011-cw2/releases/tag/v2.0.0)
— hybrid retrieval, BM25F, boolean queries, learning-to-rank, VByte
posting compression. See the [Changelog](#changelog) at the bottom.

## Highlights

- **Polite crawler**: BFS over the seed host with a strict 6-second
  request gap, robots.txt support, exponential-backoff retries, and
  content-hash dedup of alias URLs (e.g. `/tag/love/` and
  `/tag/love/page/1/`).
- **Inverted index** with term frequency, document frequency, and
  per-field positions (title vs body) per posting — positions enable
  phrase queries and BM25F field weighting. Schema is versioned;
  v1.0, v1.1, and v1.2 indexes all load.
- **Five rankers** in the same `--ranker` namespace: `frequency`,
  `tfidf`, `bm25`, `bm25f`, plus an opt-in dense / hybrid / LTR
  pipeline (numpy + sentence-transformers + lightgbm).
- **Boolean query parser** with `AND`/`OR`/`NOT` and parentheses
  (precedence `NOT > AND > OR`); plain space-separated queries still
  take the original AND-intersection path.
- **Phrase mode**: `find` requires terms in adjacent positions when
  invoked with `mode="phrase"`.
- **Two "did you mean?"** implementations: the original Levenshtein
  scan and a SymSpell deletion-dictionary that's independent of
  vocabulary size.
- **Prefix autocomplete**: `suggest <prefix>` walks a Trie over the
  vocabulary and returns the most-frequent matches.
- **Atomic JSON persistence** with schema versioning, plus an optional
  **VByte binary sidecar** for compressed posting positions.
- **332+ tests / 95% coverage / mypy --strict clean**, full suite
  runs offline in under one second via mocked HTTP and patched
  `time.sleep`.

See [`docs/design.md`](docs/design.md) for the architecture, ranking
derivations, the v1.2 schema, and the empirical evaluation against the
12-query gold standard.

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:zeyad03/COMP3011-cw2.git
   cd cw2
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate     # macOS/Linux
   # or: venv\Scripts\activate  # Windows
   ```

3. Install runtime dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   To run the test suite, additionally install the dev dependencies:

   ```bash
   pip install -r requirements-dev.txt
   ```

   The dense ranker (Phase A) needs `sentence-transformers` (~80 MB
   model download); the LTR ranker (Phase B) needs `lightgbm` and, on
   macOS, `brew install libomp`. Both are intentionally opt-in; the
   core search engine and the four sparse rankers run with only the
   base requirements.

   A `Makefile` is provided for the common loops:

   ```bash
   make install-dev   # install + dev deps
   make test          # full pytest with coverage
   make typecheck     # mypy --strict on src/
   make eval          # IR evaluation harness
   make docker        # build the runtime container
   ```

## Usage

Launch the interactive shell from the project root:

```bash
python -m src.main
```

Available commands:

| Command                                          | Description                                                  |
| ------------------------------------------------ | ------------------------------------------------------------ |
| `build`                                          | Crawl the website and build the inverted index.              |
| `load`                                           | Load a previously saved index from disk.                     |
| `print <word>`                                   | Print the inverted-index entry for a word.                   |
| `find [--ranker R] [--explain] [--snippet] <q>`  | Find pages matching `<q>`. `R` ∈ {tfidf, bm25, bm25f, frequency}. The query may use `AND`/`OR`/`NOT` and parentheses. |
| `suggest <prefix>`                               | List indexed terms beginning with `<prefix>`.                |
| `help`                                           | Show the command list.                                       |
| `exit`                                           | Exit the shell.                                              |

Example session:

```text
> load
loaded data/index.json: 67 docs, 2451 terms

> print indifference
'indifference'  df=1  cf=1
  https://quotes.toscrape.com/page/3/  tf=1  positions=[42]

> find good friends
1. https://quotes.toscrape.com/page/4/  (score=0.42)
2. https://quotes.toscrape.com/page/1/  (score=0.31)

> find --ranker bm25 (einstein OR rowling) AND magic
1. https://quotes.toscrape.com/author/J-K-Rowling  (score=2.13)
2. https://quotes.toscrape.com/author/Albert-Einstein  (score=1.05)

> suggest comp
compute, computer, computing, completion

> find xyzzy
no results for 'xyzzy'.
did you mean: ...

> exit
```

> **Note on `build`**: a full crawl of the target site takes about
> 7–10 minutes due to the mandatory 6-second politeness window. Use
> `load` for quick demos once an index has been built.

## Testing

Run the full test suite with coverage:

```bash
pytest
```

The suite is fully offline: HTTP is replaced by a fake session and
`time.sleep` is patched, so all 332+ tests finish in under one second.
Integration tests under `tests/test_integration.py` run the entire
crawler → indexer → search pipeline against saved HTML fixtures from
quotes.toscrape.com.

Static type checks:

```bash
mypy --strict src/
```

Both run on every push via the GitHub Actions workflow in
`.github/workflows/ci.yml` (matrix: Python 3.12 and 3.13). Local
contributors can additionally run

```bash
pre-commit install
```

to run `ruff`, `mypy --strict`, and the smoke-test subset of pytest on
every commit (see `.pre-commit-config.yaml`).

To generate an HTML coverage report:

```bash
pytest --cov-report=html
open htmlcov/index.html
```

Performance benchmarks:

```bash
python scripts/benchmark.py
```

Empirical evaluation (P@5, P@10, R@5, MRR, MAP across the four sparse
rankers and the 12-query gold standard):

```bash
python evaluation/evaluate.py
```

Learning-to-rank training (leave-one-query-out cross-validation):

```bash
python evaluation/learning_to_rank.py
```

## Dependencies

Runtime:

- [`requests`](https://docs.python-requests.org/) — HTTP client
- [`beautifulsoup4`](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) — HTML parsing

Development:

- `pytest`, `pytest-cov`, `pytest-mock` — testing framework + coverage
- `mypy` — static type checking
- `hypothesis` — property-based tests for the storage and ranker invariants
- `pre-commit` — local hook gate (ruff, mypy, smoke pytest)
- `numpy` — required for the dense ranker tests; lazy import elsewhere
- `lightgbm` — learning-to-rank training (macOS additionally needs
  `brew install libomp`)
- `scikit-learn` — used only by the cross-validation script
- `sentence-transformers` (optional, ~80 MB model) — needed only to
  run the dense / hybrid pipeline against the real model

## Project structure

See [`docs/design.md`](docs/design.md) for the full architecture and
design rationale.

```
cw2/
├── src/                    # Source modules
│   ├── crawler.py          # BFS crawler with content-hash dedup
│   ├── indexer.py          # v1.2 schema with per-field positions
│   ├── tokenizer.py        # Shared text normalisation
│   ├── ranker.py           # FrequencyRanker, TFIDFRanker, BM25Ranker, BM25FRanker, HybridRanker
│   ├── dense_ranker.py     # Sentence-transformer embeddings (opt-in)
│   ├── ltr_ranker.py       # LightGBM-backed learning-to-rank
│   ├── search.py           # find(), print_term(), suggest()
│   ├── query_parser.py     # AND/OR/NOT parser
│   ├── query_eval.py       # AST → set of doc_ids
│   ├── autocomplete.py     # Prefix Trie
│   ├── symspell.py         # Deletion-dictionary spell correction
│   ├── snippet.py          # --snippet excerpt generation
│   ├── storage.py          # Atomic JSON + optional VByte sidecar
│   ├── vbyte.py            # Variable-byte integer compression
│   └── main.py             # Interactive REPL
├── tests/                  # Test suite (pytest)
├── evaluation/
│   ├── evaluate.py         # IR metrics over the 12-query gold standard
│   ├── learning_to_rank.py # LambdaRank training + LOQO CV
│   ├── compare_sklearn.py  # Cross-check vs scikit-learn TF-IDF
│   ├── gold_standard.json  # 12-query relevance judgements
│   └── results.csv         # Per-query and per-ranker metrics
├── data/                   # Compiled index file
├── scripts/
│   ├── benchmark.py        # Micro-benchmarks
│   └── migrate_index_to_v12.py  # In-place v1.1 → v1.2 migration
├── docs/                   # Design notes, GenAI journal
├── Dockerfile              # Reproducible runtime image
├── Makefile                # Common targets (test/typecheck/eval/...)
└── .pre-commit-config.yaml # ruff, mypy, smoke pytest
```

## Changelog

### v2.0.0 (2026-05-07)

Major release. Headline change: the project now ships **five sparse
rankers and an opt-in dense / hybrid / learning-to-rank pipeline**,
**boolean queries**, **per-field BM25F scoring**, **VByte posting
compression**, and a **prefix-trie autocomplete** — all behind the
same `find` and `suggest` commands.

**Retrieval & ranking**
- `BM25FRanker` (`--ranker bm25f`): per-field BM25 with title weighted
  5× over body, plus per-field length normalisation. Falls back to
  plain BM25 when the index lacks field positions.
- `DenseRanker`: sentence-transformer embeddings + cosine similarity
  (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim, lazy-imported).
  Embeddings persist to `<index>.embeddings.npy`.
- `HybridRanker`: reciprocal-rank fusion (Cormack 2009, `k=60`) of any
  two list-level rankers.
- `LTRRanker`: LightGBM LambdaRank model. Trained by
  `evaluation/learning_to_rank.py` with leave-one-query-out CV across
  the 12-query gold standard (six features per query/doc pair).

**Search**
- Boolean query parser (`AND`, `OR`, `NOT`, parens; precedence
  `NOT > AND > OR`). Plain space-separated queries still take the
  unchanged AND-intersection path.
- SymSpell deletion-dictionary spell correction — output identical to
  the Levenshtein scan, lookup independent of vocabulary size.
- Prefix-trie autocomplete via the new `suggest <prefix>` command,
  ranked by collection frequency.

**Index & storage**
- v1.2 schema adds `title_length` / `body_length` per doc and
  `title_positions` / `body_positions` per posting. v1.0 and v1.1
  files still load.
- Optional VByte binary posting sidecar (`storage.save(...,
  compress_postings=True)` writes `<path>.postings.bin`). The format
  is documented in `docs/design.md` §7.1 so the index is recoverable
  from spec alone.
- `scripts/migrate_index_to_v12.py` upgrades a v1.1 index to v1.2
  in place without re-crawling.

**Crawler**
- Content-hash dedup of alias URLs (`/tag/love/` and
  `/tag/love/page/1/` both serve the same body — only one ends up in
  the index, but frontier expansion still runs on the alias).

**Tooling**
- `Dockerfile` + `Makefile` for reproducible builds.
- `.pre-commit-config.yaml` (ruff, mypy `--strict`, smoke pytest).
- 332 tests pass, 2 skip (the lightgbm native runtime is missing on
  bare macOS without `brew install libomp`; one pre-existing
  ranker-protocol skip). Coverage 95 %; `mypy --strict` clean across
  17 source files.

### v1.1.0 (2026-04-29)

- BM25 ranker; evaluation framework (gold standard + IR metrics);
  `--explain`; sklearn cross-check; delta compression; property
  tests; `--snippet` excerpts.

### v1.0.0 (2026-04-22)

- Initial release: BFS crawler, positional inverted index, TF-IDF
  ranker, phrase mode, "did you mean" Levenshtein suggestions,
  atomic versioned JSON storage, interactive REPL.

## Licence

University coursework submission — not licensed for redistribution.
