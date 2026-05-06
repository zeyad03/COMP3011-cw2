# Search Engine Tool

A Python command-line search engine that crawls
[quotes.toscrape.com](https://quotes.toscrape.com), builds a
**positional inverted index** of every word it sees, and answers single-
and multi-word queries with **TF-IDF ranking**.

> Coursework 2 for **COMP3011 — Web Services and Web Data**, University of Leeds.

## Highlights

- **Polite crawler**: BFS over the seed host with a strict 6-second
  request gap, robots.txt support, and exponential-backoff retries.
- **Inverted index** with term frequency, document frequency, and
  token positions per posting — positions enable phrase queries.
- **TF-IDF ranking** with sub-linear TF (`1 + log10(tf)`) and a
  smoothed IDF that stays positive on small corpora.
- **Phrase mode**: `find` requires terms in adjacent positions when
  invoked with `mode="phrase"`.
- **"Did you mean?"** Levenshtein suggestions for unknown query
  terms.
- **Atomic JSON persistence** with schema versioning.
- **147 tests / 97% coverage / mypy --strict clean**, full suite runs
  offline in under 0.3 seconds via mocked HTTP and patched
  `time.sleep`.

See [`docs/design.md`](docs/design.md) for the architecture, ranking
derivation, and complexity analysis.

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

## Usage

Launch the interactive shell from the project root:

```bash
python -m src.main
```

Available commands:

| Command           | Description                                           |
| ----------------- | ----------------------------------------------------- |
| `build`           | Crawl the website and build the inverted index.       |
| `load`            | Load a previously saved index from disk.              |
| `print <word>`    | Print the inverted-index entry for a word.            |
| `find <words...>` | Find pages containing the query terms (TF-IDF ranked).|
| `help`            | Show the command list.                                |
| `exit`            | Exit the shell.                                       |

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
`time.sleep` is patched, so all 147 tests finish in under 0.3 seconds.
Integration tests under `tests/test_integration.py` run the entire
crawler → indexer → search pipeline against saved HTML fixtures from
quotes.toscrape.com.

Static type checks:

```bash
mypy --strict src/
```

Both run on every push via the GitHub Actions workflow in
`.github/workflows/ci.yml` (matrix: Python 3.12 and 3.13).

To generate an HTML coverage report:

```bash
pytest --cov-report=html
open htmlcov/index.html
```

Performance benchmarks:

```bash
python scripts/benchmark.py
```

## Dependencies

Runtime:

- [`requests`](https://docs.python-requests.org/) — HTTP client
- [`beautifulsoup4`](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) — HTML parsing

Development:

- `pytest`, `pytest-cov`, `pytest-mock` — testing framework + coverage
- `mypy` — static type checking

## Project structure

See [`PLAN.md`](PLAN.md) §3 for the full architecture and design rationale.

```
cw2/
├── src/          # Source modules: crawler, indexer, search, storage, main
├── tests/        # Test suite (pytest)
├── data/         # Compiled index file
├── scripts/      # Benchmarks and utility scripts
├── docs/         # Design notes, GenAI journal
└── .github/      # CI workflows
```

## Licence

University coursework submission — not licensed for redistribution.
