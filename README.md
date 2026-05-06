# Search Engine Tool

A Python command-line search engine that crawls
[quotes.toscrape.com](https://quotes.toscrape.com), builds an inverted
index of every word it sees, and answers single- and multi-word queries
with TF-IDF ranking.

> Coursework 2 for **COMP3011 — Web Services and Web Data**, University of Leeds.

## Status

Work in progress. See [`PLAN.md`](PLAN.md) for the development roadmap.

## Installation

1. Clone the repository:

   ```bash
   git clone <repo-url>
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
> build
Crawling https://quotes.toscrape.com/ ...
Indexed 67 pages, 2451 unique terms.
Saved index to data/index.json.

> print indifference
indifference  df=1
  https://quotes.toscrape.com/page/3/  tf=1  positions=[42]

> find good friends
1. https://quotes.toscrape.com/page/1/  score=0.7421
2. https://quotes.toscrape.com/page/4/  score=0.5103
...
```

## Testing

Run the full test suite with coverage:

```bash
pytest
```

Coverage is reported to the terminal. To generate an HTML report:

```bash
pytest --cov-report=html
open htmlcov/index.html
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
