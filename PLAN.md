# COMP3011 CW2 — Search Engine Tool: Detailed Development Plan

**Module**: COMP3011 — Web Services and Web Data
**Assignment**: Coursework 2 — Search Engine Tool (30% of module)
**Submission deadline**: 8 May 2026
**Plan authored**: 6 May 2026
**Working window**: ~2 days
**Target grade band**: 80–100 (Excellent to Outstanding)

## Progress (live)

| Phase | Status | Notes |
| --- | --- | --- |
| 0 — Scaffold | ✅ | Python pkg, deps, pytest, CI, README skeleton |
| 1 — Tokeniser | ✅ | 22 tests, 100% coverage |
| 2 — Crawler | ✅ | Polite BFS, retries, robots.txt; 25 tests, 100% coverage |
| 3 — Indexer | ✅ | Positional inverted index, TypedDicts; 17 tests, 100% coverage |
| 4 — Storage | ✅ | Atomic JSON, schema versioning; 13 tests, 100% coverage |
| 5 — Search | ✅ | TF-IDF, phrase mode, suggestions; 42 tests, 100% coverage |
| 6 — CLI | ✅ | REPL with build/load/print/find; 24 tests, 96% coverage |
| 7 — Real crawl | 🟡 | Running (~6 s × ~80 pages ≈ 8–10 min) |
| 8 — Advanced | ✅ | Phrase queries + suggestions in §5; benchmark suite added |
| 9 — Polish | ✅ | mypy --strict clean, types-requests, integration fixtures |
| 10 — Docs | ✅ | README, design.md (§§1–9), GenAI journal scaffold |
| 12 — Submit | ⏳ | After tagging v1.0.0 and pushing to GitHub |

**Current totals**: 152 tests, 99% line coverage, mypy strict clean,
~7 phases worth of commits.

---

## 1. Goal

Develop a Python command-line search tool that:

1. **Crawls** every page of `https://quotes.toscrape.com/` while respecting a 6-second politeness window.
2. **Builds an inverted index** of all word occurrences, storing per-page statistics (term frequency, positions, document length, etc.).
3. **Persists** the index to disk and reloads it on demand.
4. **Searches** the index from a CLI shell, supporting single-word and multi-word queries with relevance ranking.

Submission consists of: a public GitHub repository, a 5-minute video demonstration, and the compiled index file.

---

## 2. Mark scheme analysis (where the points are)

| Criterion | Weight | Implication for plan |
|---|---|---|
| Testing & Test Coverage | **20%** | Largest single weight. Build tests *alongside* code, not after. Target ≥ 85% coverage with mocking, unit + integration + edge cases. |
| GenAI Critical Evaluation | **15%** | Keep a development journal of GenAI interactions from day 1: what was asked, what came back, what was wrong, what worked. Specific anecdotes win marks. |
| Search Functionality (print/find) | 12% | Multi-word queries must work correctly. Ranking earns top-band marks. |
| Crawling Implementation | 10% | Politeness, error handling, retry, domain restriction. |
| Indexing Implementation | 10% | Correct inverted index with statistics. |
| Code Quality & Documentation | 10% | Type hints, docstrings, PEP 8, modular design, comprehensive README. |
| Video Demonstration | 10% | Script and rehearse. 5 min hard cap. |
| Storage & Retrieval (build/load) | 8% | Round-trip correctness. JSON for transparency. |
| Version Control & Git | 5% | Semantic commits, branches, tags/releases. |

### Top grade band (80–100) explicit asks

- Advanced features beyond requirements: **TF-IDF ranking**, **advanced query processing**, **query suggestions**.
- Highly optimised algorithms with **complexity analysis and benchmarking**.
- Professional-grade test suite with **mocking** and **automated testing pipeline (CI)**.
- Professional Git workflow with **semantic commits**, **tags/releases**, **branching strategy**.
- Outstanding video discussing **algorithmic trade-offs**.
- Sophisticated GenAI evaluation discussing **ethical considerations or learning implications**.
- Publication-quality code: **type hints**, **docstrings**, modular structure.
- Evidence of **research into search engine algorithms and modern practices**.

The plan below explicitly targets each of these.

---

## 3. Architecture

### 3.1 Module map

```
cw2/
├── src/
│   ├── __init__.py
│   ├── crawler.py        # BFS web crawler with politeness + retries
│   ├── indexer.py        # Tokenisation, inverted index construction
│   ├── search.py         # Single/multi-word search + TF-IDF ranking
│   ├── storage.py        # JSON serialisation of the index
│   ├── main.py           # CLI REPL entry point
│   └── tokenizer.py      # Shared tokenisation (used by indexer + search)
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Shared fixtures (sample HTML, mock responses)
│   ├── test_crawler.py        # Mocked HTTP, URL filtering, politeness timing
│   ├── test_indexer.py        # Tokeniser edges, index correctness
│   ├── test_search.py         # Ranking, multi-word, edge cases
│   ├── test_storage.py        # Round-trip, schema validation
│   ├── test_main.py           # CLI parsing, command dispatch
│   └── fixtures/              # Saved HTML pages for offline tests
├── data/
│   └── index.json             # Compiled index (committed for marker convenience)
├── scripts/
│   └── benchmark.py           # Performance/complexity benchmarks
├── docs/
│   ├── design.md              # Architecture and design rationale
│   └── genai-journal.md       # Running log of GenAI interactions
├── .github/
│   └── workflows/
│       └── ci.yml             # Run pytest on every push
├── requirements.txt
├── pytest.ini
├── README.md
├── PLAN.md                    # This file
└── .gitignore
```

### 3.2 Component responsibilities

#### `crawler.py`
- BFS traversal starting from `https://quotes.toscrape.com/`.
- Maintains a `visited: set[str]` and a FIFO `frontier: deque[str]`.
- Restricts to the `quotes.toscrape.com` host (no external links, no subdomains).
- Honours `robots.txt` (read once at startup).
- **Politeness**: shared `_last_request_at` timestamp; `time.sleep` until `now - last >= 6.0` before every request.
- **Retries**: 3 attempts with exponential backoff on transient errors (5xx, network).
- **Returns**: `list[Page]` where `Page = {"url": str, "title": str, "text": str}`.
- Exposes a public `crawl(start_url, delay=6.0, max_pages=None) -> list[Page]`.

#### `tokenizer.py`
- Single source of truth for tokenisation (avoids drift between indexer and query parser).
- `tokenize(text: str) -> list[str]`:
  - Lowercase.
  - Strip Unicode punctuation.
  - Split on whitespace.
  - Drop empty tokens.
  - Optionally drop tokens of length 1 (configurable).

#### `indexer.py`
- Walks each page's text, calling `tokenize()`.
- Builds:
  ```python
  Index = {
    "meta": {
      "crawled_at": iso_timestamp,
      "num_docs": int,
      "num_terms": int,
      "total_tokens": int,
      "version": "1.0",
    },
    "docs": {
      doc_id: {
        "url": str,
        "title": str,
        "length": int,        # token count
      },
      ...
    },
    "terms": {
      term: {
        "df": int,            # document frequency
        "cf": int,            # collection frequency
        "postings": {
          doc_id: {
            "tf": int,
            "positions": [int, ...],
          },
          ...
        },
      },
      ...
    },
  }
  ```
- `build_index(pages: list[Page]) -> Index`.

#### `search.py`
- `print_term(index, word) -> str`: pretty-prints the postings for `word`.
- `find(index, query: str) -> list[SearchResult]`:
  - Tokenise the query using `tokenizer.tokenize`.
  - For each token, fetch its posting list. AND-intersect document sets.
  - Score each candidate doc with **TF-IDF**:
    - `tf(t,d) = 1 + log10(freq)` (sub-linear scaling, dampens spam).
    - `idf(t) = log10(N / df(t))` with smoothing (`+1` to avoid zeros).
    - Doc score = sum over query terms of `tf * idf`.
  - Optional **phrase proximity bonus**: if query terms appear in adjacent positions, multiply score by 1.5.
  - Sort descending, return list of `(url, title, score, matched_terms)`.
- `suggest(index, word, max_distance=2) -> list[str]`: Levenshtein-based "did you mean?" for unknown query terms.

#### `storage.py`
- `save(index, path)` and `load(path) -> Index`.
- JSON format (transparent, easy to grade).
- Atomic write: write to `path.tmp`, then rename.
- Schema version field; `load` validates compatibility.

#### `main.py`
- REPL loop with `> ` prompt.
- Commands: `build`, `load`, `print <word>`, `find <words...>`, `help`, `exit`.
- Maintains in-memory `index` reference between commands.
- Friendly error messages (no stack traces leaking out for expected failures).
- Argparse for command-line invocation (`python -m src.main` or `python src/main.py`).

### 3.3 Key design decisions and trade-offs

| Decision | Choice | Rationale |
|---|---|---|
| Persistence format | **JSON** | Transparent, human-readable, easy for marker to inspect. Pickle is faster but opaque and version-fragile. |
| Tokenisation | Lowercase + punctuation strip + whitespace split, **no stemming**, **no stop-word removal** | Preserves positional information needed for phrase proximity. Keeps the implementation explainable. Stemming would obscure "good"/"goodness" distinction the marker may test. |
| Crawl strategy | **BFS** | Predictable order, easier to reason about politeness budget; keeps frontier flat. |
| Index data structure | **Dict-of-dicts with positions** | Positions enable phrase proximity bonus and future phrase queries. Memory acceptable for ~70 pages. |
| Ranking | **TF-IDF** with sub-linear TF | Industry standard, lecture-aligned, easy to explain on video. BM25 is a stretch goal if time allows. |
| Politeness enforcement | Shared timestamp guarded inside crawler | Single point of control; impossible to bypass accidentally. |
| Tests | **pytest** with `unittest.mock` for HTTP, fixtures for sample HTML | Mocking lets tests run offline and fast (CI-friendly). |
| Type system | **Type hints everywhere**, validated by `mypy --strict` | Top-band marker explicitly listed. |

---

## 4. Phased delivery

Each phase has: **Goal**, **Tasks**, **Deliverables**, **Tests**, **Acceptance**, **Estimated time**, **Commit message**.

### Phase 0 — Setup (15 min)

**Goal**: Empty repo ready to receive code.

**Tasks**:
1. `git init` in `cw2/`.
2. Create `.gitignore` (Python, venv, `__pycache__`, `.pytest_cache`, IDE files).
3. Create empty directory structure as in §3.1.
4. Create `requirements.txt` with `requests`, `beautifulsoup4`, `pytest`, `pytest-cov`, `mypy`.
5. Create Python virtual environment (`python3 -m venv venv`); install deps.
6. Create `pytest.ini` with `testpaths`, coverage config.

**Deliverable**: directory layout + first commit.

**Commit**: `chore: initial project scaffolding`

---

### Phase 1 — Tokeniser (30 min)

**Goal**: Bullet-proof tokenisation, used everywhere downstream.

**Tasks**:
1. Implement `src/tokenizer.py::tokenize(text) -> list[str]`.
2. Handle Unicode (curly quotes, em-dashes, accented characters).
3. Write `tests/test_tokenizer.py`:
   - Empty string → `[]`.
   - Single word.
   - Mixed case → all lowercase.
   - Punctuation stripped.
   - Multiple whitespace, tabs, newlines.
   - Apostrophes (`don't` → `dont` or `don`/`t`? — pick and document).
   - Numbers (kept as tokens).
   - Unicode handling.

**Acceptance**: 100% branch coverage, all edge tests pass.

**Commit**: `feat(tokenizer): case-insensitive punctuation-stripping tokeniser with tests`

---

### Phase 2 — Crawler (2 hr)

**Goal**: Crawler that retrieves pages politely, with retries, fully offline-testable.

**Tasks**:
1. `src/crawler.py`:
   - `class Crawler` with configurable `delay`, `user_agent`, `timeout`, `max_pages`.
   - `_fetch(url) -> str` with retry/backoff.
   - `_extract_links(html, base_url) -> list[str]` (BeautifulSoup, filter to same host, normalise).
   - `_extract_text(html) -> tuple[title, text]`.
   - `crawl(start_url) -> list[Page]`.
   - Respect `robots.txt` via `urllib.robotparser`.
2. `tests/test_crawler.py`:
   - Mock `requests.get` with `unittest.mock`.
   - Test domain restriction (external links dropped).
   - Test fragment/query stripping.
   - Test loop avoidance (visited URL not refetched).
   - Test politeness: monkeypatch `time.sleep`, assert it was called with ≥ 6 between requests.
   - Test retry on 503.
   - Test give-up after N failures.
   - Test `max_pages` limit.

**Acceptance**: Tests pass, full crawl of fixture site (3 mocked pages) succeeds.

**Commit**: `feat(crawler): polite BFS crawler with retries and domain restriction`

---

### Phase 3 — Indexer (1.5 hr)

**Goal**: Correct inverted index built from a list of pages.

**Tasks**:
1. `src/indexer.py::build_index(pages) -> Index`.
2. `tests/test_indexer.py`:
   - Single page, single word.
   - Multiple pages → `df` correct.
   - Repeated word → `tf` and `positions` correct.
   - Empty page handled.
   - `meta` block populated.
   - Index keys match expected schema.

**Acceptance**: Schema documented in `docs/design.md`. All tests pass.

**Commit**: `feat(indexer): inverted index with TF, DF, and positional postings`

---

### Phase 4 — Storage (45 min)

**Goal**: Save/load the index correctly.

**Tasks**:
1. `src/storage.py::save(index, path)` and `load(path) -> Index`.
2. Atomic write via tempfile + rename.
3. Schema version check on load.
4. `tests/test_storage.py`:
   - Round-trip: build → save → load → equal.
   - Missing file → graceful error.
   - Corrupt JSON → graceful error.
   - Wrong schema version → graceful error.

**Acceptance**: Round-trip is idempotent and fast.

**Commit**: `feat(storage): JSON persistence with atomic writes and schema validation`

---

### Phase 5 — Search (2 hr)

**Goal**: Single-word lookup, multi-word AND, TF-IDF ranking.

**Tasks**:
1. `src/search.py::print_term(index, word) -> str`.
2. `src/search.py::find(index, query) -> list[SearchResult]`:
   - Tokenise query.
   - Intersect posting lists (start with smallest for efficiency).
   - Score with TF-IDF + optional proximity bonus.
   - Return ranked list.
3. `src/search.py::suggest(index, word) -> list[str]` for unknown terms.
4. `tests/test_search.py`:
   - Word in single page.
   - Word in multiple pages → ranked correctly (more frequent → higher score).
   - Multi-word: AND semantics enforced.
   - Word not in index → empty result + suggestion.
   - Empty query → empty result, no crash.
   - Case-insensitive query.
   - Proximity boost when terms adjacent.

**Acceptance**: Ranking is reproducible (same query, same index, same order). Tests pass.

**Commit**: `feat(search): TF-IDF ranked search with multi-word AND and "did you mean" suggestions`

---

### Phase 6 — CLI (1 hr)

**Goal**: Interactive `>` shell as specified in the brief.

**Tasks**:
1. `src/main.py`:
   - REPL with prompt `> `.
   - Commands: `build`, `load`, `print <word>`, `find <words...>`, `help`, `exit`.
   - Holds in-memory `Index | None`.
   - Sensible messages when commands run before index exists (`> find good` before `build`/`load`).
2. `tests/test_main.py`:
   - Use `monkeypatch` to feed stdin / capture stdout.
   - Test each command and the error paths.

**Acceptance**: All four required commands behave per the brief.

**Commit**: `feat(cli): interactive REPL with build, load, print, and find commands`

---

### Phase 7 — Real crawl + index file (30 min)

**Goal**: Run a full real crawl once, commit the index file as a deliverable.

**Tasks**:
1. Run `python -m src.main`, then `> build`. Wait ~10 minutes.
2. Verify index covers all pages of `quotes.toscrape.com` (manual sanity check on `> print love`, `> find good friends`).
3. Save `data/index.json` and commit it.

**Acceptance**: `> find good friends` returns a sensibly ranked list of pages.

**Commit**: `data: add compiled index from full crawl of quotes.toscrape.com`

---

### Phase 8 — Advanced features (1.5 hr)

**Goal**: Earn the top-band "advanced features" marks.

**Tasks**:
1. **Phrase queries**: `find "good friends"` (in quotes) → require terms in adjacent positions in same doc.
2. **Query suggestions**: already in §5; surface from CLI when no results.
3. **Benchmarks** (`scripts/benchmark.py`):
   - Time `build_index` on N=10, 30, 70 pages.
   - Time `find` on warm index for several queries.
   - Plot or print O(·) analysis in `docs/design.md`.
4. **Robustness**: handle malformed HTML, missing titles, empty bodies.

**Commit**: `feat: phrase queries, query suggestions, and benchmark suite`

---

### Phase 9 — Polish (1.5 hr)

**Goal**: Push code quality to publication-grade.

**Tasks**:
1. Type hints across all modules; run `mypy --strict src/`. Fix all warnings.
2. Docstrings on every public function (NumPy or Google style; pick one and apply consistently).
3. Run `python -m pyflakes` / `ruff check` and fix.
4. Increase test coverage to ≥ 85% (`pytest --cov=src --cov-report=term-missing`).
5. Add fixtures of saved HTML to `tests/fixtures/` for offline integration tests.
6. Add a `tests/test_integration.py` that runs the full pipeline against fixtures (no network).
7. CI workflow `.github/workflows/ci.yml` runs `pytest` and `mypy` on every push.
8. Tag release: `git tag -a v1.0.0 -m "Initial release"`.

**Commit chain**: small focused commits per concern (`refactor`, `test`, `docs`, `ci`).

---

### Phase 10 — Documentation (1 hr)

**Goal**: README and design doc that match top-band expectations.

**Tasks**:
1. `README.md`:
   - Project overview and purpose.
   - Architecture diagram (ASCII or Mermaid).
   - Installation: `python -m venv venv`, activate, `pip install -r requirements.txt`.
   - Usage: every command with example output.
   - Testing: `pytest` and how to read coverage.
   - Dependencies and licences.
   - GenAI declaration.
2. `docs/design.md`:
   - Why TF-IDF over plain frequency.
   - Why JSON over pickle.
   - Index schema.
   - Complexity analysis (O(N·M) build, O(k log N) search where k = avg posting length).
   - Benchmark numbers from §8.

**Commit**: `docs: comprehensive README, design rationale, and complexity analysis`

---

### Phase 11 — Video (2 hr)

**Goal**: 5-minute video that hits every requirement and stays under the cap.

**Script outline** (300 sec hard cap; padding for cuts):

| Segment | Time | Content |
|---|---|---|
| Intro | 10s | Name, module, project title. |
| Live demo | 110s | `build` (fast-forward to load), `load`, `print indifference`, `find indifference`, `find good friends`, edge cases (`find` empty, `find xyzzy`). |
| Code walkthrough | 80s | Crawler → Indexer → Search. Index data structure shown on screen. Justify TF-IDF and JSON. |
| Testing | 30s | Run `pytest --cov`. Show coverage %, mention mocking strategy. |
| Git history | 30s | `git log --oneline --graph`. Highlight semantic commits and tag. |
| GenAI critical evaluation | 30s | Two specific anecdotes (one helpful, one corrected). Reflection on learning. |
| Outro | 10s | Thank you. |

**Tasks**:
2. Dry-run, time it.
3. Record with QuickTime / OBS at 1080p, audio levels checked.
4. Upload **unlisted** to Google Drive or YouTube.
5. Test link in incognito browser (per brief).

---

### Phase 12 — Submission (30 min)

**Tasks**:
1. Push final commits and tag to GitHub. Make repo public.
2. Verify GitHub Actions CI is green.
3. Verify `data/index.json` is present and loadable.
4. Create the submission document (PDF or TXT) containing:
   - Video URL.
   - GitHub repo URL.
   - Note about index file location.
5. Attach `data/index.json` to Minerva (or include download link).
6. Submit before deadline.

---

## 5. Testing strategy

| Layer | What | Tools |
|---|---|---|
| Unit | Pure functions: tokenise, score, intersect, save, load | `pytest`, parametrised |
| Mocked | Crawler, network, sleep | `unittest.mock`, `pytest-mock` |
| Integration | Full pipeline against fixtures (`tests/fixtures/*.html`) | `pytest` |
| End-to-end | CLI commands with simulated stdin/stdout | `pytest`, `monkeypatch` |
| Regression | A locked-down "expected outputs" file for `find good friends` against the real index | `pytest` |
| Performance | `scripts/benchmark.py`, manual run | `time`, `pytest-benchmark` (optional) |

**Coverage target**: 85%+ branch coverage. Configured to fail CI if below.

**Edge cases to test explicitly** (per the brief: "edge cases (e.g., non-existent words, empty queries)"):
- Empty query.
- Whitespace-only query.
- Word with no postings.
- Word with one posting.
- Query with all unknown terms.
- Mixed known/unknown terms (intersection becomes empty).
- Unicode in queries.
- Very long query.
- Re-running `build` after `build` (idempotency).
- `load` before any `build` (no file).

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Crawl takes too long during dev | High | Medium | Add `max_pages=5` flag for dev; cache fixtures locally; full crawl only for the real submission index. |
| Site flakiness or rate-limiting | Medium | Medium | Retry with backoff. Larger politeness window if observed. |
| Test suite drifts from implementation | Medium | High | Write tests in same commit as feature. CI on every push. |
| Video over 5 min | High | Medium (mark loss) | Script with timestamps. Two dry runs before final record. |
| GenAI evaluation feels generic | High | High (15% of mark) | Keep `docs/genai-journal.md` updated *during* development, not after. |
| Forgetting to declare GenAI use | Low | Critical (academic misconduct) | Declaration drafted before video record. |
| Last-minute bug in CLI | Medium | High | Lock CLI in Phase 6, only refactor it after all tests pass. |
| Index file too large for Minerva | Low | Low | Compress; or include a download link. |

---

## 7. Git workflow

- **Branching**: `main` is always green. Feature branches per phase: `feat/crawler`, `feat/indexer`, etc. Merge with `--no-ff` to preserve history, or fast-forward if branches are short-lived. Plan favours short-lived branches given the 2-day window.
- **Commits**: Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`, `ci:`). Scope optional but encouraged.
- **Frequency**: at least one commit per phase, often more. Atomic commits — one logical change each.
- **Tags**: `v1.0.0` at submission time.
- **Release notes**: GitHub release attached to `v1.0.0` summarising features.

---

## 8. GenAI tracking

A live journal at `docs/genai-journal.md` with entries like:

```
## 2026-05-06 14:30 — Tokeniser regex
- Asked: "Python regex to strip Unicode punctuation"
- Got: \W+ pattern
- Issue: \W also drops apostrophes which I wanted to keep as word separators.
  Adjusted to use unicodedata.category() == 'P*' check instead.
- Verdict: AI accelerated discovery of unicodedata module, but its first answer
  was incorrect for my requirements.
```

These entries become the raw material for the 30-second video segment. The video must cover (per brief):
- Specific examples of where GenAI helped or hindered.
- Quality and correctness of AI-generated code.
- Reflection on learning impact.
- Challenges debugging AI code.
- Impact on time management.

---

## 9. Tech stack and dependencies

```
# requirements.txt
requests>=2.32
beautifulsoup4>=4.12
pytest>=8.0
pytest-cov>=5.0
pytest-mock>=3.12
mypy>=1.10
```

No frameworks, no databases, no async — keeps the surface small and the marker's job easy.

---

## 10. Timeline (compressed: 2 days)

### Day 1 (Wed 6 May, today)

| Time | Phase |
|---|---|
| Now → +0:15 | Phase 0 — Setup |
| +0:15 → +0:45 | Phase 1 — Tokeniser |
| +0:45 → +2:45 | Phase 2 — Crawler |
| +2:45 → +4:15 | Phase 3 — Indexer |
| +4:15 → +5:00 | Phase 4 — Storage |
| +5:00 → +7:00 | Phase 5 — Search |
| +7:00 → +8:00 | Phase 6 — CLI |
| Evening | Phase 7 — Real crawl (runs in background) |

### Day 2 (Thu 7 May)

| Time | Phase |
|---|---|
| Morning | Phase 8 — Advanced features |
| Midday | Phase 9 — Polish (types, coverage, CI) |
| Afternoon | Phase 10 — Docs |
| Evening | Phase 11 — Video script + record |

### Day 3 (Fri 8 May, deadline)

| Time | Activity |
|---|---|
| Morning | Buffer for re-record / fixes |
| By midday | Phase 12 — Submit |

Buffer is intentionally placed on the deadline morning to absorb slippage.

---

## 11. Acceptance criteria (definition of done)

The project is ready to submit when:

- [ ] All four required commands (`build`, `load`, `print`, `find`) work as specified in the brief.
- [ ] `find good friends` returns a non-empty, ranked list against the committed index.
- [ ] `find xyzzy` returns "no results" with a graceful message.
- [ ] `pytest` passes with ≥ 85% coverage.
- [ ] `mypy --strict src/` passes.
- [ ] CI workflow is green on `main`.
- [ ] `data/index.json` exists, loads without error, covers all pages of the target site.
- [ ] README has all five required sections (overview, install, usage, testing, dependencies).
- [ ] `docs/design.md` exists with schema, rationale, and complexity analysis.
- [ ] `docs/genai-journal.md` has ≥ 6 dated entries.
- [ ] Video is ≤ 5 minutes, uploaded unlisted, link verified in incognito.
- [ ] Repo is public, has at least one tagged release.
- [ ] Submission document (PDF or TXT) is prepared with both URLs and submitted via Minerva.

---

## 12. Open questions before coding starts

1. **GitHub repo name**: default to `comp3011-cw2-search-engine` unless preferred otherwise.
2. **GenAI use**: confirm tools used (e.g. Claude Code, Copilot, ChatGPT). Affects declaration text.
3. **Phrase query syntax**: brief shows `find good friends` as space-separated AND. Should `find "good friends"` mean phrase-query? **Recommend yes** — declared as advanced feature.
4. **Stretch features (skip if behind)**: BM25 ranking, snippet generation, query autocompletion. None are required.

---

## 13. References (for the README and video)

- Manning, Raghavan, Schütze. *Introduction to Information Retrieval*. CUP, 2008. — chapters 1, 6 (TF-IDF), 4 (index construction).
- Module lecture notes on web crawling, indexing, ranking algorithms.
- `requests` documentation: https://docs.python-requests.org/
- BeautifulSoup documentation: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- Target site: https://quotes.toscrape.com/
