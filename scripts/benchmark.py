"""Performance benchmarks for the indexer and search.

Run from the project root:

    python scripts/benchmark.py

Reports wall-clock time for:
  * ``build_index`` over a range of corpus sizes,
  * ``find`` for single-term, multi-term AND, and phrase queries.

Synthetic documents are used so the numbers are reproducible without
hitting the network. Each measurement is the mean over ``runs``
repetitions; the standard deviation is reported alongside.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.crawler import Page  # noqa: E402
from src.indexer import build_index  # noqa: E402
from src.search import find  # noqa: E402


def _make_pages(n: int) -> list[Page]:
    """Synthesise *n* pages with overlapping vocabulary."""
    return [
        Page(
            url=f"https://example.com/page{i}",
            title=f"Page {i}",
            text=" ".join(f"word{j}" for j in range(i, i + 100)),
        )
        for i in range(n)
    ]


def _format_ms(value: float) -> str:
    return f"{value:>10.2f}"


def benchmark_build(sizes: list[int], runs: int = 3) -> None:
    print("=" * 60)
    print("build_index")
    print("=" * 60)
    print(f"{'docs':>6}  {'mean (ms)':>10}  {'stdev (ms)':>10}")
    for n in sizes:
        pages = _make_pages(n)
        timings: list[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            build_index(pages)
            timings.append((time.perf_counter() - start) * 1000)
        sd = stdev(timings) if len(timings) > 1 else 0.0
        print(f"{n:>6}  {_format_ms(mean(timings))}  {_format_ms(sd)}")


def benchmark_search(num_docs: int = 200, runs: int = 200) -> None:
    print()
    print("=" * 60)
    print(f"find (corpus = {num_docs} docs)")
    print("=" * 60)
    pages = _make_pages(num_docs)
    index = build_index(pages)

    queries = [
        ("single",      "and",    "word100"),
        ("AND (2)",     "and",    "word100 word150"),
        ("AND (3)",     "and",    "word100 word150 word170"),
        ("phrase (2)",  "phrase", "word100 word101"),
        ("phrase (3)",  "phrase", "word100 word101 word102"),
    ]

    print(f"{'query':>14}  {'mode':>6}  {'mean (ms)':>10}  {'stdev (ms)':>10}")
    for label, mode, query in queries:
        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            find(index, query, mode=mode)  # type: ignore[arg-type]
            timings.append((time.perf_counter() - start) * 1000)
        sd = stdev(timings) if len(timings) > 1 else 0.0
        print(
            f"{label:>14}  {mode:>6}  "
            f"{_format_ms(mean(timings))}  {_format_ms(sd)}"
        )


def main() -> None:
    benchmark_build(sizes=[10, 50, 100, 500, 1000, 2000])
    benchmark_search()


if __name__ == "__main__":
    main()
