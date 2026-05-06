"""Empirical evaluation of the project's three rankers.

Loads the gold-standard query set (``evaluation/gold_standard.json``)
and the saved index (``data/index.json``), runs each query through
each ranker, and reports:

* P@5  — precision at 5
* P@10 — precision at 10
* R@5  — recall at 5
* MRR  — mean reciprocal rank
* MAP  — mean average precision

A summary table is printed to stdout; a per-query CSV is written to
``evaluation/results.csv``.

Usage
-----

    python evaluation/evaluate.py
    python evaluation/evaluate.py --top-k 20 --index data/index.json

Reproducible: no network access, runs in seconds against the saved
index.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import (  # noqa: E402
    average_precision,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.ranker import RANKERS  # noqa: E402
from src.search import find  # noqa: E402
from src.storage import load  # noqa: E402

DEFAULT_GOLD = Path(__file__).resolve().parent / "gold_standard.json"
DEFAULT_CSV = Path(__file__).resolve().parent / "results.csv"
DEFAULT_INDEX = ROOT / "data" / "index.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    index = load(args.index)
    with args.gold.open(encoding="utf-8") as fp:
        gold = json.load(fp)
    queries = gold["queries"]

    print(
        f"Evaluating {len(RANKERS)} rankers × {len(queries)} queries "
        f"against {args.index.name}"
    )
    print(f"Gold standard: {args.gold.name}")
    print()
    print(
        f"{'ranker':>10}  {'P@5':>6}  {'P@10':>6}  {'R@5':>6}  "
        f"{'MRR':>6}  {'MAP':>6}"
    )
    print("-" * 50)

    rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for name, ranker in RANKERS.items():
        p5_scores: list[float] = []
        p10_scores: list[float] = []
        r5_scores: list[float] = []
        rr_scores: list[float] = []
        ap_scores: list[float] = []

        for query in queries:
            relevant = set(query["relevant"])
            results = find(
                index, query["text"], ranker=ranker, top_k=args.top_k
            )
            retrieved = [r.url for r in results]

            p5 = precision_at_k(retrieved, relevant, 5)
            p10 = precision_at_k(retrieved, relevant, 10)
            r5 = recall_at_k(retrieved, relevant, 5)
            rr = reciprocal_rank(retrieved, relevant)
            ap = average_precision(retrieved, relevant)

            p5_scores.append(p5)
            p10_scores.append(p10)
            r5_scores.append(r5)
            rr_scores.append(rr)
            ap_scores.append(ap)

            rows.append(
                {
                    "ranker": name,
                    "query_id": query["id"],
                    "query": query["text"],
                    "p@5": f"{p5:.3f}",
                    "p@10": f"{p10:.3f}",
                    "r@5": f"{r5:.3f}",
                    "rr": f"{rr:.3f}",
                    "ap": f"{ap:.3f}",
                }
            )

        m_p5 = mean(p5_scores)
        m_p10 = mean(p10_scores)
        m_r5 = mean(r5_scores)
        mrr = mean(rr_scores)
        m_ap = mean(ap_scores)
        print(
            f"{name:>10}  {m_p5:>6.3f}  {m_p10:>6.3f}  {m_r5:>6.3f}  "
            f"{mrr:>6.3f}  {m_ap:>6.3f}"
        )
        summary_rows.append(
            {
                "ranker": name,
                "query_id": "_MEAN_",
                "query": "(mean over all queries)",
                "p@5": f"{m_p5:.3f}",
                "p@10": f"{m_p10:.3f}",
                "r@5": f"{m_r5:.3f}",
                "rr": f"{mrr:.3f}",
                "ap": f"{m_ap:.3f}",
            }
        )

    # Detailed CSV: per-query rows + per-ranker means.
    with args.csv.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = [
            "ranker", "query_id", "query", "p@5", "p@10", "r@5", "rr", "ap"
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerows(summary_rows)
    print()
    print(f"Detailed per-query results written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
