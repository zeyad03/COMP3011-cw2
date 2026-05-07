"""Train a LambdaRank LTR model on the 12-query gold standard (Phase B).

For each query, every document is treated as a candidate. The label is
1 if the URL is in the relevant set, else 0. Features are extracted by
:func:`src.ltr_ranker.extract_features`.

Cross-validation is **leave-one-query-out** (12 folds): train on 11
queries, score the held-out query, accumulate AP. The reported MAP is
the mean of the per-fold APs.

Usage::

    python evaluation/learning_to_rank.py
    python evaluation/learning_to_rank.py --save data/ltr_model.txt

Caveats
-------

* The corpus has only 12 queries. Anything within ±0.05 of BM25 should
  be read as "not significantly different on this corpus." Heavy
  regularisation (low max_depth, small num_leaves, L2) is used to
  fight overfitting.
* The script seeds NumPy and LightGBM so MAPs are reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.ltr_ranker import FEATURE_NAMES, extract_features  # noqa: E402
from src.metrics import average_precision, mean  # noqa: E402
from src.ranker import BM25Ranker  # noqa: E402
from src.search import find  # noqa: E402
from src.storage import load  # noqa: E402

DEFAULT_INDEX = ROOT / "data" / "index.json"
DEFAULT_GOLD = ROOT / "evaluation" / "gold_standard.json"
DEFAULT_MODEL_OUT = ROOT / "data" / "ltr_model.txt"

SEED = 42


def _build_dataset(
    index: dict, queries: list[dict]
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Return (X, y, group_sizes, query_ids) for LightGBM.

    X is (N, F); y is (N,) with 0/1 labels; group_sizes is the per-query
    candidate count required by LightGBM's ranking interface.
    """
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    query_ids: list[str] = []
    doc_ids = list(index["docs"].keys())
    for q in queries:
        relevant = set(q["relevant"])
        for doc_id in doc_ids:
            rows.append(extract_features(index, q["text"], doc_id))
            labels.append(
                1 if index["docs"][doc_id]["url"] in relevant else 0
            )
        groups.append(len(doc_ids))
        query_ids.append(q["id"])
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        groups,
        query_ids,
    )


def _ap_for_query(
    booster, index: dict, query: dict
) -> float:
    relevant = set(query["relevant"])
    doc_ids = list(index["docs"].keys())
    feats = [extract_features(index, query["text"], d) for d in doc_ids]
    preds = booster.predict(np.asarray(feats, dtype=np.float32))
    ranked = sorted(
        zip(preds, doc_ids), key=lambda kv: (-float(kv[0]), kv[1])
    )
    retrieved = [index["docs"][d]["url"] for _, d in ranked]
    return average_precision(retrieved, relevant)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument(
        "--save",
        type=Path,
        default=DEFAULT_MODEL_OUT,
        help="Where to save the model trained on all 12 queries.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        print(
            "lightgbm is not installed. Install with `pip install lightgbm`."
        )
        return 1

    index = load(args.index)
    with args.gold.open(encoding="utf-8") as fp:
        queries = json.load(fp)["queries"]

    X, y, groups, query_ids = _build_dataset(index, queries)
    print(
        f"Built feature matrix: {X.shape[0]} rows, {X.shape[1]} features, "
        f"{len(groups)} queries"
    )
    print(f"Features: {FEATURE_NAMES}")

    bm25 = BM25Ranker()
    fold_ap_ltr: list[float] = []
    fold_ap_bm25: list[float] = []

    base_params = dict(
        objective="lambdarank",
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_data_in_leaf=5,
        lambda_l2=1.0,
        verbose=-1,
        seed=SEED,
    )

    for fold_idx, q in enumerate(queries):
        # Train on all queries except this one.
        train_mask = np.ones(len(queries), dtype=bool)
        train_mask[fold_idx] = False
        train_qids = [
            qid for qid, keep in zip(query_ids, train_mask) if keep
        ]
        # Build slices: each group is len(doc_ids).
        per_group = groups[0]
        train_rows = []
        train_labels = []
        for i, keep in enumerate(train_mask):
            if not keep:
                continue
            train_rows.append(X[i * per_group : (i + 1) * per_group])
            train_labels.append(y[i * per_group : (i + 1) * per_group])
        X_tr = np.concatenate(train_rows)
        y_tr = np.concatenate(train_labels)
        groups_tr = [per_group] * len(train_qids)

        train_set = lgb.Dataset(X_tr, label=y_tr, group=groups_tr)
        booster = lgb.train(
            base_params, train_set, num_boost_round=80
        )

        ap_ltr = _ap_for_query(booster, index, q)
        results = find(index, q["text"], ranker=bm25, top_k=args.top_k)
        ap_bm25 = average_precision(
            [r.url for r in results], set(q["relevant"])
        )
        fold_ap_ltr.append(ap_ltr)
        fold_ap_bm25.append(ap_bm25)
        print(
            f"  fold {q['id']}: LTR AP={ap_ltr:.3f}  BM25 AP={ap_bm25:.3f}  "
            f"(diff {ap_ltr - ap_bm25:+.3f})"
        )

    print()
    print(
        f"LTR  mean AP (LOQO):  {mean(fold_ap_ltr):.3f}"
    )
    print(
        f"BM25 mean AP:         {mean(fold_ap_bm25):.3f}"
    )
    wins = sum(
        1
        for a, b in zip(fold_ap_ltr, fold_ap_bm25)
        if a > b + 1e-9
    )
    print(f"LTR strictly beat BM25 on {wins} of {len(queries)} folds")

    # Final model trained on *all* queries — the artefact saved to disk.
    full = lgb.Dataset(X, label=y, group=groups)
    final = lgb.train(base_params, full, num_boost_round=80)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(args.save))
    print(f"Saved final model to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
