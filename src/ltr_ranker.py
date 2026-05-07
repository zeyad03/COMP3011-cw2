"""Learning-to-rank scorer (Phase B).

A LightGBM LambdaRank model trained on the 12-query gold standard
acts as the fifth ranker. Training lives in
``evaluation/learning_to_rank.py``; this module is the *runtime* side:
it loads a saved model and produces per-document scores from a feature
vector built out of the existing sparse signals.

Features (six per (query-term, document) pair, summed across query
terms):

1. BM25 sum
2. TF-IDF sum
3. Raw frequency sum
4. Document length
5. Number of query terms in the document
6. Average query-term IDF

Feature ordering is stable and shared with the trainer. Changing it
here without retraining the model would silently break predictions.

LightGBM's text-format model file is the persistence boundary; the
runtime depends on lightgbm being installed but not on numpy or any
other heavy stack.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.indexer import Index
from src.ranker import BM25Ranker, FrequencyRanker, TFIDFRanker
from src.search import SearchResult
from src.tokenizer import tokenize


FEATURE_NAMES = (
    "bm25_sum",
    "tfidf_sum",
    "freq_sum",
    "doc_length",
    "matched_terms",
    "avg_query_idf",
)


def extract_features(
    index: Index,
    query: str,
    doc_id: str,
) -> list[float]:
    """Build the feature vector for one (query, doc) pair.

    The function is deterministic and pure: same inputs ⇒ same output.
    Missing query terms contribute 0 to all sums, in line with the
    additive scoring convention used by the sparse rankers.
    """
    tokens = tokenize(query)
    if not tokens or doc_id not in index["docs"]:
        return [0.0] * len(FEATURE_NAMES)
    doc = index["docs"][doc_id]
    num_docs = index["meta"]["num_docs"]
    avg_doc_length = (
        sum(d["length"] for d in index["docs"].values()) / num_docs
        if num_docs
        else 0.0
    )
    bm25 = BM25Ranker()
    tfidf = TFIDFRanker()
    freq = FrequencyRanker()

    bm25_sum = 0.0
    tfidf_sum = 0.0
    freq_sum = 0.0
    matched = 0
    idf_sum = 0.0
    idf_count = 0
    for term in tokens:
        entry = index["terms"].get(term)
        if entry is None:
            continue
        # IDF over all query terms — including those not in this doc —
        # is the average idf signal in feature 6.
        idf_count += 1
        idf_sum += math.log10(
            1.0 + (num_docs - entry["df"] + 0.5) / (entry["df"] + 0.5)
        )
        posting = entry["postings"].get(doc_id)
        if posting is None:
            continue
        matched += 1
        df = entry["df"]
        tf = posting["tf"]
        bm25_sum += bm25.score(
            tf=tf,
            df=df,
            num_docs=num_docs,
            doc_length=doc["length"],
            avg_doc_length=avg_doc_length,
        )
        tfidf_sum += tfidf.score(
            tf=tf,
            df=df,
            num_docs=num_docs,
            doc_length=doc["length"],
            avg_doc_length=avg_doc_length,
        )
        freq_sum += freq.score(
            tf=tf,
            df=df,
            num_docs=num_docs,
            doc_length=doc["length"],
            avg_doc_length=avg_doc_length,
        )
    avg_idf = idf_sum / idf_count if idf_count else 0.0
    return [
        bm25_sum,
        tfidf_sum,
        freq_sum,
        float(doc["length"]),
        float(matched),
        avg_idf,
    ]


class LTRRanker:
    """LightGBM-backed list-level ranker.

    The trained model is loaded eagerly at construction time so the
    cost is paid once, not per query. ``rank`` returns a sorted list of
    :class:`SearchResult`.
    """

    name = "ltr"

    def __init__(self, model_path: Path | str) -> None:
        try:
            import lightgbm as lgb
        except ImportError as err:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "lightgbm is required for the LTR ranker; install with "
                "`pip install lightgbm`"
            ) from err
        self._lgb = lgb
        self._booster = lgb.Booster(model_file=str(model_path))

    def rank(
        self,
        index: Index,
        query: str,
        *,
        candidate_ids: set[str] | None = None,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Score every candidate doc and return them sorted desc by score.

        If *candidate_ids* is given, only those documents are scored;
        otherwise the full corpus is scored (slow on a real index, but
        consistent with the other rankers' behaviour at small scale).
        """
        if candidate_ids is None:
            candidate_ids = set(index["docs"].keys())
        if not candidate_ids:
            return []
        features: list[list[float]] = []
        ordered_ids: list[str] = []
        for doc_id in candidate_ids:
            features.append(extract_features(index, query, doc_id))
            ordered_ids.append(doc_id)
        scores = self._booster.predict(features)
        out: list[SearchResult] = []
        for doc_id, score in zip(ordered_ids, scores):
            doc = index["docs"][doc_id]
            out.append(
                SearchResult(
                    url=doc["url"],
                    title=doc["title"],
                    score=round(float(score), 6),
                    matched_terms=tuple(tokenize(query)),
                )
            )
        out.sort(key=lambda r: (-r.score, r.url))
        if top_k is not None:
            out = out[:top_k]
        return out


def model_exists(path: Path | str) -> bool:
    """Check if a saved LTR model is on disk."""
    return Path(path).exists()


def has_lightgbm() -> bool:
    """Return True if lightgbm can be imported."""
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = [
    "FEATURE_NAMES",
    "LTRRanker",
    "extract_features",
    "has_lightgbm",
    "model_exists",
]


def _typing_helpers() -> Any:  # pragma: no cover - typing pacifier
    return None
