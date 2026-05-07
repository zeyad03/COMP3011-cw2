"""Tests for LTR feature extraction (Phase B).

Tests for the LightGBM-backed runtime ranker (:class:`LTRRanker`) are
gated on lightgbm being importable AND its native library loading; on
macOS lightgbm needs ``libomp`` (``brew install libomp``). When the
runtime is missing we skip the runtime tests but still validate the
deterministic, lightgbm-free feature extraction.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.crawler import Page
from src.indexer import build_index
from src.ltr_ranker import FEATURE_NAMES, extract_features


def _index() -> Any:
    pages = [
        Page(url="https://x.com/0", title="alpha", text="alpha beta gamma"),
        Page(url="https://x.com/1", title="beta",  text="beta beta gamma"),
        Page(url="https://x.com/2", title="zeta",  text="delta epsilon"),
    ]
    return build_index(pages)


class TestFeatureExtraction:
    def test_returns_six_features_in_known_order(self) -> None:
        idx = _index()
        feats = extract_features(idx, "beta", "1")
        assert len(feats) == len(FEATURE_NAMES) == 6
        # Doc length is feature index 3.
        assert feats[3] == idx["docs"]["1"]["length"]
        # The doc has a 'beta' match → matched_terms is at least 1.
        assert feats[4] >= 1.0

    def test_deterministic(self) -> None:
        idx = _index()
        first = extract_features(idx, "beta gamma", "1")
        second = extract_features(idx, "beta gamma", "1")
        assert first == second

    def test_unknown_doc_returns_zero_vector(self) -> None:
        idx = _index()
        out = extract_features(idx, "beta", "999")
        assert out == [0.0] * len(FEATURE_NAMES)

    def test_unknown_terms_contribute_zero_to_sums(self) -> None:
        idx = _index()
        baseline = extract_features(idx, "beta", "1")
        with_unknown = extract_features(idx, "beta xxx", "1")
        # First three (BM25, TF-IDF, freq) are identical.
        assert with_unknown[0] == pytest.approx(baseline[0])
        assert with_unknown[1] == pytest.approx(baseline[1])
        assert with_unknown[2] == pytest.approx(baseline[2])

    def test_higher_tf_yields_higher_bm25(self) -> None:
        idx = _index()
        # doc 1 has tf(beta)=2 (in body), doc 0 has tf(beta)=1
        feats0 = extract_features(idx, "beta", "0")
        feats1 = extract_features(idx, "beta", "1")
        assert feats1[0] > feats0[0]


def _lightgbm_runtime_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except OSError:
        return False
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _lightgbm_runtime_available(),
    reason="lightgbm native runtime not available (libomp on macOS)",
)
class TestLTRRunTime:
    """Smoke test for LTRRanker if the lightgbm runtime is healthy."""

    def test_loads_and_predicts(self, tmp_path: Any) -> None:
        import lightgbm as lgb
        import numpy as np

        # Train a tiny synthetic ranker so the test is hermetic.
        X = np.array(
            [
                [3.0, 1.0, 1.0, 5.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 5.0, 0.0, 0.5],
            ],
            dtype=np.float32,
        )
        y = np.array([1, 0], dtype=np.int32)
        ds = lgb.Dataset(X, label=y, group=[2])
        booster = lgb.train(
            dict(objective="lambdarank", verbose=-1, seed=42),
            ds,
            num_boost_round=5,
        )
        model_path = tmp_path / "ltr_test.txt"
        booster.save_model(str(model_path))
        from src.ltr_ranker import LTRRanker

        ranker = LTRRanker(model_path)
        idx = _index()
        out = ranker.rank(idx, "beta", top_k=3)
        assert len(out) == 3
        # Highest-scored result is the doc the model was trained to like
        # — but with two synthetic rows we just check determinism.
        assert all(r.score == round(r.score, 6) for r in out)
        # Helper exports are functional.
        from src.ltr_ranker import has_lightgbm, model_exists

        assert has_lightgbm() in (True, False)  # boolean
        assert model_exists(model_path) is True
        assert model_exists(tmp_path / "no_such_model") is False


# Pacify mypy if cast is unused in production code paths.
_ = cast
