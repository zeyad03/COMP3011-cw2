"""Cross-validate this project's TF-IDF ranker against scikit-learn.

For each gold-standard query, runs the project's :class:`TFIDFRanker`
and a ``sklearn.feature_extraction.text.TfidfVectorizer`` +
cosine-similarity baseline over the same corpus, then reports the
**top-5 overlap** — what fraction of our top-5 also appears in
sklearn's top-5.

Why this matters
----------------
Two TF-IDF implementations rarely produce *byte-identical* rankings —
they differ on document-length normalisation, IDF smoothing, and
sub-linear TF. But they should largely agree on which documents are
relevant. A high overlap is evidence that this project's hand-built
ranker is in the same ballpark as a battle-tested library; a low
overlap raises a flag worth investigating.

Run:

    python evaluation/compare_sklearn.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

from src.metrics import mean  # noqa: E402
from src.ranker import TFIDFRanker  # noqa: E402
from src.search import find  # noqa: E402
from src.storage import load  # noqa: E402
from src.tokenizer import tokenize  # noqa: E402


GOLD = Path(__file__).resolve().parent / "gold_standard.json"
INDEX = ROOT / "data" / "index.json"


def _build_sklearn_corpus(index: dict) -> tuple[list[str], list[str]]:
    """Return (urls, document_texts) reconstructed from the inverted index.

    The index stores positions per term, which lets us regenerate each
    document as a bag-of-tokens. (Order is not preserved, but TF-IDF
    is order-insensitive so cosine similarity is unchanged.)
    """
    urls: list[str] = []
    docs: list[str] = []
    for doc_id, doc_meta in sorted(
        index["docs"].items(), key=lambda kv: int(kv[0])
    ):
        tokens: list[tuple[int, str]] = []
        for term, entry in index["terms"].items():
            posting = entry["postings"].get(doc_id)
            if posting is not None:
                for pos in posting["positions"]:
                    tokens.append((pos, term))
        tokens.sort()  # restore positional order so the doc reads naturally
        urls.append(doc_meta["url"])
        docs.append(" ".join(term for _, term in tokens))
    return urls, docs


def _sklearn_top_k(
    vectoriser: TfidfVectorizer,
    matrix,
    urls: list[str],
    query: str,
    k: int,
) -> list[str]:
    query_vec = vectoriser.transform([" ".join(tokenize(query))])
    sims = cosine_similarity(query_vec, matrix)[0]
    # argsort ascending; take the last k and reverse for descending.
    top_indices = sims.argsort()[-k:][::-1]
    return [urls[i] for i in top_indices if sims[i] > 0]


def main() -> int:
    index = load(INDEX)
    urls, docs = _build_sklearn_corpus(index)

    # Match our own tokeniser by passing token_pattern=None and a
    # custom tokenizer wouldn't quite reproduce — instead we feed
    # already-tokenised text and let sklearn split on whitespace.
    vectoriser = TfidfVectorizer(
        lowercase=False,  # already lowercased by our tokeniser
        token_pattern=r"(?u)\b\S+\b",
        sublinear_tf=True,
    )
    matrix = vectoriser.fit_transform(docs)

    with GOLD.open(encoding="utf-8") as fp:
        gold = json.load(fp)
    queries = gold["queries"]

    project_ranker = TFIDFRanker()
    overlaps: list[float] = []

    print(
        f"{'query':<28}  {'ours top-5':>10}  "
        f"{'sklearn top-5':>14}  {'overlap':>8}"
    )
    print("-" * 70)
    for q in queries:
        text = q["text"]
        ours = {
            r.url
            for r in find(index, text, ranker=project_ranker, top_k=5)
        }
        theirs = set(_sklearn_top_k(vectoriser, matrix, urls, text, 5))
        if not ours and not theirs:
            overlap = 1.0  # both agree there's nothing
        elif not ours or not theirs:
            overlap = 0.0
        else:
            overlap = len(ours & theirs) / max(len(ours), len(theirs))
        overlaps.append(overlap)
        print(
            f"{text:<28}  {len(ours):>10d}  "
            f"{len(theirs):>14d}  {overlap:>8.2f}"
        )

    print("-" * 70)
    print(f"mean top-5 overlap: {mean(overlaps):.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
