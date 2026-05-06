"""Search over the inverted index.

Three public functions:

- :func:`print_term` — formats a single term's posting list for display.
- :func:`find` — multi-term search with TF-IDF ranking and an optional
  phrase mode that requires terms to appear in adjacent positions.
- :func:`suggest` — Levenshtein-based "did you mean?" for unknown query
  terms.

Ranking
-------

Per-term contribution to the document score uses a sub-linear TF and a
smoothed IDF:

    score(t, d)  = (1 + log10(tf(t, d))) * log10(1 + N / df(t))

* The ``1 + log10(tf)`` form dampens the influence of very-frequent
  terms within a single document (so a doc that says "foo" 100 times
  doesn't rank 100× higher than one that mentions it once).
* The ``1 + N/df`` form keeps IDF strictly positive even when a term
  appears in *every* document (i.e. ``df == N``), avoiding the
  degenerate all-zeros case that hits small corpora hard.

The total document score is the sum of per-term contributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import StringIO
from typing import Literal

from src.indexer import Index, Posting
from src.tokenizer import tokenize


SearchMode = Literal["and", "phrase"]


@dataclass(frozen=True)
class SearchResult:
    """A single ranked search hit."""

    url: str
    title: str
    score: float
    matched_terms: tuple[str, ...]


# ----------------------------------------------------------------- print_term


def print_term(index: Index, word: str) -> str:
    """Return a printable representation of *word*'s posting list."""
    tokens = tokenize(word)
    if not tokens:
        return "empty term"
    term = tokens[0]
    entry = index["terms"].get(term)
    if entry is None:
        return f"term {term!r} not found in index"

    out = StringIO()
    out.write(f"{term!r}  df={entry['df']}  cf={entry['cf']}\n")
    for doc_id, posting in entry["postings"].items():
        url = index["docs"][doc_id]["url"]
        out.write(
            f"  {url}  tf={posting['tf']}  positions={posting['positions']}\n"
        )
    return out.getvalue().rstrip()


# ----------------------------------------------------------------------- find


def find(
    index: Index,
    query: str,
    *,
    mode: SearchMode = "and",
    top_k: int | None = None,
) -> list[SearchResult]:
    """Find pages matching *query* using TF-IDF ranking.

    Args:
        index: The inverted index to search.
        query: User query string. Tokenised with the same tokeniser used
            during indexing, so case and punctuation match.
        mode: ``"and"`` requires every query token to appear in the
            matching document (any order). ``"phrase"`` additionally
            requires the tokens to appear in adjacent positions.
        top_k: Optional cap on the number of results returned.

    Returns:
        Ranked list of :class:`SearchResult`, highest score first.
    """
    tokens = tokenize(query)
    if not tokens:
        return []

    postings_per_term: list[dict[str, Posting]] = []
    for token in tokens:
        entry = index["terms"].get(token)
        if entry is None:
            return []
        postings_per_term.append(entry["postings"])

    # AND intersection: start from the smallest posting list for speed.
    postings_sorted = sorted(postings_per_term, key=len)
    candidate_ids: set[str] = set(postings_sorted[0])
    for postings in postings_sorted[1:]:
        candidate_ids &= postings.keys()

    if not candidate_ids:
        return []

    if mode == "phrase" and len(tokens) > 1:
        candidate_ids = {
            doc_id
            for doc_id in candidate_ids
            if _is_adjacent_in_doc(postings_per_term, doc_id)
        }
        if not candidate_ids:
            return []

    num_docs = index["meta"]["num_docs"]
    results: list[SearchResult] = []
    for doc_id in candidate_ids:
        doc = index["docs"][doc_id]
        score = 0.0
        for token, postings in zip(tokens, postings_per_term):
            df = index["terms"][token]["df"]
            tf = postings[doc_id]["tf"]
            score += _term_score(tf=tf, df=df, num_docs=num_docs)
        results.append(
            SearchResult(
                url=doc["url"],
                title=doc["title"],
                score=round(score, 6),
                matched_terms=tuple(tokens),
            )
        )

    # Stable secondary sort by URL keeps test fixtures deterministic.
    results.sort(key=lambda r: (-r.score, r.url))
    if top_k is not None:
        results = results[:top_k]
    return results


def _term_score(*, tf: int, df: int, num_docs: int) -> float:
    if tf <= 0 or df <= 0 or num_docs <= 0:
        return 0.0
    tf_weight = 1.0 + math.log10(tf)
    idf_weight = math.log10(1.0 + num_docs / df)
    return tf_weight * idf_weight


def _is_adjacent_in_doc(
    postings_per_term: list[dict[str, Posting]], doc_id: str
) -> bool:
    """True if the query terms appear in adjacent positions in *doc_id*."""
    position_sets = [set(p[doc_id]["positions"]) for p in postings_per_term]
    n = len(position_sets)
    for start in position_sets[0]:
        if all((start + i) in position_sets[i] for i in range(n)):
            return True
    return False


# -------------------------------------------------------------------- suggest


def suggest(
    index: Index,
    word: str,
    *,
    max_distance: int = 2,
    max_suggestions: int = 5,
) -> list[str]:
    """Return known terms within *max_distance* edits of *word*.

    Returns ``[]`` if *word* is itself in the index (no need to suggest)
    or empty. Suggestions are ordered by edit distance ascending, then
    alphabetically.
    """
    tokens = tokenize(word)
    if not tokens:
        return []
    target = tokens[0]
    if target in index["terms"]:
        return []

    candidates: list[tuple[int, str]] = []
    for term in index["terms"]:
        # Length-difference is a free lower bound on edit distance.
        if abs(len(term) - len(target)) > max_distance:
            continue
        distance = _levenshtein(target, term, max_distance)
        if distance <= max_distance:
            candidates.append((distance, term))
    candidates.sort()
    return [term for _, term in candidates[:max_suggestions]]


def _levenshtein(a: str, b: str, max_distance: int) -> int:
    """Levenshtein edit distance between *a* and *b*, capped at *max_distance*+1.

    Standard two-row dynamic-programming implementation. Bails out as
    soon as every cell in a row exceeds *max_distance* (no way to
    recover within the budget).
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            value = min(
                prev[j] + 1,         # delete
                curr[j - 1] + 1,     # insert
                prev[j - 1] + cost,  # substitute
            )
            curr.append(value)
            if value < row_min:
                row_min = value
        if row_min > max_distance:
            return max_distance + 1
        prev = curr
    return prev[-1]
