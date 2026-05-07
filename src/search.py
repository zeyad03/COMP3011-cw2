"""Search over the inverted index.

Three public functions:

- :func:`print_term` — formats a single term's posting list for display.
- :func:`find` — multi-term search with pluggable scoring and an
  optional phrase mode that requires terms to appear in adjacent
  positions.
- :func:`suggest` — Levenshtein-based "did you mean?" for unknown query
  terms.

Ranking is delegated to a :class:`~src.ranker.Ranker`. Three are
provided in :mod:`src.ranker`: the naive ``FrequencyRanker``
(baseline), the default ``TFIDFRanker`` (sub-linear TF, smoothed IDF),
and ``BM25Ranker`` (Okapi BM25 with tunable ``k1`` and ``b``). The
ranker is injectable so callers — including the evaluation harness —
can compare scoring functions on the same index.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Literal

from src.indexer import Index, Posting
from src.query_eval import collect_terms, evaluate
from src.query_parser import ParseError, has_boolean_operators, parse
from src.ranker import Ranker, TFIDFRanker, supports_fields
from src.tokenizer import tokenize


SearchMode = Literal["and", "phrase"]


@dataclass(frozen=True)
class TermContribution:
    """Per-term breakdown of a document's score, for ``--explain`` output."""

    term: str
    tf: int
    df: int
    contribution: float


@dataclass(frozen=True)
class SearchResult:
    """A single ranked search hit.

    ``breakdown`` is empty unless :func:`find` was called with
    ``explain=True``; see the ``--explain`` CLI flag.
    """

    url: str
    title: str
    score: float
    matched_terms: tuple[str, ...]
    breakdown: tuple[TermContribution, ...] = ()


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
    ranker: Ranker | None = None,
    top_k: int | None = None,
    explain: bool = False,
) -> list[SearchResult]:
    """Find pages matching *query* using the given ranker.

    Args:
        index: The inverted index to search.
        query: User query string. Tokenised with the same tokeniser used
            during indexing, so case and punctuation match.
        mode: ``"and"`` requires every query token to appear in the
            matching document (any order). ``"phrase"`` additionally
            requires the tokens to appear in adjacent positions.
        ranker: Scoring function. Defaults to :class:`TFIDFRanker`.
            Pass :class:`FrequencyRanker` or :class:`BM25Ranker` to
            compare alternative scoring schemes against the same index.
        top_k: Optional cap on the number of results returned.
        explain: If true, attach a per-term ``breakdown`` to each
            :class:`SearchResult` so callers can see how each query
            term contributed to the document's score.

    Returns:
        Ranked list of :class:`SearchResult`, highest score first.
    """
    if ranker is None:
        ranker = TFIDFRanker()

    # ----------- Boolean query path (Phase C). -----------
    if has_boolean_operators(query):
        try:
            ast = parse(query)
        except ParseError:
            # Malformed boolean query: fall through to plain AND search.
            ast = None
        if ast is not None:
            scoring_terms = [
                t for t in collect_terms(ast) if t in index["terms"]
            ]
            if not scoring_terms:
                return []
            bool_candidates = evaluate(ast, index)
            if not bool_candidates:
                return []
            return _score_candidates(
                index=index,
                tokens=scoring_terms,
                postings_per_term=[
                    index["terms"][t]["postings"] for t in scoring_terms
                ],
                candidate_ids=bool_candidates,
                ranker=ranker,
                explain=explain,
                top_k=top_k,
            )

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

    return _score_candidates(
        index=index,
        tokens=tokens,
        postings_per_term=postings_per_term,
        candidate_ids=candidate_ids,
        ranker=ranker,
        explain=explain,
        top_k=top_k,
    )


def _score_candidates(
    *,
    index: Index,
    tokens: list[str],
    postings_per_term: list[dict[str, Posting]],
    candidate_ids: set[str],
    ranker: Ranker,
    explain: bool,
    top_k: int | None,
) -> list[SearchResult]:
    """Shared scoring loop for the AND-search and boolean-query paths.

    *tokens* and *postings_per_term* are positional pairs. A document
    that does not appear in a given posting list contributes 0 for that
    term (the boolean evaluator may admit such documents via Or/Not).
    """
    num_docs = index["meta"]["num_docs"]
    avg_doc_length = _avg_doc_length(index)
    avg_title_length = _avg_field_length(index, "title_length")
    avg_body_length = _avg_field_length(index, "body_length")
    use_fielded = supports_fields(ranker) and _index_has_fields(index)
    results: list[SearchResult] = []
    for doc_id in candidate_ids:
        doc = index["docs"][doc_id]
        score = 0.0
        contributions: list[TermContribution] = []
        for token, postings in zip(tokens, postings_per_term):
            posting = postings.get(doc_id)
            if posting is None:
                if explain:
                    df = index["terms"][token]["df"]
                    contributions.append(
                        TermContribution(
                            term=token, tf=0, df=df, contribution=0.0
                        )
                    )
                continue
            df = index["terms"][token]["df"]
            tf = posting["tf"]
            if use_fielded:
                term_score = ranker.score_fielded(  # type: ignore[attr-defined]
                    tf_title=len(posting.get("title_positions", [])),
                    tf_body=len(posting.get("body_positions", [])),
                    df=df,
                    num_docs=num_docs,
                    title_length=doc.get("title_length", 0),
                    body_length=doc.get(
                        "body_length", doc["length"]
                    ),
                    avg_title_length=avg_title_length,
                    avg_body_length=avg_body_length,
                )
            else:
                term_score = ranker.score(
                    tf=tf,
                    df=df,
                    num_docs=num_docs,
                    doc_length=doc["length"],
                    avg_doc_length=avg_doc_length,
                )
            score += term_score
            if explain:
                contributions.append(
                    TermContribution(
                        term=token,
                        tf=tf,
                        df=df,
                        contribution=round(term_score, 6),
                    )
                )
        results.append(
            SearchResult(
                url=doc["url"],
                title=doc["title"],
                score=round(score, 6),
                matched_terms=tuple(tokens),
                breakdown=tuple(contributions),
            )
        )

    # Stable secondary sort by URL keeps test fixtures deterministic.
    results.sort(key=lambda r: (-r.score, r.url))
    if top_k is not None:
        results = results[:top_k]
    return results


def _avg_doc_length(index: Index) -> float:
    docs = index["docs"]
    if not docs:
        return 0.0
    return sum(d["length"] for d in docs.values()) / len(docs)


def _avg_field_length(index: Index, key: str) -> float:
    """Average length over a per-field token-count key, or 0 on missing key."""
    docs = index["docs"]
    if not docs:
        return 0.0
    total = 0
    seen = 0
    for d in docs.values():
        if key in d:
            total += int(d[key])  # type: ignore[literal-required]
            seen += 1
    if seen == 0:
        return 0.0
    return total / seen


def _index_has_fields(index: Index) -> bool:
    """True if the index carries per-field positions (v1.2+)."""
    docs = index["docs"]
    if not docs:
        return False
    sample = next(iter(docs.values()))
    return "title_length" in sample and "body_length" in sample


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
    symspell: object | None = None,
) -> list[str]:
    """Return known terms within *max_distance* edits of *word*.

    Returns ``[]`` if *word* is itself in the index (no need to suggest)
    or empty. Suggestions are ordered by edit distance ascending, then
    alphabetically.

    If a pre-built :class:`~src.symspell.SymSpell` index is passed via
    *symspell*, the lookup uses the SymSpell deletion-dictionary
    algorithm (O(L^d) instead of O(N·L²) for vocabulary size N). The
    output is identical to the Levenshtein scan; only the runtime
    differs. When *symspell* is ``None`` the original Levenshtein scan
    is used so callers without a pre-built index still get correct
    results.
    """
    tokens = tokenize(word)
    if not tokens:
        return []
    target = tokens[0]
    if target in index["terms"]:
        return []

    if symspell is not None:
        # Delegated path: SymSpell.lookup post-verifies with _levenshtein,
        # so output ordering matches the linear scan exactly.
        return symspell.lookup(  # type: ignore[no-any-return,attr-defined]
            target,
            max_distance=max_distance,
            max_suggestions=max_suggestions,
        )

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
