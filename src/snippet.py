"""Snippet generation for search results.

Returns a short window of tokens from a document, centred on the
position with the highest density of query-term hits. Query-term hits
within the window are wrapped in markers (default ``**…**``) so the
matched words stand out in terminal output.

The implementation reuses the positional information already stored in
the inverted index — no schema change, no additional document body
needs to be persisted. The token sequence for a document is
reconstructed by gathering ``(position, term)`` pairs from every
posting that mentions that document and sorting by position.

Runtime is ``O(T)`` per document, where ``T`` is the number of unique
terms in the index (~4{,}600 for our corpus). Reconstruction is
deliberately *not* cached: snippet generation is opt-in via the
``--snippet`` CLI flag and is not in the hot path of an evaluation
run.
"""

from __future__ import annotations

from src.indexer import Index


DEFAULT_WINDOW = 20
MARKER_LEFT = "**"
MARKER_RIGHT = "**"


def reconstruct_tokens(index: Index, doc_id: str) -> list[str]:
    """Return the document's token sequence in original positional order."""
    tokens_at: dict[int, str] = {}
    for term, entry in index["terms"].items():
        posting = entry["postings"].get(doc_id)
        if posting is None:
            continue
        for position in posting["positions"]:
            tokens_at[position] = term
    return [tokens_at[p] for p in sorted(tokens_at)]


def generate_snippet(
    index: Index,
    doc_id: str,
    query_terms: list[str],
    *,
    window: int = DEFAULT_WINDOW,
) -> str:
    """Build a short excerpt of the document's text around query hits.

    Args:
        index: The inverted index.
        doc_id: ID of the document to excerpt.
        query_terms: Already-tokenised query words.
        window: Maximum number of tokens to show.

    Returns:
        The excerpt as a string. Hits are wrapped in ``**…**``.
        Returns an empty string for an unknown ``doc_id``.
    """
    if window <= 0:
        return ""
    tokens = reconstruct_tokens(index, doc_id)
    if not tokens:
        return ""

    query_set = set(query_terms)
    n = len(tokens)
    if n <= window:
        return _format(tokens, query_set)

    # Sliding window: find the start position whose window contains the
    # most query-term hits, with O(N) total cost.
    current = sum(1 for t in tokens[:window] if t in query_set)
    best_start = 0
    best_count = current
    for i in range(1, n - window + 1):
        if tokens[i - 1] in query_set:
            current -= 1
        if tokens[i + window - 1] in query_set:
            current += 1
        if current > best_count:
            best_count = current
            best_start = i

    snippet = _format(
        tokens[best_start : best_start + window], query_set
    )
    prefix = "... " if best_start > 0 else ""
    suffix = " ..." if best_start + window < n else ""
    return f"{prefix}{snippet}{suffix}"


def _format(tokens: list[str], query_set: set[str]) -> str:
    return " ".join(
        f"{MARKER_LEFT}{t}{MARKER_RIGHT}" if t in query_set else t
        for t in tokens
    )
