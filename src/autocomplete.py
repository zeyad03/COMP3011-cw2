"""Prefix-trie autocomplete over the indexed vocabulary.

Builds a Trie at index-load time over every term in the index. The
``suggest_prefix`` function returns up to *k* terms beginning with a
given prefix, sorted by collection frequency (``cf``) descending and
then alphabetically. Empty or unmatched prefixes return ``[]``.

The Trie is intentionally lightweight: each node is a plain ``dict``
mapping a single character to a child node, plus an optional terminal
``cf`` value. For a 5,000-term vocabulary the structure occupies
roughly the same memory as the term list itself; build cost is
``O(sum of term lengths)`` and prefix lookup is ``O(|prefix| + k)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.indexer import Index


@dataclass
class TrieNode:
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    # If non-None this node terminates a term and ``cf`` is its
    # collection frequency in the index.
    cf: int | None = None
    term: str | None = None


class TermTrie:
    """Prefix index over the term vocabulary."""

    def __init__(self) -> None:
        self._root = TrieNode()

    def insert(self, term: str, cf: int) -> None:
        node = self._root
        for ch in term:
            child = node.children.get(ch)
            if child is None:
                child = TrieNode()
                node.children[ch] = child
            node = child
        node.cf = cf
        node.term = term

    def suggest(self, prefix: str, *, k: int = 10) -> list[str]:
        """Return up to *k* terms with *prefix*, ranked by cf desc then term asc."""
        if not prefix or k <= 0:
            return []
        node = self._root
        for ch in prefix:
            child = node.children.get(ch)
            if child is None:
                return []
            node = child

        # Collect every terminal node in the subtree.
        candidates: list[tuple[int, str]] = []
        stack: list[TrieNode] = [node]
        while stack:
            current = stack.pop()
            if current.cf is not None and current.term is not None:
                candidates.append((current.cf, current.term))
            stack.extend(current.children.values())

        # Highest cf first; ties broken alphabetically for determinism.
        candidates.sort(key=lambda c: (-c[0], c[1]))
        return [term for _, term in candidates[:k]]


def build_trie(index: Index) -> TermTrie:
    """Build a :class:`TermTrie` over every term in *index*."""
    trie = TermTrie()
    for term, entry in index["terms"].items():
        trie.insert(term, entry["cf"])
    return trie


__all__ = ["TermTrie", "TrieNode", "build_trie"]
