"""Text tokenisation for indexing and querying.

Tokenisation is the single source of truth shared by the indexer (which
processes crawled page text) and the search module (which processes user
queries). Keeping both sides on the same tokeniser is essential: if an
index is built with one normalisation rule and queried with another, the
two will silently disagree and pages will appear "missing" from results.

Rules
-----
1. Case-folded to lowercase. The brief specifies case-insensitive search
   ('Good' == 'good').
2. Apostrophes are collapsed: ``don't`` -> ``dont``. This keeps
   contractions as a single searchable unit; splitting them yields
   low-information fragments like ``t``.
3. Curly Unicode quotes are normalised to ASCII first, so smart-quote
   apostrophes scraped from web copy collapse the same way.
4. All other non-word characters (Unicode-aware via the ``re`` module)
   become token separators.
5. Digits are preserved as tokens (e.g. ``1984``).
6. Accented characters are preserved (e.g. ``café``). Diacritics are not
   stripped; a query for ``cafe`` will not match ``café``. This is a
   deliberate trade-off documented in ``docs/design.md``.
"""

from __future__ import annotations

import re

_QUOTE_NORMALISATION = str.maketrans(
    {
        "‘": "'",  # left single quotation mark
        "’": "'",  # right single quotation mark
        "“": '"',  # left double quotation mark
        "”": '"',  # right double quotation mark
    }
)

_NON_WORD = re.compile(r"[^\w]+")


def tokenize(text: str) -> list[str]:
    """Split *text* into normalised tokens.

    Args:
        text: Arbitrary input text. May be empty or contain Unicode.

    Returns:
        A list of lowercase tokens in the order they appear, with
        punctuation and whitespace stripped and apostrophes collapsed.
    """
    if not text:
        return []
    text = text.translate(_QUOTE_NORMALISATION)
    text = text.lower()
    text = text.replace("'", "")
    text = _NON_WORD.sub(" ", text)
    return text.split()
