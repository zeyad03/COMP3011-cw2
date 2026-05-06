"""Inverted-index construction from crawled pages.

The indexer takes the output of the crawler — a list of :class:`Page`
records — and produces an in-memory inverted index keyed by term, with
per-document term frequency, token positions, and document frequency.
The same tokeniser used here is also used by the search module, so
queries and indexed text are normalised identically.

Index schema (also reproduced in ``docs/design.md``)::

    {
      "meta": {
        "crawled_at":    ISO-8601 UTC timestamp,
        "num_docs":      int,
        "num_terms":     int,
        "total_tokens":  int,
        "version":       "1.0",
      },
      "docs": {
        "<doc_id>": {
          "url":    str,
          "title":  str,
          "length": int,            # token count (title + body)
        }
      },
      "terms": {
        "<term>": {
          "df":       int,           # document frequency
          "cf":       int,           # collection frequency
          "postings": {
            "<doc_id>": {
              "tf":        int,
              "positions": [int, ...],
            }
          }
        }
      }
    }

Document IDs are sequential string integers (``"0"``, ``"1"``, …) assigned
in the order pages are supplied. The choice of *string* keys is
deliberate: JSON object keys must be strings, and using string IDs from
the start avoids a serialisation surprise later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, TypedDict

from src.crawler import Page
from src.tokenizer import tokenize

INDEX_VERSION: str = "1.0"


class DocMeta(TypedDict):
    url: str
    title: str
    length: int


class Posting(TypedDict):
    tf: int
    positions: list[int]


class TermEntry(TypedDict):
    df: int
    cf: int
    postings: dict[str, Posting]


class IndexMeta(TypedDict):
    crawled_at: str
    num_docs: int
    num_terms: int
    total_tokens: int
    version: str


class Index(TypedDict):
    meta: IndexMeta
    docs: dict[str, DocMeta]
    terms: dict[str, TermEntry]


def build_index(pages: Iterable[Page]) -> Index:
    """Build an inverted index from *pages*.

    Args:
        pages: Iterable of crawled :class:`Page` records.

    Returns:
        An :class:`Index` dictionary matching the schema in the module
        docstring.
    """
    docs: dict[str, DocMeta] = {}
    terms: dict[str, TermEntry] = {}
    total_tokens = 0

    for doc_index, page in enumerate(pages):
        doc_id = str(doc_index)
        tokens = tokenize(f"{page.title} {page.text}")
        docs[doc_id] = {
            "url": page.url,
            "title": page.title,
            "length": len(tokens),
        }
        total_tokens += len(tokens)

        # Group token positions per term within this single document, so we
        # can update the global index with one entry per (term, doc) pair.
        local_positions: dict[str, list[int]] = {}
        for position, token in enumerate(tokens):
            local_positions.setdefault(token, []).append(position)

        for term, positions in local_positions.items():
            entry = terms.get(term)
            if entry is None:
                entry = {"df": 0, "cf": 0, "postings": {}}
                terms[term] = entry
            entry["df"] += 1
            entry["cf"] += len(positions)
            entry["postings"][doc_id] = {
                "tf": len(positions),
                "positions": positions,
            }

    return {
        "meta": {
            "crawled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "num_docs": len(docs),
            "num_terms": len(terms),
            "total_tokens": total_tokens,
            "version": INDEX_VERSION,
        },
        "docs": docs,
        "terms": terms,
    }
