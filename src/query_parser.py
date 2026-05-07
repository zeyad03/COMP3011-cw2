"""Boolean query parser with operator precedence (Phase C).

Parses queries of the form::

    (machine OR computer) AND learning NOT statistics

into an abstract syntax tree of :class:`Term`, :class:`Not`,
:class:`And`, :class:`Or` nodes. The grammar is::

    expr   := or_expr
    or_expr  := and_expr ("OR" and_expr)*
    and_expr := not_expr ("AND" not_expr)*
    not_expr := "NOT" not_expr | atom
    atom   := TERM | "(" expr ")"

Implicit AND is *not* supported: queries without explicit operators are
handled by :func:`src.search.find` directly. Operators are case-insensitive
on input but emitted as upper-case in :class:`ParseError` messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

from src.tokenizer import tokenize


@dataclass(frozen=True)
class Term:
    """A leaf node naming a single tokenised term."""

    text: str


@dataclass(frozen=True)
class Not:
    operand: "BoolExpr"


@dataclass(frozen=True)
class And:
    left: "BoolExpr"
    right: "BoolExpr"


@dataclass(frozen=True)
class Or:
    left: "BoolExpr"
    right: "BoolExpr"


BoolExpr = Union[Term, Not, And, Or]


class ParseError(ValueError):
    """Raised on malformed boolean queries."""


_OPERATORS = {"AND", "OR", "NOT", "(", ")"}
_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


def has_boolean_operators(query: str) -> bool:
    """Cheap detector for queries that need the boolean parser."""
    upper = query.upper()
    if "(" in query or ")" in query:
        return True
    return any(
        re.search(rf"(?:^|\s){op}(?:\s|$)", upper) for op in ("AND", "OR", "NOT")
    )


def _tokenize(query: str) -> list[str]:
    """Split *query* into operator/term tokens, normalising terms."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(query):
        upper = raw.upper()
        if upper in _OPERATORS:
            out.append(upper)
            continue
        # Normalise the term identically to indexing so lookups match.
        normalised = tokenize(raw)
        if not normalised:
            continue  # Punctuation-only token: drop.
        out.append(normalised[0])
    return out


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._pos = 0

    def parse(self) -> BoolExpr:
        if not self._tokens:
            raise ParseError("empty query")
        node = self._parse_or()
        if self._pos != len(self._tokens):
            raise ParseError(
                f"unexpected token {self._tokens[self._pos]!r} at position "
                f"{self._pos}"
            )
        return node

    def _peek(self) -> str | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self) -> str:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _parse_or(self) -> BoolExpr:
        left = self._parse_and()
        while self._peek() == "OR":
            self._consume()
            right = self._parse_and()
            left = Or(left, right)
        return left

    def _parse_and(self) -> BoolExpr:
        left = self._parse_not()
        while self._peek() == "AND":
            self._consume()
            right = self._parse_not()
            left = And(left, right)
        return left

    def _parse_not(self) -> BoolExpr:
        if self._peek() == "NOT":
            self._consume()
            return Not(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> BoolExpr:
        tok = self._peek()
        if tok is None:
            raise ParseError("expected term or '(' but got end of query")
        if tok == "(":
            self._consume()
            inner = self._parse_or()
            if self._peek() != ")":
                raise ParseError("missing closing ')'")
            self._consume()
            return inner
        if tok in {"AND", "OR", ")"}:
            raise ParseError(f"unexpected operator {tok!r}")
        # Plain term.
        self._consume()
        return Term(tok)


def parse(query: str) -> BoolExpr:
    """Parse *query* into a :class:`BoolExpr` AST."""
    return _Parser(_tokenize(query)).parse()


__all__ = [
    "And",
    "BoolExpr",
    "Not",
    "Or",
    "ParseError",
    "Term",
    "has_boolean_operators",
    "parse",
]
