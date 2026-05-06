"""Tests for the tokeniser.

Coverage targets:
    - Empty / whitespace-only / pure-punctuation inputs return ``[]``.
    - Case folding.
    - Punctuation stripping (leading, trailing, internal).
    - Apostrophe collapse, including curly Unicode apostrophes.
    - Whitespace handling: spaces, tabs, newlines, runs.
    - Digits preserved.
    - Accented characters preserved (no diacritic stripping).
    - Em-dashes treated as separators.
"""

from __future__ import annotations

import pytest

from src.tokenizer import tokenize


class TestEmptyAndDegenerateInput:
    def test_empty_string_returns_empty_list(self) -> None:
        assert tokenize("") == []

    def test_whitespace_only_returns_empty_list(self) -> None:
        assert tokenize("   \t\n  ") == []

    def test_punctuation_only_returns_empty_list(self) -> None:
        assert tokenize("...!?") == []


class TestCaseFolding:
    def test_uppercase_is_lowercased(self) -> None:
        assert tokenize("HELLO") == ["hello"]

    def test_mixed_case_is_lowercased(self) -> None:
        assert tokenize("Hello World") == ["hello", "world"]


class TestPunctuationStripping:
    def test_strips_trailing_punctuation(self) -> None:
        assert tokenize("hello!") == ["hello"]

    def test_strips_leading_punctuation(self) -> None:
        assert tokenize("(hello") == ["hello"]

    def test_splits_on_internal_punctuation(self) -> None:
        assert tokenize("hello-world") == ["hello", "world"]

    def test_full_sentence(self) -> None:
        assert tokenize("It's a truth, universally acknowledged.") == [
            "its",
            "a",
            "truth",
            "universally",
            "acknowledged",
        ]


class TestApostropheCollapse:
    def test_ascii_apostrophe_in_contraction_collapses(self) -> None:
        assert tokenize("don't") == ["dont"]

    def test_curly_apostrophe_collapses_like_ascii(self) -> None:
        assert tokenize("don’t") == ["dont"]

    def test_apostrophe_in_possessive_collapses(self) -> None:
        assert tokenize("Sherlock's") == ["sherlocks"]


class TestWhitespace:
    def test_multiple_spaces_collapse(self) -> None:
        assert tokenize("hello    world") == ["hello", "world"]

    def test_tabs_and_newlines_split(self) -> None:
        assert tokenize("hello\tworld\nfoo") == ["hello", "world", "foo"]


class TestDigitsAndUnicode:
    def test_digits_are_kept_as_tokens(self) -> None:
        assert tokenize("1984 was a year") == ["1984", "was", "a", "year"]

    def test_accented_characters_preserved(self) -> None:
        assert tokenize("café") == ["café"]

    def test_em_dash_is_a_separator(self) -> None:
        assert tokenize("good — friends") == ["good", "friends"]


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("", []),
        ("   ", []),
        ("foo", ["foo"]),
        ("Foo BAR", ["foo", "bar"]),
        ("foo, bar; baz!", ["foo", "bar", "baz"]),
    ],
)
def test_parametrised_basic_cases(input_text: str, expected: list[str]) -> None:
    assert tokenize(input_text) == expected
