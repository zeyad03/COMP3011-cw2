"""Tests for the CLI REPL."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from src.crawler import Page
from src.indexer import build_index
from src.main import CLI
from src.storage import save


def _run(commands: list[str], **cli_kwargs: Any) -> str:
    """Drive a fresh :class:`CLI` with the given command lines.

    Always appends ``exit`` so the REPL terminates cleanly.
    """
    stdin = StringIO("\n".join(commands + ["exit"]) + "\n")
    stdout = StringIO()
    cli = CLI(stdin=stdin, stdout=stdout, **cli_kwargs)
    cli.run()
    return stdout.getvalue()


def _saved_index(tmp_path: Path) -> Path:
    """Build and save a small index for load/print/find tests."""
    pages = [
        Page(url="https://x.com/p0", title="", text="hello world"),
        Page(url="https://x.com/p1", title="", text="hello again friends"),
        Page(url="https://x.com/p2", title="", text="goodbye friends"),
    ]
    index = build_index(pages)
    target = tmp_path / "idx.json"
    save(index, target)
    return target


# ----------------------------------------------------------------- basic loop


class TestREPL:
    def test_help_lists_all_commands(self) -> None:
        out = _run(["help"])
        for cmd in ("build", "load", "print", "find", "help", "exit"):
            assert cmd in out

    def test_unknown_command_message(self) -> None:
        out = _run(["fnord"])
        assert "unknown command" in out
        assert "fnord" in out

    def test_blank_line_does_nothing(self) -> None:
        out = _run([""])
        assert "unknown command" not in out

    def test_quit_alias_exits(self) -> None:
        # Use only "quit" without trailing exit — the REPL should still terminate.
        stdin = StringIO("quit\n")
        stdout = StringIO()
        cli = CLI(stdin=stdin, stdout=stdout)
        cli.run()
        # If we got here without hanging, the alias worked.

    def test_eof_exits_cleanly(self) -> None:
        stdin = StringIO("")
        stdout = StringIO()
        cli = CLI(stdin=stdin, stdout=stdout)
        assert cli.run() == 0


class TestNoIndexState:
    def test_print_before_build_or_load(self) -> None:
        out = _run(["print foo"])
        assert "no index loaded" in out

    def test_find_before_build_or_load(self) -> None:
        out = _run(["find foo"])
        assert "no index loaded" in out


# --------------------------------------------------------------------- load


class TestLoad:
    def test_load_existing_index_succeeds(self, tmp_path: Path) -> None:
        index_path = _saved_index(tmp_path)
        out = _run(["load"], index_path=index_path)
        assert "loaded" in out
        assert "3 docs" in out

    def test_load_missing_file_reports_error(self, tmp_path: Path) -> None:
        out = _run(["load"], index_path=tmp_path / "nope.json")
        assert "load failed" in out


# --------------------------------------------------------------------- print


class TestPrintCommand:
    def test_print_known_term(self, tmp_path: Path) -> None:
        out = _run(["load", "print hello"], index_path=_saved_index(tmp_path))
        assert "hello" in out
        assert "df=2" in out

    def test_print_unknown_term(self, tmp_path: Path) -> None:
        out = _run(["load", "print xyzzy"], index_path=_saved_index(tmp_path))
        assert "not found" in out

    def test_print_without_arg(self, tmp_path: Path) -> None:
        out = _run(["load", "print"], index_path=_saved_index(tmp_path))
        assert "usage: print" in out


# --------------------------------------------------------------------- find


class TestFindCommand:
    def test_find_single_term(self, tmp_path: Path) -> None:
        out = _run(["load", "find hello"], index_path=_saved_index(tmp_path))
        assert "https://x.com/p0" in out
        assert "https://x.com/p1" in out
        assert "score=" in out

    def test_find_multi_term_and(self, tmp_path: Path) -> None:
        out = _run(
            ["load", "find hello friends"], index_path=_saved_index(tmp_path)
        )
        # Only p1 has both 'hello' and 'friends'.
        assert "https://x.com/p1" in out
        assert "https://x.com/p0" not in out
        assert "https://x.com/p2" not in out

    def test_find_no_results_message(self, tmp_path: Path) -> None:
        out = _run(["load", "find xyzzy"], index_path=_saved_index(tmp_path))
        assert "no results" in out

    def test_find_offers_suggestion_for_typo(self, tmp_path: Path) -> None:
        out = _run(["load", "find helo"], index_path=_saved_index(tmp_path))
        assert "did you mean" in out
        assert "hello" in out

    def test_find_without_args(self, tmp_path: Path) -> None:
        out = _run(["load", "find"], index_path=_saved_index(tmp_path))
        assert "usage: find" in out

    def test_find_with_bm25_ranker(self, tmp_path: Path) -> None:
        out = _run(
            ["load", "find --ranker bm25 hello"],
            index_path=_saved_index(tmp_path),
        )
        assert "https://x.com/" in out
        assert "score=" in out

    def test_find_with_unknown_ranker(self, tmp_path: Path) -> None:
        out = _run(
            ["load", "find --ranker bogus hello"],
            index_path=_saved_index(tmp_path),
        )
        assert "unknown ranker" in out

    def test_find_only_ranker_flag_no_query(self, tmp_path: Path) -> None:
        out = _run(
            ["load", "find --ranker bm25"],
            index_path=_saved_index(tmp_path),
        )
        assert "usage: find" in out

    def test_find_explain_prints_per_term_breakdown(
        self, tmp_path: Path
    ) -> None:
        out = _run(
            ["load", "find --explain hello"],
            index_path=_saved_index(tmp_path),
        )
        # Each result should be followed by a breakdown line containing
        # tf=, df=, and contribution=.
        assert "tf=" in out
        assert "df=" in out
        assert "contribution=" in out

    def test_find_without_explain_omits_breakdown(
        self, tmp_path: Path
    ) -> None:
        out = _run(
            ["load", "find hello"],
            index_path=_saved_index(tmp_path),
        )
        assert "contribution=" not in out

    def test_find_snippet_marks_query_terms(self, tmp_path: Path) -> None:
        out = _run(
            ["load", "find --snippet hello"],
            index_path=_saved_index(tmp_path),
        )
        # Query terms should be wrapped in ** markers in the snippet.
        assert "**hello**" in out

    def test_find_without_snippet_omits_excerpt_markers(
        self, tmp_path: Path
    ) -> None:
        out = _run(
            ["load", "find hello"],
            index_path=_saved_index(tmp_path),
        )
        assert "**hello**" not in out


# -------------------------------------------------------------------- build


class TestBuildCommand:
    def test_build_pipes_pages_through_indexer_and_save(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        fake_pages = [
            Page(url="https://x.com/a", title="", text="hello world"),
            Page(url="https://x.com/b", title="", text="hello again"),
        ]
        fake_crawler = mocker.MagicMock()
        fake_crawler.crawl.return_value = fake_pages

        index_path = tmp_path / "out.json"
        out = _run(
            ["build"],
            index_path=index_path,
            crawler_factory=lambda: fake_crawler,
        )

        assert "crawled 2 pages" in out
        assert "saved" in out
        assert index_path.exists()
        fake_crawler.crawl.assert_called_once()

    def test_build_handles_crawl_error(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        fake_crawler = mocker.MagicMock()
        fake_crawler.crawl.side_effect = RuntimeError("network down")

        out = _run(
            ["build"],
            index_path=tmp_path / "out.json",
            crawler_factory=lambda: fake_crawler,
        )
        assert "crawl failed" in out
        assert "network down" in out

    def test_build_with_extra_args_prints_usage(self, tmp_path: Path) -> None:
        out = _run(["build foo"], index_path=tmp_path / "out.json")
        assert "usage: build" in out

    def test_load_with_extra_args_prints_usage(self, tmp_path: Path) -> None:
        out = _run(["load foo"], index_path=tmp_path / "out.json")
        assert "usage: load" in out

    def test_build_handles_save_error(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        from src.storage import StorageError

        fake_pages = [Page(url="https://x.com/a", title="", text="hello")]
        fake_crawler = mocker.MagicMock()
        fake_crawler.crawl.return_value = fake_pages
        mocker.patch("src.main.save", side_effect=StorageError("disk full"))

        out = _run(
            ["build"],
            index_path=tmp_path / "out.json",
            crawler_factory=lambda: fake_crawler,
        )
        assert "save failed" in out
        assert "disk full" in out


# ----------------------------------------------------------------- parsing


class TestParseErrors:
    def test_unbalanced_quote_reports_parse_error(self) -> None:
        out = _run(['find "unbalanced'])
        assert "parse error" in out


# ---------------------------------------------------------------- entry point


class TestMainEntryPoint:
    def test_main_returns_zero_on_eof(self, mocker: Any) -> None:
        from src import main as main_module

        mocker.patch.object(main_module.sys, "stdin", StringIO(""))
        mocker.patch.object(main_module.sys, "stdout", StringIO())
        assert main_module.main() == 0
