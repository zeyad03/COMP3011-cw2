"""Interactive command-line interface for the search engine.

The CLI exposes the four commands required by the brief — ``build``,
``load``, ``print``, ``find`` — plus ``help``, ``exit``, and ``quit``.
Each command runs against an in-memory :class:`~src.indexer.Index` held
between iterations of the REPL.

The class is constructed with optional ``stdin`` and ``stdout`` streams,
which allows tests to feed scripted input and capture output without
touching the user's terminal.
"""

from __future__ import annotations

import logging
import shlex
import sys
from pathlib import Path
from typing import Callable, TextIO

from src.crawler import DEFAULT_DELAY, Crawler
from src.indexer import Index, build_index
from src.ranker import RANKERS, Ranker, TFIDFRanker
from src.search import find, print_term, suggest
from src.storage import StorageError, load, save
from src.tokenizer import tokenize

DEFAULT_INDEX_PATH: Path = Path("data/index.json")
DEFAULT_START_URL: str = "https://quotes.toscrape.com/"
PROMPT: str = "> "

CommandHandler = Callable[[list[str]], bool]


class CLI:
    """REPL for the search engine.

    Args:
        index_path: File the ``build`` command writes to and the
            ``load`` command reads from.
        start_url: Seed URL for the ``build`` command.
        crawler_delay: Politeness delay (seconds) for the crawler.
        stdin: Input stream. Defaults to :data:`sys.stdin`.
        stdout: Output stream. Defaults to :data:`sys.stdout`.
        crawler_factory: Optional factory returning a configured
            :class:`Crawler`. Tests inject a fake here to avoid the
            network.
    """

    def __init__(
        self,
        *,
        index_path: Path = DEFAULT_INDEX_PATH,
        start_url: str = DEFAULT_START_URL,
        crawler_delay: float = DEFAULT_DELAY,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        crawler_factory: Callable[[], Crawler] | None = None,
    ) -> None:
        self._index_path = Path(index_path)
        self._start_url = start_url
        self._crawler_delay = crawler_delay
        self._stdin = stdin if stdin is not None else sys.stdin
        self._stdout = stdout if stdout is not None else sys.stdout
        self._crawler_factory = crawler_factory or (
            lambda: Crawler(delay=crawler_delay, verbose=True)
        )
        self._index: Index | None = None
        self._handlers: dict[str, CommandHandler] = {
            "build": self._cmd_build,
            "load": self._cmd_load,
            "print": self._cmd_print,
            "find": self._cmd_find,
            "help": self._cmd_help,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
        }

    # ---------------------------------------------------------------- run-loop

    def run(self) -> int:
        self._print("Search Engine. Type 'help' for commands, 'exit' to quit.")
        while True:
            try:
                self._stdout.write(PROMPT)
                self._stdout.flush()
                line = self._stdin.readline()
            except KeyboardInterrupt:
                self._print()
                continue
            if not line:
                self._print()
                return 0
            if not self.dispatch(line):
                return 0

    def dispatch(self, line: str) -> bool:
        """Run a single command line. Returning ``False`` terminates the REPL."""
        line = line.strip()
        if not line:
            return True
        try:
            parts = shlex.split(line)
        except ValueError as err:
            self._print(f"parse error: {err}")
            return True
        if not parts:
            return True
        cmd_name, *args = parts
        handler = self._handlers.get(cmd_name.lower())
        if handler is None:
            self._print(
                f"unknown command: {cmd_name!r}. Type 'help' for the list."
            )
            return True
        return handler(args)

    # ------------------------------------------------------------- handlers

    def _cmd_build(self, args: list[str]) -> bool:
        if args:
            self._print("usage: build")
            return True
        self._print(
            f"crawling {self._start_url} (delay={self._crawler_delay}s)..."
        )
        crawler = self._crawler_factory()
        try:
            pages = crawler.crawl(self._start_url)
        except Exception as err:
            self._print(f"crawl failed: {err}")
            return True
        self._print(f"crawled {len(pages)} pages; indexing...")
        self._index = build_index(pages)
        try:
            save(self._index, self._index_path)
        except StorageError as err:
            self._print(f"save failed: {err}")
            return True
        meta = self._index["meta"]
        self._print(
            f"indexed {meta['num_terms']} unique terms across {meta['num_docs']} docs; "
            f"saved to {self._index_path}"
        )
        return True

    def _cmd_load(self, args: list[str]) -> bool:
        if args:
            self._print("usage: load")
            return True
        try:
            self._index = load(self._index_path)
        except StorageError as err:
            self._print(f"load failed: {err}")
            return True
        meta = self._index["meta"]
        self._print(
            f"loaded {self._index_path}: "
            f"{meta['num_docs']} docs, {meta['num_terms']} terms"
        )
        return True

    def _cmd_print(self, args: list[str]) -> bool:
        if self._index is None:
            self._print("no index loaded. Run 'build' or 'load' first.")
            return True
        if len(args) != 1:
            self._print("usage: print <word>")
            return True
        self._print(print_term(self._index, args[0]))
        return True

    def _cmd_find(self, args: list[str]) -> bool:
        if self._index is None:
            self._print("no index loaded. Run 'build' or 'load' first.")
            return True
        ranker, explain, remaining = _extract_find_flags(args)
        if ranker is None:
            self._print(
                f"unknown ranker. Available: {', '.join(sorted(RANKERS))}"
            )
            return True
        if not remaining:
            self._print(
                "usage: find [--ranker <name>] [--explain] <words...>"
            )
            return True
        query = " ".join(remaining)
        results = find(self._index, query, ranker=ranker, explain=explain)
        if not results:
            self._print(f"no results for {query!r}.")
            self._maybe_suggest(query)
            return True
        for rank, result in enumerate(results, 1):
            self._print(f"{rank}. {result.url}  (score={result.score})")
            if explain:
                for c in result.breakdown:
                    self._print(
                        f"     {c.term:<14} tf={c.tf:<3} df={c.df:<4} "
                        f"contribution={c.contribution}"
                    )
        return True

    def _cmd_help(self, _args: list[str]) -> bool:
        self._print("Commands:")
        self._print("  build              crawl the website and build the index")
        self._print("  load               load the saved index from disk")
        self._print("  print <word>       print the inverted index for a word")
        self._print(
            "  find [--ranker R] [--explain] <words...>"
            "  R in {tfidf,bm25,frequency}"
        )
        self._print("  help               show this message")
        self._print("  exit | quit        leave the shell")
        return True

    def _cmd_exit(self, _args: list[str]) -> bool:
        return False

    # --------------------------------------------------------------- helpers

    def _maybe_suggest(self, query: str) -> None:
        """If any query token is unknown, print "did you mean" suggestions."""
        if self._index is None:
            return
        for token in tokenize(query):
            if token not in self._index["terms"]:
                suggestions = suggest(self._index, token)
                if suggestions:
                    self._print(
                        f"did you mean: {', '.join(suggestions)}"
                    )
                return  # Suggest for at most one unknown token.

    def _print(self, *parts: object) -> None:
        self._stdout.write(" ".join(str(p) for p in parts) + "\n")
        self._stdout.flush()


def _extract_find_flags(
    args: list[str],
) -> tuple[Ranker | None, bool, list[str]]:
    """Extract ``--ranker <name>`` and ``--explain`` flags from *args*.

    Returns ``(ranker, explain, remaining_args)``:

    * *ranker*: the matching :data:`~src.ranker.RANKERS` entry, or
      :class:`TFIDFRanker` when no flag was given. ``None`` indicates
      an unknown ranker name and the caller should report an error.
    * *explain*: ``True`` if ``--explain`` was present.
    * *remaining_args*: the positional words of the query.
    """
    ranker: Ranker = TFIDFRanker()
    explain = False
    remaining: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--ranker" and i + 1 < len(args):
            name = args[i + 1]
            if name not in RANKERS:
                return None, False, []
            ranker = RANKERS[name]
            i += 2
        elif args[i] == "--explain":
            explain = True
            i += 1
        else:
            remaining.append(args[i])
            i += 1
    return ranker, explain, remaining


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return CLI().run()


if __name__ == "__main__":
    raise SystemExit(main())
