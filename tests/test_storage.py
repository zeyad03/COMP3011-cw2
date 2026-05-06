"""Tests for index JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.crawler import Page
from src.indexer import Index, build_index
from src.storage import (
    IndexVersionError,
    StorageError,
    _delta_decode,
    _delta_encode,
    load,
    save,
)


def _index() -> Index:
    pages = [
        Page(url="https://x.com/a", title="A", text="hello world"),
        Page(url="https://x.com/b", title="B", text="hello again"),
    ]
    return build_index(pages)


class TestRoundTrip:
    def test_save_then_load_returns_equal_index(self, tmp_path: Path) -> None:
        idx = _index()
        save(idx, tmp_path / "idx.json")
        loaded = load(tmp_path / "idx.json")
        assert loaded == idx

    def test_save_accepts_string_path(self, tmp_path: Path) -> None:
        idx = _index()
        save(idx, str(tmp_path / "idx.json"))
        loaded = load(str(tmp_path / "idx.json"))
        assert loaded == idx

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "deeply" / "nested" / "idx.json"
        save(_index(), target)
        assert target.exists()

    def test_overwrite_replaces_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "idx.json"
        save(_index(), target)
        new_idx = build_index(
            [Page(url="https://x.com/c", title="C", text="different content")]
        )
        save(new_idx, target)
        loaded = load(target)
        assert loaded["docs"]["0"]["url"] == "https://x.com/c"


class TestLoadErrors:
    def test_missing_file_raises_storage_error(self, tmp_path: Path) -> None:
        with pytest.raises(StorageError, match="not found"):
            load(tmp_path / "does_not_exist.json")

    def test_corrupt_json_raises_storage_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json {")
        with pytest.raises(StorageError, match="corrupt"):
            load(bad)

    def test_non_object_root_raises_storage_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.json"
        bad.write_text("[1, 2, 3]")
        with pytest.raises(StorageError, match="root"):
            load(bad)

    def test_wrong_version_raises_version_error(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong_version.json"
        path.write_text(
            json.dumps(
                {"meta": {"version": "999.0"}, "docs": {}, "terms": {}}
            )
        )
        with pytest.raises(IndexVersionError, match="999.0"):
            load(path)

    def test_missing_meta_raises_version_error(self, tmp_path: Path) -> None:
        path = tmp_path / "no_meta.json"
        path.write_text(json.dumps({"docs": {}, "terms": {}}))
        with pytest.raises(IndexVersionError):
            load(path)


class TestAtomicity:
    def test_successful_save_leaves_no_tmp_files(self, tmp_path: Path) -> None:
        save(_index(), tmp_path / "idx.json")
        leftover = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftover == []

    def test_failed_save_cleans_up_tmp_file(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        target = tmp_path / "idx.json"
        mocker.patch("src.storage.json.dump", side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            save(_index(), target)
        # No tmp file left behind, and the target was never created.
        leftover = list(tmp_path.iterdir())
        assert leftover == []
        assert not target.exists()

    def test_failed_save_does_not_clobber_existing_file(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        target = tmp_path / "idx.json"
        save(_index(), target)
        original_bytes = target.read_bytes()

        mocker.patch("src.storage.json.dump", side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            save(_index(), target)

        assert target.read_bytes() == original_bytes


class TestSerialisationFormat:
    def test_output_is_human_readable_json(self, tmp_path: Path) -> None:
        target = tmp_path / "idx.json"
        save(_index(), target)
        text = target.read_text()
        # indent=2 produces multi-line output.
        assert "\n" in text
        # sort_keys=True puts "docs" before "meta" before "terms".
        assert text.index('"docs"') < text.index('"meta"') < text.index('"terms"')

    def test_output_handles_unicode(self, tmp_path: Path) -> None:
        idx = build_index(
            [Page(url="https://x.com/a", title="café", text="é è")]
        )
        target = tmp_path / "idx.json"
        save(idx, target)
        loaded = load(target)
        assert "café" in loaded["docs"]["0"]["title"]
        assert "é" in loaded["terms"]


class TestDeltaCompression:
    @pytest.mark.parametrize(
        "absolute,deltas",
        [
            ([], []),
            ([5], [5]),
            ([3, 7, 12], [3, 4, 5]),
            ([0, 0, 0], [0, 0, 0]),  # consecutive zero deltas survive
            ([313, 322, 331, 341, 350], [313, 9, 9, 10, 9]),
        ],
    )
    def test_encode_then_decode_round_trips(
        self, absolute: list[int], deltas: list[int]
    ) -> None:
        assert _delta_encode(absolute) == deltas
        assert _delta_decode(deltas) == absolute

    def test_save_writes_v11_format(self, tmp_path: Path) -> None:
        target = tmp_path / "idx.json"
        save(_index(), target)
        on_disk = json.loads(target.read_text())
        assert on_disk["meta"]["version"] == "1.1"

    def test_save_does_not_mutate_caller_index(self, tmp_path: Path) -> None:
        idx = _index()
        before = json.dumps(idx, sort_keys=True)
        save(idx, tmp_path / "idx.json")
        after = json.dumps(idx, sort_keys=True)
        assert before == after  # caller's dict is untouched

    def test_load_decodes_deltas_back_to_absolute_positions(
        self, tmp_path: Path
    ) -> None:
        original = _index()
        target = tmp_path / "idx.json"
        save(original, target)
        loaded = load(target)
        # Pick a term with multiple positions and verify they are absolute.
        for entry in loaded["terms"].values():
            for posting in entry["postings"].values():
                positions = posting["positions"]
                if len(positions) > 1:
                    # Strictly increasing positions => absolute, not deltas.
                    assert all(b > a for a, b in zip(positions, positions[1:]))
                    return  # one example is enough

    def test_loading_v10_file_still_works(self, tmp_path: Path) -> None:
        # Older v1.0 files stored absolute positions. Loader must accept them.
        v10_index = {
            "meta": {
                "crawled_at": "2026-05-06T00:00:00+00:00",
                "num_docs": 1,
                "num_terms": 2,
                "total_tokens": 4,
                "version": "1.0",
            },
            "docs": {
                "0": {"url": "https://x.com/a", "title": "", "length": 4}
            },
            "terms": {
                "alpha": {
                    "df": 1,
                    "cf": 2,
                    "postings": {
                        "0": {"tf": 2, "positions": [0, 3]},  # absolute
                    },
                },
                "beta": {
                    "df": 1,
                    "cf": 2,
                    "postings": {
                        "0": {"tf": 2, "positions": [1, 2]},
                    },
                },
            },
        }
        target = tmp_path / "v10.json"
        target.write_text(json.dumps(v10_index))

        loaded = load(target)
        # Positions must be absolute (no delta transform).
        assert loaded["terms"]["alpha"]["postings"]["0"]["positions"] == [0, 3]
        assert loaded["terms"]["beta"]["postings"]["0"]["positions"] == [1, 2]

    def test_unsupported_version_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "v999.json"
        target.write_text(
            json.dumps({"meta": {"version": "999.0"}, "docs": {}, "terms": {}})
        )
        with pytest.raises(IndexVersionError):
            load(target)
