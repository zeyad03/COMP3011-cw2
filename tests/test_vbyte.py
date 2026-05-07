"""Tests for the VByte integer compression module (Phase F)."""

from __future__ import annotations

import io

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.vbyte import (
    MAGIC,
    decode_block,
    decode_named_blocks,
    decode_uvarint,
    encode_block,
    encode_named_blocks,
    encode_uvarint,
)


class TestUvarint:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0, b"\x00"),
            (1, b"\x01"),
            (127, b"\x7f"),
            (128, b"\x80\x01"),
            (16383, b"\xff\x7f"),
        ],
    )
    def test_known_encodings(self, value: int, expected: bytes) -> None:
        assert encode_uvarint(value) == expected
        assert decode_uvarint(io.BytesIO(expected)) == value

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            encode_uvarint(-1)

    def test_truncated_input_raises_eof(self) -> None:
        # 0x80 alone signals "more bytes follow"; followed by EOF.
        with pytest.raises(EOFError):
            decode_uvarint(io.BytesIO(b"\x80"))

    @given(st.integers(min_value=0, max_value=2**40))
    def test_round_trip(self, value: int) -> None:
        assert decode_uvarint(io.BytesIO(encode_uvarint(value))) == value


class TestBlock:
    def test_round_trip(self) -> None:
        for block in [[], [0], [1, 2, 3], [313, 9, 9, 10, 9]]:
            assert decode_block(io.BytesIO(encode_block(block))) == block

    @given(
        st.lists(st.integers(min_value=0, max_value=2**32), max_size=200)
    )
    def test_property_round_trip(self, block: list[int]) -> None:
        assert decode_block(io.BytesIO(encode_block(block))) == block


class TestNamedBlocks:
    def test_magic_present(self) -> None:
        out = encode_named_blocks({"a": [1, 2]})
        assert out.startswith(MAGIC)

    def test_round_trip(self) -> None:
        blocks = {
            "alpha": [313, 9, 9, 10, 9],
            "beta": [],
            "gamma": [0, 0, 0],
        }
        encoded = encode_named_blocks(blocks)
        assert decode_named_blocks(encoded) == blocks

    def test_bad_magic_rejected(self) -> None:
        with pytest.raises(ValueError):
            decode_named_blocks(b"XXXX" + b"\x00")

    def test_compression_beats_decimal_json(self) -> None:
        # On a realistic delta distribution VByte must beat the decimal
        # text length JSON would emit.
        deltas = [313] + [9, 12, 7, 11, 8, 10, 14] * 200
        json_size = sum(len(str(d)) + 1 for d in deltas)  # +1 comma
        vb_size = len(encode_block(deltas))
        assert vb_size < json_size
