"""Variable-byte (VByte) integer compression for posting deltas.

This is the canonical IR posting-list compression scheme: each
non-negative integer is encoded as a sequence of bytes, where the high
bit of every byte is a *continuation* flag (1 = more bytes follow,
0 = last byte) and the low seven bits carry the next chunk of the
value, least-significant first. Numbers ≤ 127 fit in one byte; numbers
≤ 16,383 fit in two; etc.

VByte achieves ~30–50 % reduction on delta-encoded posting positions on
this corpus because the delta distribution is heavy-tailed but
right-truncated at the document length: most deltas are 1 to 100, and a
single byte represents them all. JSON's decimal-text encoding pays the
ASCII overhead per digit; VByte pays one bit of overhead per byte.

Format spec
-----------

A *block* of integers is encoded as:

    block := length_uvarint  data*
    data  := uvarint+

* ``uvarint`` is a single non-negative integer encoded with the VByte
  scheme described above.
* ``length_uvarint`` is the count of integers in *data*.
* All integers are non-negative.

A *file* of multiple named blocks (used by the storage sidecar) is
encoded as:

    file := magic  count_uvarint  ( key block )*
    magic := b"VBP\\x01"
    key   := length_uvarint  utf8_bytes

That file format is intentionally simple so the index can be recovered
from this spec alone — no other code is required.
"""

from __future__ import annotations

import io
from typing import Iterable

MAGIC = b"VBP\x01"


def encode_uvarint(value: int) -> bytes:
    """Encode a non-negative integer as VByte (continuation-bit) bytes."""
    if value < 0:
        raise ValueError("VByte does not encode negative integers")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def encode_block(values: Iterable[int]) -> bytes:
    """Encode an iterable of integers as ``length || values...``."""
    materialised = list(values)
    out = bytearray(encode_uvarint(len(materialised)))
    for v in materialised:
        out.extend(encode_uvarint(v))
    return bytes(out)


def decode_uvarint(buffer: io.BytesIO) -> int:
    """Decode a single VByte integer from *buffer*. Raises ``EOFError`` at end."""
    value = 0
    shift = 0
    while True:
        byte = buffer.read(1)
        if not byte:
            raise EOFError("unexpected end of VByte stream")
        b = byte[0]
        value |= (b & 0x7F) << shift
        if b & 0x80 == 0:
            return value
        shift += 7
        if shift > 63:
            raise ValueError("VByte integer overflow (>63 bits)")


def decode_block(buffer: io.BytesIO) -> list[int]:
    """Decode a single ``length || values...`` block from *buffer*."""
    n = decode_uvarint(buffer)
    return [decode_uvarint(buffer) for _ in range(n)]


def encode_named_blocks(blocks: dict[str, list[int]]) -> bytes:
    """Encode a mapping of name to integer-list as a single byte string.

    Layout: ``MAGIC || count || (key_length || key_bytes || block)*``.
    """
    out = bytearray(MAGIC)
    out.extend(encode_uvarint(len(blocks)))
    for key, values in blocks.items():
        key_bytes = key.encode("utf-8")
        out.extend(encode_uvarint(len(key_bytes)))
        out.extend(key_bytes)
        out.extend(encode_block(values))
    return bytes(out)


def decode_named_blocks(data: bytes) -> dict[str, list[int]]:
    """Inverse of :func:`encode_named_blocks`. Validates the magic header."""
    buffer = io.BytesIO(data)
    if buffer.read(4) != MAGIC:
        raise ValueError("not a VByte block file (magic mismatch)")
    count = decode_uvarint(buffer)
    out: dict[str, list[int]] = {}
    for _ in range(count):
        key_length = decode_uvarint(buffer)
        key = buffer.read(key_length).decode("utf-8")
        out[key] = decode_block(buffer)
    return out


__all__ = [
    "MAGIC",
    "decode_block",
    "decode_named_blocks",
    "decode_uvarint",
    "encode_block",
    "encode_named_blocks",
    "encode_uvarint",
]
