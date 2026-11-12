"""Tiny valid PNG files for tests, built with the standard library only."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def png_bytes(value: int, size: int = 4) -> bytes:
    """A size x size grey PNG; different ``value`` gives different file content."""
    rows = b"".join(b"\x00" + bytes([value % 256]) * size for _ in range(size))
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(rows))
            + _chunk(b"IEND", b""))


def write_png(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(value))
    return path
