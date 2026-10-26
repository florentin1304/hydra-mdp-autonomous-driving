"""Read uncompressed TFRecord files without TensorFlow."""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path


def iter_tfrecord_file(path: str | Path) -> Iterator[bytes]:
    """Yield raw example bytes from an uncompressed TFRecord shard."""
    with open(path, "rb") as f:
        while True:
            header = f.read(12)
            if not header:
                return
            if len(header) < 12:
                return
            (length,) = struct.unpack("<Q", header[:8])
            data = f.read(length)
            if len(data) < length:
                return
            f.read(4)  # data CRC, ignored
            yield data


def list_tfrecord_shards(raw_dir: str | Path) -> list[Path]:
    shards = sorted(Path(raw_dir).rglob("*.tfrecord*"))
    if not shards:
        raise FileNotFoundError(f"No TFRecord shards under {raw_dir}")
    return shards


def iter_tfrecord_dir(raw_dir: str | Path, *, skip_shards: int = 0) -> Iterator[bytes]:
    shards = list_tfrecord_shards(raw_dir)[int(skip_shards) :]
    for shard in shards:
        yield from iter_tfrecord_file(shard)
