"""Offline Hydra metric labels [N, K, M] aligned with a packed HDF5."""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np

CACHE_VERSION = 5  # log-future arc EP + fail-closed DAC (edge × lane corridor)
# v4 = pre-route labels; allowed for training until v1.3.1 packs + v5 recache land.
_ACCEPT_CACHE_VERSIONS = frozenset({4, CACHE_VERSION})
LABEL_DTYPE = np.uint8
# uint8 stores round(score * 255) so DDC 0.5 and continuous EP survive.


def vocab_sha(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(data).hexdigest()[:16]


def resolve_metric_cache(root: str | Path | None, packed_path: str | Path) -> Path | None:
    """Directory ``root/{stem}.h5`` or a single file. None if missing."""
    if root is None:
        return None
    p = Path(root)
    packed = Path(packed_path)
    if p.is_file():
        return p if p.exists() else None
    candidate = p / f"{packed.stem}.h5"
    return candidate if candidate.exists() else None


def create_metric_cache(
    path: str | Path,
    n: int,
    k: int,
    *,
    vocab_sha_hex: str,
    packed: str | Path,
    n_metrics: int | None = None,
    metric_names: tuple[str, ...] | None = None,
) -> h5py.File:
    if n_metrics is None:
        from womd_il.losses.hydra import METRIC_NAMES

        metric_names = metric_names or METRIC_NAMES
        n_metrics = len(metric_names)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    chunks = (min(32, n), k, n_metrics)
    f.create_dataset("labels", (n, k, n_metrics), dtype=LABEL_DTYPE, chunks=chunks, compression="lzf")
    f.attrs["version"] = int(CACHE_VERSION)
    f.attrs["vocab_sha"] = str(vocab_sha_hex)
    f.attrs["packed"] = str(packed)
    f.attrs["n"] = int(n)
    f.attrs["k"] = int(k)
    f.attrs["encoding"] = "uint8_255"
    if metric_names is not None:
        f.attrs["metrics"] = ",".join(metric_names)
    return f


def validate_metric_cache(
    handle: h5py.File,
    *,
    vocab_sha_hex: str | None = None,
    k: int | None = None,
    n: int | None = None,
) -> None:
    ver = int(handle.attrs.get("version", -1))
    if ver not in _ACCEPT_CACHE_VERSIONS:
        raise ValueError(
            f"metric cache version {ver} not in {_ACCEPT_CACHE_VERSIONS}; "
            "rebuild with scripts/cache_hydra_metrics.py"
        )
    if vocab_sha_hex is not None and str(handle.attrs.get("vocab_sha", "")) != str(vocab_sha_hex):
        raise ValueError(
            "metric cache vocab hash does not match hydra_vocab.npy; rebuild the cache"
        )
    shape = handle["labels"].shape
    if k is not None and int(shape[1]) != int(k):
        raise ValueError(f"metric cache K={shape[1]} != vocab K={k}")
    if n is not None and int(shape[0]) < int(n):
        raise ValueError(f"metric cache N={shape[0]} < dataset N={n}; rebuild the cache")
