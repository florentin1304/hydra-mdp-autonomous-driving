"""Cap train scenarios per Hydra vocabulary class.

With ``main=pdms`` the class is the vocab trajectory with the highest
cached PDMS. With ``main=imitation`` it is the ADE winner against the
logged SDC future.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from womd_il import constants as C
from womd_il.data.metric_cache import vocab_sha
from womd_il.losses.hydra import pdms_from_probs


def class_cap_indices(winner: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    """Scenario indices with at most ``cap`` rows kept for each class.

    ``winner[i] < 0`` marks a scenario with no usable future; those rows are
    left out. The result is sorted so a fixed seed yields a stable subset.
    """
    cap = int(cap)
    if cap < 1:
        raise ValueError(f"subsample cap must be >= 1, got {cap}")
    labels = np.asarray(winner)
    rng = np.random.default_rng(int(seed))
    parts: list[np.ndarray] = []
    for k in np.unique(labels):
        if int(k) < 0:
            continue
        ids = np.flatnonzero(labels == k)
        if ids.size > cap:
            ids = rng.choice(ids, size=cap, replace=False)
        parts.append(np.asarray(ids, dtype=np.int64))
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.sort(np.concatenate(parts))


def load_sdc_futures(
    path: str | Path,
    *,
    chunk: int = 256,
    current_step: int = C.CURRENT_STEP,
    future_steps: int = C.FUTURE_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Agent-centric SDC future ``[N, T, 2]`` and validity ``[N, T]``.

    Slot 0 is the SDC, matching ``target_mode: sdc``.
    """
    with h5py.File(path, "r") as f:
        n, _, t_all, _ = f["xyz"].shape
        t0 = min(int(current_step), t_all - 1)
        fut = slice(t0 + 1, min(t0 + 1 + int(future_steps), t_all))
        n_fut = fut.stop - fut.start
        gt = np.zeros((n, int(future_steps), 2), np.float32)
        valid = np.zeros((n, int(future_steps)), np.float32)
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            xyz = np.asarray(f["xyz"][i0:i1, 0], dtype=np.float32)
            yaw = np.asarray(f["yaw"][i0:i1, 0], dtype=np.float32)
            v = np.asarray(f["valid"][i0:i1, 0])
            origin_xy = xyz[:, t0, :2].copy()
            origin_yaw = yaw[:, t0].copy()
            bad = ~v[:, t0]
            origin_xy[bad] = 0.0
            origin_yaw[bad] = 0.0
            fut_xy = xyz[:, fut, :2]
            c = np.cos(origin_yaw)
            s = np.sin(origin_yaw)
            dx = fut_xy[..., 0] - origin_xy[:, None, 0]
            dy = fut_xy[..., 1] - origin_xy[:, None, 1]
            gt[i0:i1, :n_fut, 0] = dx * c[:, None] + dy * s[:, None]
            gt[i0:i1, :n_fut, 1] = -dx * s[:, None] + dy * c[:, None]
            valid[i0:i1, :n_fut] = v[:, fut].astype(np.float32)
    return gt, valid


def assign_winners(
    gt: np.ndarray,
    valid: np.ndarray,
    vocab_xy: np.ndarray,
    *,
    temperature: float = 1.0,
    front_weight: float = 1.0,
    min_valid_frac: float = 0.5,
    batch: int = 48,
    k_chunk: int = 512,
) -> np.ndarray:
    """Hard class id per scenario. ``-1`` when the future is mostly invalid.

    ADE matches ``hydra_imitation_loss`` at ``collision_weight=0``.
    ``temperature`` is unused for the hard winner; it is accepted so callers
    can pass the decoder config through unchanged.
    """
    del temperature
    keep = valid.mean(axis=1) >= float(min_valid_frac)
    index = np.flatnonzero(keep)
    vocab_t = torch.from_numpy(np.ascontiguousarray(vocab_xy[..., :2], dtype=np.float32))
    k, t, _ = vocab_t.shape
    fw = max(float(front_weight), 1.0)
    tw = torch.ones(t) if fw == 1.0 else torch.linspace(fw, 1.0, t)
    winners = np.full(len(gt), -1, np.int32)
    for i0 in range(0, len(index), batch):
        sl = index[i0 : i0 + batch]
        g = torch.from_numpy(np.ascontiguousarray(gt[sl]))
        v = torch.from_numpy(np.ascontiguousarray(valid[sl]))
        vw = v * tw
        vsum = vw.sum(-1).clamp_min(1.0).unsqueeze(-1)
        parts = []
        for k0 in range(0, k, k_chunk):
            vk = vocab_t[k0 : k0 + k_chunk]
            dist = torch.linalg.vector_norm(g[:, None] - vk[None], dim=-1)
            parts.append((dist * vw[:, None]).sum(-1) / vsum)
        winners[sl] = torch.cat(parts, dim=1).argmin(dim=-1).numpy().astype(np.int32)
        done = i0 + len(sl)
        if done == len(index) or done % (batch * 40) == 0:
            print(f"  class assign {done}/{len(index)}", flush=True)
    return winners


def assign_pdms_winners(labels: np.ndarray, *, chunk: int = 512) -> np.ndarray:
    """Hard class id = vocab index of max PDMS. ``-1`` if every mode scores 0."""
    arr = np.asarray(labels)
    if arr.ndim != 3 or arr.shape[-1] < 5:
        raise ValueError(f"metric labels must be [N,K,5+], got {tuple(arr.shape)}")
    n = int(arr.shape[0])
    winners = np.full(n, -1, np.int32)
    for i0 in range(0, n, int(chunk)):
        i1 = min(i0 + int(chunk), n)
        y = torch.from_numpy(np.ascontiguousarray(arr[i0:i1], dtype=np.float32))
        pdms = pdms_from_probs(y[..., :5])
        best, idx = pdms.max(dim=-1)
        w = idx.to(torch.int32)
        w[best <= 0] = -1
        winners[i0:i1] = w.numpy()
    return winners


def load_winners(
    train_h5: str | Path,
    vocab_path: str | Path,
    cache_path: str | Path,
    *,
    front_weight: float = 1.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Load cached hard winners, recomputing when the vocab hash differs."""
    train_h5 = Path(train_h5)
    vocab_path = Path(vocab_path)
    cache_path = Path(cache_path)
    sha = vocab_sha(vocab_path)
    if cache_path.exists():
        with np.load(cache_path) as z:
            cached_sha = str(z["vocab_sha"])
            winner = np.asarray(z["winner"])
            kind = str(z["kind"]) if "kind" in z.files else "ade"
        with h5py.File(train_h5, "r") as f:
            n = int(f["xyz"].shape[0])
        if cached_sha == sha and winner.shape == (n,) and kind == "ade":
            return winner.astype(np.int32, copy=False)
        print(f"Hydra winner cache stale ({cache_path}); recomputing")
    print(f"Assigning Hydra ADE classes on {train_h5.name} -> {cache_path}")
    gt, valid = load_sdc_futures(train_h5)
    vocab = np.load(vocab_path)
    winner = assign_winners(
        gt,
        valid,
        vocab,
        temperature=temperature,
        front_weight=front_weight,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        winner=winner.astype(np.int32),
        vocab_sha=np.array(sha),
        kind=np.array("ade"),
    )
    print(f"Wrote {cache_path} ({winner.shape[0]} scenarios)")
    return winner


def load_pdms_winners(
    metric_h5: str | Path,
    vocab_path: str | Path,
    cache_path: str | Path,
) -> np.ndarray:
    """Load cached PDMS argmax classes, recomputing when the vocab hash differs."""
    metric_h5 = Path(metric_h5)
    vocab_path = Path(vocab_path)
    cache_path = Path(cache_path)
    sha = vocab_sha(vocab_path)
    if not metric_h5.exists():
        raise FileNotFoundError(
            f"PDMS class subsample needs the metric cache at {metric_h5}"
        )
    with h5py.File(metric_h5, "r") as f:
        n = int(f["labels"].shape[0])
        cache_sha = str(f.attrs.get("vocab_sha", ""))
    if cache_sha and cache_sha != sha:
        raise ValueError(
            f"metric cache vocab hash {cache_sha} != {sha}; rebuild the cache"
        )
    if cache_path.exists():
        with np.load(cache_path) as z:
            cached_sha = str(z["vocab_sha"])
            winner = np.asarray(z["winner"])
            kind = str(z["kind"]) if "kind" in z.files else ""
        if cached_sha == sha and winner.shape == (n,) and kind == "pdms":
            return winner.astype(np.int32, copy=False)
        print(f"Hydra PDMS winner cache stale ({cache_path}); recomputing")
    print(f"Assigning Hydra PDMS classes from {metric_h5.name} -> {cache_path}")
    with h5py.File(metric_h5, "r") as f:
        winner = assign_pdms_winners(f["labels"])
    n_ok = int((winner >= 0).sum())
    n_cls = int(np.unique(winner[winner >= 0]).size) if n_ok else 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        winner=winner.astype(np.int32),
        vocab_sha=np.array(sha),
        kind=np.array("pdms"),
    )
    print(
        f"Wrote {cache_path} ({n_ok}/{winner.shape[0]} scenarios with PDMS>0, "
        f"{n_cls} classes)"
    )
    return winner
