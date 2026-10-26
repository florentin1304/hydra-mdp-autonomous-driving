#!/usr/bin/env python3
"""K-means vocabulary of SDC futures in the agent-centric frame (Hydra-MDP).

Default (4 s paper setup): stride 0.5 s windows, cluster flattened xy, snap
each center to a unique real logged window. Does not overwrite an existing
8 s centroid file unless ``--out`` points at it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from womd_il.data.dataset import WOMDPackedDataset  # noqa: E402
from womd_il.data.metric_cache import vocab_sha  # noqa: E402
from womd_il.data.schema import PackConfig  # noqa: E402
from womd_il.data.subsample import class_cap_indices  # noqa: E402
from womd_il.data.windows import (  # noqa: E402
    collect_sdc_windows,
    flatten_l2_winners,
    save_window_table,
    snap_unique,
)


def kmeans(
    x: np.ndarray, k: int, n_iter: int = 40, seed: int = 0, batch: int = 256
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    k = min(k, n)
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    for _ in tqdm(range(n_iter), desc="kmeans"):
        assign = np.empty(n, dtype=np.int32)
        for i in range(0, n, batch):
            sl = x[i : i + batch]
            d = ((sl[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            assign[i : i + batch] = d.argmin(axis=1)
        for j in range(k):
            mask = assign == j
            if np.any(mask):
                centers[j] = x[mask].mean(axis=0)
    return centers


def minibatch_kmeans(x: np.ndarray, k: int, n_iter: int = 40, seed: int = 0, batch: int = 8192) -> np.ndarray:
    try:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=max(int(batch), k), n_init=3)
        km.fit(x)
        return km.cluster_centers_.astype(np.float32)
    except ImportError:
        pass
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
    n = int(xt.size(0))
    k = min(int(k), n)
    rng = np.random.default_rng(seed)
    centers = xt[torch.as_tensor(rng.choice(n, size=k, replace=False), device=device)].clone()
    for _ in tqdm(range(n_iter), desc="kmeans-mb"):
        idx = torch.as_tensor(rng.choice(n, size=min(int(batch), n), replace=False), device=device)
        sl = xt[idx]
        assign = torch.cdist(sl, centers).argmin(dim=-1)
        for j in range(k):
            m = assign == j
            if bool(m.any()):
                centers[j] = 0.85 * centers[j] + 0.15 * sl[m].mean(dim=0)
    return centers.detach().cpu().numpy().astype(np.float32)


def _collect_first_only(h5: Path, max_scenarios: int | None, horizon: int) -> np.ndarray:
    ds = WOMDPackedDataset(h5, target_mode="sdc", max_scenarios=max_scenarios, cfg=PackConfig())
    trajs = []
    for i in tqdm(range(len(ds)), desc="collect"):
        item = ds[i]
        xy = item["gt_xy"][:horizon]
        yaw = item["gt_yaw"][:horizon]
        valid = item["gt_valid"][:horizon] > 0.5
        if valid.mean() < 0.5:
            continue
        trajs.append(np.concatenate([xy, yaw[:, None]], axis=-1))
    ds.close()
    return np.stack(trajs, axis=0).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--stride", type=int, default=5, help="Origin stride in 10 Hz steps (5 = 0.5 s).")
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snap-real", action="store_true", default=True)
    p.add_argument("--no-snap-real", action="store_false", dest="snap_real")
    p.add_argument("--first-only", action="store_true", help="Old behaviour: one window per scene at current_step.")
    p.add_argument("--cap", type=int, default=12, help="Max train windows per vocab class. 0 = keep all.")
    p.add_argument("--windows-out", type=Path, default=None)
    args = p.parse_args()

    horizon = int(args.horizon)
    if args.first_only:
        arr = _collect_first_only(args.h5, args.max_scenarios, horizon)
        scene = np.arange(arr.shape[0], dtype=np.int32)
        t0 = np.full(arr.shape[0], 10, dtype=np.int32)
    else:
        print(f"Collecting {horizon}-step windows stride={args.stride} from {args.h5}")
        arr, scene, t0 = collect_sdc_windows(args.h5, horizon=horizon, stride=int(args.stride))
        if args.max_scenarios is not None:
            keep = scene < int(args.max_scenarios)
            arr, scene, t0 = arr[keep], scene[keep], t0[keep]
    n, t, c = arr.shape
    print(f"windows {n}  T={t} C={c}  unique_scenes={int(np.unique(scene).size)}")
    k_use = min(int(args.k), n)
    flat = arr[..., :2].reshape(n, t * 2)
    if k_use < n:
        print(f"Fitting k-means k={k_use} on xy {flat.shape}")
        if n > 20_000:
            centers = minibatch_kmeans(flat, k=k_use, seed=args.seed)
        else:
            centers = kmeans(flat, k=k_use, seed=args.seed)
        if args.snap_real:
            pick = snap_unique(arr, centers.reshape(k_use, t, 2))
            vocab = arr[pick].astype(np.float32)
            print(f"Snapped {k_use} unique real windows")
        else:
            heading = np.zeros((k_use, t, 1), np.float32)
            vocab = np.concatenate([centers.reshape(k_use, t, 2), heading], axis=-1)
    else:
        vocab = arr
        print(f"requested k={args.k} but only {n} windows; using all of them")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, vocab.astype(np.float32))
    print(f"Wrote vocab {vocab.shape} -> {args.out}")

    winners = flatten_l2_winners(arr, vocab)
    cap = int(args.cap)
    if cap > 0:
        keep = class_cap_indices(winners, cap, seed=int(args.seed))
        print(f"class cap={cap}: {len(keep)}/{n} windows  classes={int(np.unique(winners[keep]).size)}")
        scene, t0, winners = scene[keep], t0[keep], winners[keep]
    win_path = args.windows_out
    if win_path is None:
        win_path = args.out.with_name(args.out.stem + "_windows.npz")
    save_window_table(win_path, scene=scene, t0=t0, winner=winners, vocab_sha_hex=vocab_sha(args.out))
    print(f"Wrote windows {len(scene)} -> {win_path}")


if __name__ == "__main__":
    main()
