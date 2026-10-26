"""Sliding 4 s SDC windows on packed WOMD tracks."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from womd_il.data.transforms import agent_centric_xy, agent_centric_yaw


def collect_sdc_windows(
    h5: str | Path,
    *,
    horizon: int = 40,
    stride: int = 5,
    min_valid_frac: float = 0.5,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ego-frame SDC futures from every valid origin on the packed track.

    Returns
    -------
    xyh : [N, horizon, 3]  (x, y, heading) after t0, in the SDC frame at t0
    scene : [N] int32 packed scenario index
    t0 : [N] int32 origin timestep
    """
    h5 = Path(h5)
    horizon = int(horizon)
    stride = max(int(stride), 1)
    xy_parts: list[np.ndarray] = []
    yaw_parts: list[np.ndarray] = []
    scene_parts: list[np.ndarray] = []
    t0_parts: list[np.ndarray] = []
    with h5py.File(h5, "r") as f:
        n, _, t_all, _ = f["xyz"].shape
        last = t_all - horizon - 1
        if last < 0:
            raise ValueError(f"{h5} has T={t_all}, need > {horizon}")
        origins = np.arange(0, last + 1, stride, dtype=np.int32)
        try:
            from tqdm import tqdm

            starts = tqdm(range(0, n, chunk), desc="windows")
        except ImportError:
            starts = range(0, n, chunk)
        for i0 in starts:
            i1 = min(i0 + chunk, n)
            xyz = np.asarray(f["xyz"][i0:i1, 0], dtype=np.float32)
            yaw = np.asarray(f["yaw"][i0:i1, 0], dtype=np.float32)
            valid = np.asarray(f["valid"][i0:i1, 0], dtype=bool)
            for t0 in origins:
                fut = slice(t0 + 1, t0 + 1 + horizon)
                ok = valid[:, t0] & (valid[:, fut].mean(axis=1) >= float(min_valid_frac))
                if not np.any(ok):
                    continue
                rows = np.flatnonzero(ok)
                origin_xy = xyz[rows, t0, :2][:, None, :]
                origin_yaw = yaw[rows, t0][:, None]
                fut_xy = agent_centric_xy(xyz[rows, fut, :2], origin_xy, origin_yaw)
                fut_yaw = agent_centric_yaw(yaw[rows, fut], origin_yaw)
                xy_parts.append(fut_xy.astype(np.float32))
                yaw_parts.append(fut_yaw.astype(np.float32))
                scene_parts.append((rows + i0).astype(np.int32))
                t0_parts.append(np.full(rows.size, t0, dtype=np.int32))
            del xyz, yaw, valid
    if not xy_parts:
        raise RuntimeError(f"no valid {horizon}-step windows in {h5}")
    xy = np.concatenate(xy_parts, axis=0)
    heading = np.concatenate(yaw_parts, axis=0)
    xyh = np.concatenate([xy, heading[..., None]], axis=-1).astype(np.float32)
    scene = np.concatenate(scene_parts, axis=0)
    t0 = np.concatenate(t0_parts, axis=0)
    return xyh, scene, t0


def snap_unique(xyh: np.ndarray, centers_xy: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Greedy nearest unused real window per center (flatten-xy L2)."""
    import torch

    n = int(xyh.shape[0])
    k = int(centers_xy.shape[0])
    t = int(xyh.shape[1])
    if centers_xy.ndim == 3:
        t = min(t, int(centers_xy.shape[1]))
        ctr = np.ascontiguousarray(centers_xy[:, :t, :2].reshape(k, -1), dtype=np.float32)
    else:
        t = min(t, int(centers_xy.shape[-1] // 2))
        ctr = np.ascontiguousarray(centers_xy.reshape(k, -1)[:, : t * 2], dtype=np.float32)
    flat = np.ascontiguousarray(xyh[:, :t, :2].reshape(n, -1), dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ct = torch.from_numpy(ctr).to(device)
    used = np.zeros(n, dtype=bool)
    picked = np.empty(k, dtype=np.int64)
    try:
        from tqdm import tqdm

        js = tqdm(range(k), desc="snap")
    except ImportError:
        js = range(k)
    for j in js:
        best_d = np.inf
        best_i = 0
        cj = ct[j]
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            sl = torch.from_numpy(flat[i0:i1]).to(device)
            d = torch.linalg.vector_norm(sl - cj, dim=-1).detach().cpu().numpy()
            local = np.arange(i0, i1)
            d[used[local]] = np.inf
            i = int(d.argmin())
            if d[i] < best_d:
                best_d = float(d[i])
                best_i = int(local[i])
        picked[j] = best_i
        used[best_i] = True
    return picked


def flatten_l2_winners(xyh: np.ndarray, vocab: np.ndarray, *, batch: int = 2048) -> np.ndarray:
    """Nearest vocab mode under flattened xy L2. ``xyh``/``vocab`` [N|K, T, 2|3]."""
    import torch

    t = min(int(xyh.shape[1]), int(vocab.shape[1]))
    vt = torch.from_numpy(
        np.ascontiguousarray(vocab[:, :t, :2].reshape(vocab.shape[0], -1), dtype=np.float32)
    )
    out = np.empty(xyh.shape[0], dtype=np.int32)
    try:
        from tqdm import tqdm

        starts = tqdm(range(0, xyh.shape[0], batch), desc="winners")
    except ImportError:
        starts = range(0, xyh.shape[0], batch)
    for i0 in starts:
        i1 = min(i0 + batch, xyh.shape[0])
        g = torch.from_numpy(np.ascontiguousarray(xyh[i0:i1, :t, :2].reshape(i1 - i0, -1), dtype=np.float32))
        out[i0:i1] = torch.cdist(g, vt, p=2).argmin(dim=-1).numpy().astype(np.int32)
    return out


def save_window_table(
    path: str | Path,
    *,
    scene: np.ndarray,
    t0: np.ndarray,
    winner: np.ndarray | None = None,
    vocab_sha_hex: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene": np.asarray(scene, dtype=np.int32),
        "t0": np.asarray(t0, dtype=np.int32),
    }
    if winner is not None:
        payload["winner"] = np.asarray(winner, dtype=np.int32)
    if vocab_sha_hex is not None:
        payload["vocab_sha"] = np.array(vocab_sha_hex)
    np.savez(path, **payload)


def load_window_table(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``windows [N,2] = (scene, t0)`` and optional flatten-L2 winners."""
    with np.load(path) as z:
        windows = np.stack([np.asarray(z["scene"], dtype=np.int32), np.asarray(z["t0"], dtype=np.int32)], axis=1)
        winner = np.asarray(z["winner"], dtype=np.int32) if "winner" in z.files else None
    return windows, winner
