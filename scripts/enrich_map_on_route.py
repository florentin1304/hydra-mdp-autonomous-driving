#!/usr/bin/env python3
"""Add / refresh packed ``map_on_route`` corridor flags on an H5 pack."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from womd_il import constants as C
from womd_il.data.features import _map_on_route_flag
from womd_il.data.schema import PackConfig
from womd_il.data.transforms import agent_centric_xy


def _flags_for_row(
    map_xyz: np.ndarray,
    map_valid: np.ndarray,
    map_type: np.ndarray,
    route_xyz: np.ndarray,
    route_valid: np.ndarray,
    route_on_route: np.ndarray,
    xyz: np.ndarray,
    yaw: np.ndarray,
    valid: np.ndarray,
    current: int,
) -> np.ndarray:
    p = int(map_type.shape[0])
    out = np.zeros((p,), bool)
    rv = np.asarray(route_valid, dtype=bool)
    counts = rv.sum(axis=-1)
    on = np.asarray(route_on_route, dtype=bool)
    usable = on & (counts >= 2)
    if not usable.any() and int(counts.max()) < 2:
        return out
    ridx = int(np.argmax(usable)) if usable.any() else int(np.argmax(counts))
    if not bool(valid[0, current]):
        return out
    origin = xyz[0, current, :2].astype(np.float32)
    yaw0 = float(yaw[0, current])
    map_xy = agent_centric_xy(map_xyz[..., :2], origin, yaw0)
    route_xy = agent_centric_xy(route_xyz[ridx, :, :2], origin, yaw0)
    return _map_on_route_flag(
        map_xy, map_valid, map_type, route_xy, rv[ridx]
    )[:, 0].astype(bool)


def enrich(path: Path, *, overwrite: bool = True) -> None:
    cfg = PackConfig()
    current = int(cfg.current_step)
    with h5py.File(path, "a") as f:
        n = int(f.attrs["n"])
        p = int(f["map_type"].shape[1])
        if "map_on_route" in f and not overwrite:
            print(f"{path}: map_on_route exists, skip")
            return
        if "map_on_route" in f:
            del f["map_on_route"]
        ds = f.create_dataset(
            "map_on_route",
            shape=(n, p),
            dtype=bool,
            chunks=(1, p),
        )
        frac = 0
        for i in tqdm(range(n), desc=f"map_on_route {path.name}", leave=True):
            flags = _flags_for_row(
                f["map_xyz"][i],
                f["map_valid"][i],
                f["map_type"][i],
                f["route_xyz"][i],
                f["route_valid"][i],
                f["route_on_route"][i],
                f["xyz"][i],
                f["yaw"][i],
                f["valid"][i],
                current,
            )
            ds[i] = flags
            frac += float(flags.any())
        f.attrs["map_on_route_version"] = 1
        print(
            f"{path}: wrote map_on_route N={n} P={p} "
            f"frac_any={frac / max(n, 1):.3f} thresh={C.ROUTE_LANE_THRESH_M}m"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("h5", nargs="+", type=Path)
    ap.add_argument("--no-overwrite", action="store_true")
    args = ap.parse_args()
    for p in args.h5:
        enrich(p, overwrite=not args.no_overwrite)


if __name__ == "__main__":
    main()
