"""Synthetic packed scenarios for tests and closed-loop smoke runs."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from womd_il import constants as C
from womd_il.data.schema import PackConfig


def make_synthetic_pack(
    n: int = 4,
    cfg: PackConfig | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    cfg = cfg or PackConfig()
    rng = np.random.default_rng(seed)
    a, t = cfg.num_agents, cfg.total_steps
    p, m = cfg.num_map_polylines, cfg.polyline_points
    l = cfg.num_traffic_lights
    r, q = cfg.num_route_paths, cfg.route_points

    xyz = np.zeros((n, a, t, 3), np.float32)
    yaw = np.zeros((n, a, t), np.float32)
    vel = np.zeros((n, a, t, 2), np.float32)
    valid = np.zeros((n, a, t), bool)
    length = np.full((n, a), 4.5, np.float32)
    width = np.full((n, a), 2.0, np.float32)
    height = np.full((n, a), 1.5, np.float32)
    types = np.ones((n, a), np.int8)
    is_sdc = np.zeros((n, a), bool)
    is_sdc[:, 0] = True
    ttp = np.zeros((n, a), bool)
    ttp[:, 0] = True
    ooi = np.zeros((n, a), bool)

    for i in range(n):
        speed = 8.0 + float(rng.normal(0, 1.0))
        heading = float(rng.uniform(-0.2, 0.2))
        x0, y0 = float(rng.uniform(-5, 5)), float(rng.uniform(-2, 2))
        for step in range(t):
            dist = speed * cfg.dt * step
            xyz[i, 0, step, 0] = x0 + dist * np.cos(heading)
            xyz[i, 0, step, 1] = y0 + dist * np.sin(heading)
            yaw[i, 0, step] = heading
            vel[i, 0, step] = (speed * np.cos(heading), speed * np.sin(heading))
            valid[i, 0, step] = True
        # Neighbor in the left lane, slightly slower.
        for slot in range(1, min(8, a)):
            lat = 3.7 * ((slot % 3) - 1)
            nspeed = speed * (0.7 + 0.1 * slot)
            nhead = heading
            nx0 = x0 - 8.0 * slot
            for step in range(t):
                dist = nspeed * cfg.dt * step
                xyz[i, slot, step, 0] = nx0 + dist * np.cos(nhead)
                xyz[i, slot, step, 1] = y0 + lat + dist * np.sin(nhead)
                yaw[i, slot, step] = nhead
                vel[i, slot, step] = (nspeed * np.cos(nhead), nspeed * np.sin(nhead))
                valid[i, slot, step] = True
            types[i, slot] = C.TYPE_VEHICLE

    map_xyz = np.zeros((n, p, m, 3), np.float32)
    map_valid = np.zeros((n, p, m), bool)
    map_type = np.zeros((n, p), np.int8)
    map_on_route = np.zeros((n, p), bool)
    for i in range(n):
        for pi in range(min(p, 32)):
            xs = np.linspace(-50, 80, m)
            map_xyz[i, pi, :, 0] = xs
            map_xyz[i, pi, :, 1] = (pi - 16) * 0.5
            map_valid[i, pi] = True
            map_type[i, pi] = C.MAP_LANE_SURFACE if pi % 4 != 0 else C.MAP_ROAD_EDGE
            # Corridor around y=0 (~6 m), same as ROUTE_LANE_THRESH_M.
            if map_type[i, pi] == C.MAP_LANE_SURFACE and abs((pi - 16) * 0.5) < 5.0:
                map_on_route[i, pi] = True

    tl_xyz = np.zeros((n, l, t, 3), np.float32)
    tl_state = np.zeros((n, l, t), np.int8)
    tl_valid = np.zeros((n, l, t), bool)
    tl_xyz[:, 0, :, 0] = 40.0
    tl_state[:, 0] = C.TL_GO
    tl_valid[:, 0] = True

    route_xyz = np.zeros((n, r, q, 3), np.float32)
    route_valid = np.zeros((n, r, q), bool)
    route_on_route = np.zeros((n, r), bool)
    xs = np.linspace(0, 80, q)
    route_xyz[:, 0, :, 0] = xs[None, :]
    route_valid[:, 0] = True
    route_on_route[:, 0] = True

    ids = np.array([f"synth_{i:04d}" for i in range(n)], dtype="S16")
    return {
        "scenario_id": ids,
        "xyz": xyz,
        "yaw": yaw,
        "vel": vel,
        "valid": valid,
        "length": length,
        "width": width,
        "height": height,
        "type": types,
        "is_sdc": is_sdc,
        "tracks_to_predict": ttp,
        "objects_of_interest": ooi,
        "map_xyz": map_xyz,
        "map_valid": map_valid,
        "map_type": map_type,
        "map_on_route": map_on_route,
        "tl_xyz": tl_xyz,
        "tl_state": tl_state,
        "tl_valid": tl_valid,
        "route_xyz": route_xyz,
        "route_valid": route_valid,
        "route_on_route": route_on_route,
        "sdc_index": np.zeros((n,), np.int32),
    }


def write_synthetic_h5(path: str | Path, n: int = 4, seed: int = 0, cfg: PackConfig | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pack = make_synthetic_pack(n=n, cfg=cfg, seed=seed)
    cfg = cfg or PackConfig()
    with h5py.File(path, "w") as f:
        f.attrs["history_steps"] = cfg.history_steps
        f.attrs["future_steps"] = cfg.future_steps
        f.attrs["current_step"] = cfg.current_step
        f.attrs["dt"] = cfg.dt
        for k, v in pack.items():
            f.create_dataset(k, data=v, compression="gzip")
    return path
