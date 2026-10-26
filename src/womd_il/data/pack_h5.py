"""Write padded scenario samples into a single HDF5 pack."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from womd_il.data.schema import DATASETS, PackConfig

_DTYPES = {
    "xyz": np.float32,
    "yaw": np.float32,
    "vel": np.float32,
    "valid": bool,
    "length": np.float32,
    "width": np.float32,
    "height": np.float32,
    "type": np.int8,
    "is_sdc": bool,
    "tracks_to_predict": bool,
    "objects_of_interest": bool,
    "map_xyz": np.float32,
    "map_valid": bool,
    "map_type": np.int8,
    "map_on_route": bool,
    "tl_xyz": np.float32,
    "tl_state": np.int8,
    "tl_valid": bool,
    "route_xyz": np.float32,
    "route_valid": bool,
    "route_on_route": bool,
    "sdc_index": np.int32,
}


def _id_bytes(sid: str) -> bytes:
    return str(sid).encode("utf-8")[:64].ljust(64, b"\0")


def _rest_shapes(cfg: PackConfig) -> dict[str, tuple[int, ...]]:
    a, t = cfg.num_agents, cfg.total_steps
    p, m = cfg.num_map_polylines, cfg.polyline_points
    l = cfg.num_traffic_lights
    r, q = cfg.num_route_paths, cfg.route_points
    return {
        "xyz": (a, t, 3),
        "yaw": (a, t),
        "vel": (a, t, 2),
        "valid": (a, t),
        "length": (a,),
        "width": (a,),
        "height": (a,),
        "type": (a,),
        "is_sdc": (a,),
        "tracks_to_predict": (a,),
        "objects_of_interest": (a,),
        "map_xyz": (p, m, 3),
        "map_valid": (p, m),
        "map_type": (p,),
        "map_on_route": (p,),
        "tl_xyz": (l, t, 3),
        "tl_state": (l, t),
        "tl_valid": (l, t),
        "route_xyz": (r, q, 3),
        "route_valid": (r, q),
        "route_on_route": (r,),
        "sdc_index": (),
    }


def create_h5(path: str | Path, cfg: PackConfig) -> h5py.File:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.attrs["history_steps"] = cfg.history_steps
    f.attrs["future_steps"] = cfg.future_steps
    f.attrs["current_step"] = cfg.current_step
    f.attrs["dt"] = cfg.dt
    f.attrs["n"] = 0
    rest = _rest_shapes(cfg)
    f.create_dataset("scenario_id", shape=(0,), maxshape=(None,), dtype="S64", chunks=(64,))
    for name in DATASETS:
        r = rest[name]
        chunks = (1,) + r if r else (64,)
        f.create_dataset(
            name,
            shape=(0,) + r,
            maxshape=(None,) + r,
            dtype=_DTYPES[name],
            chunks=chunks,
        )
    return f


def open_h5(path: str | Path) -> h5py.File:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return h5py.File(path, "a")


def append_sample(f: h5py.File, sample: dict) -> int:
    n = int(f.attrs["n"])
    f["scenario_id"].resize((n + 1,))
    f["scenario_id"][n] = _id_bytes(sample.get("scenario_id", f"{n}"))
    for name in DATASETS:
        f[name].resize((n + 1,) + f[name].shape[1:])
        f[name][n] = sample[name]
    f.attrs["n"] = n + 1
    return n
