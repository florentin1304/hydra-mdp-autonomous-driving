"""Parse a WOMD Scenario proto (or a numpy dict) into a padded sample."""

from __future__ import annotations

from typing import Any

import numpy as np

from womd_il import constants as C
from womd_il.data.schema import PackConfig


def _empty_sample(cfg: PackConfig) -> dict[str, np.ndarray | str]:
    a, t = cfg.num_agents, cfg.total_steps
    p, m = cfg.num_map_polylines, cfg.polyline_points
    l = cfg.num_traffic_lights
    r, q = cfg.num_route_paths, cfg.route_points
    return {
        "scenario_id": "",
        "xyz": np.zeros((a, t, 3), np.float32),
        "yaw": np.zeros((a, t), np.float32),
        "vel": np.zeros((a, t, 2), np.float32),
        "valid": np.zeros((a, t), bool),
        "length": np.zeros((a,), np.float32),
        "width": np.zeros((a,), np.float32),
        "height": np.zeros((a,), np.float32),
        "type": np.zeros((a,), np.int8),
        "is_sdc": np.zeros((a,), bool),
        "tracks_to_predict": np.zeros((a,), bool),
        "objects_of_interest": np.zeros((a,), bool),
        "map_xyz": np.zeros((p, m, 3), np.float32),
        "map_valid": np.zeros((p, m), bool),
        "map_type": np.zeros((p,), np.int8),
        "map_on_route": np.zeros((p,), bool),
        "tl_xyz": np.zeros((l, t, 3), np.float32),
        "tl_state": np.zeros((l, t), np.int8),
        "tl_valid": np.zeros((l, t), bool),
        "route_xyz": np.zeros((r, q, 3), np.float32),
        "route_valid": np.zeros((r, q), bool),
        "route_on_route": np.zeros((r,), bool),
        "sdc_index": np.int32(0),
    }


def _map_type_from_feature(mf: Any) -> int:
    which = mf.WhichOneof("feature_data") if hasattr(mf, "WhichOneof") else None
    mapping = {
        "lane": C.MAP_LANE_SURFACE,
        "road_line": C.MAP_ROAD_LINE,
        "road_edge": C.MAP_ROAD_EDGE,
        "stop_sign": C.MAP_STOP_SIGN,
        "crosswalk": C.MAP_CROSSWALK,
        "speed_bump": C.MAP_SPEED_BUMP,
        "driveway": C.MAP_DRIVEWAY,
    }
    if which in mapping:
        if which == "lane":
            lane_type = getattr(mf.lane, "type", 0)
            # WOMD Lane.Type: TYPE_UNDEFINED=0, FREEWAY=1, SURFACE_STREET=2, BIKE_LANE=3
            if lane_type == 1:
                return C.MAP_LANE_FREEWAY
            if lane_type == 3:
                return C.MAP_LANE_BIKE
            return C.MAP_LANE_SURFACE
        return mapping[which]
    return C.MAP_UNKNOWN


def _polyline_from_map_feature(mf: Any) -> np.ndarray | None:
    which = mf.WhichOneof("feature_data") if hasattr(mf, "WhichOneof") else None
    pts = None
    if which == "lane":
        pts = mf.lane.polyline
    elif which == "road_line":
        pts = mf.road_line.polyline
    elif which == "road_edge":
        pts = mf.road_edge.polyline
    elif which == "stop_sign":
        p = mf.stop_sign.position
        return np.array([[p.x, p.y, getattr(p, "z", 0.0)]], dtype=np.float32)
    elif which in ("crosswalk", "speed_bump", "driveway"):
        field = getattr(mf, which)
        pts = getattr(field, "polygon", None)
    if pts is None:
        return None
    arr = np.array([[p.x, p.y, getattr(p, "z", 0.0)] for p in pts], dtype=np.float32)
    return arr if arr.size else None


def _resample_polyline(pts: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample to `n` points; return (xyz[n,3], valid[n])."""
    out = np.zeros((n, 3), np.float32)
    valid = np.zeros((n,), bool)
    if pts is None or len(pts) == 0:
        return out, valid
    if len(pts) == 1:
        out[0] = pts[0]
        valid[0] = True
        return out, valid
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arclen[-1])
    if total < 1e-6:
        k = min(len(pts), n)
        out[:k] = pts[:k]
        valid[:k] = True
        return out, valid
    samples = np.linspace(0.0, total, n)
    for i, s in enumerate(samples):
        j = int(np.searchsorted(arclen, s, side="right") - 1)
        j = min(max(j, 0), len(pts) - 2)
        t = 0.0 if arclen[j + 1] == arclen[j] else (s - arclen[j]) / (arclen[j + 1] - arclen[j])
        out[i] = (1 - t) * pts[j] + t * pts[j + 1]
        valid[i] = True
    return out, valid


def parse_scenario_proto(scenario: Any, cfg: PackConfig | None = None) -> dict[str, Any]:
    """Convert a `scenario_pb2.Scenario` into a padded numpy sample."""
    cfg = cfg or PackConfig()
    sample = _empty_sample(cfg)
    sample["scenario_id"] = str(getattr(scenario, "scenario_id", "") or "")

    tracks = list(scenario.tracks)
    sdc_idx = int(getattr(scenario, "sdc_track_index", 0))
    sdc_idx = min(max(sdc_idx, 0), len(tracks) - 1) if tracks else 0

    ttp = {int(t.track_index) for t in getattr(scenario, "tracks_to_predict", [])}
    ooi = set(int(x) for x in getattr(scenario, "objects_of_interest", []))

    current = min(int(getattr(scenario, "current_time_index", cfg.current_step)), cfg.total_steps - 1)

    # Order: SDC first, then nearest currently-valid tracks.
    def _track_xy(tr) -> np.ndarray:
        states = tr.states
        if current < len(states) and states[current].valid:
            return np.array([states[current].center_x, states[current].center_y], np.float32)
        return np.array([np.inf, np.inf], np.float32)

    sdc_xy = _track_xy(tracks[sdc_idx]) if tracks else np.zeros(2, np.float32)
    order = [sdc_idx]
    others = [i for i in range(len(tracks)) if i != sdc_idx]
    dists = [float(np.linalg.norm(_track_xy(tracks[i]) - sdc_xy)) for i in others]
    others = [i for _, i in sorted(zip(dists, others))]
    order.extend(others)
    order = order[: cfg.num_agents]

    for slot, src in enumerate(order):
        tr = tracks[src]
        sample["type"][slot] = int(getattr(tr, "object_type", C.TYPE_OTHER))
        sample["is_sdc"][slot] = src == sdc_idx
        sample["tracks_to_predict"][slot] = src in ttp
        sample["objects_of_interest"][slot] = src in ooi
        for t, st in enumerate(tr.states):
            if t >= cfg.total_steps:
                break
            if not st.valid:
                continue
            sample["xyz"][slot, t] = (st.center_x, st.center_y, getattr(st, "center_z", 0.0))
            sample["yaw"][slot, t] = st.heading
            sample["vel"][slot, t] = (st.velocity_x, st.velocity_y)
            sample["valid"][slot, t] = True
            if sample["length"][slot] == 0:
                sample["length"][slot] = getattr(st, "length", 0.0)
                sample["width"][slot] = getattr(st, "width", 0.0)
                sample["height"][slot] = getattr(st, "height", 0.0)
    sample["sdc_index"] = np.int32(0)

    # Map polylines closest to SDC.
    map_feats = []
    for mf in getattr(scenario, "map_features", []):
        pts = _polyline_from_map_feature(mf)
        if pts is None:
            continue
        centroid = pts.mean(axis=0)[:2]
        dist = float(np.linalg.norm(centroid - sdc_xy[:2]))
        map_feats.append((dist, pts, _map_type_from_feature(mf)))
    map_feats.sort(key=lambda x: x[0])
    for i, (_, pts, mtype) in enumerate(map_feats[: cfg.num_map_polylines]):
        xyz, valid = _resample_polyline(pts, cfg.polyline_points)
        sample["map_xyz"][i] = xyz
        sample["map_valid"][i] = valid
        sample["map_type"][i] = mtype

    # Traffic lights: dynamic_map_states[t].lane_states
    lane_ids: dict[int, int] = {}
    dyn = list(getattr(scenario, "dynamic_map_states", []))
    for t, dms in enumerate(dyn):
        if t >= cfg.total_steps:
            break
        for ls in getattr(dms, "lane_states", []):
            lid = int(getattr(ls, "lane", 0))
            if lid not in lane_ids:
                if len(lane_ids) >= cfg.num_traffic_lights:
                    continue
                lane_ids[lid] = len(lane_ids)
            slot = lane_ids[lid]
            stop = getattr(ls, "stop_point", None)
            if stop is not None:
                sample["tl_xyz"][slot, t] = (
                    getattr(stop, "x", 0.0),
                    getattr(stop, "y", 0.0),
                    getattr(stop, "z", 0.0),
                )
            sample["tl_state"][slot, t] = int(getattr(ls, "state", 0))
            sample["tl_valid"][slot, t] = True

    # sdc_paths (WOMD >= 1.3.1). Public GCS scenario shards often still omit them;
    # fall back to SDC future (+ nearby lane centerlines) so progress/DAC stay defined.
    paths = list(getattr(scenario, "sdc_paths", [])) + list(getattr(scenario, "sdc_paths_alt", []))
    for i, path in enumerate(paths[: cfg.num_route_paths]):
        pts_src = getattr(path, "path_points", None) or getattr(path, "waypoints", None) or []
        pts = np.array(
            [[getattr(p, "x", 0.0), getattr(p, "y", 0.0), getattr(p, "z", 0.0)] for p in pts_src],
            dtype=np.float32,
        )
        xyz, valid = _resample_polyline(pts, cfg.route_points)
        sample["route_xyz"][i] = xyz
        sample["route_valid"][i] = valid
        sample["route_on_route"][i] = bool(getattr(path, "on_route", True))

    if int(np.asarray(sample["route_valid"]).sum()) < 2:
        _synthesize_routes(sample, cfg, current)

    _fill_map_on_route(sample, current)
    return sample


def _fill_map_on_route(sample: dict[str, Any], current: int) -> None:
    """Pack-time nuPlan-style on-route lane corridor (read at train/cache)."""
    from womd_il.data.features import _map_on_route_flag
    from womd_il.data.transforms import agent_centric_xy

    p = int(sample["map_type"].shape[0])
    sample["map_on_route"] = np.zeros((p,), bool)
    rv = np.asarray(sample["route_valid"], dtype=bool)
    counts = rv.sum(axis=-1)
    on = np.asarray(sample.get("route_on_route", np.zeros(counts.shape[0], bool)), dtype=bool)
    usable = on & (counts >= 2)
    if not usable.any() and int(counts.max()) < 2:
        return
    ridx = int(np.argmax(usable)) if usable.any() else int(np.argmax(counts))
    if not bool(sample["valid"][0, current]):
        return
    origin = sample["xyz"][0, current, :2].astype(np.float32)
    yaw = float(sample["yaw"][0, current])
    map_xy = agent_centric_xy(sample["map_xyz"][..., :2], origin, yaw)
    route_xy = agent_centric_xy(sample["route_xyz"][ridx, :, :2], origin, yaw)
    flags = _map_on_route_flag(
        map_xy, sample["map_valid"], sample["map_type"], route_xy, rv[ridx]
    )[:, 0]
    sample["map_on_route"] = flags.astype(bool)


def _forward_lane_polylines(
    sample: dict[str, Any],
    current: int,
    *,
    max_paths: int,
    max_dist_m: float = 25.0,
    min_align: float = 0.25,
) -> list[np.ndarray]:
    """Nearby driving-lane polylines, cropped to run forward from ego."""
    if not bool(sample["valid"][0, current]):
        return []
    origin = sample["xyz"][0, current, :2].astype(np.float32)
    yaw = float(sample["yaw"][0, current])
    forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
    lane_types = {C.MAP_LANE_FREEWAY, C.MAP_LANE_SURFACE}
    scored: list[tuple[float, float, np.ndarray]] = []
    for i in range(int(sample["map_type"].shape[0])):
        if int(sample["map_type"][i]) not in lane_types:
            continue
        mvalid = np.asarray(sample["map_valid"][i], dtype=bool)
        if int(mvalid.sum()) < 2:
            continue
        pts = np.asarray(sample["map_xyz"][i, mvalid], dtype=np.float32)
        d = np.linalg.norm(pts[:, :2] - origin[None, :], axis=-1)
        j = int(np.argmin(d))
        if float(d[j]) > max_dist_m:
            continue
        # Orient polyline so index increases roughly along ego heading.
        tang = pts[-1, :2] - pts[0, :2]
        tang_n = float(np.linalg.norm(tang))
        if tang_n < 1e-3:
            continue
        align = float(np.dot(tang / tang_n, forward))
        if align < 0.0:
            pts = pts[::-1].copy()
            j = int(len(pts) - 1 - j)
            align = -align
        if align < min_align:
            continue
        fwd = pts[j:]
        if len(fwd) < 2:
            continue
        scored.append((float(d[j]), -align, fwd))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [p for _, _, p in scored[: max(0, int(max_paths))]]


def _synthesize_routes(sample: dict[str, Any], cfg: PackConfig, current: int) -> None:
    """Approximate routes when WOMD ``sdc_paths`` are missing.

    Path 0 (on-route): SDC future track if long enough, else nearest forward lane.
    Extra slots: other nearby forward lane centerlines (off-route alternatives).
    """
    r = int(cfg.num_route_paths)
    q = int(cfg.route_points)
    fut = []
    for t in range(int(current), int(cfg.total_steps)):
        if bool(sample["valid"][0, t]):
            fut.append(sample["xyz"][0, t])
    fut_pts = np.asarray(fut, dtype=np.float32) if fut else np.zeros((0, 3), np.float32)
    fut_len = 0.0
    if len(fut_pts) >= 2:
        fut_len = float(np.linalg.norm(np.diff(fut_pts[:, :2], axis=0), axis=1).sum())

    lanes = _forward_lane_polylines(sample, current, max_paths=max(r, 4))
    primary: np.ndarray | None = None
    primary_from_lane = False
    if len(fut_pts) >= 2 and fut_len >= 5.0:
        primary = fut_pts
    elif lanes:
        primary = lanes[0]
        primary_from_lane = True
    elif len(fut_pts) >= 2:
        primary = fut_pts

    filled = 0
    if primary is not None and len(primary) >= 2:
        xyz, valid = _resample_polyline(primary, q)
        sample["route_xyz"][0] = xyz
        sample["route_valid"][0] = valid
        sample["route_on_route"][0] = True
        filled = 1

    alt_lanes = lanes[1:] if primary_from_lane else lanes
    for lane in alt_lanes:
        if filled >= r:
            break
        if filled > 0:
            a0 = sample["route_xyz"][0, 0, :2]
            a1 = sample["route_xyz"][0, -1, :2]
            if float(np.linalg.norm(lane[0, :2] - a0)) < 2.0 and float(
                np.linalg.norm(lane[-1, :2] - a1)
            ) < 8.0:
                continue
        xyz, valid = _resample_polyline(lane, q)
        if int(valid.sum()) < 2:
            continue
        sample["route_xyz"][filled] = xyz
        sample["route_valid"][filled] = valid
        sample["route_on_route"][filled] = False
        filled += 1


def parse_tfrecord_bytes(raw: bytes, cfg: PackConfig | None = None) -> dict[str, Any]:
    try:
        from womd_il.protos import scenario_pb2
    except ImportError:
        try:
            from waymo_open_dataset.protos import scenario_pb2
        except ImportError as exc:
            raise ImportError(
                "Could not import WOMD scenario protobufs. "
                "Reinstall womd-il or pip install waymo-open-dataset."
            ) from exc
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(raw)
    return parse_scenario_proto(scenario, cfg)
