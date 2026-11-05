"""Build agent-centric encoder tensors from a packed world-frame sample."""

from __future__ import annotations

import numpy as np
import torch

from womd_il import constants as C
from womd_il.data.transforms import (
    agent_centric_xy,
    agent_centric_yaw,
    inverse_agent_centric_xy,
    wrap_angle,
)


def _one_hot(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(idx.shape + (n,), dtype=np.float32)
    clipped = np.clip(idx.astype(np.int64), 0, n - 1)
    flat = out.reshape(-1, n)
    flat[np.arange(flat.shape[0]), clipped.reshape(-1)] = 1.0
    return out


def _finite_diff(x: np.ndarray, valid: np.ndarray, dt: float) -> np.ndarray:
    """Backward difference; zeros where either endpoint is invalid."""
    dx = np.zeros_like(x)
    dx[..., 1:, :] = (x[..., 1:, :] - x[..., :-1, :]) / dt
    ok = valid[..., 1:] & valid[..., :-1]
    while ok.ndim < dx.ndim:
        ok = ok[..., None]
    dx[..., 1:, :] *= ok
    return dx


_ROUTE_LANE_TYPES = (C.MAP_LANE_FREEWAY, C.MAP_LANE_SURFACE)


def _polyline_shape_feats(map_xy: np.ndarray, map_valid: np.ndarray):
    """Start/end/mid/quartiles and max lateral bow from the chord."""
    n = map_xy.shape[0]
    start = np.zeros((n, 2), np.float32)
    end = np.zeros((n, 2), np.float32)
    mid = np.zeros((n, 2), np.float32)
    q1 = np.zeros((n, 2), np.float32)
    q3 = np.zeros((n, 2), np.float32)
    bow = np.zeros((n, 1), np.float32)
    valid = np.asarray(map_valid, dtype=bool)
    for i in range(n):
        pts = map_xy[i, valid[i]]
        if pts.shape[0] == 0:
            continue
        start[i] = pts[0]
        end[i] = pts[-1]
        mid[i] = pts.mean(axis=0)
        idx = np.linspace(0, pts.shape[0] - 1, 5).round().astype(np.int64)
        q1[i] = pts[idx[1]]
        q3[i] = pts[idx[3]]
        chord = end[i] - start[i]
        clen = float(np.linalg.norm(chord))
        if clen < 1e-3:
            continue
        u = chord / clen
        rel = pts - start[i]
        lat = rel - (rel @ u)[:, None] * u
        bow[i, 0] = np.linalg.norm(lat, axis=-1).max()
    return start, end, mid, q1, q3, bow


def _map_on_route_flag(
    map_xy: np.ndarray,
    map_valid: np.ndarray,
    map_type: np.ndarray,
    route_xy: np.ndarray,
    route_valid: np.ndarray,
    thresh: float = C.ROUTE_LANE_THRESH_M,
    min_frac: float = 0.35,
    heading_cos: float = C.ROUTE_LANE_HEADING_COS,
) -> np.ndarray:
    """Flag lane centerlines in a nuPlan-style on-route corridor.

    Like nuPlan roadblocks: all same-direction lanes whose centerlines sit in a
    lateral band around the preferred path (not only the driven lane). A lane
    point counts if it is within ``thresh`` of the route **and** its local
    tangent aligns with the nearest route tangent (rejects cross streets).
    """
    n_poly = map_xy.shape[0]
    flags = np.zeros((n_poly, 1), dtype=np.float32)
    rpts = route_xy[np.asarray(route_valid, dtype=bool)].astype(np.float32)
    if rpts.shape[0] < 2:
        return flags

    # Route tangents at each route sample (forward difference).
    rt = np.diff(rpts, axis=0)
    rt = np.concatenate([rt, rt[-1:]], axis=0)
    rtn = np.linalg.norm(rt, axis=-1, keepdims=True)
    rt = rt / np.clip(rtn, 1e-3, None)

    map_xy_f = map_xy.astype(np.float32)
    map_sq = (map_xy_f**2).sum(axis=-1, keepdims=True)
    route_sq = (rpts**2).sum(axis=-1)
    cross = map_xy_f @ rpts.T
    d2 = np.clip(map_sq + route_sq - 2.0 * cross, 0.0, None)
    nn = d2.argmin(axis=-1)  # [P, M]
    min_d = np.sqrt(d2.min(axis=-1))

    # Per-point lane tangent from consecutive valid samples (fallback: zeros).
    lane_t = np.zeros_like(map_xy_f)
    lane_t[:, :-1] = map_xy_f[:, 1:] - map_xy_f[:, :-1]
    lane_t[:, -1] = lane_t[:, -2]
    lt_n = np.linalg.norm(lane_t, axis=-1, keepdims=True)
    lane_t = lane_t / np.clip(lt_n, 1e-3, None)

    # Gather route tangent at nearest route point: [P, M, 2]
    rt_nn = rt[nn]
    align = np.abs((lane_t * rt_nn).sum(axis=-1))

    valid = np.asarray(map_valid, dtype=bool)
    n_valid = valid.sum(axis=1)
    on_pt = valid & (min_d < thresh) & (align >= heading_cos)
    n_on = on_pt.sum(axis=1)
    frac = n_on / np.clip(n_valid, 1, None)
    is_lane = np.isin(map_type, _ROUTE_LANE_TYPES)
    flags[:, 0] = ((n_valid > 0) & (frac >= min_frac) & is_lane).astype(np.float32)
    return flags


def apply_origin_perturb(
    origin_xy: np.ndarray,
    origin_yaw: float,
    perturb: dict,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """PlanTF pose noise in the current ego frame, returned as a new world origin.

    The log future is *not* rewritten to a return-to-log spline; callers
    re-normalize every tensor into this frame and supervise the original
    expert trajectory there.
    """
    if not perturb:
        return origin_xy, origin_yaw
    explicit = any(k in perturb for k in ("lon", "lat", "dyaw"))
    if explicit:
        lon = float(perturb.get("lon", 0.0))
        lat = float(perturb.get("lat", 0.0))
        dyaw = float(perturb.get("dyaw", 0.0))
    else:
        prob = float(perturb.get("prob", 0.0))
        if rng is None or prob <= 0.0 or rng.random() >= prob:
            return origin_xy, origin_yaw
        # Optional PlanTF-style UniformNoise on [x, y, yaw] when dist=uniform
        # or *_{low,high} bounds are set; otherwise Gaussian via xy_std / yaw_std.
        use_uniform = (
            str(perturb.get("dist", "")).lower() == "uniform"
            or any(
                k in perturb
                for k in ("lon_low", "lon_high", "lat_low", "lat_high", "yaw_low", "yaw_high")
            )
        )
        if use_uniform:
            lon = float(
                rng.uniform(float(perturb.get("lon_low", -1.0)), float(perturb.get("lon_high", 1.0)))
            )
            lat = float(
                rng.uniform(float(perturb.get("lat_low", -0.75)), float(perturb.get("lat_high", 0.75)))
            )
            dyaw = float(
                rng.uniform(float(perturb.get("yaw_low", -0.35)), float(perturb.get("yaw_high", 0.35)))
            )
        else:
            xy_std = float(perturb.get("xy_std", 0.5))
            yaw_std = float(perturb.get("yaw_std", 0.2))
            lon = float(np.clip(rng.normal(0.0, xy_std), -3.0 * xy_std, 3.0 * xy_std))
            lat = float(np.clip(rng.normal(0.0, xy_std), -3.0 * xy_std, 3.0 * xy_std))
            dyaw = float(np.clip(rng.normal(0.0, yaw_std), -3.0 * yaw_std, 3.0 * yaw_std))
    delta = np.array([lon, lat], dtype=np.float32)
    new_xy = inverse_agent_centric_xy(delta, origin_xy, origin_yaw).astype(np.float32)
    new_yaw = float(wrap_angle(origin_yaw + dyaw))
    return new_xy, new_yaw


def build_sample_features(
    sample: dict,
    *,
    target_idx: int = 0,
    current_step: int | None = None,
    history_steps: int = C.HISTORY_STEPS,
    future_steps: int = C.FUTURE_STEPS,
    dt: float = C.DT,
    perturb: dict | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Return numpy feature dict in the *target* agent's current frame.

    ``perturb`` (train only) jitters that origin then rebuilds every
    agent-centric tensor, including the log future. Val / closed-loop
    leave it unset so the origin stays the true pose.
    """
    xyz = sample["xyz"]
    yaw = sample["yaw"]
    vel = sample["vel"]
    valid = sample["valid"]
    a, t_all, _ = xyz.shape
    t0 = C.CURRENT_STEP if current_step is None else int(current_step)
    t0 = min(max(t0, 0), t_all - 1)
    t_hist0 = max(t0 - history_steps + 1, 0)

    origin_xy = xyz[target_idx, t0, :2].astype(np.float32)
    origin_yaw = float(yaw[target_idx, t0])
    origin_valid = bool(valid[target_idx, t0])
    if not origin_valid:
        origin_xy = np.zeros(2, np.float32)
        origin_yaw = 0.0
    elif perturb:
        origin_xy, origin_yaw = apply_origin_perturb(origin_xy, origin_yaw, perturb, rng)

    hist_sl = slice(t_hist0, t0 + 1)
    fut_sl = slice(t0 + 1, min(t0 + 1 + future_steps, t_all))

    xy_w = xyz[:, :, :2]
    xy = agent_centric_xy(xy_w, origin_xy, origin_yaw)
    vel_xy = agent_centric_xy(vel, np.zeros(2, np.float32), origin_yaw)
    yaw_l = agent_centric_yaw(yaw, origin_yaw)

    hist_xy = xy[:, hist_sl]
    hist_vel = vel_xy[:, hist_sl]
    hist_yaw = yaw_l[:, hist_sl]
    hist_valid = valid[:, hist_sl]
    # Pad history on the left if the window is short (closed-loop early steps).
    need_h = history_steps - hist_xy.shape[1]
    if need_h > 0:
        pad = lambda arr, extra_dim=False: np.concatenate(
            [np.zeros((arr.shape[0], need_h) + arr.shape[2:], dtype=arr.dtype), arr],
            axis=1,
        )
        hist_xy = pad(hist_xy)
        hist_vel = pad(hist_vel)
        hist_yaw = pad(hist_yaw)
        hist_valid = pad(hist_valid)

    hist_acc = _finite_diff(hist_vel, hist_valid, dt)
    yaw_un = np.unwrap(hist_yaw, axis=1)
    yaw_rate = np.zeros_like(hist_yaw)
    yaw_rate[:, 1:] = (yaw_un[:, 1:] - yaw_un[:, :-1]) / dt
    yaw_rate[:, 1:] *= hist_valid[:, 1:] & hist_valid[:, :-1]

    types = np.broadcast_to(sample["type"][:, None], hist_valid.shape)
    size = np.stack([sample["length"], sample["width"], sample["height"]], axis=-1)
    size_b = np.broadcast_to(size[:, None, :], hist_xy.shape[:2] + (3,))
    type_oh = _one_hot(types, C.NUM_OBJECT_TYPES)

    hist_feat = np.concatenate(
        [
            hist_xy.astype(np.float32),
            hist_vel.astype(np.float32),
            hist_acc.astype(np.float32),
            np.stack([np.sin(hist_yaw), np.cos(hist_yaw)], axis=-1).astype(np.float32),
            yaw_rate[..., None].astype(np.float32),
            size_b.astype(np.float32),
            type_oh,
            hist_valid[..., None].astype(np.float32),
        ],
        axis=-1,
    )
    assert hist_feat.shape[-1] == C.HIST_IN_DIM, hist_feat.shape

    sdc_hist = hist_feat[target_idx]
    sdc_hist_mask = hist_valid[target_idx]
    neigh_idx = [i for i in range(a) if i != target_idx]
    neighbors = hist_feat[neigh_idx]
    neighbors_mask = hist_valid[neigh_idx]

    # Preferred route: first on-route path with ≥2 pts, else longest valid path.
    route_xy_w = sample["route_xyz"][..., :2]
    route_valid = sample["route_valid"]
    on_route = sample.get("route_on_route")
    counts = np.asarray(route_valid, dtype=bool).sum(axis=-1)
    ridx = 0
    if on_route is not None:
        on = np.asarray(on_route, dtype=bool) & (counts >= 2)
        if np.any(on):
            ridx = int(np.argmax(on))
        elif int(counts.max()) >= 2:
            ridx = int(np.argmax(counts))
    elif int(counts.max()) >= 2:
        ridx = int(np.argmax(counts))
    sdc_route = agent_centric_xy(route_xy_w[ridx], origin_xy, origin_yaw)
    sdc_route_mask = route_valid[ridx]

    # Map polylines (chord + interior quartiles + bow for curvature)
    map_xyz = sample["map_xyz"]
    map_valid = sample["map_valid"]
    map_type = sample["map_type"]
    map_xy = agent_centric_xy(map_xyz[..., :2], origin_xy, origin_yaw)
    start, end, mid, q1, q3, bow = _polyline_shape_feats(map_xy, map_valid)
    direction = end - start
    length = np.linalg.norm(direction, axis=-1, keepdims=True)
    unit = direction / np.clip(length, 1e-3, None)
    dist = np.linalg.norm(mid, axis=-1, keepdims=True)
    map_poly_valid = map_valid.any(axis=-1)
    # Prefer pack-time corridor flags (no per-step geometry in the train loop).
    packed = sample.get("map_on_route")
    if packed is not None:
        on_route_flag = np.asarray(packed, dtype=np.float32).reshape(-1, 1)
    else:
        on_route_flag = _map_on_route_flag(
            map_xy, map_valid, map_type, sdc_route, sdc_route_mask
        )
    map_feat = np.concatenate(
        [
            start,
            end,
            mid,
            q1,
            q3,
            unit.astype(np.float32),
            length.astype(np.float32),
            bow.astype(np.float32),
            _one_hot(map_type, C.NUM_MAP_TYPES),
            dist.astype(np.float32),
            on_route_flag,
        ],
        axis=-1,
    )
    assert map_feat.shape[-1] == C.MAP_IN_DIM, map_feat.shape

    n_route = sdc_route.shape[0]
    route_prog = np.linspace(0.0, 1.0, n_route, dtype=np.float32)[:, None]
    route_feat = np.concatenate([sdc_route.astype(np.float32), route_prog], axis=-1)
    assert route_feat.shape[-1] == C.ROUTE_IN_DIM, route_feat.shape

    # Traffic lights over the 1 s history (Wayformer [T, S_tls]); t=0 + future for TLC.
    tl_xy = agent_centric_xy(sample["tl_xyz"][:, t0, :2], origin_xy, origin_yaw)
    tl_state = sample["tl_state"][:, t0]
    tl_valid = sample["tl_valid"][:, t0]
    tl_xy_h = agent_centric_xy(sample["tl_xyz"][:, hist_sl, :2], origin_xy, origin_yaw)
    tl_state_h = sample["tl_state"][:, hist_sl]
    tl_valid_h = sample["tl_valid"][:, hist_sl]
    if need_h > 0:
        zxy = np.zeros((tl_xy_h.shape[0], need_h, 2), np.float32)
        zs = np.zeros((tl_state_h.shape[0], need_h), dtype=tl_state_h.dtype)
        zv = np.zeros((tl_valid_h.shape[0], need_h), dtype=tl_valid_h.dtype)
        tl_xy_h = np.concatenate([zxy, tl_xy_h], 1)
        tl_state_h = np.concatenate([zs, tl_state_h], 1)
        tl_valid_h = np.concatenate([zv, tl_valid_h], 1)
    tl_feat = np.concatenate(
        [
            tl_xy_h.astype(np.float32),
            _one_hot(tl_state_h, C.NUM_TL_STATES),
        ],
        axis=-1,
    )
    assert tl_feat.shape[-1] == C.TL_IN_DIM, tl_feat.shape
    # Future TL timeline aligned with gt / other_fut (t0+1 …).
    tl_fut_xy = agent_centric_xy(sample["tl_xyz"][:, fut_sl, :2], origin_xy, origin_yaw)
    tl_fut_state = sample["tl_state"][:, fut_sl]
    tl_fut_valid = sample["tl_valid"][:, fut_sl]
    need_f_tl = future_steps - tl_fut_xy.shape[1]
    if need_f_tl > 0:
        zxy = np.zeros((tl_fut_xy.shape[0], need_f_tl, 2), np.float32)
        zs = np.zeros((tl_fut_state.shape[0], need_f_tl), dtype=tl_fut_state.dtype)
        zv = np.zeros((tl_fut_valid.shape[0], need_f_tl), dtype=tl_fut_valid.dtype)
        tl_fut_xy = np.concatenate([tl_fut_xy, zxy], 1)
        tl_fut_state = np.concatenate([tl_fut_state, zs], 1)
        tl_fut_valid = np.concatenate([tl_fut_valid, zv], 1)

    speed = float(np.linalg.norm(vel[target_idx, t0]))
    last_acc = hist_acc[target_idx, -1]
    ego_state = np.array(
        [
            speed,
            float(last_acc[0]),
            float(last_acc[1]),
            float(np.sin(0.0)),
            float(np.cos(0.0)),
            float(sample["length"][target_idx]),
            float(sample["width"][target_idx]),
            float(sample["height"][target_idx]),
            float(sample["type"][target_idx]),
            1.0 if origin_valid else 0.0,
            float(t0),
            dt,
        ],
        dtype=np.float32,
    )
    assert ego_state.shape[-1] == C.EGO_IN_DIM

    fut_xy = xy[target_idx, fut_sl]
    fut_yaw = yaw_l[target_idx, fut_sl]
    fut_valid = valid[target_idx, fut_sl]
    other_fut_xy = xy[neigh_idx][:, fut_sl]
    other_fut_yaw = yaw_l[neigh_idx][:, fut_sl]
    other_fut_valid = valid[neigh_idx][:, fut_sl]
    other_lw = size[neigh_idx, :2]
    need_f = future_steps - fut_xy.shape[0]
    if need_f > 0:
        fut_xy = np.concatenate([fut_xy, np.zeros((need_f, 2), np.float32)], 0)
        fut_yaw = np.concatenate([fut_yaw, np.zeros((need_f,), np.float32)], 0)
        fut_valid = np.concatenate([fut_valid, np.zeros((need_f,), bool)], 0)
        zxy = np.zeros((other_fut_xy.shape[0], need_f, 2), np.float32)
        zyaw = np.zeros((other_fut_yaw.shape[0], need_f), np.float32)
        zv = np.zeros((other_fut_valid.shape[0], need_f), dtype=other_fut_valid.dtype)
        other_fut_xy = np.concatenate([other_fut_xy, zxy], 1)
        other_fut_yaw = np.concatenate([other_fut_yaw, zyaw], 1)
        other_fut_valid = np.concatenate([other_fut_valid, zv], 1)

    return {
        "sdc_hist": sdc_hist.astype(np.float32),
        "sdc_hist_mask": sdc_hist_mask.astype(np.float32),
        "neighbors": neighbors.astype(np.float32),
        "neighbors_mask": neighbors_mask.astype(np.float32),
        "map_feat": map_feat.astype(np.float32),
        "map_mask": map_poly_valid.astype(np.float32),
        "map_xy": map_xy.astype(np.float32),
        "map_pt_valid": map_valid.astype(np.float32),
        "map_type": map_type.astype(np.int32),
        "route_feat": route_feat.astype(np.float32),
        "tl_feat": tl_feat.astype(np.float32),
        "tl_mask": tl_valid_h.astype(np.float32),
        "tl_xy": tl_xy.astype(np.float32),
        "tl_state": tl_state.astype(np.int32),
        "tl_valid": tl_valid.astype(np.float32),
        "tl_fut_xy": tl_fut_xy.astype(np.float32),
        "tl_fut_state": tl_fut_state.astype(np.int32),
        "tl_fut_valid": tl_fut_valid.astype(np.float32),
        "ego_state": ego_state,
        "current_xyh_world": np.array(
            [origin_xy[0], origin_xy[1], origin_yaw], dtype=np.float32
        ),
        "gt_xy": fut_xy.astype(np.float32),
        "gt_yaw": fut_yaw.astype(np.float32),
        "gt_valid": fut_valid.astype(np.float32),
        "other_fut_xy": other_fut_xy.astype(np.float32),
        "other_fut_yaw": other_fut_yaw.astype(np.float32),
        "other_fut_valid": other_fut_valid.astype(np.float32),
        "other_lw": other_lw.astype(np.float32),
        "ego_lw": size[target_idx, :2].astype(np.float32),
        "sdc_route": sdc_route.astype(np.float32),
        "sdc_route_mask": sdc_route_mask.astype(np.float32),
        "speed": np.float32(speed),
        "target_idx": np.int32(target_idx),
    }


def features_to_tensors(feat: dict[str, np.ndarray], device=None) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in feat.items():
        if k in ("scenario_id",):
            continue
        t = torch.from_numpy(np.asarray(v))
        if device is not None:
            t = t.to(device)
        out[k] = t
    return out
