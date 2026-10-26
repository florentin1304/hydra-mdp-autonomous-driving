"""Simple IDM speed control along a logged polyline (numpy fallback for Waymax IDM)."""

from __future__ import annotations

import numpy as np


def idm_accel(
    v: float,
    v0: float,
    gap: float,
    v_lead: float,
    a_max: float = 2.0,
    b: float = 4.0,
    delta: float = 4.0,
    s0: float = 2.0,
    t_head: float = 1.5,
) -> float:
    s_star = s0 + max(0.0, v * t_head + v * (v - v_lead) / (2.0 * np.sqrt(max(a_max * b, 1e-6))))
    gap = max(gap, 0.1)
    return a_max * (1.0 - (v / max(v0, 0.1)) ** delta - (s_star / gap) ** 2)


def nearest_arclength(path: np.ndarray, xy: np.ndarray) -> tuple[float, int]:
    d = np.linalg.norm(path - xy[None, :], axis=-1)
    i = int(np.argmin(d))
    if i >= len(path) - 1:
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return float(seg.sum()), i
    return float(np.linalg.norm(np.diff(path[: i + 1], axis=0), axis=1).sum()) if i > 0 else 0.0, i


def advance_on_path(path: np.ndarray, idx: int, dist: float) -> tuple[np.ndarray, float, int]:
    """Walk `dist` meters along `path` starting at segment `idx`."""
    xy = path[min(idx, len(path) - 1)].copy()
    remaining = dist
    i = idx
    while remaining > 0 and i < len(path) - 1:
        delta = path[i + 1] - xy
        seglen = float(np.linalg.norm(delta))
        if seglen < 1e-6:
            i += 1
            xy = path[i].copy()
            continue
        if remaining >= seglen:
            remaining -= seglen
            i += 1
            xy = path[i].copy()
        else:
            xy = xy + delta * (remaining / seglen)
            remaining = 0.0
    heading = 0.0
    if i < len(path) - 1:
        d = path[min(i + 1, len(path) - 1)] - path[i]
        heading = float(np.arctan2(d[1], d[0]))
    elif i > 0:
        d = path[i] - path[i - 1]
        heading = float(np.arctan2(d[1], d[0]))
    return xy, heading, i


def lead_gap_on_heading(
    xy: np.ndarray,
    yaw: float,
    others_xy: np.ndarray,
    others_speed: np.ndarray,
    others_valid: np.ndarray,
    cone_cos: float = 0.5,
    max_range: float = 40.0,
) -> tuple[float, float]:
    """Return (gap, lead_speed) for the nearest agent roughly ahead."""
    if len(others_xy) == 0:
        return max_range, 0.0
    rel = others_xy - xy[None, :]
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    proj = rel @ fwd
    lat = np.linalg.norm(rel - proj[:, None] * fwd[None, :], axis=-1)
    dist = np.linalg.norm(rel, axis=-1)
    ahead = others_valid & (proj > 0.5) & (dist < max_range) & (lat < 2.5)
    if not np.any(ahead):
        return max_range, 0.0
    dist_a = np.where(ahead, dist, np.inf)
    j = int(np.argmin(dist_a))
    return float(dist_a[j]), float(others_speed[j])
