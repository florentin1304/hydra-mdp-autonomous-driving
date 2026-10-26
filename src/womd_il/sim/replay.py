"""Packed-scenario closed-loop simulator (log-replay or IDM neighbors)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from womd_il import constants as C
from womd_il.data.features import build_sample_features
from womd_il.sim.bicycle import bicycle_step
from womd_il.sim.idm import advance_on_path, idm_accel, lead_gap_on_heading, nearest_arclength


@dataclass
class SimState:
    t: int
    sample: dict
    xyz: np.ndarray
    yaw: np.ndarray
    vel: np.ndarray
    valid: np.ndarray
    log_xyz: np.ndarray
    log_yaw: np.ndarray
    log_vel: np.ndarray
    log_valid: np.ndarray
    executed_xy: list = field(default_factory=list)
    executed_yaw: list = field(default_factory=list)
    collisions: list = field(default_factory=list)
    offroad: list = field(default_factory=list)
    predicted_xy: list = field(default_factory=list)
    predicted_t: list = field(default_factory=list)
    # Optional viz: at each replan, top-N world trajs + probs + pick index.
    # Each entry: {"t": int, "xy": [N,T,2], "probs": [N], "pick": int}
    predicted_modes: list = field(default_factory=list)

    @property
    def sdc(self) -> int:
        return 0

    def current_speed(self) -> float:
        return float(np.linalg.norm(self.vel[self.sdc, self.t]))


def _boxes_overlap(xy1, yaw1, lw1, xy2, yaw2, lw2) -> bool:
    """Coarse oriented-box overlap via circle test then AABB in box1 frame."""
    r1 = 0.5 * float(np.hypot(*lw1))
    r2 = 0.5 * float(np.hypot(*lw2))
    if np.linalg.norm(xy1 - xy2) > r1 + r2:
        return False
    c, s = np.cos(-yaw1), np.sin(-yaw1)
    d = xy2 - xy1
    local = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
    return abs(local[0]) < 0.5 * lw1[0] + 0.5 * lw2[0] and abs(local[1]) < 0.5 * lw1[1] + 0.5 * lw2[1]


def _seg_cross(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """Proper 2D segment intersection of pq and ab."""
    def cross(u, v):
        return float(u[0] * v[1] - u[1] * v[0])

    pq, ab = q - p, b - a
    o1, o2 = cross(pq, a - p), cross(pq, b - p)
    o3, o4 = cross(ab, p - a), cross(ab, q - a)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


# Keep in sync with ``losses.hydra._DAC_MAX_LANE_M`` (open-loop DAC lane gate).
_OFFROAD_MAX_LANE_M = 4.0
_OFFROAD_LANE_TYPES = (C.MAP_LANE_FREEWAY, C.MAP_LANE_SURFACE)


def _point_to_polylines_m(xy: np.ndarray, polys: np.ndarray, valid: np.ndarray) -> float:
    """Min distance from ``xy`` to any valid polyline segment. ``inf`` if none."""
    q = np.asarray(xy, dtype=np.float64).reshape(2)
    best = float("inf")
    for poly, ok in zip(polys, valid):
        pts = np.asarray(poly[ok], dtype=np.float64)
        if pts.shape[0] < 2:
            continue
        a, b = pts[:-1], pts[1:]
        ab = b - a
        denom = np.maximum((ab * ab).sum(-1), 1e-12)
        t = np.clip(((q - a) * ab).sum(-1) / denom, 0.0, 1.0)
        d = np.linalg.norm(q - a - t[:, None] * ab, axis=-1).min()
        if d < best:
            best = float(d)
    return best


def _lane_mask(sample: dict) -> np.ndarray:
    """Driving-lane polylines; prefer on-route corridor when packed."""
    mtype = sample["map_type"]
    lane = np.zeros(mtype.shape[0], dtype=bool)
    for t in _OFFROAD_LANE_TYPES:
        lane |= mtype == int(t)
    on_route = sample.get("map_on_route")
    if on_route is not None:
        corridor = lane & np.asarray(on_route, dtype=bool).reshape(-1)
        if np.any(corridor):
            return corridor
    return lane


def _offroad(sample: dict, xy: np.ndarray, prev_xy: np.ndarray | None = None) -> bool:
    """True if off the driving surface — matches open-loop Hydra DAC.

    Fails if:
      1. ``prev_xy → xy`` crosses a road-edge polyline (needs a segment), or
      2. ``xy`` is farther than ``_OFFROAD_MAX_LANE_M`` from a driving lane
         (on-route lanes when ``map_on_route`` is packed, else any lane).

    (2) catches gap/island cuts that never cross a digitized edge — the case
    closed-loop previously missed after DAC gained the lane gate.
    """
    q = np.asarray(xy, dtype=np.float64).reshape(2)
    if prev_xy is not None:
        edges = sample["map_type"] == C.MAP_ROAD_EDGE
        if np.any(edges):
            polys = sample["map_xyz"][edges, :, :2]
            valid = sample["map_valid"][edges]
            p = np.asarray(prev_xy, dtype=np.float64).reshape(2)
            for poly, ok in zip(polys, valid):
                pts = poly[ok]
                if pts.shape[0] < 2:
                    continue
                for i in range(pts.shape[0] - 1):
                    if _seg_cross(p, q, pts[i], pts[i + 1]):
                        return True
    lanes = _lane_mask(sample)
    if np.any(lanes):
        d = _point_to_polylines_m(q, sample["map_xyz"][lanes, :, :2], sample["map_valid"][lanes])
        if np.isfinite(d) and d > _OFFROAD_MAX_LANE_M:
            return True
    return False


class PackedSimulator:
    def __init__(self, sample: dict, neighbors: str = "log", dt: float = C.DT):
        if neighbors not in ("log", "idm"):
            raise ValueError("neighbors must be 'log' or 'idm'")
        self.neighbors = neighbors
        self.dt = dt
        self.log_xyz = np.array(sample["xyz"], copy=True)
        self.log_yaw = np.array(sample["yaw"], copy=True)
        self.log_vel = np.array(sample["vel"], copy=True)
        self.log_valid = np.array(sample["valid"], copy=True)
        self.sample_template = sample
        self.state = self.reset()

    def reset(self) -> SimState:
        t0 = C.CURRENT_STEP
        st = SimState(
            t=t0,
            sample=self.sample_template,
            xyz=self.log_xyz.copy(),
            yaw=self.log_yaw.copy(),
            vel=self.log_vel.copy(),
            valid=self.log_valid.copy(),
            log_xyz=self.log_xyz,
            log_yaw=self.log_yaw,
            log_vel=self.log_vel,
            log_valid=self.log_valid,
        )
        st.executed_xy.append(st.xyz[0, t0, :2].copy())
        st.executed_yaw.append(float(st.yaw[0, t0]))
        self.state = st
        return st

    def features(self) -> dict:
        st = self.state
        sample = dict(self.sample_template)
        sample["xyz"] = st.xyz
        sample["yaw"] = st.yaw
        sample["vel"] = st.vel
        sample["valid"] = st.valid
        return build_sample_features(sample, target_idx=0, current_step=st.t)

    def _step_neighbors(self, nxt: int) -> None:
        st = self.state
        a = st.xyz.shape[0]
        if self.neighbors == "log":
            for i in range(1, a):
                st.xyz[i, nxt] = st.log_xyz[i, nxt]
                st.yaw[i, nxt] = st.log_yaw[i, nxt]
                st.vel[i, nxt] = st.log_vel[i, nxt]
                st.valid[i, nxt] = st.log_valid[i, nxt]
            return
        # IDM along each neighbor's logged path.
        sdc_xy = st.xyz[0, st.t, :2]
        sdc_speed = st.current_speed()
        for i in range(1, a):
            if not st.valid[i, st.t]:
                st.valid[i, nxt] = False
                continue
            path = st.log_xyz[i, :, :2]
            valid_path = st.log_valid[i]
            pts = path[valid_path]
            if len(pts) < 2:
                st.xyz[i, nxt] = st.xyz[i, st.t]
                st.yaw[i, nxt] = st.yaw[i, st.t]
                st.vel[i, nxt] = st.vel[i, st.t]
                st.valid[i, nxt] = True
                continue
            xy = st.xyz[i, st.t, :2]
            yaw = float(st.yaw[i, st.t])
            speed = float(np.linalg.norm(st.vel[i, st.t]))
            v0 = max(speed, float(np.linalg.norm(st.log_vel[i, st.t])), 5.0)
            others_xy = np.vstack([sdc_xy, st.xyz[1:i, st.t, :2], st.xyz[i + 1 :, st.t, :2]])
            others_sp = np.array(
                [sdc_speed]
                + [float(np.linalg.norm(st.vel[j, st.t])) for j in range(1, a) if j != i]
            )
            others_ok = np.array(
                [True] + [bool(st.valid[j, st.t]) for j in range(1, a) if j != i]
            )
            gap, v_lead = lead_gap_on_heading(xy, yaw, others_xy, others_sp, others_ok)
            acc = float(np.clip(idm_accel(speed, v0, gap, v_lead), -6.0, 2.0))
            dist = max(0.0, speed * self.dt + 0.5 * acc * self.dt**2)
            _, idx = nearest_arclength(pts, xy)
            new_xy, new_yaw, _ = advance_on_path(pts, idx, dist)
            new_speed = max(0.0, speed + acc * self.dt)
            st.xyz[i, nxt, :2] = new_xy
            st.xyz[i, nxt, 2] = st.xyz[i, st.t, 2]
            st.yaw[i, nxt] = new_yaw
            st.vel[i, nxt] = (new_speed * np.cos(new_yaw), new_speed * np.sin(new_yaw))
            st.valid[i, nxt] = True

    def step(self, accel: float, kappa: float) -> SimState:
        st = self.state
        nxt = st.t + 1
        if nxt >= st.xyz.shape[1]:
            return st
        x, y = st.xyz[0, st.t, :2]
        yaw = float(st.yaw[0, st.t])
        speed = st.current_speed()
        nx, ny, nyaw, nspeed = bicycle_step(float(x), float(y), yaw, speed, accel, kappa, dt=self.dt)
        st.xyz[0, nxt, 0] = nx
        st.xyz[0, nxt, 1] = ny
        st.xyz[0, nxt, 2] = st.xyz[0, st.t, 2]
        st.yaw[0, nxt] = nyaw
        st.vel[0, nxt] = (nspeed * np.cos(nyaw), nspeed * np.sin(nyaw))
        st.valid[0, nxt] = True
        self._step_neighbors(nxt)
        st.t = nxt
        st.executed_xy.append(np.array([nx, ny], np.float32))
        st.executed_yaw.append(nyaw)
        st.collisions.append(self._sdc_collision())
        st.offroad.append(_offroad(st.sample, np.array([nx, ny]), prev_xy=np.array([x, y])))
        return st

    def _sdc_collision(self) -> bool:
        st = self.state
        xy = st.xyz[0, st.t, :2]
        yaw = float(st.yaw[0, st.t])
        lw = (float(st.sample["length"][0]), float(st.sample["width"][0]))
        for i in range(1, st.xyz.shape[0]):
            if not st.valid[i, st.t]:
                continue
            lw2 = (float(st.sample["length"][i]), float(st.sample["width"][i]))
            if lw2[0] <= 0:
                lw2 = (1.0, 1.0)
            if _boxes_overlap(xy, yaw, lw, st.xyz[i, st.t, :2], float(st.yaw[i, st.t]), lw2):
                return True
        return False
