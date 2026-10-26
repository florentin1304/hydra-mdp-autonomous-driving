"""Invertible bicycle model matching Waymax's (accel, curvature) action."""

from __future__ import annotations

import numpy as np

from womd_il import constants as C
from womd_il.data.transforms import wrap_angle


def bicycle_step(
    x: float,
    y: float,
    yaw: float,
    speed: float,
    accel: float,
    kappa: float,
    dt: float = C.DT,
    accel_clip: float = C.ACCEL_CLIP,
    kappa_clip: float = C.KAPPA_CLIP,
    max_speed: float = C.MAX_SPEED,
) -> tuple[float, float, float, float]:
    a = float(np.clip(accel, -accel_clip, accel_clip))
    k = float(np.clip(kappa, -kappa_clip, kappa_clip))
    new_speed = float(np.clip(speed + a * dt, 0.0, max_speed))
    dist = speed * dt + 0.5 * a * dt * dt
    if dist < 0.0:
        dist = 0.0
    new_yaw = float(wrap_angle(yaw + k * dist))
    new_x = x + dist * np.cos(yaw)
    new_y = y + dist * np.sin(yaw)
    return float(new_x), float(new_y), new_yaw, new_speed


def bicycle_inverse(
    x: float,
    y: float,
    yaw: float,
    speed: float,
    next_x: float,
    next_y: float,
    next_yaw: float | None = None,
    dt: float = C.DT,
    accel_clip: float = C.ACCEL_CLIP,
    kappa_clip: float = C.KAPPA_CLIP,
) -> tuple[float, float]:
    dx = next_x - x
    dy = next_y - y
    dist = float(np.hypot(dx, dy))
    new_speed = dist / max(dt, 1e-6)
    accel = (new_speed - speed) / max(dt, 1e-6)
    accel = float(np.clip(accel, -accel_clip, accel_clip))
    if next_yaw is None:
        next_yaw = float(np.arctan2(dy, dx)) if dist > 1e-6 else yaw
    yaw_diff = float(wrap_angle(next_yaw - yaw))
    denom = speed * dt + 0.5 * accel * dt * dt
    if abs(denom) < 1e-4:
        kappa = 0.0
    else:
        kappa = yaw_diff / denom
    kappa = float(np.clip(kappa, -kappa_clip, kappa_clip))
    return accel, kappa


def traj_to_actions(
    xy: np.ndarray,
    yaw_seq: np.ndarray | None,
    x0: float,
    y0: float,
    yaw0: float,
    speed0: float,
    n: int = 1,
    dt: float = C.DT,
    tracker: str = "lqr",
    lookahead: int = 10,
) -> list[tuple[float, float]]:
    """Convert the first `n` waypoints into bicycle (accel, kappa) actions.

    `xy` is [T, 2] in the **same frame as (x0,y0)** (typically world).

    `tracker='lqr'` is the NAVSIM / nuPlan protocol: track the plan's speed
    profile and path, not the Euclidean catch-up to the time-indexed waypoint.
    Inverse-bicycle catch-up clips accel whenever the plan is faster than the
    current speed (one 0.1 s step cannot close a several-metre gap).
    """
    if str(tracker).lower() in ("inverse", "catchup", "waypoint"):
        return _inverse_traj_to_actions(xy, yaw_seq, x0, y0, yaw0, speed0, n, dt)
    return _lqr_traj_to_actions(
        xy, yaw_seq, x0, y0, yaw0, speed0, n, dt, lookahead=lookahead
    )


def _inverse_traj_to_actions(
    xy: np.ndarray,
    yaw_seq: np.ndarray | None,
    x0: float,
    y0: float,
    yaw0: float,
    speed0: float,
    n: int,
    dt: float,
) -> list[tuple[float, float]]:
    actions = []
    x, y, yaw, speed = x0, y0, yaw0, speed0
    n = min(int(n), len(xy))
    for i in range(n):
        nx, ny = float(xy[i, 0]), float(xy[i, 1])
        nyaw = float(yaw_seq[i]) if yaw_seq is not None else None
        a, k = bicycle_inverse(x, y, yaw, speed, nx, ny, nyaw, dt=dt)
        actions.append((a, k))
        x, y, yaw, speed = bicycle_step(x, y, yaw, speed, a, k, dt=dt)
    return actions


def _plan_speed_and_kappa(
    xy: np.ndarray,
    yaw_seq: np.ndarray | None,
    x0: float,
    y0: float,
    yaw0: float,
    n: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step plan speed and curvature from origin + n waypoints."""
    n = min(int(n), len(xy))
    ref_xy = np.vstack([np.array([x0, y0], dtype=np.float64), np.asarray(xy[:n], dtype=np.float64)])
    if yaw_seq is None:
        dxy = np.diff(ref_xy, axis=0)
        ref_yaw = np.arctan2(dxy[:, 1], dxy[:, 0])
        ref_yaw = np.concatenate([[yaw0], ref_yaw])
        for i in range(1, len(ref_yaw)):
            if np.hypot(dxy[i - 1, 0], dxy[i - 1, 1]) < 1e-6:
                ref_yaw[i] = ref_yaw[i - 1]
    else:
        ref_yaw = np.concatenate(
            [[float(yaw0)], np.asarray(yaw_seq[:n], dtype=np.float64)]
        )
    ds = np.linalg.norm(np.diff(ref_xy, axis=0), axis=1)
    v_plan = ds / max(dt, 1e-6)
    d_yaw = wrap_angle(np.diff(ref_yaw))
    kappa_plan = np.divide(d_yaw, ds, out=np.zeros_like(d_yaw), where=ds > 1e-4)
    return v_plan, kappa_plan, ref_xy[1:]


def _lqr_traj_to_actions(
    xy: np.ndarray,
    yaw_seq: np.ndarray | None,
    x0: float,
    y0: float,
    yaw0: float,
    speed0: float,
    n: int,
    dt: float,
    lookahead: int = 10,
) -> list[tuple[float, float]]:
    """Decoupled longitudinal / lateral tracker (nuPlan LQR, kappa bicycle).

    Longitudinal: one-step LQR on velocity with a 1 s lookahead, Q=R=1.
    Lateral: plan curvature feedforward plus P on cross-track / heading.
    Stopping uses the nuPlan P-controller when both speeds are near zero.
    """
    n = min(int(n), len(xy))
    if n <= 0:
        return []
    xy = np.asarray(xy, dtype=np.float64)
    v_plan, kappa_plan, ref_xy = _plan_speed_and_kappa(xy, yaw_seq, x0, y0, yaw0, n, dt)
    if yaw_seq is None:
        dxy = ref_xy - np.vstack([np.array([x0, y0], dtype=np.float64), ref_xy[:-1]])
        ref_yaw = np.arctan2(dxy[:, 1], dxy[:, 0])
        for i in range(n):
            if np.hypot(dxy[i, 0], dxy[i, 1]) < 1e-6:
                ref_yaw[i] = yaw0 if i == 0 else ref_yaw[i - 1]
    else:
        ref_yaw = np.asarray(yaw_seq[:n], dtype=np.float64)

    n_look = max(int(lookahead), 1)
    b_long = n_look * dt
    long_gain = b_long / (b_long * b_long + 1.0)
    actions: list[tuple[float, float]] = []
    x, y, yaw, speed = float(x0), float(y0), float(yaw0), float(speed0)
    for i in range(n):
        look = min(i + n_look - 1, n - 1)
        v_ref = float(v_plan[look])
        if v_ref <= 0.5 and speed <= 0.5:
            accel = float(np.clip(-1.5 * speed, -C.ACCEL_CLIP, C.ACCEL_CLIP))
            kappa = 0.0
        else:
            accel = float(np.clip(long_gain * (v_ref - speed), -C.ACCEL_CLIP, C.ACCEL_CLIP))
            rx, ry = float(ref_xy[i, 0]), float(ref_xy[i, 1])
            heading_ref = float(ref_yaw[i])
            lat = -(x - rx) * np.sin(heading_ref) + (y - ry) * np.cos(heading_ref)
            head_err = wrap_angle(yaw - heading_ref)
            s = max(speed, 1.0) * max(n_look * dt, dt)
            kappa = float(kappa_plan[i]) - head_err / s - 2.0 * lat / (s * s)
            kappa = float(np.clip(kappa, -C.KAPPA_CLIP, C.KAPPA_CLIP))
        actions.append((accel, kappa))
        x, y, yaw, speed = bicycle_step(x, y, yaw, speed, accel, kappa, dt=dt)
    return actions
