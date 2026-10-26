"""Closed-loop collision / off-road / progress / log-ADE metrics."""

from __future__ import annotations

import numpy as np

from womd_il.data.transforms import wrap_angle


def closed_loop_metrics(state, sample: dict, dt: float = 0.1) -> dict[str, float]:
    exec_xy = np.asarray(state.executed_xy, dtype=np.float32)
    t0 = 10
    n = len(exec_xy)
    log_xy = sample["xyz"][0, t0 : t0 + n, :2]
    valid = sample["valid"][0, t0 : t0 + n]
    m = min(len(log_xy), n)
    err = np.linalg.norm(exec_xy[:m] - log_xy[:m], axis=-1)
    err = np.where(valid[:m], err, np.nan)
    log_ade = float(np.nanmean(err)) if np.any(np.isfinite(err)) else 0.0
    log_fde = float(err[np.isfinite(err)][-1]) if np.any(np.isfinite(err)) else 0.0
    collisions = [bool(x) for x in getattr(state, "collisions", [])]
    offroad = [bool(x) for x in getattr(state, "offroad", [])]
    progress = 0.0
    if n >= 2:
        progress = float(np.linalg.norm(exec_xy[-1] - exec_xy[0]))
    log_progress = 0.0
    if m >= 2:
        log_progress = float(np.linalg.norm(log_xy[m - 1] - log_xy[0]))

    # Comfort: peak |accel| and |yaw rate| along the executed path.
    peak_a = 0.0
    peak_yawrate = 0.0
    yaws = np.asarray(getattr(state, "executed_yaw", []), dtype=np.float32)
    if n >= 3:
        speed = np.linalg.norm(np.diff(exec_xy, axis=0), axis=-1) / dt
        acc = np.diff(speed) / dt
        peak_a = float(np.max(np.abs(acc))) if len(acc) else 0.0
        if len(yaws) >= 2:
            peak_yawrate = float(np.max(np.abs(wrap_angle(np.diff(yaws)))) / dt)

    # Off-route: distance from executed points to the preferred route polyline.
    off_route = 0.0
    route = sample["route_xyz"][0, :, :2]
    rv = sample["route_valid"][0]
    rpts = route[rv]
    if len(rpts) and n:
        d = np.linalg.norm(exec_xy[:, None, :] - rpts[None, :, :], axis=-1).min(axis=-1)
        off_route = float(np.mean(d > 5.0))

    kin_infeas = float(peak_a > 6.0 or peak_yawrate > 1.5)
    return {
        "collision": float(any(collisions)),
        "collision_rate": float(np.mean(collisions)) if collisions else 0.0,
        "offroad": float(any(offroad)),
        "offroad_rate": float(np.mean(offroad)) if offroad else 0.0,
        "off_route": off_route,
        "progress": progress,
        "log_progress": log_progress,
        "progress_ratio": progress / max(log_progress, 1e-3),
        "log_ade": log_ade,
        "log_fde": log_fde,
        "peak_accel": peak_a,
        "peak_yawrate": peak_yawrate,
        "kinematic_infeasible": kin_infeas,
        "steps": float(n),
    }
