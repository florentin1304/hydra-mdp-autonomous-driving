#!/usr/bin/env python3
"""BEV plots of synthetic/official routes with green on-route lane overlay."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from womd_il import constants as C
from womd_il.data.features import _map_on_route_flag
from womd_il.data.parse_scenario import parse_tfrecord_bytes
from womd_il.data.schema import PackConfig
from womd_il.data.tfrecord import iter_tfrecord_file, list_tfrecord_shards
from womd_il.data.transforms import agent_centric_xy


def _preferred_route(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    rv = np.asarray(sample["route_valid"], dtype=bool)
    counts = rv.sum(axis=-1)
    on = np.asarray(sample.get("route_on_route", np.zeros(counts.shape[0], bool)), dtype=bool)
    usable = on & (counts >= 2)
    ridx = int(np.argmax(usable)) if usable.any() else int(np.argmax(counts))
    return sample["route_xyz"][ridx, :, :2], rv[ridx]


def _lane_ribbon(pts: np.ndarray, half_w: float = 1.85) -> np.ndarray | None:
    """Approximate a lane polygon from its centerline (nuPlan-style fill)."""
    if len(pts) < 2:
        return None
    d = np.diff(pts, axis=0)
    d = np.concatenate([d, d[-1:]], axis=0)
    nrm = np.linalg.norm(d, axis=-1, keepdims=True)
    d = d / np.clip(nrm, 1e-3, None)
    left = pts + half_w * np.stack([-d[:, 1], d[:, 0]], axis=-1)
    right = pts - half_w * np.stack([-d[:, 1], d[:, 0]], axis=-1)
    return np.concatenate([left, right[::-1]], axis=0)


def plot_scene(sample: dict, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    t0 = int(C.CURRENT_STEP)
    origin = sample["xyz"][0, t0, :2].astype(np.float32)
    yaw = float(sample["yaw"][0, t0])

    map_xy = agent_centric_xy(sample["map_xyz"][..., :2], origin, yaw)
    map_valid = np.asarray(sample["map_valid"], dtype=bool)
    map_type = np.asarray(sample["map_type"])
    route_xy_w, route_mask = _preferred_route(sample)
    route_xy = agent_centric_xy(route_xy_w, origin, yaw)

    on_route = _map_on_route_flag(
        map_xy, map_valid, map_type, route_xy, route_mask
    )[:, 0].astype(bool)

    fig, ax = plt.subplots(figsize=(7, 7))
    # Base map
    for i in range(len(map_type)):
        pts = map_xy[i, map_valid[i]]
        if len(pts) < 2:
            continue
        if int(map_type[i]) in (C.MAP_LANE_FREEWAY, C.MAP_LANE_SURFACE, C.MAP_LANE_BIKE):
            ax.plot(pts[:, 0], pts[:, 1], color="#bbbbbb", lw=0.8, zorder=1)
        elif int(map_type[i]) == C.MAP_ROAD_EDGE:
            ax.plot(pts[:, 0], pts[:, 1], color="#666666", lw=1.0, zorder=1)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color="#dddddd", lw=0.5, zorder=0)

    # Green filled corridor: on-route lane ribbons (nuPlan roadblock look)
    first_green = True
    for i in np.where(on_route)[0]:
        pts = map_xy[i, map_valid[i]]
        poly = _lane_ribbon(pts)
        if poly is None:
            continue
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor="#2ca02c",
                edgecolor="#1a7a1a",
                linewidth=0.4,
                alpha=0.45,
                zorder=3,
                label="on-route lanes" if first_green else None,
            )
        )
        first_green = False

    # Alt routes (off-route path slots) — under preferred
    rv = np.asarray(sample["route_valid"], dtype=bool)
    on = np.asarray(sample.get("route_on_route", np.zeros(rv.shape[0], bool)))
    shown_alt = False
    for i in range(rv.shape[0]):
        if on[i] or int(rv[i].sum()) < 2:
            continue
        alt = agent_centric_xy(sample["route_xyz"][i, :, :2], origin, yaw)
        pts = alt[rv[i]]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="#ff7f0e",
            lw=1.2,
            alpha=0.7,
            zorder=2,
            label="alt routes" if not shown_alt else None,
        )
        shown_alt = True

    # Log SDC future (under preferred so preferred stays visible)
    log = agent_centric_xy(sample["xyz"][0, :, :2], origin, yaw)
    logv = np.asarray(sample["valid"][0], dtype=bool)
    fut = logv.copy()
    fut[:t0] = False
    ax.plot(log[fut, 0], log[fut, 1], color="black", lw=2.2, zorder=4, label="log SDC future")

    # Preferred route on top — magenta dashes (often coincides with log)
    rpts = route_xy[route_mask]
    if len(rpts) >= 2:
        ax.plot(
            rpts[:, 0],
            rpts[:, 1],
            color="#c51b8a",
            lw=2.4,
            ls="--",
            zorder=5,
            label="preferred route",
        )
        ax.scatter(
            rpts[:: max(1, len(rpts) // 12), 0],
            rpts[:: max(1, len(rpts) // 12), 1],
            c="#c51b8a",
            s=18,
            zorder=6,
            marker="o",
        )
    ax.scatter([0.0], [0.0], c="red", s=36, zorder=7, label="ego now")

    # Neighbors at t0
    nxy = agent_centric_xy(sample["xyz"][1:, t0, :2], origin, yaw)
    nv = np.asarray(sample["valid"][1:, t0], dtype=bool)
    ax.scatter(nxy[nv, 0], nxy[nv, 1], c="tab:blue", s=10, zorder=4, label="agents")

    ax.set_aspect("equal")
    ax.set_xlim(-40, 80)
    ax.set_ylim(-60, 60)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_xlabel("ego-x (m)")
    ax.set_ylabel("ego-y (m)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=Path("data/womd/raw/validation"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/route_overlay"))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = PackConfig()
    shards = list_tfrecord_shards(args.raw_dir)
    rng = np.random.default_rng(args.seed)

    # Prefer scenes where preferred route matches lane centerlines (green overlay).
    candidates: list[tuple[float, int, dict, str]] = []
    scanned = 0
    for sh in shards[:6]:
        for b in iter_tfrecord_file(sh):
            if scanned >= 400:
                break
            sample = parse_tfrecord_bytes(b, cfg)
            scanned += 1
            route_xy, mask = _preferred_route(sample)
            if int(mask.sum()) < 2:
                continue
            t0 = int(C.CURRENT_STEP)
            origin = sample["xyz"][0, t0, :2]
            yaw = float(sample["yaw"][0, t0])
            map_xy = agent_centric_xy(sample["map_xyz"][..., :2], origin, yaw)
            route_local = agent_centric_xy(route_xy, origin, yaw)
            n_on = int(
                _map_on_route_flag(
                    map_xy, sample["map_valid"], sample["map_type"], route_local, mask
                )[:, 0].sum()
            )
            if n_on < 1:
                continue
            pts = route_local[mask]
            length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            lat = float(np.max(np.abs(pts[:, 1]))) if len(pts) else 0.0
            score = lat * 2.0 + min(length, 120.0) * 0.05 + float(n_on)
            candidates.append((score, n_on, sample, str(sample.get("scenario_id", scanned))))
        if scanned >= 400:
            break

    candidates.sort(key=lambda x: -x[0])
    top = candidates[: max(args.n // 2, 1)]
    rest = candidates[max(args.n // 2, 1) :]
    if rest:
        pick = rng.choice(len(rest), size=min(args.n - len(top), len(rest)), replace=False)
        chosen = top + [rest[i] for i in pick]
    else:
        chosen = top[: args.n]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for j, (score, n_on, sample, sid) in enumerate(chosen[: args.n]):
        title = f"{j:02d} id={sid[:12]} score={score:.1f} on_route_lanes={n_on}"
        out = args.out_dir / f"{j:02d}_{sid[:16]}_route.png"
        plot_scene(sample, out, title)
        print(f"wrote {out}  ({title})")


if __name__ == "__main__":
    main()
