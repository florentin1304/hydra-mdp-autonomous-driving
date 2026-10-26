"""Bird's-eye snapshot of map, log SDC, predicted modes, and executed path."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_bev(
    path: str | Path,
    sample: dict,
    executed_xy: np.ndarray | None = None,
    pred_xy_world: np.ndarray | None = None,
    title: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    map_xyz = sample["map_xyz"]
    map_valid = sample["map_valid"]
    map_type = sample["map_type"]
    for i in range(len(map_type)):
        pts = map_xyz[i, map_valid[i], :2]
        if len(pts) < 2:
            continue
        color = "#888" if map_type[i] >= 4 else "#ccc"
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=0.6, zorder=0)

    t0 = 10
    log = sample["xyz"][0, t0:, :2]
    logv = sample["valid"][0, t0:]
    ax.plot(log[logv, 0], log[logv, 1], color="black", linewidth=1.5, label="log SDC", zorder=3)
    ax.scatter(sample["xyz"][0, t0, 0], sample["xyz"][0, t0, 1], c="black", s=20, zorder=4)

    neigh = sample["xyz"][1:, t0, :2]
    nv = sample["valid"][1:, t0]
    ax.scatter(neigh[nv, 0], neigh[nv, 1], c="tab:blue", s=12, label="agents", zorder=2)

    if pred_xy_world is not None:
        pred = np.asarray(pred_xy_world)
        if pred.ndim == 2:
            pred = pred[None, ...]
        for k, traj in enumerate(pred[:6]):
            ax.plot(traj[:, 0], traj[:, 1], color="tab:orange", alpha=0.7, linewidth=1.0, zorder=3)

    if executed_xy is not None and len(executed_xy):
        ex = np.asarray(executed_xy)
        ax.plot(ex[:, 0], ex[:, 1], color="tab:red", linewidth=2.0, label="executed", zorder=5)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
