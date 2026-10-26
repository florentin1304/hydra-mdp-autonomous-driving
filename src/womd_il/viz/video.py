"""Animated BEV videos for open-loop prediction and closed-loop rollout."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    return plt, patches, FuncAnimation, FFMpegWriter, PillowWriter


def _rotated_box(xy, yaw, length, width):
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = 0.5 * max(length, 1.0), 0.5 * max(width, 0.5)
    corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    rot = np.stack([c * corners[:, 0] - s * corners[:, 1], s * corners[:, 0] + c * corners[:, 1]], axis=-1)
    return rot + xy


def _draw_map(ax, sample, center, radius=45.0):
    map_xyz = sample["map_xyz"]
    map_valid = sample["map_valid"]
    map_type = sample["map_type"]
    for i in range(len(map_type)):
        pts = map_xyz[i, map_valid[i], :2]
        if len(pts) < 2:
            continue
        if np.linalg.norm(pts.mean(0) - center) > radius + 20:
            continue
        color = "#7a7a7a" if int(map_type[i]) >= 4 else "#d0d0d0"
        lw = 1.0 if int(map_type[i]) >= 4 else 0.6
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw, zorder=0)


def _plan_at(state, t: int) -> np.ndarray | None:
    if not getattr(state, "predicted_xy", None):
        return None
    chosen = None
    for pt, xy in zip(state.predicted_t, state.predicted_xy):
        if pt <= t:
            chosen = xy
    return chosen


def _modes_at(state, t: int) -> dict | None:
    modes = getattr(state, "predicted_modes", None) or []
    chosen = None
    for entry in modes:
        if int(entry.get("t", -1)) <= t:
            chosen = entry
    return chosen


def _draw_mode_fan(ax, modes: dict, *, cmap_name: str = "plasma") -> None:
    """Color-grade top-N by log p; highlight the selected pick in cyan."""
    from matplotlib import cm
    from matplotlib.colors import Normalize

    xy = np.asarray(modes["xy"])
    probs = np.asarray(modes["probs"], dtype=np.float64)
    pick = int(modes.get("pick", 0))
    if xy.ndim != 3 or len(probs) == 0:
        return
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    # Stretch contrast across the displayed set.
    vmin = float(np.percentile(logp, 5))
    vmax = float(logp.max())
    if vmax <= vmin:
        vmax = vmin + 1e-3
    norm = Normalize(vmin=vmin, vmax=vmax)
    try:
        cmap = cm.get_cmap(cmap_name)
    except AttributeError:
        cmap = cm.colormaps[cmap_name]
    order = np.argsort(probs)  # low first so high sits on top
    for j in order:
        if j == pick:
            continue
        traj = xy[int(j)]
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=cmap(norm(logp[j])),
            lw=0.7,
            alpha=0.45,
            zorder=2,
        )
    # Best / executed pick.
    best = xy[pick]
    ax.plot(
        best[:, 0],
        best[:, 1],
        color="cyan",
        lw=2.4,
        alpha=0.95,
        zorder=4,
        label=f"pick p={probs[pick]:.3f}",
    )


def _save_animation(fig, update, n_frames: int, path: Path, fps: int = 10) -> Path:
    plt, _, FuncAnimation, FFMpegWriter, PillowWriter = _setup_mpl()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    mp4 = path.with_suffix(".mp4")
    gif = path.with_suffix(".gif")
    written = gif
    try:
        anim.save(str(mp4), writer=FFMpegWriter(fps=fps, bitrate=1800))
        written = mp4
    except Exception:
        anim.save(str(gif), writer=PillowWriter(fps=fps))
        written = gif
    plt.close(fig)
    return written


def render_open_loop_video(
    path: str | Path,
    sample: dict,
    pred_xy_world: np.ndarray,
    scores: np.ndarray | None = None,
    fps: int = 10,
    title: str = "open loop",
    n_modes: int = 128,
) -> Path:
    """pred_xy_world: [K, T, 2] in world frame, aligned to the 8 s future."""
    plt, patches, *_ = _setup_mpl()
    pred = np.asarray(pred_xy_world)
    if pred.ndim == 2:
        pred = pred[None, ...]
    t0 = 10
    horizon = min(pred.shape[1], sample["xyz"].shape[1] - t0 - 1)
    if scores is None:
        probs = np.ones(len(pred), dtype=np.float64) / max(len(pred), 1)
    else:
        sc = np.asarray(scores, dtype=np.float64)
        # Already probabilities (e.g. top-k im_scores) vs raw logits/costs.
        if sc.min() >= 0.0 and sc.max() <= 1.0 + 1e-5:
            probs = sc / max(float(sc.sum()), 1e-12)
        else:
            sc = np.where(sc > -1e8, sc, -1e9)
            sc = sc - sc.max()
            probs = np.exp(sc)
            probs = probs / max(probs.sum(), 1e-12)
    order = np.argsort(-probs)
    keep = order[: min(int(n_modes), len(order))]
    modes = {
        "xy": pred[keep, :horizon],
        "probs": probs[keep],
        "pick": 0,  # highest-prob among keep
    }
    fig, ax = plt.subplots(figsize=(7, 7))

    def update(fi: int):
        ax.clear()
        t = t0 + fi
        sdc = sample["xyz"][0, t, :2]
        _draw_map(ax, sample, sdc)
        log = sample["xyz"][0, t0 : t0 + horizon, :2]
        ax.plot(log[:, 0], log[:, 1], color="black", lw=1.6, label="log SDC", zorder=5)
        _draw_mode_fan(ax, modes)
        for i in range(sample["xyz"].shape[0]):
            if not sample["valid"][i, t]:
                continue
            xy = sample["xyz"][i, t, :2]
            yaw = float(sample["yaw"][i, t])
            poly = _rotated_box(xy, yaw, float(sample["length"][i] or 4.5), float(sample["width"][i] or 2.0))
            color = "black" if i == 0 else "tab:blue"
            ax.add_patch(patches.Polygon(poly, closed=True, fc=color, ec="none", alpha=0.85, zorder=6))
        ax.scatter([sdc[0]], [sdc[1]], c="red", s=18, zorder=7)
        ax.set_aspect("equal")
        ax.set_xlim(sdc[0] - 40, sdc[0] + 40)
        ax.set_ylim(sdc[1] - 40, sdc[1] + 40)
        ax.set_title(
            f"{title}  t={fi * 0.1:.1f}s  top-{len(keep)}  "
            f"p1={probs[keep[0]]:.3f}"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    return _save_animation(fig, update, horizon, Path(path), fps=fps)


def render_closed_loop_video(
    path: str | Path,
    sample: dict,
    state,
    fps: int = 10,
    title: str = "closed loop",
) -> Path:
    plt, patches, *_ = _setup_mpl()
    t0 = 10
    executed = np.asarray(state.executed_xy)
    n = min(len(executed), state.xyz.shape[1] - t0)
    fig, ax = plt.subplots(figsize=(7, 7))

    def update(fi: int):
        ax.clear()
        t = t0 + fi
        sdc = state.xyz[0, t, :2]
        _draw_map(ax, sample, sdc)
        log = sample["xyz"][0, t0 : t0 + n, :2]
        ax.plot(log[:, 0], log[:, 1], color="black", lw=1.4, label="log SDC", zorder=5)
        ax.plot(executed[: fi + 1, 0], executed[: fi + 1, 1], color="tab:red", lw=2.0, label="executed", zorder=6)
        modes = _modes_at(state, t)
        p_txt = ""
        if modes is not None:
            _draw_mode_fan(ax, modes)
            p_txt = f"  top-{len(modes['probs'])} p1={float(np.max(modes['probs'])):.3f}"
        else:
            plan = _plan_at(state, t)
            if plan is not None:
                ax.plot(plan[:, 0], plan[:, 1], color="tab:orange", lw=1.6, alpha=0.8, label="replan", zorder=3)
        for i in range(state.xyz.shape[0]):
            if not state.valid[i, t]:
                continue
            xy = state.xyz[i, t, :2]
            yaw = float(state.yaw[i, t])
            length = float(sample["length"][i] or 4.5)
            width = float(sample["width"][i] or 2.0)
            poly = _rotated_box(xy, yaw, length, width)
            color = "tab:red" if i == 0 else "tab:blue"
            ax.add_patch(patches.Polygon(poly, closed=True, fc=color, ec="none", alpha=0.85, zorder=7))
        ax.set_aspect("equal")
        ax.set_xlim(sdc[0] - 40, sdc[0] + 40)
        ax.set_ylim(sdc[1] - 40, sdc[1] + 40)
        hit = state.collisions[fi - 1] if fi > 0 and fi - 1 < len(state.collisions) else False
        ax.set_title(f"{title}  t={fi * 0.1:.1f}s{p_txt}" + ("  COLLISION" if hit else ""))
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    return _save_animation(fig, update, n, Path(path), fps=fps)
