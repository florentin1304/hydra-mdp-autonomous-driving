"""Closed-loop episode capture and metric-filtered trajectory drawing.

At every replan the full vocabulary is scored two ways:

* ground truth — rule labels (nc, dac, …) plus log-ADE imitation and rule EPDMS
* predicted — imitation / EPO softmax and, when the checkpoint has metric heads,
  per-metric probabilities

The notebook draws the trajectories that clear the checked scores.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from womd_il import constants as C
from womd_il.data.dataset import WOMDPackedDataset
from womd_il.data.schema import PackConfig
from womd_il.losses.hydra import METRIC_INDEX, METRIC_NAMES, hydra_metric_labels, pdms_from_probs
from womd_il.sim.rollout import rollout_policy
from womd_il.utils import load_policy_from_ckpt
from womd_il.viz.video import _draw_map, _rotated_box

# Channels the inspector can filter on. Imitation and EPO sit beside the rule metrics.
CHANNELS = ("imitation", "epo", *METRIC_NAMES)
_CACHE_VERSION = 1
_TL_COLOR = {
    C.TL_ARROW_STOP: "#d62728",
    C.TL_STOP: "#d62728",
    C.TL_FLASHING_STOP: "#d62728",
    C.TL_ARROW_CAUTION: "#ff7f0e",
    C.TL_CAUTION: "#ff7f0e",
    C.TL_FLASHING_CAUTION: "#ff7f0e",
    C.TL_ARROW_GO: "#2ca02c",
    C.TL_GO: "#2ca02c",
}


@dataclass
class ClosedLoopEpisode:
    """One rollout plus per-replan scores over the full vocabulary."""

    scenario_index: int
    scenario_id: str
    ckpt: str
    h5: str
    mode_select: str
    t0: int
    final_t: int
    vocab_xy: np.ndarray  # [K, T, 2] agent frame
    origins: np.ndarray  # [S, 3] world xy + yaw at each replan
    sim_t: np.ndarray  # [S] simulator time of each replan
    pick: np.ndarray  # [S] vocab index actually executed
    pred_im: np.ndarray  # [S, K] softmax
    pred_epo: np.ndarray  # [S, K] softmax
    pred_metrics: np.ndarray | None  # [S, K, M] probabilities, or None
    gt_metrics: np.ndarray  # [S, K, M]
    gt_ade: np.ndarray  # [S, K] meters vs log future
    gt_pdms: np.ndarray  # [S, K] rule EPDMS
    ego_xy: np.ndarray  # [time, 2] executed ego (log before t0)
    ego_yaw: np.ndarray  # [time]
    collisions: np.ndarray  # [steps] after each bicycle step
    offroad: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.final_t - self.t0 + 1)

    @property
    def k(self) -> int:
        return int(self.vocab_xy.shape[0])


def episode_summary(ep: ClosedLoopEpisode) -> str:
    seconds = (ep.final_t - ep.t0) * float(C.DT)
    collided = bool(np.asarray(ep.collisions).any()) if len(ep.collisions) else False
    off = bool(np.asarray(ep.offroad).any()) if len(ep.offroad) else False
    metrics = "yes" if ep.pred_metrics is not None else "no"
    return (
        f"{ep.scenario_id}  idx={ep.scenario_index}  "
        f"{ep.n_frames} frames ({seconds:.1f}s)  replans={len(ep.sim_t)}  "
        f"collision={collided}  offroad={off}  K={ep.k}  "
        f"predicted metrics={metrics}  select={ep.mode_select}"
    )


def _peak_norm(values: np.ndarray) -> np.ndarray:
    """Scale a distribution so the best mode is 1. Worst stays 0 when the peak is 0."""
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    peak = float(v.max()) if v.size else 0.0
    if peak <= 1e-12:
        return np.zeros_like(v)
    return v / peak


def channel_scores(
    ep: ClosedLoopEpisode,
    replan_i: int,
    source: str,
    ade_scale: float,
) -> dict[str, np.ndarray]:
    """Per-vocab scores in ``[0, 1]`` at one replan.

    Ground-truth imitation is ``exp(-ADE / ade_scale)`` against the log future
    in the executed ego frame. Predicted imitation and EPO are that head's
    softmax divided by its peak, so the threshold means "fraction of the best
    mode". Metric channels are the rule label or the predicted probability.
    Ground-truth EPO is rule EPDMS.
    """
    i = int(replan_i)
    out: dict[str, np.ndarray] = {}
    if source == "gt":
        labels = np.asarray(ep.gt_metrics[i], dtype=np.float32)
        for name, mi in METRIC_INDEX.items():
            out[name] = labels[:, mi]
        ade = np.asarray(ep.gt_ade[i], dtype=np.float32)
        scale = max(float(ade_scale), 1e-3)
        out["imitation"] = np.exp(-ade / scale).astype(np.float32)
        out["epo"] = np.asarray(ep.gt_pdms[i], dtype=np.float32)
        return out
    if source != "pred":
        raise ValueError(f"source must be 'gt' or 'pred', got {source!r}")
    out["imitation"] = _peak_norm(ep.pred_im[i])
    out["epo"] = _peak_norm(ep.pred_epo[i])
    if ep.pred_metrics is not None:
        labels = np.asarray(ep.pred_metrics[i], dtype=np.float32)
        for name, mi in METRIC_INDEX.items():
            out[name] = labels[:, mi]
    return out


def passing_indices(
    scores: dict[str, np.ndarray],
    selected: Sequence[str],
    threshold: float,
    max_draw: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Indices that clear every selected channel, plus a color score and the full count.

    When more than ``max_draw`` pass and the scores are tied, keep a stride
    through the vocabulary so the fan shape stays visible. Otherwise keep the
    highest color scores (minimum of the checked channels).
    """
    selected = [name for name in selected if name in scores]
    if not selected:
        return np.zeros(0, np.int64), np.zeros(0, np.float32), 0
    k = int(np.asarray(scores[selected[0]]).shape[0])
    mask = np.ones(k, dtype=bool)
    color = np.ones(k, dtype=np.float32)
    thr = float(threshold)
    for name in selected:
        s = np.asarray(scores[name], dtype=np.float32).reshape(-1)
        mask &= s >= thr
        color = np.minimum(color, s)
    idx = np.flatnonzero(mask)
    total = int(idx.size)
    if total == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32), 0
    cap = max(int(max_draw), 1)
    if total > cap:
        sc = color[idx]
        order = np.argsort(-sc, kind="stable")
        if float(sc[order[0]] - sc[order[cap - 1]]) <= 1e-3:
            stride = int(np.ceil(total / cap))
            take = np.arange(0, total, stride)[:cap]
        else:
            take = order[:cap]
        idx = idx[take]
        sc = sc[take]
    else:
        sc = color[idx]
    return idx.astype(np.int64), sc.astype(np.float32), total


def frame_sim_t(ep: ClosedLoopEpisode, frame: int) -> int:
    frame = int(np.clip(frame, 0, max(ep.n_frames - 1, 0)))
    return int(ep.t0 + frame)


def replan_index(ep: ClosedLoopEpisode, sim_t: int) -> int:
    """Latest replan at or before ``sim_t``."""
    if len(ep.sim_t) == 0:
        raise RuntimeError("episode has no replans")
    i = int(np.searchsorted(ep.sim_t, int(sim_t), side="right") - 1)
    return int(np.clip(i, 0, len(ep.sim_t) - 1))


def _meta_dict(
    ckpt: Path,
    h5: Path,
    scenario_index: int,
    horizon: int,
    h_exec: int,
    mode_select: str,
    neighbors: str,
    early_stop: bool,
) -> dict:
    return {
        "version": _CACHE_VERSION,
        "ckpt": str(ckpt.resolve()),
        "ckpt_mtime_ns": ckpt.stat().st_mtime_ns,
        "h5": str(h5.resolve()),
        "scenario_index": int(scenario_index),
        "horizon": int(horizon),
        "h_exec": int(h_exec),
        "mode_select": str(mode_select),
        "neighbors": str(neighbors),
        "early_stop": bool(early_stop),
    }


def _save_episode(ep: ClosedLoopEpisode, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = ep.pred_metrics if ep.pred_metrics is not None else np.zeros((0,), np.float16)
    np.savez_compressed(
        path,
        meta=np.array(json.dumps(meta)),
        scenario_index=np.int32(ep.scenario_index),
        scenario_id=np.array(ep.scenario_id),
        ckpt=np.array(ep.ckpt),
        h5=np.array(ep.h5),
        mode_select=np.array(ep.mode_select),
        t0=np.int32(ep.t0),
        final_t=np.int32(ep.final_t),
        vocab_xy=ep.vocab_xy,
        origins=ep.origins,
        sim_t=ep.sim_t,
        pick=ep.pick,
        pred_im=ep.pred_im,
        pred_epo=ep.pred_epo,
        pred_metrics=pred,
        has_pred_metrics=np.bool_(ep.pred_metrics is not None),
        gt_metrics=ep.gt_metrics,
        gt_ade=ep.gt_ade,
        gt_pdms=ep.gt_pdms,
        ego_xy=ep.ego_xy,
        ego_yaw=ep.ego_yaw,
        collisions=np.asarray(ep.collisions, dtype=bool),
        offroad=np.asarray(ep.offroad, dtype=bool),
    )


def _load_episode(path: Path) -> tuple[ClosedLoopEpisode, dict]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        has = bool(z["has_pred_metrics"])
        pred = np.asarray(z["pred_metrics"]) if has else None
        ep = ClosedLoopEpisode(
            scenario_index=int(z["scenario_index"]),
            scenario_id=str(z["scenario_id"]),
            ckpt=str(z["ckpt"]),
            h5=str(z["h5"]),
            mode_select=str(z["mode_select"]),
            t0=int(z["t0"]),
            final_t=int(z["final_t"]),
            vocab_xy=np.asarray(z["vocab_xy"]),
            origins=np.asarray(z["origins"]),
            sim_t=np.asarray(z["sim_t"]),
            pick=np.asarray(z["pick"]),
            pred_im=np.asarray(z["pred_im"]),
            pred_epo=np.asarray(z["pred_epo"]),
            pred_metrics=pred,
            gt_metrics=np.asarray(z["gt_metrics"]),
            gt_ade=np.asarray(z["gt_ade"]),
            gt_pdms=np.asarray(z["gt_pdms"]),
            ego_xy=np.asarray(z["ego_xy"]),
            ego_yaw=np.asarray(z["ego_yaw"]),
            collisions=np.asarray(z["collisions"]),
            offroad=np.asarray(z["offroad"]),
        )
    return ep, meta


def _as_vec(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().reshape(-1).cpu().numpy()


def capture_episode(
    ckpt: str | Path,
    h5: str | Path,
    scenario_index: int = 540,
    *,
    device: str = "cuda",
    horizon: int = 80,
    h_exec: int = 1,
    mode_select: str = "geomean",
    neighbors: str = "log",
    early_stop_on_fail: bool = True,
    cache_path: str | Path | None = None,
    reuse_cache: bool = True,
) -> tuple[ClosedLoopEpisode, dict]:
    """Roll a packed scenario and score every vocab trajectory at each replan.

    Returns the episode and the world sample (map, log agents, traffic lights).
    A matching ``cache_path`` skips the rollout.
    """
    ckpt = Path(ckpt)
    h5 = Path(h5)
    meta = _meta_dict(
        ckpt,
        h5,
        scenario_index,
        horizon,
        h_exec,
        mode_select,
        neighbors,
        early_stop_on_fail,
    )
    cache = Path(cache_path) if cache_path else None
    if reuse_cache and cache is not None and cache.exists():
        ep, saved = _load_episode(cache)
        if saved == meta:
            ds = WOMDPackedDataset(h5, target_mode="sdc", cfg=PackConfig())
            try:
                sample = ds.get_world_sample(int(scenario_index))
            finally:
                ds.close()
            return ep, sample

    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    policy, _cfg = load_policy_from_ckpt(ckpt, device=dev)
    policy.eval()
    ds = WOMDPackedDataset(h5, target_mode="sdc", cfg=PackConfig())
    try:
        sample = ds.get_world_sample(int(scenario_index))
    finally:
        ds.close()

    vocab_xy: list[np.ndarray] = []
    origins: list[np.ndarray] = []
    sim_ts: list[int] = []
    picks: list[int] = []
    pred_im: list[np.ndarray] = []
    pred_epo: list[np.ndarray] = []
    pred_metrics: list[np.ndarray] = []
    gt_metrics: list[np.ndarray] = []
    gt_ade: list[np.ndarray] = []
    gt_pdms: list[np.ndarray] = []
    has_metrics = True

    pbar = tqdm(total=int(horizon), desc=f"score i{scenario_index}", leave=False)

    def on_replan(*, sim, bundle, batch, pick, origin) -> None:
        nonlocal has_metrics
        extras = bundle.extras or {}
        if not vocab_xy:
            vocab_xy.append(bundle.xyh[0, :, :, :2].detach().float().cpu().numpy())
        pred_im.append(_as_vec(extras["im_scores"]).astype(np.float16))
        pred_epo.append(_as_vec(extras["epo_scores"]).astype(np.float16))
        mlog = extras.get("metric_logits")
        if mlog is None:
            has_metrics = False
        else:
            prob = torch.sigmoid(mlog[0]).detach().float().cpu().numpy().astype(np.float16)
            pred_metrics.append(prob)
        yaw = bundle.xyh[..., 2] if bundle.xyh.size(-1) >= 3 else None
        labels = hydra_metric_labels(bundle.xyh[..., :2], batch, pred_yaw=yaw, max_k=None)
        pdms = pdms_from_probs(labels)
        t = min(int(bundle.xyh.size(2)), int(batch["gt_xy"].size(1)))
        gt = batch["gt_xy"][:, :t]
        valid = batch["gt_valid"][:, :t].float()
        dist = torch.linalg.norm(gt[:, None] - bundle.xyh[:, :, :t, :2], dim=-1)
        ade = (dist * valid[:, None]).sum(-1) / valid.sum(-1, keepdim=True).clamp_min(1.0)
        gt_metrics.append(labels[0].detach().float().cpu().numpy().astype(np.float16))
        gt_pdms.append(pdms[0].detach().float().cpu().numpy().astype(np.float16))
        gt_ade.append(ade[0].detach().float().cpu().numpy().astype(np.float16))
        picks.append(int(pick))
        origins.append(np.asarray(origin, dtype=np.float32).reshape(3))
        sim_ts.append(int(sim.state.t))
        pbar.update(1)

    state = rollout_policy(
        policy,
        sample,
        neighbors=neighbors,
        horizon=int(horizon),
        device=dev,
        h_exec=int(h_exec),
        mode_select=mode_select,
        n_viz_modes=0,
        early_stop_on_fail=bool(early_stop_on_fail),
        on_replan=on_replan,
    )
    pbar.close()
    if not sim_ts:
        raise RuntimeError(f"scenario {scenario_index} produced no replans")

    ep = ClosedLoopEpisode(
        scenario_index=int(scenario_index),
        scenario_id=str(sample.get("scenario_id", scenario_index)),
        ckpt=str(ckpt.resolve()),
        h5=str(h5.resolve()),
        mode_select=str(mode_select),
        t0=int(C.CURRENT_STEP),
        final_t=int(state.t),
        vocab_xy=vocab_xy[0].astype(np.float32),
        origins=np.stack(origins).astype(np.float32),
        sim_t=np.asarray(sim_ts, dtype=np.int32),
        pick=np.asarray(picks, dtype=np.int32),
        pred_im=np.stack(pred_im),
        pred_epo=np.stack(pred_epo),
        pred_metrics=np.stack(pred_metrics) if has_metrics and pred_metrics else None,
        gt_metrics=np.stack(gt_metrics),
        gt_ade=np.stack(gt_ade),
        gt_pdms=np.stack(gt_pdms),
        ego_xy=np.asarray(state.xyz[0, :, :2], dtype=np.float32),
        ego_yaw=np.asarray(state.yaw[0], dtype=np.float32),
        collisions=np.asarray(state.collisions, dtype=bool),
        offroad=np.asarray(state.offroad, dtype=bool),
    )
    if cache is not None:
        _save_episode(ep, cache, meta)
    return ep, sample


def _to_world(xy: np.ndarray, origin: np.ndarray) -> np.ndarray:
    yaw = float(origin[2])
    c, s = np.cos(yaw), np.sin(yaw)
    x = xy[..., 0] * c - xy[..., 1] * s + float(origin[0])
    y = xy[..., 0] * s + xy[..., 1] * c + float(origin[1])
    return np.stack([x, y], axis=-1).astype(np.float32)


def _draw_route(ax, sample: dict, center: np.ndarray, radius: float) -> None:
    route = sample.get("route_xyz")
    if route is None:
        return
    valid = sample["route_valid"]
    on = sample.get("route_on_route")
    for i in range(len(route)):
        if on is not None and not bool(on[i]):
            continue
        pts = route[i, valid[i], :2]
        if len(pts) < 2:
            continue
        if np.linalg.norm(pts.mean(0) - center) > radius + 40:
            continue
        ax.plot(pts[:, 0], pts[:, 1], color="#8fd19e", lw=2.2, alpha=0.85, zorder=1)


def _draw_lights(ax, sample: dict, sim_t: int, center: np.ndarray, radius: float) -> None:
    xyz = sample.get("tl_xyz")
    if xyz is None:
        return
    valid = sample["tl_valid"]
    state = sample["tl_state"]
    if sim_t >= xyz.shape[1]:
        return
    for i in range(xyz.shape[0]):
        if not valid[i, sim_t]:
            continue
        xy = xyz[i, sim_t, :2]
        if np.linalg.norm(xy - center) > radius:
            continue
        color = _TL_COLOR.get(int(state[i, sim_t]), "#bbbbbb")
        ax.scatter(
            [xy[0]],
            [xy[1]],
            c=color,
            s=28,
            marker="o",
            zorder=8,
            edgecolors="black",
            linewidths=0.4,
        )


def _flag_at(flags: np.ndarray, frame: int) -> bool:
    if frame <= 0 or len(flags) == 0:
        return False
    j = frame - 1
    if j >= len(flags):
        return bool(flags[-1])
    return bool(flags[j])


def render_episode_png(
    ep: ClosedLoopEpisode,
    sample: dict,
    *,
    frame: int,
    source: str,
    selected: Sequence[str],
    threshold: float = 0.5,
    ade_scale: float = 2.0,
    max_draw: int = 800,
    radius: float = 40.0,
    show_pick: bool = True,
    show_log: bool = True,
    show_executed: bool = True,
    show_agents: bool = True,
) -> tuple[bytes, str]:
    """Draw one timestep. Returns PNG bytes and an HTML status line."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    sim_t = frame_sim_t(ep, frame)
    ri = replan_index(ep, sim_t)
    plan_t = int(ep.sim_t[ri])
    origin = ep.origins[ri]
    center = ep.ego_xy[sim_t, :2]
    scores = channel_scores(ep, ri, source, ade_scale)
    selected = list(selected)
    missing = [name for name in selected if name not in scores]
    active = [name for name in selected if name in scores]
    idx, color_s, total = passing_indices(scores, active, threshold, max_draw)

    fig = Figure(figsize=(8.2, 8.0), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    _draw_map(ax, sample, center, radius=float(radius))
    _draw_route(ax, sample, center, float(radius))
    _draw_lights(ax, sample, sim_t, center, float(radius))

    if show_log:
        log = sample["xyz"][0, ep.t0 :, :2]
        log_v = sample["valid"][0, ep.t0 :]
        if log_v.any():
            ax.plot(log[log_v, 0], log[log_v, 1], color="black", lw=1.0, alpha=0.35, zorder=4)
        end = min(sim_t + ep.vocab_xy.shape[1], sample["xyz"].shape[1])
        window = sample["xyz"][0, sim_t:end, :2]
        win_v = sample["valid"][0, sim_t:end]
        if win_v.any():
            ax.plot(
                window[win_v, 0],
                window[win_v, 1],
                color="black",
                lw=1.8,
                zorder=5,
                label="log",
            )

    if show_executed:
        ex = ep.ego_xy[ep.t0 : sim_t + 1]
        if len(ex) >= 2:
            ax.plot(ex[:, 0], ex[:, 1], color="tab:red", lw=2.2, zorder=6, label="executed")

    if len(idx):
        local = ep.vocab_xy[idx]
        if len(idx) > 250:
            local = local[:, ::2]
        world = _to_world(local, origin)
        norm = Normalize(vmin=float(threshold), vmax=1.0)
        cmap = ScalarMappable(norm=norm, cmap="plasma").cmap
        rgba = cmap(norm(np.clip(color_s, float(threshold), 1.0)))
        rgba[:, 3] = 0.22 if len(idx) > 400 else 0.55
        ax.add_collection(
            LineCollection(world, colors=rgba, linewidths=0.8, zorder=3)
        )
        fig.colorbar(
            ScalarMappable(norm=norm, cmap="plasma"),
            ax=ax,
            fraction=0.046,
            pad=0.02,
            label="weakest checked score",
        )

    pick = int(ep.pick[ri])
    pick_passes = False
    if show_pick and 0 <= pick < ep.k:
        if active:
            pick_passes = all(float(scores[name][pick]) >= float(threshold) for name in active)
        else:
            pick_passes = True
        pick_w = _to_world(ep.vocab_xy[pick], origin)
        ax.plot(
            pick_w[:, 0],
            pick_w[:, 1],
            color="cyan",
            lw=2.4,
            ls="-" if pick_passes or not active else "--",
            zorder=7,
            label="executed plan",
        )

    if show_agents:
        from matplotlib import patches

        n_agents = sample["xyz"].shape[0]
        for i in range(n_agents):
            if i == 0:
                if sim_t >= len(ep.ego_xy):
                    continue
                xy = ep.ego_xy[sim_t, :2]
                yaw = float(ep.ego_yaw[sim_t])
                color = "tab:red"
            else:
                if sim_t >= sample["valid"].shape[1] or not sample["valid"][i, sim_t]:
                    continue
                xy = sample["xyz"][i, sim_t, :2]
                yaw = float(sample["yaw"][i, sim_t])
                color = "tab:blue"
            length = float(sample["length"][i] or 4.5)
            width = float(sample["width"][i] or 2.0)
            poly = _rotated_box(xy, yaw, length, width)
            ax.add_patch(
                patches.Polygon(poly, closed=True, fc=color, ec="none", alpha=0.9, zorder=8)
            )

    ax.set_aspect("equal")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_xticks([])
    ax.set_yticks([])
    src = "ground truth" if source == "gt" else "predicted"
    names = "+".join(active) if active else "none"
    clock = (sim_t - ep.t0) * C.DT
    title = f"i{ep.scenario_index}  {ep.scenario_id[:12]}  t={clock:.1f}s  {src}  {names}"
    if _flag_at(ep.collisions, frame):
        title += "  COLLISION"
    elif _flag_at(ep.offroad, frame):
        title += "  OFFROAD"
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    png = buf.getvalue()

    per = []
    for name in active:
        n_pass = int((np.asarray(scores[name]) >= float(threshold)).sum())
        per.append(f"{name} {n_pass}/{ep.k}")
    shown = f"drawing {len(idx)} of {total}" if total > len(idx) else f"drawing {total}"
    if not active:
        shown = "check a metric to draw trajectories"
    elif total == 0:
        shown = "no trajectory clears every checked score"
    bits = [
        f"<b>{src}</b> ≥ {float(threshold):.2f} &nbsp; {shown}",
        " · ".join(per) if per else "",
    ]
    if missing:
        bits.append(
            "no predicted head for " + ", ".join(missing) + " (this checkpoint has no metric head)"
        )
    if plan_t != sim_t:
        bits.append(f"plan anchored at t={(plan_t - ep.t0) * C.DT:.1f}s")
    if show_pick and active:
        if pick_passes:
            bits.append("executed plan passes the filter")
        else:
            bits.append("executed plan misses the filter (dashed)")
    if source == "gt" and "imitation" in active:
        ade_at = -float(ade_scale) * float(np.log(max(float(threshold), 1e-6)))
        bits.append(
            f"imitation ground truth: ADE ≤ {ade_at:.2f} m (scale {float(ade_scale):.1f} m)"
        )
    if source == "pred" and ("imitation" in active or "epo" in active):
        bits.append("imitation / EPO predicted: score is that head’s softmax ÷ its peak")
    html = "<br>".join(bit for bit in bits if bit)
    return png, html
