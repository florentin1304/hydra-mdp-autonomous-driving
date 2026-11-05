"""Closed-loop metrics + video dump used at the end of training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from womd_il.data.features import build_sample_features
from womd_il.data.transforms import inverse_agent_centric_xy
from womd_il.metrics.closed_loop import closed_loop_metrics
from womd_il.metrics.open_loop import nms_trajectories
from womd_il.sim.obs import features_to_batch
from womd_il.sim.rollout import rollout_policy
from womd_il.viz.video import render_closed_loop_video, render_open_loop_video

HEADLINE = ("collision", "offroad", "progress", "progress_ratio", "log_ade", "log_fde")


@torch.no_grad()
def open_loop_world_preds(policy, sample: dict, device, num_keep: int = 128, nms: bool = False):
    feat = build_sample_features(sample, target_idx=0)
    batch = features_to_batch(feat, device)
    bundle = policy.infer(batch)
    # Prefer imitation softmax for ranking when present (true pick probs).
    extras = bundle.extras or {}
    rank = extras.get("im_scores", bundle.scores)
    if nms:
        xy, scores = nms_trajectories(bundle.xyh[..., :2], rank, num_keep=num_keep)
        local = xy[0]
        sc = scores[0]
    else:
        k = min(int(num_keep), int(rank.size(-1)))
        top = torch.topk(rank[0], k)
        local = bundle.xyh[0, top.indices, :, :2]
        sc = top.values
    origin = batch["current_xyh_world"][0].detach().cpu().numpy()
    world = inverse_agent_centric_xy(local.detach().cpu().numpy(), origin[:2], float(origin[2]))
    return world, sc.detach().cpu().numpy()


def closed_loop_eval(
    policy,
    dataset,
    *,
    neighbors: str,
    horizon: int,
    device,
    max_scenarios: int | None = None,
    desc: str | None = None,
    h_exec: int = 1,
    mode_select: str = "argmax",
    cost: dict | None = None,
    early_stop_on_fail: bool = False,
) -> dict[str, Any]:
    n = len(dataset)
    if max_scenarios is not None:
        n = min(n, int(max_scenarios))
    rows = []
    label = desc or f"closed-loop[{neighbors}]"
    h_exec = int(h_exec)
    for i in tqdm(range(n), desc=label):
        sample = dataset.get_world_sample(i)
        state = rollout_policy(
            policy,
            sample,
            neighbors=neighbors,
            horizon=horizon,
            device=device,
            h_exec=h_exec,
            mode_select=mode_select,
            cost=cost,
            early_stop_on_fail=early_stop_on_fail,
        )
        m = closed_loop_metrics(state, sample)
        m["scenario_id"] = str(sample.get("scenario_id", i))
        m["scenario_index"] = i
        rows.append(m)
    if not rows:
        return {
            "summary": {
                "n": 0.0,
                "collision": 0.0,
                "offroad": 0.0,
                "progress": 0.0,
                "log_progress": 0.0,
                "collision_count": 0.0,
                "offroad_count": 0.0,
            },
            "per_scenario": [],
            "neighbors": neighbors,
        }
    keys = [k for k in rows[0] if k not in ("scenario_id", "scenario_index")]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary["n"] = float(n)
    summary["collision_count"] = float(sum(r["collision"] > 0.5 for r in rows))
    summary["offroad_count"] = float(sum(r["offroad"] > 0.5 for r in rows))
    return {"summary": summary, "per_scenario": rows, "neighbors": neighbors}


def print_closed_loop_report(payload: dict[str, Any]) -> None:
    s = payload["summary"]
    n = int(s.get("n", 0))
    neigh = payload.get("neighbors", "?")
    print()
    print(f"========== closed-loop ({neigh})  n={n} ==========")
    print(
        f"  collision       {s['collision']:.3f}   "
        f"({int(s.get('collision_count', 0))}/{n} scenes)"
    )
    print(
        f"  offroad         {s['offroad']:.3f}   "
        f"({int(s.get('offroad_count', 0))}/{n} scenes)"
    )
    print(f"  progress        {s['progress']:.2f} m")
    print(f"  log_progress    {s['log_progress']:.2f} m")
    print(f"  progress_ratio  {s['progress_ratio']:.3f}")
    print(f"  log_ade         {s['log_ade']:.3f} m")
    print(f"  log_fde         {s['log_fde']:.3f} m")
    print("================================================")


def render_report_videos(
    policy,
    dataset,
    out_dir: Path,
    *,
    device,
    n_videos: int,
    neighbors: list[str],
    horizon: int,
    open_loop: bool = True,
    h_exec: int = 1,
    mode_select: str = "argmax",
    cost: dict | None = None,
) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    n = min(int(n_videos), len(dataset))
    h_exec = int(h_exec)
    for i in tqdm(range(n), desc="videos"):
        sample = dataset.get_world_sample(i)
        sid = str(sample.get("scenario_id", i)).replace("/", "_")
        if open_loop:
            pred, scores = open_loop_world_preds(policy, sample, device, num_keep=128, nms=False)
            path = render_open_loop_video(
                out_dir / f"{i:04d}_{sid}_open",
                sample,
                pred,
                scores,
                title=f"open-loop {sid}",
                n_modes=128,
            )
            written.append(str(path))
        for neigh in neighbors:
            state = rollout_policy(
                policy,
                sample,
                neighbors=neigh,
                horizon=horizon,
                device=device,
                h_exec=h_exec,
                mode_select=mode_select,
                cost=cost,
            )
            path = render_closed_loop_video(
                out_dir / f"{i:04d}_{sid}_closed_{neigh}",
                sample,
                state,
                title=f"closed-loop {neigh} {sid}",
            )
            written.append(str(path))
    return written


def run_end_of_training_report(
    policy,
    dataset,
    *,
    out_dir: Path,
    device,
    max_scenarios: int = 64,
    viz_n: int = 8,
    neighbors: list[str] | tuple[str, ...] = ("log", "idm"),
    horizon: int = 80,
    wandb_run=None,
    h_exec: int | None = None,
) -> dict[str, Any]:
    """Always-on post-training closed-loop metrics + videos."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.eval()
    neighbors = [str(x) for x in neighbors]
    decoder = getattr(policy, "decoder", None)
    if h_exec is None:
        h_exec = int(getattr(decoder, "h_exec", 1) or 1)
    raw_select = str(getattr(decoder, "cl_select", "argmax") or "argmax")
    mode_select = (
        "argmax"
        if raw_select in ("argmax", "epdms", "pdms", "imitation", "hydra_mdp_w")
        else raw_select
    )
    cost = {"topk": 256, "look_ahead": 20, "w_ov": 16.0, "w_off": 8.0, "w_speed": 0.2, "w_im": 1.0}
    report: dict[str, Any] = {"closed_loop": {}, "videos": []}
    for neigh in neighbors:
        payload = closed_loop_eval(
            policy,
            dataset,
            neighbors=neigh,
            horizon=horizon,
            device=device,
            max_scenarios=max_scenarios,
            h_exec=h_exec,
            mode_select=mode_select,
            cost=cost,
        )
        print_closed_loop_report(payload)
        report["closed_loop"][neigh] = payload
        if wandb_run is not None:
            wandb_run.log({f"closed/{neigh}/{k}": v for k, v in payload["summary"].items()})
    videos = render_report_videos(
        policy,
        dataset,
        out_dir / "videos",
        device=device,
        n_videos=viz_n,
        neighbors=neighbors,
        horizon=horizon,
        open_loop=True,
        h_exec=h_exec,
        mode_select=mode_select,
        cost=cost,
    )
    report["videos"] = videos
    json_path = out_dir / "closed_loop.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"metrics JSON -> {json_path}")
    print(f"videos ({len(videos)}) -> {out_dir / 'videos'}")
    return report
