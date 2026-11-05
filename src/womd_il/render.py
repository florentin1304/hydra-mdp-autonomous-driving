"""Render open-loop and closed-loop BEV videos from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from womd_il.data.dataset import WOMDPackedDataset
from womd_il.data.schema import PackConfig
from womd_il.report import open_loop_world_preds
from womd_il.sim.rollout import rollout_policy
from womd_il.utils import load_policy_from_ckpt, resolve_device
from womd_il.viz.video import render_closed_loop_video, render_open_loop_video


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--h5", type=Path, default=Path("data/womd/packed/val.h5"))
    p.add_argument("--out", type=Path, default=None, help="Default: <checkpoint dir>/videos")
    p.add_argument("--n", type=int, default=6, help="How many scenarios to render")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--neighbors", default="log", choices=("log", "idm", "both"))
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--open-loop", action="store_true", default=True)
    p.add_argument("--no-open-loop", action="store_false", dest="open_loop")
    p.add_argument("--closed-loop", action="store_true", default=True)
    p.add_argument("--no-closed-loop", action="store_false", dest="closed_loop")
    args = p.parse_args()
    if args.out is None:
        args.out = args.ckpt.parent / "videos"

    device = resolve_device(args.device)
    policy, _ = load_policy_from_ckpt(args.ckpt, device=device)
    ds = WOMDPackedDataset(args.h5, target_mode="sdc", cfg=PackConfig())
    n = min(args.n, len(ds) - args.start)
    args.out.mkdir(parents=True, exist_ok=True)
    neigh_modes = ["log", "idm"] if args.neighbors == "both" else [args.neighbors]

    for j in range(n):
        idx = args.start + j
        sample = ds.get_world_sample(idx)
        sid = str(sample.get("scenario_id", idx)).replace("/", "_")
        print(f"[{j+1}/{n}] scenario {idx} {sid}")
        if args.open_loop:
            pred, scores = open_loop_world_preds(policy, sample, device)
            out = render_open_loop_video(
                args.out / f"{idx:04d}_{sid}_open",
                sample,
                pred,
                scores,
                title=f"open-loop {sid}",
            )
            print(f"  open-loop  -> {out}")
        if args.closed_loop:
            for neigh in neigh_modes:
                state = rollout_policy(
                    policy,
                    sample,
                    neighbors=neigh,
                    horizon=args.horizon,
                    device=device,
                )
                out = render_closed_loop_video(
                    args.out / f"{idx:04d}_{sid}_closed_{neigh}",
                    sample,
                    state,
                    title=f"closed-loop {neigh} {sid}",
                )
                print(f"  closed {neigh} -> {out}")


if __name__ == "__main__":
    main()
