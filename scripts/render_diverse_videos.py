#!/usr/bin/env python3
"""Diverse closed-log videos (same scene set as hydra_4s_epo/videos_diverse_log)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from womd_il.data.dataset import WOMDPackedDataset
from womd_il.data.schema import PackConfig
from womd_il.metrics.closed_loop import closed_loop_metrics
from womd_il.sim.rollout import rollout_policy
from womd_il.utils import load_policy_from_ckpt
from womd_il.viz.video import render_closed_loop_video

def load_picked(suite: Path | None) -> list[tuple[int, str, str]]:
    if suite is None or not suite.exists():
        suite = Path("configs/eval/diversity_suite.json")
    rows = json.loads(suite.read_text())
    return [(int(r["idx"]), str(r["tag"]), str(r.get("sid") or "")) for r in rows]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/hydra_4s_epo_try/best.pt"))
    p.add_argument("--h5", type=Path, default=Path("data/womd/packed/val.h5"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/hydra_4s_epo_try/videos_diverse_log"))
    p.add_argument("--h-exec", type=int, default=1)
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--device", default="cuda")
    p.add_argument("--suite", type=Path, default=Path("configs/eval/diversity_suite.json"))
    p.add_argument(
        "--select",
        default="argmax",
        help="Mode select: argmax | avg | geomean | pdms | … "
        "(avg/geomean = blend im_scores & epo_scores)",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    picked = load_picked(args.suite)
    print(f"loading {args.ckpt} on {device}  select={args.select}  n={len(picked)}", flush=True)
    policy, _ = load_policy_from_ckpt(args.ckpt, device=device)
    policy.eval()
    ds = WOMDPackedDataset(args.h5, target_mode="sdc", cfg=PackConfig())

    rows = []
    manifest_lines = []
    for j, (idx, tag, expect_sid) in enumerate(tqdm(picked, desc=f"diverse:{args.select}")):
        sample = ds.get_world_sample(idx)
        sid = str(sample.get("scenario_id", idx)).replace("/", "_")
        if expect_sid and expect_sid not in sid and sid != expect_sid:
            print(f"WARN idx={idx}: sid={sid} expected~{expect_sid}", flush=True)
        state = rollout_policy(
            policy,
            sample,
            neighbors="log",
            horizon=args.horizon,
            device=device,
            h_exec=args.h_exec,
            mode_select=args.select,
            early_stop_on_fail=True,
        )
        m = closed_loop_metrics(state, sample)
        m["scenario_index"] = idx
        m["tag"] = tag
        m["scenario_id"] = sid
        rows.append(m)
        stem = args.out / f"{j:04d}_{tag}_i{idx}_{sid}_closed_log"
        path = render_closed_loop_video(
            stem,
            sample,
            state,
            title=f"{tag} i={idx} sel={args.select} h_exec={args.h_exec} {sid}",
        )
        line = (
            f"{j:04d}\t{tag}\tidx={idx}\t{sid}\t"
            f"coll={int(m['collision'] > 0.5)}\toff={int(m['offroad'] > 0.5)}\t"
            f"prog={m['progress']:.1f}\tratio={m['progress_ratio']:.3f}\tade={m['log_ade']:.3f}"
        )
        manifest_lines.append(line)
        print(line, flush=True)
        print(f"  -> {path}", flush=True)

    summary = {
        k: float(np.mean([r[k] for r in rows]))
        for k in (
            "collision",
            "offroad",
            "progress",
            "log_progress",
            "progress_ratio",
            "log_ade",
            "log_fde",
        )
    }
    summary["n"] = float(len(rows))
    summary["collision_count"] = float(sum(r["collision"] > 0.5 for r in rows))
    summary["offroad_count"] = float(sum(r["offroad"] > 0.5 for r in rows))
    # per-tag rollup
    by_tag: dict[str, list] = {}
    for r in rows:
        by_tag.setdefault(str(r["tag"]), []).append(r)
    tag_summary = {}
    for tag, rs in by_tag.items():
        tag_summary[tag] = {
            "n": len(rs),
            "collision": float(np.mean([x["collision"] for x in rs])),
            "offroad": float(np.mean([x["offroad"] for x in rs])),
            "progress": float(np.mean([x["progress"] for x in rs])),
            "log_ade": float(np.mean([x["log_ade"] for x in rs])),
            "collision_count": float(sum(x["collision"] > 0.5 for x in rs)),
            "offroad_count": float(sum(x["offroad"] > 0.5 for x in rs)),
        }
    payload = {
        "summary": summary,
        "by_tag": tag_summary,
        "per_scenario": rows,
        "ckpt": str(args.ckpt),
        "h_exec": args.h_exec,
        "mode_select": args.select,
    }
    (args.out / "metrics.json").write_text(json.dumps(payload, indent=2, default=float))
    (args.out / "manifest.txt").write_text("\n".join(manifest_lines) + "\n")
    print("DONE", args.out, flush=True)
    print(json.dumps({"select": args.select, **summary, "by_tag": tag_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
