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

# Same variety set as checkpoints/hydra_4s_epo/videos_diverse_log/manifest.txt
PICKED = [
    (1349, "straight", "2b5de09e898af8aa"),
    (1101, "straight", "9ace9b3267cddfe6"),
    (2211, "straight", "c5321022ff385608"),
    (540, "turn", "ce0be5afd6b03891"),
    (1642, "turn", "90b672a3a529af98"),
    (4142, "turn", "1a9e189a99369a8c"),
    (1386, "traffic_light", "781d434bee8a6447"),
    (171, "traffic_light", "797779c184e54223"),
    (2093, "traffic_light", "76bb4b8a12314fb2"),
    (3367, "yield", "9cd6397da0030636"),
    (1028, "yield", "def8bd3ba11bf807"),
    (4169, "yield", "fb356ccbe025fbba"),
    (2770, "stop_sign", "b580633b6a506c11"),
    (246, "stop_sign", "fb059a2c9e505e9d"),
    (23, "crosswalk", "137a96e2448a70f2"),
    (1754, "crosswalk", "d01f4dddfbc907c0"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/hydra_4s_epo_try/best.pt"))
    p.add_argument("--h5", type=Path, default=Path("data/womd/packed/val.h5"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/hydra_4s_epo_try/videos_diverse_log"))
    p.add_argument("--h-exec", type=int, default=1)
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--select",
        default="argmax",
        help="Mode select: argmax | avg | geomean | pdms | … "
        "(avg/geomean = blend im_scores & epo_scores)",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"loading {args.ckpt} on {device}  select={args.select}", flush=True)
    policy, _ = load_policy_from_ckpt(args.ckpt, device=device)
    policy.eval()
    ds = WOMDPackedDataset(args.h5, target_mode="sdc", cfg=PackConfig())

    rows = []
    manifest_lines = []
    for j, (idx, tag, expect_sid) in enumerate(tqdm(PICKED, desc=f"diverse:{args.select}")):
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
