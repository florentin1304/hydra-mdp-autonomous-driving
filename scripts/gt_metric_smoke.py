#!/usr/bin/env python3
"""Score logged SDC futures with Hydra metric labels vs LDiP human-agent numbers.

LDiP Table 1 (Navtest v2) Human Agent:
  NC 100  DAC 100  DDC 99.8  TLC 100  EP 87.4  TTC 100  LK 100  HC 98.1  EC 90.1
  EPDMS 90.3
Table 2 (Navtest v1) Human: PDMS 94.8  (EP 87.5 is PDM-closed-loop; open-loop
self-progress should be ~100).

Do not recache or train until DAC/NC/TTC/DDC/TLC/LK/HC are in the same ballpark.
EC cannot be labeled from a single plan.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from womd_il.data.dataset import WOMDPackedDataset, packed_collate  # noqa: E402
from womd_il.data.schema import PackConfig  # noqa: E402
from womd_il.losses.hydra import (  # noqa: E402
    METRIC_NAMES,
    hydra_metric_labels,
    pdms_from_probs,
    pdms_v1_from_probs,
)

# LDiP arXiv:2609.07049 Table 1 / Table 2, percent.
PAPER_HUMAN_V2 = {
    "nc": 100.0,
    "dac": 100.0,
    "ddc": 99.8,
    "tlc": 100.0,
    "progress": 87.4,
    "ttc": 100.0,
    "lk": 100.0,
    "hc": 98.1,
    "ec": 90.1,
    "epdms": 90.3,
}
PAPER_HUMAN_V1 = {"pdms": 94.8}

# Open-loop logged GT should match the 100-ish multipliers. EP in the paper is
# closed-loop PDM; self-progress here should be ~100, not 87.
BARS = {
    "nc": 0.95,
    "dac": 0.95,
    "ddc": 0.90,
    "tlc": 0.90,
    "ttc": 0.95,
    "lk": 0.85,
    "hc": 0.90,
    "progress": 0.90,
    "pdms_v1": 0.85,
    "epdms": 0.80,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=Path, default=ROOT / "data/womd/packed/val.h5")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ds = WOMDPackedDataset(args.h5, target_mode="sdc", max_scenarios=int(args.n), cfg=PackConfig())
    loader = torch.utils.data.DataLoader(
        ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=packed_collate, num_workers=0
    )
    sums = {name: 0.0 for name in METRIC_NAMES}
    sums["pdms_v1"] = 0.0
    sums["epdms"] = 0.0
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            tensors = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            gt = tensors["gt_xy"]
            b, t, _ = gt.shape
            vocab = gt[:, None]
            yaw = tensors.get("gt_yaw")
            pred_yaw = yaw[:, None] if yaw is not None else None
            y = hydra_metric_labels(vocab, tensors, pred_yaw=pred_yaw, max_k=None)
            for i, name in enumerate(METRIC_NAMES):
                sums[name] += float(y[..., i].sum().cpu())
            sums["pdms_v1"] += float(pdms_v1_from_probs(y).sum().cpu())
            sums["epdms"] += float(pdms_from_probs(y).sum().cpu())
            n_seen += b
    ds.close()
    means = {k: v / max(n_seen, 1) for k, v in sums.items()}

    print(f"GT open-loop labels  n={n_seen}  {args.h5}  {device}")
    print(f"{'metric':>10}  {'ours%':>8}  {'paper%':>8}  {'bar':>8}  pass")
    failed = []
    order = list(METRIC_NAMES) + ["pdms_v1", "epdms"]
    for name in order:
        ours = 100.0 * means[name]
        paper = PAPER_HUMAN_V2.get(name, PAPER_HUMAN_V1.get(name.replace("pdms_v1", "pdms"), float("nan")))
        if name == "progress":
            paper_note = f"{PAPER_HUMAN_V2['progress']:.1f}*"
        elif name == "pdms_v1":
            paper_note = f"{PAPER_HUMAN_V1['pdms']:.1f}"
            paper = PAPER_HUMAN_V1["pdms"]
        elif name == "epdms":
            paper_note = f"{PAPER_HUMAN_V2['epdms']:.1f}"
            paper = PAPER_HUMAN_V2["epdms"]
        else:
            paper_note = f"{paper:.1f}"
        bar = BARS.get(name)
        ok = True if bar is None else means[name] >= bar
        if not ok:
            failed.append(name)
        flag = "ok" if ok else "FAIL"
        print(f"{name:>10}  {ours:8.1f}  {paper_note:>8}  {100.0 * (bar or 0):8.0f}  {flag}")
    print("* paper EP is PDM closed-loop; open-loop GT vs itself should be ~100.")
    print("EC is not labeled (needs two consecutive planner outputs).")
    if failed:
        print(f"FAILED: {', '.join(failed)}  — do not recache or train.")
        raise SystemExit(1)
    print("PASS: human-log labels are in the LDiP ballpark. Safe to recache.")


if __name__ == "__main__":
    main()
