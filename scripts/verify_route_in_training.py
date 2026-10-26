#!/usr/bin/env python3
"""Fail if packed corridor flags don't reach map_feat / the encoder (XY route tokens off)."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from womd_il.data.dataset import WOMDPackedDataset, packed_collate
from womd_il.data.schema import PackConfig
from womd_il.models.encoder import WayformerEncoder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        if "map_on_route" not in f:
            raise SystemExit(f"FATAL: {args.h5} has no map_on_route dataset")
        n_pack = int(f.attrs["n"])
        frac_any = float(np.asarray(f["map_on_route"][:: max(1, n_pack // 2000)]).any(-1).mean())

    ds = WOMDPackedDataset(args.h5, target_mode="sdc", cfg=PackConfig())
    n = min(int(args.n), len(ds))
    on_ok = 0
    on_lanes = []
    for i in range(n):
        feat = ds[i]
        on = np.asarray(feat["map_feat"][..., -1])
        on_lanes.append(float(on.sum()))
        if float(on.sum()) >= 1.0:
            on_ok += 1

    print(
        f"verify {args.h5}: pack_frac_any_on={frac_any:.3f}  "
        f"on_route_in_map_feat={on_ok}/{n}  mean_on_lanes={np.mean(on_lanes):.1f}"
    )
    if on_ok < int(0.5 * n):
        raise SystemExit(
            f"FATAL: map_feat on_route bit dead ({on_ok}/{n}) — model would not see corridor"
        )

    batch = packed_collate([ds[i] for i in range(min(4, n))])
    tb = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            tb[k] = v
        elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            tb[k] = torch.as_tensor(v)
        # skip scenario_id / other non-tensor fields
    enc = WayformerEncoder(
        agent_pool="lstm",
        agent_lstm_layers=2,
        use_route_tokens=False,
        factorized=True,
    )
    with torch.no_grad():
        scene = enc(tb)
        assert enc._route_tokens(tb) is None, "route XY tokens must stay off"
    on_mass = float(tb["map_feat"][..., -1].sum())
    if on_mass < 1.0:
        raise SystemExit("FATAL: batch map_feat on_route mass is 0")
    print(
        f"  encoder OK (corridor-only): tokens={tuple(scene.tokens.shape)}  "
        f"batch_on_route_mass={on_mass:.0f}  use_route_tokens=False"
    )
    print("corridor visibility check OK")


if __name__ == "__main__":
    main()
