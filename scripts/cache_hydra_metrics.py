#!/usr/bin/env python3
"""Precompute Hydra EPDMS-proxy labels for the full vocabulary.

Writes ``{out_dir}/{packed_stem}.h5`` with uint8 labels [N, K, 8]
(nc, dac, ddc, tlc, ttc, lk, hc, progress). EC is omitted (needs two plans).

CPU: split scenes across ``--jobs`` processes (one torch thread each).
GPU: one process streams into H5 (``--device cuda``). With multiple GPUs,
``--jobs N`` shards scenes across ``cuda:0..N-1``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from womd_il.data.build_metric_cache import cache_metrics  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=Path, required=True, help="Packed train or val HDF5")
    p.add_argument("--vocab", type=Path, default=ROOT / "data/artifacts/hydra_vocab.npy")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/artifacts/hydra_metrics")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (single-job / GPU)")
    p.add_argument("--jobs", type=int, default=None, help="Processes. Default: 1 on CUDA, nproc on CPU")
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu | cuda. Default: cuda if available")
    p.add_argument("--windows", type=Path, default=None, help="Optional (scene, t0) table for windowed train")
    args = p.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu = torch.cuda.device_count() if args.device.startswith("cuda") else 0
    if args.jobs is None:
        if args.device.startswith("cuda"):
            args.jobs = max(1, n_gpu)
        else:
            args.jobs = max(1, os.cpu_count() or 4)
    if args.device.startswith("cuda") and args.jobs > 1 and n_gpu <= 1:
        print("Single GPU: using one process (map ops are already GPU-bound). Pass --jobs only for multi-GPU.")
        args.jobs = 1
    if args.batch_size is None:
        # Map heads are per-scene; small B keeps host↔device light. NC/TTC still batch.
        args.batch_size = 8 if args.device.startswith("cuda") else 4

    cache_metrics(
        args.h5,
        args.vocab,
        args.out_dir,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        jobs=int(args.jobs),
        max_scenarios=args.max_scenarios,
        device=str(args.device),
        windows_path=args.windows,
    )


if __name__ == "__main__":
    main()
