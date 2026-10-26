#!/usr/bin/env python3
"""Parse WOMD Scenario TFRecords into a packed HDF5 file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from womd_il.data.pack_h5 import append_sample, create_h5, open_h5  # noqa: E402
from womd_il.data.parse_scenario import parse_tfrecord_bytes  # noqa: E402
from womd_il.data.schema import PackConfig  # noqa: E402
from womd_il.data.tfrecord import iter_tfrecord_dir  # noqa: E402


def _has_usable_route(sample: dict) -> bool:
    """Prefer an on-route path with enough valid points; else any path."""
    import numpy as np

    valid = np.asarray(sample["route_valid"])
    on = np.asarray(sample.get("route_on_route", np.zeros(valid.shape[0], bool)))
    counts = valid.sum(axis=-1)
    if np.any(on & (counts >= 2)):
        return True
    return bool(np.any(counts >= 2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--num-agents", type=int, default=64)
    p.add_argument("--num-map-polylines", type=int, default=512)
    p.add_argument("--append", action="store_true", help="Append to an existing pack.")
    p.add_argument("--skip-shards", type=int, default=0, help="Skip the first N sorted shards.")
    p.add_argument(
        "--require-route",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip scenarios with empty sdc_paths (default: true; needs WOMD ≥1.3.1).",
    )
    args = p.parse_args()

    cfg = PackConfig(
        num_agents=args.num_agents,
        num_map_polylines=args.num_map_polylines,
    )
    added = 0
    skipped_no_route = 0
    f = open_h5(args.out) if args.append else create_h5(args.out, cfg)
    with f:
        n = int(f.attrs["n"])
        for raw in tqdm(
            iter_tfrecord_dir(args.raw_dir, skip_shards=args.skip_shards),
            desc="preprocess",
        ):
            if args.max_scenarios is not None and n >= args.max_scenarios:
                break
            sample = parse_tfrecord_bytes(raw, cfg)
            if args.require_route and not _has_usable_route(sample):
                skipped_no_route += 1
                continue
            if "map_on_route" not in sample:
                raise RuntimeError("parse must fill map_on_route before pack")
            append_sample(f, sample)
            n += 1
            added += 1
    print(
        f"Wrote {added} new scenarios (total {n}) -> {args.out}"
        + (f"  skipped_no_route={skipped_no_route}" if skipped_no_route else "")
    )


if __name__ == "__main__":
    main()
