#!/usr/bin/env python3
"""Plot open-/closed-loop metrics across DAgger iterations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "iters" in data:
        return list(data["iters"])
    if isinstance(data, list):
        return data
    raise ValueError(f"unrecognized history format: {path}")


def _series(rows: list[dict], *keys: str) -> list[float]:
    out = []
    for r in rows:
        v = r
        for k in keys:
            if not isinstance(v, dict) or k not in v:
                v = None
                break
            v = v[k]
        out.append(float(v) if v is not None else float("nan"))
    return out


def plot_dagger(history: list[dict], out: Path) -> None:
    if not history:
        raise SystemExit("empty history")
    xs = [int(r.get("iter", i)) for i, r in enumerate(history)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)

    panels = [
        (axes[0, 0], _series(history, "ol", "argmaxADE@80"), "OL argmaxADE@80 ↓", "m"),
        (axes[0, 1], _series(history, "ol", "minADE@80"), "OL minADE@80 ↓", "m"),
        (axes[0, 2], _series(history, "cl", "log_ade"), "CL log_ade ↓", "m"),
        (axes[1, 0], _series(history, "cl", "collision"), "CL collision rate ↓", ""),
        (axes[1, 1], _series(history, "cl", "offroad"), "CL offroad rate ↓", ""),
        (axes[1, 2], _series(history, "cl", "progress_ratio"), "CL progress_ratio ↑", ""),
    ]
    for ax, ys, title, ylab in panels:
        ax.plot(xs, ys, marker="o")
        ax.set_title(title)
        ax.set_xlabel("DAgger iter")
        if ylab:
            ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(xs)

    # Pool size twin on progress panel annotation
    pool = _series(history, "pool_size")
    if np.any(np.isfinite(pool)):
        axes[1, 2].text(
            0.02,
            0.98,
            f"pool@last={int(pool[-1]) if np.isfinite(pool[-1]) else 0}",
            transform=axes[1, 2].transAxes,
            va="top",
            fontsize=8,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--history",
        type=Path,
        default=Path("checkpoints/hydra_4s_dagger/history.json"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/hydra_4s_dagger/dagger_curves.png"),
    )
    args = p.parse_args()
    plot_dagger(load_history(args.history), args.out)


if __name__ == "__main__":
    main()
