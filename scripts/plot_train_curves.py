#!/usr/bin/env python3
"""Plot Hydra / LDiP train curves from a tqdm train log.

Example:
  python scripts/plot_train_curves.py \\
      --log logs/train_hydra_4s_ldip.log \\
      --out checkpoints/hydra_4s_ldip/loss_curves.png
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = ("nc", "dac", "ddc", "tlc", "ttc", "lk", "hc", "progress")
TRAIN_KEYS = ("loss", "im", "im16", "im256", "im4096", "m", "ade", *METRICS)


def _grab(line: str, key: str) -> float | None:
    m = re.search(rf"\b{re.escape(key)}=([0-9.]+)", line)
    return float(m.group(1)) if m else None


def parse_train_medians(log_text: str) -> tuple[list[int], dict[str, list[float]]]:
    by_ep: dict[int, list[dict[str, float | None]]] = defaultdict(list)
    for line in log_text.replace("\r", "\n").split("\n"):
        if not line.startswith("epoch") or "loss=" not in line:
            continue
        em = re.match(r"epoch\s+(\d+):", line)
        if not em:
            continue
        ep = int(em.group(1))
        row = {k: _grab(line, k) for k in TRAIN_KEYS}
        if row["loss"] is not None:
            by_ep[ep].append(row)

    epochs = sorted(by_ep)

    def med(ep: int, key: str) -> float:
        xs = [d[key] for d in by_ep[ep] if d.get(key) is not None]
        return float(np.median(xs)) if xs else float("nan")

    series = {k: [med(e, k) for e in epochs] for k in TRAIN_KEYS}
    return epochs, series


def parse_val_argmax(log_text: str) -> list[tuple[int, float]]:
    vals = [float(x) for x in re.findall(r"argmaxADE@80=([0-9.]+)", log_text)]
    return list(enumerate(vals))


def plot_curves(
    epochs: list[int],
    series: dict[str, list[float]],
    val_ade: list[tuple[int, float]],
    out: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True)

    ax = axes[0]
    ax.plot(epochs, series["loss"], "o-", label="loss (total)", color="#1f4e79", lw=2, ms=5)
    ax.plot(epochs, series["im"], "s-", label="im = Σ stage CE", color="#c45c26", lw=1.6, ms=4)
    ax.plot(epochs, series["m"], "^-", label="metric BCE (sum)", color="#2a9d8f", lw=1.6, ms=4)
    ax.set_ylabel("train (median)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    ax = axes[1]
    ax.plot(epochs, series["im4096"], "o-", label="im4096", color="#1f77b4", lw=1.6)
    ax.plot(epochs, series["im256"], "s-", label="im256", color="#ff7f0e", lw=1.6)
    ax.plot(epochs, series["im16"], "^-", label="im16", color="#2ca02c", lw=1.6)
    ax.axhline(np.log(4096), color="#1f77b4", ls=":", alpha=0.45)
    ax.axhline(np.log(256), color="#ff7f0e", ls=":", alpha=0.45)
    ax.axhline(np.log(16), color="#2ca02c", ls=":", alpha=0.45)
    ax.set_ylabel("stage CE (dotted = random)")
    ax.grid(True, alpha=0.3)
    if val_ade:
        ax2 = ax.twinx()
        ve, va = zip(*val_ade)
        ax2.plot(ve, va, "D--", color="#d62728", label="val argmaxADE@80", lw=1.6, ms=5)
        ax2.set_ylabel("val argmaxADE@80 (m)", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=8)
    else:
        ax.legend(loc="upper right", frameon=False)

    ax = axes[2]
    colors = {
        "nc": "#e63946",
        "dac": "#457b9d",
        "ddc": "#1d3557",
        "tlc": "#2a9d8f",
        "ttc": "#e9c46a",
        "lk": "#f4a261",
        "hc": "#264653",
        "progress": "#a8dadc",
    }
    for name in METRICS:
        ys = series[name]
        if all(np.isnan(ys)):
            continue
        ax.plot(epochs, ys, "o-", label=name, color=colors[name], lw=1.5, ms=4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("per-metric BCE (median)")
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Metric head BCEs")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--log",
        type=Path,
        default=Path("logs/train_hydra_4s_ldip.log"),
        help="Train stdout log (tqdm postfixes + val tables)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("checkpoints/hydra_4s_ldip/loss_curves.png"),
        help="Output PNG path",
    )
    p.add_argument("--title", default="Hydra 4s LDiP — losses over epochs")
    args = p.parse_args()

    text = args.log.read_bytes().decode("utf-8", "replace")
    epochs, series = parse_train_medians(text)
    val_ade = parse_val_argmax(text)
    if not epochs:
        raise SystemExit(f"no epoch train rows found in {args.log}")

    plot_curves(epochs, series, val_ade, args.out, args.title)
    print(f"wrote {args.out.resolve()}")
    print(f"epochs {epochs[0]}..{epochs[-1]}  ({len(epochs)} train, {len(val_ade)} val)")
    if val_ade:
        best_e, best_a = min(val_ade, key=lambda t: t[1])
        print(f"best val argmaxADE@80={best_a:.3f} @ ep{best_e}")
        print("val:", ", ".join(f"{e}:{a:.3f}" for e, a in val_ade))


if __name__ == "__main__":
    main()
