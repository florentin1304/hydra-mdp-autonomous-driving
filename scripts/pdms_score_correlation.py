"""Correlate EPO scores with ground-truth PDMS and with metric-head PDMS.

One forward pass per validation scene, for each iteration checkpoint.
EPO score is sigmoid(EPO logit), so it sits in (0, 1) like PDMS. Rank
correlation is invariant to that monotone map.

Within-scene correlations are the headline: each scene has its own 4096
modes, and pooling them would mix different PDMS baselines.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from womd_il.data.dataset import WOMDPackedDataset, packed_collate
from womd_il.data.metric_cache import resolve_metric_cache, vocab_sha
from womd_il.data.schema import PackConfig
from womd_il.losses.hydra import pdms_from_metric_logits, pdms_from_probs
from womd_il.utils import config_dir, load_policy_from_ckpt, resolve_device

ROOT = Path("checkpoints/hydra_4s_dagger")
OUT = ROOT / "pdms_correlation"
ITERS = (0, 1, 2, 3, 4)
N_BINS = 10
SCATTER_PER_SCENE = 4


def _corr_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    return (x * y).sum(dim=-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp_min(1e-8)


def _spearman_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    rx = x.argsort(dim=-1).argsort(dim=-1).float()
    ry = y.argsort(dim=-1).argsort(dim=-1).float()
    return _corr_rows(rx, ry)


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def score_checkpoint(policy, loader, device) -> dict:
    spearman = {k: [] for k in ("epo_gt", "epo_head", "head_gt")}
    pearson = {k: [] for k in ("epo_gt", "epo_head", "head_gt")}
    dec_gt = np.zeros(N_BINS, np.float64)
    dec_head = np.zeros(N_BINS, np.float64)
    dec_epo = np.zeros(N_BINS, np.float64)
    n_scenes = 0
    top_epo_gt = []
    top_epo_head = []
    selected = []
    oracle = []
    scatter = []
    policy.eval()
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        labels = batch.get("metric_labels")
        if labels is None:
            raise RuntimeError("validation batch has no metric_labels")
        bundle = policy.infer(batch)
        epo = bundle.extras["epo_logits"].float()
        mlog = bundle.extras["metric_logits"].float()
        gt = pdms_from_probs(labels.float())
        head = pdms_from_metric_logits(mlog)
        if gt.shape != epo.shape or head.shape != epo.shape:
            raise RuntimeError(f"shape mismatch epo {tuple(epo.shape)} gt {tuple(gt.shape)} head {tuple(head.shape)}")
        sig = epo.sigmoid()
        spearman["epo_gt"].extend(_spearman_rows(epo, gt).tolist())
        spearman["epo_head"].extend(_spearman_rows(epo, head).tolist())
        spearman["head_gt"].extend(_spearman_rows(head, gt).tolist())
        pearson["epo_gt"].extend(_corr_rows(sig, gt).tolist())
        pearson["epo_head"].extend(_corr_rows(sig, head).tolist())
        pearson["head_gt"].extend(_corr_rows(head, gt).tolist())

        b, k = epo.shape
        order = epo.argsort(dim=-1)
        edges = torch.linspace(0, k, N_BINS + 1, device=epo.device).long()
        for i in range(N_BINS):
            idx = order[:, edges[i] : edges[i + 1]]
            dec_gt[i] += gt.gather(1, idx).mean(dim=1).sum().item()
            dec_head[i] += head.gather(1, idx).mean(dim=1).sum().item()
            dec_epo[i] += sig.gather(1, idx).mean(dim=1).sum().item()
        n_scenes += b

        epo_pick = epo.argmax(dim=-1)
        gt_pick = gt.argmax(dim=-1)
        head_pick = head.argmax(dim=-1)
        row = torch.arange(b, device=epo.device)
        top_epo_gt.append(float((epo_pick == gt_pick).float().mean()))
        top_epo_head.append(float((epo_pick == head_pick).float().mean()))
        selected.append(float(gt[row, epo_pick].mean()))
        oracle.append(float(gt[row, gt_pick].mean()))

        take = torch.randint(0, k, (b, SCATTER_PER_SCENE), device=epo.device)
        pct = epo.argsort(dim=-1).argsort(dim=-1).float() / max(k - 1, 1)
        scatter.append(
            torch.stack(
                [pct.gather(1, take), gt.gather(1, take), head.gather(1, take)],
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, 3)
        )
        if n_scenes % 512 < b:
            print(f"  scenes {n_scenes}", flush=True)
    pts = np.concatenate(scatter, axis=0).astype(np.float32)
    return {
        "n_scenes": n_scenes,
        "spearman_mean": {k: _mean(v) for k, v in spearman.items()},
        "spearman_median": {k: float(np.median(v)) for k, v in spearman.items()},
        "pearson_mean": {k: _mean(v) for k, v in pearson.items()},
        "pearson_median": {k: float(np.median(v)) for k, v in pearson.items()},
        "top1_epo_matches_gt": _mean(top_epo_gt),
        "top1_epo_matches_head": _mean(top_epo_head),
        "gt_pdms_of_epo_argmax": _mean(selected),
        "gt_pdms_oracle": _mean(oracle),
        "decile_gt": (dec_gt / max(n_scenes, 1)).tolist(),
        "decile_head": (dec_head / max(n_scenes, 1)).tolist(),
        "decile_epo_sigmoid": (dec_epo / max(n_scenes, 1)).tolist(),
        "scatter": pts,
    }


def _plot(results: list[dict]) -> None:
    labels = [f"iter {r['iter']}" for r in results]
    x = np.arange(len(results))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    keys = (
        ("epo_gt", "EPO vs ground-truth PDMS"),
        ("epo_head", "EPO vs metric-head PDMS"),
        ("head_gt", "Metric-head vs ground-truth PDMS"),
    )
    for i, (key, name) in enumerate(keys):
        vals = [r["spearman_mean"][key] for r in results]
        ax.bar(x + (i - 1) * width, vals, width, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean within-scene Spearman")
    ax.set_xlabel("Checkpoint")
    ax.set_title("PDMS rank correlation on the validation set")
    ax.legend(fontsize=8, loc="lower right")
    ax.axhline(0, color="black", lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "spearman_bars.png", dpi=140)
    plt.close(fig)

    deciles = np.arange(1, N_BINS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True)
    for r in results:
        axes[0].plot(deciles, r["decile_gt"], marker="o", ms=3.5, label=f"iter {r['iter']}")
        axes[1].plot(deciles, r["decile_head"], marker="o", ms=3.5, label=f"iter {r['iter']}")
    axes[0].set_title("Ground-truth PDMS by EPO rank decile")
    axes[1].set_title("Metric-head PDMS by EPO rank decile")
    for ax in axes:
        ax.set_xlabel("EPO rank decile (1 = lowest score)")
        ax.set_ylabel("Mean PDMS")
        ax.set_xticks(deciles)
        ax.legend(fontsize=8)
    fig.suptitle("Validation set, mean over scenes", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "decile_curves.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(results), figsize=(2.5 * len(results), 5.4), sharex=True, sharey=True)
    for col, r in enumerate(results):
        pts = r["scatter"]
        axes[0, col].hexbin(pts[:, 0], pts[:, 1], gridsize=35, cmap="viridis", mincnt=1)
        axes[1, col].hexbin(pts[:, 0], pts[:, 2], gridsize=35, cmap="viridis", mincnt=1)
        axes[0, col].set_title(f"iter {r['iter']}")
    axes[0, 0].set_ylabel("Ground-truth PDMS")
    axes[1, 0].set_ylabel("Metric-head PDMS")
    for ax in axes[1]:
        ax.set_xlabel("EPO rank percentile")
    fig.suptitle("Subsample of modes (4 per validation scene)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "hexbin_epo.png", dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(config_dir()), version_base=None):
        cfg = compose(config_name="dagger")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    device = resolve_device(cfg.device)
    sha = vocab_sha(cfg.decoder.vocab_path)
    cache = resolve_metric_cache(cfg.decoder.metric_cache, cfg.data.val_file)
    ds = WOMDPackedDataset(
        cfg.data.val_file,
        target_mode=cfg.data.target_mode,
        cfg=PackConfig.from_cfg(cfg.data),
        metric_cache=cache,
        vocab_sha=sha,
        vocab_size=int(cfg.decoder.vocab_size) if cfg.decoder.get("vocab_size") else None,
    )
    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        collate_fn=packed_collate,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    print(f"val scenes {len(ds)} cache {cache}", flush=True)
    saved = []
    for it in ITERS:
        ckpt = ROOT / f"iter{it:02d}" / "best.pt"
        print(f"iter {it} {ckpt}", flush=True)
        policy, _ = load_policy_from_ckpt(ckpt, device=device)
        row = score_checkpoint(policy, loader, device)
        scatter = row.pop("scatter")
        row["iter"] = it
        row["ckpt"] = str(ckpt)
        print(
            f"  spearman epo-gt {row['spearman_mean']['epo_gt']:.3f}  "
            f"epo-head {row['spearman_mean']['epo_head']:.3f}  "
            f"head-gt {row['spearman_mean']['head_gt']:.3f}",
            flush=True,
        )
        saved.append({**row, "scatter": scatter})
        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _plot(saved)
    payload = []
    for row in saved:
        item = {k: v for k, v in row.items() if k != "scatter"}
        payload.append(item)
    (OUT / "correlation.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}", flush=True)
    ds.close()


if __name__ == "__main__":
    main()
