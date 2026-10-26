"""Open-loop evaluation: minADE / minFDE / miss rate / buffered IoU."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from womd_il.data.dataset import WOMDPackedDataset, packed_collate
from womd_il.data.schema import PackConfig
from womd_il.metrics.open_loop import format_open_loop_table, open_loop_metrics
from womd_il.models.factory import build_policy
from womd_il.utils import config_dir, load_checkpoint, resolve_device


@torch.no_grad()
def evaluate(cfg: DictConfig) -> dict[str, float]:
    device = resolve_device(cfg.device)
    data_cfg = PackConfig.from_cfg(cfg.data)
    path = cfg.data.val_file if cfg.split == "val" else cfg.data.train_file
    ds = WOMDPackedDataset(
        path,
        target_mode=cfg.get("target_mode", cfg.data.target_mode),
        max_scenarios=cfg.max_scenarios,
        cfg=data_cfg,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=packed_collate)
    policy = build_policy(cfg.model, cfg.decoder).to(device)
    load_checkpoint(cfg.ckpt, policy, map_location=device)
    policy.eval()

    acc: dict[str, list[float]] = {}
    for batch in tqdm(loader, desc="open-loop"):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        bundle = policy.infer(batch)
        xy = bundle.xyh[..., :2]
        m = open_loop_metrics(
            xy,
            bundle.scores,
            batch["gt_xy"],
            batch["gt_valid"],
            horizons=tuple(int(h) for h in cfg.horizons),
            miss_lat=tuple(cfg.miss_lat) if cfg.get("miss_lat") is not None else None,
            miss_lon=tuple(cfg.miss_lon) if cfg.get("miss_lon") is not None else None,
            iou_buffers=tuple(cfg.get("iou_buffers", (0.5, 1.0, 2.0))),
            last_xy_vel=batch["sdc_hist"][:, -1, 2:4],
            num_keep=int(getattr(cfg.decoder, "num_keep", 6)),
        )
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
    metrics = {k: float(np.mean(v)) for k, v in acc.items()}
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(format_open_loop_table(metrics))
    return metrics


@hydra.main(version_base=None, config_path=str(config_dir()), config_name="eval/open_loop")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    evaluate(cfg)


if __name__ == "__main__":
    main()
