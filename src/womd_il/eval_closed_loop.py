"""Closed-loop evaluation: drive the SDC, replay or IDM the rest."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from womd_il.data.dataset import WOMDPackedDataset
from womd_il.data.schema import PackConfig
from womd_il.metrics.closed_loop import closed_loop_metrics
from womd_il.models.factory import build_policy
from womd_il.sim.rollout import rollout_policy
from womd_il.utils import config_dir, load_checkpoint, resolve_device
from womd_il.viz.bev import save_bev


def evaluate(cfg: DictConfig) -> dict:
    device = resolve_device(cfg.device)
    data_cfg = PackConfig.from_cfg(cfg.data)
    path = cfg.data.val_file if cfg.split == "val" else cfg.data.train_file
    ds = WOMDPackedDataset(
        path,
        target_mode="sdc",
        max_scenarios=cfg.max_scenarios,
        cfg=data_cfg,
    )
    policy = build_policy(cfg.model, cfg.decoder).to(device)
    load_checkpoint(cfg.ckpt, policy, map_location=device)

    h_exec = int(cfg.get("h_exec", 1))
    mode_select = str(cfg.get("mode_select", "argmax"))
    cost_cfg = {}
    raw_cost = cfg.get("cost")
    if raw_cost is not None:
        cost_cfg = OmegaConf.to_container(raw_cost, resolve=True) or {}
        if not isinstance(cost_cfg, dict):
            cost_cfg = {}
    rows = []
    viz_dir = Path(cfg.viz_dir) if cfg.viz_dir else None
    n = min(len(ds), int(cfg.max_scenarios) if cfg.max_scenarios else len(ds))
    for i in tqdm(range(n), desc=f"closed-loop[{cfg.neighbors}]"):
        sample = ds.get_world_sample(i)
        state = rollout_policy(
            policy,
            sample,
            neighbors=str(cfg.neighbors),
            horizon=int(cfg.horizon),
            device=device,
            h_exec=h_exec,
            mode_select=mode_select,
            cost=cost_cfg,
        )
        m = closed_loop_metrics(state, sample)
        m["scenario_id"] = sample.get("scenario_id", str(i))
        rows.append(m)
        if viz_dir is not None and (i % int(cfg.viz_every) == 0):
            save_bev(
                viz_dir / f"{i:04d}_{cfg.neighbors}.png",
                sample,
                executed_xy=np.asarray(state.executed_xy),
                title=f"{m['scenario_id']} {cfg.neighbors}",
            )
    keys = [k for k in rows[0] if k != "scenario_id"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    payload = {"summary": summary, "per_scenario": rows, "neighbors": str(cfg.neighbors)}
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    return payload


@hydra.main(version_base=None, config_path=str(config_dir()), config_name="eval/closed_loop")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    evaluate(cfg)


if __name__ == "__main__":
    main()
