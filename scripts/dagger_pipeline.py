#!/usr/bin/env python3
"""Offline DAgger loop: eval → collect far-from-log → PDMS cache → fine-tune."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from womd_il.data.dagger import (
    cache_dagger_metrics,
    collect_dagger_pool,
    pool_iter_extent,
    pool_len,
)
from womd_il.data.dataset import WOMDPackedDataset, packed_collate
from womd_il.data.schema import PackConfig
from womd_il.metrics.open_loop import open_loop_metrics
from womd_il.report import closed_loop_eval, print_closed_loop_report
from womd_il.sim.rollout import rollout_policy
from womd_il.train import train
from womd_il.utils import config_dir, load_policy_from_ckpt, resolve_device
from womd_il.viz.video import render_closed_loop_video


def _load_suite(path: Path) -> list[tuple[int, str, str]]:
    rows = json.loads(Path(path).read_text())
    out = []
    for r in rows:
        out.append((int(r["idx"]), str(r["tag"]), str(r.get("sid") or "")))
    return out


def _append_history(path: Path, row: dict) -> None:
    """Replace the row for this iter if it already exists (restart-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"iters": []}
    if path.exists():
        data = json.loads(path.read_text())
        if isinstance(data, list):
            data = {"iters": data}
    iters = [r for r in data.get("iters", []) if int(r.get("iter", -1)) != int(row["iter"])]
    iters.append(row)
    iters.sort(key=lambda r: int(r.get("iter", 0)))
    path.write_text(json.dumps({"iters": iters}, indent=2, default=float))


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"iter": 0, "phase": "start"}
    return json.loads(path.read_text())


def _save_state(path: Path, iter_id: int, phase: str) -> None:
    path.write_text(json.dumps({"iter": int(iter_id), "phase": str(phase)}, indent=2))


def _iter_phases(iter_id: int) -> tuple[str, ...]:
    # Iter 0 evaluates the init ckpt then collects. Later iters train first.
    if int(iter_id) == 0:
        return ("eval", "collect")
    return ("train", "eval", "collect")


def _phase_done(state: dict, iter_id: int, phase: str) -> bool:
    """True if ``state`` is at or past ``(iter_id, phase)`` in that iter's order."""
    si = int(state.get("iter", -1))
    if si > iter_id:
        return True
    if si < iter_id:
        return False
    order = _iter_phases(iter_id)
    cur = str(state.get("phase", ""))
    if cur not in order or phase not in order:
        return False
    return order.index(cur) >= order.index(phase)


def _free_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def eval_open_loop(policy, h5: Path, *, device, max_scenarios: int, batch_size: int = 32) -> dict:
    from torch.utils.data import DataLoader

    ds = WOMDPackedDataset(h5, target_mode="sdc", max_scenarios=max_scenarios, cfg=PackConfig())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=packed_collate)
    acc: dict[str, list[float]] = {}
    for batch in tqdm(loader, desc="open-loop"):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        bundle = policy.infer(batch)
        m = open_loop_metrics(
            bundle.xyh[..., :2],
            bundle.scores,
            batch["gt_xy"],
            batch["gt_valid"],
            last_xy_vel=batch["sdc_hist"][:, -1, 2:4],
        )
        for k, v in m.items():
            acc.setdefault(k, []).append(float(v))
    ds.close()
    return {k: float(np.mean(vs)) for k, vs in acc.items()}


def render_diversity(
    policy,
    *,
    h5: Path,
    suite: list[tuple[int, str, str]],
    out_dir: Path,
    device,
    h_exec: int,
    mode_select: str,
    horizon: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = WOMDPackedDataset(h5, target_mode="sdc", cfg=PackConfig())
    rows = []
    lines = []
    for j, (idx, tag, expect_sid) in enumerate(tqdm(suite, desc="diversity")):
        if idx >= len(ds):
            print(f"WARN skip idx={idx} (val has {len(ds)})", flush=True)
            continue
        sample = ds.get_world_sample(idx)
        sid = str(sample.get("scenario_id", idx)).replace("/", "_")
        if expect_sid and expect_sid not in sid:
            print(f"WARN idx={idx}: sid={sid} expected~{expect_sid}", flush=True)
        state = rollout_policy(
            policy,
            sample,
            neighbors="log",
            horizon=horizon,
            device=device,
        h_exec=h_exec,
        mode_select=mode_select,
        early_stop_on_fail=False,
    )
        from womd_il.metrics.closed_loop import closed_loop_metrics

        m = closed_loop_metrics(state, sample)
        m["scenario_index"] = idx
        m["tag"] = tag
        m["scenario_id"] = sid
        rows.append(m)
        stem = out_dir / f"{j:04d}_{tag}_i{idx}_{sid}_closed_log"
        path = render_closed_loop_video(
            stem,
            sample,
            state,
            title=f"{tag} i={idx} sel={mode_select} h_exec={h_exec}",
        )
        line = (
            f"{j:04d}\t{tag}\tidx={idx}\t{sid}\t"
            f"coll={int(m['collision'] > 0.5)}\toff={int(m['offroad'] > 0.5)}\t"
            f"ade={m['log_ade']:.3f}\t-> {path}"
        )
        lines.append(line)
        print(line, flush=True)
    ds.close()
    summary = {
        k: float(np.mean([r[k] for r in rows]))
        for k in ("collision", "offroad", "progress_ratio", "log_ade")
        if rows
    }
    summary["n"] = float(len(rows))
    (out_dir / "manifest.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps({"summary": summary, "per_scenario": rows}, indent=2, default=float))
    return summary


def run_iter_eval(cfg: DictConfig, ckpt: Path, iter_dir: Path, iter_id: int, pool_size: int) -> dict:
    device = resolve_device(cfg.device)
    policy, _ = load_policy_from_ckpt(ckpt, device=device)
    policy.eval()
    ev = cfg.eval
    ol = eval_open_loop(
        policy,
        Path(cfg.data.val_file),
        device=device,
        max_scenarios=int(ev.ol_max_scenarios),
        batch_size=int(cfg.batch_size),
    )
    (iter_dir / "metrics_ol.json").write_text(json.dumps(ol, indent=2))

    val_ds = WOMDPackedDataset(
        cfg.data.val_file,
        target_mode="sdc",
        max_scenarios=int(ev.cl_max_scenarios),
        cfg=PackConfig.from_cfg(cfg.data),
    )
    cl = closed_loop_eval(
        policy,
        val_ds,
        neighbors=str(ev.neighbors),
        horizon=int(ev.horizon),
        device=device,
        max_scenarios=int(ev.cl_max_scenarios),
        h_exec=int(ev.h_exec),
        mode_select=str(ev.mode_select),
        early_stop_on_fail=bool(ev.get("early_stop_on_fail", False)),
    )
    print_closed_loop_report(cl)
    (iter_dir / "metrics_cl.json").write_text(json.dumps(cl, indent=2, default=float))
    val_ds.close()

    suite = _load_suite(Path(str(ev.diversity_suite)))
    div = render_diversity(
        policy,
        h5=Path(cfg.data.val_file),
        suite=suite,
        out_dir=iter_dir / "videos_diverse",
        device=device,
        h_exec=int(ev.h_exec),
        mode_select=str(ev.get("diversity_select", ev.mode_select)),
        horizon=int(ev.horizon),
    )

    row = {
        "iter": iter_id,
        "ckpt": str(ckpt),
        "ol": ol,
        "cl": cl["summary"],
        "diversity": div,
        "pool_size": pool_size,
    }
    del policy
    _free_cuda()
    return row


def run_collect_and_cache(cfg: DictConfig, ckpt: Path, iter_id: int, pool: Path, metrics: Path) -> dict:
    device = resolve_device(cfg.device)
    policy, _ = load_policy_from_ckpt(ckpt, device=device)
    policy.eval()
    col = cfg.collect
    stats = collect_dagger_pool(
        policy,
        cfg.data.train_file,
        pool,
        iter_id=iter_id,
        device=device,
        max_scenarios=col.get("max_scenarios"),
        neighbors=str(col.neighbors),
        horizon=int(col.horizon),
        h_exec=int(col.h_exec),
        mode_select=str(col.mode_select),
        log_ade_thresh=float(col.log_ade_thresh),
        max_per_scenario=int(col.max_per_scenario),
        pdms_horizon=int(col.pdms_horizon),
        min_valid_frac=float(col.min_valid_frac),
        save_stride=int(col.get("save_stride", 1)),
        flush_every=int(col.get("flush_every", 100)),
        infer_batch=int(col.get("infer_batch", 8)),
    )
    print(f"collect iter={iter_id}: {json.dumps(stats)}", flush=True)
    del policy
    _free_cuda()
    n_pool, _ = pool_iter_extent(pool, iter_id) if pool.exists() else (0, -1)
    if pool_len(pool) > 0:
        cache_dagger_metrics(
            cfg.data.train_file,
            pool,
            cfg.decoder.vocab_path,
            metrics,
            batch_size=int(cfg.cache.batch_size),
            num_workers=int(cfg.cache.num_workers),
            device=str(cfg.device),
        )
        print(f"PDMS cache → {metrics} (N={pool_len(pool)} this_iter={n_pool})", flush=True)
    return stats


def run_train_iter(cfg: DictConfig, iter_dir: Path, warm: Path, pool: Path, metrics: Path) -> Path:
    tcfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    # Drop pipeline-only blocks that train() ignores safely, keep dagger hooks.
    tcfg.ckpt_dir = str(iter_dir)
    tcfg.warm_start = str(warm)
    tcfg.resume = None
    tcfg.dagger_pool = str(pool)
    tcfg.dagger_metric_cache = str(metrics)
    tcfg.dagger_pool_frac = float(cfg.get("dagger_pool_frac", 0.3))
    tcfg.report = {"max_scenarios": 0, "viz_n": 0, "neighbors": ["log"], "horizon": 80}
    train(tcfg)
    _free_cuda()
    best = iter_dir / "best.pt"
    if not best.exists():
        best = iter_dir / "last.pt"
    return best


@hydra.main(version_base=None, config_path=str(config_dir()), config_name="dagger")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    root = Path(str(cfg.ckpt_root))
    root.mkdir(parents=True, exist_ok=True)
    pool = root / "pool.h5"
    metrics = root / "metrics_pool.h5"
    history_path = root / "history.json"
    init = Path(str(cfg.init_ckpt))
    if not init.exists():
        raise FileNotFoundError(init)

    # Seed iter0 directory with a copy of the init checkpoint for bookkeeping.
    iter0 = root / "iter00"
    iter0.mkdir(parents=True, exist_ok=True)
    ckpt = iter0 / "best.pt"
    if not ckpt.exists():
        shutil.copy2(init, ckpt)

    current = ckpt
    n_iters = int(cfg.n_iters)
    state_path = root / "state.json"
    state = _load_state(state_path)
    print(f"resume state={state}", flush=True)

    def _plot() -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "plot_dagger_curves", Path(__file__).resolve().parent / "plot_dagger_curves.py"
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        if history_path.exists():
            mod.plot_dagger(mod.load_history(history_path), root / "dagger_curves.png")

    for it in range(0, n_iters + 1):
        iter_dir = root / f"iter{it:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        if it > 0:
            prev = root / f"iter{it - 1:02d}" / "best.pt"
            if not prev.exists():
                prev = root / f"iter{it - 1:02d}" / "last.pt"
            current = prev
            if not _phase_done(state, it, "train"):
                current = run_train_iter(cfg, iter_dir, current, pool, metrics)
                _save_state(state_path, it, "train")
                state = _load_state(state_path)
            else:
                trained = iter_dir / "best.pt"
                current = trained if trained.exists() else iter_dir / "last.pt"
                print(f"skip train iter={it} (state={state})", flush=True)

        if not _phase_done(state, it, "eval"):
            row = run_iter_eval(cfg, current, iter_dir, it, pool_len(pool))
            _append_history(history_path, row)
            _plot()
            _save_state(state_path, it, "eval")
            state = _load_state(state_path)
        else:
            print(f"skip eval iter={it} (state={state})", flush=True)

        if it >= n_iters:
            break
        if not _phase_done(state, it, "collect"):
            run_collect_and_cache(cfg, current, it, pool, metrics)
            _save_state(state_path, it, "collect")
            state = _load_state(state_path)
        else:
            print(f"skip collect iter={it} (state={state})", flush=True)

    print(f"DONE dagger root={root} history={history_path}", flush=True)


if __name__ == "__main__":
    main()
