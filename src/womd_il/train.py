"""Train a Wayformer policy with a swappable decoder."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from womd_il.data.dataset import WOMDPackedDataset, packed_collate
from womd_il.data.metric_cache import resolve_metric_cache, vocab_sha
from womd_il.data.subsample import class_cap_indices, load_pdms_winners, load_winners
from womd_il.data.windows import load_window_table
from womd_il.data.schema import PackConfig
from womd_il.losses.hydra import (
    format_metric_table,
    format_pdms_table,
    gather_or_compute_metric_labels,
    hydra_metric_eval,
    hydra_pdms_eval,
    pdms_from_probs,
)
from womd_il.metrics.open_loop import DEFAULT_HORIZONS, format_open_loop_table, open_loop_metrics
from womd_il.models.factory import build_policy
from womd_il.report import run_end_of_training_report
from womd_il.utils import (
    apply_torch_compile,
    config_dir,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
)


def _maybe_wandb(cfg: DictConfig):
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed; continuing without logging")
        return None
    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=OmegaConf.to_container(cfg, True))
    return wandb


def _run_open_loop(
    policy,
    loader,
    device,
    max_batches: int = 20,
    iou_buffers: tuple[float, ...] = (0.5, 1.0, 2.0),
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, float]:
    policy.eval()
    acc: dict[str, list[float]] = {}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            bundle = policy.infer(batch)
            xy = bundle.xyh[..., :2]
            m = open_loop_metrics(
                xy,
                bundle.scores,
                batch["gt_xy"],
                batch["gt_valid"],
                horizons=horizons,
                iou_buffers=iou_buffers,
                last_xy_vel=batch["sdc_hist"][:, -1, 2:4],
            )
            for k, v in m.items():
                acc.setdefault(k, []).append(v)
    policy.train()
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _run_metric_heads(policy, loader, device, max_batches: int = 20) -> dict[str, float]:
    """Val metric-head BCE/acc and PDMS rank of the Hydra vocab head."""
    decoder = getattr(policy, "decoder", None)
    if decoder is None:
        return {}
    has_metric = getattr(decoder, "metric_head", None) is not None
    main_pdms = str(getattr(decoder, "main", "")) == "pdms"
    if not has_metric and not main_pdms:
        return {}
    policy.eval()
    acc: dict[str, list[float]] = {}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            bundle = policy.infer(batch)
            logits = bundle.extras.get("metric_logits")
            idx = bundle.extras.get("stage_index")
            if has_metric and logits is not None and idx is not None:
                labels = gather_or_compute_metric_labels(
                    decoder.vocab[..., :2][idx],
                    batch,
                    idx,
                    pred_yaw=decoder.vocab[..., 2][idx],
                )
                for k, v in hydra_metric_eval(logits, labels).items():
                    acc.setdefault(k, []).append(v)
            cached = batch.get("metric_labels")
            if cached is not None:
                pdms = pdms_from_probs(cached.float())
                # Rank each policy head separately so mix/epo runs are comparable.
                for key, prefix in (
                    ("im_logits", "pdms"),
                    ("epo_logits", "pdms_epo"),
                ):
                    score_logits = bundle.extras.get(key)
                    if score_logits is None:
                        continue
                    filled = score_logits.masked_fill(~torch.isfinite(score_logits), -1e9)
                    for k, v in hydra_pdms_eval(filled, pdms).items():
                        # Keep legacy pdms/* for the imitation head; epo gets pdms_epo/*.
                        out_k = k if prefix == "pdms" else k.replace("pdms/", f"{prefix}/", 1)
                        acc.setdefault(out_k, []).append(v)
    policy.train()
    return {k: float(np.mean(v)) for k, v in acc.items()} if acc else {}


def should_early_stop(
    *,
    improved: bool,
    stale: int,
    epoch: int,
    patience: int,
    min_epochs: int,
) -> tuple[int, bool]:
    """Count consecutive non-improving val epochs; stop after ``patience``.

    ``patience=0`` disables. ``min_epochs`` is a floor so the first noisy
    vals cannot halt the run.
    """
    if int(patience) <= 0:
        return 0, False
    stale_n = 0 if improved else int(stale) + 1
    stop = (int(epoch) + 1) >= int(min_epochs) and stale_n >= int(patience)
    return stale_n, stop


def _early_stop_cfg(cfg: DictConfig) -> tuple[int, int, float]:
    raw = cfg.get("early_stop") or {}
    return (
        int(raw.get("patience", 0) or 0),
        int(raw.get("min_epochs", 0) or 0),
        float(raw.get("min_delta", 0.0) or 0.0),
    )


def _maybe_class_subsample(cfg: DictConfig, train_ds: WOMDPackedDataset):
    """Keep at most ``decoder.subsample_cap`` scenarios for each vocab class."""
    cap = cfg.decoder.get("subsample_cap", None)
    if cap is None:
        return train_ds
    cap_i = int(cap)
    if str(cfg.data.target_mode) != "sdc":
        raise ValueError(
            "decoder.subsample_cap indexes scenarios in order, which matches "
            "target_mode=sdc. Unset subsample_cap for other target modes."
        )
    vocab_path = cfg.decoder.get("vocab_path")
    if not vocab_path or not Path(str(vocab_path)).exists():
        raise FileNotFoundError(
            f"decoder.subsample_cap={cap_i} needs decoder.vocab_path, missing {vocab_path}"
        )
    main = str(cfg.decoder.get("main", "pdms")).lower()
    # Reward-driven objectives (pdms / epo / mix) balance by PDMS winners;
    # pure imitation still uses ADE winners.
    if main in ("pdms", "epo", "mix"):
        cache = cfg.decoder.get("winner_cache", "data/artifacts/hydra_winner_pdms.npz")
        metric_h5 = resolve_metric_cache(cfg.decoder.get("metric_cache"), cfg.data.train_file)
        if metric_h5 is None:
            raise FileNotFoundError(
                f"decoder.main={main} class subsample needs decoder.metric_cache "
                "(run scripts/cache_hydra_metrics.py)."
            )
        winners = load_pdms_winners(metric_h5, vocab_path, cache)
        cap_tag = "PDMS"
    else:
        cache = cfg.decoder.get("winner_cache", "data/artifacts/hydra_winner.npz")
        winners = load_winners(
            cfg.data.train_file,
            vocab_path,
            cache,
            front_weight=float(cfg.decoder.get("front_weight", 1.0)),
            temperature=float(cfg.decoder.get("temperature", 1.0)),
        )
        cap_tag = "ADE"
    n = train_ds.n_scenarios
    if len(winners) < n:
        raise ValueError(
            f"Winner cache has {len(winners)} rows but the train set has {n} scenarios"
        )
    idx = class_cap_indices(winners[:n], cap_i, seed=int(cfg.seed))
    print(f"Hydra {cap_tag} class subsample cap={cap_i}: {len(idx)}/{n} scenarios")
    return Subset(train_ds, idx.tolist())


def train(cfg: DictConfig) -> None:
    device = resolve_device(cfg.device)
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    data_cfg = PackConfig.from_cfg(cfg.data)
    cache_root = cfg.decoder.get("metric_cache")
    vocab_path = cfg.decoder.get("vocab_path")
    sha = vocab_sha(vocab_path) if vocab_path and Path(str(vocab_path)).exists() else None
    raw_k = cfg.decoder.get("vocab_size")
    k_vocab = int(raw_k) if raw_k else None
    train_cache = resolve_metric_cache(cache_root, cfg.data.train_file)
    val_cache = resolve_metric_cache(cache_root, cfg.data.val_file)
    if train_cache is not None:
        print(f"Hydra metric cache (train): {train_cache}")
    elif cache_root:
        print(
            f"Hydra metric cache missing under {cache_root} "
            "(run scripts/cache_hydra_metrics.py). Occupancy labels only for K<=256."
        )
    windows = None
    win_winners = None
    win_path = cfg.decoder.get("windows_path")
    if win_path and Path(str(win_path)).exists() and str(cfg.data.target_mode) == "sdc":
        windows, win_winners = load_window_table(win_path)
        print(f"Hydra train windows: {len(windows)} from {win_path}")
    train_ds = WOMDPackedDataset(
        cfg.data.train_file,
        target_mode=cfg.data.target_mode,
        max_scenarios=cfg.data.max_train_scenarios,
        cfg=data_cfg,
        augment=True,
        perturb=cfg.data.get("perturb"),
        metric_cache=train_cache,
        vocab_sha=sha,
        vocab_size=k_vocab,
        windows=windows,
    )
    val_path = Path(cfg.data.val_file)
    val_ds = None
    val_loader = None
    if val_path.exists():
        val_ds = WOMDPackedDataset(
            val_path,
            target_mode=cfg.data.target_mode,
            max_scenarios=cfg.data.max_val_scenarios,
            cfg=data_cfg,
            metric_cache=val_cache,
            vocab_sha=sha,
            vocab_size=k_vocab,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=packed_collate,
            num_workers=int(cfg.num_workers),
        )
    if windows is not None:
        cap = cfg.decoder.get("subsample_cap", None)
        if cap is not None and win_winners is not None:
            idx = class_cap_indices(win_winners, int(cap), seed=int(cfg.seed))
            print(f"Hydra window class subsample cap={int(cap)}: {len(idx)}/{len(windows)}")
            if len(idx) < len(windows):
                train_ds = Subset(train_ds, idx.tolist())
    else:
        train_ds = _maybe_class_subsample(cfg, train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=packed_collate,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )

    # Mission signal = on-route lane corridor in map_feat (not preferred-path XY tokens).
    _probe = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    _n_probe = min(32, len(_probe))
    _on_ok = 0
    for _i in range(_n_probe):
        _feat = _probe[_i]
        if float(np.asarray(_feat["map_feat"][..., -1]).sum()) >= 1.0:
            _on_ok += 1
    _use_rt = bool(cfg.model.get("use_route_tokens", False))
    print(
        f"Mission probe: map_on_route_bit={_on_ok}/{_n_probe}  "
        f"use_route_tokens={_use_rt}"
    )
    if _on_ok < int(0.4 * _n_probe):
        raise RuntimeError(
            f"Training map_feat on_route bit empty ({_on_ok}/{_n_probe}). "
            "Repack so H5 contains map_on_route."
        )

    policy = build_policy(cfg.model, cfg.decoder).to(device)
    policy = apply_torch_compile(policy, cfg)
    base_lr = float(cfg.lr)
    min_lr = float(cfg.get("min_lr", 1e-6))
    opt = torch.optim.AdamW(policy.parameters(), lr=base_lr, weight_decay=float(cfg.weight_decay))
    n_epochs = int(cfg.epochs)
    use_amp = bool(cfg.amp) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    wb = _maybe_wandb(cfg)

    global_step = 0
    start_epoch = 0
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Reward-driven Hydra runs checkpoint on val PDMS of the active policy head.
    _main = str(cfg.decoder.get("main", "pdms")).lower()
    _policy_head = str(cfg.decoder.get("policy_head", "imitation")).lower()
    if str(cfg.decoder.get("name", "")) == "hydra" and _main in ("pdms", "epo", "mix"):
        if _main == "epo" or (_main == "mix" and _policy_head == "epo"):
            pdms_key = "pdms_epo/selected"
        else:
            pdms_key = "pdms/selected"
        pdms_best = True
    else:
        pdms_key = "pdms/selected"
        pdms_best = False
    best = -float("inf") if pdms_best else float("inf")
    collapse_factor = float(cfg.get("collapse_factor", 2.5))
    es_patience, es_min_epochs, es_min_delta = _early_stop_cfg(cfg)
    stale = 0

    resume_raw = cfg.get("resume", None)
    resume_path: Path | None = None
    if resume_raw is True or str(resume_raw).lower() in ("true", "1", "yes", "last"):
        resume_path = ckpt_dir / "last.pt"
    elif resume_raw:
        resume_path = Path(str(resume_raw))
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        try:
            resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            resume_ckpt = torch.load(resume_path, map_location=device)
        load_checkpoint(resume_path, policy, map_location=device)
        global_step = int(resume_ckpt.get("step", 0) or 0)
        # ``epoch`` in the ckpt is the last finished epoch index.
        start_epoch = int(resume_ckpt.get("epoch", -1) or -1) + 1
        if "best" in resume_ckpt and resume_ckpt["best"] is not None:
            best = float(resume_ckpt["best"])
        best_path = ckpt_dir / "best.pt"
        if best_path.exists() and (
            best == float("inf") or best == -float("inf")
        ):
            try:
                best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            except TypeError:
                best_ckpt = torch.load(best_path, map_location="cpu")
            if "best" in best_ckpt and best_ckpt["best"] is not None:
                best = float(best_ckpt["best"])
            elif isinstance(best_ckpt.get("val"), dict):
                val = best_ckpt["val"]
                if pdms_best and pdms_key in val:
                    best = float(val[pdms_key])
                elif pdms_best and "pdms/selected" in val:
                    best = float(val["pdms/selected"])
                else:
                    best = float(
                        val.get(
                            "argmaxADE@80",
                            val.get("minADE", val.get("minADE@80", best)),
                        )
                    )
        if "stale" in resume_ckpt and resume_ckpt["stale"] is not None:
            stale = int(resume_ckpt["stale"])
        print(
            f"resume {resume_path} → epoch={start_epoch}/{n_epochs} "
            f"step={global_step} best={best:.4g} stale={stale}"
        )

    # Align cosine schedule with completed epochs (opt state is not restored).
    for g in opt.param_groups:
        g.setdefault("initial_lr", base_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(n_epochs, 1),
        eta_min=min_lr,
        last_epoch=start_epoch - 1,
    )
    print(f"train lr={base_lr:g} cosine→{min_lr:g} batch={cfg.batch_size} epochs={n_epochs}")
    print(f"current lr={opt.param_groups[0]['lr']:.3e} (after resume align)")
    if pdms_best:
        print(f"best.pt tracks val {pdms_key} (higher is better), not minADE")
    else:
        print("best.pt tracks val argmaxADE@80 (executed mode), not NMS minADE")
    if es_patience > 0:
        print(
            f"early stop patience={es_patience} min_epochs={es_min_epochs} "
            f"min_delta={es_min_delta:g}"
        )
    if start_epoch >= n_epochs:
        print(f"resume start_epoch={start_epoch} >= epochs={n_epochs}; nothing to train")
        return

    for epoch in range(start_epoch, n_epochs):
        policy.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                stats = policy.train_step(batch)
                loss = stats["loss"]
            # NaN: skip without touching GradScaler. Inf: still backward so AMP
            # can drop the scale (calling scaler.update() here crashes).
            if torch.isnan(loss.detach()):
                print(f"skip NaN loss at step {global_step}")
                del stats, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.grad_clip))
            scaler.step(opt)
            scaler.update()
            global_step += 1
            logs = {
                k: float(v.detach().cpu())
                for k, v in stats.items()
                if torch.is_tensor(v) and v.ndim == 0
            }
            postfix = {
                "loss": f"{logs.get('loss', float(loss.detach())):.3f}",
                "lr": f"{opt.param_groups[0]['lr']:.1e}",
            }
            if "pdms_pred" in logs:
                postfix["pdms"] = f"{logs['pdms_pred']:.3f}"
            if "loss_ade" in logs:
                postfix["ade"] = f"{logs['loss_ade']:.2f}"
            # Imitation KL(q||p); per-stage LDiP KL is `im4096`/…
            if "loss_im" in logs:
                postfix["im"] = f"{logs['loss_im']:.3f}"
            for key in sorted(
                (k for k in logs if k.startswith("loss_im@")),
                key=lambda k: -int(k.split("@", 1)[1]),
            ):
                postfix["im" + key.split("@", 1)[1]] = f"{logs[key]:.2f}"
            # EPO normalized regret (and optional raw advantage).
            if "loss_epo" in logs:
                postfix["epo"] = f"{logs['loss_epo']:.3f}"
            for key in sorted(
                (k for k in logs if k.startswith("loss_epo@")),
                key=lambda k: -int(k.split("@", 1)[1]),
            ):
                postfix["epo" + key.split("@", 1)[1]] = f"{logs[key]:.2f}"
            if "epo_regret" in logs:
                postfix["reg"] = f"{logs['epo_regret']:.3f}"
            if "epo_entropy" in logs:
                postfix["H"] = f"{logs['epo_entropy']:.2f}"
            if "epo_ent_pen" in logs:
                postfix["Hpen"] = f"{logs['epo_ent_pen']:.3f}"
            if "epo_adv" in logs:
                postfix["adv"] = f"{logs['epo_adv']:.2f}"
            if "hit@256" in logs:
                postfix["h256"] = f"{logs['hit@256']:.2f}"
            if "hit@16" in logs:
                postfix["h16"] = f"{logs['hit@16']:.2f}"
            if "loss_pdms" in logs:
                postfix["s"] = f"{logs['loss_pdms']:.3f}"
            if "loss_metric" in logs:
                postfix["m"] = f"{logs['loss_metric']:.3f}"
            for name in ("nc", "dac", "ddc", "tlc", "ttc", "lk", "hc", "progress"):
                key = f"loss_{name}"
                if key in logs:
                    postfix[name] = f"{logs[key]:.2f}"
            pbar.set_postfix(**postfix)
            if global_step % int(cfg.log_every) == 0 and wb:
                wb.log({f"train/{k}": v for k, v in logs.items()}, step=global_step)
            if cfg.max_train_steps and global_step >= int(cfg.max_train_steps):
                break
        save_checkpoint(
            ckpt_dir / "last.pt",
            policy,
            cfg,
            extra={"step": global_step, "epoch": epoch, "best": best, "stale": stale},
        )
        if val_loader is not None and (epoch + 1) % int(cfg.val_every) == 0:
            metrics = _run_open_loop(
                policy,
                val_loader,
                device,
                iou_buffers=tuple(cfg.get("iou_buffers", (0.5, 1.0, 2.0))),
                horizons=tuple(int(h) for h in cfg.get("horizons", DEFAULT_HORIZONS)),
            )
            # Closed-loop executes argmax, not NMS-6 minADE.
            ade = metrics.get(
                "argmaxADE@80", metrics.get("minADE", metrics.get("minADE@80", 1e9))
            )
            print(format_open_loop_table(metrics, epoch=epoch, lr=opt.param_groups[0]["lr"]))
            head_metrics = _run_metric_heads(policy, val_loader, device)
            if head_metrics:
                if any(k.startswith("metric/") for k in head_metrics):
                    print(format_metric_table(head_metrics, epoch=epoch, lr=opt.param_groups[0]["lr"]))
                if "pdms/selected" in head_metrics or "pdms_epo/selected" in head_metrics:
                    print(format_pdms_table(head_metrics, epoch=epoch, lr=opt.param_groups[0]["lr"]))
                metrics.update(head_metrics)
            if wb:
                wb.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)
            improved = False
            if pdms_best and pdms_key in metrics:
                score = float(metrics[pdms_key])
                if score > best + es_min_delta:
                    best = score
                    improved = True
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        policy,
                        cfg,
                        extra={
                            "step": global_step,
                            "epoch": epoch,
                            "val": metrics,
                            "best": best,
                            "stale": 0,
                        },
                    )
                elif best > -float("inf") and score < best / max(collapse_factor, 1.0):
                    best_path = ckpt_dir / "best.pt"
                    if best_path.exists():
                        new_lr = max(opt.param_groups[0]["lr"] * 0.5, min_lr)
                        print(
                            f"val {pdms_key} collapsed {score:.3f} vs best {best:.3f}; "
                            f"restoring {best_path} and lr→{new_lr:.2e}"
                        )
                        load_checkpoint(best_path, policy, map_location=device)
                        opt = torch.optim.AdamW(
                            policy.parameters(), lr=new_lr, weight_decay=float(cfg.weight_decay)
                        )
                        left = max(n_epochs - epoch - 1, 1)
                        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                            opt, T_max=left, eta_min=min_lr
                        )
            elif not pdms_best:
                if ade < best - es_min_delta:
                    best = ade
                    improved = True
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        policy,
                        cfg,
                        extra={
                            "step": global_step,
                            "epoch": epoch,
                            "val": metrics,
                            "best": best,
                            "stale": 0,
                        },
                    )
                elif best < float("inf") and ade > collapse_factor * best:
                    best_path = ckpt_dir / "best.pt"
                    if best_path.exists():
                        new_lr = max(opt.param_groups[0]["lr"] * 0.5, min_lr)
                        print(
                            f"val collapsed {ade:.3f} vs best {best:.3f}; "
                            f"restoring {best_path} and lr→{new_lr:.2e}"
                        )
                        load_checkpoint(best_path, policy, map_location=device)
                        opt = torch.optim.AdamW(
                            policy.parameters(), lr=new_lr, weight_decay=float(cfg.weight_decay)
                        )
                        left = max(n_epochs - epoch - 1, 1)
                        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                            opt, T_max=left, eta_min=min_lr
                        )
            stale, stop = should_early_stop(
                improved=improved,
                stale=stale,
                epoch=epoch,
                patience=es_patience,
                min_epochs=es_min_epochs,
            )
            if stop:
                print(
                    f"early stop at epoch={epoch} (patience={es_patience}, "
                    f"best={pdms_key + ' ' + f'{best:.3f}' if pdms_best else 'minADE ' + f'{best:.3f}'})"
                )
                break
        sched.step()
        if cfg.max_train_steps and global_step >= int(cfg.max_train_steps):
            break
    save_checkpoint(ckpt_dir / "last.pt", policy, cfg, extra={"step": global_step})
    named = ckpt_dir / f"{policy.decoder_name}.pt"
    save_checkpoint(named, policy, cfg, extra={"step": global_step})

    report_cfg = cfg.get("report") or {}
    max_scenarios = int(report_cfg.get("max_scenarios", 64))
    viz_n = int(report_cfg.get("viz_n", 8))
    # Skip closed-loop + videos when both are disabled (e.g. memory probes).
    if max_scenarios > 0 or viz_n > 0:
        report_ds = val_ds if val_ds is not None else train_ds
        eval_policy = policy
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            load_checkpoint(best_path, eval_policy, map_location=device)
            print(f"end-of-training report uses {best_path}")
        else:
            print("end-of-training report uses last weights")
        neighbors = list(report_cfg.get("neighbors", ["log", "idm"]))
        run_end_of_training_report(
            eval_policy,
            report_ds,
            out_dir=ckpt_dir,
            device=device,
            max_scenarios=max_scenarios,
            viz_n=viz_n,
            neighbors=neighbors,
            horizon=int(report_cfg.get("horizon", 80)),
            wandb_run=wb,
        )
    else:
        print("end-of-training report skipped (report.max_scenarios=0 and report.viz_n=0)")
    if wb:
        wb.finish()


@hydra.main(version_base=None, config_path=str(config_dir()), config_name="train")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
