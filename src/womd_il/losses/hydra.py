"""Soft cross-entropy of Hydra scores toward nearest vocabulary trajectories."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from womd_il import constants as C
from womd_il.losses.overlap import horizon_weights, overlap_rate

# NAVSIM v2 / LDiP EPDMS channels we can label from packed WOMD (no two-plan EC).
METRIC_NAMES = ("nc", "dac", "ddc", "tlc", "ttc", "lk", "hc", "progress")
METRIC_INDEX = {name: i for i, name in enumerate(METRIC_NAMES)}
_MAX_K_METRIC = 256
_HIT_T = 40
_PDM_T = 40  # 4 s @ 10 Hz, same horizon as NAVSIM / LDiP
_K_CHUNK = 256
_P_CHUNK = 4096
_Q_CHUNK = 16384
_STOPPED_SPEED = 0.05
# PDM TTC: constant-velocity / heading probes at 0.3 s over a 0.9 s bound.
_TTC_OFFSET_STEPS = (0, 3, 6, 9)
# NAVSIM v1 PDMS: NC * DAC * (5 TTC + 5 EP + 2 C) / 12.
_PDMS_TTC_W = 5.0
_PDMS_EP_W = 5.0
_PDMS_COMFORT_W = 2.0
_PDMS_LK_W = 2.0
_PDMS_HC_W = 2.0
_PDMS_DEN = 12.0
_EPDMS_DEN_NO_EC = 14.0  # 5 TTC + 5 EP + 2 LK + 2 HC (EC omitted)
_MAP_STRIDE = 2  # map queries at 5 Hz; 4 s still has 20 samples
_LK_LATERAL_M = 1.0  # NAVSIM is 0.5 m; WOMD polylines are coarser
_LK_SECONDS = 2.0
_LK_IN_LANE_M = 4.0
_DDC_OK_M = 2.0
_DDC_HALF_M = 6.0
_DDC_WINDOW = 10  # 1 s
_TLC_RADIUS_M = 4.0
_TLC_PAST_M = 2.0
_EP_REF_M = 5.0
# Route-progress: also require the projection to stay near the route polyline.
_EP_ROUTE_LAT_M = 6.0
# DAC: fail if any (strided) plan point is farther than this from a driving lane.
# Catches "left the road through a gap" that edge-crossing alone misses.
# ~1 lane width; beyond this the plan is off the driving surface even if it
# never crosses a digitized road-edge (gaps / open boundaries).
_DAC_MAX_LANE_M = 4.0
_COMFORT_YAWRATE = 1.5
_RED_TL = (C.TL_ARROW_STOP, C.TL_STOP, C.TL_FLASHING_STOP)
_LANE_TYPES = (C.MAP_LANE_FREEWAY, C.MAP_LANE_SURFACE)
_EDGE_CHUNK = 512
# Drop map segments whose AABB is outside the plan cloud (+ pad). Cuts ~4–5× work.
_MAP_BBOX_PAD_M = 15.0
# DAC intersection: modes × edge-segs tile (keeps peak GPU memory bounded).
_DAC_K_CHUNK = 512


def _ch(probs: Tensor, name: str) -> Tensor:
    return probs[..., METRIC_INDEX[name]]


def pdms_from_probs(probs: Tensor) -> Tensor:
    """PDMS / EPDMS-without-EC from metric channels.

    ``[..., 5]`` keeps NAVSIM v1: NC×DAC×(5 TTC + 5 EP + 2 C)/12.
    ``[..., 8]`` is LDiP/NAVSIM v2 without EC:
    NC×DAC×DDC×TLC×(5 TTC + 5 EP + 2 LK + 2 HC)/14.
    """
    n = int(probs.size(-1))
    if n == 5:
        nc, dac, ttc, comfort, progress = probs.unbind(-1)
        weighted = (
            _PDMS_TTC_W * ttc + _PDMS_EP_W * progress + _PDMS_COMFORT_W * comfort
        ) / _PDMS_DEN
        return nc * dac * weighted
    if n != len(METRIC_NAMES):
        raise ValueError(f"expected 5 or {len(METRIC_NAMES)} metric channels, got {n}")
    mult = _ch(probs, "nc") * _ch(probs, "dac") * _ch(probs, "ddc") * _ch(probs, "tlc")
    weighted = (
        _PDMS_TTC_W * _ch(probs, "ttc")
        + _PDMS_EP_W * _ch(probs, "progress")
        + _PDMS_LK_W * _ch(probs, "lk")
        + _PDMS_HC_W * _ch(probs, "hc")
    ) / _EPDMS_DEN_NO_EC
    return mult * weighted


def pdms_v1_from_probs(probs: Tensor) -> Tensor:
    """NAVSIM v1 PDMS using NC, DAC, TTC, comfort/HC, EP (ignores DDC/TLC/LK)."""
    n = int(probs.size(-1))
    if n == 5:
        return pdms_from_probs(probs)
    comfort = _ch(probs, "hc")
    weighted = (
        _PDMS_TTC_W * _ch(probs, "ttc")
        + _PDMS_EP_W * _ch(probs, "progress")
        + _PDMS_COMFORT_W * comfort
    ) / _PDMS_DEN
    return _ch(probs, "nc") * _ch(probs, "dac") * weighted


def pdms_from_metric_logits(metric_logits: Tensor) -> Tensor:
    return pdms_from_probs(metric_logits.sigmoid())


def hydra_pdms_loss(
    logits: Tensor,
    pdms: Tensor,
    temperature: float = 1.0,
    soft_k: int | None = 16,
    bce_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Train a vocab head to match NAVSIM v1 PDMS (no EC / two-plan terms).

    Ranking: soft CE toward a PDMS-weighted distribution over remaining modes.
    Calibration: BCE-with-logits vs the scalar PDMS in ``[0, 1]``.
    """
    pdms = pdms.float().clamp(0.0, 1.0)
    logits = logits.float()
    tau = max(float(temperature), 1e-3)
    k = logits.size(-1)
    k_soft = 0 if soft_k is None else min(int(soft_k), k)
    if k_soft > 0 and k_soft < k:
        vals, idx = pdms.topk(k_soft, dim=-1)
        target = torch.zeros_like(pdms).scatter(-1, idx, F.softmax(vals / tau, dim=-1))
        winner = idx.gather(-1, vals.argmax(dim=-1, keepdim=True)).squeeze(-1)
    else:
        target = F.softmax(pdms / tau, dim=-1)
        winner = pdms.argmax(dim=-1)
    log_p = F.log_softmax(logits, dim=-1)
    ce = -(target * log_p).sum(dim=-1).mean()
    bce = F.binary_cross_entropy_with_logits(logits, pdms)
    loss = ce + float(bce_weight) * bce
    pred = logits.argmax(dim=-1)
    b = pred.size(0)
    row = torch.arange(b, device=logits.device)
    return {
        "loss": loss,
        "loss_pdms": loss.detach(),
        "loss_pdms_ce": ce.detach(),
        "loss_pdms_bce": bce.detach(),
        "loss_im": loss.detach(),
        "winner": winner,
        "pdms": pdms,
        "pdms_pred": pdms[row, pred].mean().detach(),
        "pdms_oracle": pdms[row, winner].mean().detach(),
    }


def hydra_epo_loss(
    logits: Tensor,
    pdms: Tensor,
    *,
    entropy_coef: float = 0.2,
    temperature: float = 1.0,
    entropy_min_k: float = 256.0,
) -> dict[str, Tensor]:
    """Exhaustive Policy Optimization (ZTRS): maximize ``E_π[Ψ]`` over the vocab.

    ``Ψ`` is the per-sample z-score of rule PDMS / EPDMS over ``K`` (detached).
    With ``π = softmax(logits / τ)``, the regret term's ``∇_π`` matches paper
    Eq. 7 up to a positive per-sample scale.

    We minimize **normalized regret**
    ``(max Ψ - E_π[Ψ]) / max(max Ψ, ε) ∈ [0, 1]``.

    Entropy is **not** maximized globally. A collapse floor
    ``H_floor = ln(entropy_min_k)`` triggers a squared penalty
    ``coef * relu(H_floor - H(π))^2`` only when effective support drops below
    ``entropy_min_k`` modes; above the floor the entropy term is zero so π
    may sharpen freely. Default ``τ=1`` (no temperature softening).

    Comfort correction ``b`` is omitted here (optional later).
    """
    pdms = pdms.float()
    logits = logits.float()
    if pdms.shape != logits.shape:
        raise ValueError(f"pdms {tuple(pdms.shape)} must match logits {tuple(logits.shape)}")
    mu = pdms.mean(dim=-1, keepdim=True)
    std = pdms.std(dim=-1, keepdim=True).clamp_min(1e-6)
    psi = ((pdms - mu) / std).detach()
    tau = max(float(temperature), 1e-3)
    log_pi = F.log_softmax(logits / tau, dim=-1)
    pi = log_pi.exp()
    adv = (pi * psi).sum(dim=-1)
    psi_max = psi.max(dim=-1).values
    # Non-negative, ≤1; denominator constant w.r.t. π → same ∇ direction as -adv.
    regret = (psi_max - adv) / psi_max.clamp_min(1e-6)
    ent = -(pi * log_pi).sum(dim=-1)
    coef = float(entropy_coef)
    k_floor = max(float(entropy_min_k), 1.0)
    h_floor = math.log(k_floor)
    # Only punish collapse below ~entropy_min_k effective modes; square the deficit.
    ent_deficit = (h_floor - ent).clamp_min(0.0)
    ent_pen = coef * (ent_deficit * ent_deficit)
    loss = regret.mean() + ent_pen.mean()
    winner = pdms.argmax(dim=-1)
    pred = logits.argmax(dim=-1)
    b = pred.size(0)
    row = torch.arange(b, device=logits.device)
    return {
        "loss": loss,
        "loss_epo": loss.detach(),
        "loss_epo_regret": regret.mean().detach(),
        "loss_epo_ent": ent.mean().detach(),
        "loss_epo_ent_pen": ent_pen.mean().detach(),
        "loss_im": loss.detach(),
        "epo_adv": adv.mean().detach(),
        "epo_regret": regret.mean().detach(),
        "epo_entropy": ent.mean().detach(),
        "epo_ent_pen": ent_pen.mean().detach(),
        "winner": winner,
        "pdms": pdms,
        "pdms_pred": pdms[row, pred].mean().detach(),
        "pdms_oracle": pdms[row, winner].mean().detach(),
        # Placeholders so train logging that expects ADE keys stays quiet.
        "loss_ade": pdms.new_zeros(()),
        "ade": pdms,
    }


def hydra_pdms_eval(score_logits: Tensor, pdms: Tensor) -> dict[str, float]:
    """Oracle PDMS of the predicted argmax vs the true PDMS argmax."""
    pdms = pdms.float()
    pred = score_logits.argmax(dim=-1)
    oracle = pdms.argmax(dim=-1)
    row = torch.arange(pred.size(0), device=pred.device)
    return {
        "pdms/selected": float(pdms[row, pred].mean().detach().cpu()),
        "pdms/oracle": float(pdms[row, oracle].mean().detach().cpu()),
        "pdms/top1": float((pred == oracle).float().mean().detach().cpu()),
        "pdms/mean": float(pdms.mean().detach().cpu()),
        "pdms/pred_mean": float(score_logits.sigmoid().mean().detach().cpu()),
    }


def hydra_imitation_loss(
    logits: Tensor,
    vocab_xy: Tensor,
    gt_xy: Tensor,
    gt_valid: Tensor,
    temperature: float = 1.0,
    soft_k: int | None = 16,
    front_weight: float = 2.0,
    collision_weight: float = 8.0,
    other_xy: Tensor | None = None,
    other_valid: Tensor | None = None,
    ego_lw: Tensor | None = None,
    other_lw: Tensor | None = None,
    pdms: Tensor | None = None,
    pdms_scale: float = 0.0,
    force_idx: Tensor | None = None,
    distance: str = "ade",
    gt_yaw: Tensor | None = None,
    imi_times: tuple[int, ...] | list[int] | None = None,
) -> dict[str, Tensor]:
    """
    logits: [B, K]
    vocab_xy: [K, T, 2|3] or per-sample remaining set [B, K, T, 2|3]
    gt_xy: [B, T, 2]
    gt_valid: [B, T]

    ``distance='sse'`` is Hydra-MDP Eq. 9 / LDiP Eq. 5:
    ``q ∝ exp(-||v - a||_2^2 / σ)`` over the remaining set, using every
    valid ``(x, y, heading)`` step of the action chunk (``σ=0.5`` in yaml).
    ``imi_times`` is an optional subsample; leave it ``None`` for the papers.
    ``distance='ade'`` is the older mean-Euclidean kernel.

    With ``pdms_scale > 0`` the soft target is
    ``q ∝ PDMS^{pdms_scale} * exp(-dist / τ)``. Partial PDMS in ``(0, 1)``
    scales the kernel continuously (0.5 has half the mass of 1.0 at the same
    distance). ``pdms_scale=0`` is vanilla distance.
    """
    gt_xy = gt_xy.float()
    vocab_xy = vocab_xy.float()
    logits = logits.float()
    if vocab_xy.dim() == 3:
        vocab_b = vocab_xy.unsqueeze(0).expand(gt_xy.size(0), -1, -1, -1)
    elif vocab_xy.dim() == 4:
        vocab_b = vocab_xy
    else:
        raise ValueError(f"vocab_xy must be [K,T,C] or [B,K,T,C], got {tuple(vocab_xy.shape)}")
    if vocab_b.size(-1) not in (2, 3):
        raise ValueError(f"vocab last dim must be 2 or 3, got {vocab_b.size(-1)}")
    _, k, t, _ = vocab_b.shape
    vocab_xy2 = vocab_b[..., :2]
    valid = gt_valid.float().unsqueeze(1)  # [B,1,T]
    tw = horizon_weights(gt_xy.size(1), front_weight, gt_xy.device, gt_xy.dtype)
    tw_sum = (valid * tw).sum(dim=-1).clamp_min(1.0)
    diff = gt_xy.unsqueeze(1) - vocab_xy2  # [B,K,T,2]
    sq_xy = diff.pow(2).sum(-1)
    dist = sq_xy.sqrt()
    ade = (dist * valid * tw).sum(-1) / tw_sum  # [B,K]
    kind = str(distance or "ade").lower()
    if kind not in ("ade", "sse", "flatten_l2"):
        raise ValueError(f"distance must be 'ade', 'sse', or 'flatten_l2', got {distance!r}")
    if kind == "flatten_l2":
        t_use = min(int(gt_xy.size(1)), int(vocab_xy2.size(2)))
        g = gt_xy[:, :t_use].reshape(gt_xy.size(0), -1)
        v = vocab_xy2[:, :, :t_use].reshape(gt_xy.size(0), k, -1)
        dist_term = torch.linalg.vector_norm(g.unsqueeze(1) - v, dim=-1)
    elif kind == "sse":
        pose_v = vocab_xy2
        pose_g = gt_xy
        yaw_v = vocab_b[..., 2] if vocab_b.size(-1) == 3 else None
        yaw_g = gt_yaw.float() if gt_yaw is not None else None
        valid_t = valid
        if imi_times:
            keep = [int(i) for i in imi_times if 0 <= int(i) < t]
            if keep:
                idx = torch.tensor(keep, device=vocab_b.device, dtype=torch.long)
                pose_v = pose_v.index_select(-2, idx)
                pose_g = pose_g.index_select(-2, idx)
                valid_t = valid.index_select(-1, idx)
                if yaw_v is not None:
                    yaw_v = yaw_v.index_select(-1, idx)
                if yaw_g is not None:
                    yaw_g = yaw_g.index_select(-1, idx)
        sq = (pose_g.unsqueeze(1) - pose_v).pow(2).sum(-1)
        if yaw_v is not None and yaw_g is not None:
            sq = sq + (yaw_v - yaw_g.unsqueeze(1)).pow(2)
        dist_term = (sq * valid_t).sum(dim=-1)
    else:
        dist_term = ade
    tau = max(float(temperature), 1e-3)
    scale = float(pdms_scale)
    log_pdms = torch.zeros_like(ade)
    if pdms is not None and scale > 0:
        if tuple(pdms.shape[-1:]) != (k,):
            raise ValueError(f"pdms must be [B, K={k}], got {tuple(pdms.shape)}")
        log_pdms = scale * pdms.float().clamp(0.0, 1.0).clamp_min(1e-6).log()
    rank = log_pdms - dist_term / tau
    k_soft = 0 if soft_k is None else min(int(soft_k), k)
    use_k = k_soft > 0 and k_soft < k
    if other_xy is not None and other_xy.size(-2) != t:
        other_xy = other_xy[..., :t, :]
    if other_valid is not None and other_valid.size(-1) != t:
        other_valid = other_valid[..., :t]
    collide = (
        collision_weight > 0
        and other_xy is not None
        and other_valid is not None
        and ego_lw is not None
        and other_lw is not None
    )
    if use_k:
        vals, idx = rank.topk(k_soft, dim=-1, largest=True)
        if force_idx is not None:
            fi = force_idx.long()
            missing = ~(idx == fi.unsqueeze(-1)).any(dim=-1)
            if missing.any():
                idx = idx.clone()
                vals = vals.clone()
                idx[missing, -1] = fi[missing]
                vals[missing, -1] = rank[missing, fi[missing]]
        if collide:
            sel = vocab_xy2.gather(1, idx[:, :, None, None].expand(-1, -1, t, 2))
            ov = overlap_rate(sel, other_xy, other_valid, ego_lw, other_lw, pred_valid=gt_valid)
            dist_s = dist_term.gather(1, idx) + float(collision_weight) * ov
            vals = log_pdms.gather(1, idx) - dist_s / tau
        target = torch.zeros_like(ade).scatter(-1, idx, F.softmax(vals, dim=-1))
        winner = idx.gather(-1, vals.argmax(dim=-1, keepdim=True)).squeeze(-1)
    else:
        if collide:
            ov = overlap_rate(vocab_xy2, other_xy, other_valid, ego_lw, other_lw, pred_valid=gt_valid)
            rank = log_pdms - (dist_term + float(collision_weight) * ov) / tau
        target = F.softmax(rank, dim=-1)
        winner = rank.argmax(dim=-1)
    # Soft CE = KL(q||p) + H(q). With soft_k=null, H(q) is large and dominates
    # the scalar, so CE does not read as “classifier quality”. Train & log KL;
    # gradients match CE because H(q) does not depend on logits.
    log_p = F.log_softmax(logits, dim=-1)
    log_q = target.clamp_min(1e-12).log()
    ce = -(target * log_p).sum(dim=-1)
    ent = -(target * log_q).sum(dim=-1)
    kl = ce - ent
    loss = kl.mean()
    return {
        "loss": loss,
        "loss_im": loss.detach(),
        "loss_im_ce": ce.mean().detach(),
        "loss_im_H": ent.mean().detach(),
        "loss_ade": ade.min(dim=-1).values.mean().detach(),
        "winner": winner,
        "ade": ade,
    }


def vocab_ade_winner(
    vocab_xy: Tensor,
    gt_xy: Tensor,
    gt_valid: Tensor,
    front_weight: float = 1.0,
) -> Tensor:
    """Global vocab index of the min-ADE trajectory. [B]."""
    if vocab_xy.dim() != 3:
        raise ValueError(f"vocab_xy must be [K,T,2], got {tuple(vocab_xy.shape)}")
    gt_xy = gt_xy.float()
    vocab_b = vocab_xy.float().unsqueeze(0).expand(gt_xy.size(0), -1, -1, -1)
    valid = gt_valid.float().unsqueeze(1)
    tw = horizon_weights(gt_xy.size(1), front_weight, gt_xy.device, gt_xy.dtype)
    tw_sum = (valid * tw).sum(dim=-1).clamp_min(1.0)
    dist = (gt_xy.unsqueeze(1) - vocab_b).pow(2).sum(-1).sqrt()
    ade = (dist * valid * tw).sum(dim=-1) / tw_sum
    return ade.argmin(dim=-1)


def _balanced_class_weights(
    pos_rate: Tensor, cap: float | None = None
) -> tuple[Tensor, Tensor]:
    """Inverse-frequency weights so P(y=0) and P(y=1) contribute equally.

    ``w1 = 1/(2p)``, ``w0 = 1/(2(1-p))`` with ``p = P(y=1)``. Labels stay
    1=good / 0=bad; rare failures (usually y=0) get the large weight. Optional
    ``cap`` clamps each of ``w0``/``w1``.
    """
    p = pos_rate.float().clamp(1e-4, 1.0 - 1e-4)
    w1 = 0.5 / p
    w0 = 0.5 / (1.0 - p)
    if cap is not None and float(cap) > 0:
        w1 = w1.clamp(max=float(cap))
        w0 = w0.clamp(max=float(cap))
    return w0, w1


def _metric_sample_weight(
    labels: Tensor,
    *,
    pos_rate: Tensor | None = None,
    pos_weight_cap: float | None = None,
) -> Tensor:
    """Per-example balanced weight matching ``labels`` shape ``[..., M]``."""
    y = labels.float()
    rate = (
        pos_rate.float()
        if pos_rate is not None
        else y.mean(dim=tuple(range(y.ndim - 1)))
    )
    w0, w1 = _balanced_class_weights(rate, pos_weight_cap)
    while w0.ndim < y.ndim:
        w0 = w0.unsqueeze(0)
        w1 = w1.unsqueeze(0)
    return torch.where(y >= 0.5, w1, w0)


def hydra_metric_bce(
    head_logits: Tensor,
    labels: Tensor,
    *,
    class_weighted: bool = True,
    pos_weight_cap: float | None = None,
    pos_rate: Tensor | None = None,
) -> Tensor:
    """Balanced BCE on metric heads. ``head_logits``/``labels``: [B, K, M].

    When ``class_weighted``, each example is scaled by inverse class frequency
    (``w1=1/(2p)``, ``w0=1/(2(1-p))``). ``pos_rate`` is an optional [M] prior
    (e.g. per-stage EMA); otherwise the batch mean is used.
    """
    y = labels.float()
    if not class_weighted:
        return F.binary_cross_entropy_with_logits(head_logits, y)
    weight = _metric_sample_weight(y, pos_rate=pos_rate, pos_weight_cap=pos_weight_cap)
    loss = F.binary_cross_entropy_with_logits(head_logits, y, reduction="none")
    return (loss * weight).mean()


def hydra_metric_bce_per_head(
    head_logits: Tensor,
    labels: Tensor,
    *,
    class_weighted: bool = True,
    pos_weight_cap: float | None = None,
    pos_rate: Tensor | None = None,
) -> Tensor:
    """Mean BCE per metric channel. Returns [M]."""
    y = labels.float()
    loss = F.binary_cross_entropy_with_logits(head_logits, y, reduction="none")
    if class_weighted:
        weight = _metric_sample_weight(y, pos_rate=pos_rate, pos_weight_cap=pos_weight_cap)
        loss = loss * weight
    return loss.mean(dim=tuple(range(loss.ndim - 1)))


def hydra_metric_eval(head_logits: Tensor, labels: Tensor) -> dict[str, float]:
    """Per-head BCE / accuracy / F1 / MCC / positive rates on a val batch."""
    labels = labels.float()
    logits = head_logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=(0, 1))
    pred = (logits.sigmoid() >= 0.5).float()
    y_bin = (labels >= 0.5).float()
    acc = (pred == y_bin).float().mean(dim=(0, 1))
    pos = labels.mean(dim=(0, 1))
    pred_pos = pred.mean(dim=(0, 1))
    tp = (pred * y_bin).sum(dim=(0, 1))
    fp = (pred * (1.0 - y_bin)).sum(dim=(0, 1))
    fn = ((1.0 - pred) * y_bin).sum(dim=(0, 1))
    tn = ((1.0 - pred) * (1.0 - y_bin)).sum(dim=(0, 1))
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
    # Matthews correlation: uses all four confusion cells; ≈0 for chance /
    # always-majority under imbalance (unlike accuracy / lone F1).
    denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).clamp_min(1e-6)
    mcc = (tp * tn - fp * fn) / denom
    out: dict[str, float] = {
        "metric/bce": float(bce.mean().detach().cpu()),
        "metric/acc": float(acc.mean().detach().cpu()),
        "metric/macro_f1": float(f1.mean().detach().cpu()),
        "metric/macro_mcc": float(mcc.mean().detach().cpu()),
    }
    n = min(int(bce.numel()), len(METRIC_NAMES))
    for i in range(n):
        name = METRIC_NAMES[i]
        out[f"metric/{name}/bce"] = float(bce[i].detach().cpu())
        out[f"metric/{name}/acc"] = float(acc[i].detach().cpu())
        out[f"metric/{name}/f1"] = float(f1[i].detach().cpu())
        out[f"metric/{name}/mcc"] = float(mcc[i].detach().cpu())
        out[f"metric/{name}/label+"] = float(pos[i].detach().cpu())
        out[f"metric/{name}/pred+"] = float(pred_pos[i].detach().cpu())
    return out


def format_metric_table(
    metrics: dict[str, float],
    *,
    epoch: int | None = None,
    lr: float | None = None,
) -> str:
    """Pretty-print Hydra metric-head BCE / acc / F1 / MCC / class balance."""
    parts = ["val metrics"]
    if epoch is not None:
        parts.append(f"epoch={epoch}")
    if lr is not None:
        parts.append(f"lr={lr:.2e}")
    header = (
        f"{'metric':>10}  {'BCE':>8}  {'acc':>8}  {'F1':>8}  {'MCC':>8}  "
        f"{'label+':>8}  {'pred+':>8}"
    )
    lines = [" ".join(parts), header, "-" * len(header)]
    names = [n for n in METRIC_NAMES if f"metric/{n}/bce" in metrics]
    for name in names:
        lines.append(
            f"{name:>10}  {metrics[f'metric/{name}/bce']:8.3f}  "
            f"{metrics[f'metric/{name}/acc']:8.3f}  "
            f"{metrics.get(f'metric/{name}/f1', float('nan')):8.3f}  "
            f"{metrics.get(f'metric/{name}/mcc', float('nan')):8.3f}  "
            f"{metrics[f'metric/{name}/label+']:8.3f}  "
            f"{metrics[f'metric/{name}/pred+']:8.3f}"
        )
    if "metric/bce" in metrics:
        lines.append(
            f"{'mean':>10}  {metrics['metric/bce']:8.3f}  "
            f"{metrics.get('metric/acc', float('nan')):8.3f}  "
            f"{metrics.get('metric/macro_f1', float('nan')):8.3f}  "
            f"{metrics.get('metric/macro_mcc', float('nan')):8.3f}"
        )
    return "\n".join(lines)


def format_pdms_table(
    metrics: dict[str, float],
    *,
    epoch: int | None = None,
    lr: float | None = None,
) -> str:
    parts = ["val PDMS rank"]
    if epoch is not None:
        parts.append(f"epoch={epoch}")
    if lr is not None:
        parts.append(f"lr={lr:.2e}")
    lines = [" ".join(parts)]
    for label, prefix in (("im_head", "pdms"), ("epo_head", "pdms_epo")):
        sel = metrics.get(f"{prefix}/selected")
        if sel is None:
            continue
        ora = metrics.get(f"{prefix}/oracle", float("nan"))
        top = metrics.get(f"{prefix}/top1", float("nan"))
        lines.append(
            f"  {label:8s}  selected={sel:.3f}  oracle={ora:.3f}  top1={top:.3f}"
        )
    if len(lines) == 1:
        return lines[0] + "  (no pdms keys)"
    return "\n".join(lines)


def _circle_hits(
    pred_xy: Tensor,
    other_xy: Tensor,
    other_valid: Tensor,
    ego_lw: Tensor,
    other_lw: Tensor,
) -> Tensor:
    """Boolean [B, K, T] — any neighbor circle overlap at that step."""
    r_ego = 0.5 * torch.linalg.vector_norm(ego_lw.float(), dim=-1).clamp_min(0.5)
    r_oth = 0.5 * torch.linalg.vector_norm(other_lw.float(), dim=-1).clamp_min(0.5)
    thresh = r_ego[:, None, None, None] + r_oth[:, None, :, None]
    parts = []
    k = pred_xy.size(1)
    for i in range(0, k, _K_CHUNK):
        sl = pred_xy[:, i : i + _K_CHUNK]
        delta = sl.unsqueeze(2) - other_xy.unsqueeze(1)
        dist = torch.linalg.vector_norm(delta, dim=-1)
        hit = (dist < thresh) & other_valid[:, None]
        parts.append(hit.any(dim=2))
    return torch.cat(parts, dim=1)


def _obb_hits(
    pred_xy: Tensor,
    pred_yaw: Tensor,
    other_xy: Tensor,
    other_valid: Tensor,
    ego_lw: Tensor,
    other_lw: Tensor,
    *,
    at_fault: bool = True,
) -> Tensor:
    """Boolean [B, K, T] oriented-box overlap (same AABB-in-ego-frame as the sim).

    With ``at_fault``, a hit behind the ego rear bumper is ignored (PDM rear-end).
    """
    half_e = 0.5 * ego_lw.float().clamp_min(0.2)
    half_o = 0.5 * other_lw.float().clamp_min(0.2)
    hx = half_e[:, 0, None, None, None] + half_o[:, None, :, 0, None]
    hy = half_e[:, 1, None, None, None] + half_o[:, None, :, 1, None]
    rear = half_e[:, 0, None, None]
    parts = []
    k = pred_xy.size(1)
    for i in range(0, k, _K_CHUNK):
        sl = pred_xy[:, i : i + _K_CHUNK]
        yaw = pred_yaw[:, i : i + _K_CHUNK]
        c = torch.cos(-yaw)
        s = torch.sin(-yaw)
        delta = other_xy.unsqueeze(1) - sl.unsqueeze(2)
        lx = c.unsqueeze(2) * delta[..., 0] - s.unsqueeze(2) * delta[..., 1]
        ly = s.unsqueeze(2) * delta[..., 0] + c.unsqueeze(2) * delta[..., 1]
        hit = (lx.abs() < hx) & (ly.abs() < hy) & other_valid[:, None]
        if at_fault:
            hit = hit & (lx >= -rear.unsqueeze(1))
        parts.append(hit.any(dim=2))
    return torch.cat(parts, dim=1)


def _plan_speed_yaw(pred_xy: Tensor, pred_yaw: Tensor | None, dt: float) -> tuple[Tensor, Tensor]:
    dxy = pred_xy[:, :, 1:] - pred_xy[:, :, :-1]
    step = torch.linalg.vector_norm(dxy.float(), dim=-1)
    speed = torch.cat([step[:, :, :1], step], dim=-1) / dt
    if pred_yaw is not None:
        return speed, pred_yaw.float()
    yaw = torch.atan2(dxy[..., 1], dxy[..., 0])
    yaw = torch.cat([yaw[:, :, :1], yaw], dim=-1)
    return speed, yaw


def _timestep_hits(
    pred_xy: Tensor,
    other_xy: Tensor,
    other_valid: Tensor,
    ego_lw: Tensor,
    other_lw: Tensor,
    pred_yaw: Tensor | None = None,
    *,
    at_fault: bool = True,
) -> Tensor:
    """Boolean [B, K, T] occupancy vs logged neighbors (boxes if yaw is given)."""
    t = min(int(pred_xy.size(2)), int(other_xy.size(2)), _HIT_T)
    pred_xy = pred_xy[:, :, :t].float()
    other_xy = other_xy[:, :, :t].float()
    other_valid = other_valid[:, :, :t].bool()
    if pred_yaw is None:
        return _circle_hits(pred_xy, other_xy, other_valid, ego_lw, other_lw)
    return _obb_hits(
        pred_xy,
        pred_yaw[:, :, :t].float(),
        other_xy,
        other_valid,
        ego_lw,
        other_lw,
        at_fault=at_fault,
    )


def _ttc_clear(
    pred_xy: Tensor,
    pred_yaw: Tensor | None,
    other_xy: Tensor,
    other_valid: Tensor,
    ego_lw: Tensor,
    other_lw: Tensor,
    dt: float = C.DT,
) -> Tensor:
    """1 if no CV box (or circle) hit within 0.9 s at any plan step. [B, K]."""
    speed, yaw = _plan_speed_yaw(pred_xy.float(), pred_yaw, dt)
    moved = (speed >= _STOPPED_SPEED)
    direction = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    t_plan = pred_xy.size(2)
    t_oth = other_xy.size(2)
    any_hit: Tensor | None = None
    for off in _TTC_OFFSET_STEPS:
        t_use = min(t_plan, t_oth - off, _HIT_T)
        if t_use <= 0:
            continue
        ego_p = pred_xy[:, :, :t_use].float() + direction[:, :, :t_use] * (
            speed[:, :, :t_use].unsqueeze(-1) * (off * dt)
        )
        oth_p = other_xy[:, :, off : off + t_use].float()
        oth_v = other_valid[:, :, off : off + t_use].bool()
        hit = _timestep_hits(ego_p, oth_p, oth_v, ego_lw, other_lw, yaw[:, :, :t_use])
        hit = hit & moved[:, :, :t_use]
        reduced = hit.any(dim=-1)
        any_hit = reduced if any_hit is None else any_hit | reduced
    if any_hit is None:
        return torch.ones(pred_xy.size(0), pred_xy.size(1), device=pred_xy.device, dtype=torch.float32)
    return (~any_hit).float()


def _comfort_ok(xy: Tensor, dt: float = C.DT) -> Tensor:
    """1 if accel / yaw-rate on ``xy`` [B,K,T,2] stay inside bounds.

    Yaw-rate from finite-differenced xy is noise while stopped, so it is
    ignored below 1 m/s. A single 10 Hz spike (logged WOMD, not PDM-tracked)
    does not fail; two consecutive frames above the bound do.
    """
    if xy.size(2) < 3:
        return torch.ones(xy.size(0), xy.size(1), device=xy.device, dtype=torch.float32)
    dxy = xy[:, :, 1:] - xy[:, :, :-1]
    speed = torch.linalg.vector_norm(dxy.float(), dim=-1) / dt
    acc = (speed[..., 1:] - speed[..., :-1]) / dt
    heading = torch.atan2(dxy[..., 1], dxy[..., 0])
    dyaw = torch.atan2(
        torch.sin(heading[..., 1:] - heading[..., :-1]),
        torch.cos(heading[..., 1:] - heading[..., :-1]),
    )
    yawrate = dyaw.abs() / dt
    moving = torch.minimum(speed[..., 1:], speed[..., :-1]) >= 1.0
    yawrate = yawrate.masked_fill(~moving, 0.0)
    bad_a = acc.abs() >= C.ACCEL_CLIP
    bad_w = yawrate >= _COMFORT_YAWRATE
    fail_a = (bad_a[..., 1:] & bad_a[..., :-1]).any(dim=-1) if bad_a.size(-1) > 1 else bad_a.any(dim=-1)
    fail_w = (bad_w[..., 1:] & bad_w[..., :-1]).any(dim=-1) if bad_w.size(-1) > 1 else bad_w.any(dim=-1)
    return (~(fail_a | fail_w)).float()


def _history_comfort(vocab_xy: Tensor, batch: dict[str, Tensor]) -> Tensor:
    """HC: comfort on (valid history ≳1 s) stitched onto the plan."""
    hist = batch.get("sdc_hist")
    if hist is None:
        return _comfort_ok(vocab_xy)
    hist_xy = hist[..., :2].float()
    mask = batch.get("sdc_hist_mask")
    if mask is not None:
        mask = mask.bool()
        filled = hist_xy.clone()
        for i in range(hist_xy.size(1) - 2, -1, -1):
            use = mask[:, i]
            filled[:, i] = torch.where(use[:, None], filled[:, i], filled[:, i + 1])
        hist_xy = filled
    b, k, _, _ = vocab_xy.shape
    hist_k = hist_xy[:, None].expand(b, k, -1, -1)
    return _comfort_ok(torch.cat([hist_k, vocab_xy.float()], dim=2))


def _type_mask(map_type: Tensor, types: tuple[int, ...]) -> Tensor:
    want = torch.zeros_like(map_type, dtype=torch.bool)
    for t in types:
        want = want | (map_type == int(t))
    return want


def _compact_segments(
    map_xy: Tensor, map_pt_valid: Tensor, map_type: Tensor, types: tuple[int, ...]
) -> tuple[Tensor, Tensor, Tensor]:
    """Valid segments for one scene: ``a, b, unit_dir`` each [N, 2]."""
    want = _type_mask(map_type, types)
    a = map_xy[:, :-1].float()
    b = map_xy[:, 1:].float()
    ok = map_pt_valid[:, :-1].bool() & map_pt_valid[:, 1:].bool() & want[:, None]
    a, b = a[ok], b[ok]
    ab = b - a
    nrm = torch.linalg.vector_norm(ab, dim=-1, keepdim=True).clamp_min(1e-6)
    return a, b, ab / nrm


def _plan_aabb(xy: Tensor, pad: float = _MAP_BBOX_PAD_M) -> tuple[float, float, float, float]:
    """Axis-aligned pad around plan points. ``xy`` [..., 2]."""
    flat = xy.reshape(-1, 2).float()
    mn = flat.min(dim=0).values
    mx = flat.max(dim=0).values
    return (
        float(mn[0] - pad),
        float(mx[0] + pad),
        float(mn[1] - pad),
        float(mx[1] + pad),
    )


def _filter_segments_aabb(
    a: Tensor, b: Tensor, direc: Tensor, xmin: float, xmax: float, ymin: float, ymax: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Keep segments whose AABB intersects the query box."""
    if a.size(0) == 0:
        return a, b, direc
    sx0 = torch.minimum(a[:, 0], b[:, 0])
    sx1 = torch.maximum(a[:, 0], b[:, 0])
    sy0 = torch.minimum(a[:, 1], b[:, 1])
    sy1 = torch.maximum(a[:, 1], b[:, 1])
    keep = (sx1 >= xmin) & (sx0 <= xmax) & (sy1 >= ymin) & (sy0 <= ymax)
    return a[keep], b[keep], direc[keep]


def _nearest_segments(q: Tensor, a: Tensor, b: Tensor, direc: Tensor) -> tuple[Tensor, Tensor]:
    """Point-to-segment nearest. ``q`` [Q,2], segments [N,2] → dist [Q], dir [Q,2]."""
    qn = q.size(0)
    device, dtype = q.device, q.dtype
    if a.size(0) == 0:
        return torch.full((qn,), 1e6, device=device, dtype=dtype), torch.zeros(qn, 2, device=device, dtype=dtype)
    # Track squared distance; sqrt once at the end.
    min_d2 = torch.full((qn,), 1e12, device=device, dtype=dtype)
    best = torch.zeros(qn, 2, device=device, dtype=dtype)
    ab = b - a
    ab2 = (ab * ab).sum(-1).clamp_min(1e-6)
    n = a.size(0)
    for j in range(0, n, _P_CHUNK):
        aj, abj, ab2j, dj = a[j : j + _P_CHUNK], ab[j : j + _P_CHUNK], ab2[j : j + _P_CHUNK], direc[j : j + _P_CHUNK]
        for i in range(0, qn, _Q_CHUNK):
            qi = q[i : i + _Q_CHUNK]
            sl = slice(i, i + _Q_CHUNK)
            t = ((qi[:, None, :] - aj[None, :, :]) * abj[None, :, :]).sum(-1) / ab2j[None, :]
            t = t.clamp(0.0, 1.0)
            closest = aj[None, :, :] + t.unsqueeze(-1) * abj[None, :, :]
            d2 = ((qi[:, None, :] - closest) ** 2).sum(-1)
            d_min, idx = d2.min(dim=-1)
            better = d_min < min_d2[sl]
            min_d2[sl] = torch.where(better, d_min, min_d2[sl])
            best[sl] = torch.where(better[:, None], dj[idx], best[sl])
    return min_d2.sqrt(), best


def _lane_query(
    vocab_xy: Tensor, batch: dict[str, Tensor]
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Per-plan nearest-lane distance + direction. None if no map polylines."""
    map_xy = batch.get("map_xy")
    map_pt_valid = batch.get("map_pt_valid")
    map_type = batch.get("map_type")
    if map_xy is None or map_pt_valid is None or map_type is None:
        return None
    b, k, t, _ = vocab_xy.shape
    lane_d = torch.full((b, k, t), 1e6, device=vocab_xy.device, dtype=torch.float32)
    lane_dir = torch.zeros(b, k, t, 2, device=vocab_xy.device, dtype=torch.float32)
    has_lane = torch.zeros(b, dtype=torch.bool, device=vocab_xy.device)
    # Shared vocab cloud → one AABB for the whole batch.
    xmin, xmax, ymin, ymax = _plan_aabb(vocab_xy)
    for bi in range(b):
        q = vocab_xy[bi].reshape(k * t, 2).float()
        la, lb, ld = _compact_segments(map_xy[bi], map_pt_valid[bi], map_type[bi], _LANE_TYPES)
        has_lane[bi] = la.size(0) > 0
        if la.size(0) == 0:
            continue
        la, lb, ld = _filter_segments_aabb(la, lb, ld, xmin, xmax, ymin, ymax)
        if la.size(0) == 0:
            continue
        d, direc = _nearest_segments(q, la, lb, ld)
        lane_d[bi] = d.view(k, t)
        lane_dir[bi] = direc.view(k, t, 2)
    return lane_d, lane_dir, has_lane


def _cross2(u: Tensor, v: Tensor) -> Tensor:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _traj_crosses_edges(xy: Tensor, ea: Tensor, eb: Tensor) -> Tensor:
    """True per mode if any consecutive plan segment properly intersects an edge.

    ``xy`` [K,T,2], ``ea``/``eb`` [N,2] → ``[K]``. Modes are tiled by
    ``_DAC_K_CHUNK`` so peak memory stays O(chunk × T × edges).
    """
    k, t, _ = xy.shape
    device = xy.device
    if t < 2 or ea.size(0) == 0:
        return torch.zeros(k, dtype=torch.bool, device=device)
    hit = torch.zeros(k, dtype=torch.bool, device=device)
    n = ea.size(0)
    for i0 in range(0, k, _DAC_K_CHUNK):
        sl = slice(i0, i0 + _DAC_K_CHUNK)
        p = xy[sl, :-1].float()
        q = xy[sl, 1:].float()
        pq = q - p
        chunk_hit = torch.zeros(p.size(0), dtype=torch.bool, device=device)
        for j in range(0, n, _EDGE_CHUNK):
            aj = ea[j : j + _EDGE_CHUNK].float()
            bj = eb[j : j + _EDGE_CHUNK].float()
            ab = bj - aj
            o1 = _cross2(pq[:, :, None, :], aj[None, None, :, :] - p[:, :, None, :])
            o2 = _cross2(pq[:, :, None, :], bj[None, None, :, :] - p[:, :, None, :])
            o3 = _cross2(ab[None, None, :, :], p[:, :, None, :] - aj[None, None, :, :])
            o4 = _cross2(ab[None, None, :, :], q[:, :, None, :] - aj[None, None, :, :])
            cross = (o1 * o2 < 0) & (o3 * o4 < 0)
            chunk_hit = chunk_hit | cross.any(dim=-1).any(dim=-1)
        hit[sl] = chunk_hit
    return hit


def _dac_from_edge_cross(
    vocab_xy: Tensor, batch: dict[str, Tensor]
) -> tuple[Tensor, Tensor]:
    """Per-mode edge-cross score + whether the scene had any road-edge geometry.

    Returns ``(ok [B,K], has_edge [B])``. ``ok`` is 1 iff no plan segment
    crosses a road-edge. Scenes with **no** edges get ``ok=1`` placeholder and
    ``has_edge=False`` — caller must not treat that as “on road.”
    """
    map_xy = batch.get("map_xy")
    map_pt_valid = batch.get("map_pt_valid")
    map_type = batch.get("map_type")
    b, k, _, _ = vocab_xy.shape
    ones = torch.ones(b, k, device=vocab_xy.device, dtype=torch.float32)
    has_edge = torch.zeros(b, device=vocab_xy.device, dtype=torch.bool)
    if map_xy is None or map_pt_valid is None or map_type is None:
        return ones, has_edge
    xmin, xmax, ymin, ymax = _plan_aabb(vocab_xy)
    out = []
    for bi in range(b):
        ea, eb, ed = _compact_segments(map_xy[bi], map_pt_valid[bi], map_type[bi], (C.MAP_ROAD_EDGE,))
        if ea.size(0) == 0:
            out.append(ones[bi])
            continue
        ea, eb, _ = _filter_segments_aabb(ea, eb, ed, xmin, xmax, ymin, ymax)
        if ea.size(0) == 0:
            out.append(ones[bi])
            continue
        has_edge[bi] = True
        crossed = _traj_crosses_edges(vocab_xy[bi], ea, eb)
        out.append((~crossed).float())
    return torch.stack(out, dim=0), has_edge


def _dac_from_lane_prox(
    lane_d: Tensor, has_lane: Tensor, *, max_lane_m: float = _DAC_MAX_LANE_M
) -> Tensor:
    """DAC=0 if any plan sample is farther than ``max_lane_m`` from a driving lane."""
    far = lane_d > float(max_lane_m)
    ok = (~far.any(dim=-1)).float()
    # Scenes without lanes: placeholder 1; ``_dac_ok`` gates on has_lane.
    return torch.where(has_lane[:, None], ok, torch.ones_like(ok))


def _dac_ok(vocab_xy: Tensor, batch: dict[str, Tensor], lane_q=None) -> Tensor:
    """On-road: must not cross a road-edge AND stay near a driving lane.

    Fail-closed: if the scene has neither road edges nor vehicle lanes in the
    map crop, every mode scores 0 (no free “DAC=1” pass).
    """
    edge_ok, has_edge = _dac_from_edge_cross(vocab_xy, batch)
    q = lane_q if lane_q is not None else _lane_query(vocab_xy, batch)
    if q is None:
        has_lane = torch.zeros(vocab_xy.size(0), device=vocab_xy.device, dtype=torch.bool)
        lane_ok = torch.ones_like(edge_ok)
    else:
        lane_d, _, has_lane = q
        lane_ok = _dac_from_lane_prox(lane_d, has_lane)
    usable = has_edge | has_lane
    score = edge_ok * lane_ok
    # If edges exist, edge_ok already applied; if only lanes, lane_ok applies.
    # Scenes with neither geometry → 0.
    return torch.where(usable[:, None], score, torch.zeros_like(score))


def _ddc_from_dists(
    vocab_xy: Tensor,
    lane_d: Tensor,
    lane_dir: Tensor,
    pred_yaw: Tensor | None,
    *,
    dt: float = C.DT,
) -> Tensor:
    """Oncoming vs nearest-lane direction; score {0, 0.5, 1} from 1 s travel."""
    del lane_d  # association is nearest-lane already; no extra distance gate
    _, yaw = _plan_speed_yaw(vocab_xy.float(), pred_yaw, float(dt))
    ego_dir = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    dxy = vocab_xy[:, :, 1:] - vocab_xy[:, :, :-1]
    step = torch.linalg.vector_norm(dxy.float(), dim=-1)
    step = torch.cat([step[:, :, :1], step], dim=-1)
    oncoming = (lane_dir * ego_dir).sum(-1) < 0
    against = step * oncoming.float()
    w = min(max(1, int(round(1.0 / float(dt)))), against.size(-1))
    win = against.unfold(-1, w, 1).sum(-1).amax(dim=-1)
    score = torch.ones_like(win)
    score = torch.where(win >= _DDC_OK_M, torch.full_like(score, 0.5), score)
    score = torch.where(win >= _DDC_HALF_M, torch.zeros_like(score), score)
    return score


def _lk_from_dists(lane_d: Tensor, *, dt: float = C.DT) -> Tensor:
    need = int(round(_LK_SECONDS / float(dt)))
    bad = (lane_d > _LK_LATERAL_M) & (lane_d < _LK_IN_LANE_M)
    run = torch.zeros(lane_d.size(0), lane_d.size(1), device=lane_d.device)
    mx = torch.zeros_like(run)
    for step in range(lane_d.size(-1)):
        run = torch.where(bad[:, :, step], run + 1.0, torch.zeros_like(run))
        mx = torch.maximum(mx, run)
    return (mx < need).float()


def _ddc_ok(vocab_xy: Tensor, batch: dict[str, Tensor], pred_yaw: Tensor | None) -> Tensor:
    """Driving-direction compliance: {0, 0.5, 1} from oncoming travel in 1 s."""
    q = _lane_query(vocab_xy, batch)
    if q is None or vocab_xy.size(2) < 2:
        return torch.ones(vocab_xy.size(0), vocab_xy.size(1), device=vocab_xy.device)
    lane_d, lane_dir, has_lane = q
    score = _ddc_from_dists(vocab_xy, lane_d, lane_dir, pred_yaw)
    return torch.where(has_lane[:, None], score, torch.ones_like(score))


def _tl_red_mask(state: Tensor) -> Tensor:
    red = torch.zeros_like(state, dtype=torch.bool)
    for s in _RED_TL:
        red = red | (state == int(s))
    return red


def _tlc_ok(vocab_xy: Tensor, batch: dict[str, Tensor]) -> Tensor:
    """Traffic-light compliance using per-step future TL state when available.

    Fail if at any plan step the ego is >2 m past a stop point (within 4 m
    laterally) while that light is red at that same step. Falls back to
    broadcasting ``tl_state`` at t=0 when ``tl_fut_*`` is absent.
    """
    tl_xy = batch.get("tl_xy")
    b, k, t, _ = vocab_xy.shape
    ones = torch.ones(b, k, device=vocab_xy.device)
    if tl_xy is None:
        return ones
    stop = tl_xy.float()  # [B, L, 2]
    ahead0 = stop[..., 0] > 0.5

    fut_state = batch.get("tl_fut_state")
    fut_valid = batch.get("tl_fut_valid")
    if fut_state is not None and fut_valid is not None:
        # Align plan step i with future index i (same as other_fut / gt).
        tf = min(int(t), int(fut_state.size(-1)))
        state_t = fut_state[..., :tf]
        valid_t = fut_valid[..., :tf].bool()
        red_t = _tl_red_mask(state_t) & valid_t & ahead0[..., None]  # [B,L,Tf]
        pts = vocab_xy[:, :, :tf].float()
    else:
        tl_state = batch.get("tl_state")
        tl_valid = batch.get("tl_valid")
        if tl_state is None or tl_valid is None:
            return ones
        red0 = _tl_red_mask(tl_state) & tl_valid.bool() & ahead0
        tf = int(t)
        pts = vocab_xy.float()
        red_t = red0[..., None].expand(-1, -1, tf)

    parts = []
    for i in range(0, k, _K_CHUNK):
        sl = pts[:, i : i + _K_CHUNK]  # [B,k',T,2]
        dy = (sl.unsqueeze(3) - stop[:, None, None, :, :]).abs()[..., 1]
        past = sl[..., 0].unsqueeze(3) > (stop[:, None, None, :, 0] + _TLC_PAST_M)
        # red_t: [B,L,T] → [B,1,T,L]
        active = red_t.permute(0, 2, 1)[:, None, :, :]
        hit = (active & (dy < _TLC_RADIUS_M) & past).any(dim=-1).any(dim=-1)
        parts.append((~hit).float())
    return torch.cat(parts, dim=1)


def _lk_ok(vocab_xy: Tensor, batch: dict[str, Tensor]) -> Tensor:
    """Lane keeping: 0 if |lat| > 1 m for ≥2 s while still near a lane."""
    q = _lane_query(vocab_xy, batch)
    if q is None:
        return torch.ones(vocab_xy.size(0), vocab_xy.size(1), device=vocab_xy.device)
    lane_d, _, has_lane = q
    score = _lk_from_dists(lane_d)
    return torch.where(has_lane[:, None], score, torch.ones_like(score))


def _route_arc_length(route_xy: Tensor, route_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Cumulative arc length along a polyline. Returns (cum[Q], total)."""
    pts = route_xy.float()
    valid = route_mask.bool()
    # Zero out invalid points so diffs involving them don't contribute.
    pts = torch.where(valid[:, None], pts, torch.zeros_like(pts))
    d = torch.linalg.vector_norm(pts[1:] - pts[:-1], dim=-1)
    both = valid[1:] & valid[:-1]
    d = torch.where(both, d, torch.zeros_like(d))
    cum = torch.zeros(pts.size(0), device=pts.device, dtype=pts.dtype)
    if d.numel():
        cum[1:] = torch.cumsum(d, dim=0)
    total = cum[valid].max() if valid.any() else cum.new_zeros(())
    return cum, total


def _project_onto_route(
    xy: Tensor, route_xy: Tensor, route_mask: Tensor, cum: Tensor
) -> tuple[Tensor, Tensor]:
    """Project points onto a route polyline → (arc_s, lat_dist).

    ``xy`` [..., 2], route [Q,2] → same leading shape.
    """
    lead = xy.shape[:-1]
    q = xy.reshape(-1, 2).float()
    pts = route_xy.float()
    valid = route_mask.bool()
    device = q.device
    if int(valid.sum()) < 2:
        z = torch.zeros(q.size(0), device=device, dtype=q.dtype)
        return z.view(*lead), torch.full((*lead,), 1e6, device=device, dtype=q.dtype)
    # Compact to contiguous valid vertices (keeps cum alignment via index).
    idx = valid.nonzero(as_tuple=False).squeeze(-1)
    a = pts[idx[:-1]]
    b = pts[idx[1:]]
    cum_a = cum[idx[:-1]]
    ab = b - a
    ab2 = (ab * ab).sum(-1).clamp_min(1e-6)
    # Nearest segment for each query.
    t = ((q[:, None, :] - a[None, :, :]) * ab[None, :, :]).sum(-1) / ab2[None, :]
    t = t.clamp(0.0, 1.0)
    closest = a[None, :, :] + t.unsqueeze(-1) * ab[None, :, :]
    d2 = ((q[:, None, :] - closest) ** 2).sum(-1)
    d_min, j = d2.min(dim=-1)
    t_best = t.gather(1, j[:, None]).squeeze(-1)
    s = cum_a[j] + t_best * ab2[j].sqrt()
    return s.view(*lead), d_min.sqrt().view(*lead)


def _ego_progress(
    vocab_xy: Tensor,
    gt_xy: Tensor,
    batch: dict[str, Tensor] | None = None,
) -> Tensor:
    """Ego progress in [0, 1] along the **log SDC future** (``gt_xy``).

    Reference polyline is the logged trajectory (not ``sdc_route`` / map path).
    Score = arc(proposal end on log) / arc(log end), clamped to [0, 1].
    Proposals with lateral distance > ``_EP_ROUTE_LAT_M`` from the log → 0.
    Short logs (< ``_EP_REF_M``) score all ones (don't force progress).

    ``batch`` is unused; kept for call-site compatibility.
    """
    del batch  # progress is log-relative; route corridor is a separate map feature
    device = vocab_xy.device
    b, k, _, _ = vocab_xy.shape
    out = torch.zeros(b, k, device=device, dtype=torch.float32)

    for bi in range(b):
        ref_xy = gt_xy[bi].float()
        ref_mask = torch.ones(ref_xy.size(0), device=device, dtype=torch.bool)
        cum, total = _route_arc_length(ref_xy, ref_mask)
        s_gt_v = float(total)
        if s_gt_v < _EP_REF_M:
            # Logged barely advances — don't force progress.
            out[bi] = 1.0
            continue
        s_pred, lat_pred = _project_onto_route(
            vocab_xy[bi, :, -1, :], ref_xy, ref_mask, cum
        )
        ratio = (s_pred / max(s_gt_v, 1e-3)).clamp(0.0, 1.0)
        on_path = lat_pred <= _EP_ROUTE_LAT_M
        out[bi] = torch.where(on_path, ratio, torch.zeros_like(ratio))
    return out


def hydra_metric_labels(
    vocab_xy: Tensor,
    batch: dict[str, Tensor],
    pred_yaw: Tensor | None = None,
    *,
    max_k: int | None = _MAX_K_METRIC,
) -> Tensor:
    """EPDMS-proxy labels [B, K, 8] = (nc, dac, ddc, tlc, ttc, lk, hc, progress).

    Horizon is 4 s (NAVSIM / LDiP). NC: no at-fault box overlap vs logged
    neighbors (circles if ``pred_yaw`` is omitted). DAC: fail if any plan
    segment crosses a road-edge **or** any sample is >``_DAC_MAX_LANE_M`` from
    a driving lane. DDC uses nearest-lane direction (no distance gate). TLC
    uses ``tl_fut_state`` per future step when present. HC stitches ~1 s of
    history. EP is arc-length ratio along the log SDC future (``gt_xy``).
    Occupancy / map heads are skipped when K exceeds ``max_k`` (set
    ``max_k=None`` when filling the offline cache). EC cannot be labeled from
    a single plan.
    """
    if vocab_xy.dim() != 4:
        raise ValueError(f"vocab_xy must be [B,K,T,2], got {tuple(vocab_xy.shape)}")
    t = min(int(vocab_xy.size(2)), _PDM_T)
    vocab_xy = vocab_xy[:, :, :t]
    if pred_yaw is not None:
        pred_yaw = pred_yaw[:, :, :t]
    gt_xy = batch["gt_xy"][:, :t]
    b, k, _, _ = vocab_xy.shape
    device = vocab_xy.device
    ones = torch.ones(b, k, device=device, dtype=torch.float32)
    nc = ones
    ttc = ones
    dac = ones
    ddc = ones
    tlc = ones
    lk = ones
    run_occ = max_k is None or k <= int(max_k)
    if run_occ and batch.get("other_fut_xy") is not None:
        hits = _timestep_hits(
            vocab_xy,
            batch["other_fut_xy"],
            batch["other_fut_valid"],
            batch["ego_lw"],
            batch["other_lw"],
            pred_yaw,
        )
        nc = (~hits.any(dim=-1)).float()
        ttc = _ttc_clear(
            vocab_xy,
            pred_yaw,
            batch["other_fut_xy"],
            batch["other_fut_valid"],
            batch["ego_lw"],
            batch["other_lw"],
        )
    if run_occ:
        tlc = _tlc_ok(vocab_xy, batch)
        xy_m = vocab_xy[:, :, ::_MAP_STRIDE]
        yaw_m = pred_yaw[:, :, ::_MAP_STRIDE] if pred_yaw is not None else None
        dt_m = float(C.DT) * _MAP_STRIDE
        q = _lane_query(xy_m, batch)
        dac = _dac_ok(vocab_xy, batch, lane_q=q)
        if q is not None:
            lane_d, lane_dir, has_lane = q
            ddc = torch.where(
                has_lane[:, None],
                _ddc_from_dists(xy_m, lane_d, lane_dir, yaw_m, dt=dt_m),
                ones,
            )
            lk = torch.where(has_lane[:, None], _lk_from_dists(lane_d, dt=dt_m), ones)
    hc = _history_comfort(vocab_xy, batch)
    progress = _ego_progress(vocab_xy, gt_xy, batch)
    return torch.stack([nc, dac, ddc, tlc, ttc, lk, hc, progress], dim=-1)


def gather_or_compute_metric_labels(
    vocab_xy: Tensor,
    batch: dict[str, Tensor],
    stage_idx: Tensor,
    pred_yaw: Tensor | None = None,
) -> Tensor:
    """Use a full-vocab cache [B, K_full, M] when present, else compute."""
    cached = batch.get("metric_labels")
    if cached is not None:
        n_m = cached.size(-1)
        idx = stage_idx.unsqueeze(-1).expand(-1, -1, n_m)
        return cached.gather(1, idx).float()
    return hydra_metric_labels(vocab_xy, batch, pred_yaw=pred_yaw)
