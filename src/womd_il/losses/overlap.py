"""Open-loop overlap of predicted modes vs logged neighbor futures."""

from __future__ import annotations

import torch
from torch import Tensor


def horizon_weights(steps: int, front_weight: float, device, dtype) -> Tensor:
    """Linear weights: first timestep `front_weight`, last timestep 1.0."""
    fw = max(float(front_weight), 1.0)
    if steps <= 1 or fw == 1.0:
        return torch.ones(steps, device=device, dtype=dtype)
    return torch.linspace(fw, 1.0, steps, device=device, dtype=dtype)


def overlap_rate(
    pred_xy: Tensor,
    other_xy: Tensor,
    other_valid: Tensor,
    ego_lw: Tensor,
    other_lw: Tensor,
    pred_valid: Tensor | None = None,
) -> Tensor:
    """Fraction of timesteps a mode's center hits another agent's circle.

    pred_xy: [B, K, T, 2]
    other_xy: [B, A, T, 2]
    other_valid: [B, A, T]
    ego_lw: [B, 2] length, width
    other_lw: [B, A, 2]
    returns [B, K] in [0, 1]
    """
    b, k, t, _ = pred_xy.shape
    r_ego = 0.5 * torch.linalg.vector_norm(ego_lw.float(), dim=-1).clamp_min(0.5)  # [B]
    r_oth = 0.5 * torch.linalg.vector_norm(other_lw.float(), dim=-1).clamp_min(0.5)  # [B,A]
    thresh = r_ego[:, None, None, None] + r_oth[:, None, :, None]  # [B,1,A,1]
    delta = pred_xy.unsqueeze(2) - other_xy.unsqueeze(1)  # [B,K,A,T,2]
    dist = torch.linalg.vector_norm(delta.float(), dim=-1)
    hit = (dist < thresh) & other_valid[:, None].bool()
    any_hit = hit.any(dim=2)  # [B,K,T]
    if pred_valid is not None:
        w = pred_valid.float().unsqueeze(1)
        return (any_hit.float() * w).sum(dim=-1) / w.sum(dim=-1).clamp_min(1.0)
    return any_hit.float().mean(dim=-1)
