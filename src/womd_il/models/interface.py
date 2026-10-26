"""Shared encoder/decoder contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor, nn


@dataclass
class SceneEncoding:
    tokens: Tensor  # [B, L, D]
    token_mask: Tensor  # [B, L]  True = valid
    ego_state: Tensor  # [B, D_e]  (projected or raw)
    current_xyh: Tensor  # [B, 3] world frame
    sdc_route: Tensor  # [B, Q, 2]
    extras: dict[str, Any] = field(default_factory=dict)
    # Factorized banks (optional). dual_cross: agents→map; triple: map/ego/agents
    # (sequential or parallel+fuse via decoder.triple_parallel).
    agent_tokens: Tensor | None = None  # [B, A, D] neighbours only
    agent_mask: Tensor | None = None  # [B, A] True = valid
    map_tokens: Tensor | None = None  # [B, P, D]
    map_mask: Tensor | None = None  # [B, P] True = valid
    ego_tokens: Tensor | None = None  # [B, 1, D] after social/map fusion
    ego_mask: Tensor | None = None  # [B, 1] True = valid


@dataclass
class TrajectoryBundle:
    xyh: Tensor  # [B, K, T, 3] agent frame
    scores: Tensor  # [B, K]
    extras: dict[str, Any] = field(default_factory=dict)


class Decoder(Protocol):
    name: str

    def train_step(self, enc: SceneEncoding, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        ...

    def infer(self, enc: SceneEncoding, **kwargs) -> TrajectoryBundle:
        ...


class DecoderBase(nn.Module):
    name = "base"

    def train_step(self, enc: SceneEncoding, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError

    def infer(self, enc: SceneEncoding, **kwargs) -> TrajectoryBundle:
        raise NotImplementedError
