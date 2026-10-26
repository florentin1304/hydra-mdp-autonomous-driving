"""PlanTF State Dropout Encoder (SDE).

Port of ``StateAttentionEncoder`` from jchengai/planTF
(``src/models/planTF/modules/agent_encoder.py``): each state scalar is embedded
independently, a learnable query aggregates via cross-attention, and during
training every token except the first three (pose / heading) is dropped with
probability ``state_dropout`` (PlanTF default 0.75).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StateAttentionEncoder(nn.Module):
    """Attention-based ego state encoder with train-time kinematic dropout."""

    def __init__(self, state_channel: int, dim: int, state_dropout: float = 0.75) -> None:
        super().__init__()
        if state_channel < 3:
            raise ValueError("state_channel must be >= 3 (pose/heading kept visible)")
        self.state_channel = int(state_channel)
        self.state_dropout = float(state_dropout)
        self.linears = nn.ModuleList([nn.Linear(1, dim) for _ in range(self.state_channel)])
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.pos_embed = nn.Parameter(torch.Tensor(1, self.state_channel, dim))
        self.query = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """``x`` [B, C] → ego token [B, D]."""
        if x.size(-1) != self.state_channel:
            raise ValueError(f"expected C={self.state_channel}, got {tuple(x.shape)}")
        x_embed = torch.stack(
            [linear(x[:, i, None]) for i, linear in enumerate(self.linears)],
            dim=1,
        )
        x_embed = x_embed + self.pos_embed.expand(x_embed.size(0), -1, -1)

        key_padding_mask = None
        if self.training and self.state_dropout > 0:
            # First 3 channels (pose / heading) always visible; drop the rest.
            visible = torch.zeros(x.size(0), 3, device=x.device, dtype=torch.bool)
            drop = (
                torch.rand(x.size(0), self.state_channel - 3, device=x.device)
                < self.state_dropout
            )
            key_padding_mask = torch.cat([visible, drop], dim=1)

        query = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(
            query=query,
            key=x_embed,
            value=x_embed,
            key_padding_mask=key_padding_mask,
        )
        return out[:, 0]


def plantf_state6_from_ego(ego: Tensor) -> Tensor:
    """Build PlanTF-style state6 from our ``ego_state`` [B, 12].

    Channels: ``[px, py, heading, speed, ax, ay]``. In the agent-centric frame
    pose/heading are identically zero (never dropped by SDE); kinematics are
    ``ego[:, 0:3]`` (speed, ax, ay) and are dropout candidates.
    """
    zeros = ego.new_zeros(ego.size(0), 3)
    return torch.cat([zeros, ego[:, :3]], dim=-1)
