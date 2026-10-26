"""Transformer building blocks used by the Wayformer encoder and decoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PreNormSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        h = self.norm(x)
        y, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        return x + self.drop(y)


class PreNormCrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        key_padding_mask: Tensor | None = None,
        residual: bool = True,
    ) -> Tensor:
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        y, _ = self.attn(qn, kvn, kvn, key_padding_mask=key_padding_mask, need_weights=False)
        y = self.drop(y)
        return q + y if residual else y


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = PreNormSelfAttention(d_model, nhead, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ffn, dropout)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        x = self.self_attn(x, key_padding_mask)
        return x + self.ff(self.norm(x))


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = PreNormSelfAttention(d_model, nhead, dropout)
        self.cross_attn = PreNormCrossAttention(d_model, nhead, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        q_padding_mask: Tensor | None = None,
        kv_padding_mask: Tensor | None = None,
    ) -> Tensor:
        q = self.self_attn(q, q_padding_mask)
        q = self.cross_attn(q, kv, kv_padding_mask)
        return q + self.ff(self.norm(q))


class CrossOnlyDecoderBlock(nn.Module):
    """Action queries cross-attend to the scene; no self-attention over the vocab."""

    def __init__(self, d_model: int, nhead: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = PreNormCrossAttention(d_model, nhead, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        kv_padding_mask: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        q = self.cross_attn(q, kv, kv_padding_mask)
        return q + self.ff(self.norm(q))


class AgentMapSocialBlock(nn.Module):
    """Agent bank: social self-attn → map cross-attn → social self-attn → FF.

    Grounds neighbors on the map, then lets them re-interact with that context.
    """

    def __init__(self, d_model: int, nhead: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.social_pre = PreNormSelfAttention(d_model, nhead, dropout)
        self.cross_map = PreNormCrossAttention(d_model, nhead, dropout)
        self.social_post = PreNormSelfAttention(d_model, nhead, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        agents: Tensor,
        map_kv: Tensor,
        agent_padding_mask: Tensor | None = None,
        map_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.social_pre(agents, agent_padding_mask)
        x = self.cross_map(x, map_kv, map_padding_mask)
        x = self.social_post(x, agent_padding_mask)
        return x + self.ff(self.norm(x))


class DualCrossOnlyDecoderBlock(nn.Module):
    """Traj queries cross-attend agent bank, then map bank (legacy DIPP order)."""

    def __init__(self, d_model: int, nhead: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.cross_agent = PreNormCrossAttention(d_model, nhead, dropout)
        self.cross_map = PreNormCrossAttention(d_model, nhead, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        q: Tensor,
        agent_kv: Tensor,
        map_kv: Tensor,
        agent_padding_mask: Tensor | None = None,
        map_padding_mask: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        q = self.cross_agent(q, agent_kv, agent_padding_mask)
        q = self.cross_map(q, map_kv, map_padding_mask)
        return q + self.ff(self.norm(q))


class DualDecoderBlock(nn.Module):
    """Self-attn over queries, then agent cross, then map cross."""

    def __init__(self, d_model: int, nhead: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = PreNormSelfAttention(d_model, nhead, dropout)
        self.cross_agent = PreNormCrossAttention(d_model, nhead, dropout)
        self.cross_map = PreNormCrossAttention(d_model, nhead, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        q: Tensor,
        agent_kv: Tensor,
        map_kv: Tensor,
        q_padding_mask: Tensor | None = None,
        agent_padding_mask: Tensor | None = None,
        map_padding_mask: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        q = self.self_attn(q, q_padding_mask)
        q = self.cross_agent(q, agent_kv, agent_padding_mask)
        q = self.cross_map(q, map_kv, map_padding_mask)
        return q + self.ff(self.norm(q))


class TripleCrossOnlyDecoderBlock(nn.Module):
    """Traj queries: map → ego → neighbours (no self-attn on queries)."""

    def __init__(self, d_model: int, nhead: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.cross_map = PreNormCrossAttention(d_model, nhead, dropout)
        self.cross_ego = PreNormCrossAttention(d_model, nhead, dropout)
        self.cross_agent = PreNormCrossAttention(d_model, nhead, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        q: Tensor,
        map_kv: Tensor,
        ego_kv: Tensor,
        agent_kv: Tensor,
        map_padding_mask: Tensor | None = None,
        ego_padding_mask: Tensor | None = None,
        agent_padding_mask: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        q = self.cross_map(q, map_kv, map_padding_mask)
        q = self.cross_ego(q, ego_kv, ego_padding_mask)
        q = self.cross_agent(q, agent_kv, agent_padding_mask)
        return q + self.ff(self.norm(q))


class TripleDecoderBlock(nn.Module):
    """Self-attn on queries, then map → ego → neighbours cross."""

    def __init__(self, d_model: int, nhead: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = PreNormSelfAttention(d_model, nhead, dropout)
        self.cross_map = PreNormCrossAttention(d_model, nhead, dropout)
        self.cross_ego = PreNormCrossAttention(d_model, nhead, dropout)
        self.cross_agent = PreNormCrossAttention(d_model, nhead, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        q: Tensor,
        map_kv: Tensor,
        ego_kv: Tensor,
        agent_kv: Tensor,
        q_padding_mask: Tensor | None = None,
        map_padding_mask: Tensor | None = None,
        ego_padding_mask: Tensor | None = None,
        agent_padding_mask: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        q = self.self_attn(q, q_padding_mask)
        q = self.cross_map(q, map_kv, map_padding_mask)
        q = self.cross_ego(q, ego_kv, ego_padding_mask)
        q = self.cross_agent(q, agent_kv, agent_padding_mask)
        return q + self.ff(self.norm(q))


def _zero_init_last_linear(mlp: MLP) -> None:
    """Last Linear of ``MLP`` → 0 so residual fuse starts as identity."""
    last = mlp.net[-1]
    assert isinstance(last, nn.Linear)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class ParallelTripleCrossOnlyDecoderBlock(nn.Module):
    """Traj queries: parallel map/ego/agent cross → concat MLP fuse → FF.

    Each bank attends from the same ``q`` (no sequential overwrite). Fuse is
    ``q + MLP([h_m; h_e; h_a])`` with zero-init last Linear so training starts
    near traj-only. Stack layers for refinement (same depth as sequential triple).
    """

    def __init__(self, d_model: int, nhead: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.cross_map = PreNormCrossAttention(d_model, nhead, dropout)
        self.cross_ego = PreNormCrossAttention(d_model, nhead, dropout)
        self.cross_agent = PreNormCrossAttention(d_model, nhead, dropout)
        self.fuse = MLP(3 * d_model, d_ffn, d_model, dropout)
        _zero_init_last_linear(self.fuse)
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        q: Tensor,
        map_kv: Tensor,
        ego_kv: Tensor,
        agent_kv: Tensor,
        map_padding_mask: Tensor | None = None,
        ego_padding_mask: Tensor | None = None,
        agent_padding_mask: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        h_m = self.cross_map(q, map_kv, map_padding_mask, residual=False)
        h_e = self.cross_ego(q, ego_kv, ego_padding_mask, residual=False)
        h_a = self.cross_agent(q, agent_kv, agent_padding_mask, residual=False)
        q = q + self.fuse(torch.cat([h_m, h_e, h_a], dim=-1))
        return q + self.ff(self.norm(q))


class ParallelTripleDecoderBlock(nn.Module):
    """Self-attn on queries, then parallel map/ego/agent cross + fuse."""

    def __init__(self, d_model: int, nhead: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = PreNormSelfAttention(d_model, nhead, dropout)
        self.cross_map = PreNormCrossAttention(d_model, nhead, dropout)
        self.cross_ego = PreNormCrossAttention(d_model, nhead, dropout)
        self.cross_agent = PreNormCrossAttention(d_model, nhead, dropout)
        self.fuse = MLP(3 * d_model, d_ffn, d_model, dropout)
        _zero_init_last_linear(self.fuse)
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        q: Tensor,
        map_kv: Tensor,
        ego_kv: Tensor,
        agent_kv: Tensor,
        q_padding_mask: Tensor | None = None,
        map_padding_mask: Tensor | None = None,
        ego_padding_mask: Tensor | None = None,
        agent_padding_mask: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        q = self.self_attn(q, q_padding_mask)
        h_m = self.cross_map(q, map_kv, map_padding_mask, residual=False)
        h_e = self.cross_ego(q, ego_kv, ego_padding_mask, residual=False)
        h_a = self.cross_agent(q, agent_kv, agent_padding_mask, residual=False)
        q = q + self.fuse(torch.cat([h_m, h_e, h_a], dim=-1))
        return q + self.ff(self.norm(q))


class LatentQueryEncoder(nn.Module):
    """Perceiver-style: learned latents cross-attend to scene tokens, then self-attend."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ffn: int,
        num_latents: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.zeros(num_latents, d_model))
        nn.init.trunc_normal_(self.latents, std=0.02)
        self.cross = PreNormCrossAttention(d_model, nhead, dropout)
        self.cross_ff_norm = nn.LayerNorm(d_model)
        self.cross_ff = FeedForward(d_model, d_ffn, dropout)
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, nhead, d_ffn, dropout) for _ in range(num_layers)]
        )

    def forward(self, tokens: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        b = tokens.size(0)
        q = self.latents.unsqueeze(0).expand(b, -1, -1)
        q = self.cross(q, tokens, pad_mask)
        q = q + self.cross_ff(self.cross_ff_norm(q))
        for layer in self.layers:
            q = layer(q, key_padding_mask=None)
        return q
