"""Wayformer early-fusion encoder with optional DIPP-style factorized banks."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from womd_il import constants as C
from womd_il.models.interface import SceneEncoding
from womd_il.models.layers import AgentMapSocialBlock, EncoderBlock, LatentQueryEncoder, MLP
from womd_il.models.sde import StateAttentionEncoder, plantf_state6_from_ego


def pack_valid_tokens(tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Gather valid tokens to the front and trim to the batch max length.

    ``mask`` True = keep. PyTorch attention still pays for padded slots, so
    dropping them is the actual speedup.
    """
    valid = mask.bool()
    order = valid.to(dtype=tokens.dtype).argsort(dim=1, descending=True, stable=True)
    packed = tokens.gather(1, order.unsqueeze(-1).expand_as(tokens))
    packed_mask = valid.gather(1, order)
    nmax = int(packed_mask.sum(dim=1).max().item()) if packed_mask.numel() else 1
    nmax = max(nmax, 1)
    packed = packed[:, :nmax]
    packed_mask = packed_mask[:, :nmax]
    empty = ~packed_mask.any(dim=1)
    if bool(empty.any()):
        packed_mask = packed_mask.clone()
        packed_mask[empty, 0] = True
    return packed, packed_mask


def _pool_time(x: Tensor, mask: Tensor, mode: str) -> tuple[Tensor, Tensor]:
    """Pool [B, N, T, D] over time → [B, N, D]. ``mask`` True = valid [B, N, T]."""
    valid = mask.bool()
    agent_ok = valid.any(dim=-1)  # [B, N]
    if mode == "last":
        # Index of last True along T; 0 if none.
        t_idx = valid.to(dtype=torch.long) * torch.arange(
            valid.size(-1), device=valid.device, dtype=torch.long
        )
        last = t_idx.max(dim=-1).values  # [B, N]
        gather_i = last.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, x.size(-1))
        pooled = x.gather(2, gather_i).squeeze(2)
    else:
        w = valid.to(dtype=x.dtype).unsqueeze(-1)
        denom = w.sum(dim=2).clamp_min(1.0)
        pooled = (x * w).sum(dim=2) / denom
    return pooled, agent_ok


def _lstm_history(
    lstm: nn.LSTM, x: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor]:
    """DIPP/GET-DIPP: shared LSTM over time; keep last (current) hidden. [B,N,T,D]→[B,N,D]."""
    valid = mask.bool()
    agent_ok = valid.any(dim=-1)
    b, n, t, d = x.shape
    flat = x.reshape(b * n, t, d)
    lengths = valid.reshape(b * n, t).sum(dim=1).clamp_min(1)
    packed = nn.utils.rnn.pack_padded_sequence(
        flat, lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    _, (h_n, _) = lstm(packed)
    # Last LSTM layer, current-step state (GET-DIPP).
    out = h_n[-1].reshape(b, n, d)
    return out, agent_ok


class WayformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        nhead: int = 8,
        dropout: float = 0.1,
        num_latents: int = 0,
        encoder_layers: int = 2,
        hist_in_dim: int = C.HIST_IN_DIM,
        map_in_dim: int = C.MAP_IN_DIM,
        tl_in_dim: int = C.TL_IN_DIM,
        ego_in_dim: int = C.EGO_IN_DIM,
        route_in_dim: int = C.ROUTE_IN_DIM,
        max_hist: int = 16,
        max_neigh: int = 64,
        max_map: int = 512,
        max_tl: int = 32,
        max_route: int = 128,
        drop_ego_hist: bool = False,
        ego_kin_dropout: float = 0.0,
        # Preferred-path XY tokens. Default off: mission = map on-route corridor only.
        use_route_tokens: bool = False,
        # Factorized banks: map self-attn; agents social→map→social.
        # Decoder dual_cross: traj→agents then traj→map.
        factorized: bool = False,
        agent_pool: str = "lstm",  # lstm (DIPP) | mean | last
        agent_lstm_layers: int = 2,
        map_encoder_layers: int | None = None,
        # Ego token: mlp (legacy ego_proj) | sde (PlanTF StateAttentionEncoder).
        ego_encoder: str = "mlp",
        ego_sde_dropout: float = 0.75,
        ego_sde_channels: int = 6,
        # Concat ego into the agent social pool (factorized), then split after fusion.
        ego_in_social: bool = False,
        **_: object,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_latents = int(num_latents)
        self.drop_ego_hist = bool(drop_ego_hist)
        self.ego_kin_dropout = float(ego_kin_dropout)
        self.use_route_tokens = bool(use_route_tokens)
        self.factorized = bool(factorized)
        self.agent_pool = str(agent_pool or "lstm").lower()
        if self.agent_pool not in ("lstm", "mean", "last"):
            raise ValueError(f"agent_pool must be 'lstm', 'mean', or 'last', got {agent_pool!r}")
        self.ego_encoder = str(ego_encoder or "mlp").lower()
        if self.ego_encoder not in ("mlp", "sde"):
            raise ValueError(f"ego_encoder must be 'mlp' or 'sde', got {ego_encoder!r}")
        self.ego_sde_dropout = float(ego_sde_dropout)
        self.ego_sde_channels = int(ego_sde_channels)
        self.ego_in_social = bool(ego_in_social)
        self.hist_proj = MLP(hist_in_dim, d_model, d_model, dropout)
        self.map_proj = MLP(map_in_dim, d_model, d_model, dropout)
        self.tl_proj = MLP(tl_in_dim, d_model, d_model, dropout)
        # Kept for ckpt compat; unused when use_route_tokens=False.
        self.route_proj = MLP(route_in_dim, d_model, d_model, dropout)
        self.ego_proj = MLP(ego_in_dim, d_model, d_model, dropout)
        self.ego_sde = (
            StateAttentionEncoder(self.ego_sde_channels, d_model, self.ego_sde_dropout)
            if self.ego_encoder == "sde"
            else None
        )
        # Learned positional embeddings, zero-init (Wayformer).
        self.pos_hist = nn.Parameter(torch.zeros(1, max_hist, d_model))
        self.pos_neigh_agent = nn.Parameter(torch.zeros(1, max_neigh, 1, d_model))
        self.pos_neigh_time = nn.Parameter(torch.zeros(1, 1, max_hist, d_model))
        self.pos_map = nn.Parameter(torch.zeros(1, max_map, d_model))
        self.pos_tl = nn.Parameter(torch.zeros(1, max_tl, 1, d_model))
        self.pos_tl_time = nn.Parameter(torch.zeros(1, 1, max_hist, d_model))
        self.pos_route = nn.Parameter(torch.zeros(1, max_route, d_model))

        map_layers = encoder_layers if map_encoder_layers is None else int(map_encoder_layers)

        if self.factorized:
            if self.num_latents > 0:
                raise ValueError("factorized encoder does not support num_latents > 0")
            self.encoder = None
            self.scene_layers = None
            # DIPP: shared multi-layer LSTM over agent history → current-step feature.
            n_lstm = max(int(agent_lstm_layers), 1)
            self.agent_hist_lstm = (
                nn.LSTM(d_model, d_model, num_layers=n_lstm, batch_first=True)
                if self.agent_pool == "lstm"
                else None
            )
            # Agents: social → map cross → social (per layer).
            self.agent_layers = nn.ModuleList(
                [
                    AgentMapSocialBlock(d_model, nhead, d_ffn, dropout)
                    for _ in range(encoder_layers)
                ]
            )
            self.map_layers = nn.ModuleList(
                [EncoderBlock(d_model, nhead, d_ffn, dropout) for _ in range(map_layers)]
            )
        elif self.num_latents > 0:
            self.encoder = LatentQueryEncoder(
                d_model=d_model,
                nhead=nhead,
                d_ffn=d_ffn,
                num_latents=self.num_latents,
                num_layers=encoder_layers,
                dropout=dropout,
            )
            self.scene_layers = None
            self.agent_layers = None
            self.map_layers = None
            self.agent_hist_lstm = None
        else:
            self.encoder = None
            self.scene_layers = nn.ModuleList(
                [EncoderBlock(d_model, nhead, d_ffn, dropout) for _ in range(encoder_layers)]
            )
            self.agent_layers = None
            self.map_layers = None
            self.agent_hist_lstm = None

    def _drop_ego_kinematics(self, ego: Tensor) -> Tensor:
        """Legacy independent speed/ax/ay zeroing (mlp ego path only)."""
        if not self.training or self.ego_kin_dropout <= 0 or ego.size(-1) < 3:
            return ego
        ego = ego.clone()
        drop = torch.rand(ego.size(0), 3, device=ego.device) < self.ego_kin_dropout
        ego[:, :3] = ego[:, :3].masked_fill(drop, 0.0)
        return ego

    def _encode_ego_token(self, batch: dict[str, Tensor]) -> Tensor:
        """Project raw ``ego_state`` → [B, D] (SDE or MLP)."""
        raw = batch["ego_state"]
        if self.ego_sde is not None:
            state = plantf_state6_from_ego(raw.float())
            if state.size(-1) != self.ego_sde_channels:
                # Pad / trim if config channels differ from built state6.
                c = self.ego_sde_channels
                if state.size(-1) < c:
                    state = torch.cat(
                        [state, state.new_zeros(state.size(0), c - state.size(-1))],
                        dim=-1,
                    )
                else:
                    state = state[:, :c]
            return self.ego_sde(state)
        return self.ego_proj(self._drop_ego_kinematics(raw))

    def _tl_tokens(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        feat = batch["tl_feat"]
        mask = batch["tl_mask"]
        if feat.dim() == 3:
            tl = self.tl_proj(feat)
            l = tl.size(1)
            tl = tl + self.pos_tl[:, :l, 0]
            return tl, mask.reshape(tl.size(0), l) > 0.5
        if feat.dim() != 4:
            raise ValueError(f"tl_feat must be [B,L,C] or [B,L,T,C], got {tuple(feat.shape)}")
        tl = self.tl_proj(feat)
        b, l, t, d = tl.shape
        tl = tl + self.pos_tl[:, :l] + self.pos_tl_time[:, :, :t]
        return tl.reshape(b, l * t, d), mask.reshape(b, l * t) > 0.5

    def _route_tokens(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor] | None:
        """Optional preferred-path XY tokens. Off by default (corridor-only mission)."""
        if not self.use_route_tokens:
            return None
        feat = batch.get("route_feat")
        mask = batch.get("sdc_route_mask")
        if feat is None:
            xy = batch.get("sdc_route")
            if xy is None or mask is None:
                return None
            q = xy.size(1)
            prog = torch.linspace(0.0, 1.0, q, device=xy.device, dtype=xy.dtype)
            feat = torch.cat([xy, prog.expand(xy.size(0), -1).unsqueeze(-1)], dim=-1)
        rt = self.route_proj(feat.float())
        q = rt.size(1)
        rt = rt + self.pos_route[:, :q]
        if mask is None:
            mask = torch.ones(rt.size(0), q, device=rt.device, dtype=torch.bool)
        return rt, mask.reshape(rt.size(0), q) > 0.5

    def _encode_fused(self, batch: dict[str, Tensor]) -> SceneEncoding:
        neigh = self.hist_proj(batch["neighbors"])  # [B, A, T, D]
        a, tn = neigh.size(1), neigh.size(2)
        neigh = (
            neigh
            + self.pos_neigh_agent[:, :a]
            + self.pos_neigh_time[:, :, :tn]
        )
        b = neigh.size(0)
        neigh = neigh.reshape(b, a * tn, -1)
        neigh_mask = batch["neighbors_mask"].reshape(b, a * tn) > 0.5

        mp = self.map_proj(batch["map_feat"])
        p = mp.size(1)
        mp = mp + self.pos_map[:, :p]
        map_mask = batch["map_mask"] > 0.5

        tl, tl_mask = self._tl_tokens(batch)

        parts = [neigh, mp, tl]
        masks = [neigh_mask, map_mask, tl_mask]
        route = self._route_tokens(batch)
        if route is not None:
            rt, rt_mask = route
            parts.append(rt)
            masks.append(rt_mask)
        if not self.drop_ego_hist:
            sdc = self.hist_proj(batch["sdc_hist"])
            t = sdc.size(1)
            sdc = sdc + self.pos_hist[:, :t]
            parts = [sdc, *parts]
            masks = [batch["sdc_hist_mask"] > 0.5, *masks]

        tokens = torch.cat(parts, dim=1)
        mask = torch.cat(masks, dim=1)
        tokens, mask = pack_valid_tokens(tokens, mask)
        pad = ~mask
        if self.encoder is not None:
            latents = self.encoder(tokens, pad)
            out_tokens = latents
            out_mask = torch.ones(latents.shape[:2], device=latents.device, dtype=torch.bool)
        else:
            x = tokens
            for layer in self.scene_layers or []:
                x = layer(x, key_padding_mask=pad)
            out_tokens, out_mask = x, mask
        ego = self._encode_ego_token(batch)
        return SceneEncoding(
            tokens=out_tokens,
            token_mask=out_mask,
            ego_state=ego,
            current_xyh=batch["current_xyh_world"],
            sdc_route=batch["sdc_route"],
            extras={
                "raw_ego": batch["ego_state"],
                "speed": batch.get("speed"),
                "last_xy_vel": batch["sdc_hist"][:, -1, 2:4],
                "n_scene": out_tokens.size(1),
                "factorized": False,
            },
        )

    def _encode_factorized(self, batch: dict[str, Tensor]) -> SceneEncoding:
        """Map self-attn; neighbours (+ optional ego) social → map → social; split."""
        # --- Map bank (geometry / TL / route) before agents, so agents can ground. ---
        mp = self.map_proj(batch["map_feat"])
        p = mp.size(1)
        mp = mp + self.pos_map[:, :p]
        map_mask = batch["map_mask"] > 0.5

        parts = [mp]
        masks = [map_mask]
        tl, tl_mask = self._tl_tokens(batch)
        parts.append(tl)
        masks.append(tl_mask)
        route = self._route_tokens(batch)
        if route is not None:
            rt, rt_mask = route
            parts.append(rt)
            masks.append(rt_mask)

        map_bank = torch.cat(parts, dim=1)
        map_bank_mask = torch.cat(masks, dim=1)
        map_bank, map_bank_mask = pack_valid_tokens(map_bank, map_bank_mask)
        for layer in self.map_layers or []:
            map_bank = layer(map_bank, key_padding_mask=~map_bank_mask)

        # --- Neighbours: DIPP LSTM (or pool) over history. ---
        neigh = self.hist_proj(batch["neighbors"])  # [B, A, T, D]
        a, tn = neigh.size(1), neigh.size(2)
        neigh = neigh + self.pos_neigh_time[:, :, :tn]
        neigh_tmask = batch["neighbors_mask"] > 0.5  # [B, A, T]
        if self.agent_pool == "lstm":
            if self.agent_hist_lstm is None:
                raise RuntimeError("agent_pool=lstm requires agent_hist_lstm")
            agents, agent_mask = _lstm_history(self.agent_hist_lstm, neigh, neigh_tmask)
        else:
            agents, agent_mask = _pool_time(neigh, neigh_tmask, self.agent_pool)
        agents = agents + self.pos_neigh_agent[:, :a, 0, :]

        ego = self._encode_ego_token(batch)  # [B, D]
        ego_tok = ego.unsqueeze(1)  # [B, 1, D]
        ego_mask = torch.ones(ego.size(0), 1, device=ego.device, dtype=torch.bool)

        if self.ego_in_social:
            # Ego participates in social ↔ map fusion, then is split back out.
            fused = torch.cat([ego_tok, agents], dim=1)
            fused_mask = torch.cat([ego_mask, agent_mask], dim=1)
        else:
            fused = agents
            fused_mask = agent_mask

        # Ensure at least one slot is attendable (MHA forbids all-True pad).
        empty = ~fused_mask.any(dim=1)
        if bool(empty.any()):
            fused_mask = fused_mask.clone()
            fused_mask[empty, 0] = True

        for layer in self.agent_layers or []:
            fused = layer(
                fused,
                map_bank,
                agent_padding_mask=~fused_mask,
                map_padding_mask=~map_bank_mask,
            )

        if self.ego_in_social:
            ego_tok = fused[:, :1]
            agents = fused[:, 1:]
            agent_mask = fused_mask[:, 1:]
            ego_mask = fused_mask[:, :1]
        # else: ego_tok stays the pre-social SDE/MLP token (still exposed for KV)

        # Neighbours-only bank for decoder; keep a valid slot if all padded.
        empty_a = ~agent_mask.any(dim=1)
        if bool(empty_a.any()):
            agent_mask = agent_mask.clone()
            agent_mask[empty_a, 0] = True

        tokens = torch.cat([ego_tok, agents, map_bank], dim=1)
        token_mask = torch.cat([ego_mask, agent_mask, map_bank_mask], dim=1)

        return SceneEncoding(
            tokens=tokens,
            token_mask=token_mask,
            ego_state=ego_tok.squeeze(1),
            current_xyh=batch["current_xyh_world"],
            sdc_route=batch["sdc_route"],
            agent_tokens=agents,
            agent_mask=agent_mask,
            map_tokens=map_bank,
            map_mask=map_bank_mask,
            ego_tokens=ego_tok,
            ego_mask=ego_mask,
            extras={
                "raw_ego": batch["ego_state"],
                "speed": batch.get("speed"),
                "last_xy_vel": batch["sdc_hist"][:, -1, 2:4],
                "n_scene": tokens.size(1),
                "n_agent": agents.size(1),
                "n_map": map_bank.size(1),
                "factorized": True,
                "ego_in_social": self.ego_in_social,
                "ego_encoder": self.ego_encoder,
            },
        )

    def forward(self, batch: dict[str, Tensor]) -> SceneEncoding:
        if self.factorized:
            return self._encode_factorized(batch)
        return self._encode_fused(batch)
