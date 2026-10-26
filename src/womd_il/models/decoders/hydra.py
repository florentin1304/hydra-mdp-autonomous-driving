"""Hydra-MDP classifier with LDiP iterative / stochastic scoring."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from womd_il.losses.hydra import (
    METRIC_NAMES,
    gather_or_compute_metric_labels,
    hydra_epo_loss,
    hydra_imitation_loss,
    hydra_metric_bce,
    hydra_metric_bce_per_head,
    hydra_pdms_loss,
    pdms_from_metric_logits,
    pdms_from_probs,
    vocab_ade_winner,
)
from womd_il.models.interface import DecoderBase, SceneEncoding, TrajectoryBundle
from womd_il.models.layers import (
    MLP,
    CrossOnlyDecoderBlock,
    DecoderBlock,
    DualCrossOnlyDecoderBlock,
    DualDecoderBlock,
    EncoderBlock,
    ParallelTripleCrossOnlyDecoderBlock,
    ParallelTripleDecoderBlock,
    TripleCrossOnlyDecoderBlock,
    TripleDecoderBlock,
    compact_kv_pad,
)


class ParallelMetricHeads(nn.Module):
    """Independent MLP per metric; all M heads computed in fused GEMMs (no head loop).

    ``n_layers=2`` → (D→H)×M → GELU → (H→1)×M
    ``n_layers=3`` → (D→H)×M → GELU → (H→H)×M → GELU → (H→1)×M

    - ``in_proj``: one ``Linear(D, M·H)`` (= M separate D→H blocks in one matmul)
    - mid/out: ``bmm`` over the M dimension so every head runs together
    - ``chunk_k`` tiles K to keep the ``[B,Kt,M,H]`` activation bounded
    """

    def __init__(
        self,
        d_model: int,
        n_metrics: int,
        *,
        hidden: int = 256,
        n_layers: int = 3,
        chunk_k: int = 1024,
    ):
        super().__init__()
        if int(n_layers) not in (2, 3):
            raise ValueError(f"metric head n_layers must be 2 or 3, got {n_layers}")
        self.n_metrics = int(n_metrics)
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.chunk_k = max(int(chunk_k), 1)
        m, hid = self.n_metrics, self.hidden
        self.in_proj = nn.Linear(int(d_model), m * hid)
        if self.n_layers == 3:
            self.w_mid = nn.Parameter(torch.empty(m, hid, hid))
            self.b_mid = nn.Parameter(torch.zeros(m, hid))
            nn.init.xavier_uniform_(self.w_mid)
        else:
            self.register_parameter("w_mid", None)
            self.register_parameter("b_mid", None)
        self.w_out = nn.Parameter(torch.empty(m, hid, 1))
        self.b_out = nn.Parameter(torch.zeros(m))
        nn.init.xavier_uniform_(self.w_out)

    def _bmm_heads(self, x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        """``x`` [B,K,M,H], ``w`` [M,H,H|1], ``b`` [M,…] → same layout as w @ x."""
        bsz, k, m, hid = x.shape
        # [M, B*K, H] @ [M, H, Hout]
        xt = x.reshape(bsz * k, m, hid).permute(1, 0, 2).contiguous()
        yt = torch.bmm(xt, w)
        if b.ndim == 1:
            yt = yt.squeeze(-1) + b[:, None]
            return yt.permute(1, 0).reshape(bsz, k, m)
        yt = yt + b[:, None, :]
        return yt.permute(1, 0, 2).reshape(bsz, k, m, -1)

    def _forward_tile(self, h: Tensor) -> Tensor:
        b, k, _ = h.shape
        m, hid = self.n_metrics, self.hidden
        x = F.gelu(self.in_proj(h).view(b, k, m, hid))
        if self.w_mid is not None and self.b_mid is not None:
            x = F.gelu(self._bmm_heads(x, self.w_mid, self.b_mid))
        return self._bmm_heads(x, self.w_out, self.b_out)

    def forward(self, h: Tensor) -> Tensor:
        """``h`` [B, K, D] → logits [B, K, M]."""
        k = h.size(1)
        if k <= self.chunk_k:
            return self._forward_tile(h)
        parts = [self._forward_tile(h[:, i : i + self.chunk_k]) for i in range(0, k, self.chunk_k)]
        return torch.cat(parts, dim=1)


class ConvActionTokenizer(nn.Module):
    """1D conv over an action chunk [K, T, C] → one token [K, D]."""

    def __init__(self, in_dim: int, d_model: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, d_model, kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size, padding=pad),
            nn.GELU(),
        )
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        h = self.net(x.transpose(1, 2))
        return self.out(h.mean(dim=-1))


class PCATokenizer(nn.Module):
    """Flattened [K, T, C] → one whitened PCA ``nn.Linear``.

    ``y_i = sqrt(N) (x - mean) · v_i / s_i`` so each component is ~unit
    variance on the vocab, not raw scores of size ~100–400.
    """

    def __init__(self, vocab: Tensor, d_model: int):
        super().__init__()
        if vocab.dim() != 3:
            raise ValueError(f"vocab must be [K,T,C], got {tuple(vocab.shape)}")
        k, t, c = vocab.shape
        flat = vocab.reshape(k, t * c).float()
        mean = flat.mean(dim=0)
        xc = flat - mean
        _u, s, vh = torch.linalg.svd(xc, full_matrices=False)
        n = min(int(d_model), int(vh.size(0)))
        scale = (float(k) ** 0.5) / s[:n].clamp_min(s[0] * 1e-6)
        weight = torch.zeros(d_model, t * c, dtype=torch.float32)
        weight[:n] = vh[:n] * scale.unsqueeze(1)
        bias = -(mean @ weight.T)
        self.proj = nn.Linear(t * c, d_model, bias=True)
        self.proj.weight.data.copy_(weight)
        self.proj.bias.data.copy_(bias)
        self.proj.weight.requires_grad_(False)
        self.proj.bias.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x.reshape(x.size(0), -1))


class HydraMLPTokenizer(nn.Module):
    """Hydra-MDP: flatten each vocab traj, MLP → query, plus a learned mode ID.

    Sequence models + LayerNorm collapse near-duplicate modes. A linear
    MLP on scaled xy keeps endpoint/speed, and the ID embedding keeps the
    4096 queries distinct so CE can actually move.
    """

    def __init__(self, k: int, t: int, c: int, d_model: int, xy_scale: float = 10.0):
        super().__init__()
        self.xy_scale = float(xy_scale)
        # Konev / Hydra-MDP tokenizer: flattened xy only (heading is not an input).
        del c
        self.mlp = nn.Sequential(
            nn.Linear(t * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.id_emb = nn.Embedding(k, d_model)
        # IDs must be O(1), not 0.02, or ego addition swamps them and all queries tie.
        nn.init.normal_(self.id_emb.weight, std=0.5)

    def forward(self, x: Tensor) -> Tensor:
        y = x[..., :2].float() / self.xy_scale
        return self.mlp(y.reshape(y.size(0), -1)) + self.id_emb.weight


def _sinusoidal_pe(max_t: int, d_model: int) -> Tensor:
    pe = torch.zeros(1, max_t, d_model)
    pos = torch.arange(max_t).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div)
    return pe


class TrajTransformer(nn.Module):
    """Self-attention over time on each vocab trajectory, plus sinusoidal PE.

    Vocab modes are scene-independent, so this runs once on ``[K, T, C]`` and
    the resulting ``[K, D]`` tokens are broadcast across the batch. Dropout is
    off: the same mode must map to the same token every step.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        nhead: int,
        d_ffn: int,
        dropout: float,
        num_layers: int,
        max_t: int,
        chunk: int,
    ):
        super().__init__()
        self.chunk = max(int(chunk), 1)
        self.in_proj = nn.Linear(in_dim, d_model)
        self.register_buffer("pos", _sinusoidal_pe(max_t, d_model), persistent=False)
        scale = torch.ones(in_dim)
        scale[: min(2, in_dim)] = 10.0
        self.register_buffer("in_scale", scale, persistent=False)
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, nhead, d_ffn, dropout) for _ in range(num_layers)]
        )

    def _encode(self, x: Tensor) -> Tensor:
        h = self.in_proj(x / self.in_scale) + self.pos[:, : x.size(1)]
        for layer in self.layers:
            h = layer(h)
        return h[:, -1]

    def forward(self, x: Tensor) -> Tensor:
        # fp32: AMP + raw meters in attention NaN'd around step 17 last run.
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            k = x.size(0)
            chunk = min(self.chunk, k)
            parts = []
            for i in range(0, k, chunk):
                xi = x[i : i + chunk]
                if self.training:
                    parts.append(checkpoint(self._encode, xi, use_reentrant=False))
                else:
                    parts.append(self._encode(xi))
            return torch.cat(parts, dim=0)


def _gumbel_noise(scores: Tensor) -> Tensor:
    u = torch.rand_like(scores).clamp(min=1e-6, max=1.0 - 1e-6)
    return -torch.log(-torch.log(u))


def nms_global_indices(
    scores: Tensor,
    glob_idx: Tensor,
    points: Tensor,
    n: int,
    radius: float,
) -> Tensor:
    """Greedy endpoint NMS. ``scores``/``glob_idx`` [B,K], ``points`` [K_full,2] → [B,n].

    Score top-k keeps near-duplicates. NMS keeps modes whose endpoints are at
    least ``radius`` metres apart, then fills with next-best unique ids if
    fewer than ``n`` packs exist.
    """
    b, k = scores.shape
    n = min(int(n), k)
    if n <= 0:
        return glob_idx[:, :0]
    if radius <= 0 or n >= k:
        return glob_idx.gather(1, scores.topk(n, dim=-1).indices)
    sc = scores.detach().float().cpu().numpy()
    ids = glob_idx.detach().cpu().numpy()
    pts = points.detach().float().cpu().numpy()
    r2 = float(radius) * float(radius)
    out = np.empty((b, n), dtype=np.int64)
    for bi in range(b):
        order = np.argsort(-sc[bi], kind="stable")
        ranked = ids[bi, order]
        kept: list[int] = []
        kept_xy: list[np.ndarray] = []
        leftover: list[int] = []
        for gid in ranked:
            gi = int(gid)
            p = pts[gi]
            if kept_xy:
                d2 = ((np.stack(kept_xy, axis=0) - p) ** 2).sum(axis=-1)
                if np.any(d2 < r2):
                    leftover.append(gi)
                    continue
            kept.append(gi)
            kept_xy.append(p)
            if len(kept) >= n:
                break
        if len(kept) < n:
            seen = set(kept)
            for gi in leftover:
                if gi not in seen:
                    kept.append(gi)
                    seen.add(gi)
                    if len(kept) >= n:
                        break
            if len(kept) < n:
                for gi in ranked:
                    gii = int(gi)
                    if gii not in seen:
                        kept.append(gii)
                        seen.add(gii)
                        if len(kept) >= n:
                            break
        out[bi] = np.asarray(kept[:n], dtype=np.int64)
    return torch.as_tensor(out, device=scores.device, dtype=glob_idx.dtype)


def _int_tuple(xs) -> tuple[int, ...]:
    if xs is None:
        return ()
    return tuple(int(x) for x in xs)


class HydraDecoder(DecoderBase):
    name = "hydra"

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        vocab_path: str | None = None,
        vocab: np.ndarray | Tensor | None = None,
        vocab_size: int = 4096,
        future_steps: int = 80,
        decoder_layers: int = 2,
        temperature: float = 1.0,
        soft_k: int | None = None,
        query_chunk: int = 256,
        front_weight: float = 2.0,
        collision_weight: float = 8.0,
        metric_heads: bool = False,
        weights: dict | None = None,
        prune_to: tuple[int, ...] | list[int] | None = (),
        gumbel_tau: float = 0.0,
        tokenizer: str = "mlp",
        cross_only: bool = True,
        conv_kernel: int = 5,
        main: str = "pdms",
        # Which policy logits feed hydra_mdp_w / argmax: imitation | epo.
        # Only matters when both heads exist (main=mix, or always dual).
        policy_head: str = "imitation",
        # Loss scales (NOT the hydra_mdp_w selection weights in ``weights``).
        imitation_loss_scale: float = 1.0,
        epo_loss_scale: float = 1.0,
        # Collapse floor for EPO: squared penalty only when H < ln(epo_entropy_min_k).
        epo_entropy: float = 0.05,
        epo_temperature: float = 1.0,
        epo_entropy_min_k: float = 256.0,
        metric_stopgrad: bool = True,
        # Scale on metric BCE before it is added to the policy loss (unfreeze-safe).
        metric_loss_scale: float = 1.0,
        # Balanced BCE: w1=1/(2p), w0=1/(2(1-p)); rare class (usually y=0) upweighted.
        metric_class_weighted: bool = True,
        # EMA momentum for per-stage P(y=1). Separate buffer per LDiP stage K.
        # 0 disables EMA (batch rates only).
        metric_rate_ema: float = 0.99,
        # Clamp each of w0/w1 when unfreezing trunk; None = uncapped.
        metric_pos_weight_cap: float | None = None,
        metric_head_layers: int = 3,
        metric_head_hidden: int = 256,
        pdms_scale: float = 0.0,
        # LDiP / Hydra-MDP: 4 s @ 10 Hz. Vocab is 8 s; we commit the first 4 s.
        h_exec: int = 40,
        cl_select: str = "hydra_mdp_w",
        keep_winner: bool = False,
        nms_radius: float = 0.0,
        hard_last: bool = False,
        distance: str = "sse",
        imi_times: tuple[int, ...] | list[int] | None = None,
        traj_mlp_layers: int = 2,
        traj_encoder_layers: int = 2,
        traj_chunk: int = 256,
        xy_scale: float = 10.0,
        vocab_self_attn_layers: int = 0,
        # DIPP-style: traj queries cross-attend agent bank then map bank.
        dual_cross: bool = False,
        # map → ego → neighbours (needs encoder ego_tokens). Overrides dual order.
        triple_cross: bool = False,
        # If true with triple_cross: parallel map/ego/agent cross + concat MLP fuse
        # per layer (instead of sequential residual crosses).
        triple_parallel: bool = False,
        # Fuse ego into each traj query: concat(traj, ego)→MLP + traj skip.
        # Set false for traj-only queries (ego only via KV / triple_cross).
        ego_on_query: bool = True,
        **_: object,
    ):
        super().__init__()
        self.temperature = temperature
        self.h_exec = max(int(h_exec), 1)
        self.cl_select = str(cl_select or "argmax")
        self.keep_winner = bool(keep_winner)
        self.nms_radius = float(nms_radius)
        self.hard_last = bool(hard_last)
        self.soft_k = soft_k
        self.query_chunk = max(int(query_chunk), 1)
        self.dual_cross = bool(dual_cross)
        self.triple_cross = bool(triple_cross)
        self.triple_parallel = bool(triple_parallel)
        self.ego_on_query = bool(ego_on_query)
        if self.triple_parallel and not self.triple_cross:
            raise ValueError("decoder.triple_parallel=true requires decoder.triple_cross=true")
        if self.triple_cross and not self.dual_cross:
            # triple implies factorized banks; keep dual_cross flag true for asserts.
            self.dual_cross = True
        # Q = traj + MLP([traj; ego]); residual keeps mode identity (vs bare add).
        self.ego_traj_fuse = (
            MLP(2 * d_model, d_model, d_model, dropout) if self.ego_on_query else None
        )
        self.front_weight = front_weight
        self.collision_weight = collision_weight
        self.metric_heads_enabled = metric_heads
        self.main = str(main or "pdms").lower()
        if self.main not in ("pdms", "imitation", "epo", "mix"):
            raise ValueError(
                f"decoder.main must be 'pdms', 'imitation', 'epo', or 'mix', got {main!r}"
            )
        self.policy_head = str(policy_head or "imitation").lower()
        if self.policy_head not in ("imitation", "epo"):
            raise ValueError(
                f"decoder.policy_head must be 'imitation' or 'epo', got {policy_head!r}"
            )
        if self.main == "pdms":
            default_w = {"pdms": 1.0}
        elif self.main == "epo":
            default_w = {"epo": 1.0}
        elif self.main == "mix":
            default_w = {"imitation": 1.0, "epo": 1.0}
        else:
            default_w = {"imitation": 1.0}
        self.sel_weights = weights or default_w
        self.imitation_loss_scale = float(imitation_loss_scale)
        self.epo_loss_scale = float(epo_loss_scale)
        self.epo_entropy = float(epo_entropy)
        self.epo_temperature = float(epo_temperature)
        self.epo_entropy_min_k = float(epo_entropy_min_k)
        self.metric_stopgrad = bool(metric_stopgrad)
        self.metric_loss_scale = float(metric_loss_scale)
        self.metric_class_weighted = bool(metric_class_weighted)
        self.metric_rate_ema = float(metric_rate_ema)
        self.metric_pos_weight_cap = (
            None if metric_pos_weight_cap is None else float(metric_pos_weight_cap)
        )
        self.metric_head_layers = int(metric_head_layers)
        self.metric_head_hidden = int(metric_head_hidden)
        self.pdms_scale = float(pdms_scale)
        self.distance = str(distance or "sse").lower()
        if self.distance not in ("ade", "sse", "flatten_l2"):
            raise ValueError(f"decoder.distance must be 'ade', 'sse', or 'flatten_l2', got {distance!r}")
        self.vocab_self_attn_layers = max(int(vocab_self_attn_layers), 0)
        self.imi_times = None if imi_times is None else tuple(int(i) for i in imi_times)
        self.prune_to = _int_tuple(prune_to)
        self.gumbel_tau = float(gumbel_tau)
        self.tokenizer_name = str(tokenizer)
        self.cross_only = bool(cross_only)
        if vocab is None:
            vocab = self._load_or_random(vocab_path, vocab_size, future_steps)
            if (
                vocab_path
                and Path(vocab_path).exists()
                and vocab.shape[0] < 32
                and int(vocab_size) >= 256
            ):
                raise ValueError(
                    f"Hydra vocab {vocab_path} has K={vocab.shape[0]} proposals, but "
                    f"decoder.vocab_size={vocab_size}. Rebuild with "
                    "scripts/build_hydra_vocab.py --k 4096 (this is usually a leftover "
                    "smoke-test file)."
                )
        vocab_t = torch.as_tensor(vocab, dtype=torch.float32)
        if vocab_t.dim() != 3:
            raise ValueError(f"vocab must be [K,T,3] or [K,T,2], got {tuple(vocab_t.shape)}")
        if vocab_t.size(-1) == 2:
            z = torch.zeros(*vocab_t.shape[:-1], 1)
            vocab_t = torch.cat([vocab_t, z], dim=-1)
        self.register_buffer("vocab", vocab_t)  # [K,T,3]
        self.register_buffer("vocab_end_xy", vocab_t[:, -1, :2].contiguous())
        k, t, c = vocab_t.shape
        self.future_steps = t
        stages = self._stage_sizes(k)
        self.traj_mlp_layers = max(int(traj_mlp_layers), 0)
        self.traj_encoder_layers = max(int(traj_encoder_layers), 1)
        print(
            f"Hydra decoder: {k} trajectory proposals (T={t}) stages={stages} "
            f"tokenizer={self.tokenizer_name} "
            f"cross_only={self.cross_only} dual_cross={self.dual_cross} "
            f"triple_cross={self.triple_cross} triple_parallel={self.triple_parallel} "
            f"ego_on_query={self.ego_on_query} "
            f"main={self.main} "
            f"policy_head={self.policy_head} "
            f"imi_scale={self.imitation_loss_scale:g} epo_scale={self.epo_loss_scale:g} "
            f"epo_ent={self.epo_entropy:g} epo_tau={self.epo_temperature:g} "
            f"epo_ent_min_k={self.epo_entropy_min_k:g} "
            f"metric_stopgrad={self.metric_stopgrad} "
            f"metric_loss_scale={self.metric_loss_scale:g} "
            f"metric_class_weighted={self.metric_class_weighted} "
            f"metric_rate_ema={self.metric_rate_ema:g} "
            f"metric_pos_weight_cap={self.metric_pos_weight_cap} "
            f"nms_radius={self.nms_radius:g} keep_winner={self.keep_winner} "
            f"vocab_self_attn={self.vocab_self_attn_layers} "
            f"metric_mlp={self.metric_head_layers}x{self.metric_head_hidden}"
        )
        if self.tokenizer_name == "conv":
            self.embed = ConvActionTokenizer(c, d_model, kernel_size=int(conv_kernel))
        elif self.tokenizer_name == "mlp":
            self.embed = HydraMLPTokenizer(k, t, c, d_model, xy_scale=float(xy_scale))
        elif self.tokenizer_name == "pca":
            self.embed = PCATokenizer(vocab_t, d_model)
        elif self.tokenizer_name == "traj":
            self.embed = TrajTransformer(
                c,
                d_model,
                nhead,
                d_ffn,
                dropout=0.0,
                num_layers=self.traj_encoder_layers,
                max_t=t,
                chunk=int(traj_chunk),
            )
        else:
            raise ValueError(
                f"tokenizer must be 'conv', 'mlp', 'pca', or 'traj', got {self.tokenizer_name!r}"
            )
        self.vocab_encoder = nn.ModuleList(
            [EncoderBlock(d_model, nhead, d_ffn, 0.0) for _ in range(self.vocab_self_attn_layers)]
        )
        if self.triple_cross:
            if self.triple_parallel:
                block = (
                    ParallelTripleCrossOnlyDecoderBlock
                    if self.cross_only
                    else ParallelTripleDecoderBlock
                )
            else:
                block = TripleCrossOnlyDecoderBlock if self.cross_only else TripleDecoderBlock
        elif self.dual_cross:
            block = DualCrossOnlyDecoderBlock if self.cross_only else DualDecoderBlock
        else:
            block = CrossOnlyDecoderBlock if self.cross_only else DecoderBlock
        self.layers = nn.ModuleList(
            [block(d_model, nhead, d_ffn, dropout) for _ in range(decoder_layers)]
        )
        # Dual policy heads: imitation (human soft-CE) and EPO (reward max).
        # Always allocate both so main / policy_head knobs can switch without
        # rebuilding; unused head simply gets no loss / no selection.
        self.im_head = nn.Linear(d_model, 1)
        self.epo_head = nn.Linear(d_model, 1)
        if metric_heads:
            self.metric_head = ParallelMetricHeads(
                d_model,
                len(METRIC_NAMES),
                hidden=self.metric_head_hidden,
                n_layers=self.metric_head_layers,
                chunk_k=max(self.query_chunk, 1),
            )
            # Per-stage EMA of P(y=1): LDiP cascade has different label mixes at
            # K=4096 vs 256 vs 16, so rates must not share one running average.
            n_m = len(METRIC_NAMES)
            self.register_buffer(
                "metric_pos_rate_ema",
                torch.full((len(stages), n_m), 0.5),
            )
            self.register_buffer(
                "metric_pos_rate_ema_init",
                torch.zeros(len(stages)),
            )
        else:
            self.metric_head = None

    def _stage_sizes(self, k_full: int) -> tuple[int, ...]:
        sizes = [int(k_full)]
        for raw in self.prune_to:
            nxt = min(max(int(raw), 1), sizes[-1])
            if nxt < sizes[-1]:
                sizes.append(nxt)
        return tuple(sizes)

    @torch.no_grad()
    def _metric_pos_rate_for_stage(self, stage_k: int, labels: Tensor) -> Tensor | None:
        """Per-stage P(y=1) for balanced BCE; EMA keyed by cascade K.

        Stages (4096 → 256 → 16) see different failure rates after pruning, so
        each K keeps its own EMA. Returns ``None`` when class weighting is off.
        """
        if not self.metric_class_weighted:
            return None
        batch_rate = labels.detach().float().mean(dim=tuple(range(labels.ndim - 1)))
        if (
            self.metric_rate_ema <= 0
            or self.metric_head is None
            or not hasattr(self, "metric_pos_rate_ema")
        ):
            return batch_rate
        stages = self._stage_sizes(int(self.vocab.size(0)))
        try:
            si = stages.index(int(stage_k))
        except ValueError:
            return batch_rate
        if float(self.metric_pos_rate_ema_init[si]) < 0.5:
            self.metric_pos_rate_ema[si].copy_(batch_rate)
            self.metric_pos_rate_ema_init[si].fill_(1.0)
            return batch_rate
        rate = self.metric_pos_rate_ema[si].clone()
        decay = float(self.metric_rate_ema)
        self.metric_pos_rate_ema[si].mul_(decay).add_(batch_rate, alpha=1.0 - decay)
        return rate

    @staticmethod
    def _load_or_random(path: str | None, k: int, t: int) -> np.ndarray:
        if path and Path(path).exists():
            arr = np.load(path)
            return arr.astype(np.float32)
        rng = np.random.default_rng(0)
        speed = rng.uniform(2.0, 16.0, size=(k, 1))
        yaw_rate = rng.normal(0.0, 0.05, size=(k, 1))
        steps = np.arange(1, t + 1)[None, :] * 0.1
        heading = yaw_rate * steps
        x = speed * steps * np.cos(heading)
        y = speed * steps * np.sin(heading)
        return np.stack([x, y, heading], axis=-1).astype(np.float32)

    def _traj_tokens(self) -> Tensor:
        """One [K, D] token per vocab mode; broadcast later, not per sample.

        Paper Eq. 6: self-attn over the vocab once (``[1,K,D]``), then ego is
        added in ``_action_queries``. Not per-batch ``[B,K,D]`` (that OOMs).
        """
        z = self.embed(self.vocab)
        if self.vocab_encoder:
            h = z.unsqueeze(0)
            for layer in self.vocab_encoder:
                h = layer(h)
            z = h.squeeze(0)
        return z

    def _action_queries(self, enc: SceneEncoding, traj: Tensor) -> Tensor:
        """Per-mode query: traj only, or ``traj + MLP(concat(traj, ego))``."""
        b = enc.tokens.size(0)
        q = traj.unsqueeze(0).expand(b, -1, -1)
        if not self.ego_on_query or self.ego_traj_fuse is None:
            return q
        ego = enc.ego_state.unsqueeze(1).expand_as(q)
        return q + self.ego_traj_fuse(torch.cat([q, ego], dim=-1))

    def _fuse_traj(self, h: Tensor, traj_b: Tensor) -> Tensor:
        # Heads see decoder output V'' only. Residual blocks already keep q.
        return h

    @staticmethod
    def _pad_mask(mask: Tensor | None) -> Tensor | None:
        if mask is None:
            return None
        return ~mask if mask.dtype == torch.bool else mask < 0.5

    def _decode_queries(self, q: Tensor, enc: SceneEncoding) -> Tensor:
        chunk = min(self.query_chunk, q.size(1))
        if self.triple_cross:
            if (
                enc.agent_tokens is None
                or enc.map_tokens is None
                or enc.ego_tokens is None
            ):
                raise ValueError(
                    "decoder.triple_cross=true requires factorized encoder with "
                    "ego_tokens / agent_tokens / map_tokens"
                )
            agent_kv, agent_pad = compact_kv_pad(
                enc.agent_tokens, self._pad_mask(enc.agent_mask)
            )
            map_kv, map_pad = compact_kv_pad(
                enc.map_tokens, self._pad_mask(enc.map_mask)
            )
            ego_kv, ego_pad = compact_kv_pad(
                enc.ego_tokens, self._pad_mask(enc.ego_mask)
            )

            def _run_tri(qi: Tensor) -> Tensor:
                for layer in self.layers:
                    qi = layer(
                        qi,
                        map_kv,
                        ego_kv,
                        agent_kv,
                        map_padding_mask=map_pad,
                        ego_padding_mask=ego_pad,
                        agent_padding_mask=agent_pad,
                    )
                return qi

            if chunk >= q.size(1):
                return _run_tri(q)
            parts = [_run_tri(q[:, i : i + chunk]) for i in range(0, q.size(1), chunk)]
            return torch.cat(parts, dim=1)

        if self.dual_cross:
            if enc.agent_tokens is None or enc.map_tokens is None:
                raise ValueError(
                    "decoder.dual_cross=true requires encoder.factorized=true "
                    "(missing agent_tokens / map_tokens on SceneEncoding)"
                )
            agent_kv, agent_pad = compact_kv_pad(
                enc.agent_tokens, self._pad_mask(enc.agent_mask)
            )
            map_kv, map_pad = compact_kv_pad(
                enc.map_tokens, self._pad_mask(enc.map_mask)
            )

            def _run(qi: Tensor) -> Tensor:
                for layer in self.layers:
                    qi = layer(
                        qi,
                        agent_kv,
                        map_kv,
                        agent_padding_mask=agent_pad,
                        map_padding_mask=map_pad,
                    )
                return qi

            if chunk >= q.size(1):
                return _run(q)
            parts = [_run(q[:, i : i + chunk]) for i in range(0, q.size(1), chunk)]
            return torch.cat(parts, dim=1)

        pad = self._pad_mask(enc.token_mask)
        if chunk >= q.size(1):
            for layer in self.layers:
                q = layer(q, enc.tokens, kv_padding_mask=pad)
            return q
        parts = []
        for i in range(0, q.size(1), chunk):
            qi = q[:, i : i + chunk]
            for layer in self.layers:
                qi = layer(qi, enc.tokens, kv_padding_mask=pad)
            parts.append(qi)
        return torch.cat(parts, dim=1)

    def _policy_source(self) -> str:
        """Which head provides policy logits for selection / prune."""
        if self.main == "epo":
            return "epo"
        if self.main == "mix":
            return self.policy_head
        return "imitation"

    def _active_policy_logits(
        self, im_logits: Tensor, epo_logits: Tensor | None
    ) -> Tensor:
        if self._policy_source() == "epo" and epo_logits is not None:
            return epo_logits
        return im_logits

    def _select_logits(
        self,
        im_logits: Tensor,
        metric_logits: Tensor | None,
        epo_logits: Tensor | None = None,
    ) -> Tensor:
        """Hydra-MDP-W cost. Multiplicative NC/DAC; combined (5 TTC + 2 C + 5 EP).

        Paper:  f̃ = -(w_imi log S_im + w_NC log S_NC + w_DAC log S_DAC
                       + w_rule log(5 S_TTC + 2 S_C + 5 S_EP)).
        Independent log S_progress was letting the EP head pick highway modes.

        When both ``weights.imitation`` and ``weights.epo`` are >0 and the epo
        head exists, policy terms are a **weighted average** of the two heads
        (runtime blend). Otherwise the active ``policy_head`` is used alone.
        """
        w = self.sel_weights
        w_im = float(w.get("imitation", 0.0))
        w_epo = float(w.get("epo", 0.0))
        w_pdms = float(w.get("pdms", 0.0))
        blend = (
            self.main != "pdms"
            and epo_logits is not None
            and w_im > 0.0
            and w_epo > 0.0
        )

        metric_w = 0.0
        if metric_logits is not None:
            metric_w = sum(float(w.get(name, 0.0)) for name in METRIC_NAMES) + float(
                w.get("rule", 0.0)
            )

        if self.main == "pdms":
            # Legacy single-head PDMS scores on im_head.
            w_main = w_pdms if w_pdms > 0.0 else w_im
            if metric_logits is None or metric_w == 0.0:
                return im_logits
            scores = torch.sigmoid(im_logits).clamp_min(1e-6)
            cost = -w_main * torch.log(scores)
        elif blend:
            # Runtime weighted average of ADE (im) and EPO heads.
            if metric_logits is None or metric_w == 0.0:
                # No metric bank: mix log-softmax then convert back to logits-ish
                # scores via the blended log-probs (argmax-equivalent up to ties).
                log_im = F.log_softmax(im_logits, dim=-1)
                log_epo = F.log_softmax(epo_logits, dim=-1)
                return w_im * log_im + w_epo * log_epo
            p_im = torch.softmax(im_logits, dim=-1).clamp_min(1e-6)
            p_epo = torch.softmax(epo_logits, dim=-1).clamp_min(1e-6)
            cost = -w_im * torch.log(p_im) - w_epo * torch.log(p_epo)
        else:
            policy = self._active_policy_logits(im_logits, epo_logits)
            src = self._policy_source()
            w_main = w_epo if src == "epo" else w_im
            if w_main <= 0.0:
                w_main = 1.0
            if metric_logits is None or metric_w == 0.0:
                return policy
            scores = torch.softmax(policy, dim=-1).clamp_min(1e-6)
            cost = -w_main * torch.log(scores)

        mprob = torch.sigmoid(metric_logits)
        idx = {name: i for i, name in enumerate(METRIC_NAMES)}
        # Hydra-MDP Eq. 11: only NC and DAC are independent log terms.
        for name in ("nc", "dac"):
            wi = float(w.get(name, 0.0))
            if wi:
                cost = cost - wi * torch.log(mprob[..., idx[name]].clamp_min(1e-6))
        w_ttc = float(w.get("ttc", 0.0))
        w_hc = float(w.get("hc", 0.0))
        w_prog = float(w.get("progress", 0.0))
        w_lk = float(w.get("lk", 0.0))
        if w_ttc or w_hc or w_prog or w_lk:
            combo = (
                w_ttc * mprob[..., idx["ttc"]]
                + w_hc * mprob[..., idx["hc"]]
                + w_prog * mprob[..., idx["progress"]]
                + w_lk * mprob[..., idx["lk"]]
            )
            w_rule = float(w.get("rule", 1.0))
            cost = cost - w_rule * torch.log(combo.clamp_min(1e-6))
        return -cost

    def _score_stages(
        self,
        enc: SceneEncoding,
        *,
        stochastic: bool,
        keep_idx: Tensor | None = None,
        stages: tuple[int, ...] | None = None,
    ) -> tuple[
        list[Tensor], list[Tensor], list[Tensor], list[Tensor | None], list[Tensor]
    ]:
        """Score remaining vocab at each stage.

        Returns (global indices, im logits, epo logits, metric logits, raw prune hits).
        ``keep_idx`` [B] is teacher-forced into every surviving set so the last
        CE still sees the ADE winner. Raw hits are whether it would have
        survived without that force.

        ``stages`` defaults to the LDiP cascade from ``prune_to``. Pass
        ``(K,)`` to score the full vocab once (used at inference).
        """
        traj = self._traj_tokens()
        q_all = self._action_queries(enc, traj)
        b, k_full, d_model = q_all.shape
        traj_b = traj.unsqueeze(0).expand(b, -1, -1)
        stages = self._stage_sizes(k_full) if stages is None else tuple(int(s) for s in stages)
        index = torch.arange(k_full, device=q_all.device).unsqueeze(0).expand(b, -1)
        all_idx: list[Tensor] = []
        all_im: list[Tensor] = []
        all_epo: list[Tensor] = []
        all_m: list[Tensor | None] = []
        raw_hits: list[Tensor] = []
        for t, _k_t in enumerate(stages):
            q = q_all if t == 0 else q_all.gather(1, index.unsqueeze(-1).expand(-1, -1, d_model))
            traj_t = (
                traj_b
                if t == 0
                else traj_b.gather(1, index.unsqueeze(-1).expand(-1, -1, d_model))
            )
            h = self._fuse_traj(self._decode_queries(q, enc), traj_t)
            im_logits = self.im_head(h).squeeze(-1)
            epo_logits = self.epo_head(h).squeeze(-1)
            if self.metric_head is None:
                mlog = None
            else:
                h_m = h.detach() if self.metric_stopgrad else h
                mlog = self.metric_head(h_m)
            all_idx.append(index)
            all_im.append(im_logits)
            all_epo.append(epo_logits)
            all_m.append(mlog)
            if t + 1 >= len(stages):
                break
            # LDiP: Gumbel-Top-k on selection scores (τ>0 at train). With
            # nms_radius=0 this is plain top-k after Gumbel; radius>0 is the
            # geometric-NMS variant (not paper LDiP).
            sel = self._select_logits(im_logits, mlog, epo_logits)
            if stochastic and self.gumbel_tau > 0:
                sel = sel + self.gumbel_tau * _gumbel_noise(sel)
            nxt = nms_global_indices(
                sel, index, self.vocab_end_xy, stages[t + 1], self.nms_radius
            )
            if keep_idx is not None:
                raw_hits.append((nxt == keep_idx.unsqueeze(-1)).any(dim=-1).float())
                missing = ~(nxt == keep_idx.unsqueeze(-1)).any(dim=-1)
                if missing.any():
                    nxt = nxt.clone()
                    nxt[missing, -1] = keep_idx[missing]
            else:
                raw_hits.append(torch.ones(b, device=index.device))
            index = nxt
        return all_idx, all_im, all_epo, all_m, raw_hits

    def infer(self, enc: SceneEncoding, **kwargs) -> TrajectoryBundle:
        rank = str(kwargs.get("rank", "") or "")
        if rank == "pdms":
            return self._infer_pdms(enc)
        # Inference always scores the full vocab — LDiP prune is train-only.
        k_full = int(self.vocab.size(0))
        idx, ims, epos, mlogs, _hits = self._score_stages(
            enc, stochastic=False, stages=(k_full,)
        )
        last_idx, last_im, last_epo, last_m = idx[-1], ims[-1], epos[-1], mlogs[-1]
        extras: dict = {
            "im_logits": last_im,
            "epo_logits": last_epo,
            "stage_index": last_idx,
            "policy_source": self._policy_source(),
        }
        if last_m is not None:
            extras["metric_logits"] = last_m
            extras["epdms"] = pdms_from_metric_logits(last_m)
        plan = self._select_logits(last_im, last_m, last_epo)
        extras["im_scores"] = torch.softmax(last_im, dim=-1)
        extras["epo_scores"] = torch.softmax(last_epo, dim=-1)
        xyh = self.vocab.unsqueeze(0).expand(last_im.size(0), -1, -1, -1)
        return TrajectoryBundle(xyh=xyh, scores=plan, extras=extras)

    def _infer_pdms(self, enc: SceneEncoding) -> TrajectoryBundle:
        """Rank the full vocab by NAVSIM v1 PDMS from metric heads. Imitation off."""
        if self.metric_head is None:
            raise RuntimeError("rank=pdms requires decoder.metric_heads=true")
        traj = self._traj_tokens()
        q = self._action_queries(enc, traj)
        traj_b = traj.unsqueeze(0).expand(q.size(0), -1, -1)
        h = self._fuse_traj(self._decode_queries(q, enc), traj_b)
        im = self.im_head(h).squeeze(-1)
        epo = self.epo_head(h).squeeze(-1)
        h_m = h.detach() if self.metric_stopgrad else h
        mlog = self.metric_head(h_m)
        pdms = pdms_from_metric_logits(mlog)
        extras = {
            "im_logits": im,
            "epo_logits": epo,
            "metric_logits": mlog,
            "pdms": pdms,
        }
        xyh = self.vocab.unsqueeze(0).expand(im.size(0), -1, -1, -1)
        return TrajectoryBundle(xyh=xyh, scores=pdms, extras=extras)

    def train_step(self, enc: SceneEncoding, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        needs_pdms = self.main in ("pdms", "epo", "mix") or self.pdms_scale > 0
        if (
            needs_pdms
            and batch.get("metric_labels") is None
            and int(self.vocab.size(0)) > 256
        ):
            raise RuntimeError(
                "EPO / PDMS / pdms_scale needs batch['metric_labels'] "
                "for K>256 (run scripts/cache_hydra_metrics.py)."
            )
        t_v = int(self.vocab.size(1))
        gt_xy = batch["gt_xy"][:, :t_v]
        gt_valid = batch["gt_valid"][:, :t_v]
        gt_yaw = batch.get("gt_yaw")
        if gt_yaw is not None:
            gt_yaw = gt_yaw[:, :t_v]
        keep = None
        if self.keep_winner and self.main not in ("pdms", "epo"):
            keep = vocab_ade_winner(
                self.vocab[..., :2],
                gt_xy,
                gt_valid,
                front_weight=self.front_weight,
            )
        idx, ims, epos, mlogs, raw_hits = self._score_stages(
            enc, stochastic=self.gumbel_tau > 0, keep_idx=keep
        )
        vocab_xyh = self.vocab
        losses = []
        stage_ces: list[tuple[int, Tensor]] = []
        metric_losses = []
        per_head = []
        stats0 = None
        stats_epo0 = None
        last_i = len(idx) - 1
        s_im = float(self.imitation_loss_scale)
        s_epo = float(self.epo_loss_scale)
        for si, (stage_idx, im_logits, epo_logits, mlog) in enumerate(
            zip(idx, ims, epos, mlogs)
        ):
            vocab_t = vocab_xyh[stage_idx]
            labels_t = None
            if (
                self.main in ("pdms", "epo", "mix")
                or mlog is not None
                or self.pdms_scale > 0
            ):
                labels_t = gather_or_compute_metric_labels(
                    vocab_t[..., :2], batch, stage_idx, pred_yaw=vocab_t[..., 2]
                )
            pdms_t = pdms_from_probs(labels_t) if labels_t is not None else None
            stage_loss = im_logits.new_zeros(())
            stats = None
            if self.main == "pdms":
                if pdms_t is None:
                    raise RuntimeError("PDMS train_step missing metric labels")
                stats = hydra_pdms_loss(
                    im_logits,
                    pdms_t,
                    temperature=self.temperature,
                    soft_k=self.soft_k,
                )
                stage_loss = stage_loss + stats["loss"]
            if self.main in ("imitation", "mix"):
                local = None
                if keep is not None:
                    local = (stage_idx == keep.unsqueeze(-1)).float().argmax(dim=-1)
                pdms_for_imi = pdms_t if self.pdms_scale > 0 else None
                stats = hydra_imitation_loss(
                    im_logits,
                    vocab_t,
                    gt_xy,
                    gt_valid,
                    temperature=self.temperature,
                    soft_k=self.soft_k,
                    front_weight=self.front_weight,
                    collision_weight=self.collision_weight,
                    other_xy=batch.get("other_fut_xy"),
                    other_valid=batch.get("other_fut_valid"),
                    ego_lw=batch.get("ego_lw"),
                    other_lw=batch.get("other_lw"),
                    pdms=pdms_for_imi,
                    pdms_scale=self.pdms_scale,
                    force_idx=local,
                    distance=self.distance,
                    gt_yaw=gt_yaw,
                    imi_times=self.imi_times,
                )
                stage_loss = stage_loss + s_im * stats["loss"]
            if self.main in ("epo", "mix"):
                if pdms_t is None:
                    raise RuntimeError("EPO train_step missing metric labels")
                stats_epo = hydra_epo_loss(
                    epo_logits,
                    pdms_t,
                    entropy_coef=self.epo_entropy,
                    temperature=self.epo_temperature,
                    entropy_min_k=self.epo_entropy_min_k,
                )
                if stats_epo0 is None:
                    stats_epo0 = stats_epo
                stage_loss = stage_loss + s_epo * stats_epo["loss"]
                if stats is None:
                    stats = stats_epo
            # LDiP: equal loss at every scoring stage.
            losses.append(stage_loss)
            stage_ces.append((int(stage_idx.size(1)), stage_loss.detach()))
            if stats0 is None:
                stats0 = stats
            if mlog is not None:
                pos_rate = self._metric_pos_rate_for_stage(
                    int(stage_idx.size(1)), labels_t
                )
                metric_losses.append(
                    hydra_metric_bce(
                        mlog,
                        labels_t,
                        class_weighted=self.metric_class_weighted,
                        pos_weight_cap=self.metric_pos_weight_cap,
                        pos_rate=pos_rate,
                    )
                )
                per_head.append(
                    hydra_metric_bce_per_head(
                        mlog,
                        labels_t,
                        class_weighted=self.metric_class_weighted,
                        pos_weight_cap=self.metric_pos_weight_cap,
                        pos_rate=pos_rate,
                    )
                )
            if self.hard_last and si == last_i and keep is not None:
                local = (stage_idx == keep.unsqueeze(-1)).float().argmax(dim=-1)
                losses.append(F.cross_entropy(im_logits, local))
        loss = torch.stack(losses).sum()
        out = {
            "loss": loss,
            "winner": keep if keep is not None else (
                stats0["winner"] if stats0 is not None else idx[0][:, 0]
            ),
        }
        stage_key = "loss_epo" if self.main == "epo" else "loss_im"
        for k_stage, stage_ce in stage_ces:
            out[f"{stage_key}@{k_stage}"] = stage_ce
        if raw_hits:
            for hit, k_next in zip(raw_hits, self._stage_sizes(int(self.vocab.size(0)))[1:]):
                out[f"hit@{k_next}"] = hit.mean().detach()
        if self.main == "pdms":
            out["loss_pdms"] = stats0["loss_pdms"]
            out["loss_pdms_ce"] = stats0["loss_pdms_ce"]
            out["loss_pdms_bce"] = stats0["loss_pdms_bce"]
            out["pdms_pred"] = stats0["pdms_pred"]
            out["pdms_oracle"] = stats0["pdms_oracle"]
        else:
            if self.main in ("imitation", "mix") and stats0 is not None:
                out["loss_im"] = stats0["loss"].detach()
                out["loss_ade"] = stats0.get("loss_ade", loss.detach() * 0)
                if "ade" in stats0:
                    out["ade"] = stats0["ade"]
            if stats_epo0 is not None:
                out["loss_epo"] = stats_epo0["loss_epo"]
                out["epo_adv"] = stats_epo0["epo_adv"]
                out["epo_regret"] = stats_epo0["epo_regret"]
                out["epo_entropy"] = stats_epo0["epo_entropy"]
                if "epo_ent_pen" in stats_epo0:
                    out["epo_ent_pen"] = stats_epo0["epo_ent_pen"]
                out["pdms_pred"] = stats_epo0["pdms_pred"]
                out["pdms_oracle"] = stats_epo0["pdms_oracle"]
            if self.main == "epo" and stats_epo0 is not None:
                # No imitation head trained — leave loss_im unset so the bar
                # shows epo= instead of a fake im= alias.
                out["loss_ade"] = stats_epo0["loss_ade"]
        if metric_losses:
            bce = torch.stack(metric_losses).sum()
            scale = self.metric_loss_scale
            out["loss"] = out["loss"] + scale * bce
            out["loss_metric"] = bce.detach()
            out["loss_metric_scaled"] = (scale * bce).detach()
            heads = torch.stack(per_head, dim=0).mean(dim=0)
            for name, val in zip(METRIC_NAMES, heads):
                out[f"loss_{name}"] = val.detach()
        return out
