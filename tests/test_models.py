from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from womd_il import constants as C
from womd_il.data.dataset import WOMDPackedDataset, packed_collate
from womd_il.metrics.open_loop import format_open_loop_table, open_loop_metrics
from womd_il.models.decoders.diffusion import DiffusionDecoder
from womd_il.models.decoders.dummy import StraightLineDecoder
from womd_il.models.decoders.gmm import GMMDecoder
from womd_il.losses.hydra import METRIC_NAMES, vocab_ade_winner
from womd_il.models.decoders.hydra import (
    ConvActionTokenizer,
    HydraDecoder,
    HydraMLPTokenizer,
    PCATokenizer,
    TrajTransformer,
    nms_global_indices,
)
from womd_il.models.encoder import WayformerEncoder, pack_valid_tokens
from womd_il.models.pca import PCABasis
from womd_il.models.policy import WayformerPolicy


def _tiny_encoder() -> WayformerEncoder:
    return WayformerEncoder(
        d_model=32,
        d_ffn=64,
        nhead=4,
        dropout=0.0,
        num_latents=8,
        encoder_layers=1,
        hist_in_dim=C.HIST_IN_DIM,
        map_in_dim=C.MAP_IN_DIM,
        tl_in_dim=C.TL_IN_DIM,
        ego_in_dim=C.EGO_IN_DIM,
    )


def _batch(synthetic_h5, pack_cfg):
    ds = WOMDPackedDataset(synthetic_h5, cfg=pack_cfg)
    loader = DataLoader(ds, batch_size=2, collate_fn=packed_collate)
    return next(iter(loader))


def test_pack_valid_tokens_drops_padding():
    tokens = torch.arange(12, dtype=torch.float32).view(2, 3, 2)
    mask = torch.tensor([[True, False, True], [False, False, True]])
    packed, packed_mask = pack_valid_tokens(tokens, mask)
    assert packed.shape[1] == 2
    assert packed_mask[0].tolist() == [True, True]
    assert packed_mask[1].tolist() == [True, False]
    assert packed[1, 0].tolist() == tokens[1, 2].tolist()


def test_encoder_packs_scene_corridor_not_route_xy(synthetic_h5, pack_cfg):
    batch = _batch(synthetic_h5, pack_cfg)
    enc = WayformerEncoder(
        d_model=32,
        d_ffn=64,
        nhead=4,
        dropout=0.0,
        num_latents=0,
        encoder_layers=1,
        hist_in_dim=C.HIST_IN_DIM,
        map_in_dim=C.MAP_IN_DIM,
        tl_in_dim=C.TL_IN_DIM,
        ego_in_dim=C.EGO_IN_DIM,
        route_in_dim=C.ROUTE_IN_DIM,
    )
    enc.eval()
    raw = (
        batch["neighbors"].numel() // batch["neighbors"].shape[-1]
        + batch["map_feat"].size(1)
        + batch["tl_feat"].shape[1] * batch["tl_feat"].shape[2]
        + batch["sdc_hist"].size(1)
    )
    out = enc(batch)
    assert out.token_mask.dtype == torch.bool
    assert int(out.token_mask.sum().item()) <= raw
    assert out.tokens.size(1) == int(out.token_mask.size(1))
    # Optional XY route tokens (default off): enabling them grows the scene.
    enc_rt = WayformerEncoder(
        d_model=32,
        d_ffn=64,
        nhead=4,
        dropout=0.0,
        num_latents=0,
        encoder_layers=1,
        hist_in_dim=C.HIST_IN_DIM,
        map_in_dim=C.MAP_IN_DIM,
        tl_in_dim=C.TL_IN_DIM,
        ego_in_dim=C.EGO_IN_DIM,
        route_in_dim=C.ROUTE_IN_DIM,
        use_route_tokens=True,
    )
    # Default encoder: corridor via map_feat only (no route XY tokens).
    assert enc.use_route_tokens is False
    assert enc._route_tokens(batch) is None
    n_off = int(out.token_mask.sum(dim=1).max().item())
    n_on = int(enc_rt(batch).token_mask.sum(dim=1).max().item())
    assert n_on > n_off


def test_factorized_encoder_and_dual_cross_hydra(synthetic_h5, pack_cfg):
    batch = _batch(synthetic_h5, pack_cfg)
    enc = WayformerEncoder(
        d_model=32,
        d_ffn=64,
        nhead=4,
        dropout=0.0,
        num_latents=0,
        encoder_layers=1,
        hist_in_dim=C.HIST_IN_DIM,
        map_in_dim=C.MAP_IN_DIM,
        tl_in_dim=C.TL_IN_DIM,
        ego_in_dim=C.EGO_IN_DIM,
        route_in_dim=C.ROUTE_IN_DIM,
        factorized=True,
        agent_pool="mean",
    )
    enc.eval()
    out = enc(batch)
    assert out.agent_tokens is not None and out.map_tokens is not None
    assert out.agent_tokens.shape[:2] == out.agent_mask.shape
    assert out.map_tokens.shape[:2] == out.map_mask.shape
    # Time-pooled neighbors only (ego stays on traj queries, not AgentKV).
    a = batch["neighbors"].size(1)
    assert out.agent_tokens.size(1) == a
    assert out.extras.get("factorized") is True

    vocab = np.zeros((8, 80, 3), np.float32)
    vocab[:, :, 0] = np.linspace(0.1, 8.0, 80)
    policy = WayformerPolicy(
        enc,
        HydraDecoder(
            d_model=32,
            nhead=4,
            d_ffn=64,
            dropout=0.0,
            vocab=vocab,
            decoder_layers=1,
            soft_k=4,
            prune_to=(4, 2),
            gumbel_tau=0.5,
            main="imitation",
            tokenizer="mlp",
            dual_cross=True,
            vocab_self_attn_layers=0,
        ),
    )
    stats = policy.train_step(batch)
    assert torch.isfinite(stats["loss"])
    stats["loss"].backward()
    policy.eval()
    bundle = policy.infer(batch)
    assert bundle.xyh.shape[1] == 8


def test_ego_kin_dropout_only_in_train():
    enc = WayformerEncoder(
        d_model=32,
        d_ffn=64,
        nhead=4,
        dropout=0.0,
        num_latents=8,
        encoder_layers=1,
        ego_kin_dropout=1.0,
        drop_ego_hist=True,
    )
    ego = torch.ones(4, C.EGO_IN_DIM)
    enc.train()
    dropped = enc._drop_ego_kinematics(ego)
    assert torch.count_nonzero(dropped[:, :3]) == 0
    enc.eval()
    kept = enc._drop_ego_kinematics(ego)
    assert torch.allclose(kept[:, :3], ego[:, :3])


def test_gmm_modes_depend_on_ego_speed(synthetic_h5, pack_cfg):
    batch = _batch(synthetic_h5, pack_cfg)
    policy = WayformerPolicy(
        _tiny_encoder(),
        GMMDecoder(d_model=32, nhead=4, d_ffn=64, dropout=0.0, num_modes=4, decoder_layers=1),
    )
    policy.eval()
    a = policy.infer(batch).xyh
    batch2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch2["ego_state"] = batch2["ego_state"].clone()
    batch2["ego_state"][:, 0] = 25.0
    b = policy.infer(batch2).xyh
    assert not torch.allclose(a, b, atol=1e-6)


def test_gmm_predicts_absolute_xy_not_cv_residual(synthetic_h5, pack_cfg):
    batch = _batch(synthetic_h5, pack_cfg)
    batch["sdc_hist"] = batch["sdc_hist"].clone()
    batch["sdc_hist"][:, -1, 2] = 25.0
    batch["sdc_hist"][:, -1, 3] = 0.0
    policy = WayformerPolicy(
        _tiny_encoder(),
        GMMDecoder(d_model=32, nhead=4, d_ffn=64, dropout=0.0, num_modes=4, decoder_layers=1),
    )
    policy.eval()
    xy = policy.infer(batch).xyh[0, :, -1, :2]
    # A 25 m/s CV prior would sit ~200 m at 8 s. Direct head starts near 0.
    assert float(xy.abs().max().detach()) < 20.0


def test_gmm_cv_residual_adds_constant_velocity(synthetic_h5, pack_cfg):
    batch = _batch(synthetic_h5, pack_cfg)
    batch["speed"] = torch.full((batch["gt_xy"].size(0),), 10.0)
    dec = GMMDecoder(
        d_model=32, nhead=4, d_ffn=64, dropout=0.0, num_modes=4, decoder_layers=1, cv_residual=True
    )
    policy = WayformerPolicy(_tiny_encoder(), dec)
    policy.eval()
    xy = policy.infer(batch).xyh[0, 0, :, :2]
    # Agent-frame CV at 10 m/s, 8 s → x≈80 m, y≈0.
    assert float(xy[-1, 0].detach()) > 60.0
    assert float(xy[-1, 1].abs().detach()) < 15.0


def test_gmm_train_and_infer(synthetic_h5, pack_cfg):
    batch = _batch(synthetic_h5, pack_cfg)
    policy = WayformerPolicy(
        _tiny_encoder(),
        GMMDecoder(d_model=32, nhead=4, d_ffn=64, dropout=0.0, num_modes=4, decoder_layers=1),
    )
    stats = policy.train_step(batch)
    assert torch.isfinite(stats["loss"])
    stats["loss"].backward()
    bundle = policy.infer(batch)
    assert bundle.xyh.shape == (2, 4, 80, 3)
    m = open_loop_metrics(
        bundle.xyh[..., :2], bundle.scores, batch["gt_xy"], batch["gt_valid"], num_keep=4
    )
    assert "minADE@80" in m


def test_pca_tokenizer_pc1_tracks_speed():
    t = torch.arange(1, 81).float() * 0.1
    vocab = torch.zeros(3, 80, 3)
    vocab[0, :, 0] = 4.0 * t
    vocab[1, :, 0] = 8.0 * t
    vocab[2, :, 0] = 16.0 * t
    tok = PCATokenizer(vocab, d_model=8)
    assert not tok.proj.weight.requires_grad
    assert not tok.proj.bias.requires_grad
    z = tok(vocab)
    # PC1 is the first linear row; faster end-x should be farther from the mean.
    assert z[2, 0].abs() > z[0, 0].abs()
    assert not torch.allclose(z[0], z[2], atol=1e-4)
    assert float(z.detach().abs().max()) < 5.0


def test_hydra_mlp_tokenizer_distinct_ids():
    vocab = torch.zeros(4, 80, 3)
    vocab[:, :, 0] = torch.linspace(0.1, 8.0, 80)
    tok = HydraMLPTokenizer(k=4, t=80, c=3, d_model=16)
    z = tok(vocab)
    assert z.shape == (4, 16)
    # Same geometry, different mode IDs → different queries.
    assert not torch.allclose(z[0], z[1], atol=1e-5)
    # Heading is not an MLP input.
    tok.id_emb.weight.data.zero_()
    same_xy = vocab[:2].clone()
    same_xy[0, :, 2] = 2.0
    same_xy[1, :, 2] = -2.0
    # Tokenizer always adds the full [K,D] ID table; pad to K.
    padded = torch.zeros_like(vocab)
    padded[:2] = same_xy
    z2 = tok(padded)
    assert torch.allclose(z2[0], z2[1], atol=1e-5)


def test_traj_transformer_separates_speeds():
    t = torch.arange(1, 81).float() * 0.1
    vocab = torch.zeros(3, 80, 3)
    vocab[0, :, 0] = 4.0 * t
    vocab[1, :, 0] = 8.0 * t
    vocab[2, :, 0] = 16.0 * t
    tok = TrajTransformer(
        in_dim=3, d_model=16, nhead=4, d_ffn=32, dropout=0.0, num_layers=1, max_t=80, chunk=2
    )
    tok.train()
    z = tok(vocab)
    assert z.shape == (3, 16)
    z.sum().backward()
    assert tok.in_proj.weight.grad is not None
    assert not torch.allclose(z[0], z[2], atol=1e-4)
    tok.eval()
    z_a, z_b = tok(vocab), tok(vocab)
    assert torch.allclose(z_a, z_b)


def test_nms_global_indices_drops_near_dups():
    # Three twin pairs, endpoints 0 / 10 / 20 m. Top scores are the twins at 0.
    scores = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    glob = torch.arange(6).unsqueeze(0)
    pts = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0], [20.0, 0.0], [20.1, 0.0]]
    )
    top = nms_global_indices(scores, glob, pts, n=3, radius=0.0)
    assert top[0].tolist() == [0, 1, 2]
    nms = nms_global_indices(scores, glob, pts, n=3, radius=2.5)
    assert sorted(nms[0].tolist()) == [0, 2, 4]


def test_hydra_train_and_infer(synthetic_h5, pack_cfg):
    batch = _batch(synthetic_h5, pack_cfg)
    vocab = np.zeros((8, 80, 3), np.float32)
    vocab[:, :, 0] = np.linspace(0.1, 8.0, 80)[None, :] * np.linspace(0.5, 1.5, 8)[:, None]
    policy = WayformerPolicy(
        _tiny_encoder(),
        HydraDecoder(
            d_model=32,
            nhead=4,
            d_ffn=64,
            dropout=0.0,
            vocab=vocab,
            decoder_layers=1,
            soft_k=4,
            prune_to=(4, 2),
            gumbel_tau=0.5,
            main="imitation",
            tokenizer="conv",
        ),
    )
    assert isinstance(policy.decoder.embed, ConvActionTokenizer)
    assert not hasattr(policy.decoder.layers[0], "self_attn")
    stats = policy.train_step(batch)
    assert torch.isfinite(stats["loss"])
    assert "hit@4" in stats
    assert "hit@2" in stats
    assert "loss_im@8" in stats
    assert "loss_im@4" in stats
    assert "loss_im@2" in stats
    stats["loss"].backward()
    policy.eval()
    bundle = policy.infer(batch)
    assert bundle.xyh.shape[1] == 8
    # Infer scores the full vocab (LDiP prune is train-only).
    surviving = (bundle.scores > -1e8).sum(dim=-1)
    assert torch.all(surviving == 8)


def test_imitation_force_idx_keeps_winner_in_soft_k():
    from womd_il.losses.hydra import hydra_imitation_loss

    vocab = torch.zeros(8, 80, 2)
    vocab[:, :, 0] = torch.linspace(0.1, 8.0, 80) * torch.linspace(0.5, 2.0, 8)[:, None]
    gt = vocab[7:].clone()
    valid = torch.ones(1, 80)
    logits = torch.zeros(1, 8)
    stats = hydra_imitation_loss(
        logits, vocab, gt, valid, temperature=0.2, soft_k=3, force_idx=torch.tensor([7])
    )
    assert int(stats["winner"].item()) == 7


def test_hydra_infer_plans_with_epdms(synthetic_h5, pack_cfg):
    batch = _batch(synthetic_h5, pack_cfg)
    vocab = np.zeros((8, 80, 3), np.float32)
    vocab[:, :, 0] = np.linspace(0.1, 8.0, 80)
    dec = HydraDecoder(
        d_model=32,
        nhead=4,
        d_ffn=64,
        dropout=0.0,
        vocab=vocab,
        decoder_layers=1,
        prune_to=(4, 2),
        gumbel_tau=0.0,
        metric_heads=True,
        main="imitation",
        pdms_scale=0.0,
    )
    policy = WayformerPolicy(_tiny_encoder(), dec)
    policy.eval()
    bundle = policy.infer(batch)
    assert "epdms" in bundle.extras
    assert bundle.extras["epdms"].shape[-1] == 8
    assert torch.isfinite(bundle.scores).all()


def test_vocab_ade_winner_is_nearest():
    vocab = torch.zeros(3, 80, 2)
    vocab[0, :, 0] = torch.linspace(0.1, 4.0, 80)
    vocab[1, :, 0] = torch.linspace(0.1, 8.0, 80)
    vocab[2, :, 0] = torch.linspace(0.1, 16.0, 80)
    gt = vocab[1].unsqueeze(0)
    valid = torch.ones(1, 80)
    assert int(vocab_ade_winner(vocab, gt, valid).item()) == 1


def test_hydra_metric_heads_train_and_infer(synthetic_h5, pack_cfg):
    batch = _batch(synthetic_h5, pack_cfg)
    vocab = np.zeros((8, 80, 3), np.float32)
    vocab[:, :, 0] = np.linspace(0.1, 8.0, 80)[None, :] * np.linspace(0.5, 1.5, 8)[:, None]
    policy = WayformerPolicy(
        _tiny_encoder(),
        HydraDecoder(
            d_model=32,
            nhead=4,
            d_ffn=64,
            dropout=0.0,
            vocab=vocab,
            decoder_layers=1,
            soft_k=4,
            prune_to=(4, 2),
            gumbel_tau=0.0,
            metric_heads=True,
            main="pdms",
            weights={"pdms": 1.0, "nc": 0.0, "dac": 0.0, "ttc": 0.0, "comfort": 0.0, "progress": 0.0},
        ),
    )
    assert policy.decoder.metric_head is not None
    assert policy.decoder.main == "pdms"
    stats = policy.train_step(batch)
    assert "loss_pdms" in stats
    assert "loss_metric" in stats
    assert torch.isfinite(stats["loss"])
    stats["loss"].backward()
    assert policy.decoder.metric_stopgrad is False
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in policy.decoder.metric_head.parameters()
    )
    policy.eval()
    bundle = policy.infer(batch)
    assert "metric_logits" in bundle.extras
    from womd_il.losses.hydra import METRIC_NAMES

    assert bundle.extras["metric_logits"].shape[-1] == len(METRIC_NAMES)
    pdms = policy.infer(batch, rank="pdms")
    k = policy.decoder.vocab.size(0)
    assert pdms.scores.shape == (bundle.scores.shape[0], k)
    assert "pdms" in pdms.extras
    assert torch.isfinite(pdms.scores).all()
    assert (pdms.scores >= 0).all()


def test_hydra_infer_is_deterministic(synthetic_h5, pack_cfg):
    batch = _batch(synthetic_h5, pack_cfg)
    vocab = np.linspace(0.1, 8.0, 80, dtype=np.float32)[None, :, None]
    vocab = np.broadcast_to(vocab * np.linspace(0.5, 1.5, 16)[:, None, None], (16, 80, 3)).copy()
    dec = HydraDecoder(
        d_model=32,
        nhead=4,
        d_ffn=64,
        dropout=0.0,
        vocab=vocab,
        decoder_layers=1,
        prune_to=(8, 2),
        gumbel_tau=1.0,
        main="imitation",
    )
    policy = WayformerPolicy(_tiny_encoder(), dec)
    policy.eval()
    a = policy.infer(batch).scores
    b = policy.infer(batch).scores
    assert torch.allclose(a, b)


def test_hydra_vocab_self_attn_shape(synthetic_h5, pack_cfg):
    vocab = np.zeros((8, 40, 3), np.float32)
    vocab[:, :, 0] = np.linspace(0.1, 4.0, 40)
    dec = HydraDecoder(
        d_model=32,
        nhead=4,
        d_ffn=64,
        dropout=0.0,
        vocab=vocab,
        decoder_layers=1,
        prune_to=(),
        main="imitation",
        vocab_self_attn_layers=2,
    )
    z = dec._traj_tokens()
    assert z.shape == (8, 32)
    assert len(dec.vocab_encoder) == 2
    dec.query_chunk = 3
    batch = _batch(synthetic_h5, pack_cfg)
    policy = WayformerPolicy(_tiny_encoder(), dec)
    stats = policy.train_step(batch)
    assert torch.isfinite(stats["loss"])
    policy.eval()
    bundle = policy.infer(batch)
    assert bundle.scores.shape[-1] == 8


def test_hydra_oneshot_scores_full_vocab(synthetic_h5, pack_cfg):
    batch = _batch(synthetic_h5, pack_cfg)
    vocab = np.zeros((8, 80, 3), np.float32)
    vocab[:, :, 0] = np.linspace(0.1, 8.0, 80)
    policy = WayformerPolicy(
        _tiny_encoder(),
        HydraDecoder(
            d_model=32,
            nhead=4,
            d_ffn=64,
            dropout=0.0,
            vocab=vocab,
            decoder_layers=1,
            prune_to=(),
            gumbel_tau=0.0,
            main="imitation",
        ),
    )
    assert isinstance(policy.decoder.embed, HydraMLPTokenizer)
    stats = policy.train_step(batch)
    stats["loss"].backward()
    assert policy.decoder.embed.mlp[0].weight.grad is not None
    assert policy.decoder.embed.id_emb.weight.grad is not None
    assert "loss_im@8" in stats
    assert "loss_im@4" not in stats
    assert "hit@256" not in stats
    assert "hit@16" not in stats
    policy.eval()
    bundle = policy.infer(batch)
    surviving = (bundle.scores > -1e8).sum(dim=-1)
    assert torch.all(surviving == 8)


def test_hydra_mdp_w_combines_ttc_comfort_progress():
    vocab = np.zeros((4, 80, 3), np.float32)
    vocab[:, :, 0] = np.linspace(0.1, 8.0, 80)
    dec = HydraDecoder(
        d_model=32,
        nhead=4,
        d_ffn=64,
        dropout=0.0,
        vocab=vocab,
        decoder_layers=1,
        prune_to=(),
        gumbel_tau=0.0,
        metric_heads=True,
        main="imitation",
        weights={
            "imitation": 0.1,
            "nc": 1.0,
            "dac": 1.0,
            "ttc": 5.0,
            "hc": 2.0,
            "progress": 5.0,
            "rule": 1.0,
        },
    )
    im = torch.zeros(1, 4)
    im[0, 0] = 4.0
    m = torch.zeros(1, 4, len(METRIC_NAMES))
    m[0, 0, :] = -8.0
    m[0, 0, -1] = 8.0
    m[0, 1, :] = 8.0
    m[0, 1, -1] = -8.0
    sel = dec._select_logits(im, m)
    assert sel[0, 1] > sel[0, 0]


def test_hydra_rejects_smoke_vocab_file(tmp_path: Path):
    path = tmp_path / "hydra_vocab.npy"
    np.save(path, np.zeros((8, 80, 3), np.float32))
    with pytest.raises(ValueError, match="K=8"):
        HydraDecoder(
            d_model=32,
            nhead=4,
            d_ffn=64,
            dropout=0.0,
            vocab_path=str(path),
            vocab_size=4096,
            decoder_layers=1,
        )


def test_diffusion_train_and_infer(synthetic_h5, pack_cfg):
    batch = _batch(synthetic_h5, pack_cfg)
    pca = PCABasis.fit(batch["gt_xy"].numpy(), k=4)
    policy = WayformerPolicy(
        _tiny_encoder(),
        DiffusionDecoder(
            d_model=32,
            nhead=4,
            d_ffn=64,
            dropout=0.0,
            pca=pca,
            pca_dim=4,
            num_layers=1,
            num_sample_steps=3,
        ),
    )
    stats = policy.train_step(batch)
    assert torch.isfinite(stats["loss"])
    bundle = policy.infer(batch)
    assert bundle.xyh.shape == (2, 1, 80, 3)
    # Warm-start path
    bundle2 = policy.infer(batch, x_init=bundle.xyh[:, 0, :, :2])
    assert bundle2.xyh.shape[-1] == 3


def test_dummy_infer(synthetic_h5, pack_cfg):
    batch = _batch(synthetic_h5, pack_cfg)
    policy = WayformerPolicy(_tiny_encoder(), StraightLineDecoder())
    bundle = policy.infer(batch)
    assert bundle.xyh.shape[1] == 1
    assert bundle.xyh[0, 0, -1, 0] > 0


def test_format_open_loop_table():
    table = format_open_loop_table(
        {
            "minADE@30": 0.31364179253578184,
            "minFDE@30": 0.8287274837493896,
            "miss@30": 0.1329166680574417,
            "minADE@50": 0.8780457139015198,
            "minFDE@50": 2.4681678771972657,
            "miss@50": 0.18687500357627868,
            "minADE@80": 2.3887058019638063,
            "minFDE@80": 6.735514450073242,
            "miss@80": 0.25416666865348814,
            "minADE": 2.3887058019638063,
            "minFDE": 6.735514450073242,
            "oracleADE@80": 2.045500302314758,
            "IoU0.5@30": 0.71,
            "IoU1@30": 0.84,
            "IoU2@30": 0.91,
            "IoU0.5@50": 0.55,
            "IoU1@50": 0.72,
            "IoU2@50": 0.85,
            "IoU0.5@80": 0.41,
            "IoU1@80": 0.63,
            "IoU2@80": 0.78,
            "cvADE@30": 0.90,
            "beatCV@30": 0.80,
            "cvADE@50": 2.10,
            "beatCV@50": 0.70,
            "cvADE@80": 4.20,
            "beatCV@80": 0.61,
        },
        epoch=10,
        lr=2.55e-05,
    )
    assert "val epoch=10 lr=2.55e-05" in table
    assert "horizon" in table and "minADE" in table and "minFDE" in table and "miss" in table
    assert "IoU@0.5" in table and "IoU@1" in table and "IoU@2" in table
    assert "cvADE" in table and "beatCV" in table
    assert "@30" in table and "@50" in table and "@80" in table
    assert "0.314" in table and "6.736" in table
    assert "0.710" in table and "0.780" in table
    assert "4.200" in table and "0.610" in table
    assert "oracleADE@80=2.046" in table
    assert "minADE@" not in table.split("\n")[0]
    assert "IoU0.5" not in table.split("\n")[-1]


def test_hydra_epo_and_mix_one_stage(synthetic_h5, pack_cfg):
    from womd_il.losses.hydra import METRIC_NAMES

    batch = _batch(synthetic_h5, pack_cfg)
    k, t = 8, 40
    vocab = np.zeros((k, t, 3), np.float32)
    vocab[:, :, 0] = np.linspace(0.1, 4.0, t)
    # Cached labels so EPO does not recompute (and works past K>256 later).
    batch = {**batch, "metric_labels": torch.ones(batch["gt_xy"].size(0), k, len(METRIC_NAMES))}

    epo = WayformerPolicy(
        _tiny_encoder(),
        HydraDecoder(
            d_model=32,
            nhead=4,
            d_ffn=64,
            dropout=0.0,
            vocab=vocab,
            decoder_layers=1,
            prune_to=(),
            main="epo",
            policy_head="epo",
            metric_heads=True,
            metric_stopgrad=True,
        ),
    )
    assert epo.decoder._stage_sizes(k) == (k,)
    stats = epo.train_step(batch)
    assert torch.isfinite(stats["loss"])
    assert "loss_epo" in stats
    stats["loss"].backward()
    assert epo.decoder.epo_head.weight.grad is not None
    # Imitation head unused under main=epo → no grad.
    assert epo.decoder.im_head.weight.grad is None
    epo.eval()
    bundle = epo.infer(batch)
    assert "epo_logits" in bundle.extras
    assert bundle.extras["policy_source"] == "epo"

    mix = WayformerPolicy(
        _tiny_encoder(),
        HydraDecoder(
            d_model=32,
            nhead=4,
            d_ffn=64,
            dropout=0.0,
            vocab=vocab,
            decoder_layers=1,
            prune_to=(),
            main="mix",
            policy_head="imitation",
            metric_heads=True,
            weights={"imitation": 1.0, "epo": 0.5},
        ),
    )
    stats = mix.train_step(batch)
    assert torch.isfinite(stats["loss"])
    assert "loss_epo" in stats and "loss_im" in stats
    stats["loss"].backward()
    assert mix.decoder.im_head.weight.grad is not None
    assert mix.decoder.epo_head.weight.grad is not None
    mix.eval()
    assert mix.infer(batch).extras["policy_source"] == "imitation"
    mix.decoder.policy_head = "epo"
    assert mix.infer(batch).extras["policy_source"] == "epo"
