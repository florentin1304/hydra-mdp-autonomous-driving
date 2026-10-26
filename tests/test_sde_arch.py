"""PlanTF SDE + traj↔ego fuse + triple-cross smoke tests."""

from __future__ import annotations

import torch

from womd_il.models.decoders.hydra import HydraDecoder
from womd_il.models.encoder import WayformerEncoder
from womd_il.models.policy import WayformerPolicy
from womd_il.models.sde import StateAttentionEncoder, plantf_state6_from_ego


def test_state_attention_encoder_shapes():
    enc = StateAttentionEncoder(6, dim=32, state_dropout=1.0)
    enc.train()
    x = torch.randn(4, 6)
    assert enc(x).shape == (4, 32)
    enc.eval()
    assert enc(x).shape == (4, 32)


def test_plantf_state6_keeps_pose_zero():
    ego = torch.zeros(2, 12)
    ego[:, 0] = 7.0
    ego[:, 1] = -1.0
    ego[:, 2] = 2.0
    s = plantf_state6_from_ego(ego)
    assert s.shape == (2, 6)
    assert torch.allclose(s[:, :3], torch.zeros(2, 3))
    assert torch.allclose(s[:, 3:], ego[:, :3])


def test_ego_traj_fuse_residual(synthetic_h5, pack_cfg):
    from womd_il.data.dataset import WOMDPackedDataset
    from womd_il.data.features import build_sample_features
    from womd_il.sim.obs import features_to_batch

    ds = WOMDPackedDataset(synthetic_h5, cfg=pack_cfg)
    feat = build_sample_features(ds.get_world_sample(0), target_idx=0)
    batch = features_to_batch(feat, "cpu")

    enc = WayformerEncoder(
        d_model=32,
        d_ffn=64,
        nhead=4,
        dropout=0.0,
        num_latents=0,
        encoder_layers=1,
        factorized=True,
        agent_pool="lstm",
        agent_lstm_layers=1,
        ego_encoder="sde",
        ego_in_social=True,
    )
    dec = HydraDecoder(
        d_model=32,
        nhead=4,
        d_ffn=64,
        dropout=0.0,
        decoder_layers=1,
        vocab_size=8,
        vocab_path=None,
        cross_only=True,
        dual_cross=True,
        triple_cross=True,
        ego_on_query=True,
        metric_heads=False,
        prune_to=[],
    )
    assert dec.ego_traj_fuse is not None
    policy = WayformerPolicy(enc, dec)
    policy.eval()
    with torch.no_grad():
        scene = policy.encode(batch)
        traj = dec._traj_tokens()
        q = dec._action_queries(scene, traj)
        assert q.shape == (1, traj.size(0), 32)
        out = policy.infer(batch)
    assert out.scores.shape[0] == 1
