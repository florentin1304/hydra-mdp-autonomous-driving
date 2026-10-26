from __future__ import annotations

import numpy as np
import torch

from womd_il.data.dataset import WOMDPackedDataset
from womd_il.metrics.closed_loop import closed_loop_metrics
from womd_il.models.decoders.dummy import StraightLineDecoder
from womd_il.models.decoders.gmm import GMMDecoder
from womd_il.models.encoder import WayformerEncoder
from womd_il.models.policy import WayformerPolicy
from womd_il.sim.bicycle import bicycle_inverse, bicycle_step, traj_to_actions
from womd_il.sim.rollout import rollout_policy


def test_bicycle_inverse_roundtrip():
    x, y, yaw, speed = 0.0, 0.0, 0.2, 8.0
    accel, kappa = 1.0, 0.05
    nx, ny, nyaw, nspeed = bicycle_step(x, y, yaw, speed, accel, kappa)
    a2, k2 = bicycle_inverse(x, y, yaw, speed, nx, ny, nyaw)
    nx2, ny2, nyaw2, nspeed2 = bicycle_step(x, y, yaw, speed, a2, k2)
    assert abs(nx2 - nx) < 0.05
    assert abs(ny2 - ny) < 0.05


def test_offroad_lane_gate_matches_dac():
    """Far from on-route lane → offroad even without an edge cross (DAC parity)."""
    from womd_il.sim.replay import _offroad

    # One on-route lane along +x; no road edges.
    map_xyz = np.zeros((1, 4, 3), np.float32)
    map_xyz[0, :, 0] = [0.0, 10.0, 20.0, 30.0]
    sample = {
        "map_type": np.array([2], np.int32),  # MAP_LANE_SURFACE
        "map_xyz": map_xyz,
        "map_valid": np.ones((1, 4), bool),
        "map_on_route": np.array([True]),
    }
    assert _offroad(sample, np.array([5.0, 0.0]), prev_xy=np.array([4.0, 0.0])) is False
    assert _offroad(sample, np.array([5.0, 5.0]), prev_xy=np.array([5.0, 4.5])) is True


def _tiny_policy(decoder):
    enc = WayformerEncoder(
        d_model=32,
        d_ffn=64,
        nhead=4,
        dropout=0.0,
        num_latents=8,
        encoder_layers=1,
    )
    return WayformerPolicy(enc, decoder)


def test_dummy_rollout_log_and_idm(synthetic_h5, pack_cfg):
    ds = WOMDPackedDataset(synthetic_h5, cfg=pack_cfg)
    sample = ds.get_world_sample(0)
    policy = _tiny_policy(StraightLineDecoder())
    for neigh in ("log", "idm"):
        state = rollout_policy(policy, sample, neighbors=neigh, horizon=12, device="cpu", h_exec=1)
        assert len(state.executed_xy) == 13  # current + 12 steps
        m = closed_loop_metrics(state, sample)
        assert m["steps"] == 13
        assert np.isfinite(m["log_ade"])
        # Dummy drives forward; some progress expected.
        assert m["progress"] > 1.0


def test_dummy_rollout_commits_multi_step(synthetic_h5, pack_cfg):
    ds = WOMDPackedDataset(synthetic_h5, cfg=pack_cfg)
    sample = ds.get_world_sample(0)
    policy = _tiny_policy(StraightLineDecoder())
    state = rollout_policy(
        policy, sample, neighbors="log", horizon=12, device="cpu", h_exec=4
    )
    assert len(state.executed_xy) == 13
    assert closed_loop_metrics(state, sample)["progress"] > 1.0


def test_gmm_short_rollout(synthetic_h5, pack_cfg):
    ds = WOMDPackedDataset(synthetic_h5, cfg=pack_cfg)
    sample = ds.get_world_sample(1)
    policy = _tiny_policy(
        GMMDecoder(d_model=32, nhead=4, d_ffn=64, dropout=0.0, num_modes=3, decoder_layers=1)
    )
    state = rollout_policy(policy, sample, neighbors="log", horizon=5, device="cpu")
    assert len(state.executed_xy) == 6


def test_occupancy_mode_prefers_clear_path():
    from types import SimpleNamespace

    from womd_il.sim.rollout import _occupancy_mode

    # Ego at origin heading +x. Neighbor sits 8 m ahead. Mode 0 drives into it;
    # mode 1 goes left; mode 1 has a slightly lower score.
    t = 10
    a, steps = 2, 20
    xyz = np.zeros((a, steps, 3), np.float32)
    vel = np.zeros((a, steps, 2), np.float32)
    valid = np.ones((a, steps), bool)
    xyz[1, t, 0] = 8.0
    sample = {"length": np.array([4.5, 4.5], np.float32), "width": np.array([1.8, 1.8], np.float32)}
    sim = SimpleNamespace(
        state=SimpleNamespace(t=t, xyz=xyz, vel=vel, valid=valid, sample=sample)
    )
    scores = torch.tensor([0.7, 0.3])
    xyh = torch.zeros(2, 20, 3)
    xyh[0, :, 0] = torch.linspace(0.5, 15.0, 20)
    xyh[1, :, 0] = torch.linspace(0.5, 15.0, 20)
    xyh[1, :, 1] = 6.0
    origin = np.array([0.0, 0.0, 0.0], np.float32)
    assert _occupancy_mode(scores, xyh, origin, sim, topk=2, look_ahead=20) == 1


def test_cost_mode_trades_imitation_for_clearance():
    from types import SimpleNamespace

    from womd_il.sim.rollout import _cost_mode

    t = 10
    a, steps = 2, 20
    xyz = np.zeros((a, steps, 3), np.float32)
    vel = np.zeros((a, steps, 2), np.float32)
    vel[0, t] = (8.0, 0.0)
    valid = np.ones((a, steps), bool)
    xyz[1, t, 0] = 8.0
    sample = {
        "length": np.array([4.5, 4.5], np.float32),
        "width": np.array([1.8, 1.8], np.float32),
        "map_type": np.zeros((1,), np.int32),
        "map_xyz": np.zeros((1, 2, 3), np.float32),
        "map_valid": np.zeros((1, 2), bool),
    }

    class _St(SimpleNamespace):
        def current_speed(self):
            return float(np.linalg.norm(self.vel[0, self.t]))

    sim = SimpleNamespace(state=_St(t=t, xyz=xyz, vel=vel, valid=valid, sample=sample))
    scores = torch.tensor([0.8, 0.2])
    xyh = torch.zeros(2, 20, 3)
    xyh[0, :, 0] = torch.linspace(0.8, 16.0, 20)
    xyh[1, :, 0] = torch.linspace(0.8, 16.0, 20)
    xyh[1, :, 1] = 6.0
    origin = np.array([0.0, 0.0, 0.0], np.float32)
    assert (
        _cost_mode(
            scores,
            xyh,
            origin,
            sim,
            topk=2,
            look_ahead=20,
            w_im=1.0,
            w_ov=8.0,
            w_speed=0.0,
            w_off=0.0,
        )
        == 1
    )


def test_lqr_tracks_speed_profile_without_clip():
    dt = 0.1
    v = 8.0
    t = np.arange(1, 41) * dt
    xy = np.stack([v * t, np.zeros_like(t)], axis=1)
    yaw = np.zeros(40)
    acts = traj_to_actions(xy, yaw, 0.0, 0.0, 0.0, v, n=40, tracker="lqr")
    acc = np.array([a for a, _ in acts])
    kap = np.array([k for _, k in acts])
    assert float(np.max(np.abs(acc))) < 1.0
    assert float(np.max(np.abs(kap))) < 0.05
    x = y = yaw_s = 0.0
    speed = v
    for a, k in acts:
        x, y, yaw_s, speed = bicycle_step(x, y, yaw_s, speed, a, k, dt=dt)
    assert abs(speed - v) < 0.5
    assert abs(x - v * 4.0) < 2.0


def test_inverse_clips_when_plan_is_faster_lqr_does_not_catch_up():
    dt = 0.1
    v_plan = 8.0
    t = np.arange(1, 41) * dt
    xy = np.stack([v_plan * t, np.zeros_like(t)], axis=1)
    yaw = np.zeros(40)
    inv = traj_to_actions(xy, yaw, 0.0, 0.0, 0.0, 5.0, n=40, tracker="inverse")
    lqr = traj_to_actions(xy, yaw, 0.0, 0.0, 0.0, 5.0, n=40, tracker="lqr")
    assert abs(inv[0][0]) >= 5.9
    assert 0.0 < lqr[0][0] < 3.0
