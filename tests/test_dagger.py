"""DAgger pool / overlay / far-from-log collection helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from womd_il import constants as C
from womd_il.data.dagger import (
    DaggerOverlayDataset,
    append_pool_rows,
    patch_ego_track,
    pool_len,
)
from womd_il.data.dataset import WOMDPackedDataset, packed_collate
from womd_il.data.features import agent_centric_xy, build_sample_features
from womd_il.data.metric_cache import create_metric_cache
from womd_il.data.schema import PackConfig
from womd_il.sim.replay import PackedSimulator


def test_patch_ego_keeps_log_future(synthetic_h5, pack_cfg):
    ds = WOMDPackedDataset(synthetic_h5, cfg=pack_cfg)
    sample = ds.get_world_sample(0)
    t = C.CURRENT_STEP + 5
    ego_xy = sample["xyz"][0, :, :2].copy()
    ego_yaw = sample["yaw"][0].copy()
    ego_vel = sample["vel"][0].copy()
    # Shift ego at t away from log.
    ego_xy[C.CURRENT_STEP : t + 1] += np.array([2.0, 0.5], np.float32)
    patched = patch_ego_track(sample, ego_xy, ego_yaw, ego_vel, t)
    np.testing.assert_allclose(patched["xyz"][0, t, :2], ego_xy[t], atol=1e-5)
    # Future after t still log.
    np.testing.assert_allclose(
        patched["xyz"][0, t + 1 :, :2], sample["xyz"][0, t + 1 :, :2], atol=1e-5
    )
    feat = build_sample_features(patched, current_step=t)
    origin = feat["current_xyh_world"][:2]
    yaw = float(feat["current_xyh_world"][2])
    fut_sl = slice(t + 1, t + 1 + pack_cfg.future_steps)
    expected = agent_centric_xy(sample["xyz"][0, fut_sl, :2], origin, yaw)
    np.testing.assert_allclose(feat["gt_xy"][: expected.shape[0]], expected, atol=1e-4)
    ds.close()


def test_pool_append_and_overlay(synthetic_h5, pack_cfg, tmp_path):
    ds = WOMDPackedDataset(synthetic_h5, cfg=pack_cfg)
    sample = ds.get_world_sample(0)
    t = C.CURRENT_STEP + 3
    sim = PackedSimulator(sample, neighbors="log")
    sim.reset()
    # Nudge ego offline-ish without needing a policy.
    for _ in range(3):
        sim.step(1.0, 0.05)
    st = sim.state
    t = int(st.t)
    row = {
        "scenario_index": 0,
        "t": t,
        "iter": 0,
        "ego_xy": st.xyz[0, :, :2].astype(np.float32).copy(),
        "ego_yaw": st.yaw[0].astype(np.float32).copy(),
        "ego_vel": st.vel[0].astype(np.float32).copy(),
    }
    pool = tmp_path / "pool.h5"
    n = append_pool_rows(pool, [row], packed=Path(synthetic_h5), log_ade_thresh=1.5)
    assert n == 1
    assert pool_len(pool) == 1

    k, m = 8, 8
    cache = tmp_path / "metrics.h5"
    handle = create_metric_cache(cache, 1, k, vocab_sha_hex="deadbeef", packed=pool, n_metrics=m)
    handle["labels"][:] = 128
    handle.attrs["rows_written"] = 1
    handle.close()

    overlay = DaggerOverlayDataset(
        synthetic_h5,
        pool,
        cfg=pack_cfg,
        metric_cache=cache,
        vocab_sha="deadbeef",
        vocab_size=k,
    )
    assert len(overlay) == 1
    feat = overlay[0]
    assert "gt_xy" in feat and "metric_labels" in feat
    assert feat["metric_labels"].shape == (k, m)
    batch = packed_collate([feat, overlay[0]])
    assert batch["gt_xy"].shape[0] == 2
    overlay.close()
    ds.close()


def test_collect_requires_full_pdms_future(synthetic_h5, pack_cfg):
    """Saved rows must keep a full 4 s log future (flatten_l2 ignores gt_valid)."""
    import womd_il.data.dagger as dagger

    ds = WOMDPackedDataset(synthetic_h5, cfg=pack_cfg)
    sample = ds.get_world_sample(0)
    t_all = sample["xyz"].shape[1]

    class _Bundle:
        extras: dict = {}

        def __init__(self, k):
            self.scores = torch.zeros(1, k)
            self.xyh = torch.zeros(1, k, 40, 3)
            # Drift hard to the ego left so every step is far from the log.
            self.xyh[0, 0, :, 1] = torch.linspace(0.0, 12.0, 40)

    class _Policy:
        def eval(self):
            return self

        def infer(self, batch, **kw):
            return _Bundle(4)

    rows, stats = dagger.collect_scenario_rows(
        _Policy(),
        sample,
        scenario_index=0,
        iter_id=0,
        device=torch.device("cpu"),
        horizon=80,
        h_exec=1,
        mode_select="argmax",
        log_ade_thresh=0.0,
        max_per_scenario=1000,
    )
    assert rows, "expected at least one far-from-log row"
    t_last = t_all - 40 - 1
    assert max(int(r["t"]) for r in rows) <= t_last
    # The kept rows really do have 40 renormalized log future steps.
    for r in rows[-1:]:
        patched = dagger.patch_ego_track(
            sample, r["ego_xy"], r["ego_yaw"], r["ego_vel"], int(r["t"])
        )
        feat = build_sample_features(patched, current_step=int(r["t"]))
        assert float(feat["gt_valid"][:40].mean()) >= 0.5

    # Guard itself: a horizon that leaves no room must save nothing.
    rows_none, stats_none = dagger.collect_scenario_rows(
        _Policy(),
        sample,
        scenario_index=0,
        iter_id=0,
        device=torch.device("cpu"),
        horizon=80,
        h_exec=1,
        mode_select="argmax",
        log_ade_thresh=0.0,
        max_per_scenario=1000,
        pdms_horizon=t_all - 1,
    )
    assert rows_none == []
    assert stats_none["short_future"] == 1
    ds.close()


def test_pool_resume_skips_existing_iter(synthetic_h5, pack_cfg, tmp_path):
    from womd_il.data.dagger import append_pool_rows, collect_dagger_pool, pool_iter_extent

    ds = WOMDPackedDataset(synthetic_h5, cfg=pack_cfg)
    sample = ds.get_world_sample(0)
    t = C.CURRENT_STEP + 3
    row = {
        "scenario_index": 0,
        "t": t,
        "iter": 0,
        "ego_xy": sample["xyz"][0, :, :2].astype(np.float32).copy(),
        "ego_yaw": sample["yaw"][0].astype(np.float32).copy(),
        "ego_vel": sample["vel"][0].astype(np.float32).copy(),
    }
    pool = tmp_path / "pool.h5"
    append_pool_rows(pool, [row], packed=Path(synthetic_h5), log_ade_thresh=1.5)
    n, last = pool_iter_extent(pool, 0)
    assert n == 1 and last == 0

    class _P:
        def eval(self):
            return self

        def infer(self, batch, **kw):
            raise AssertionError("collect must not re-roll already-finished iter")

    stats = collect_dagger_pool(
        _P(),
        synthetic_h5,
        pool,
        iter_id=0,
        device=torch.device("cpu"),
        max_scenarios=1,
    )
    assert stats["already_complete"] == 1
    assert stats["new_rows"] == 0
    ds.close()


def test_save_stride_thins_consecutive_steps(synthetic_h5, pack_cfg):
    import womd_il.data.dagger as dagger

    ds = WOMDPackedDataset(synthetic_h5, cfg=pack_cfg)
    sample = ds.get_world_sample(0)

    class _Bundle:
        extras: dict = {}

        def __init__(self, k):
            self.scores = torch.zeros(1, k)
            self.xyh = torch.zeros(1, k, 40, 3)
            self.xyh[0, 0, :, 1] = torch.linspace(0.0, 12.0, 40)

    class _Policy:
        def eval(self):
            return self

        def infer(self, batch, **kw):
            return _Bundle(4)

    dense, _ = dagger.collect_scenario_rows(
        _Policy(),
        sample,
        scenario_index=0,
        iter_id=0,
        device=torch.device("cpu"),
        horizon=80,
        h_exec=1,
        mode_select="argmax",
        log_ade_thresh=0.0,
        max_per_scenario=1000,
        save_stride=1,
    )
    thin, _ = dagger.collect_scenario_rows(
        _Policy(),
        sample,
        scenario_index=0,
        iter_id=0,
        device=torch.device("cpu"),
        horizon=80,
        h_exec=1,
        mode_select="argmax",
        log_ade_thresh=0.0,
        max_per_scenario=1000,
        save_stride=5,
    )
    assert dense, "need some far-from-log steps"
    if len(dense) >= 5:
        assert len(thin) < len(dense)
        ts = [int(r["t"]) for r in thin]
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        assert all(g >= 5 for g in gaps)
    ds.close()


def test_batched_collect_matches_serial(synthetic_h5, pack_cfg):
    import womd_il.data.dagger as dagger

    ds = WOMDPackedDataset(synthetic_h5, cfg=pack_cfg)

    class _Bundle:
        def __init__(self, b, k):
            self.extras: dict = {}
            self.scores = torch.zeros(b, k)
            self.xyh = torch.zeros(b, k, 40, 3)
            self.xyh[:, 0, :, 1] = torch.linspace(0.0, 12.0, 40)

    class _Policy:
        def eval(self):
            return self

        def infer(self, batch, **kw):
            xy = batch["gt_xy"]
            b = int(xy.shape[0]) if xy.dim() == 3 else 1
            return _Bundle(b, 4)

    kw = dict(
        iter_id=0,
        device=torch.device("cpu"),
        horizon=30,
        h_exec=1,
        mode_select="argmax",
        log_ade_thresh=0.0,
        max_per_scenario=8,
        save_stride=5,
    )
    serial = []
    for i in range(len(ds)):
        rows, _ = dagger.collect_scenario_rows(
            _Policy(), ds.get_world_sample(i), scenario_index=i, **kw
        )
        serial.append([int(r["t"]) for r in rows])
    batched = []
    for i, rows, _st in dagger.iter_collected_scenarios(
        _Policy(),
        ds,
        start=0,
        stop=len(ds),
        neighbors="log",
        pdms_horizon=40,
        min_valid_frac=0.5,
        infer_batch=2,
        **kw,
    ):
        batched.append((i, [int(r["t"]) for r in rows]))
    assert [i for i, _ in batched] == list(range(len(ds)))
    assert [ts for _, ts in batched] == serial
    ds.close()


def test_metric_cache_rejects_partial_fill(tmp_path, synthetic_h5):
    from womd_il.data.metric_cache import create_metric_cache, validate_metric_cache

    path = tmp_path / "partial.h5"
    handle = create_metric_cache(path, 4, 8, vocab_sha_hex="deadbeef", packed=synthetic_h5)
    handle.close()
    with h5py.File(path, "r") as f:
        try:
            validate_metric_cache(f, vocab_sha_hex="deadbeef", k=8, n=4)
            raised = False
        except ValueError as exc:
            raised = True
            assert "filled rows" in str(exc)
    assert raised


def test_early_stop_rollout_flag():
    import inspect

    from womd_il.sim.rollout import rollout_policy

    assert "early_stop_on_fail" in inspect.signature(rollout_policy).parameters
