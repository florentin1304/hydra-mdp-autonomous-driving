from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from womd_il import constants as C
from womd_il.data.dataset import WOMDPackedDataset, packed_collate
from womd_il.data.features import apply_origin_perturb, build_sample_features
from womd_il.data.parse_scenario import _resample_polyline
from womd_il.data.synthetic import make_synthetic_pack
from womd_il.data.transforms import agent_centric_xy, agent_centric_yaw, wrap_angle


def test_resample_polyline_length():
    pts = np.stack([np.linspace(0, 10, 50), np.zeros(50), np.zeros(50)], axis=1)
    xyz, valid = _resample_polyline(pts, 20)
    assert xyz.shape == (20, 3)
    assert valid.all()
    np.testing.assert_allclose(xyz[0, 0], 0.0, atol=1e-4)
    np.testing.assert_allclose(xyz[-1, 0], 10.0, atol=1e-4)


def test_feature_shapes(pack_cfg):
    pack = make_synthetic_pack(n=1, cfg=pack_cfg, seed=0)
    sample = {k: v[0] for k, v in pack.items() if k != "scenario_id"}
    feat = build_sample_features(sample, target_idx=0, current_step=pack_cfg.current_step)
    assert feat["sdc_hist"].shape == (C.HISTORY_STEPS, C.HIST_IN_DIM)
    assert feat["neighbors"].shape[0] == C.NUM_AGENTS - 1
    assert feat["neighbors"].shape[-1] == C.HIST_IN_DIM
    assert feat["map_feat"].shape == (C.NUM_MAP_POLYLINES, C.MAP_IN_DIM)
    assert feat["tl_feat"].shape == (C.NUM_TRAFFIC_LIGHTS, C.HISTORY_STEPS, C.TL_IN_DIM)
    assert feat["tl_mask"].shape == (C.NUM_TRAFFIC_LIGHTS, C.HISTORY_STEPS)
    assert feat["tl_fut_state"].shape == (C.NUM_TRAFFIC_LIGHTS, C.FUTURE_STEPS)
    assert feat["tl_fut_valid"].shape == (C.NUM_TRAFFIC_LIGHTS, C.FUTURE_STEPS)
    assert feat["tl_fut_xy"].shape == (C.NUM_TRAFFIC_LIGHTS, C.FUTURE_STEPS, 2)
    assert feat["route_feat"].shape[-1] == C.ROUTE_IN_DIM
    assert feat["gt_xy"].shape == (C.FUTURE_STEPS, 2)
    assert feat["other_fut_xy"].shape == (C.NUM_AGENTS - 1, C.FUTURE_STEPS, 2)
    assert feat["other_fut_yaw"].shape == (C.NUM_AGENTS - 1, C.FUTURE_STEPS)
    assert feat["other_lw"].shape == (C.NUM_AGENTS - 1, 2)
    assert feat["ego_lw"].shape == (2,)
    assert feat["ego_state"].shape == (C.EGO_IN_DIM,)
    # SDC future should mostly go forward in the agent frame.
    assert feat["gt_xy"][10, 0] > 1.0


def test_map_on_route_flag(pack_cfg):
    pack = make_synthetic_pack(n=1, cfg=pack_cfg, seed=0)
    sample = {k: v[0] for k, v in pack.items() if k != "scenario_id"}
    feat = build_sample_features(sample, target_idx=0, current_step=pack_cfg.current_step)
    flags = feat["map_feat"][:, -1]
    types = sample["map_type"]
    # Synthetic route is y=0; lanes sit at y=(pi-16)*0.5. Corridor ≈6 m.
    lat = np.abs(np.arange(len(types)) - 16) * 0.5
    close_lane = (types == C.MAP_LANE_SURFACE) & (lat < 5.0)
    far_lane = (types == C.MAP_LANE_SURFACE) & (lat > 8.0)
    assert flags[close_lane].sum() > 0
    assert np.all(flags[close_lane] == 1.0)
    assert np.all(flags[far_lane] == 0.0)
    assert np.all(flags[types == C.MAP_ROAD_EDGE] == 0.0)


def test_snap_unique_returns_real_rows_not_means():
    from womd_il.data.windows import snap_unique

    rng = np.random.default_rng(0)
    xyh = np.zeros((20, 8, 3), np.float32)
    xyh[:, :, :2] = rng.normal(size=(20, 8, 2)).astype(np.float32)
    centers = np.stack(
        [
            xyh[:5, :, :2].mean(axis=0),
            xyh[5:10, :, :2].mean(axis=0),
            xyh[10:15, :, :2].mean(axis=0),
        ],
        axis=0,
    )
    pick = snap_unique(xyh, centers)
    assert pick.shape == (3,)
    assert len(set(pick.tolist())) == 3
    snapped = xyh[pick]
    for i, p in enumerate(pick):
        np.testing.assert_allclose(snapped[i], xyh[p])
        assert not np.allclose(snapped[i, :, :2], centers[i], atol=1e-5)


def test_collect_sdc_windows_is_strided(synthetic_h5):
    from womd_il.data.windows import collect_sdc_windows

    xyh, scene, t0 = collect_sdc_windows(synthetic_h5, horizon=40, stride=5)
    assert xyh.shape[1:] == (40, 3)
    assert xyh.shape[0] > 4
    assert set(np.diff(np.unique(t0)).tolist()).issubset({5})


def test_windowed_dataset_uses_t0(synthetic_h5, pack_cfg):
    windows = np.array([[0, 10], [0, 15]], np.int32)
    ds = WOMDPackedDataset(synthetic_h5, target_mode="sdc", cfg=pack_cfg, windows=windows)
    assert len(ds) == 2
    a, b = ds[0], ds[1]
    assert a["gt_xy"].shape == (80, 2)
    assert not np.allclose(a["gt_xy"][:10], b["gt_xy"][:10])


def test_dataset_roundtrip(synthetic_h5, pack_cfg):
    ds = WOMDPackedDataset(synthetic_h5, target_mode="sdc", cfg=pack_cfg)
    assert len(ds) == 4
    item = ds[0]
    assert item["gt_xy"].shape == (80, 2)
    loader = DataLoader(ds, batch_size=2, collate_fn=packed_collate)
    batch = next(iter(loader))
    assert batch["sdc_hist"].shape[0] == 2
    assert isinstance(batch["sdc_hist"], torch.Tensor)
    world = ds.get_world_sample(0)
    assert world["xyz"].shape[0] == pack_cfg.num_agents


def test_origin_perturb_is_ego_frame_then_world(pack_cfg):
    origin_xy = np.array([10.0, 4.0], np.float32)
    origin_yaw = 0.0
    new_xy, new_yaw = apply_origin_perturb(
        origin_xy, origin_yaw, {"lon": 1.0, "lat": 2.0, "dyaw": 0.3}
    )
    np.testing.assert_allclose(new_xy, [11.0, 6.0], atol=1e-5)
    np.testing.assert_allclose(new_yaw, 0.3, atol=1e-5)
    # yaw=pi/2: ego +x is world +y, ego +y is world -x
    new_xy, new_yaw = apply_origin_perturb(
        origin_xy, np.pi / 2, {"lon": 1.0, "lat": 0.0, "dyaw": 0.0}
    )
    np.testing.assert_allclose(new_xy, [10.0, 5.0], atol=1e-5)
    np.testing.assert_allclose(new_yaw, np.pi / 2, atol=1e-5)


def test_perturb_renorm_uses_log_future_not_corrected(pack_cfg):
    pack = make_synthetic_pack(n=1, cfg=pack_cfg, seed=0)
    sample = {k: v[0] for k, v in pack.items() if k != "scenario_id"}
    t0 = pack_cfg.current_step
    log_xy = sample["xyz"][0, t0, :2].astype(np.float32)
    log_yaw = float(sample["yaw"][0, t0])
    perturb = {"lon": 1.0, "lat": 0.25, "dyaw": 0.15}
    feat = build_sample_features(
        sample, target_idx=0, current_step=t0, perturb=perturb
    )
    origin_xy, origin_yaw = apply_origin_perturb(log_xy, log_yaw, perturb)
    np.testing.assert_allclose(feat["current_xyh_world"][:2], origin_xy, atol=1e-5)
    np.testing.assert_allclose(feat["current_xyh_world"][2], origin_yaw, atol=1e-5)
    assert not np.allclose(feat["current_xyh_world"][:2], log_xy, atol=1e-3)

    fut_sl = slice(t0 + 1, t0 + 1 + pack_cfg.future_steps)
    expected_xy = agent_centric_xy(sample["xyz"][0, fut_sl, :2], origin_xy, origin_yaw)
    expected_yaw = agent_centric_yaw(sample["yaw"][0, fut_sl], origin_yaw)
    np.testing.assert_allclose(feat["gt_xy"], expected_xy, atol=1e-4)
    np.testing.assert_allclose(feat["gt_yaw"], expected_yaw, atol=1e-4)
    # Neighbor futures live in the same re-normalized frame.
    expected_other = agent_centric_xy(sample["xyz"][1, fut_sl, :2], origin_xy, origin_yaw)
    np.testing.assert_allclose(feat["other_fut_xy"][0], expected_other, atol=1e-4)


def test_dataset_perturb_train_only(synthetic_h5, pack_cfg):
    perturb = {"prob": 1.0, "lon": 2.0, "lat": 0.0, "dyaw": 0.0}
    plain = WOMDPackedDataset(synthetic_h5, cfg=pack_cfg)
    off = WOMDPackedDataset(
        synthetic_h5, cfg=pack_cfg, augment=False, perturb=perturb
    )
    on = WOMDPackedDataset(
        synthetic_h5, cfg=pack_cfg, augment=True, perturb=perturb
    )
    np.testing.assert_allclose(plain[0]["gt_xy"], off[0]["gt_xy"])
    np.testing.assert_allclose(plain[0]["current_xyh_world"], off[0]["current_xyh_world"])
    assert not np.allclose(on[0]["gt_xy"], plain[0]["gt_xy"], atol=1e-3)
    assert not np.allclose(on[0]["current_xyh_world"], plain[0]["current_xyh_world"], atol=1e-3)


def test_random_origin_perturb_is_seeded():
    origin_xy = np.zeros(2, np.float32)
    origin_yaw = 0.0
    perturb = {"prob": 1.0, "xy_std": 0.5, "yaw_std": 0.2}
    a_xy, a_yaw = apply_origin_perturb(
        origin_xy, origin_yaw, perturb, np.random.default_rng(7)
    )
    b_xy, b_yaw = apply_origin_perturb(
        origin_xy, origin_yaw, perturb, np.random.default_rng(7)
    )
    c_xy, c_yaw = apply_origin_perturb(
        origin_xy, origin_yaw, perturb, np.random.default_rng(8)
    )
    np.testing.assert_allclose(a_xy, b_xy)
    np.testing.assert_allclose(a_yaw, b_yaw)
    assert not np.allclose(a_xy, c_xy) or not np.isclose(a_yaw, c_yaw)
    np.testing.assert_allclose(float(wrap_angle(a_yaw)), a_yaw, atol=1e-6)


def test_metric_cache_loads_and_skips_on_perturb(synthetic_h5, pack_cfg, tmp_path):
    from womd_il.data.metric_cache import create_metric_cache

    k = 8
    path = tmp_path / "metrics.h5"
    handle = create_metric_cache(path, 4, k, vocab_sha_hex="deadbeef", packed=synthetic_h5)
    handle["labels"][0] = 0
    handle["labels"][1:] = 1
    handle.attrs["rows_written"] = 4
    handle.close()
    ds = WOMDPackedDataset(
        synthetic_h5,
        cfg=pack_cfg,
        metric_cache=path,
        vocab_sha="deadbeef",
        vocab_size=k,
    )
    from womd_il.losses.hydra import METRIC_NAMES

    labels = ds[0]["metric_labels"]
    assert labels.shape == (k, len(METRIC_NAMES))
    assert labels.max() == 0
    ds.close()
    perturbed = WOMDPackedDataset(
        synthetic_h5,
        cfg=pack_cfg,
        augment=True,
        perturb={"prob": 0.5, "xy_std": 0.2, "yaw_std": 0.1},
        metric_cache=path,
        vocab_sha="deadbeef",
        vocab_size=k,
    )
    assert "metric_labels" not in perturbed[0]


def test_shard_bounds_cover_n():
    from womd_il.data.build_metric_cache import shard_bounds

    n, jobs = 10, 3
    spans = [shard_bounds(n, jobs, r) for r in range(jobs)]
    assert spans[0] == (0, 3)
    assert spans[-1][1] == n
    covered = []
    for a, b in spans:
        covered.extend(range(a, b))
    assert covered == list(range(n))


def test_parallel_metric_cache_matches_serial(synthetic_h5, tmp_path):
    from womd_il.data.build_metric_cache import cache_metrics
    import h5py

    vocab = np.zeros((8, 80, 3), np.float32)
    vocab[:, :, 0] = np.linspace(0.1, 8.0, 80)
    vp = tmp_path / "vocab.npy"
    np.save(vp, vocab)
    serial = cache_metrics(
        synthetic_h5, vp, tmp_path / "j1", jobs=1, batch_size=2, num_workers=0, device="cpu"
    )
    parallel = cache_metrics(
        synthetic_h5, vp, tmp_path / "j2", jobs=2, batch_size=2, num_workers=0, device="cpu"
    )
    with h5py.File(serial, "r") as fa, h5py.File(parallel, "r") as fb:
        np.testing.assert_array_equal(fa["labels"][:], fb["labels"][:])
