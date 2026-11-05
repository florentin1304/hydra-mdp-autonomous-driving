"""Filter logic for the closed-loop episode inspector."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from womd_il.losses.hydra import METRIC_NAMES
from womd_il.viz.episode_inspect import (
    ClosedLoopEpisode,
    channel_scores,
    passing_indices,
)


def _episode(**overrides) -> ClosedLoopEpisode:
    k = 4
    m = len(METRIC_NAMES)
    gt = np.zeros((1, k, m), np.float32)
    gt[0, :, 0] = [0.0, 1.0, 1.0, 0.2]  # nc
    gt[0, :, 4] = [1.0, 1.0, 0.0, 1.0]  # ttc
    base = dict(
        scenario_index=540,
        scenario_id="abc",
        ckpt="ckpt",
        h5="h5",
        mode_select="geomean",
        t0=10,
        final_t=10,
        vocab_xy=np.zeros((k, 3, 2), np.float32),
        origins=np.zeros((1, 3), np.float32),
        sim_t=np.array([10], np.int32),
        pick=np.array([1], np.int32),
        pred_im=np.array([[0.2, 0.1, 0.05, 0.0]], np.float32),
        pred_epo=np.array([[0.01, 0.04, 0.02, 0.0]], np.float32),
        pred_metrics=gt.copy(),
        gt_metrics=gt,
        gt_ade=np.array([[0.0, 2.0 * np.log(2.0), 10.0, 1.0]], np.float32),
        gt_pdms=np.array([[0.9, 0.4, 0.6, 0.1]], np.float32),
        ego_xy=np.zeros((11, 2), np.float32),
        ego_yaw=np.zeros((11,), np.float32),
        collisions=np.zeros((0,), bool),
        offroad=np.zeros((0,), bool),
    )
    base.update(overrides)
    return ClosedLoopEpisode(**base)


def test_passing_indices_intersects_channels():
    scores = {
        "nc": np.array([0.0, 1.0, 1.0, 0.2], np.float32),
        "ttc": np.array([1.0, 1.0, 0.0, 1.0], np.float32),
    }
    idx, color, total = passing_indices(scores, ["nc", "ttc"], 0.5, max_draw=10)
    assert total == 1
    assert idx.tolist() == [1]
    assert color.tolist() == [1.0]


def test_passing_indices_strides_ties():
    scores = {"nc": np.ones(10, np.float32)}
    idx, _color, total = passing_indices(scores, ["nc"], 0.5, max_draw=4)
    assert total == 10
    assert len(idx) == 4
    assert idx.tolist() == [0, 3, 6, 9]


def test_passing_indices_keeps_highest_when_scores_differ():
    scores = {"progress": np.linspace(0.0, 1.0, 6).astype(np.float32)}
    idx, _color, total = passing_indices(scores, ["progress"], 0.5, max_draw=2)
    assert total == 3
    assert set(idx.tolist()) == {4, 5}


def test_channel_scores_gt_imitation_and_pred_peak():
    ep = _episode()
    gt = channel_scores(ep, 0, "gt", ade_scale=2.0)
    assert np.isclose(gt["imitation"][0], 1.0)
    assert np.isclose(gt["imitation"][1], 0.5, atol=1e-5)
    assert gt["imitation"][2] < 0.5
    assert np.isclose(gt["epo"][0], 0.9)
    assert np.isclose(gt["nc"][1], 1.0)

    pred = channel_scores(ep, 0, "pred", ade_scale=2.0)
    assert np.isclose(pred["imitation"][0], 1.0)
    assert np.isclose(pred["imitation"][1], 0.5)
    assert np.isclose(pred["epo"][1], 1.0)
    assert "nc" in pred


def test_channel_scores_pred_without_metric_head():
    ep = _episode(pred_metrics=None)
    pred = channel_scores(ep, 0, "pred", ade_scale=2.0)
    assert "imitation" in pred and "nc" not in pred
    idx, _c, total = passing_indices(pred, ["nc"], 0.5, max_draw=10)
    assert total == 0
    assert len(idx) == 0


def test_episode_cache_roundtrip(tmp_path):
    from womd_il.viz.episode_inspect import _load_episode, _save_episode

    ep = _episode()
    path = tmp_path / "ep.npz"
    meta = {"version": 1, "scenario_index": 540}
    _save_episode(ep, path, meta)
    loaded, saved = _load_episode(path)
    assert saved["scenario_index"] == 540
    assert loaded.scenario_id == "abc"
    assert loaded.pred_metrics is not None
    assert loaded.gt_metrics.shape == ep.gt_metrics.shape
    assert np.allclose(loaded.pred_im.astype(np.float32), ep.pred_im)

    ep2 = _episode(pred_metrics=None)
    _save_episode(ep2, path, meta)
    loaded2, _ = _load_episode(path)
    assert loaded2.pred_metrics is None


def test_vocab_path_walks_up_from_nested_cwd(tmp_path: Path, monkeypatch):
    from womd_il.models.decoders.hydra import HydraDecoder, resolve_vocab_path

    repo = tmp_path / "repo"
    nested = repo / "notebooks"
    nested.mkdir(parents=True)
    vocab_dir = repo / "data" / "artifacts"
    vocab_dir.mkdir(parents=True)
    np.save(vocab_dir / "hydra_vocab_4s.npy", np.zeros((4, 40, 3), np.float32))
    monkeypatch.chdir(nested)
    found = resolve_vocab_path("data/artifacts/hydra_vocab_4s.npy")
    assert found is not None
    arr = HydraDecoder._load_or_random("data/artifacts/hydra_vocab_4s.npy", 4, 80)
    assert arr.shape == (4, 40, 3)
