from __future__ import annotations

import torch

from womd_il.metrics.open_loop import (
    DEFAULT_HORIZONS,
    constant_velocity_xy,
    max_buffered_iou,
    miss_box_for_horizon,
    open_loop_metrics,
)


def _line(x_end: float, y_end: float = 0.0, t: int = 80) -> torch.Tensor:
    x = torch.linspace(0.0, x_end, t)
    y = torch.linspace(0.0, y_end, t)
    return torch.stack([x, y], dim=-1)


def test_buffered_iou_identical_is_one():
    gt = _line(20.0)[None]
    pred = gt[:, None]
    valid = torch.ones(1, 80)
    iou = max_buffered_iou(pred, gt, valid, buffer=1.0)
    assert iou.item() > 0.95


def test_buffered_iou_opposite_is_low():
    gt = _line(20.0)[None]
    pred = _line(-20.0)[None, None]
    valid = torch.ones(1, 80)
    iou = max_buffered_iou(pred, gt, valid, buffer=1.0)
    assert iou.item() < 0.2


def test_buffered_iou_grows_with_buffer():
    gt = _line(20.0)[None]
    pred = (_line(20.0) + torch.tensor([0.0, 1.2]))[None, None]
    valid = torch.ones(1, 80)
    ious = [max_buffered_iou(pred, gt, valid, buffer=b).item() for b in (0.5, 1.0, 2.0)]
    assert ious[0] < ious[1] <= ious[2]
    assert ious[0] < 0.15
    assert ious[2] > 0.35


def test_buffered_iou_picks_best_mode():
    gt = _line(20.0)[None]
    pred = torch.stack([_line(-20.0), _line(20.0)], dim=0)[None]
    valid = torch.ones(1, 80)
    iou = max_buffered_iou(pred, gt, valid, buffer=1.0)
    assert iou.item() > 0.95


def test_open_loop_metrics_include_iou(synthetic_h5, pack_cfg):
    from torch.utils.data import DataLoader

    from womd_il.data.dataset import WOMDPackedDataset, packed_collate

    ds = WOMDPackedDataset(synthetic_h5, cfg=pack_cfg)
    batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=packed_collate)))
    pred = batch["gt_xy"][:, None] + 0.1
    scores = torch.zeros(pred.shape[0], 1)
    m = open_loop_metrics(pred, scores, batch["gt_xy"], batch["gt_valid"], num_keep=1)
    assert "IoU0.5@80" in m and "IoU1@80" in m and "IoU2@80" in m
    assert 0.0 <= m["IoU0.5@80"] <= m["IoU2@80"] <= 1.0
    assert "cvADE@80" not in m


def test_constant_velocity_coasts_with_last_vel():
    vel = torch.tensor([[10.0, 0.0], [0.0, 5.0]])
    xy = constant_velocity_xy(vel, steps=10, dt=0.1)
    assert xy.shape == (2, 10, 2)
    assert torch.allclose(xy[0, -1], torch.tensor([10.0, 0.0]))
    assert torch.allclose(xy[1, -1], torch.tensor([0.0, 5.0]))


def test_beat_cv_is_zero_when_pred_is_dummy():
    vel = torch.tensor([[8.0, 0.0]])
    dummy = constant_velocity_xy(vel, steps=80, dt=0.1)
    pred = dummy[:, None]
    scores = torch.ones(1, 1)
    valid = torch.ones(1, 80)
    # GT is a turn, so dummy ADE is high, but pred == dummy so it cannot beat CV.
    gt = _line(20.0, 15.0)[None]
    m = open_loop_metrics(pred, scores, gt, valid, num_keep=1, last_xy_vel=vel, iou_buffers=())
    assert m["beatCV@80"] == 0.0
    assert m["cvADE@80"] == m["minADE@80"]


def test_beat_cv_is_one_when_pred_is_gt():
    vel = torch.tensor([[8.0, 0.0]])
    gt = _line(8.0, 12.0)[None]
    pred = gt[:, None]
    scores = torch.ones(1, 1)
    valid = torch.ones(1, 80)
    m = open_loop_metrics(pred, scores, gt, valid, num_keep=1, last_xy_vel=vel, iou_buffers=())
    assert m["minADE@80"] < 1e-5
    assert m["argmaxADE@80"] < 1e-5
    assert m["cvADE@80"] > 1.0
    assert m["beatCV@80"] == 1.0


def test_miss_box_matches_womd_anchors():
    assert miss_box_for_horizon(30) == (1.0, 2.0)
    assert miss_box_for_horizon(50) == (1.8, 3.6)
    assert miss_box_for_horizon(80) == (3.0, 6.0)
    lat10, lon10 = miss_box_for_horizon(10)
    assert 0.0 < lat10 < 1.0 and 0.0 < lon10 < 2.0


def test_default_horizons_are_every_10_steps():
    assert DEFAULT_HORIZONS == (10, 20, 30, 40, 50, 60, 70, 80)
    gt = _line(20.0)[None]
    pred = gt[:, None]
    m = open_loop_metrics(
        pred, torch.ones(1, 1), gt, torch.ones(1, 80), num_keep=1, iou_buffers=()
    )
    for h in DEFAULT_HORIZONS:
        assert f"minADE@{h}" in m
        assert f"miss@{h}" in m
