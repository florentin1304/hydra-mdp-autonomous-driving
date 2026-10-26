from __future__ import annotations

import numpy as np
import torch

from womd_il.data.transforms import (
    agent_centric_xy,
    headings_from_xy,
    inverse_agent_centric_xy,
    wrap_angle,
)


def test_roundtrip_numpy():
    rng = np.random.default_rng(1)
    xy = rng.normal(size=(7, 2)).astype(np.float32)
    origin = np.array([3.0, -2.0], np.float32)
    yaw = 0.7
    local = agent_centric_xy(xy, origin, yaw)
    world = inverse_agent_centric_xy(local, origin, yaw)
    np.testing.assert_allclose(world, xy, atol=1e-5)


def test_roundtrip_torch():
    xy = torch.tensor([[10.0, 2.0], [11.0, 2.5]])
    origin = torch.tensor([10.0, 2.0])
    yaw = torch.tensor(np.pi / 2)
    local = agent_centric_xy(xy, origin, yaw)
    # After +90° rotation, the second point (1, 0.5) world-delta -> (0.5, -1)
    world = inverse_agent_centric_xy(local, origin, yaw)
    torch.testing.assert_close(world, xy, atol=1e-5, rtol=1e-5)


def test_heading_plus_x_is_zero():
    origin = np.zeros(2, np.float32)
    yaw = 0.3
    ahead = np.array([[np.cos(yaw), np.sin(yaw)]], np.float32) * 5
    local = agent_centric_xy(ahead, origin, yaw)
    np.testing.assert_allclose(local[0, 1], 0.0, atol=1e-5)
    assert local[0, 0] > 0


def test_wrap_angle():
    w = float(wrap_angle(np.array(3 * np.pi)))
    assert abs(abs(w) - np.pi) < 1e-5


def test_headings_from_xy():
    xy = torch.zeros(1, 4, 2)
    xy[0, :, 0] = torch.arange(1, 5).float()
    h = headings_from_xy(xy)
    assert torch.allclose(h, torch.zeros_like(h), atol=1e-5)
