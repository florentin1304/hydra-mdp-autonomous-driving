"""Agent-centric frame transforms (SDC at origin, heading along +x)."""

from __future__ import annotations

import numpy as np
import torch

Array = np.ndarray | torch.Tensor


def wrap_angle(angle: Array) -> Array:
    """Wrap radians to (-pi, pi]."""
    if isinstance(angle, torch.Tensor):
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    return np.arctan2(np.sin(angle), np.cos(angle))


def yaw_to_rotation(yaw: Array) -> tuple[Array, Array]:
    """Return (cos(yaw), sin(yaw))."""
    if isinstance(yaw, torch.Tensor):
        return torch.cos(yaw), torch.sin(yaw)
    return np.cos(yaw), np.sin(yaw)


def agent_centric_xy(xy: Array, origin_xy: Array, origin_yaw: Array) -> Array:
    """Rotate/translate world XY into an agent frame (heading = +x).

    World-to-agent:
        x' =  (x - ox) * cos(yaw) + (y - oy) * sin(yaw)
        y' = -(x - ox) * sin(yaw) + (y - oy) * cos(yaw)
    """
    c, s = yaw_to_rotation(origin_yaw)
    dx = xy[..., 0] - origin_xy[..., 0]
    dy = xy[..., 1] - origin_xy[..., 1]
    x = dx * c + dy * s
    y = -dx * s + dy * c
    if isinstance(xy, torch.Tensor):
        return torch.stack([x, y], dim=-1)
    return np.stack([x, y], axis=-1)


def inverse_agent_centric_xy(xy_local: Array, origin_xy: Array, origin_yaw: Array) -> Array:
    """Agent-frame XY back to world."""
    c, s = yaw_to_rotation(origin_yaw)
    x = xy_local[..., 0] * c - xy_local[..., 1] * s + origin_xy[..., 0]
    y = xy_local[..., 0] * s + xy_local[..., 1] * c + origin_xy[..., 1]
    if isinstance(xy_local, torch.Tensor):
        return torch.stack([x, y], dim=-1)
    return np.stack([x, y], axis=-1)


def agent_centric_yaw(yaw: Array, origin_yaw: Array) -> Array:
    return wrap_angle(yaw - origin_yaw)


def inverse_agent_centric_yaw(yaw_local: Array, origin_yaw: Array) -> Array:
    return wrap_angle(yaw_local + origin_yaw)


def headings_from_xy(xy: Array, fallback: Array | float = 0.0) -> Array:
    """Finite-difference headings from a trajectory that starts after t=0.

    `xy` is [..., T, 2] in the agent frame (displacement from current pose).
    """
    if isinstance(xy, torch.Tensor):
        prev = torch.zeros_like(xy)
        prev[..., 1:, :] = xy[..., :-1, :]
        delta = xy - prev
        heading = torch.atan2(delta[..., 1], delta[..., 0])
        zero = (delta.pow(2).sum(dim=-1) < 1e-8)
        if isinstance(fallback, torch.Tensor):
            heading = torch.where(zero, fallback.expand_as(heading), heading)
        else:
            heading = torch.where(zero, torch.full_like(heading, float(fallback)), heading)
        return wrap_angle(heading)
    prev = np.zeros_like(xy)
    prev[..., 1:, :] = xy[..., :-1, :]
    delta = xy - prev
    heading = np.arctan2(delta[..., 1], delta[..., 0])
    zero = np.sum(delta * delta, axis=-1) < 1e-8
    heading = np.where(zero, fallback, heading)
    return wrap_angle(heading)
