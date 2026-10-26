"""Open-loop minADE / minFDE / miss-rate and buffered trajectory IoU."""

from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F
from torch import Tensor

# Rasterized buffered-polyline IoU (Medium: sausage polygons around each path).
_IOU_RES = 0.25
_IOU_MAX_PIXELS = 160_000
_IOU_KEY = re.compile(r"^IoU([0-9.]+)@(\d+)$")
DEFAULT_HORIZONS = tuple(range(10, 81, 10))
# Official WOMD miss boxes at 3s / 5s / 8s. Other 10 Hz steps interpolate
# from the origin (0,0) through these anchors.
_MISS_ANCHORS = ((0, 0.0, 0.0), (30, 1.0, 2.0), (50, 1.8, 3.6), (80, 3.0, 6.0))


def nms_trajectories(
    xy: Tensor,
    scores: Tensor,
    num_keep: int = 6,
    dist_thresh: float = 2.5,
) -> tuple[Tensor, Tensor]:
    """Greedy cover in ADE space. xy [B,K,T,2], scores [B,K]."""
    b, k, t, _ = xy.shape
    device = xy.device
    keep_xy = []
    keep_sc = []
    for bi in range(b):
        order = torch.argsort(scores[bi], descending=True)
        chosen: list[int] = []
        for idx in order.tolist():
            if len(chosen) >= num_keep:
                break
            cand = xy[bi, idx]
            if chosen:
                refs = xy[bi, chosen]
                ade = (refs - cand).pow(2).sum(-1).sqrt().mean(-1)
                if torch.any(ade < dist_thresh):
                    continue
            chosen.append(idx)
        if not chosen:
            chosen = [int(order[0].item())]
        while len(chosen) < num_keep:
            chosen.append(chosen[-1])
        keep_xy.append(xy[bi, chosen])
        keep_sc.append(scores[bi, chosen])
    return torch.stack(keep_xy, 0), torch.stack(keep_sc, 0)


def _iou_key(buffer: float, horizon: int) -> str:
    return f"IoU{buffer:g}@{horizon}"


def miss_box_for_horizon(h: int) -> tuple[float, float]:
    """Lateral / longitudinal miss thresholds (m) at a 10 Hz step index."""
    if h <= 0:
        return 0.0, 0.0
    if h >= _MISS_ANCHORS[-1][0]:
        return _MISS_ANCHORS[-1][1], _MISS_ANCHORS[-1][2]
    for (h0, lat0, lon0), (h1, lat1, lon1) in zip(_MISS_ANCHORS, _MISS_ANCHORS[1:]):
        if h0 <= h <= h1:
            t = (h - h0) / (h1 - h0)
            return lat0 + t * (lat1 - lat0), lon0 + t * (lon1 - lon0)
    return _MISS_ANCHORS[-1][1], _MISS_ANCHORS[-1][2]


def _horizon_miss_boxes(
    horizons: tuple[int, ...],
    miss_lat: tuple[float, ...] | None,
    miss_lon: tuple[float, ...] | None,
) -> list[tuple[int, float, float]]:
    if (
        miss_lat is not None
        and miss_lon is not None
        and len(miss_lat) == len(horizons)
        and len(miss_lon) == len(horizons)
    ):
        return list(zip(horizons, miss_lat, miss_lon))
    return [(h, *miss_box_for_horizon(h)) for h in horizons]


def constant_velocity_xy(last_xy_vel: Tensor, steps: int, dt: float = 0.1) -> Tensor:
    """Coast with the last observed velocity. last_xy_vel [B,2] -> xy [B,T,2]."""
    vel = last_xy_vel.reshape(-1, 2).to(dtype=torch.float32)
    t = torch.arange(1, steps + 1, device=vel.device, dtype=vel.dtype) * dt
    return vel.unsqueeze(1) * t.view(1, -1, 1)


def _disk_kernel(buffer: float, res: float, device, dtype) -> Tensor:
    r_px = max(1, int(math.ceil(buffer / res)))
    ys = torch.arange(-r_px, r_px + 1, device=device, dtype=dtype)
    xs = torch.arange(-r_px, r_px + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    disk = (xx * res).pow(2) + (yy * res).pow(2) <= buffer * buffer
    return disk.to(dtype).view(1, 1, *disk.shape)


def _dense_points(xy: Tensor, valid: Tensor, step: float) -> tuple[Tensor, Tensor]:
    n, t, _ = xy.shape
    valid_b = valid > 0.5
    if t == 1:
        return xy, valid_b
    seg = xy[:, 1:] - xy[:, :-1]
    # Cover ~8 m / 0.1 s (highway) without a GPU sync on max segment length.
    n_per = max(2, min(48, int(math.ceil(8.0 / max(step, 1e-3)))))
    alpha = torch.linspace(0.0, 1.0, n_per, device=xy.device, dtype=xy.dtype)
    pts = xy[:, :-1, None, :] + seg[:, :, None, :] * alpha[None, None, :, None]
    pts = pts.reshape(n, (t - 1) * n_per, 2)
    seg_valid = valid_b[:, :-1] & valid_b[:, 1:]
    dense_valid = seg_valid[:, :, None].expand(n, t - 1, n_per).reshape(n, (t - 1) * n_per)
    return pts, dense_valid


def _make_grid(
    pred_xy: Tensor, gt_xy: Tensor, pad: float, res: float
) -> tuple[float, float, float, int, int]:
    pts = torch.cat([pred_xy.reshape(-1, 2), gt_xy.reshape(-1, 2)], dim=0)
    mins = pts.amin(0)
    maxs = pts.amax(0)
    x0 = float(mins[0].item()) - pad
    y0 = float(mins[1].item()) - pad
    span_x = max(float((maxs[0] - mins[0]).item()) + 2 * pad, res)
    span_y = max(float((maxs[1] - mins[1]).item()) + 2 * pad, res)
    area = (span_x / res) * (span_y / res)
    if area > _IOU_MAX_PIXELS:
        res = res * math.sqrt(area / _IOU_MAX_PIXELS)
    width = max(1, int(math.ceil(span_x / res)) + 1)
    height = max(1, int(math.ceil(span_y / res)) + 1)
    return x0, y0, res, height, width


def _stamp_polylines(
    xy: Tensor, valid: Tensor, x0: float, y0: float, res: float, height: int, width: int
) -> Tensor:
    """1-pixel occupancy [N,H,W] of densified polylines on a shared grid."""
    pts, pts_valid = _dense_points(xy, valid, step=0.5 * res)
    px = ((pts[..., 0] - x0) / res).round().long()
    py = ((pts[..., 1] - y0) / res).round().long()
    n, s = px.shape
    occ = torch.zeros(n, height, width, dtype=torch.bool, device=xy.device)
    n_idx = torch.arange(n, device=xy.device).unsqueeze(1).expand(n, s)
    inside = pts_valid & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if inside.any():
        occ[n_idx[inside], py[inside], px[inside]] = True
    return occ


def _dilate(occ: Tensor, kernel: Tensor, chunk: int = 16) -> Tensor:
    out = torch.empty_like(occ)
    pad = kernel.size(-1) // 2
    for i in range(0, occ.size(0), chunk):
        sl = slice(i, i + chunk)
        conv = F.conv2d(occ[sl].float().unsqueeze(1), kernel, padding=pad)
        out[sl] = conv.squeeze(1) > 0
    return out


def max_buffered_iou(
    pred_xy: Tensor,
    gt_xy: Tensor,
    gt_valid: Tensor,
    buffer: float,
    res: float = _IOU_RES,
) -> Tensor:
    """Max-over-K IoU of disk-buffered polylines. pred [B,K,T,2], gt [B,T,2].

    Each trajectory is a sausage polygon (polyline Minkowski-sum a disk of
    radius `buffer`). IoU is rasterized on an agent-frame grid. Higher is
    better; opposite-direction paths score near 0 even if ADE is similar.
    """
    return _max_buffered_iou_multi(pred_xy, gt_xy, gt_valid, (buffer,), res)[buffer]


def _max_buffered_iou_multi(
    pred_xy: Tensor,
    gt_xy: Tensor,
    gt_valid: Tensor,
    buffers: tuple[float, ...],
    res: float = _IOU_RES,
) -> dict[float, Tensor]:
    if not buffers:
        return {}
    b, k, t, _ = pred_xy.shape
    valid = gt_valid[:, :t]
    pred_flat = pred_xy.reshape(b * k, t, 2)
    valid_flat = valid[:, None].expand(b, k, t).reshape(b * k, t)
    pad = max(buffers) + 2.0 * res
    x0, y0, res, height, width = _make_grid(pred_xy, gt_xy[:, :t], pad, res)
    stamp = (x0, y0, res, height, width)
    gt_pts = _stamp_polylines(gt_xy[:, :t], valid, *stamp)
    pred_pts = _stamp_polylines(pred_flat, valid_flat, *stamp)
    out: dict[float, Tensor] = {}
    for buf in buffers:
        kernel = _disk_kernel(buf, res, pred_xy.device, torch.float32)
        gt_occ = _dilate(gt_pts, kernel)
        pred_occ = _dilate(pred_pts, kernel)
        gt_exp = gt_occ[:, None].expand(b, k, height, width).reshape(b * k, height, width)
        inter = (pred_occ & gt_exp).flatten(1).sum(1).float()
        union = (pred_occ | gt_exp).flatten(1).sum(1).float().clamp_min(1.0)
        out[buf] = (inter / union).reshape(b, k).max(dim=-1).values
    return out


def _horizon_metrics(
    pred_xy: Tensor,
    gt_xy: Tensor,
    gt_valid: Tensor,
    horizon: int,
    miss_lat: float,
    miss_lon: float,
) -> dict[str, Tensor]:
    h = min(horizon, pred_xy.size(-2), gt_xy.size(-2))
    p = pred_xy[:, :, :h]
    g = gt_xy[:, None, :h]
    v = gt_valid[:, None, :h].float()
    err = p - g
    dist = err.pow(2).sum(-1).sqrt()  # [B,K,H]
    ade = (dist * v).sum(-1) / v.sum(-1).clamp_min(1.0)
    fde = dist[:, :, -1]
    min_ade = ade.min(dim=-1).values
    min_fde = fde.min(dim=-1).values
    # Axis-aligned miss in the agent frame: lon=x, lat=y.
    lon = err[..., 0].abs()
    lat = err[..., 1].abs()
    match = (lon[:, :, -1] <= miss_lon) & (lat[:, :, -1] <= miss_lat)
    miss = (~match.any(dim=-1)).float()
    return {"minADE": min_ade, "minFDE": min_fde, "miss": miss}


def open_loop_metrics(
    pred_xy: Tensor,
    scores: Tensor,
    gt_xy: Tensor,
    gt_valid: Tensor,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    miss_lat: tuple[float, ...] | None = None,
    miss_lon: tuple[float, ...] | None = None,
    iou_buffers: tuple[float, ...] = (0.5, 1.0, 2.0),
    last_xy_vel: Tensor | None = None,
    dt: float = 0.1,
    num_keep: int = 6,
    nms_thresh: float = 2.5,
) -> dict[str, float]:
    """minADE/minFDE/miss plus max-over-K buffered trajectory IoU at each horizon.

    IoU keys look like ``IoU0.5@80``: disk-buffer the GT and predicted polylines
    by that radius (m) and take intersection/union of the two sausage polygons.

    If ``last_xy_vel`` is given, also score a constant-velocity dummy (copy last
    velocity, coast) as ``cvADE@h`` and ``beatCV@h`` (fraction of agents whose
    minADE is strictly better than that dummy).
    """
    xy, sc = nms_trajectories(pred_xy, scores, num_keep=num_keep, dist_thresh=nms_thresh)
    dummy = None
    if last_xy_vel is not None:
        dummy = constant_velocity_xy(last_xy_vel, pred_xy.size(-2), dt).unsqueeze(1)
    out: dict[str, float] = {}
    for h, mlat, mlon in _horizon_miss_boxes(horizons, miss_lat, miss_lon):
        m = _horizon_metrics(xy, gt_xy, gt_valid, h, mlat, mlon)
        for k, v in m.items():
            out[f"{k}@{h}"] = float(v.detach().mean().cpu())
        if dummy is not None:
            dm = _horizon_metrics(dummy, gt_xy, gt_valid, h, mlat, mlon)
            out[f"cvADE@{h}"] = float(dm["minADE"].detach().mean().cpu())
            out[f"beatCV@{h}"] = float((m["minADE"] < dm["minADE"]).float().mean().cpu())
        ious = _max_buffered_iou_multi(xy[:, :, :h], gt_xy[:, :h], gt_valid[:, :h], iou_buffers)
        for buf, iou in ious.items():
            out[_iou_key(buf, h)] = float(iou.detach().mean().cpu())
    last_h = horizons[-1] if horizons else 80
    out["minADE"] = out.get(f"minADE@{last_h}", 0.0)
    out["minFDE"] = out.get(f"minFDE@{last_h}", 0.0)
    # Best of all K modes, ignoring scores/NMS. If this stays low while
    # minADE@80 blows up, the GMM means are fine and ranking collapsed.
    oracle = _horizon_metrics(pred_xy, gt_xy, gt_valid, 80, 3.0, 6.0)
    out["oracleADE@80"] = float(oracle["minADE"].detach().mean().cpu())
    # Mode closed-loop actually executes (argmax of scores, no NMS).
    row = torch.arange(pred_xy.size(0), device=pred_xy.device)
    arg_xy = pred_xy[row, scores.argmax(dim=-1)].unsqueeze(1)
    arg = _horizon_metrics(arg_xy, gt_xy, gt_valid, 80, 3.0, 6.0)
    out["argmaxADE@80"] = float(arg["minADE"].detach().mean().cpu())
    out["argmaxFDE@80"] = float(arg["minFDE"].detach().mean().cpu())
    return out


def format_open_loop_table(
    metrics: dict[str, float],
    *,
    epoch: int | None = None,
    lr: float | None = None,
) -> str:
    """Pretty-print minADE / minFDE / miss / vs-dummy / buffered IoU by horizon."""
    horizons = sorted(int(k.split("@", 1)[1]) for k in metrics if k.startswith("minADE@"))
    buffers = sorted({float(m.group(1)) for k in metrics if (m := _IOU_KEY.match(k))})
    has_cv = any(k.startswith("cvADE@") for k in metrics)
    parts = ["val"]
    if epoch is not None:
        parts.append(f"epoch={epoch}")
    if lr is not None:
        parts.append(f"lr={lr:.2e}")
    cv_hdr = f"  {'cvADE':>8}  {'beatCV':>8}" if has_cv else ""
    iou_hdr = "".join(f"  {f'IoU@{b:g}':>8}" for b in buffers)
    header = f"{'horizon':>8}  {'minADE':>8}  {'minFDE':>8}  {'miss':>8}{cv_hdr}{iou_hdr}"
    lines = [" ".join(parts), header, "-" * len(header)]
    for h in horizons:
        row = (
            f"{f'@{h}':>8}  {metrics.get(f'minADE@{h}', float('nan')):8.3f}  "
            f"{metrics.get(f'minFDE@{h}', float('nan')):8.3f}  "
            f"{metrics.get(f'miss@{h}', float('nan')):8.3f}"
        )
        if has_cv:
            row += (
                f"  {metrics.get(f'cvADE@{h}', float('nan')):8.3f}  "
                f"{metrics.get(f'beatCV@{h}', float('nan')):8.3f}"
            )
        for b in buffers:
            row += f"  {metrics.get(_iou_key(b, h), float('nan')):8.3f}"
        lines.append(row)
    table_keys = {f"{name}@{h}" for h in horizons for name in ("minADE", "minFDE", "miss")}
    table_keys.update({"minADE", "minFDE"})
    if has_cv:
        table_keys.update(f"{name}@{h}" for h in horizons for name in ("cvADE", "beatCV"))
    table_keys.update(_iou_key(b, h) for h in horizons for b in buffers)
    extras = [f"{k}={v:.3f}" for k, v in metrics.items() if k not in table_keys]
    if extras:
        lines.append("  " + "  ".join(extras))
    return "\n".join(lines)
