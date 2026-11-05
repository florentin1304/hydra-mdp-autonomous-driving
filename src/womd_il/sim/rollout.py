"""Closed-loop rollout of a policy on a packed scenario."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from womd_il.data.transforms import inverse_agent_centric_xy, wrap_angle
from womd_il.losses.hydra import METRIC_INDEX
from womd_il.sim.bicycle import traj_to_actions
from womd_il.sim.obs import features_to_batch
from womd_il.sim.replay import PackedSimulator, SimState, _offroad

# Hydra-MDP Eq. 11, same point as configs/decoder/hydra.yaml, EPO weight 0.
# Metric sigmoids point the trained way: 1 is compliant (no collision, on road).
_HYDRA_W_IM = 0.1
_HYDRA_W_NC = 1.0
_HYDRA_W_DAC = 1.0
_HYDRA_W_RULE = 1.0
_HYDRA_W_TTC = 5.0
_HYDRA_W_C = 2.0
_HYDRA_W_EP = 5.0


def hydra_mdp_scores(im_scores: torch.Tensor, metric_logits: torch.Tensor) -> torch.Tensor:
    """Score to maximize: imitation log plus NC, DAC, and the 5/2/5 rule log."""
    mprob = torch.sigmoid(metric_logits.float())
    s_nc = mprob[..., METRIC_INDEX["nc"]]
    s_dac = mprob[..., METRIC_INDEX["dac"]]
    s_ttc = mprob[..., METRIC_INDEX["ttc"]]
    s_c = mprob[..., METRIC_INDEX["hc"]]
    s_ep = mprob[..., METRIC_INDEX["progress"]]
    rule = _HYDRA_W_TTC * s_ttc + _HYDRA_W_C * s_c + _HYDRA_W_EP * s_ep
    return (
        _HYDRA_W_IM * torch.log(im_scores.float().clamp_min(1e-6))
        + _HYDRA_W_NC * torch.log(s_nc.clamp_min(1e-6))
        + _HYDRA_W_DAC * torch.log(s_dac.clamp_min(1e-6))
        + _HYDRA_W_RULE * torch.log(rule.clamp_min(1e-6))
    )


def _topk_indices(scores: torch.Tensor, topk: int) -> np.ndarray:
    k_all = int(scores.numel())
    if k_all <= 1:
        return np.array([0], dtype=np.int64)
    k = min(int(topk), k_all)
    return torch.topk(scores, k).indices.detach().cpu().numpy()


def _neighbor_cv(sim: PackedSimulator, look_ahead: int, dt: float):
    """Constant-velocity neighbors for `look_ahead` steps, or None if empty."""
    st = sim.state
    t = int(st.t)
    others_xy, others_vel, others_lw = [], [], []
    for i in range(1, st.xyz.shape[0]):
        if t >= st.valid.shape[1] or not st.valid[i, t]:
            continue
        others_xy.append(st.xyz[i, t, :2].copy())
        others_vel.append(st.vel[i, t].copy())
        others_lw.append((float(st.sample["length"][i]), float(st.sample["width"][i])))
    if not others_xy:
        return None
    others_xy = np.stack(others_xy, axis=0)
    others_vel = np.stack(others_vel, axis=0)
    others_lw = np.asarray(others_lw, dtype=np.float32)
    times = (np.arange(int(look_ahead), dtype=np.float32) * float(dt))[:, None]
    other_fut = others_xy[:, None, :] + others_vel[:, None, :] * times[None, :, :]
    r_ego = 0.5 * float(np.linalg.norm([st.sample["length"][0], st.sample["width"][0]]))
    r_oth = 0.5 * np.linalg.norm(others_lw, axis=-1)
    thresh = (r_ego + r_oth).astype(np.float32)
    return other_fut, thresh


def _overlap_hit(world_xy: np.ndarray, other_fut: np.ndarray, thresh: np.ndarray) -> float:
    t_use = min(world_xy.shape[0], other_fut.shape[1])
    dist = np.linalg.norm(world_xy[None, :t_use, :] - other_fut[:, :t_use, :], axis=-1)
    return float((dist < thresh[:, None]).any(axis=0).mean())


def _occupancy_mode(
    scores: torch.Tensor,
    xyh: torch.Tensor,
    origin: np.ndarray,
    sim: PackedSimulator,
    *,
    topk: int = 8,
    look_ahead: int = 20,
    dt: float = 0.1,
) -> int:
    """Among top-k scored modes, pick the one that overlaps current neighbors least."""
    idxs = _topk_indices(scores, topk)
    neigh = _neighbor_cv(sim, look_ahead, dt)
    if neigh is None:
        return int(torch.argmax(scores).item())
    other_fut, thresh = neigh
    t_use = min(int(look_ahead), int(xyh.shape[1]))
    xy_np = xyh[:, :t_use, :2].detach().cpu().numpy()
    hits = []
    for i in idxs:
        world = inverse_agent_centric_xy(xy_np[int(i)], origin[:2], float(origin[2]))
        hit = _overlap_hit(world, other_fut, thresh)
        hits.append((hit, -float(scores[int(i)]), int(i)))
    hits.sort()
    return int(hits[0][2])


def _plan_speed(xy_agent: np.ndarray, dt: float) -> float:
    """Implied speed from the first ~0.3 s of an agent-centric plan."""
    n = min(3, int(xy_agent.shape[0]))
    if n <= 0:
        return 0.0
    return float(np.linalg.norm(xy_agent[n - 1, :2])) / max(n * float(dt), 1e-6)


def _offroad_frac(sample: dict, world_xy: np.ndarray, stride: int = 4) -> float:
    """Fraction of consecutive plan segments that fail closed-loop offroad (DAC)."""
    if world_xy.shape[0] < 2:
        return 0.0
    step = max(int(stride), 1)
    pts = world_xy[::step]
    if pts.shape[0] < 2:
        pts = world_xy
    flags = [_offroad(sample, pts[i + 1], prev_xy=pts[i]) for i in range(pts.shape[0] - 1)]
    return float(np.mean(flags)) if flags else 0.0


def _cost_mode(
    scores: torch.Tensor,
    xyh: torch.Tensor,
    origin: np.ndarray,
    sim: PackedSimulator,
    *,
    topk: int = 256,
    look_ahead: int = 20,
    dt: float = 0.1,
    w_im: float = 1.0,
    w_ov: float = 16.0,
    w_speed: float = 0.2,
    w_off: float = 8.0,
) -> int:
    """Min cost among top-k: -log p_im + overlap + |Δv| + off-road.

    Hydra's last-stage 16 carry essentially all softmax mass, so topk=16 is
    'the proposals Hydra actually scored'. GMM collapse makes this a no-op.
    """
    idxs = _topk_indices(scores, topk)
    neigh = _neighbor_cv(sim, look_ahead, dt)
    t_use = min(int(look_ahead), int(xyh.shape[1]))
    xy_np = xyh[:, :t_use, :2].detach().cpu().numpy()
    v_now = float(sim.state.current_speed())
    sample = sim.state.sample
    best_i = int(idxs[0])
    best_cost = 1e18
    for i in idxs:
        i = int(i)
        p = float(scores[i].clamp(min=1e-8))
        world = inverse_agent_centric_xy(xy_np[i], origin[:2], float(origin[2]))
        ov = _overlap_hit(world, neigh[0], neigh[1]) if neigh is not None else 0.0
        dv = abs(_plan_speed(xy_np[i], dt) - v_now)
        off = _offroad_frac(sample, world)
        cost = (
            float(w_im) * (-np.log(p))
            + float(w_ov) * ov
            + float(w_speed) * dv
            + float(w_off) * off
        )
        if ov > 0.0:
            cost += 100.0
        if cost < best_cost:
            best_cost = cost
            best_i = i
    return best_i


def _topk_mode_overlay(
    bundle,
    scores: torch.Tensor,
    pick: int,
    origin: np.ndarray,
    *,
    n_modes: int = 128,
) -> dict:
    """World-frame top-N modes for video overlay (probs from im_scores if present)."""
    extras = bundle.extras or {}
    im_p = extras.get("im_scores")
    if im_p is not None:
        probs = im_p[0].float()
    else:
        # Scores may be Hydra-MDP-W costs with -1e9 fillers; mask those out.
        finite = scores > -1e8
        logits = scores.float().clone()
        logits = torch.where(finite, logits, torch.full_like(logits, -1e9))
        probs = torch.softmax(logits, dim=-1)
    k_all = int(probs.numel())
    n = min(max(int(n_modes), 1), k_all)
    top = torch.topk(probs, n)
    idxs = top.indices.detach().cpu().numpy().astype(np.int64)
    p = top.values.detach().cpu().numpy().astype(np.float32)
    local = bundle.xyh[0, idxs, :, :2].detach().cpu().numpy()
    world = np.stack(
        [inverse_agent_centric_xy(local[i], origin[:2], float(origin[2])) for i in range(n)],
        axis=0,
    ).astype(np.float32)
    pick_local = int(np.where(idxs == int(pick))[0][0]) if int(pick) in set(idxs.tolist()) else 0
    # If W-pick is outside top-N by im-prob, prepend it so the highlight exists.
    if int(pick) not in set(idxs.tolist()):
        pick_xy = bundle.xyh[0, int(pick), :, :2].detach().cpu().numpy()
        pick_w = inverse_agent_centric_xy(pick_xy, origin[:2], float(origin[2])).astype(np.float32)
        pick_p = float(probs[int(pick)].detach().cpu())
        world = np.concatenate([pick_w[None], world[: n - 1]], axis=0)
        p = np.concatenate([np.array([pick_p], np.float32), p[: n - 1]], axis=0)
        pick_local = 0
    return {"xy": world, "probs": p, "pick": pick_local}


@torch.no_grad()
def rollout_policy(
    policy,
    sample: dict,
    *,
    neighbors: str = "log",
    horizon: int = 80,
    device: torch.device | str = "cpu",
    h_exec: int = 1,
    mode_select: str = "argmax",
    cost: dict | None = None,
    n_viz_modes: int = 128,
    early_stop_on_fail: bool = False,
    on_replan: Callable[..., None] | None = None,
) -> SimState:
    device = torch.device(device)
    policy.eval()
    sim = PackedSimulator(sample, neighbors=neighbors)
    sim.reset()
    action_q: list[tuple[float, float]] = []
    steps = 0
    # Caller ``h_exec`` wins (0.1 s = 1). Decoder default is only a fallback.
    n_commit = max(int(h_exec), 1)
    max_t = sample["xyz"].shape[1] - 1
    while steps < horizon and sim.state.t < max_t:
        if not action_q:
            feat = sim.features()
            batch = features_to_batch(feat, device)
            kwargs = {}
            select = str(mode_select)
            if select == "pdms":
                kwargs["rank"] = "pdms"
            bundle = policy.infer(batch, **kwargs)
            scores = bundle.scores[0]
            pool = bundle.extras.get("pool_scores") if bundle.extras else None
            if pool is not None and select in ("cost", "occupancy"):
                scores = pool[0]
            # Blend / pick imitation + EPO softmax scores (mix checkpoints).
            extras = bundle.extras or {}
            if select in ("im", "imitation", "imi") and "im_scores" in extras:
                scores = extras["im_scores"][0]
            elif select in ("epo",) and "epo_scores" in extras:
                scores = extras["epo_scores"][0]
            elif select in ("avg", "mean", "arith") and "im_scores" in extras and "epo_scores" in extras:
                scores = 0.5 * extras["im_scores"][0] + 0.5 * extras["epo_scores"][0]
            elif select in ("geomean", "geometric") and "im_scores" in extras and "epo_scores" in extras:
                im_s = extras["im_scores"][0].clamp_min(1e-12)
                epo_s = extras["epo_scores"][0].clamp_min(1e-12)
                scores = torch.sqrt(im_s * epo_s)
            elif select in ("hydra", "hydra_mdp") and "im_scores" in extras and "metric_logits" in extras:
                scores = hydra_mdp_scores(extras["im_scores"][0], extras["metric_logits"][0])
            origin = batch["current_xyh_world"][0].detach().cpu().numpy()
            if select == "occupancy":
                k = _occupancy_mode(scores, bundle.xyh[0], origin, sim)
            elif select == "cost":
                allowed = ("topk", "look_ahead", "dt", "w_im", "w_ov", "w_speed", "w_off")
                kw = {kk: v for kk, v in dict(cost or {}).items() if kk in allowed and v is not None}
                k = _cost_mode(scores, bundle.xyh[0], origin, sim, **kw)
            else:
                k = int(torch.argmax(scores).item())
            if on_replan is not None:
                on_replan(sim=sim, bundle=bundle, batch=batch, pick=k, origin=origin)
            xyh = bundle.xyh[0, k]
            xy_agent = xyh[:, :2].detach().cpu().numpy()
            world_xy = inverse_agent_centric_xy(xy_agent, origin[:2], float(origin[2]))
            yaw_seq = None
            if xyh.size(-1) >= 3:
                yaw_seq = wrap_angle(
                    float(origin[2]) + xyh[:, 2].detach().cpu().numpy()
                )
            speed = sim.state.current_speed()
            action_q = traj_to_actions(
                world_xy,
                yaw_seq,
                float(origin[0]),
                float(origin[1]),
                float(origin[2]),
                speed,
                n=n_commit,
            )
            sim.state.predicted_xy.append(np.asarray(world_xy, dtype=np.float32))
            sim.state.predicted_t.append(int(sim.state.t))
            if int(n_viz_modes) > 0:
                overlay = _topk_mode_overlay(
                    bundle, scores, k, origin, n_modes=int(n_viz_modes)
                )
                overlay["t"] = int(sim.state.t)
                sim.state.predicted_modes.append(overlay)
        accel, kappa = action_q.pop(0)
        sim.step(accel, kappa)
        steps += 1
        if early_stop_on_fail and sim.state.collisions and sim.state.offroad:
            if sim.state.collisions[-1] or sim.state.offroad[-1]:
                break
    return sim.state
