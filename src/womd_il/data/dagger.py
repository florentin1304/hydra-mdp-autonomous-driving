"""DAgger pool: far-from-log closed-loop states with log-future IM targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from womd_il.data.dataset import WOMDPackedDataset
from womd_il.data.features import build_sample_features
from womd_il.data.metric_cache import create_metric_cache, validate_metric_cache, vocab_sha
from womd_il.data.schema import DATASETS, PackConfig
from womd_il.data.transforms import inverse_agent_centric_xy, wrap_angle
from womd_il.losses.hydra import METRIC_NAMES, _PDM_T
from womd_il.sim.obs import features_to_batch
from womd_il.sim.replay import PackedSimulator
from womd_il.sim.rollout import apply_mode_select, traj_to_actions


POOL_VERSION = 1


def _ensure_pool(path: Path, *, packed: Path, t_steps: int, log_ade_thresh: float) -> h5py.File:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return h5py.File(path, "a")
    f = h5py.File(path, "w")
    f.attrs["version"] = POOL_VERSION
    f.attrs["packed"] = str(packed)
    f.attrs["log_ade_thresh"] = float(log_ade_thresh)
    f.attrs["t_steps"] = int(t_steps)
    for name, dtype, shape in (
        ("scenario_index", np.int32, (0,)),
        ("t", np.int32, (0,)),
        ("iter", np.int32, (0,)),
        ("far_from_log", np.uint8, (0,)),
        ("end_collision", np.uint8, (0,)),
        ("end_offroad", np.uint8, (0,)),
    ):
        f.create_dataset(name, shape=shape, maxshape=(None,), dtype=dtype, chunks=True)
    f.create_dataset(
        "ego_xy",
        shape=(0, t_steps, 2),
        maxshape=(None, t_steps, 2),
        dtype=np.float32,
        chunks=(1, t_steps, 2),
    )
    f.create_dataset(
        "ego_yaw",
        shape=(0, t_steps),
        maxshape=(None, t_steps),
        dtype=np.float32,
        chunks=(1, t_steps),
    )
    f.create_dataset(
        "ego_vel",
        shape=(0, t_steps, 2),
        maxshape=(None, t_steps, 2),
        dtype=np.float32,
        chunks=(1, t_steps, 2),
    )
    return f


def pool_len(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    with h5py.File(path, "r") as f:
        return int(f["scenario_index"].shape[0])


def pool_iter_extent(path: str | Path, iter_id: int) -> tuple[int, int]:
    """Return ``(n_rows, last_scenario)`` already stored for ``iter_id``.

    ``last_scenario`` is -1 when this iteration has no rows yet.
    """
    path = Path(path)
    if not path.exists():
        return 0, -1
    with h5py.File(path, "r") as f:
        it = np.asarray(f["iter"][:], dtype=np.int32)
        mask = it == int(iter_id)
        n = int(mask.sum())
        if n == 0:
            return 0, -1
        scenes = np.asarray(f["scenario_index"][:], dtype=np.int32)[mask]
        return n, int(scenes.max())


def append_pool_rows(
    path: Path, rows: list[dict[str, Any]], *, packed: Path, log_ade_thresh: float
) -> int:
    """Append collected rows; returns new total length."""
    if not rows:
        return pool_len(path)
    t_steps = int(rows[0]["ego_xy"].shape[0])
    f = _ensure_pool(path, packed=packed, t_steps=t_steps, log_ade_thresh=log_ade_thresh)
    try:
        n0 = int(f["scenario_index"].shape[0])
        n = n0 + len(rows)
        for key in (
            "scenario_index",
            "t",
            "iter",
            "far_from_log",
            "end_collision",
            "end_offroad",
            "ego_xy",
            "ego_yaw",
            "ego_vel",
        ):
            f[key].resize(n, axis=0)
        for i, row in enumerate(rows):
            j = n0 + i
            f["scenario_index"][j] = int(row["scenario_index"])
            f["t"][j] = int(row["t"])
            f["iter"][j] = int(row["iter"])
            f["far_from_log"][j] = 1
            f["end_collision"][j] = int(row.get("end_collision", 0))
            f["end_offroad"][j] = int(row.get("end_offroad", 0))
            f["ego_xy"][j] = np.asarray(row["ego_xy"], dtype=np.float32)
            f["ego_yaw"][j] = np.asarray(row["ego_yaw"], dtype=np.float32)
            f["ego_vel"][j] = np.asarray(row["ego_vel"], dtype=np.float32)
        f.flush()
        return n
    finally:
        f.close()


def patch_ego_track(
    sample: dict, ego_xy: np.ndarray, ego_yaw: np.ndarray, ego_vel: np.ndarray, t: int
) -> dict:
    """Copy sample and overwrite ego history through ``t``; keep log future after ``t``."""
    out = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
    t = int(t)
    xy = np.asarray(ego_xy, dtype=np.float32)
    yaw = np.asarray(ego_yaw, dtype=np.float32)
    vel = np.asarray(ego_vel, dtype=np.float32)
    out["xyz"][0, : t + 1, :2] = xy[: t + 1]
    out["yaw"][0, : t + 1] = yaw[: t + 1]
    out["vel"][0, : t + 1] = vel[: t + 1]
    out["valid"][0, : t + 1] = True
    return out


def select_mode_scores(scores: torch.Tensor, bundle, select: str) -> torch.Tensor:
    """Per-scene scores [B, K] used to pick the executed mode."""
    return apply_mode_select(scores, getattr(bundle, "extras", None), select)


def select_mode_index(scores: torch.Tensor, bundle, select: str) -> int:
    s = select_mode_scores(scores, bundle, select)
    if s.dim() == 1:
        return int(torch.argmax(s).item())
    return int(torch.argmax(s[0]).item())


def _collate_feats(feats: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    keys = feats[0].keys()
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        arr = np.stack([np.asarray(f[k]) for f in feats], axis=0)
        t = torch.from_numpy(arr)
        if t.dtype == torch.float64:
            t = t.float()
        out[k] = t.to(device, non_blocking=True)
    return out


@torch.no_grad()
def collect_scenario_rows(
    policy,
    sample: dict,
    *,
    scenario_index: int,
    iter_id: int,
    device: torch.device,
    neighbors: str = "log",
    horizon: int = 80,
    h_exec: int = 1,
    mode_select: str = "hydra_mdp_w",
    log_ade_thresh: float = 1.5,
    max_per_scenario: int = 16,
    pdms_horizon: int = _PDM_T,
    min_valid_frac: float = 0.5,
    save_stride: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Roll out with h_exec=1; save far-from-log valid states; stop on coll/offroad.

    A row is only savable while a full ``pdms_horizon`` log future still
    exists after ``t``: the vocab and the PDMS labels span 4 s, and
    ``flatten_l2`` imitation ignores ``gt_valid``, so a truncated (zero-padded)
    future would supervise "stop at the origin". Mirrors the bound and the
    validity gate of ``windows.collect_sdc_windows``.
    """
    policy.eval()
    sim = PackedSimulator(sample, neighbors=neighbors)
    sim.reset()
    action_q: list[tuple[float, float]] = []
    steps = 0
    n_commit = max(int(h_exec), 1)
    t_all = sample["xyz"].shape[1]
    max_t = t_all - 1
    fut_h = int(pdms_horizon)
    t_last_save = t_all - fut_h - 1
    log_xy = np.asarray(sample["xyz"][0, :, :2], dtype=np.float32)
    log_valid = np.asarray(sample["valid"][0], dtype=bool)
    rows: list[dict[str, Any]] = []
    stats = {"saved": 0, "end_collision": 0, "end_offroad": 0, "horizon": 0, "short_future": 0}
    n_qual = 0
    stride = max(int(save_stride), 1)
    cap = max(int(max_per_scenario), 0)

    while steps < horizon and sim.state.t < max_t:
        if not action_q:
            feat = sim.features()
            batch = features_to_batch(feat, device)
            kwargs = {"rank": "pdms"} if str(mode_select) == "pdms" else {}
            bundle = policy.infer(batch, **kwargs)
            scores = bundle.scores[0]
            k = select_mode_index(scores, bundle, str(mode_select))
            origin = batch["current_xyh_world"][0].detach().cpu().numpy()
            xyh = bundle.xyh[0, k]
            xy_agent = xyh[:, :2].detach().cpu().numpy()
            world_xy = inverse_agent_centric_xy(xy_agent, origin[:2], float(origin[2]))
            yaw_seq = None
            if xyh.size(-1) >= 3:
                yaw_seq = wrap_angle(float(origin[2]) + xyh[:, 2].detach().cpu().numpy())
            action_q = traj_to_actions(
                world_xy,
                yaw_seq,
                float(origin[0]),
                float(origin[1]),
                float(origin[2]),
                sim.state.current_speed(),
                n=n_commit,
            )
        accel, kappa = action_q.pop(0)
        sim.step(accel, kappa)
        steps += 1
        st = sim.state
        collided = bool(st.collisions[-1]) if st.collisions else False
        off = bool(st.offroad[-1]) if st.offroad else False
        if collided or off:
            stats["end_collision"] = int(collided)
            stats["end_offroad"] = int(off and not collided)
            break
        t = int(st.t)
        if t > t_last_save:
            # No full 4 s log future left; nothing further is savable.
            stats["short_future"] = 1
            break
        if log_valid[t]:
            fut_ok = float(log_valid[t + 1 : t + 1 + fut_h].mean())
            dist = float(np.linalg.norm(st.xyz[0, t, :2] - log_xy[t]))
            if dist > float(log_ade_thresh) and fut_ok >= float(min_valid_frac):
                if cap > 0 and n_qual % stride == 0 and len(rows) < cap:
                    rows.append(
                        {
                            "scenario_index": int(scenario_index),
                            "t": t,
                            "iter": int(iter_id),
                            "ego_xy": st.xyz[0, :, :2].astype(np.float32).copy(),
                            "ego_yaw": st.yaw[0].astype(np.float32).copy(),
                            "ego_vel": st.vel[0].astype(np.float32).copy(),
                            "end_collision": 0,
                            "end_offroad": 0,
                        }
                    )
                    stats["saved"] += 1
                n_qual += 1
    else:
        stats["horizon"] = 1
    if rows and (stats["end_collision"] or stats["end_offroad"]):
        rows[-1]["end_collision"] = stats["end_collision"]
        rows[-1]["end_offroad"] = stats["end_offroad"]
    return rows, stats


@torch.no_grad()
def _plan_actions(policy, sims, *, device: torch.device, mode_select: str, n_commit: int):
    """One batched forward for every simulator that needs a new plan."""
    feats = [sim.features() for sim in sims]
    batch = _collate_feats(feats, device)
    kwargs = {"rank": "pdms"} if str(mode_select) == "pdms" else {}
    # fp32: fp16 flips the argmax on 4096 scores often enough to change the rollout.
    bundle = policy.infer(batch, **kwargs)
    scores = select_mode_scores(bundle.scores, bundle, str(mode_select)).float()
    ks = torch.argmax(scores, dim=-1)
    origins = batch["current_xyh_world"].float().detach().cpu().numpy()
    xyh = bundle.xyh
    queues = []
    for j, sim in enumerate(sims):
        k = int(ks[j])
        origin = origins[j]
        traj = xyh[j, k].float().detach().cpu().numpy()
        world_xy = inverse_agent_centric_xy(traj[:, :2], origin[:2], float(origin[2]))
        yaw_seq = None
        if traj.shape[-1] >= 3:
            yaw_seq = wrap_angle(float(origin[2]) + traj[:, 2])
        queues.append(
            traj_to_actions(
                world_xy,
                yaw_seq,
                float(origin[0]),
                float(origin[1]),
                float(origin[2]),
                sim.state.current_speed(),
                n=n_commit,
            )
        )
    return queues


@torch.no_grad()
@torch.no_grad()
def iter_collected_scenarios(
    policy,
    ds: WOMDPackedDataset,
    *,
    start: int,
    stop: int,
    iter_id: int,
    device: torch.device,
    neighbors: str,
    horizon: int,
    h_exec: int,
    mode_select: str,
    log_ade_thresh: float,
    max_per_scenario: int,
    pdms_horizon: int,
    min_valid_frac: float,
    save_stride: int,
    infer_batch: int,
):
    """Roll scenes out together. Yields ``(index, rows, stats)`` in index order.

    In-order yield is what makes a crash resumable: a higher index is never
    handed to the caller before every lower index has finished.
    """
    policy.eval()
    n_commit = max(int(h_exec), 1)
    stride = max(int(save_stride), 1)
    cap = max(int(max_per_scenario), 0)
    fut_h = int(pdms_horizon)
    batch_n = max(int(infer_batch), 1)

    slots: list[dict[str, Any]] = []
    cursor = int(start)
    done: dict[int, tuple[list, dict]] = {}
    next_yield = int(start)

    def _open(index: int) -> dict[str, Any]:
        sample = ds.get_world_sample(index)
        sim = PackedSimulator(sample, neighbors=neighbors)
        sim.reset()
        t_all = sample["xyz"].shape[1]
        return {
            "index": index,
            "sim": sim,
            "action_q": [],
            "steps": 0,
            "rows": [],
            "n_qual": 0,
            "log_xy": np.asarray(sample["xyz"][0, :, :2], dtype=np.float32),
            "log_valid": np.asarray(sample["valid"][0], dtype=bool),
            "max_t": t_all - 1,
            "t_last_save": t_all - fut_h - 1,
            "stats": {
                "saved": 0,
                "end_collision": 0,
                "end_offroad": 0,
                "horizon": 0,
                "short_future": 0,
            },
        }

    def _fill() -> None:
        nonlocal cursor
        while len(slots) < batch_n and cursor < stop:
            slots.append(_open(cursor))
            cursor += 1

    def _finish(slot: dict[str, Any], why: str) -> None:
        if why == "collision":
            pass
        elif why == "short":
            slot["stats"]["short_future"] = 1
        else:
            slot["stats"]["horizon"] = 1
        if slot["rows"] and (
            slot["stats"]["end_collision"] or slot["stats"]["end_offroad"]
        ):
            slot["rows"][-1]["end_collision"] = slot["stats"]["end_collision"]
            slot["rows"][-1]["end_offroad"] = slot["stats"]["end_offroad"]
        done[slot["index"]] = (slot["rows"], slot["stats"])

    _fill()
    while slots:
        need = [s for s in slots if not s["action_q"]]
        if need:
            queues = _plan_actions(
                policy, [s["sim"] for s in need], device=device, mode_select=mode_select, n_commit=n_commit
            )
            for s, q in zip(need, queues):
                s["action_q"] = q
        alive: list[dict[str, Any]] = []
        for s in slots:
            accel, kappa = s["action_q"].pop(0)
            s["sim"].step(accel, kappa)
            s["steps"] += 1
            st = s["sim"].state
            collided = bool(st.collisions[-1]) if st.collisions else False
            off = bool(st.offroad[-1]) if st.offroad else False
            if collided or off:
                s["stats"]["end_collision"] = int(collided)
                s["stats"]["end_offroad"] = int(off and not collided)
                _finish(s, "collision")
                continue
            t = int(st.t)
            if t > s["t_last_save"]:
                _finish(s, "short")
                continue
            if s["log_valid"][t]:
                fut_ok = float(s["log_valid"][t + 1 : t + 1 + fut_h].mean())
                dist = float(np.linalg.norm(st.xyz[0, t, :2] - s["log_xy"][t]))
                if dist > float(log_ade_thresh) and fut_ok >= float(min_valid_frac):
                    if cap > 0 and s["n_qual"] % stride == 0 and len(s["rows"]) < cap:
                        s["rows"].append(
                            {
                                "scenario_index": int(s["index"]),
                                "t": t,
                                "iter": int(iter_id),
                                "ego_xy": st.xyz[0, :, :2].astype(np.float32).copy(),
                                "ego_yaw": st.yaw[0].astype(np.float32).copy(),
                                "ego_vel": st.vel[0].astype(np.float32).copy(),
                                "end_collision": 0,
                                "end_offroad": 0,
                            }
                        )
                        s["stats"]["saved"] += 1
                    s["n_qual"] += 1
            if s["steps"] >= horizon or st.t >= s["max_t"]:
                _finish(s, "horizon")
                continue
            alive.append(s)
        slots = alive
        _fill()
        while next_yield in done:
            rows, stats = done.pop(next_yield)
            yield next_yield, rows, stats
            next_yield += 1


@torch.no_grad()
def collect_dagger_pool(
    policy,
    packed: str | Path,
    pool_path: str | Path,
    *,
    iter_id: int,
    device: torch.device | str = "cuda",
    max_scenarios: int | None = None,
    neighbors: str = "log",
    horizon: int = 80,
    h_exec: int = 1,
    mode_select: str = "hydra_mdp_w",
    log_ade_thresh: float = 1.5,
    max_per_scenario: int = 16,
    pdms_horizon: int = _PDM_T,
    min_valid_frac: float = 0.5,
    save_stride: int = 1,
    flush_every: int = 100,
    infer_batch: int = 8,
) -> dict[str, Any]:
    device = torch.device(device)
    packed = Path(packed)
    pool_path = Path(pool_path)
    ds = WOMDPackedDataset(packed, target_mode="sdc", cfg=PackConfig())
    n = len(ds) if max_scenarios is None else min(len(ds), int(max_scenarios))
    already, last_scene = pool_iter_extent(pool_path, iter_id)
    start_i = last_scene + 1
    totals = {
        "saved": already,
        "end_collision": 0,
        "end_offroad": 0,
        "horizon": 0,
        "short_future": 0,
        "scenarios": n,
        "resumed_from": start_i,
        "skipped_existing": already,
    }
    pending: list[dict[str, Any]] = []
    new_rows = 0

    def _flush() -> None:
        nonlocal pending, new_rows
        if not pending:
            return
        append_pool_rows(
            pool_path, pending, packed=packed, log_ade_thresh=float(log_ade_thresh)
        )
        new_rows += len(pending)
        pending = []

    if start_i >= n:
        ds.close()
        totals["pool_size"] = pool_len(pool_path)
        totals["new_rows"] = 0
        totals["already_complete"] = 1
        return totals

    stream = iter_collected_scenarios(
        policy,
        ds,
        start=start_i,
        stop=n,
        iter_id=iter_id,
        device=device,
        neighbors=neighbors,
        horizon=horizon,
        h_exec=h_exec,
        mode_select=mode_select,
        log_ade_thresh=log_ade_thresh,
        max_per_scenario=max_per_scenario,
        pdms_horizon=pdms_horizon,
        min_valid_frac=min_valid_frac,
        save_stride=save_stride,
        infer_batch=infer_batch,
    )
    for i, rows, st in tqdm(stream, total=n - start_i, desc=f"dagger-collect[{iter_id}]"):
        pending.extend(rows)
        totals["saved"] += int(st["saved"])
        for k in ("end_collision", "end_offroad", "horizon", "short_future"):
            totals[k] += int(st[k])
        if len(pending) >= int(flush_every) or (i + 1) % int(flush_every) == 0:
            _flush()
    _flush()
    ds.close()
    totals["pool_size"] = pool_len(pool_path)
    totals["new_rows"] = new_rows
    totals["already_complete"] = 0
    return totals


class DaggerOverlayDataset(Dataset):
    """Rebuild features from packed world + ego CL history; log future as GT."""

    def __init__(
        self,
        packed: str | Path,
        pool: str | Path,
        *,
        cfg: PackConfig | None = None,
        metric_cache: str | Path | None = None,
        vocab_sha: str | None = None,
        vocab_size: int | None = None,
    ):
        self.packed = Path(packed)
        self.pool_path = Path(pool)
        self.cfg = cfg or PackConfig()
        self._file: h5py.File | None = None
        self._pool: h5py.File | None = None
        self._cache_file: h5py.File | None = None
        self.metric_cache_path = Path(metric_cache) if metric_cache else None
        self._vocab_sha = vocab_sha
        self._vocab_size = vocab_size
        if not self.pool_path.exists():
            raise FileNotFoundError(self.pool_path)
        with h5py.File(self.pool_path, "r") as f:
            self._n = int(f["scenario_index"].shape[0])

    def _h5(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.packed, "r")
        return self._file

    def _pool_h5(self) -> h5py.File:
        if self._pool is None:
            self._pool = h5py.File(self.pool_path, "r")
        return self._pool

    def _metric_h5(self) -> h5py.File | None:
        if self.metric_cache_path is None:
            return None
        if self._cache_file is None:
            self._cache_file = h5py.File(self.metric_cache_path, "r")
            validate_metric_cache(
                self._cache_file,
                vocab_sha_hex=self._vocab_sha,
                k=self._vocab_size,
                n=self._n,
            )
        return self._cache_file

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        p = self._pool_h5()
        scen_i = int(p["scenario_index"][idx])
        t = int(p["t"][idx])
        f = self._h5()
        sample = {k: f[k][scen_i] for k in DATASETS if k in f}
        sid = f["scenario_id"][scen_i]
        if isinstance(sid, bytes):
            sid = sid.decode("utf-8", errors="ignore")
        sample = patch_ego_track(
            sample,
            np.asarray(p["ego_xy"][idx]),
            np.asarray(p["ego_yaw"][idx]),
            np.asarray(p["ego_vel"][idx]),
            t,
        )
        feat = build_sample_features(
            sample,
            target_idx=0,
            current_step=t,
            history_steps=self.cfg.history_steps,
            future_steps=self.cfg.future_steps,
            dt=self.cfg.dt,
        )
        feat["scenario_id"] = str(sid)
        feat["scenario_index"] = np.int32(scen_i)
        cache = self._metric_h5()
        if cache is not None:
            raw = np.asarray(cache["labels"][idx], dtype=np.float32)
            if int(cache.attrs.get("version", 0)) >= 3:
                raw = raw / 255.0
            feat["metric_labels"] = raw
        return feat

    def close(self) -> None:
        for attr in ("_file", "_pool", "_cache_file"):
            handle = getattr(self, attr)
            if handle is not None:
                handle.close()
                setattr(self, attr, None)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def cache_dagger_metrics(
    packed: str | Path,
    pool: str | Path,
    vocab_path: str | Path,
    out_path: str | Path,
    *,
    batch_size: int = 4,
    num_workers: int = 0,
    device: str = "cuda",
) -> Path:
    """Write PDMS metric labels aligned 1:1 with pool rows.

    Append-only: an existing cache is grown in place and only rows
    ``[rows_written, N)`` are labeled. The previous labels are never read into
    RAM and the file is never unlinked, so an interrupted fill can resume and a
    multi-day label pass is not one crash away from being lost.
    """
    from torch.utils.data import Subset

    from womd_il.data.build_metric_cache import fill_labels

    packed = Path(packed)
    pool = Path(pool)
    vocab_path = Path(vocab_path)
    out_path = Path(out_path)
    vocab = np.load(vocab_path).astype(np.float32)
    k = int(vocab.shape[0])
    sha = vocab_sha(vocab_path)
    ds = DaggerOverlayDataset(packed, pool, cfg=PackConfig())
    n = len(ds)
    if n == 0:
        ds.close()
        raise ValueError(f"empty dagger pool: {pool}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle: h5py.File
    if out_path.exists():
        handle = h5py.File(out_path, "a")
        try:
            validate_metric_cache(handle, vocab_sha_hex=sha, k=k)
            written = int(handle.attrs.get("rows_written", handle["labels"].shape[0]))
            if written > n:
                raise ValueError(
                    f"metric cache has {written} rows but the pool only has {n}; "
                    "pool and cache are out of sync — delete both and recollect"
                )
            if int(handle["labels"].shape[0]) < n:
                handle["labels"].resize(n, axis=0)
                handle.attrs["n"] = n
        except Exception:
            handle.close()
            ds.close()
            raise
    else:
        handle = create_metric_cache(
            out_path,
            n,
            k,
            vocab_sha_hex=sha,
            packed=pool,
            n_metrics=len(METRIC_NAMES),
            metric_names=METRIC_NAMES,
        )
        written = 0

    offset = min(int(written), n)

    def _mark(total: int) -> None:
        handle.attrs["rows_written"] = int(total)

    try:
        if offset < n:
            subset = Subset(ds, list(range(offset, n)))
            fill_labels(
                subset,
                torch.from_numpy(vocab),
                device=torch.device(device),
                batch_size=batch_size,
                num_workers=num_workers,
                n_local=n - offset,
                out_h5=handle["labels"],
                out_offset=offset,
                desc=f"dagger-labels[{offset}:{n}]",
                on_rows=_mark,
            )
        _mark(n)
        handle.flush()
    finally:
        handle.close()
        ds.close()
    return out_path
