"""HDF5 packed WOMD dataset with on-the-fly agent-centric features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from womd_il.data.features import build_sample_features
from womd_il.data.metric_cache import validate_metric_cache
from womd_il.data.schema import DATASETS, PackConfig


def _as_perturb_dict(perturb: Any) -> dict[str, float]:
    if perturb is None:
        return {}
    out: dict[str, float] = {}
    for k in ("prob", "xy_std", "yaw_std", "lon", "lat", "dyaw"):
        if k in perturb:
            out[k] = float(perturb[k])
    return out


class WOMDPackedDataset(Dataset):
    """Reads packed world-frame scenarios and emits encoder-ready tensors."""

    def __init__(
        self,
        path: str | Path,
        *,
        target_mode: str = "sdc",
        max_scenarios: int | None = None,
        cfg: PackConfig | None = None,
        augment: bool = False,
        perturb: dict | None = None,
        metric_cache: str | Path | None = None,
        vocab_sha: str | None = None,
        vocab_size: int | None = None,
        windows: np.ndarray | None = None,
    ):
        self.path = Path(path)
        self.target_mode = target_mode
        self.cfg = cfg or PackConfig()
        self.augment = bool(augment)
        self.perturb = _as_perturb_dict(perturb)
        self._file: h5py.File | None = None
        self._cache_file: h5py.File | None = None
        # Origin perturb moves neighbors vs the frozen vocab, so cached
        # labels would be wrong. Skip the cache whenever perturb can fire.
        can_perturb = self.augment and self.perturb.get("prob", 0.0) > 0.0
        self.metric_cache_path = None if can_perturb or metric_cache is None else Path(metric_cache)
        self._vocab_sha = vocab_sha
        self._vocab_size = vocab_size
        if not self.path.exists():
            raise FileNotFoundError(
                f"Packed dataset not found: {self.path}. "
                "Run scripts/preprocess.py or write_synthetic_h5()."
            )
        with h5py.File(self.path, "r") as f:
            n = int(f["xyz"].shape[0])
            self.n_scenarios = n if max_scenarios is None else min(n, int(max_scenarios))
            ttp = f["tracks_to_predict"][: self.n_scenarios]
        self._index: list[tuple[int, int]] = []
        if target_mode == "tracks_to_predict":
            for i in range(self.n_scenarios):
                hits = np.where(ttp[i])[0]
                if len(hits) == 0:
                    self._index.append((i, 0))
                else:
                    for j in hits:
                        self._index.append((i, int(j)))
        else:
            self._index = [(i, 0) for i in range(self.n_scenarios)]
        self._t0: list[int] | None = None
        if windows is not None:
            win = np.asarray(windows)
            if win.ndim != 2 or win.shape[1] < 2:
                raise ValueError(f"windows must be [N,2] (scene, t0), got {tuple(win.shape)}")
            self._index = [(int(s), 0) for s in win[:, 0]]
            self._t0 = [int(t) for t in win[:, 1]]

    def _h5(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def _metric_h5(self) -> h5py.File | None:
        if self.metric_cache_path is None:
            return None
        if self._cache_file is None:
            self._cache_file = h5py.File(self.metric_cache_path, "r")
            validate_metric_cache(
                self._cache_file,
                vocab_sha_hex=self._vocab_sha,
                k=self._vocab_size,
                n=len(self._index),
            )
        return self._cache_file

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        scen_i, target = self._index[idx]
        f = self._h5()
        sample = {k: f[k][scen_i] for k in DATASETS if k in f}
        sid = f["scenario_id"][scen_i]
        if isinstance(sid, bytes):
            sid = sid.decode("utf-8", errors="ignore")
        use_perturb = self.augment and self.perturb
        t0 = self._t0[idx] if self._t0 is not None else self.cfg.current_step
        feat = build_sample_features(
            sample,
            target_idx=target,
            current_step=t0,
            history_steps=self.cfg.history_steps,
            future_steps=self.cfg.future_steps,
            dt=self.cfg.dt,
            perturb=self.perturb if use_perturb else None,
            rng=np.random.default_rng() if use_perturb else None,
        )
        feat["scenario_id"] = str(sid)
        feat["scenario_index"] = np.int32(scen_i)
        cache = self._metric_h5()
        if cache is not None:
            row = idx if self._t0 is not None else scen_i
            raw = np.asarray(cache["labels"][row], dtype=np.float32)
            if int(cache.attrs.get("version", 0)) >= 3:
                raw = raw / 255.0
            feat["metric_labels"] = raw
        return feat

    def get_world_sample(self, scen_i: int) -> dict[str, np.ndarray]:
        f = self._h5()
        sample = {k: np.array(f[k][scen_i]) for k in DATASETS if k in f}
        sid = f["scenario_id"][scen_i]
        if isinstance(sid, bytes):
            sid = sid.decode("utf-8", errors="ignore")
        sample["scenario_id"] = str(sid)
        return sample

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._cache_file is not None:
            self._cache_file.close()
            self._cache_file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def packed_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [b.pop("scenario_id") for b in batch]
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = torch.from_numpy(np.stack([b[k] for b in batch], axis=0))
    out["scenario_id"] = ids
    return out
