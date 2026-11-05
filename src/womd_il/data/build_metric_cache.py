"""Build the offline Hydra metric-label HDF5 (CPU process shards or GPU)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from womd_il.data.dataset import WOMDPackedDataset, packed_collate
from womd_il.data.metric_cache import CACHE_VERSION, create_metric_cache, vocab_sha
from womd_il.data.schema import PackConfig
from womd_il.data.windows import load_window_table
from womd_il.losses.hydra import METRIC_NAMES, hydra_metric_labels


def shard_bounds(n: int, jobs: int, rank: int) -> tuple[int, int]:
    jobs = max(int(jobs), 1)
    start = rank * n // jobs
    end = (rank + 1) * n // jobs
    return start, end


def _label_loader(ds, *, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=packed_collate,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(num_workers > 0),
    )


def fill_labels(
    ds,
    vocab: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    n_local: int,
    desc: str = "labels",
    out_h5: h5py.Dataset | None = None,
    out_offset: int = 0,
    on_rows=None,
) -> np.ndarray | None:
    """Label ``n_local`` rows. Write into ``out_h5[out_offset:]`` when given, else return uint8 array.

    ``on_rows(total_written)`` is called after each chunk so a caller can
    checkpoint progress (see ``metric_cache`` ``rows_written``).
    """
    k = int(vocab.size(0))
    out: np.ndarray | None = None
    if out_h5 is None:
        out = np.zeros((n_local, k, len(METRIC_NAMES)), dtype=np.uint8)
    vocab_t = vocab.to(device, non_blocking=True)
    pin = device.type == "cuda"
    cursor = 0
    with torch.no_grad():
        for batch in tqdm(
            _label_loader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin),
            desc=desc,
        ):
            b = int(batch["gt_xy"].size(0))
            tensors = {
                key: (val.to(device, non_blocking=True) if torch.is_tensor(val) else val)
                for key, val in batch.items()
            }
            vocab_xy = vocab_t[None, :, :, :2].expand(b, -1, -1, -1)
            pred_yaw = vocab_t[None, :, :, 2].expand(b, -1, -1) if vocab_t.size(-1) > 2 else None
            y = hydra_metric_labels(vocab_xy, tensors, pred_yaw=pred_yaw, max_k=None)
            arr = np.clip(np.rint(y.detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
            if out_h5 is not None:
                out_h5[out_offset + cursor : out_offset + cursor + b] = arr
            else:
                assert out is not None
                out[cursor : cursor + b] = arr
            cursor += b
            if on_rows is not None:
                on_rows(out_offset + cursor)
    if cursor != n_local:
        raise RuntimeError(f"wrote {cursor} rows, expected {n_local}")
    return out


def _shard_worker(payload: dict) -> str:
    """Spawn-friendly worker: write one ``*.part{rank}.npy`` shard."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    rank = int(payload["rank"])
    jobs = int(payload["jobs"])
    device_str = str(payload.get("device", "cpu"))
    if device_str.startswith("cuda") and ":" not in device_str:
        # Multi-process single-host: bind rank → GPU.
        n_gpu = max(1, torch.cuda.device_count())
        device_str = f"cuda:{rank % n_gpu}"
    device = torch.device(device_str)
    vocab = np.load(payload["vocab"]).astype(np.float32)
    vocab_t = torch.from_numpy(vocab)
    windows = None
    windows_path = payload.get("windows")
    if windows_path:
        windows, _ = load_window_table(windows_path)
    ds = WOMDPackedDataset(
        payload["h5"],
        target_mode="sdc",
        max_scenarios=payload["max_scenarios"],
        cfg=PackConfig(),
        windows=windows,
    )
    n = len(ds)
    start, end = shard_bounds(n, jobs, rank)
    path = Path(payload["shard"])
    if end <= start:
        np.save(path, np.zeros((0, vocab.shape[0], len(METRIC_NAMES)), np.uint8))
        ds.close()
        return str(path)
    subset = Subset(ds, list(range(start, end)))
    labels = fill_labels(
        subset,
        vocab_t,
        device=device,
        batch_size=int(payload["batch_size"]),
        num_workers=0,
        n_local=end - start,
        desc=f"labels[{rank}@{device}]",
    )
    np.save(path, labels)
    ds.close()
    return str(path)


def write_h5_from_array(path: Path, labels: np.ndarray, *, vocab_sha_hex: str, packed: Path) -> None:
    n, k, m = labels.shape
    handle = create_metric_cache(
        path, n, k, vocab_sha_hex=vocab_sha_hex, packed=packed, n_metrics=m, metric_names=METRIC_NAMES
    )
    handle["labels"][:] = labels
    handle.attrs["rows_written"] = int(n)
    handle.flush()
    handle.close()


def _print_stats(labels: np.ndarray | h5py.Dataset, *, path: Path, sha: str, k: int, n: int) -> None:
    if isinstance(labels, h5py.Dataset):
        # Reservoir mean over rows to avoid loading full cache.
        step = max(1, n // 512)
        idx = np.arange(0, n, step)
        sample = np.asarray(labels[idx], dtype=np.float64)
        means = sample.mean(axis=(0, 1)) / 255.0
        pos = (sample.max(axis=1) > 0).mean(axis=0)
    else:
        means = labels.astype(np.float64).mean(axis=(0, 1)) / 255.0
        pos = (labels.max(axis=1) > 0).mean(axis=0)
    print(
        f"Wrote {path}  version={CACHE_VERSION}  metrics={','.join(METRIC_NAMES)}  "
        f"vocab_sha={sha}  K={k}  N={n}"
    )
    for name, mean in zip(METRIC_NAMES, means):
        print(f"  cache mean {name:10s}  {100.0 * mean:6.2f}%")
    print("  scenes with any vocab traj > 0:")
    for name, rate in zip(METRIC_NAMES, pos):
        print(f"    {name:10s}  {100.0 * rate:6.2f}%")


def cache_metrics(
    h5: Path,
    vocab_path: Path,
    out_dir: Path,
    *,
    batch_size: int = 4,
    num_workers: int = 4,
    jobs: int = 1,
    max_scenarios: int | None = None,
    device: str = "cpu",
    windows_path: Path | None = None,
) -> Path:
    h5 = Path(h5)
    vocab_path = Path(vocab_path)
    out_dir = Path(out_dir)
    vocab = np.load(vocab_path).astype(np.float32)
    if vocab.ndim != 3 or vocab.shape[-1] < 2:
        raise ValueError(f"vocab must be [K,T,2|3], got {vocab.shape}")
    k = int(vocab.shape[0])
    sha = vocab_sha(vocab_path)
    windows = None
    win_arg = None
    if windows_path is not None and Path(windows_path).exists():
        windows, _w = load_window_table(windows_path)
        win_arg = str(windows_path)
        print(f"Metric cache uses {len(windows)} windows from {windows_path}")
    probe = WOMDPackedDataset(
        h5, target_mode="sdc", max_scenarios=max_scenarios, cfg=PackConfig(), windows=windows
    )
    n = len(probe)
    probe.close()
    out_path = out_dir / f"{h5.stem}.h5"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Caching {n} rows × {k} trajectories → {out_path}  jobs={jobs} device={device}")

    if jobs <= 1:
        torch_device = torch.device(device)
        ds = WOMDPackedDataset(
            h5, target_mode="sdc", max_scenarios=max_scenarios, cfg=PackConfig(), windows=windows
        )
        handle = create_metric_cache(
            out_path, n, k, vocab_sha_hex=sha, packed=h5, n_metrics=len(METRIC_NAMES), metric_names=METRIC_NAMES
        )
        def _mark(written: int) -> None:
            handle.attrs["rows_written"] = int(written)

        try:
            fill_labels(
                ds,
                torch.from_numpy(vocab),
                device=torch_device,
                batch_size=batch_size,
                num_workers=num_workers,
                n_local=n,
                out_h5=handle["labels"],
                out_offset=0,
                on_rows=_mark,
            )
            handle.flush()
            _print_stats(handle["labels"], path=out_path, sha=sha, k=k, n=n)
        finally:
            handle.close()
            ds.close()
    else:
        shards = [out_dir / f".{h5.stem}.part{rank:03d}.npy" for rank in range(jobs)]
        payloads = [
            {
                "rank": rank,
                "jobs": jobs,
                "h5": str(h5),
                "vocab": str(vocab_path),
                "max_scenarios": max_scenarios,
                "batch_size": batch_size,
                "shard": str(shards[rank]),
                "windows": win_arg,
                "device": device,
            }
            for rank in range(jobs)
        ]
        ctx = torch.multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
            futs = [pool.submit(_shard_worker, payload) for payload in payloads]
            for fut in as_completed(futs):
                fut.result()
        parts = [np.load(p) for p in shards]
        labels = np.concatenate(parts, axis=0)
        write_h5_from_array(out_path, labels, vocab_sha_hex=sha, packed=h5)
        for p in shards:
            p.unlink(missing_ok=True)
        _print_stats(labels, path=out_path, sha=sha, k=k, n=n)
    return out_path
