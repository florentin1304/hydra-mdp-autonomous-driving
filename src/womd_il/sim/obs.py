"""Batch a single-scene feature dict for the PyTorch policy."""

from __future__ import annotations

import numpy as np
import torch


def features_to_batch(feat: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch = {}
    for k, v in feat.items():
        if k in ("scenario_id",):
            continue
        arr = np.asarray(v)
        t = torch.from_numpy(arr)
        if t.dtype == torch.float64:
            t = t.float()
        batch[k] = t.unsqueeze(0).to(device)
    return batch
