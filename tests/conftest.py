from __future__ import annotations

from pathlib import Path

import pytest

from womd_il.data.schema import PackConfig
from womd_il.data.synthetic import write_synthetic_h5


@pytest.fixture
def pack_cfg() -> PackConfig:
    return PackConfig()


@pytest.fixture
def synthetic_h5(tmp_path: Path, pack_cfg: PackConfig) -> Path:
    return write_synthetic_h5(tmp_path / "synth.h5", n=4, seed=0, cfg=pack_cfg)
