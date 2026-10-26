from __future__ import annotations

from womd_il.data.dataset import WOMDPackedDataset
from womd_il.models.decoders.dummy import StraightLineDecoder
from womd_il.models.encoder import WayformerEncoder
from womd_il.models.policy import WayformerPolicy
from womd_il.report import closed_loop_eval, print_closed_loop_report


def test_closed_loop_eval_headlines(synthetic_h5, pack_cfg):
    ds = WOMDPackedDataset(synthetic_h5, cfg=pack_cfg)
    policy = WayformerPolicy(
        WayformerEncoder(d_model=32, d_ffn=64, nhead=4, dropout=0.0, num_latents=8, encoder_layers=1),
        StraightLineDecoder(),
    )
    payload = closed_loop_eval(
        policy, ds, neighbors="log", horizon=8, device="cpu", max_scenarios=2
    )
    s = payload["summary"]
    assert "collision" in s and "offroad" in s and "progress" in s
    assert s["n"] == 2
    print_closed_loop_report(payload)
