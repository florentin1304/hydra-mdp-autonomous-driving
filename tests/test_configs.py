from __future__ import annotations

from hydra import compose, initialize_config_dir

from womd_il.models.factory import build_policy
from womd_il.utils import config_dir


def test_train_configs_compose_and_build():
    with initialize_config_dir(config_dir=str(config_dir()), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "decoder=hydra",
                "decoder.vocab_path=null",
                "decoder.vocab_size=8",
                "decoder.decoder_layers=1",
                "model.d_model=32",
                "model.nhead=4",
                "model.d_ffn=64",
                "model.encoder_layers=1",
            ],
        )
        policy = build_policy(cfg.model, cfg.decoder)
        assert policy.decoder_name == "hydra"


def test_eval_configs_are_global():
    with initialize_config_dir(config_dir=str(config_dir()), version_base=None):
        cl = compose(config_name="eval/closed_loop")
        assert "data" in cl
        ol = compose(config_name="eval/open_loop")
        assert "ckpt" in ol
        assert "data" in ol
