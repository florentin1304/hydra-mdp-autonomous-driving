"""Build encoder/decoder/policy from Hydra configs."""

from __future__ import annotations

from typing import Any

from womd_il.models.decoders.hydra import HydraDecoder
from womd_il.models.encoder import WayformerEncoder
from womd_il.models.policy import WayformerPolicy


def _cfg_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            return dict(OmegaConf.to_container(cfg, resolve=True) or {})
    except Exception:
        pass
    if hasattr(cfg, "items"):
        return {k: cfg[k] for k in cfg}
    return dict(cfg)


def build_encoder(model_cfg: Any) -> WayformerEncoder:
    c = _cfg_dict(model_cfg)
    c.pop("name", None)
    c.pop("decoder_layers", None)
    return WayformerEncoder(**c)


def build_decoder(decoder_cfg: Any, model_cfg: Any) -> HydraDecoder:
    d = _cfg_dict(decoder_cfg)
    m = _cfg_dict(model_cfg)
    name = str(d.pop("name", "hydra"))
    if name != "hydra":
        raise ValueError(f"Unknown decoder '{name}'. This package only supports 'hydra'.")
    kwargs = dict(d)
    for extra in ("metric_cache", "windows_path", "subsample_cap", "winner_cache"):
        kwargs.pop(extra, None)
    kwargs.setdefault("d_model", m.get("d_model", 256))
    kwargs.setdefault("nhead", m.get("nhead", 8))
    kwargs.setdefault("d_ffn", m.get("d_ffn", 1024))
    kwargs.setdefault("dropout", m.get("dropout", 0.1))
    return HydraDecoder(**kwargs)


def build_policy(model_cfg: Any, decoder_cfg: Any) -> WayformerPolicy:
    encoder = build_encoder(model_cfg)
    decoder = build_decoder(decoder_cfg, model_cfg)
    return WayformerPolicy(encoder, decoder)
