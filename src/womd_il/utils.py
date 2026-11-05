"""Small shared helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from womd_il import constants as C


def _config_parent(path: str) -> str:
    return str(Path(str(path)).parent)


if not OmegaConf.has_resolver("parent"):
    OmegaConf.register_new_resolver("parent", _config_parent)


def config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs"


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def unwrap_compiled(module: torch.nn.Module) -> torch.nn.Module:
    """Return the original nn.Module if ``torch.compile`` wrapped it."""
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod  # type: ignore[assignment]
    return module


def apply_torch_compile(policy: torch.nn.Module, cfg) -> torch.nn.Module:
    """Compile the train/infer hot paths (policy has no useful ``forward``).

    Accepts ``compile: true`` or a dict::

        compile:
          enabled: true
          mode: default          # default | reduce-overhead | max-autotune
          dynamic: false         # true recompiles on shape change; can break
                                 # fixed Linear in-dims under inductor
    """
    raw = cfg.get("compile", False) if cfg is not None else False
    if raw is None or raw is False:
        return policy
    if raw is True:
        mode, dynamic = "default", False
    else:
        if not bool(raw.get("enabled", True)):
            return policy
        mode = str(raw.get("mode", "default"))
        dyn = raw.get("dynamic", False)
        dynamic = None if dyn is None else bool(dyn)
    kwargs: dict = {"mode": mode, "fullgraph": False}
    if dynamic is not None:
        kwargs["dynamic"] = dynamic
    # Method compile covers encoder+decoder used by train/eval; compiling the
    # nn.Module alone would only optimize unused Module.forward.
    try:
        policy.train_step = torch.compile(policy.train_step, **kwargs)  # type: ignore[method-assign]
        policy.infer = torch.compile(policy.infer, **kwargs)  # type: ignore[method-assign]
    except Exception as exc:  # pragma: no cover - backend / version issues
        print(f"torch.compile setup failed ({exc}); continuing eager")
        return policy
    print(f"torch.compile train_step+infer mode={mode} dynamic={dynamic}")
    return policy


def save_checkpoint(path: str | Path, policy, cfg, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mod = unwrap_compiled(policy)
    payload = {
        "model": mod.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "decoder": str(getattr(policy, "decoder_name", getattr(mod, "decoder_name", ""))),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def pad_state_dict_to_model(state: dict, model: torch.nn.Module) -> dict:
    """Zero-pad a narrower map_proj input so old checkpoints keep working.

    The on-route map channel was added as dim 21. Extra input weights stay 0,
    so a 20-d checkpoint ignores the new bit until retrained.
    """
    out = dict(state)
    key = "encoder.map_proj.net.0.weight"
    dst = model.state_dict()
    if key not in out or key not in dst:
        return out
    src_w = out[key]
    dst_w = dst[key]
    if (
        src_w.ndim == 2
        and dst_w.ndim == 2
        and src_w.shape[0] == dst_w.shape[0]
        and src_w.shape[1] < dst_w.shape[1]
    ):
        w = torch.zeros_like(dst_w)
        w[:, : src_w.shape[1]] = src_w.to(dtype=dst_w.dtype, device=dst_w.device)
        out[key] = w
    return out


def _bump_map_in_dim(cfg) -> None:
    model = cfg.get("model") if cfg is not None else None
    if model is None:
        return
    mid = int(model.get("map_in_dim", C.MAP_IN_DIM) or C.MAP_IN_DIM)
    if mid < C.MAP_IN_DIM:
        model.map_in_dim = C.MAP_IN_DIM


def load_policy_from_ckpt(path: str | Path, device="cpu"):
    """Rebuild encoder/decoder from the config stored in a checkpoint."""
    from womd_il.models.factory import build_policy

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    cfg = OmegaConf.create(ckpt.get("cfg", {}))
    if cfg.get("model") is None or cfg.get("decoder") is None:
        raise ValueError(f"{path} has no saved model/decoder config")
    _bump_map_in_dim(cfg)
    # Vocab is a registered buffer: use the checkpoint tensor so the tokenizer
    # width matches the weights even if ``vocab_path`` is missing from cwd.
    vocab = ckpt.get("model", {}).get("decoder.vocab")
    policy = build_policy(cfg.model, cfg.decoder, vocab=vocab).to(device)
    policy.load_state_dict(pad_state_dict_to_model(ckpt["model"], policy), strict=False)
    policy.eval()
    return policy, cfg


def load_checkpoint(path: str | Path, policy, map_location="cpu") -> dict:
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    mod = unwrap_compiled(policy)
    mod.load_state_dict(pad_state_dict_to_model(ckpt["model"], mod), strict=False)
    return ckpt
