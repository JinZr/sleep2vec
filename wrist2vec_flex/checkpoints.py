from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import typing as t

import torch

_EPOCH_RE = re.compile(r"epoch[=\-](\d+)")
_SOURCE_MODULE_ROOT = "channel_source_encoder_mapping."
_SOURCE_MODULE_PARTS = (".source_embedding.", ".source_fusion.")
_STATE_WRAPPER_PREFIXES = ("model.backbone.", "backbone.", "model.", "ema_model.", "running_mean_model.")


@dataclass
class PretrainInitLoadResult:
    used_prefix: str
    loaded_keys: int
    missing_keys: list[str]
    unexpected_keys: list[str]


def is_source_module_state_key(key: str) -> bool:
    stripped = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in _STATE_WRAPPER_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                changed = True
                break
    return stripped.startswith(_SOURCE_MODULE_ROOT) and any(part in stripped for part in _SOURCE_MODULE_PARTS)


def _parse_epoch(path: Path) -> int | None:
    match = _EPOCH_RE.search(path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def load_checkpoint(path: Path | str, device: torch.device | str) -> t.Any:
    return torch.load(Path(path), map_location=torch.device(device), weights_only=False)


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    ckpt = load_checkpoint(path, device)
    return get_state_dict_from_checkpoint(ckpt)


def get_state_dict_from_checkpoint(ckpt: t.Any) -> dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Checkpoint payload must be a Lightning .ckpt with top-level 'state_dict'.")

    state = ckpt["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint payload 'state_dict' must be a mapping.")
    return state


def extract_pretrain_init_state_dict(
    ckpt: t.Any,
    *,
    prefixes: t.Sequence[str] = ("ema_model.", "model."),
) -> tuple[dict[str, torch.Tensor], str]:
    state_dict = get_state_dict_from_checkpoint(ckpt)
    for prefix in prefixes:
        filtered = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if filtered:
            return filtered, prefix

    preview = ", ".join(list(state_dict.keys())[:8])
    raise ValueError(
        "Checkpoint does not contain any pretrain-model subtree matching "
        f"{list(prefixes)}. Example keys: [{preview}]"
    )


def load_pretrain_init_weights(
    module: torch.nn.Module,
    ckpt_path: Path | str,
    *,
    device: torch.device | str = torch.device("cpu"),
    strict: bool = False,
    prefixes: t.Sequence[str] = ("ema_model.", "model."),
) -> PretrainInitLoadResult:
    ckpt = load_checkpoint(ckpt_path, device)
    filtered_state_dict, used_prefix = extract_pretrain_init_state_dict(ckpt, prefixes=prefixes)
    load_info = module.load_state_dict(filtered_state_dict, strict=strict)
    return PretrainInitLoadResult(
        used_prefix=used_prefix,
        loaded_keys=len(filtered_state_dict),
        missing_keys=list(load_info.missing_keys),
        unexpected_keys=list(load_info.unexpected_keys),
    )


def select_checkpoints(
    ckpt_dir: Path,
    *,
    end_ckpt: Path | None,
    num_ckpts: int,
) -> list[Path]:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise ValueError(f"No .ckpt files found under {ckpt_dir}")

    if end_ckpt is not None:
        end_ckpt = Path(end_ckpt)
        if not end_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {end_ckpt}")

    epoch_pairs: list[tuple[int, Path]] = []
    for path in ckpts:
        epoch = _parse_epoch(path)
        if epoch is not None:
            epoch_pairs.append((epoch, path))

    if epoch_pairs:
        epoch_pairs.sort(key=lambda item: item[0])
        if end_ckpt is not None:
            end_epoch = _parse_epoch(end_ckpt)
            if end_epoch is not None:
                epoch_pairs = [item for item in epoch_pairs if item[0] <= end_epoch]
            else:
                try:
                    end_idx = [p for _, p in epoch_pairs].index(end_ckpt)
                    epoch_pairs = epoch_pairs[: end_idx + 1]
                except ValueError:
                    pass
        selected = [p for _, p in epoch_pairs][-num_ckpts:]
        if len(selected) == num_ckpts:
            return selected

    ckpts_sorted = sorted(ckpts, key=lambda p: p.stat().st_mtime)
    if end_ckpt is not None:
        end_mtime = end_ckpt.stat().st_mtime
        ckpts_sorted = [p for p in ckpts_sorted if p.stat().st_mtime <= end_mtime + 1e-6]
    selected = ckpts_sorted[-num_ckpts:]
    if len(selected) < num_ckpts:
        raise ValueError(f"Not enough checkpoints to average: requested {num_ckpts}, found {len(selected)}")
    return selected


def average_checkpoints(
    filenames: t.Sequence[Path],
    *,
    device: torch.device | str = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    if not filenames:
        raise ValueError("No checkpoints provided for averaging.")

    device = torch.device(device)
    state = _load_state_dict(Path(filenames[0]), device=device)
    avg: dict[str, torch.Tensor] = {}

    for key, value in state.items():
        if torch.is_tensor(value):
            avg[key] = value.clone()

    for path in filenames[1:]:
        state_dict = _load_state_dict(Path(path), device=device)
        for key, value in avg.items():
            if key not in state_dict:
                raise KeyError(f"Missing key '{key}' in checkpoint {path}")
            avg[key] += state_dict[key]

    n = len(filenames)
    for key, value in avg.items():
        if value.is_floating_point():
            value.div_(n)
        else:
            value //= n

    logging.info("Averaged %d checkpoints", n)
    return avg


__all__ = [
    "PretrainInitLoadResult",
    "average_checkpoints",
    "extract_pretrain_init_state_dict",
    "get_state_dict_from_checkpoint",
    "is_source_module_state_key",
    "load_checkpoint",
    "load_pretrain_init_weights",
    "select_checkpoints",
]
