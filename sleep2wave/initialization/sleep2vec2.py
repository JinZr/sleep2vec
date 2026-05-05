from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import typing as t

import torch

from sleep2wave.checkpoints import get_state_dict_from_checkpoint, load_checkpoint
from sleep2wave.generative.config import InitializationConfig

INITIALIZATION_GROUPS = {
    "tokenizers",
    "backbone",
    "projection",
    "autoencoder_encoders",
    "diffusion_transformer",
}

_PREFERRED_PREFIXES = ("ema_model.", "model.")


@dataclass
class Sleep2Vec2InitializationReport:
    used_prefix: str | None = None
    loaded_groups: list[str] = field(default_factory=list)
    loaded_keys: list[str] = field(default_factory=list)
    skipped_missing_target: list[str] = field(default_factory=list)
    skipped_shape_mismatch: list[str] = field(default_factory=list)
    skipped_disabled_group: list[str] = field(default_factory=list)
    skipped_unknown_group: list[str] = field(default_factory=list)
    skipped_unknown_prefix: list[str] = field(default_factory=list)


def _group_for_key(key: str) -> str | None:
    if key.startswith("tokenizer_mapping."):
        return "tokenizers"
    if key.startswith("embedding_projection.") or key.startswith("proj_head."):
        return "projection"
    if key.startswith("encoder.") or key.startswith("cls_embedding."):
        return "backbone"
    if key.startswith("modality_autoencoders.") and ".encoder." in key:
        return "autoencoder_encoders"
    if (
        key.startswith("input_projection.")
        or key.startswith("output_projection.")
        or key.startswith("diffusion_step_embedding.")
        or key.startswith("modality_embedding.")
        or key.startswith("epoch_position_embedding.")
        or key.startswith("sleep_night_position_projection.")
        or key.startswith("availability_embedding.")
        or key.startswith("quality_projection.")
        or key.startswith("layers.")
        or key.startswith("final_norm.")
    ):
        return "diffusion_transformer"
    return None


def _filter_by_prefix(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], str | None, list[str]]:
    unknown_prefix_keys = [
        key for key in state_dict if not any(key.startswith(prefix) for prefix in _PREFERRED_PREFIXES)
    ]
    for prefix in _PREFERRED_PREFIXES:
        filtered = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
        if filtered:
            return filtered, prefix, sorted(unknown_prefix_keys)
    return {}, None, sorted(unknown_prefix_keys)


def load_sleep2vec2_initialization(
    target: torch.nn.Module,
    checkpoint_path: Path | str | None,
    config: InitializationConfig | None,
    *,
    target_groups: t.Collection[str],
    device: torch.device | str = "cpu",
) -> Sleep2Vec2InitializationReport:
    if config is None:
        return Sleep2Vec2InitializationReport()
    resolved_checkpoint_path = checkpoint_path if checkpoint_path is not None else config.sleep2vec2_checkpoint
    if resolved_checkpoint_path is None:
        return Sleep2Vec2InitializationReport()

    unknown_target_groups = sorted(set(target_groups) - INITIALIZATION_GROUPS)
    if unknown_target_groups:
        raise ValueError(f"Unknown sleep2wave initialization target groups: {unknown_target_groups}")
    unknown_load_groups = sorted(set(config.load_groups) - INITIALIZATION_GROUPS)
    if unknown_load_groups:
        raise ValueError(f"Unknown sleep2wave initialization load groups: {unknown_load_groups}")

    ckpt = load_checkpoint(resolved_checkpoint_path, device)
    source_state_dict = get_state_dict_from_checkpoint(ckpt)
    filtered_state_dict, used_prefix, unknown_prefix_keys = _filter_by_prefix(source_state_dict)
    target_state_dict = target.state_dict()
    compatible_state_dict: dict[str, torch.Tensor] = {}
    report = Sleep2Vec2InitializationReport(
        used_prefix=used_prefix,
        skipped_unknown_prefix=unknown_prefix_keys,
    )

    enabled_groups = {group for group, enabled in config.load_groups.items() if enabled}
    for key, value in sorted(filtered_state_dict.items()):
        group = _group_for_key(key)
        if group is None:
            report.skipped_unknown_group.append(key)
            continue
        if group not in target_groups or group not in enabled_groups:
            report.skipped_disabled_group.append(key)
            continue
        if key not in target_state_dict:
            report.skipped_missing_target.append(key)
            continue
        target_value = target_state_dict[key]
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(target_value.shape):
            report.skipped_shape_mismatch.append(key)
            continue
        compatible_state_dict[key] = value

    if config.strict_compatible and report.skipped_shape_mismatch:
        preview = ", ".join(report.skipped_shape_mismatch[:5])
        raise ValueError(f"sleep2wave initialization shape mismatch for enabled groups: {preview}")

    if compatible_state_dict:
        target.load_state_dict(compatible_state_dict, strict=False)
        report.loaded_keys = sorted(compatible_state_dict)
        report.loaded_groups = sorted({_group_for_key(key) for key in compatible_state_dict if _group_for_key(key)})

    if config.require_any_loaded and not report.loaded_keys:
        raise ValueError(
            "sleep2wave initialization did not load any compatible sleep2vec2 checkpoint keys. "
            f"Used prefix: {report.used_prefix!r}."
        )

    return report


__all__ = [
    "INITIALIZATION_GROUPS",
    "Sleep2Vec2InitializationReport",
    "load_sleep2vec2_initialization",
]
