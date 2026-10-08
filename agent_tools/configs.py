"""Thin shell that resolves a config path to its structured summary.

Mixed bridge: dispatches through the ``adapters.config_providers`` tables for
config families claimed by shape, and otherwise delegates the summary body to
``domain.finetune_summary``. Also owns the recipe-level loader that summarizes
a recipe's ``inputs.config`` with the recipe's path-validation context. Never
imports the adapter registry.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

from .adapters.config_providers import CONFIG_SHAPE_SUMMARIES, CONFIG_SUMMARY_PROVIDERS
from .decision_paths import path_context, path_validation
from .domain.finetune_summary import finetune_summary_body, guess_variant
from .models import (
    CONFIG_FINETUNE_SECTION,
    REPO_ROOT,
    YAML_SAFE_LOADER,
    ConfigSummary,
    load_yaml,
    repo_relative,
    resolve_repo_path,
)


def config_summary(
    config_path: str | Path,
    *,
    variant: str | None = None,
    validate_survival_local_paths: bool = True,
    local_path_base: str | Path | None = None,
    config_bytes: bytes | None = None,
    validated_sidecar_keys: dict[str, set[str]] | None = None,
) -> ConfigSummary:
    resolved = resolve_repo_path(config_path)
    if resolved is None:
        raise FileNotFoundError("Config path is required.")
    snapshot = None
    summary_path = resolved
    if config_bytes is None:
        data = load_yaml(resolved)
    else:
        data = yaml.load(config_bytes, Loader=YAML_SAFE_LOADER)
        if not isinstance(data, dict):
            raise ValueError(f"YAML must be a mapping: {resolved}")
        # Summary loaders accept paths but preserve configured relative strings, so an immutable snapshot keeps
        # validation byte-exact; the original source path metadata is restored below.
        snapshot = NamedTemporaryFile(suffix=resolved.suffix or ".yaml")
        snapshot.write(config_bytes)
        snapshot.flush()
        summary_path = Path(snapshot.name)
    summary: ConfigSummary
    try:
        for matches, summarize in CONFIG_SHAPE_SUMMARIES:
            if matches(data):
                summary = summarize(summary_path)
                summary["config_path"] = repo_relative(resolved)
                return summary
        for provider in CONFIG_SUMMARY_PROVIDERS:
            structural_match = provider.matches(data)
            if (provider.force_variant is not None and variant == provider.force_variant) or structural_match:
                summary = provider.summarize(
                    summary_path,
                    validate_survival_local_paths=validate_survival_local_paths,
                    local_path_base=local_path_base,
                    validated_sidecar_keys=validated_sidecar_keys,
                )
                summary["config_path"] = repo_relative(resolved)
                # Structural ownership is authoritative; variant_guess may only reflect the config's directory name.
                if structural_match and provider.force_variant is not None:
                    summary["authoritative_variant"] = provider.force_variant
                return summary
        summary = finetune_summary_body(
            summary_path,
            validate_survival_local_paths=validate_survival_local_paths,
            local_path_base=local_path_base,
            validated_sidecar_keys=validated_sidecar_keys,
        )
        summary["config_path"] = repo_relative(resolved)
        summary["variant_guess"] = guess_variant(resolved)
        return summary
    finally:
        if snapshot is not None:
            snapshot.close()


def load_config_summary_for_recipe(
    recipe: dict,
    *,
    config_bytes: bytes | None = None,
    validated_sidecar_keys: dict[str, set[str]] | None = None,
) -> ConfigSummary | None:
    inputs = recipe["inputs"] if isinstance(recipe.get("inputs"), dict) else {}
    config = inputs.get("config")
    if not config:
        return None
    resolved = resolve_repo_path(config)
    if resolved is None or (config_bytes is None and not resolved.exists()):
        return None
    try:
        config_data = load_yaml(config) if config_bytes is None else yaml.load(config_bytes, Loader=YAML_SAFE_LOADER)
    except Exception:
        config_data = {}
    return config_summary(
        config,
        variant=recipe.get("variant"),
        validate_survival_local_paths=not skips_local_path_validation(
            recipe,
            survival_validation_paths(config_data),
        ),
        local_path_base=runtime_path_base(recipe),
        config_bytes=config_bytes,
        validated_sidecar_keys=validated_sidecar_keys,
    )


def skips_local_path_validation(recipe: dict, raw_paths: list[Any] | None = None) -> bool:
    for raw_path in raw_paths or [""]:
        context = path_context(recipe, raw_path, relative_to_workdir=True)
        if context == "remote" and path_validation(recipe, context) in {"defer", "ssh", "remote"}:
            return True
    return False


def runtime_path_base(recipe: dict) -> Path:
    execution = recipe["execution"] if isinstance(recipe.get("execution"), dict) else {}
    workdir = execution.get("workdir")
    if workdir not in (None, "") and Path(str(workdir)).is_absolute():
        return Path(str(workdir))
    return REPO_ROOT


def survival_validation_paths(config_data: dict | None) -> list[Any]:
    if not isinstance(config_data, dict):
        return []
    data = config_data["data"] if isinstance(config_data.get("data"), dict) else {}
    finetune = (
        config_data[CONFIG_FINETUNE_SECTION] if isinstance(config_data.get(CONFIG_FINETUNE_SECTION), dict) else {}
    )
    survival = finetune["survival"] if isinstance(finetune.get("survival"), dict) else {}
    multilabel = finetune["multilabel"] if isinstance(finetune.get("multilabel"), dict) else {}
    paths = [data.get("finetune_data_index"), data.get("finetune_preset_path")]
    paths.extend(data.get(field) for field in ("kaldi_data_root", "kaldi_manifest"))
    paths.extend(
        survival.get(field)
        for field in ("disease_columns_index", "event_time_index", "is_event_index", "has_label_index")
    )
    paths.extend(multilabel.get(field) for field in ("disease_columns_index", "label_index", "has_label_index"))
    return [path for path in paths if path not in (None, "")]
