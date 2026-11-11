"""Thin shell that resolves a config path to its structured summary.

Config families claimed by shape come first: a sleep2stat config, then a
``sex_age_baseline`` config (also forced by that variant). Everything else gets
the generic body from ``domain.finetune_summary``. Also owns the recipe-level
loader that summarizes a recipe's ``inputs.config`` relative to the recipe's
runtime workdir. Never imports the adapter registry.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

from .adapters.sleep2stat import looks_like_sleep2stat_config_data, sleep2stat_config_summary
from .domain.finetune_summary import finetune_summary_body, guess_variant
from .domain.sex_age_summary import looks_like_sex_age_baseline_config_data, sex_age_baseline_config_summary
from .models import REPO_ROOT, YAML_SAFE_LOADER, ConfigSummary, load_yaml, repo_relative, resolve_repo_path


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
        # A shape claim wins over any variant the caller names.
        if looks_like_sleep2stat_config_data(data):
            summary = sleep2stat_config_summary(summary_path)
            summary["config_path"] = repo_relative(resolved)
            return summary
        structural_match = looks_like_sex_age_baseline_config_data(data)
        if variant == "sex_age_baseline" or structural_match:
            summary = sex_age_baseline_config_summary(
                summary_path,
                validate_survival_local_paths=validate_survival_local_paths,
                local_path_base=local_path_base,
                validated_sidecar_keys=validated_sidecar_keys,
            )
            summary["config_path"] = repo_relative(resolved)
            # Structural ownership is authoritative; variant_guess may only reflect the config's directory name.
            if structural_match:
                summary["authoritative_variant"] = "sex_age_baseline"
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
    return config_summary(
        config,
        variant=recipe.get("variant"),
        local_path_base=runtime_path_base(recipe),
        config_bytes=config_bytes,
        validated_sidecar_keys=validated_sidecar_keys,
    )


def runtime_path_base(recipe: dict) -> Path:
    execution = recipe["execution"] if isinstance(recipe.get("execution"), dict) else {}
    workdir = execution.get("workdir")
    if workdir not in (None, "") and Path(str(workdir)).is_absolute():
        return Path(str(workdir))
    return REPO_ROOT
