from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

from .adapters import all_adapters
from .adapters.config_providers import CONFIG_SUMMARY_PROVIDERS
from .adapters.sleep2stat import sleep2stat_config_summary  # noqa: F401 -- test-frozen import path
from .domain.finetune_summary import finetune_summary_body, guess_variant
from .models import load_yaml, repo_relative, resolve_repo_path  # noqa: F401 -- load_yaml re-exported for importers


def config_summary(
    config_path: str | Path,
    *,
    variant: str | None = None,
    validate_survival_local_paths: bool = True,
    config_bytes: bytes | None = None,
) -> dict[str, Any]:
    resolved = resolve_repo_path(config_path)
    if resolved is None:
        raise FileNotFoundError("Config path is required.")
    snapshot = None
    summary_path = resolved
    if config_bytes is None:
        data = load_yaml(resolved)
    else:
        data = yaml.safe_load(config_bytes)
        if not isinstance(data, dict):
            raise ValueError(f"YAML must be a mapping: {resolved}")
        # Summary loaders accept paths but preserve configured relative strings, so an immutable snapshot keeps
        # validation byte-exact; the original source path metadata is restored below.
        snapshot = NamedTemporaryFile(suffix=resolved.suffix or ".yaml")
        snapshot.write(config_bytes)
        snapshot.flush()
        summary_path = Path(snapshot.name)
    try:
        for adapter in all_adapters():
            if adapter.matches_config_data(data):
                summary = adapter.config_summary(summary_path)
                summary["config_path"] = repo_relative(resolved)
                return summary
        for provider in CONFIG_SUMMARY_PROVIDERS:
            structural_match = provider.matches(data)
            if (provider.force_variant is not None and variant == provider.force_variant) or structural_match:
                summary = provider.summarize(
                    summary_path,
                    validate_survival_local_paths=validate_survival_local_paths,
                )
                summary["config_path"] = repo_relative(resolved)
                # Structural ownership is authoritative; variant_guess may only reflect the config's directory name.
                if structural_match and provider.force_variant is not None:
                    summary["authoritative_variant"] = provider.force_variant
                return summary
        summary = finetune_summary_body(
            summary_path,
            validate_survival_local_paths=validate_survival_local_paths,
        )
        summary["config_path"] = repo_relative(resolved)
        summary["variant_guess"] = guess_variant(resolved)
        return summary
    finally:
        if snapshot is not None:
            snapshot.close()
