from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters import all_adapters
from .adapters.config_providers import CONFIG_SUMMARY_PROVIDERS
from .adapters.sleep2stat import sleep2stat_config_summary  # noqa: F401 -- test-frozen import path
from .domain.finetune_summary import finetune_summary_body
from .models import (  # noqa: F401 -- load_yaml re-exported for existing importers
    load_yaml,
    resolve_repo_path,
)


def config_summary(
    config_path: str | Path,
    *,
    variant: str | None = None,
    validate_survival_local_paths: bool = True,
) -> dict[str, Any]:
    resolved = resolve_repo_path(config_path)
    if resolved is None:
        raise FileNotFoundError("Config path is required.")
    data = load_yaml(resolved)
    for adapter in all_adapters():
        if adapter.matches_config_data(data):
            return adapter.config_summary(resolved)
    for provider in CONFIG_SUMMARY_PROVIDERS:
        if (provider.force_variant is not None and variant == provider.force_variant) or provider.matches(data):
            return provider.summarize(resolved, validate_survival_local_paths=validate_survival_local_paths)
    return finetune_summary_body(resolved, validate_survival_local_paths=validate_survival_local_paths)
