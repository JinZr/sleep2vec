from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_VARIANTS = ("sleep2vec", "sleep2vec2", "sleep2expert", "sex_age_baseline")
VARIANTLESS_TASKS = {"sleep2stat"}
# Domain-config schema section name. Shares its spelling with the finetune
# task name but is config vocabulary, not task dispatch (the adapter leak
# guard matches raw task-name constants, so kernel modules read the section
# through this constant).
CONFIG_FINETUNE_SECTION = "finetune"


def recipe_name(recipe: dict[str, Any]) -> str:
    return str(recipe.get("name") or Path(str(recipe.get("_recipe_path", "recipe"))).stem)


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    if resolved is None:
        raise FileNotFoundError("Config path is required.")
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {resolved}")
    return data


def task_requires_variant(task: str | None) -> bool:
    return task not in VARIANTLESS_TASKS


def module_for_variant(variant: str, entrypoint: str) -> str:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    return f"{variant}.{entrypoint}"


def coerce_list(value: Any) -> list[Any]:
    if value in (None, "", "ASK_USER"):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def repo_relative(path: str | Path | None) -> str | None:
    if path in (None, ""):
        return None
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(REPO_ROOT.resolve()))
    except (OSError, ValueError):
        return str(raw)


def resolve_repo_path(path: str | Path | None, *, relative_to: str | Path | None = None) -> Path | None:
    if path in (None, ""):
        return None
    candidate = Path(path)
    if relative_to is None:
        candidate = candidate.expanduser()
    if not candidate.is_absolute():
        candidate = Path(relative_to) / candidate if relative_to is not None else REPO_ROOT / candidate
    return candidate
