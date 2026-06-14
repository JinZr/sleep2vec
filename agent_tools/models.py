from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_VARIANTS = ("sleep2vec", "sleep2vec2", "sleep2expert")
VARIANTLESS_TASKS = {"sleep2stat"}


def task_requires_variant(task: str | None) -> bool:
    return task not in VARIANTLESS_TASKS


def module_for_variant(variant: str, entrypoint: str) -> str:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    return f"{variant}.{entrypoint}"


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


def resolve_repo_path(path: str | Path | None) -> Path | None:
    if path in (None, ""):
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate
