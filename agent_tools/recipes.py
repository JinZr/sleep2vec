from __future__ import annotations

from pathlib import Path
from typing import Any

from .experiment_workspace import read_managed_yaml_mapping
from .models import repo_relative, resolve_repo_path


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    if resolved is None:
        raise FileNotFoundError("Path is required.")
    return read_managed_yaml_mapping(resolved.read_text(), source=f"YAML file {resolved}")


def load_recipe(path: str | Path) -> dict[str, Any]:
    recipe = load_yaml_file(path)
    reserved = sorted(str(key) for key in recipe if str(key).startswith("_"))
    if reserved:
        raise ValueError(f"Recipe cannot define reserved internal field(s): {', '.join(reserved)}")
    recipe["_recipe_path"] = repo_relative(resolve_repo_path(path))
    return recipe


def load_recipe_with_base(path: str | Path) -> dict[str, Any]:
    recipe_path = resolve_repo_path(path)
    if recipe_path is None:
        raise FileNotFoundError("Path is required.")
    recipe = load_recipe(recipe_path)
    base_path = recipe.get("base_recipe")
    if base_path in (None, ""):
        return recipe
    if not isinstance(base_path, (str, Path)):
        raise ValueError("base_recipe must be a path string.")
    base = load_recipe(_resolve_base_recipe_path(base_path, recipe_path))
    merged = _deep_merge(base, recipe)
    merged["_base_recipe"] = base
    merged["_local_recipe"] = recipe
    merged["_recipe_path"] = recipe["_recipe_path"]
    return merged


def load_user_decisions(path: str | Path | None) -> dict[str, Any]:
    if path in (None, ""):
        return {}
    data = load_yaml_file(path)
    decisions = data.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError(f"User-decision file must contain a decisions mapping: {resolve_repo_path(path)}")
    unknown = sorted(set(data) - {"decisions"})
    if unknown:
        raise ValueError(f"Unknown user-decision file field(s): {', '.join(str(field) for field in unknown)}")
    return decisions


def load_consultation_policy() -> dict[str, Any]:
    return load_yaml_file("agent_policies/consultation_policy.yaml")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_base_recipe_path(base_path: str | Path, recipe_path: Path | None) -> Path | str:
    candidate = Path(base_path).expanduser()
    if candidate.is_absolute() or recipe_path is None:
        return candidate
    recipe_relative = recipe_path.parent / candidate
    if recipe_relative.exists():
        return recipe_relative
    return base_path


def recipe_name(recipe: dict[str, Any]) -> str:
    return str(recipe.get("name") or Path(str(recipe.get("_recipe_path", "recipe"))).stem)
