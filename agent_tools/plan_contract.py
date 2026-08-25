from __future__ import annotations

import copy
from pathlib import Path
import re
import sys
from typing import Any

from . import plan_rendering as rendering
from .experiment_workspace import run_identity, safe_artifact_name
from .models import REPO_ROOT, recipe_name

FROZEN_FINAL_EVAL_CONFIG_NAME = "config.final_eval.yaml"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PLAN_CONTEXT_FIELDS = {"home", "python", "repo_root"}


def bind_plan_context(recipe: dict[str, Any]) -> None:
    recipe["_plan_context"] = {
        "home": str(Path.home()),
        "python": sys.executable,
        "repo_root": str(REPO_ROOT),
    }


def frozen_plan_context(recipe: dict[str, Any]) -> dict[str, str]:
    context = recipe.get("_plan_context")
    if (
        not isinstance(context, dict)
        or set(context) != _PLAN_CONTEXT_FIELDS
        or any(not isinstance(context[field], str) or not context[field] for field in _PLAN_CONTEXT_FIELDS)
        or not Path(context["home"]).is_absolute()
        or not Path(context["python"]).is_absolute()
        or not Path(context["repo_root"]).is_absolute()
    ):
        raise ValueError("Frozen recipe must define an exact absolute _plan_context.")
    return dict(context)


def resolve_frozen_repo_path(recipe: dict[str, Any], path: Any) -> Path | None:
    if path in (None, ""):
        return None
    context = frozen_plan_context(recipe)
    text = str(path)
    if text == "~":
        return Path(context["home"])
    if text.startswith("~/"):
        return Path(context["home"]) / text[2:]
    candidate = Path(text)
    return candidate if candidate.is_absolute() else Path(context["repo_root"]) / candidate


def frozen_input_snapshots(recipe: dict[str, Any]) -> list[dict[str, str]]:
    snapshots = recipe.get("input_snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("Frozen recipe must define input_snapshots.")
    normalized = []
    for snapshot in snapshots:
        if (
            not isinstance(snapshot, dict)
            or set(snapshot) != {"field", "path", "sha256"}
            or any(not isinstance(snapshot[field], str) or not snapshot[field] for field in ("field", "path"))
            or not isinstance(snapshot["sha256"], str)
            or _SHA256_RE.fullmatch(snapshot["sha256"]) is None
        ):
            raise ValueError("Frozen recipe input_snapshots must define field, path, and SHA-256.")
        normalized.append(dict(snapshot))
    if normalized != sorted(normalized, key=lambda item: (item["field"], item["path"])):
        raise ValueError("Frozen recipe input_snapshots must use stable field/path ordering.")
    if len({item["field"] for item in normalized}) != len(normalized):
        raise ValueError("Frozen recipe input_snapshots must define each field once.")
    return normalized


def frozen_input_snapshot(recipe: dict[str, Any], field: str) -> dict[str, str]:
    matches = [snapshot for snapshot in frozen_input_snapshots(recipe) if snapshot["field"] == field]
    if len(matches) != 1:
        raise ValueError(f"Frozen recipe must define exactly one {field} input snapshot.")
    return matches[0]


def bind_frozen_input_snapshot(recipe: dict[str, Any], field: str, path: str | Path, sha256: str) -> None:
    if _SHA256_RE.fullmatch(sha256) is None:
        raise ValueError(f"Frozen input snapshot for {field} requires a lowercase SHA-256.")
    snapshots = recipe.get("input_snapshots")
    retained = (
        [snapshot for snapshot in snapshots if isinstance(snapshot, dict) and snapshot.get("field") != field]
        if isinstance(snapshots, list)
        else []
    )
    retained.append({"field": field, "path": str(path), "sha256": sha256})
    recipe["input_snapshots"] = sorted(retained, key=lambda item: (item["field"], item["path"]))


def generic_run_contract(
    recipe: dict[str, Any],
    plan_dir: Path,
    run_index: int,
    adapter: Any,
) -> dict[str, Any]:
    context = frozen_plan_context(recipe)
    declared_name = safe_artifact_name((recipe.get("artifacts") or {}).get("version_name") or recipe_name(recipe))
    identity = run_identity(recipe, run_index, {}, run_name=declared_name)
    run_dir = plan_dir / "runs" / f"{identity['run_id']}--{identity['run_name']}"
    runtime_recipe = copy.deepcopy(recipe)
    runtime_recipe.setdefault("inputs", {})["config"] = str(run_dir / "config.yaml")
    runtime_recipe.setdefault("artifacts", {})["version_name"] = identity["version"]
    runtime_recipe.setdefault("execution", {}).setdefault("workdir", context["repo_root"])
    runtime_dir = adapter.managed_runtime_dir(runtime_recipe, identity["version"])
    checkpoint_dir = runtime_dir / "checkpoints" if runtime_dir is not None else None
    return {
        "experiment_id": (recipe.get("experiment") or {}).get("id"),
        "step_id": (recipe.get("step") or {}).get("id"),
        **identity,
        "config": str(run_dir / "config.yaml"),
        "script": str(run_dir / "launch.sh"),
        "run_dir": str(run_dir),
        "artifacts": str(run_dir / "artifacts.json"),
        "runtime_dir": str(runtime_dir) if runtime_dir is not None else "",
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else "",
    }


def generic_commands(
    recipe: dict[str, Any],
    run: dict[str, Any],
    adapter: Any,
    config_bytes: bytes,
) -> list[str]:
    runtime_recipe = copy.deepcopy(recipe)
    runtime_recipe.setdefault("inputs", {})["config"] = run["config"]
    runtime_recipe.setdefault("artifacts", {})["version_name"] = run["version"]
    return adapter.frozen_commands(runtime_recipe, config_bytes)


def generic_script_text(
    recipe: dict[str, Any],
    run: dict[str, Any],
    adapter: Any,
    commands: list[str],
    input_snapshots: list[dict[str, str]],
) -> str:
    context = frozen_plan_context(recipe)
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    runtime_identity = (
        execution
        if adapter.supports_runtime_identity
        and all(field in execution for field in ("python", "runtime_commit", "workdir"))
        else {}
    )
    experiment = recipe.get("experiment") if isinstance(recipe.get("experiment"), dict) else {}
    return (
        "\n".join(
            rendering.script_lines(
                commands,
                run_cwd=Path(str(execution.get("workdir") or context["repo_root"])),
                experiment_root=Path(str(experiment.get("root"))),
                step_id=run["step_id"],
                run_id=run["run_id"],
                lifecycle_python=runtime_identity.get("python") or context["python"],
                expected_runtime_commit=runtime_identity.get("runtime_commit"),
                input_snapshots=input_snapshots,
            )
        )
        + "\n"
    )


def validate_final_eval_contract(
    plan: dict[str, Any],
    recipe: dict[str, Any],
    plan_dir: Path,
    contract: dict[str, Any],
) -> tuple[Path | None, str | None]:
    required = bool(contract.get("final_eval_config_required"))
    present = "final_eval_config" in plan
    if required != present:
        requirement = "missing" if required else "unexpected"
        raise ValueError(f"Registered plan has {requirement} final_eval_config: {plan_dir / 'plan.json'}")
    final_path = None
    if present:
        descriptor = plan["final_eval_config"]
        expected_sha256 = contract.get("final_eval_config_sha256")
        required_fields = {"path", "sha256", "source_path"}
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != required_fields
            or any(not isinstance(descriptor[field], str) or not descriptor[field].strip() for field in required_fields)
            or _SHA256_RE.fullmatch(descriptor["sha256"]) is None
        ):
            raise ValueError(
                f"Registered final_eval_config must define path, sha256, and source_path: {plan_dir / 'plan.json'}"
            )
        if descriptor["sha256"] != expected_sha256:
            raise ValueError(
                f"Registered final_eval_config differs from its frozen recipe digest: {plan_dir / 'plan.json'}"
            )
        final_path = plan_dir / FROZEN_FINAL_EVAL_CONFIG_NAME
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        if descriptor["path"] != str(final_path) or descriptor["source_path"] != str(
            inputs.get("final_eval_config_path") or ""
        ):
            raise ValueError(f"Registered final_eval_config differs from its frozen recipe: {plan_dir / 'plan.json'}")
    return final_path, contract.get("final_command")
