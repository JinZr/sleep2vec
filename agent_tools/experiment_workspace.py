from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any

import yaml

from . import experiment_io as exp_io
from .models import REPO_ROOT, json_ready

PHASES = {"prepare", "train", "evaluate", "analyze"}
TERMINAL_STATUSES = {"completed", "failed", "finished", "launch_failed", "stopped", "superseded"}
EXECUTION_IDENTITY_FIELDS = {
    "target",
    "host",
    "workdir",
    "gpus",
    "pid_path",
    "log_path",
    "command",
}
FROZEN_RUN_FIELDS = {
    "experiment_id",
    "step_id",
    "run_id",
    "run_name",
    "parameter_summary",
    "version",
    "config",
    "config_sha256",
    "script",
    "script_sha256",
    "run_dir",
    "artifacts",
    "runtime_dir",
    "checkpoint_dir",
} | EXECUTION_IDENTITY_FIELDS
MANAGED_RUN_PATH_FIELDS = {
    "artifacts",
    "checkpoint_dir",
    "checkpoint_path",
    "config",
    "log_path",
    "pid_path",
    "progress_dir",
    "round_dir",
    "run_dir",
    "run_manifest",
    "runtime_dir",
    "script",
    "test_logits_path",
    "test_predictions_path",
    "val_logits_path",
    "val_predictions_path",
    "workdir",
}


def experiment_metadata_issues(
    recipe: dict[str, Any],
    *,
    require_values: bool = True,
    source_layer: str | None = None,
    allow_step_io: bool = False,
) -> list[dict[str, Any]]:
    experiment = recipe.get("experiment")
    step = recipe.get("step")
    issues = []
    contract_evidence = {"preflight_before_workspace": True}
    if source_layer is not None:
        contract_evidence["source_layer"] = source_layer
    if not isinstance(experiment, dict):
        if experiment is not None:
            issues.append(
                {
                    "status": "FAIL",
                    "field": "experiment",
                    "message": "experiment must be a mapping.",
                    "evidence": {**contract_evidence, "value": experiment},
                }
            )
        elif require_values:
            issues.append(
                {
                    "status": "NEEDS_USER_INPUT",
                    "field": "experiment",
                    "message": "Recipe is not bound to an experiment workspace.",
                    "question": "What experiment id, title, objective, root, and baseline should own this task?",
                }
            )
    else:
        for field in sorted(set(experiment) - {"id", "title", "objective", "root", "baseline"}):
            issues.append(
                {
                    "status": "FAIL",
                    "field": f"experiment.{field}",
                    "message": f"Unknown experiment field: {field}.",
                    "evidence": {**contract_evidence, field: experiment[field]},
                }
            )
        if require_values:
            for field in ("id", "title", "objective", "root", "baseline"):
                if experiment.get(field) in (None, "", "ASK_USER"):
                    issues.append(
                        {
                            "status": "NEEDS_USER_INPUT",
                            "field": f"experiment.{field}",
                            "message": f"experiment.{field} is not explicitly resolved.",
                            "question": f"What should experiment.{field} be for this task?",
                        }
                    )
        experiment_id = experiment.get("id")
        if experiment_id not in (None, "", "ASK_USER"):
            if not isinstance(experiment_id, str):
                issues.append(
                    {
                        "status": "FAIL",
                        "field": "experiment.id",
                        "message": "experiment.id must be a string.",
                    }
                )
            elif not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", experiment_id):
                issues.append(
                    {
                        "status": "FAIL",
                        "field": "experiment.id",
                        "message": "experiment.id must use lowercase letters, digits, hyphens, and underscores.",
                    }
                )
    if not isinstance(step, dict):
        if step is not None:
            issues.append(
                {
                    "status": "FAIL",
                    "field": "step",
                    "message": "step must be a mapping.",
                    "evidence": {**contract_evidence, "value": step},
                }
            )
        elif require_values:
            issues.append(
                {
                    "status": "NEEDS_USER_INPUT",
                    "field": "step",
                    "message": "Recipe does not define its experiment step.",
                    "question": "What step id, phase, and purpose should describe this task?",
                }
            )
        return issues
    step_fields = {"id", "phase", "purpose"}
    if allow_step_io:
        step_fields.update({"inputs", "outputs"})
    for field in sorted(set(step) - step_fields):
        issues.append(
            {
                "status": "FAIL",
                "field": f"step.{field}",
                "message": f"Unknown step field: {field}.",
                "evidence": {**contract_evidence, field: step[field]},
            }
        )
    if require_values:
        for field in ("id", "phase", "purpose"):
            if step.get(field) in (None, "", "ASK_USER"):
                issues.append(
                    {
                        "status": "NEEDS_USER_INPUT",
                        "field": f"step.{field}",
                        "message": f"step.{field} is not explicitly resolved.",
                        "question": f"What should step.{field} be for this task?",
                    }
                )
    step_id = step.get("id")
    if step_id not in (None, "", "ASK_USER"):
        if not isinstance(step_id, str):
            issues.append(
                {
                    "status": "FAIL",
                    "field": "step.id",
                    "message": "step.id must be a string.",
                }
            )
        elif not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", step_id):
            issues.append(
                {
                    "status": "FAIL",
                    "field": "step.id",
                    "message": "step.id must use lowercase letters, digits, hyphens, and underscores.",
                }
            )
    phase = step.get("phase")
    if phase not in (None, "", "ASK_USER") and phase not in PHASES:
        issues.append(
            {
                "status": "FAIL",
                "field": "step.phase",
                "message": f"step.phase must be one of {sorted(PHASES)}.",
            }
        )
    return issues


def experiment_root(recipe: dict[str, Any]) -> Path | None:
    raw = (recipe.get("experiment") or {}).get("root")
    if raw in (None, "", "ASK_USER"):
        return None
    return canonical_local_experiment_root(raw, REPO_ROOT)


def canonical_local_experiment_root(raw: str | Path, base_dir: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(base_dir).expanduser() / path
    path = Path(os.path.normpath(path))
    if path.is_symlink():
        raise ValueError(f"Local experiment root must not be a symlink: {path}")
    return path.resolve()


def read_managed_yaml_mapping(text: str, *, source: str | Path) -> dict[str, Any]:
    label = str(source)
    if not text.strip():
        raise ValueError(f"{label} is empty.")
    document = yaml.compose(text)
    pending = [(document, False)]
    active_nodes = set()
    visited_nodes = set()
    while pending:
        node, leaving = pending.pop()
        node_id = id(node)
        if leaving:
            active_nodes.remove(node_id)
            visited_nodes.add(node_id)
            continue
        if node_id in active_nodes:
            raise ValueError(f"{label} has a recursive YAML alias.")
        if node_id in visited_nodes:
            continue
        active_nodes.add(node_id)
        pending.append((node, True))
        if isinstance(node, yaml.MappingNode):
            keys = set()
            for key_node, value_node in node.value:
                if not isinstance(key_node, yaml.ScalarNode):
                    raise ValueError(f"{label} has a non-scalar key.")
                key = (key_node.tag, key_node.value)
                if key in keys:
                    raise ValueError(f"{label} has a duplicate key: {key_node.value}.")
                keys.add(key)
                pending.append((value_node, False))
        elif isinstance(node, yaml.SequenceNode):
            pending.extend((item, False) for item in node.value)
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{label} must contain a non-empty mapping.")
    return payload


def validate_plan_output(recipe: dict[str, Any], output_dir: str | Path) -> str | None:
    root = experiment_root(recipe)
    if root is None:
        return None
    if _is_nonempty_unmanaged_root(root):
        return f"Experiment root is non-empty and has no experiment.yaml: {root}"
    manifest_path = root / "experiment.yaml"
    if manifest_path.exists():
        manifest = read_managed_yaml_mapping(
            manifest_path.read_text(), source=f"Managed experiment manifest {manifest_path}"
        )
        experiment = manifest.get("experiment") if isinstance(manifest, dict) else None
        if isinstance(experiment, dict) and experiment.get("status") == "completed":
            return f"Experiment is completed and cannot accept new plans: {root}"
    out = Path(output_dir).expanduser()
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    else:
        out = out.resolve()
    try:
        out.relative_to(root.resolve())
    except ValueError:
        return f"Plan output must be inside experiment.root: {root}"
    return None


def merge_step_manifest(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    allowed_fields = {"step", "experiment_id", "recipe_path", "plans"}
    for source, payload in (("existing", existing), ("incoming", incoming)):
        if not isinstance(payload, dict):
            raise ValueError(f"{source} step manifest must be a mapping.")
        unexpected = sorted(set(payload) - allowed_fields)
        if unexpected:
            raise ValueError(f"Unexpected {source} step manifest fields: {', '.join(unexpected)}")
        if "step" in payload and not isinstance(payload["step"], dict):
            raise ValueError(f"{source} step manifest step must be a mapping.")
        if "plans" in payload and not isinstance(payload["plans"], list):
            raise ValueError(f"{source} step manifest plans must be a list.")

    merged_step = dict(existing.get("step") or {})
    for field, value in (incoming.get("step") or {}).items():
        existing_value = merged_step.get(field)
        if existing_value not in (None, "") and value not in (None, ""):
            if json_ready(existing_value) != json_ready(value):
                raise ValueError(f"Step metadata differs from the existing step manifest: {field}")
        elif existing_value in (None, "") and value not in (None, ""):
            merged_step[field] = json_ready(value)

    existing_experiment_id = existing.get("experiment_id")
    incoming_experiment_id = incoming.get("experiment_id")
    if (
        existing_experiment_id not in (None, "")
        and incoming_experiment_id not in (None, "")
        and str(existing_experiment_id) != str(incoming_experiment_id)
    ):
        raise ValueError("Step belongs to a different experiment.")

    plans = []
    for path in [*(existing.get("plans") or []), *(incoming.get("plans") or [])]:
        path = str(path)
        if path not in plans:
            plans.append(path)

    return {
        "step": merged_step,
        "experiment_id": existing_experiment_id or incoming_experiment_id or "",
        "recipe_path": existing.get("recipe_path") or incoming.get("recipe_path") or "",
        "plans": plans,
    }


def read_step_manifest(
    root: str | Path,
    step_id: str,
    *,
    remote: str | None = None,
    allow_missing: bool = False,
) -> dict[str, Any] | None:
    path = Path(root) / "steps" / str(step_id) / "step.yaml"
    if not exp_io.path_exists_at(path, remote=remote):
        if allow_missing:
            return None
        raise FileNotFoundError(f"Managed step manifest does not exist: {path}")
    text = exp_io.read_text_at(path, remote=remote)
    payload = read_managed_yaml_mapping(text, source=f"Managed step manifest {path}")
    normalized = merge_step_manifest(payload, {})
    if normalized != payload:
        raise ValueError(f"Managed step manifest has an incomplete canonical envelope: {path}")
    step = payload["step"]
    for field in ("id", "phase", "purpose"):
        if not str(step.get(field) or "").strip():
            raise ValueError(f"Managed step manifest is missing step.{field}: {path}")
    if step["phase"] not in PHASES:
        raise ValueError(f"Managed step manifest has invalid step.phase: {path}")
    if str(step["id"]) != str(step_id):
        raise ValueError(f"Managed step manifest id differs from its directory: {path}")
    if not str(payload["experiment_id"] or "").strip():
        raise ValueError(f"Managed step manifest is missing experiment_id: {path}")
    recipe_path = payload["recipe_path"]
    if not isinstance(recipe_path, str) or (recipe_path and not Path(recipe_path).is_absolute()):
        raise ValueError(f"Managed step manifest recipe_path must be empty or absolute: {path}")
    if any(not Path(str(plan)).is_absolute() for plan in payload["plans"]):
        raise ValueError(f"Managed step manifest plan paths must be absolute: {path}")
    return payload


def initialize_run_manifest(root: str | Path, *, remote: str | None = None) -> Path:
    path = Path(root) / "run_manifest.tsv"
    if exp_io.path_exists_at(path, remote=remote):
        raise ValueError(f"Managed run manifest already exists: {path}")
    exp_io.write_text_at(path, "step_id\trun_id\n", remote=remote)
    return path


def read_run_manifest(root: str | Path, *, remote: str | None = None) -> list[dict[str, str]]:
    path = Path(root) / "run_manifest.tsv"
    # The canonical path itself is part of the ownership proof; aliases are not managed state.
    exp_io.validate_managed_output_paths(root, [path], remote=remote)
    if not exp_io.path_exists_at(path, remote=remote):
        raise FileNotFoundError(f"Managed run manifest is missing: {path}")
    text = exp_io.read_text_at(path, remote=remote)
    return _parse_run_manifest(text, path)


def _parse_run_manifest(text: str, path: Path) -> list[dict[str, str]]:
    if not text.strip():
        raise ValueError(f"Managed run manifest is empty: {path}")
    try:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t", strict=True)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Managed run manifest has no header: {path}")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"Managed run manifest has duplicate header fields: {path}")
        if "trial_id" in fieldnames:
            raise ValueError(f"Historical trial_id rows are read-only and unsupported by {path}.")
        legacy_parameters = sorted(field for field in fieldnames if field.startswith("param."))
        if legacy_parameters:
            raise ValueError(
                f"Historical parameter fields are read-only and unsupported by {path}: " + ", ".join(legacy_parameters)
            )
        missing_fields = [field for field in ("step_id", "run_id") if field not in fieldnames]
        if missing_fields:
            raise ValueError(f"Managed run manifest header is missing {', '.join(missing_fields)}: {path}")
        rows = list(reader)
    except csv.Error as exc:
        raise ValueError(f"Managed run manifest is malformed: {path}") from exc
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Managed run manifest has a non-rectangular row: {path}")
    if rows and "experiment_id" not in fieldnames:
        raise ValueError(f"Managed run manifest rows must define experiment_id: {path}")
    if any(
        not str(row.get("experiment_id") or "").strip()
        or str(row["experiment_id"]) != str(row["experiment_id"]).strip()
        for row in rows
    ):
        raise ValueError(f"Managed run manifest rows must define a non-blank experiment_id: {path}")
    validate_managed_run_rows(rows, source=str(path), cardinality="one_per_run")
    return rows


def validate_existing_experiment_manifest(existing_text: str, experiment: dict[str, Any], root: Path) -> dict[str, Any]:
    existing = read_managed_yaml_mapping(
        existing_text, source=f"Managed experiment manifest {root / 'experiment.yaml'}"
    )
    existing_experiment = existing.get("experiment") if isinstance(existing, dict) else None
    if not isinstance(existing_experiment, dict) or existing_experiment.get("id") != experiment.get("id"):
        raise ValueError(f"Experiment root belongs to a different experiment: {root}")
    for field in ("title", "objective", "root", "baseline"):
        if existing_experiment.get(field) != experiment.get(field):
            raise ValueError(f"experiment.{field} differs from the existing experiment manifest.")
    return existing


def write_initial_experiment_manifest(root: Path, experiment: dict[str, Any], *, remote: str | None = None) -> None:
    exp_io.write_text_at(
        root / "experiment.yaml", yaml.safe_dump({"experiment": experiment}, sort_keys=False), remote=remote
    )
    initialize_run_manifest(root, remote=remote)


def ensure_experiment_workspace(recipe: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    root = experiment_root(recipe)
    if root is None:
        raise ValueError("experiment.root is required.")
    output_issue = validate_plan_output(recipe, output_dir)
    if output_issue:
        raise ValueError(output_issue)
    recipe["experiment"]["root"] = str(root)
    experiment = _public_mapping(recipe.get("experiment") or {})
    step = _public_mapping(recipe.get("step") or {})
    manifest_path = root / "experiment.yaml"
    manifest_exists = manifest_path.exists()
    if manifest_exists:
        validate_existing_experiment_manifest(manifest_path.read_text(), experiment, root)
        for row in read_run_manifest(root):
            if row["experiment_id"] != experiment["id"]:
                raise ValueError("run_manifest.tsv contains a run owned by a different experiment.")
    plan_path = Path(output_dir).expanduser()
    if not plan_path.is_absolute():
        plan_path = (Path.cwd() / plan_path).resolve()
    else:
        plan_path = plan_path.resolve()
    existing_step_payload = read_step_manifest(root, str(step["id"]), allow_missing=True)
    step_payload = {
        "step": step,
        "experiment_id": experiment["id"],
        "recipe_path": recipe.get("_recipe_path", ""),
        "plans": [str(plan_path)],
    }
    merged_step_payload = merge_step_manifest(existing_step_payload or {}, step_payload)
    step_manifest = root / "steps" / str(step["id"]) / "step.yaml"
    exp_io.validate_managed_output_paths(
        root,
        [manifest_path, root / "run_manifest.tsv", root / "events.jsonl", root / "README.md", step_manifest],
    )

    root.mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    step_dir = root / "steps" / str(step["id"])
    step_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_exists:
        write_initial_experiment_manifest(root, experiment)
        append_event(root, "experiment_initialized", {"experiment_id": experiment["id"]})
    if merged_step_payload != existing_step_payload:
        step_manifest.write_text(yaml.safe_dump(merged_step_payload, sort_keys=False))
    if existing_step_payload is None:
        append_event(root, "step_registered", {"step_id": step["id"], "phase": step["phase"]})
    _write_readme(root, experiment)
    return root, step_dir


def append_event(root: str | Path, event_type: str, payload: dict[str, Any]) -> None:
    root = Path(root)
    path = root / "events.jsonl"
    exp_io.validate_managed_output_paths(root, [path])
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": _now(), "event_type": event_type, **json_ready(payload)}
    with path.open("a") as file_obj:
        file_obj.write(json.dumps(row, sort_keys=True) + "\n")


def run_identity(
    recipe: dict[str, Any], index: int, parameters: dict[str, Any], *, run_name: str | None = None
) -> dict[str, str]:
    run_id = f"run-{index:03d}"
    semantic_name = safe_artifact_name(run_name) if run_name is not None else semantic_run_name(parameters)
    experiment_id = str((recipe.get("experiment") or {}).get("id"))
    step_id = str((recipe.get("step") or {}).get("id"))
    version = _bounded_slug(f"{experiment_id}__{step_id}__{run_id}__{semantic_name}", 180)
    return {"run_id": run_id, "run_name": semantic_name, "version": version}


def managed_run_key(row: dict[str, Any]) -> tuple[str, str] | None:
    step_id = str(row.get("step_id") or "")
    run_id = str(row.get("run_id") or "")
    if not step_id.strip() or not run_id.strip():
        return None
    return step_id, run_id


def managed_run_parameters(row: dict[str, Any]) -> dict[str, Any]:
    legacy_fields = sorted(str(key) for key in row if str(key).startswith("param."))
    if legacy_fields:
        raise ValueError(f"Historical parameter fields are read-only and unsupported: {', '.join(legacy_fields)}")
    return {
        str(key): value
        for key, value in row.items()
        if str(key).startswith("runtime.") or str(key).startswith("yaml:/")
    }


def validate_managed_run_rows(rows: list[dict[str, Any]], *, source: str, cardinality: str) -> None:
    if cardinality not in {"one_per_run", "many_per_run"}:
        raise ValueError(f"Unsupported managed row cardinality for {source}: {cardinality}")
    seen = set()
    for index, row in enumerate(rows):
        if "trial_id" in row:
            raise ValueError(f"Historical trial_id rows are read-only and unsupported by {source}.")
        managed_run_parameters(row)
        key = managed_run_key(row)
        if key is None:
            raise ValueError(
                f"{source} row {index} must define step_id and run_id as non-blank managed identity fields."
            )
        if any(str(row[field]) != str(row[field]).strip() for field in ("step_id", "run_id")):
            raise ValueError(f"{source} row {index} has surrounding whitespace in its managed identity.")
        if cardinality == "one_per_run" and key in seen:
            raise ValueError(f"Duplicate managed run identity in {source}: {key[0]} / {key[1]}")
        relative_paths = [
            field
            for field in MANAGED_RUN_PATH_FIELDS
            if row.get(field) not in (None, "") and not Path(str(row[field])).is_absolute()
        ]
        if relative_paths:
            raise ValueError(f"{source} row {index} has non-absolute paths: {', '.join(sorted(relative_paths))}")
        seen.add(key)


def run_evidence_key(row: dict[str, Any]) -> tuple[str, ...] | None:
    managed_key = managed_run_key(row)
    if managed_key is not None:
        return ("managed", *managed_key)
    external_id = str(row.get("version") or "")
    return ("external", external_id) if external_id else None


def resolve_run_row(rows: list[dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any] | None:
    key = managed_run_key(evidence)
    if key is not None:
        matches = [row for row in rows if managed_run_key(row) == key]
        matched = matches[-1] if matches else None
        if matched is not None:
            evidence_experiment = str(evidence.get("experiment_id") or "")
            managed_experiment = str(matched.get("experiment_id") or "")
            if evidence_experiment and managed_experiment and evidence_experiment != managed_experiment:
                return None
        return matched

    version = str(evidence.get("version") or "")
    if version:
        matches = [row for row in rows if str(row.get("version") or "") == version]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous runtime version matches multiple managed runs: {version}")
        if matches:
            matched = matches[0]
            evidence_experiment = str(evidence.get("experiment_id") or "")
            managed_experiment = str(matched.get("experiment_id") or "")
            if evidence_experiment and managed_experiment and evidence_experiment != managed_experiment:
                return None
            return matched

    return None


def resolve_external_run_row(rows: list[dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any] | None:
    if evidence.get("experiment_id") in (None, ""):
        return resolve_run_row(rows, {"version": evidence.get("version")})
    matched = resolve_run_row(rows, evidence)
    if matched is None or str(matched.get("experiment_id") or "") != str(evidence["experiment_id"]):
        return None
    return matched


def next_run_index(recipe: dict[str, Any]) -> int:
    root = experiment_root(recipe)
    if root is None:
        return 0
    if not (root / "experiment.yaml").exists():
        return 0
    step_id = str((recipe.get("step") or {}).get("id") or "")
    indices = []
    rows = read_run_manifest(root)
    for row in rows:
        if str(row.get("step_id") or "") != step_id:
            continue
        match = re.fullmatch(r"run-(\d+)", str(row.get("run_id") or ""))
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


def semantic_run_name(parameters: dict[str, Any]) -> str:
    if not parameters:
        return "default"
    pieces = []
    used = set()
    for key, value in parameters.items():
        field = _parameter_field(key)
        if field in used:
            field = _bounded_slug(str(key), 32)
        used.add(field)
        pieces.append(_setting_slug(field, value))
    return _bounded_slug("__".join(pieces), 100)


def safe_artifact_name(value: Any) -> str:
    return _bounded_slug(str(value), 100) or "default"


def parameter_summary(parameters: dict[str, Any]) -> str:
    return "; ".join(f"{key}={_display_value(value)}" for key, value in parameters.items())


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_run_snapshot(run: dict[str, Any]) -> None:
    for path_field, hash_field in (("config", "config_sha256"), ("script", "script_sha256")):
        path = run.get(path_field)
        expected = run.get(hash_field)
        if path and expected and file_sha256(path) != expected:
            raise ValueError(f"Run snapshot hash changed after planning: {path}")


def merge_run_row(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    existing_status = merged.get("status")
    incoming_status = incoming.get("status")
    merged.update(json_ready(incoming))
    if existing_status in TERMINAL_STATUSES:
        if incoming_status == "failed" and existing_status in {"completed", "finished"}:
            merged["status"] = "failed"
        else:
            merged["status"] = existing_status
    elif incoming_status == "superseded" and existing_status not in {"planned", "pending"}:
        merged["status"] = existing_status
    elif existing_status in {"launched", "running", "unknown_remote", "missing_pid"} and incoming_status in {
        "planned",
        "pending",
    }:
        merged["status"] = existing_status
    return merged


def validate_frozen_run_update(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    *,
    require_checkpoint_ownership: bool = False,
    allow_execution_identity_fill: bool = False,
) -> None:
    key = managed_run_key(existing) or managed_run_key(incoming)
    managed_run_parameters(existing)
    incoming_parameters = managed_run_parameters(incoming)
    execution_identity_initialized = existing.get("target") not in (None, "")
    for field, incoming_value in incoming.items():
        if field not in FROZEN_RUN_FIELDS and field not in incoming_parameters:
            continue
        existing_value = existing.get(field)
        if field in EXECUTION_IDENTITY_FIELDS:
            if not execution_identity_initialized:
                if allow_execution_identity_fill:
                    continue
                step_id, run_id = key or ("", "")
                raise ValueError(f"Canonical execution identity is missing for {step_id} / {run_id}: {field}")
            changed = str(json_ready(incoming_value)) != str(json_ready(existing_value))
        elif existing_value in (None, ""):
            continue
        else:
            changed = incoming_value in (None, "") or str(json_ready(incoming_value)) != str(json_ready(existing_value))
        if changed:
            step_id, run_id = key or ("", "")
            raise ValueError(f"Frozen run field differs for {step_id} / {run_id}: {field}")
    checkpoint_path = incoming.get("checkpoint_path")
    if require_checkpoint_ownership and checkpoint_path not in (None, ""):
        checkpoint_dir = existing.get("checkpoint_dir")
        candidate = Path(str(checkpoint_path))
        frozen_dir = Path(str(checkpoint_dir)) if checkpoint_dir not in (None, "") else None
        if frozen_dir is None or candidate.parent != frozen_dir:
            step_id, run_id = key or ("", "")
            raise ValueError(f"checkpoint_path is outside the frozen checkpoint_dir for {step_id} / {run_id}.")
        # Lexical containment cannot prove ownership when an existing checkpoint entry is an alias.
        if frozen_dir.is_symlink() or candidate.is_symlink() or (candidate.exists() and not candidate.is_file()):
            step_id, run_id = key or ("", "")
            raise ValueError(f"checkpoint_path is not a regular managed checkpoint for {step_id} / {run_id}.")


def merge_run_manifest(
    root: str | Path, rows: list[dict[str, Any]], *, remote: str | None = None, lock_held: bool = False
) -> list[dict[str, Any]]:
    root = Path(root)
    path = root / "run_manifest.tsv"
    lock_path = path.with_name(path.name + ".lock")
    exp_io.validate_managed_output_paths(
        root,
        [
            path,
            lock_path,
            root / "run_matrix.csv",
            root / "reports" / "run_matrix.md",
            root / "events.jsonl",
        ],
        remote=remote,
    )
    validate_managed_run_rows(rows, source="incoming run manifest", cardinality="one_per_run")
    lock_file = None
    if not remote and not lock_held:
        lock_file = lock_path.open("a+")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    try:
        for _attempt in range(3 if remote else 1):
            if not exp_io.path_exists_at(path, remote=remote):
                raise FileNotFoundError(f"Managed run manifest is missing: {path}")
            current_text = exp_io.read_text_at(path, remote=remote)
            existing = _parse_run_manifest(current_text, path)
            by_id = {managed_run_key(row): dict(row) for row in existing}
            order = [managed_run_key(row) for row in existing]
            new_rows = [row for row in rows if managed_run_key(row) not in by_id]
            if new_rows:
                for row in new_rows:
                    experiment_id = str(row.get("experiment_id") or "")
                    if not experiment_id.strip() or experiment_id != experiment_id.strip():
                        key = managed_run_key(row)
                        raise ValueError(f"New canonical run must define experiment_id: {key[0]} / {key[1]}")
                experiment_path = root / "experiment.yaml"
                experiment_text = exp_io.read_text_at(experiment_path, remote=remote)
                if not experiment_text:
                    raise ValueError(f"Managed experiment manifest is missing: {experiment_path}")
                manifest = read_managed_yaml_mapping(
                    experiment_text, source=f"Managed experiment manifest {experiment_path}"
                )
                experiment = manifest.get("experiment") if isinstance(manifest, dict) else None
                workspace_experiment_id = str(experiment.get("id") or "") if isinstance(experiment, dict) else ""
                if not workspace_experiment_id:
                    raise ValueError(f"Managed experiment manifest is missing experiment.id: {experiment_path}")
                if any(str(row["experiment_id"]) != workspace_experiment_id for row in new_rows):
                    raise ValueError("New canonical run belongs to a different experiment.")
            for row in rows:
                key = managed_run_key(row)
                if key not in by_id:
                    order.append(key)
                else:
                    validate_frozen_run_update(by_id[key], row, allow_execution_identity_fill=True)
                by_id[key] = merge_run_row(by_id.get(key, {}), row)
            committed = [by_id[key] for key in order if key in by_id]
            if committed:
                buffer = io.StringIO()
                fieldnames = sorted({key for row in committed for key in row})
                writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(committed)
                replacement = buffer.getvalue()
            else:
                replacement = "step_id\trun_id\n"
            expected_sha256 = hashlib.sha256(current_text.encode()).hexdigest()
            if exp_io.conditional_atomic_replace_text_at(
                path,
                replacement,
                expected_sha256,
                remote=remote,
            ):
                break
        else:
            raise RuntimeError(f"Canonical run manifest changed during three commit attempts: {path}")
        write_run_matrix(root, committed, remote=remote)
    finally:
        if lock_file is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
    return committed


def write_run_matrix(root: str | Path, rows: list[dict[str, Any]], *, remote: str | None = None) -> Path:
    root = Path(root)
    validate_managed_run_rows(rows, source="run_manifest.tsv", cardinality="one_per_run")
    matrix_path = root / "run_matrix.csv"
    if rows:
        exp_io.write_rows_at(matrix_path, rows, remote=remote)
    else:
        exp_io.write_text_at(matrix_path, "step_id,run_id\n", remote=remote)
    path = root / "reports" / "run_matrix.md"
    lines = ["# Run Matrix", ""]
    if not rows:
        lines.append("No runs registered.")
    else:
        lines.extend(
            [
                "| run | setting | status | metric | score | checkpoint | W&B |",
                "|---|---|---|---|---:|---|---|",
            ]
        )
        for row in rows:
            label = f"{row.get('step_id', '')} / {row.get('run_id', '')} — {row.get('run_name', '')}"
            lines.append(
                "| {label} | {setting} | {status} | {metric} | {score} | `{checkpoint}` | {wandb} |".format(
                    label=label.replace("|", "/"),
                    setting=str(row.get("parameter_summary", "")).replace("|", "/"),
                    status=row.get("status", ""),
                    metric=row.get("metric", ""),
                    score=row.get("score", ""),
                    checkpoint=row.get("checkpoint_path", ""),
                    wandb=row.get("wandb_url", ""),
                )
            )
    exp_io.write_text_at(path, "\n".join(lines) + "\n", remote=remote)
    return matrix_path


def write_status_report(root: str | Path) -> Path:
    root = Path(root)
    rows = read_run_manifest(root)
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "unknown") or "unknown"
        counts[status] = counts.get(status, 0) + 1
    path = root / "reports" / "status.md"
    lines = ["# Experiment Status", "", f"Runs: {len(rows)}", ""]
    lines.extend(f"- {status}: {count}" for status, count in sorted(counts.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def experiment_readme_text(experiment: dict[str, Any]) -> str:
    baseline = experiment.get("baseline")
    baseline_text = json.dumps(json_ready(baseline), ensure_ascii=False, sort_keys=True)
    lines = [
        f"# {experiment['title']}",
        "",
        f"Experiment id: `{experiment['id']}`",
        "",
        "## Objective",
        "",
        str(experiment["objective"]),
        "",
        "## Baseline",
        "",
        baseline_text,
        "",
        "## Navigation",
        "",
        "- `steps/`: preparation, training, evaluation, and analysis steps",
        "- `run_manifest.tsv`: current run state and settings",
        "- `events.jsonl`: append-only experiment history",
        "- `reports/`: human-readable status, run matrix, ranking, and final report",
    ]
    return "\n".join(lines) + "\n"


def _write_readme(root: Path, experiment: dict[str, Any]) -> None:
    (root / "README.md").write_text(experiment_readme_text(experiment))


def _parameter_field(key: str) -> str:
    text = str(key)
    if text.startswith("yaml:"):
        text = text.rsplit("/", 1)[-1]
    elif "." in text:
        text = text.rsplit(".", 1)[-1]
    return _slug(text.replace("_", "-")) or "param"


def _value_slug(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return _slug(f"{value:g}".replace("e-0", "e-").replace("e+0", "e+"))
    if isinstance(value, (list, tuple)):
        return _bounded_slug("-".join(_value_slug(item) for item in value), 48)
    if isinstance(value, dict):
        text = json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))
        return "map-" + hashlib.sha256(text.encode()).hexdigest()[:8]
    return _slug(str(value)) or "empty"


def _setting_slug(field: str, value: Any) -> str:
    if not isinstance(value, bool):
        return f"{field}-{_value_slug(value)}"
    if field.endswith("-frozen"):
        return field if value else f"{field.removesuffix('-frozen')}-trainable"
    if field.endswith("-freeze"):
        stem = field.removesuffix("-freeze")
        return f"{stem}-frozen" if value else f"{stem}-trainable"
    return f"{field}-{'on' if value else 'off'}"


def _display_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))
    return str(value)


def _bounded_slug(value: str, limit: int) -> str:
    clean = "__".join(_slug(part) for part in value.split("__"))
    if len(clean) <= limit:
        return clean
    digest = hashlib.sha256(clean.encode()).hexdigest()[:8]
    return f"{clean[: limit - 11].rstrip('-_')}--h{digest}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.-]+", "-", value).strip("-.").lower()


def _public_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {key: json_ready(item) for key, item in value.items() if not str(key).startswith("_")}


def _is_nonempty_unmanaged_root(root: Path) -> bool:
    return root.exists() and any(root.iterdir()) and not (root / "experiment.yaml").exists()


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
