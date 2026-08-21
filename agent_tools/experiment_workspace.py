from __future__ import annotations

from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any

import yaml

from . import experiment_io as exp_io, transport
from .models import REPO_ROOT, json_ready

PHASES = {"prepare", "train", "evaluate", "analyze"}
TERMINAL_STATUSES = {"completed", "failed", "finished", "launch_failed", "stopped", "superseded"}
MONITOR_EXIT_CODE_PREFIX = "AGENT_TOOLS_EXIT_CODE="
PROCESS_IDENTITY_FIELDS = {"pid", "process_group_id", "process_start_token"}
SCHEDULER_PLAN_IDENTITY_FIELDS = {
    "scheduler_type",
    "scheduler_submit_token",
    "scheduler_script",
    "scheduler_script_sha256",
    "scheduler_result_path",
    "allocation_identity_path",
}
SCHEDULER_BINDING_FIELDS = {"scheduler_job_id", "scheduler_cluster", "execution_snapshot_sha256"}
SCHEDULER_IDENTITY_FIELDS = SCHEDULER_PLAN_IDENTITY_FIELDS | SCHEDULER_BINDING_FIELDS
EXECUTION_IDENTITY_FIELDS = {
    "target",
    "host",
    "workdir",
    "gpus",
    "pid_path",
    "log_path",
    "command",
} | PROCESS_IDENTITY_FIELDS
FROZEN_RUN_FIELDS = (
    {
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
        "pipeline_id",
        "job_id",
        "attempt",
        "result_root",
        "terminal_status_owner",
    }
    | EXECUTION_IDENTITY_FIELDS
    | SCHEDULER_IDENTITY_FIELDS
)
MANAGED_RUN_PATH_FIELDS = {
    "artifacts",
    "checkpoint_dir",
    "checkpoint_path",
    "config",
    "log_path",
    "pid_path",
    "progress_dir",
    "round_dir",
    "result_root",
    "run_dir",
    "run_manifest",
    "runtime_dir",
    "script",
    "test_logits_path",
    "test_predictions_path",
    "val_logits_path",
    "val_predictions_path",
    "workdir",
    "scheduler_script",
    "scheduler_result_path",
    "allocation_identity_path",
}
RESEARCH_LOG_NAME = "RESEARCH_LOG.md"
RESEARCH_LOG_KINDS = {"action", "observation", "interpretation", "decision", "conclusion"}
RESEARCH_LOG_AUTHORITIES = {"human", "policy", "canonical_decision"}
RESEARCH_LOG_ENTRY_FIELDS = {
    "id",
    "recorded_at",
    "kind",
    "title",
    "actor",
    "source",
    "evidence",
    "body",
    "occurred_at",
    "authority",
    "scope",
    "supersedes",
}
RESEARCH_LOG_PREAMBLE = """# Research Log

This append-only log records meaningful research actions, observations, interpretations, decisions, and conclusions.
`run_manifest.tsv` remains the sole authority for execution lifecycle state.

"""
RESEARCH_LOG_MARKER_RE = re.compile(
    r'^<!-- agent-tools-research-entry id="([a-z0-9][a-z0-9._-]{0,127})" sha256="([0-9a-f]{64})" -->\n',
    re.MULTILINE,
)
RESEARCH_LOG_ENTRY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def _validated_step_manifest(text: str, path: Path, step_id: str) -> dict[str, Any]:
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
    return _validated_step_manifest(exp_io.read_text_at(path, remote=remote), path, str(step_id))


def commit_step_manifest(
    root: str | Path,
    incoming: dict[str, Any],
    *,
    remote: str | None = None,
) -> tuple[dict[str, Any], bool]:
    root = Path(root)
    step = incoming.get("step") if isinstance(incoming.get("step"), dict) else {}
    step_id = str(step.get("id") or "")
    if not step_id:
        raise ValueError("Incoming step manifest is missing step.id.")
    path = root / "steps" / step_id / "step.yaml"
    exp_io.validate_managed_output_paths(root, [path], remote=remote)
    for _attempt in range(3):
        exists = exp_io.path_exists_at(path, remote=remote)
        current_text = exp_io.read_text_at(path, remote=remote) if exists else ""
        existing = _validated_step_manifest(current_text, path, step_id) if exists else {}
        merged = merge_step_manifest(existing, incoming)
        if merged == existing:
            return merged, False
        replacement = yaml.safe_dump(merged, sort_keys=False)
        _validated_step_manifest(replacement, path, step_id)
        expected_sha256 = hashlib.sha256(current_text.encode()).hexdigest() if exists else None
        if exp_io.conditional_atomic_replace_text_at(
            path,
            replacement,
            expected_sha256,
            remote=remote,
        ):
            return merged, not exists
        # A competing planner won the compare-and-swap; merge its plan on the next attempt.
    raise RuntimeError(f"Managed step manifest changed during three commit attempts: {path}")


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


def _research_log_single_line(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Research log entry {field} must be a non-empty string.")
    if value != value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"Research log entry {field} must be a trimmed single-line string.")
    return value


def _research_log_timestamp(value: Any, field: str) -> str:
    text = _research_log_single_line(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Research log entry {field} must be an ISO timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"Research log entry {field} must be in UTC.")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalized_research_log_entry(
    entry: dict[str, Any],
    *,
    experiment_id: str,
    managed_rows: list[dict[str, Any]],
    root: Path,
    remote: str | None,
) -> dict[str, Any]:
    unexpected = sorted(set(entry) - RESEARCH_LOG_ENTRY_FIELDS)
    if unexpected:
        raise ValueError(f"Unexpected research log entry fields: {', '.join(unexpected)}")
    missing = sorted(
        field
        for field in ("id", "recorded_at", "kind", "title", "actor", "source", "evidence", "body")
        if field not in entry
    )
    if missing:
        raise ValueError(f"Research log entry is missing required fields: {', '.join(missing)}")

    entry_id = _research_log_single_line(entry["id"], "id")
    if not RESEARCH_LOG_ENTRY_ID_RE.fullmatch(entry_id):
        raise ValueError("Research log entry id must match [a-z0-9][a-z0-9._-]{0,127}.")
    kind = _research_log_single_line(entry["kind"], "kind")
    if kind not in RESEARCH_LOG_KINDS:
        raise ValueError(f"Research log entry kind must be one of: {', '.join(sorted(RESEARCH_LOG_KINDS))}.")

    authority = entry.get("authority")
    if authority is not None:
        authority = _research_log_single_line(authority, "authority")
        if authority not in RESEARCH_LOG_AUTHORITIES:
            raise ValueError(
                f"Research log entry authority must be one of: {', '.join(sorted(RESEARCH_LOG_AUTHORITIES))}."
            )
    if kind == "decision" and authority is None:
        raise ValueError("Research log decision entries require authority.")

    body = entry["body"]
    if not isinstance(body, str) or not body.strip():
        raise ValueError("Research log entry body must be a non-empty string.")
    body = body.replace("\r\n", "\n").replace("\r", "\n").strip()

    evidence = entry["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Research log entry evidence must be a non-empty list.")
    normalized_evidence = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise ValueError(f"Research log entry evidence[{index}] must be a mapping.")
        unexpected_evidence = sorted(set(item) - {"label", "locator", "sha256"})
        if unexpected_evidence:
            raise ValueError(
                f"Unexpected research log entry evidence[{index}] fields: {', '.join(unexpected_evidence)}"
            )
        missing_evidence = sorted(field for field in ("label", "locator") if field not in item)
        if missing_evidence:
            raise ValueError(
                f"Research log entry evidence[{index}] is missing required fields: {', '.join(missing_evidence)}"
            )
        normalized_item = {
            "label": _research_log_single_line(item["label"], f"evidence[{index}].label"),
            "locator": _research_log_single_line(item["locator"], f"evidence[{index}].locator"),
        }
        if "sha256" in item:
            digest = _research_log_single_line(item["sha256"], f"evidence[{index}].sha256")
            if not SHA256_RE.fullmatch(digest):
                raise ValueError(f"Research log entry evidence[{index}].sha256 must be a lowercase SHA-256.")
            normalized_item["sha256"] = digest
        normalized_evidence.append(normalized_item)

    scope = entry.get("scope")
    normalized_scope: dict[str, Any] = {}
    if "scope" in entry:
        if not isinstance(scope, dict) or not scope:
            raise ValueError("Research log entry scope must be a non-empty mapping.")
        unexpected_scope = sorted(set(scope) - {"step_id", "run_ids"})
        if unexpected_scope:
            raise ValueError(f"Unexpected research log entry scope fields: {', '.join(unexpected_scope)}")
        if "step_id" not in scope:
            raise ValueError("Research log entry scope.run_ids requires scope.step_id.")
        step_id = _research_log_single_line(scope["step_id"], "scope.step_id")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", step_id):
            raise ValueError(
                "Research log entry scope.step_id must use lowercase letters, digits, hyphens, and underscores."
            )
        step_manifest = read_step_manifest(root, step_id, remote=remote)
        if step_manifest["experiment_id"] != experiment_id:
            raise ValueError("Research log entry scope.step_id belongs to a different experiment.")
        normalized_scope["step_id"] = step_id
        if "run_ids" in scope:
            run_ids = scope["run_ids"]
            if not isinstance(run_ids, list) or not run_ids:
                raise ValueError("Research log entry scope.run_ids must be a non-empty list.")
            normalized_run_ids = [
                _research_log_single_line(run_id, f"scope.run_ids[{index}]") for index, run_id in enumerate(run_ids)
            ]
            if len(normalized_run_ids) != len(set(normalized_run_ids)):
                raise ValueError("Research log entry scope.run_ids must not contain duplicates.")
            managed_keys = {managed_run_key(row) for row in managed_rows}
            missing_runs = [run_id for run_id in normalized_run_ids if (step_id, run_id) not in managed_keys]
            if missing_runs:
                raise ValueError(f"Research log entry scope references unknown managed runs: {', '.join(missing_runs)}")
            normalized_scope["run_ids"] = normalized_run_ids

    supersedes = entry.get("supersedes")
    normalized_supersedes = []
    if supersedes is not None:
        if not isinstance(supersedes, list):
            raise ValueError("Research log entry supersedes must be a list.")
        normalized_supersedes = [
            _research_log_single_line(superseded_id, f"supersedes[{index}]")
            for index, superseded_id in enumerate(supersedes)
        ]
        if len(normalized_supersedes) != len(set(normalized_supersedes)):
            raise ValueError("Research log entry supersedes must not contain duplicates.")
        if any(not RESEARCH_LOG_ENTRY_ID_RE.fullmatch(superseded_id) for superseded_id in normalized_supersedes):
            raise ValueError("Research log entry supersedes values must be valid research log entry ids.")
        if entry_id in normalized_supersedes:
            raise ValueError("Research log entry cannot supersede itself.")

    normalized = {
        "id": entry_id,
        "recorded_at": _research_log_timestamp(entry["recorded_at"], "recorded_at"),
        "kind": kind,
        "title": _research_log_single_line(entry["title"], "title"),
        "actor": _research_log_single_line(entry["actor"], "actor"),
        "source": _research_log_single_line(entry["source"], "source"),
        "evidence": normalized_evidence,
        "body": body,
    }
    if "occurred_at" in entry:
        normalized["occurred_at"] = _research_log_timestamp(entry["occurred_at"], "occurred_at")
    if authority is not None:
        normalized["authority"] = authority
    if normalized_scope:
        normalized["scope"] = normalized_scope
    if normalized_supersedes:
        normalized["supersedes"] = normalized_supersedes
    return normalized


def _research_log_blocks(text: str, path: Path) -> dict[str, str]:
    markers = list(RESEARCH_LOG_MARKER_RE.finditer(text))
    if not markers:
        if text != RESEARCH_LOG_PREAMBLE:
            raise ValueError(f"Managed research log has an invalid preamble or unmarked content: {path}")
        return {}
    if text[: markers[0].start()] != RESEARCH_LOG_PREAMBLE:
        raise ValueError(f"Managed research log has an invalid preamble: {path}")

    entries = {}
    for index, marker in enumerate(markers):
        entry_id, expected_digest = marker.groups()
        if entry_id in entries:
            raise ValueError(f"Managed research log has a duplicate entry id: {entry_id}")
        block_end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        block = text[marker.end() : block_end]
        actual_digest = hashlib.sha256(block.encode()).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(f"Managed research log entry digest differs from its content: {entry_id}")
        if not block.startswith("## ") or f"- Entry ID: `{entry_id}`\n" not in block:
            raise ValueError(f"Managed research log entry has invalid Markdown structure: {entry_id}")
        entries[entry_id] = actual_digest
    return entries


def _research_log_block(entry: dict[str, Any], experiment_id: str) -> str:
    lines = [
        f"## {entry['title']}",
        "",
        f"- Entry ID: `{entry['id']}`",
        f"- Kind: `{entry['kind']}`",
        f"- Recorded at: `{entry['recorded_at']}`",
    ]
    if "occurred_at" in entry:
        lines.append(f"- Occurred at: `{entry['occurred_at']}`")
    lines.extend(
        [
            f"- Actor: {entry['actor']}",
            f"- Source: {entry['source']}",
            f"- Experiment: `{experiment_id}`",
        ]
    )
    scope = entry.get("scope") or {}
    if "step_id" in scope:
        lines.append(f"- Step: `{scope['step_id']}`")
    if "run_ids" in scope:
        lines.append("- Runs:")
        lines.extend(f"  - `{run_id}`" for run_id in scope["run_ids"])
    if "authority" in entry:
        lines.append(f"- Authority: `{entry['authority']}`")
    if "supersedes" in entry:
        lines.append("- Supersedes: " + ", ".join(f"`{entry_id}`" for entry_id in entry["supersedes"]))
    lines.extend(["", "### Evidence", ""])
    for item in entry["evidence"]:
        lines.append(f"- Label: {item['label']}")
        lines.append(f"  Locator: {item['locator']}")
        if "sha256" in item:
            lines.append(f"  - SHA-256: `{item['sha256']}`")
    lines.extend(["", "### Record", "", entry["body"], ""])
    return "\n".join(lines) + "\n"


def append_research_log(
    root: str | Path,
    entry: dict[str, Any],
    *,
    experiment_id: str,
    managed_rows: list[dict[str, Any]],
    remote: str | None = None,
) -> tuple[Path, str, bool]:
    root = Path(root)
    path = root / RESEARCH_LOG_NAME
    lock_path = Path(f"{path}.lock") if remote else path.with_name(f".{path.name}.cas.lock")
    managed_paths = [path, lock_path]
    scope = entry.get("scope") if isinstance(entry, dict) else None
    if isinstance(scope, dict) and isinstance(scope.get("step_id"), str):
        managed_paths.append(root / "steps" / scope["step_id"] / "step.yaml")
    exp_io.validate_managed_output_paths(root, managed_paths, remote=remote)
    normalized = _normalized_research_log_entry(
        entry,
        experiment_id=experiment_id,
        managed_rows=managed_rows,
        root=root,
        remote=remote,
    )
    block = _research_log_block(normalized, experiment_id)
    digest = hashlib.sha256(block.encode()).hexdigest()
    marker = f'<!-- agent-tools-research-entry id="{normalized["id"]}" sha256="{digest}" -->\n'

    for _attempt in range(3):
        exists = exp_io.path_exists_at(path, remote=remote)
        current = exp_io.read_text_at(path, remote=remote) if exists else RESEARCH_LOG_PREAMBLE
        entries = _research_log_blocks(current, path)
        existing_digest = entries.get(normalized["id"])
        if existing_digest is not None:
            if existing_digest == digest:
                return path, normalized["id"], False
            raise ValueError(f"Research log entry id already exists with different content: {normalized['id']}")
        missing_superseded = [
            superseded_id for superseded_id in normalized.get("supersedes", []) if superseded_id not in entries
        ]
        if missing_superseded:
            raise ValueError(f"Research log entry supersedes unknown entry ids: {', '.join(missing_superseded)}")
        replacement = current + marker + block
        _research_log_blocks(replacement, path)
        expected_sha256 = hashlib.sha256(current.encode()).hexdigest() if exists else None
        if exp_io.conditional_atomic_replace_text_at(path, replacement, expected_sha256, remote=remote):
            return path, normalized["id"], True
    raise RuntimeError(f"Managed research log changed during three append attempts: {path}")


def write_initial_experiment_manifest(root: Path, experiment: dict[str, Any], *, remote: str | None = None) -> None:
    exp_io.write_text_at(
        root / "experiment.yaml", yaml.safe_dump({"experiment": experiment}, sort_keys=False), remote=remote
    )
    initialize_run_manifest(root, remote=remote)
    exp_io.write_text_at(root / RESEARCH_LOG_NAME, RESEARCH_LOG_PREAMBLE, remote=remote)


def ensure_experiment_workspace(
    recipe: dict[str, Any], output_dir: str | Path, *, register_step: bool = True
) -> tuple[Path, Path]:
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
    step_payload = {
        "step": step,
        "experiment_id": experiment["id"],
        "recipe_path": recipe.get("_recipe_path", ""),
        "plans": [str(plan_path)],
    }
    step_manifest = root / "steps" / str(step["id"]) / "step.yaml"
    exp_io.validate_managed_output_paths(
        root,
        [
            manifest_path,
            root / "run_manifest.tsv",
            root / RESEARCH_LOG_NAME,
            root / "events.jsonl",
            root / "README.md",
            step_manifest,
        ],
    )
    existing_step = read_step_manifest(root, step["id"], allow_missing=True)
    if existing_step is not None:
        merge_step_manifest(existing_step, step_payload)

    root.mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    step_dir = root / "steps" / str(step["id"])
    step_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_exists:
        write_initial_experiment_manifest(root, experiment)
        append_event(root, "experiment_initialized", {"experiment_id": experiment["id"]})
    if register_step:
        _merged_step_payload, created_step = commit_step_manifest(root, step_payload)
        if created_step:
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
    for path_field, hash_field in (
        ("config", "config_sha256"),
        ("script", "script_sha256"),
        ("scheduler_script", "scheduler_script_sha256"),
    ):
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
    elif existing_status in {
        "submitting",
        "queued",
        "launched",
        "running",
        "stopping",
        "unknown_remote",
        "unknown_scheduler",
        "missing_pid",
    } and incoming_status in {
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
        if field == "scheduler_type":
            changed = scheduler_type(existing) != str(incoming_value or "direct")
        elif field in SCHEDULER_BINDING_FIELDS:
            if existing_value in (None, "") and allow_execution_identity_fill:
                continue
            changed = str(json_ready(incoming_value)) != str(json_ready(existing_value))
        elif field in PROCESS_IDENTITY_FIELDS:
            if existing_value in (None, "") and allow_execution_identity_fill:
                continue
            changed = str(json_ready(incoming_value)) != str(json_ready(existing_value))
        elif field in EXECUTION_IDENTITY_FIELDS:
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
    if require_checkpoint_ownership:
        validate_checkpoint_ownership(existing, incoming)
        checkpoint_path = incoming.get("checkpoint_path")
        if checkpoint_path not in (None, ""):
            frozen_dir = Path(str(existing.get("checkpoint_dir")))
            candidate = Path(str(checkpoint_path))
            if frozen_dir.is_symlink() or candidate.is_symlink() or (candidate.exists() and not candidate.is_file()):
                step_id, run_id = key or ("", "")
                raise ValueError(f"checkpoint_path is not a regular managed checkpoint for {step_id} / {run_id}.")


def scheduler_type(row: dict[str, Any]) -> str:
    value = str(row.get("scheduler_type") or "direct")
    if value not in {"direct", "slurm"}:
        raise ValueError(f"scheduler_type must be direct or slurm: {value!r}")
    return value


def validate_scheduler_run_identity(row: dict[str, Any]) -> None:
    backend = scheduler_type(row)
    populated_process = {field for field in PROCESS_IDENTITY_FIELDS if row.get(field) not in (None, "")}
    populated_scheduler = {
        field for field in SCHEDULER_IDENTITY_FIELDS - {"scheduler_type"} if row.get(field) not in (None, "")
    }
    if backend == "direct":
        if populated_scheduler:
            raise ValueError("Direct managed run cannot define Slurm scheduler identity.")
        return
    if populated_process:
        raise ValueError("Slurm managed run cannot define PID process identity.")
    required_plan_fields = SCHEDULER_PLAN_IDENTITY_FIELDS - {"scheduler_type"}
    missing_plan_fields = sorted(field for field in required_plan_fields if row.get(field) in (None, ""))
    if missing_plan_fields:
        raise ValueError(f"Slurm managed run is missing scheduler plan identity: {', '.join(missing_plan_fields)}")
    job_id = str(row.get("scheduler_job_id") or "")
    if job_id and re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise ValueError(f"Slurm scheduler_job_id must be a positive integer: {job_id!r}")
    if row.get("scheduler_cluster") not in (None, "") and not job_id:
        raise ValueError("Slurm scheduler_cluster requires scheduler_job_id.")
    execution_snapshot_sha256 = str(row.get("execution_snapshot_sha256") or "")
    if execution_snapshot_sha256 and re.fullmatch(r"[0-9a-f]{64}", execution_snapshot_sha256) is None:
        raise ValueError("Slurm execution_snapshot_sha256 must be 64 lowercase hexadecimal characters.")
    status = str(row.get("status") or "planned")
    if status in {"planned", "pending", "submitting", "launch_failed", "superseded"} and job_id:
        raise ValueError(f"Slurm managed run status {status} cannot define scheduler_job_id.")
    if (
        status
        in {
            "queued",
            "running",
            "stopping",
            "unknown_scheduler",
            "completed",
            "finished",
            "failed",
            "stopped",
        }
        and not job_id
    ):
        raise ValueError(f"Slurm managed run status {status} requires scheduler_job_id.")


def validate_checkpoint_ownership(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> None:
    checkpoint_path = incoming.get("checkpoint_path")
    if checkpoint_path in (None, ""):
        return
    key = managed_run_key(existing) or managed_run_key(incoming)
    checkpoint_dir = existing.get("checkpoint_dir")
    candidate = Path(str(checkpoint_path))
    frozen_dir = Path(str(checkpoint_dir)) if checkpoint_dir not in (None, "") else None
    if frozen_dir is None or candidate.parent != frozen_dir:
        step_id, run_id = key or ("", "")
        raise ValueError(f"checkpoint_path is outside the frozen checkpoint_dir for {step_id} / {run_id}.")


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
    lock_stack = ExitStack()
    if not remote and not lock_held:
        lock_stack.enter_context(exp_io.blocking_file_lock(lock_path))
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
                merged = merge_run_row(by_id.get(key, {}), row)
                validate_scheduler_run_identity(merged)
                by_id[key] = merged
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
        if remote:
            projection_rows = committed
            projection_manifest_text = replacement
            for _projection_attempt in range(3):
                if _write_remote_run_matrix_if_current(root, projection_rows, projection_manifest_text, remote):
                    committed = projection_rows
                    break
                # A concurrent canonical commit won the shared lock; project only its newly observed version.
                current_text = exp_io.read_text_at(path, remote=remote)
                projection_rows = _parse_run_manifest(current_text, path)
                projection_manifest_text = current_text
            else:
                raise RuntimeError(f"Canonical run manifest changed during three projection attempts: {path}")
        else:
            write_run_matrix(root, committed)
    finally:
        lock_stack.close()
    return committed


def _run_matrix_text(rows: list[dict[str, Any]]) -> tuple[str, str]:
    if rows:
        buffer = io.StringIO()
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        matrix_text = buffer.getvalue()
    else:
        matrix_text = "step_id,run_id\n"
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
    return matrix_text, "\n".join(lines) + "\n"


def _write_remote_run_matrix_if_current(
    root: Path,
    rows: list[dict[str, Any]],
    manifest_text: str,
    remote: str,
) -> bool:
    matrix_path = root / "run_matrix.csv"
    report_path = root / "reports" / "run_matrix.md"
    matrix_text, report_text = _run_matrix_text(rows)
    script = f"""
import fcntl
import hashlib
import json
import os
import sys
import tempfile

manifest_path, matrix_path, report_path, expected = sys.argv[1:]
payload = json.load(sys.stdin)
with open(manifest_path + ".lock", "a+") as lock_file:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    try:
        with open(manifest_path, "rb") as file_obj:
            current = file_obj.read()
    except FileNotFoundError:
        raise SystemExit({exp_io.REMOTE_CONFLICT_RETURN_CODE})
    if hashlib.sha256(current).hexdigest() != expected:
        raise SystemExit({exp_io.REMOTE_CONFLICT_RETURN_CODE})
    staged = []
    try:
        for path, text in ((matrix_path, payload["matrix"]), (report_path, payload["report"])):
            parent = os.path.dirname(path) or "."
            os.makedirs(parent, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", dir=parent)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file_obj:
                file_obj.write(text)
                os.fchmod(file_obj.fileno(), 0o644)
                file_obj.flush()
                os.fsync(file_obj.fileno())
            staged.append((temporary, path))
        for temporary, path in staged:
            os.replace(temporary, path)
    finally:
        for temporary, _path in staged:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
"""
    result = transport.run_ssh(
        remote,
        transport.remote_python_command(
            script,
            str(root / "run_manifest.tsv"),
            str(matrix_path),
            str(report_path),
            hashlib.sha256(manifest_text.encode()).hexdigest(),
        ),
        input=json.dumps({"matrix": matrix_text, "report": report_text}),
        text=True,
    )
    if result.returncode == exp_io.REMOTE_CONFLICT_RETURN_CODE:
        return False
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"Remote run-matrix projection failed on {remote}: {detail}")
    return True


def write_run_matrix(root: str | Path, rows: list[dict[str, Any]], *, remote: str | None = None) -> Path:
    root = Path(root)
    validate_managed_run_rows(rows, source="run_manifest.tsv", cardinality="one_per_run")
    matrix_path = root / "run_matrix.csv"
    matrix_text, report_text = _run_matrix_text(rows)
    exp_io.write_text_at(matrix_path, matrix_text, remote=remote)
    path = root / "reports" / "run_matrix.md"
    exp_io.write_text_at(path, report_text, remote=remote)
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
        "- `RESEARCH_LOG.md`: append-only research actions, evidence, decisions, and conclusions",
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
