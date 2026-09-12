"""Read and rank adaptive round evidence without updating lifecycle state.

Combines frozen plans, canonical run manifests, runtime/checkpoint evidence,
and already-synced training history. Adaptive orchestration retains monitoring,
proposal history validation, digest publication, and incumbent/event writes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from . import (
    adaptive_proposals,
    checkpoint_test_results,
    experiment_sources,
    plan_rendering,
    run_artifacts as artifacts,
    run_evidence as evidence,
)
from .experiment_workspace import managed_run_key, managed_run_parameters, read_run_manifest


def digest_rows(
    round_dir: Path,
    round_index: int,
    workspace: Path,
    objective: adaptive_proposals.ProposalObjective | dict[str, str],
) -> list[dict[str, Any]]:
    plan = artifacts.read_hparam_plan(round_dir)
    recipe_value = plan.get("recipe")
    recipe = recipe_value if isinstance(recipe_value, dict) else {}
    evaluation_value = recipe.get("evaluation_policy")
    evaluation = evaluation_value if isinstance(evaluation_value, dict) else {}
    selection_split = str(evaluation.get("selection_split") or "")
    plan_keys = {managed_run_key(run) for run in plan.get("runs", [])}
    status_rows = {
        managed_run_key(row): row for row in read_run_manifest(workspace) if managed_run_key(row) in plan_keys
    }
    rows = []
    for run in plan.get("runs", []):
        run_id = str(run["run_id"])
        version = str(run["version"])
        status = status_rows.get(managed_run_key(run), {})
        artifact_row = {**run, **status}
        observed_artifacts = evidence.runtime_artifacts(artifact_row)
        if observed_artifacts is None:
            # Canonical success participates atomically, regardless of where the objective is stored.
            if status.get("status") in {"completed", "finished"}:
                raise ValueError(
                    f"Completed adaptive run has unavailable runtime artifact evidence: {run['step_id']} / {run_id}"
                )
            manifest_path = str(status.get("run_manifest") or "")
            manifest: dict[str, Any] = {}
            checkpoint_names: list[str] = []
        else:
            manifest_path, manifest, checkpoint_names = observed_artifacts
        checkpoint_dir = str(artifact_row.get("checkpoint_dir") or "")
        checkpoint_path = (
            artifacts.fixed_checkpoint_path_from_names(manifest, checkpoint_dir, checkpoint_names)
            if evidence.is_remote_row(artifact_row)
            else artifacts.fixed_checkpoint_path(manifest, Path(checkpoint_dir))
        )
        row = {
            "round": round_index,
            "experiment_id": run["experiment_id"],
            "step_id": run["step_id"],
            "run_id": run_id,
            "run_name": run["run_name"],
            "version": version,
            "external_optimized": True,
            "config": run.get("config", ""),
            "checkpoint_path": checkpoint_path,
            "run_manifest": str(manifest_path or ""),
            "log_path": artifact_row.get("log_path", ""),
            "log_failed": evidence.log_has_failure(artifact_row.get("log_path"), artifact_row),
            "log_tail": evidence.log_tail(artifact_row.get("log_path"), artifact_row, lines=4),
        }
        row.update(managed_run_parameters(run))
        row.update(manifest_metrics(manifest))
        if status.get("status") in {"completed", "finished"}:
            monitor = manifest.get("monitor")
            if isinstance(monitor, str) and monitor:
                row[monitor] = artifacts.metric_value(manifest, monitor)
        checkpoint_test_objective = selection_split == "test" and objective["metric"].startswith("test_")
        if checkpoint_test_objective:
            # Checkpoint test evidence changes identity and is valid only after canonical successful completion.
            row["monitor_checkpoint_path"] = checkpoint_path
            row.pop(objective["metric"], None)
            row.pop("epoch", None)
            row["checkpoint_path"] = ""
            if status.get("status") in {"completed", "finished"}:
                checkpoint_evidence = _test_checkpoint_evidence(
                    manifest,
                    objective,
                    checkpoint_dir,
                    checkpoint_names,
                )
                # Completed test-selected runs participate atomically; partial evidence cannot steer later rounds.
                if checkpoint_evidence is None:
                    raise ValueError(
                        f"Completed test-selected adaptive run lacks complete checkpoint test evidence: "
                        f"{run['step_id']} / {run_id}"
                    )
                checkpoint_objective, checkpoint_results = checkpoint_evidence
                selected = next(
                    result
                    for result in checkpoint_results
                    if result["checkpoint_path"] == checkpoint_objective["checkpoint_path"]
                )
                row = {key: value for key, value in row.items() if not key.startswith("test_")}
                row.update(selected["metrics"])
                row["checkpoint_test_results"] = json.dumps(
                    checkpoint_results, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                row[objective["metric"]] = checkpoint_objective["score"]
                row["checkpoint_path"] = checkpoint_objective["checkpoint_path"]
                row["epoch"] = checkpoint_objective["epoch"]
        elif status.get("status") in {"completed", "finished"}:
            raw_objective = row.get(objective["metric"])
            if isinstance(raw_objective, bool) or artifacts.float_or_none(raw_objective) is None:
                raise ValueError(
                    f"Completed adaptive run lacks finite {objective['metric']} objective evidence: "
                    f"{run['step_id']} / {run_id}"
                )
        row["status"] = status.get("status", "")
        row["stop_reason"] = status.get("stop_reason", "")
        history_monitor = manifest.get("monitor")
        if not history_monitor:
            run_config = yaml.safe_load(Path(run["config"]).read_text())
            task = run_config.get("finetune", {}).get("task") or {}
            history_monitor = task.get("monitor")
            if not history_monitor:
                task_args = SimpleNamespace()
                plan_rendering.apply_finetune_task_flags(task_args, recipe, task)
                history_monitor = task_args.monitor
        training_history = experiment_sources.read_wandb_training_history(
            workspace,
            status,
            monitor=str(history_monitor),
            objective=objective["metric"],
        )
        row["training_history"] = (
            json.dumps(training_history, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if training_history is not None
            else ""
        )
        row["pid"] = status.get("pid", "")
        rows.append(row)
    return rows


def manifest_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    metrics_value = manifest.get("metrics")
    metrics = metrics_value if isinstance(metrics_value, dict) else {}
    row = {key: value for key, value in metrics.items() if isinstance(key, str) and key != "status"}
    for key in ("best_model_score", "epoch", "monitor", "monitor_mode"):
        if manifest.get(key) is not None:
            row[key] = manifest.get(key)
    return row


def _test_checkpoint_evidence(
    manifest: dict[str, Any],
    objective: adaptive_proposals.ProposalObjective | dict[str, str],
    checkpoint_dir: str,
    checkpoint_names: list[str],
) -> tuple[checkpoint_test_results.CheckpointTestResult, list[dict[str, Any]]] | None:
    results = manifest.get("checkpoint_test_results")
    if manifest.get("test_all_checkpoints_after_fit") is not True or not isinstance(results, list):
        return None
    try:
        expected = checkpoint_test_results.expected_epoch_checkpoints(
            checkpoint_dir,
            checkpoint_names,
            step_id="adaptive",
            run_id="candidate",
        )
        candidates = checkpoint_test_results.validate_checkpoint_test_results(
            results,
            objective["metric"],
            expected,
            step_id="adaptive",
            run_id="candidate",
        )
    except ValueError:
        return None
    metrics_by_path = {
        result["checkpoint_path"]: {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in result["metrics"].items()
        }
        for result in results
    }
    trajectory = [
        {
            "checkpoint_path": candidate["checkpoint_path"],
            "epoch": candidate["epoch"],
            "metrics": metrics_by_path[candidate["checkpoint_path"]],
        }
        for candidate in sorted(candidates, key=lambda row: (row["epoch"], row["checkpoint_path"]))
    ]
    return checkpoint_test_results.best_checkpoint_test_result(candidates, objective["mode"]), trajectory


def rank_rows(
    rows: list[dict[str, Any]], objective: adaptive_proposals.ProposalObjective | dict[str, str]
) -> list[dict[str, Any]]:
    reverse = objective["mode"] == "max"

    def score(row: dict[str, Any]) -> float | None:
        try:
            value = float(row.get(objective["metric"], ""))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    scored = [(value, row) for row in rows if (value := score(row)) is not None]
    return [row for value, row in sorted(scored, key=lambda item: item[0], reverse=reverse)]
