from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import yaml

from . import experiment_io as exp_io, run_artifacts as artifacts, run_evidence as evidence
from .experiment_workspace import (
    append_event as _write_experiment_event,
    canonical_local_experiment_root,
    ensure_experiment_workspace,
    experiment_root,
    managed_run_key,
    managed_run_parameters,
    merge_run_manifest,
    read_run_manifest,
    validate_frozen_run_update,
    validate_managed_run_rows,
)
from .hparam_runtime import launch_hparam_runs, monitor_hparam_runs, stop_hparam_run
from .manifests import read_json, read_rows, utc_now, write_json, write_rows, write_text
from .models import resolve_repo_path
from .plans import build_plan, preflight_plan
from .recipes import load_recipe_with_base, recipe_name


def init_adaptive_workflow(recipe_path: str | Path, output_dir: str | Path) -> Path:
    root = canonical_local_experiment_root(output_dir, Path.cwd())
    resolved_recipe_path = resolve_repo_path(recipe_path)
    if resolved_recipe_path is None:
        raise FileNotFoundError("Path is required.")
    recipe_path = resolved_recipe_path.resolve()
    recipe = load_recipe_with_base(recipe_path)
    recipe["_recipe_path"] = str(recipe_path)
    _validate_adaptive_recipe(recipe)
    adaptive_dir = root / "adaptive"
    round_dir = adaptive_dir / "rounds" / "round_000"
    _, _, preflight = preflight_plan(recipe_path=recipe_path, output_dir=round_dir)
    if preflight.exit_code != 0:
        raise RuntimeError(f"Round 000 plan failed preflight with exit code {preflight.exit_code}.")
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Adaptive workflow is not bound to an experiment workspace.")
    initial_run_count = _hparam_count(recipe)
    initial_run_count = min(initial_run_count, int((recipe.get("search") or {}).get("max_runs") or initial_run_count))
    if initial_run_count > int(_adaptive(recipe).get("max_runs_total") or 10**9):
        raise ValueError("Round 000 would exceed adaptive.max_runs_total.")
    exp_io.validate_managed_output_paths(
        workspace,
        [
            round_dir / "round_recipe.yaml",
            adaptive_dir / "workflow.json",
            adaptive_dir / "run_registry.tsv",
            adaptive_dir / "README.md",
            workspace / "events.jsonl",
        ],
    )
    ensure_experiment_workspace(recipe, round_dir)
    round_recipe = _write_round_recipe(recipe, recipe_path, round_dir, 0)
    report = build_plan(recipe_path=round_recipe, output_dir=round_dir)
    if report.exit_code != 0:
        raise RuntimeError(f"Round 000 plan failed with exit code {report.exit_code}.")
    workflow = {
        "recipe_path": str(recipe_path),
        "root": str(root),
        "external_optimized": True,
        "objective_metric": _adaptive(recipe).get("objective_metric", "test_auroc"),
        "objective_mode": _adaptive(recipe).get("objective_mode", "max"),
    }
    write_json(adaptive_dir / "workflow.json", workflow)
    _append_event(root, "adaptive_init", {"round": 0, "recipe_path": str(recipe_path), "round_dir": str(round_dir)})
    _append_registry_rows(root, 0, round_dir)
    write_text(adaptive_dir / "README.md", _adaptive_readme(workflow))
    return root


def digest_hparam_run(run_dir: str | Path) -> Path:
    root = canonical_local_experiment_root(run_dir, Path.cwd())
    workflow_root, round_dir, round_index = _resolve_workflow_round(root)
    if (workflow_root / "adaptive" / "workflow.json").exists():
        _workflow(workflow_root)
    plan = artifacts.read_hparam_plan(round_dir)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Adaptive workflow is not bound to an experiment workspace.")
    out_dir = workflow_root / "adaptive" / "digests"
    out = out_dir / f"round_{round_index:03d}.csv"
    # Digest outputs must be safe before monitor is allowed to update canonical state.
    exp_io.validate_managed_output_paths(
        workspace,
        [
            out,
            out_dir / f"round_{round_index:03d}.md",
            workflow_root / "adaptive" / "incumbents.tsv",
            workspace / "events.jsonl",
        ],
    )
    monitor_hparam_runs(round_dir)
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
            manifest_path = str(status.get("run_manifest") or "")
            manifest = {}
            checkpoint_names = []
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
        row.update(_manifest_metrics(manifest))
        row["status"] = status.get("status", "")
        row["pid"] = status.get("pid", "")
        rows.append(row)
    write_rows(out, rows)
    write_text(out_dir / f"round_{round_index:03d}.md", _digest_markdown(rows, _objective(workflow_root, recipe)))
    _append_event(workflow_root, "digest", {"round": round_index, "path": str(out), "rows": len(rows)})
    _write_incumbent(workflow_root, rows, _objective(workflow_root, recipe), round_index)
    return out


def suggest_next_round(workflow_dir: str | Path) -> Path:
    root = canonical_local_experiment_root(workflow_dir, Path.cwd())
    workflow = _workflow(root)
    recipe = load_recipe_with_base(workflow["recipe_path"])
    current_round = _latest_round_index(root)
    next_round = current_round + 1
    digest = _latest_digest(root)
    rows = read_rows(digest)
    objective = _objective(root, recipe)
    ranked = _rank_rows(rows, objective)
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Adaptive workflow is not bound to an experiment workspace.")
    out_dir = root / "adaptive" / "suggestions"
    out = out_dir / f"round_{next_round:03d}.yaml"
    exp_io.validate_managed_output_paths(
        workspace,
        [out, out_dir / f"round_{next_round:03d}.md", workspace / "events.jsonl"],
    )
    if not ranked:
        _append_event(root, "suggest_blocked", {"round": next_round, "reason": "no_scored_runs"})
        raise ValueError(f"No digest rows with finite {objective['metric']} are available for suggestion.")
    best = ranked[0]
    suggested = copy.deepcopy(recipe)
    suggested_root = experiment_root(suggested)
    if suggested_root is not None:
        suggested["experiment"]["root"] = str(suggested_root)
    suggested["name"] = f"{recipe_name(recipe)}_adaptive_round_{next_round:03d}"
    suggested.setdefault("search", {})["parameters"] = _suggest_parameters(recipe, ranked)
    suggested["search"]["max_runs"] = int(_adaptive(recipe).get("round_size") or _hparam_count(suggested))
    if suggested.get("base_recipe"):
        suggested["base_recipe"] = str(_resolve_base_recipe(workflow["recipe_path"], suggested["base_recipe"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(_strip_internal_recipe_keys(suggested), sort_keys=False))
    rationale = _suggestion_rationale(next_round, objective, best, suggested["search"]["parameters"])
    write_text(out_dir / f"round_{next_round:03d}.md", rationale)
    _append_event(root, "suggest", {"round": next_round, "path": str(out), "best_run": best.get("run_id")})
    return out


def adaptive_step(workflow_dir: str | Path, *, execute: bool = False) -> Path:
    root = canonical_local_experiment_root(workflow_dir, Path.cwd())
    workflow = _workflow(root)
    recipe = load_recipe_with_base(workflow["recipe_path"])
    current_round = _latest_round_index(root)
    round_dir = _round_dir(root, current_round)
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Adaptive workflow is not bound to an experiment workspace.")
    next_round = current_round + 1
    next_dir = _round_dir(root, next_round)
    targets = [workspace / "events.jsonl"]
    if execute:
        targets.extend([root / "adaptive" / "run_registry.tsv", next_dir / "round_recipe.yaml"])
    exp_io.validate_managed_output_paths(workspace, targets)
    digest = digest_hparam_run(round_dir)
    suggestion = suggest_next_round(root)
    next_recipe, _, preflight = preflight_plan(recipe_path=suggestion, output_dir=next_dir)
    if preflight.exit_code != 0:
        raise RuntimeError(f"Round {next_round:03d} plan failed preflight with exit code {preflight.exit_code}.")
    next_run_count = _hparam_count(next_recipe)
    next_max_runs = (next_recipe.get("search") or {}).get("max_runs")
    if next_max_runs not in (None, ""):
        next_run_count = min(next_run_count, int(next_max_runs))
    # Retiring current runs is allowed only when the complete replacement round fits the remaining budget.
    budget_exhausted = execute and _budget_exhausted(root, recipe, prospective_runs=next_run_count)
    if execute and not budget_exhausted:
        round_recipe = _write_round_recipe(load_recipe_with_base(suggestion), suggestion, next_dir, next_round)
        report = build_plan(recipe_path=round_recipe, output_dir=next_dir)
        if report.exit_code != 0:
            raise RuntimeError(f"Round {next_round:03d} plan failed with exit code {report.exit_code}.")
        _append_registry_rows(root, next_round, next_dir)
        current_plan = artifacts.read_hparam_plan(round_dir)
        bad_run_keys = _bad_running_run_keys(root, round_dir, recipe)
        ordered_bad_run_keys = [
            managed_run_key(run) for run in current_plan["runs"] if managed_run_key(run) in bad_run_keys
        ]
        next_plan_keys = {managed_run_key(run) for run in artifacts.read_hparam_plan(next_dir)["runs"]}
        canonical_rows = read_run_manifest(workspace)
        started_keys = {
            managed_run_key(row)
            for row in canonical_rows
            if managed_run_key(row) in next_plan_keys and row.get("status") in {"launched", "running"}
        }
        launch_failed_keys = {
            managed_run_key(row)
            for row in canonical_rows
            if managed_run_key(row) in next_plan_keys and row.get("status") == "launch_failed"
        }
        try:
            launch_hparam_runs(next_dir, dry_run=False)
        except Exception as exc:
            canonical_rows = read_run_manifest(workspace)
            next_round_rows = [row for row in canonical_rows if managed_run_key(row) in next_plan_keys]
            refreshed_started_keys = {
                managed_run_key(row) for row in next_round_rows if row.get("status") in {"launched", "running"}
            }
            newly_launch_failed = {
                managed_run_key(row) for row in next_round_rows if row.get("status") == "launch_failed"
            } - launch_failed_keys
            round_committed = bool(refreshed_started_keys - started_keys) and not newly_launch_failed
            if round_committed:
                _append_event(root, "launch_round", {"round": next_round, "round_dir": str(next_dir)})
                _supersede_pending_runs(root, round_dir)
            committed = "is already committed" if round_committed else "was not committed"
            raise RuntimeError(
                f"Adaptive replacement launch failed; round {next_round:03d} {committed}. "
                "Confirmed stopped current runs: none. No current runs were retired."
            ) from exc
        canonical_rows = read_run_manifest(workspace)
        next_round_rows = [row for row in canonical_rows if managed_run_key(row) in next_plan_keys]
        refreshed_started_keys = {
            managed_run_key(row) for row in next_round_rows if row.get("status") in {"launched", "running"}
        }
        newly_launch_failed = {
            managed_run_key(row) for row in next_round_rows if row.get("status") == "launch_failed"
        } - launch_failed_keys
        if newly_launch_failed:
            failed_ids = ", ".join(sorted(key[1] for key in newly_launch_failed))
            raise RuntimeError(
                f"Adaptive replacement launch failed for {failed_ids}; round {next_round:03d} was not committed. "
                "Confirmed stopped current runs: none. No current runs were retired."
            )
        retirement_credit = len(refreshed_started_keys - started_keys)
        started_keys = refreshed_started_keys
        round_committed = False
        stopped_run_keys: list[tuple[str, str]] = []
        if retirement_credit:
            _append_event(root, "launch_round", {"round": next_round, "round_dir": str(next_dir)})
            round_committed = True
            _supersede_pending_runs(root, round_dir)

        bad_index = 0
        while bad_index < len(ordered_bad_run_keys):
            canonical_rows = read_run_manifest(workspace)
            next_round_rows = [row for row in canonical_rows if managed_run_key(row) in next_plan_keys]
            pending = any(row.get("status") in {"planned", "pending"} for row in next_round_rows)
            if retirement_credit <= 0 and not pending:
                break
            run_key = ordered_bad_run_keys[bad_index]
            bad_index += 1
            try:
                stopped = _stop_bad_running_runs(root, round_dir, recipe, run_keys={run_key})
            except Exception as exc:
                canonical_by_key = {managed_run_key(row): row for row in read_run_manifest(workspace)}
                if canonical_by_key[run_key].get("status") == "stopped" and run_key not in stopped_run_keys:
                    stopped_run_keys.append(run_key)
                committed = "is already committed" if round_committed else "was not committed"
                stopped_ids = ", ".join(key[1] for key in stopped_run_keys) or "none"
                raise RuntimeError(
                    f"Adaptive replacement failed while stopping {run_key[1]}; round {next_round:03d} "
                    f"{committed}. Confirmed stopped current runs: {stopped_ids}. "
                    "No additional current runs were retired."
                ) from exc
            if stopped:
                stopped_run_keys.extend(stopped)
                retirement_credit -= len(stopped)
            canonical_rows = read_run_manifest(workspace)
            next_round_rows = [row for row in canonical_rows if managed_run_key(row) in next_plan_keys]
            if not any(row.get("status") in {"planned", "pending"} for row in next_round_rows):
                continue
            before_launch = started_keys
            before_launch_failed = {
                managed_run_key(row) for row in next_round_rows if row.get("status") == "launch_failed"
            }
            try:
                launch_hparam_runs(next_dir, dry_run=False)
            except Exception as exc:
                canonical_rows = read_run_manifest(workspace)
                next_round_rows = [row for row in canonical_rows if managed_run_key(row) in next_plan_keys]
                refreshed_started_keys = {
                    managed_run_key(row) for row in next_round_rows if row.get("status") in {"launched", "running"}
                }
                newly_launch_failed = {
                    managed_run_key(row) for row in next_round_rows if row.get("status") == "launch_failed"
                } - before_launch_failed
                if refreshed_started_keys - before_launch and not newly_launch_failed and not round_committed:
                    _append_event(root, "launch_round", {"round": next_round, "round_dir": str(next_dir)})
                    round_committed = True
                    _supersede_pending_runs(root, round_dir)
                committed = "is already committed" if round_committed else "was not committed"
                stopped_ids = ", ".join(key[1] for key in stopped_run_keys) or "none"
                raise RuntimeError(
                    f"Adaptive replacement launch failed after the stop attempt for {run_key[1]}; "
                    f"round {next_round:03d} "
                    f"{committed}. Confirmed stopped current runs: {stopped_ids}. "
                    "No additional current runs were retired."
                ) from exc
            canonical_rows = read_run_manifest(workspace)
            next_round_rows = [row for row in canonical_rows if managed_run_key(row) in next_plan_keys]
            started_keys = {
                managed_run_key(row) for row in next_round_rows if row.get("status") in {"launched", "running"}
            }
            newly_launch_failed = {
                managed_run_key(row) for row in next_round_rows if row.get("status") == "launch_failed"
            } - before_launch_failed
            if newly_launch_failed:
                failed_ids = ", ".join(sorted(key[1] for key in newly_launch_failed))
                committed = "is already committed" if round_committed else "was not committed"
                stopped_ids = ", ".join(key[1] for key in stopped_run_keys) or "none"
                raise RuntimeError(
                    f"Adaptive replacement launch failed for {failed_ids} after the stop attempt for {run_key[1]}; "
                    f"round {next_round:03d} {committed}. Confirmed stopped current runs: {stopped_ids}. "
                    "No additional current runs were retired."
                )
            newly_started = started_keys - before_launch
            if not newly_started:
                statuses = ", ".join(sorted({str(row.get("status") or "") for row in next_round_rows})) or "none"
                committed = "is already committed" if round_committed else "was not committed"
                stopped_ids = ", ".join(key[1] for key in stopped_run_keys) or "none"
                stop_attempt = f"stopping {run_key[1]}" if stopped else f"the stop attempt for {run_key[1]}"
                raise RuntimeError(
                    f"Round {next_round:03d} started no additional runs after {stop_attempt} "
                    f"(statuses: {statuses}); the round {committed}. Confirmed stopped current runs: {stopped_ids}. "
                    f"No additional current runs were retired. The registered replacement round remains at {next_dir}."
                )
            if not round_committed:
                _append_event(root, "launch_round", {"round": next_round, "round_dir": str(next_dir)})
                round_committed = True
                _supersede_pending_runs(root, round_dir)
            retirement_credit += len(newly_started)

        if not round_committed:
            statuses = ", ".join(sorted({str(row.get("status") or "") for row in next_round_rows})) or "none"
            raise RuntimeError(
                f"Round {next_round:03d} started no runs (statuses: {statuses}); the round was not committed and "
                f"current runs were not retired. The registered replacement round remains at {next_dir}."
            )
    elif budget_exhausted:
        _append_event(
            root,
            "adaptive_budget_exhausted",
            {"round": current_round, "digest": str(digest), "suggestion": str(suggestion)},
        )
    else:
        _append_event(
            root,
            "adaptive_step_dry_run",
            {"round": current_round, "digest": str(digest), "suggestion": str(suggestion)},
        )
    return suggestion


def adaptive_loop(workflow_dir: str | Path, *, execute: bool = False) -> Path:
    root = canonical_local_experiment_root(workflow_dir, Path.cwd())
    recipe = load_recipe_with_base(_workflow(root)["recipe_path"])
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Adaptive workflow is not bound to an experiment workspace.")
    exp_io.validate_managed_output_paths(workspace, [workspace / "events.jsonl"])
    last = root
    while not _budget_exhausted(root, recipe):
        previous_round = _latest_round_index(root)
        last = adaptive_step(root, execute=execute)
        if not execute:
            break
        if _latest_round_index(root) == previous_round:
            break
        time.sleep(float(_adaptive(recipe).get("poll_seconds") or 60))
    _append_event(root, "adaptive_loop_done", {"path": str(last)})
    return Path(last)


def _validate_adaptive_recipe(recipe: dict[str, Any]) -> None:
    adaptive = _adaptive(recipe)
    if not adaptive.get("enabled"):
        raise ValueError("adaptive.enabled must be true for adaptive workflow.")
    objective = str(adaptive.get("objective_metric") or "test_auroc")
    uses_external = objective.startswith("test_") or objective.startswith("external_")
    if uses_external and adaptive.get("test_feedback_for_selection") is not True:
        raise ValueError("adaptive.test_feedback_for_selection=true is required for test/external objectives.")


def _adaptive(recipe: dict[str, Any]) -> dict[str, Any]:
    return recipe.get("adaptive") if isinstance(recipe.get("adaptive"), dict) else {}


def _append_event(root: Path, event_type: str, payload: dict[str, Any]) -> None:
    workflow_path = root / "adaptive" / "workflow.json"
    target = root
    if workflow_path.exists():
        workflow = json.loads(workflow_path.read_text())
        recipe = load_recipe_with_base(workflow["recipe_path"])
        target = experiment_root(recipe) or root
    _write_experiment_event(target, event_type, payload)


def _objective(root: Path, recipe: dict[str, Any]) -> dict[str, str]:
    workflow = _workflow(root) if (root / "adaptive" / "workflow.json").exists() else {}
    adaptive = _adaptive(recipe)
    return {
        "metric": str(workflow.get("objective_metric") or adaptive.get("objective_metric") or "test_auroc"),
        "mode": str(workflow.get("objective_mode") or adaptive.get("objective_mode") or "max"),
    }


def _workflow(root: Path) -> dict[str, Any]:
    path = root / "adaptive" / "workflow.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing adaptive workflow: {path}")
    workflow = read_json(path)
    if not isinstance(workflow, dict):
        raise ValueError(f"Adaptive workflow must contain a mapping: {path}")
    if str(workflow.get("root") or "") != str(root):
        raise ValueError(f"Adaptive workflow root differs from the requested workspace: {root}")
    recipe_path = Path(str(workflow.get("recipe_path") or ""))
    if not recipe_path.is_absolute():
        raise ValueError(f"Adaptive workflow recipe_path must be absolute: {path}")
    legacy_registry = root / "adaptive" / "trial_registry.tsv"
    if legacy_registry.exists():
        raise ValueError(f"Legacy adaptive registry is read-only and cannot be managed: {legacy_registry}")
    registry_path = root / "adaptive" / "run_registry.tsv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing adaptive run registry: {registry_path}")
    registry_rows = read_rows(registry_path, require_managed_identity=True)
    validate_managed_run_rows(registry_rows, source=str(registry_path), cardinality="one_per_run")
    round_index = _latest_round_index(root)
    round_dir = _round_dir(root, round_index)
    plan = artifacts.read_hparam_plan(round_dir)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Adaptive workflow is not bound to an experiment workspace.")
    canonical_by_key = {managed_run_key(row): row for row in read_run_manifest(workspace)}
    for registered in registry_rows:
        canonical = canonical_by_key.get(managed_run_key(registered))
        if canonical is None:
            raise ValueError(
                f"Adaptive registry row is outside the canonical manifest: "
                f"{registered.get('step_id', '')} / {registered.get('run_id', '')}"
            )
        validate_frozen_run_update(canonical, registered)
    registry_by_key = {managed_run_key(row): row for row in registry_rows}
    for run in plan.get("runs", []):
        key = managed_run_key(run)
        registered = registry_by_key.get(key)
        if registered is None:
            raise ValueError(f"Adaptive registry is missing the current plan run: {key[0]} / {key[1]}")
        if str(registered.get("round") or "") != str(round_index) or str(registered.get("round_dir") or "") != str(
            round_dir
        ):
            raise ValueError(f"Adaptive registry round binding differs for run: {key[0]} / {key[1]}")
        validate_frozen_run_update(run, registered)
    return workflow


def _write_round_recipe(
    recipe: dict[str, Any], source_recipe_path: str | Path, round_dir: Path, round_index: int
) -> Path:
    round_dir.mkdir(parents=True, exist_ok=True)
    copied = _strip_internal_recipe_keys(copy.deepcopy(recipe))
    copied_root = experiment_root(copied)
    if copied_root is not None:
        copied["experiment"]["root"] = str(copied_root)
    if copied.get("base_recipe"):
        copied["base_recipe"] = str(_resolve_base_recipe(source_recipe_path, copied["base_recipe"]))
    copied["name"] = f"{recipe_name(recipe)}-round-{round_index:03d}"
    target = round_dir / "round_recipe.yaml"
    target.write_text(yaml.safe_dump(copied, sort_keys=False))
    return target


def _resolve_base_recipe(recipe_path: str | Path, base_recipe: str | Path) -> Path:
    raw = Path(base_recipe).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    source = Path(recipe_path)
    if source.exists():
        candidate = source.parent / raw
        if candidate.exists():
            return candidate.resolve()
    resolved = resolve_repo_path(raw)
    return resolved.resolve() if resolved is not None else raw.resolve()


def _strip_internal_recipe_keys(recipe: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in recipe.items() if not str(key).startswith("_")}


def _append_registry_rows(root: Path, round_index: int, round_dir: Path) -> None:
    path = root / "adaptive" / "run_registry.tsv"
    plan = artifacts.read_hparam_plan(round_dir)
    rows = read_rows(path, require_managed_identity=True) if path.exists() else []
    for run in plan.get("runs", []):
        rows.append(
            {
                "round": round_index,
                "experiment_id": run.get("experiment_id"),
                "step_id": run.get("step_id"),
                "run_id": run.get("run_id"),
                "run_name": run.get("run_name"),
                "version": run.get("version"),
                "config": run.get("config"),
                "script": run.get("script"),
                "round_dir": str(round_dir),
                "registered_at": utc_now(),
            }
        )
    write_rows(path, rows)


def _resolve_workflow_round(path: Path) -> tuple[Path, Path, int]:
    if (path / "adaptive" / "workflow.json").exists():
        idx = _latest_round_index(path)
        return path, _round_dir(path, idx), idx
    parts = path.parts
    if "rounds" in parts:
        idx = int(path.name.split("_")[-1])
        workflow_root = Path(*parts[: parts.index("adaptive")]) if "adaptive" in parts else path
        return workflow_root, path, idx
    return path, path, 0


def _latest_round_index(root: Path) -> int:
    registry_path = root / "adaptive" / "run_registry.tsv"
    if not registry_path.exists():
        return 0
    registry = read_rows(registry_path, require_managed_identity=True)
    registered_rounds = {int(row["round"]) for row in registry}
    initial_plan = artifacts.read_hparam_plan(_round_dir(root, 0))
    recipe = initial_plan.get("recipe") if isinstance(initial_plan.get("recipe"), dict) else {}
    workspace = experiment_root(recipe)
    events_path = workspace / "events.jsonl" if workspace is not None else None
    if events_path is None:
        return 0
    exp_io.validate_managed_output_paths(workspace, [events_path])
    if not events_path.exists():
        return 0
    committed = []
    for line in events_path.read_text().splitlines():
        event = json.loads(line)
        if event.get("event_type") != "launch_round":
            continue
        round_index = int(event["round"])
        if round_index in registered_rounds and event.get("round_dir") == str(_round_dir(root, round_index)):
            committed.append(round_index)
    return max(committed, default=0)


def _round_dir(root: Path, index: int) -> Path:
    return root / "adaptive" / "rounds" / f"round_{index:03d}"


def _manifest_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {}
    row = {key: value for key, value in metrics.items() if isinstance(key, str) and key != "status"}
    for key in ("best_model_score", "epoch", "monitor", "monitor_mode"):
        if manifest.get(key) is not None:
            row[key] = manifest.get(key)
    return row


def _digest_markdown(rows: list[dict[str, Any]], objective: dict[str, str]) -> str:
    ranked = _rank_rows(rows, objective)
    lines = [
        "# Adaptive Hparam Digest",
        "",
        "external_optimized: true",
        f"objective: {objective['metric']} ({objective['mode']})",
        "",
        "## Top runs",
        "",
    ]
    for row in ranked[:5]:
        lines.append(
            f"- {row.get('run_id')}: {objective['metric']}={row.get(objective['metric'], '')} "
            f"status={row.get('status', '')} checkpoint={row.get('checkpoint_path', '')}"
        )
    return "\n".join(lines) + "\n"


def _write_incumbent(root: Path, rows: list[dict[str, Any]], objective: dict[str, str], round_index: int) -> None:
    ranked = _rank_rows(rows, objective)
    if not ranked:
        return
    best = ranked[0]
    path = root / "adaptive" / "incumbents.tsv"
    incumbents = read_rows(path) if path.exists() else []
    incumbents.append(
        {
            "round": round_index,
            "experiment_id": best.get("experiment_id", ""),
            "step_id": best.get("step_id", ""),
            "run_id": best.get("run_id", ""),
            "run_name": best.get("run_name", ""),
            "version": best.get("version", ""),
            "objective_metric": objective["metric"],
            "objective_mode": objective["mode"],
            "objective_score": best.get(objective["metric"], ""),
            "checkpoint_path": best.get("checkpoint_path", ""),
            "external_optimized": True,
            "selected_at": utc_now(),
        }
    )
    write_rows(path, incumbents)


def _rank_rows(rows: list[dict[str, Any]], objective: dict[str, str]) -> list[dict[str, Any]]:
    reverse = objective["mode"] == "max"

    def score(row: dict[str, Any]) -> float | None:
        try:
            value = float(row.get(objective["metric"], ""))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    scored = [(value, row) for row in rows if (value := score(row)) is not None]
    return [row for value, row in sorted(scored, key=lambda item: item[0], reverse=reverse)]


def _latest_digest(root: Path) -> Path:
    digests = sorted((root / "adaptive" / "digests").glob("round_*.csv"))
    if not digests:
        raise FileNotFoundError("No adaptive digest exists. Run hparam-digest first.")
    return digests[-1]


def _suggest_parameters(recipe: dict[str, Any], ranked: list[dict[str, Any]]) -> dict[str, list[Any]]:
    params = (recipe.get("search") or {}).get("parameters") or {}
    best = ranked[0]
    top = ranked[:3]
    suggested: dict[str, list[Any]] = {}
    for key, values in params.items():
        if key not in best:
            suggested[key] = values
            continue
        value = _coerce_like(best[key], values[0] if values else best[key])
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            suggested[key] = _numeric_neighbors(value)
        else:
            seen = [row[key] for row in top if row.get(key) not in (None, "")]
            suggested[key] = list(dict.fromkeys([value, *seen]))[:3]
    return suggested


def _coerce_like(value: Any, example: Any) -> Any:
    if isinstance(example, bool):
        return str(value).lower() in {"1", "true", "yes"}
    if isinstance(example, int) and not isinstance(example, bool):
        return int(float(value))
    if isinstance(example, float):
        return float(value)
    return value


def _numeric_neighbors(value: int | float) -> list[int | float]:
    if isinstance(value, int):
        return sorted(set([max(1, value - 1), value, value + 1]))
    if value == 0:
        return [0.0, 1e-6, 3e-6]
    return sorted(set([float(f"{value * 0.5:.6g}"), float(f"{value:.6g}"), float(f"{value * 1.5:.6g}")]))


def _hparam_count(recipe: dict[str, Any]) -> int:
    params = (recipe.get("search") or {}).get("parameters") or {}
    count = 1
    for choices in params.values():
        count *= len(choices)
    return count


def _suggestion_rationale(
    round_index: int, objective: dict[str, str], best: dict[str, Any], params: dict[str, list[Any]]
) -> str:
    lines = [
        f"# Adaptive Suggestion Round {round_index:03d}",
        "",
        "external_optimized: true",
        f"objective: {objective['metric']} ({objective['mode']})",
        f"best_run: {best.get('run_id', '')}",
        f"best_score: {best.get(objective['metric'], '')}",
        "",
        "## Parameters",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in params.items())
    return "\n".join(lines) + "\n"


def _supersede_pending_runs(root: Path, round_dir: Path) -> None:
    plan = artifacts.read_hparam_plan(round_dir)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    launch_path = round_dir / "launch_manifest.tsv"
    targets = [
        workspace / "run_manifest.tsv",
        workspace / "run_matrix.csv",
        workspace / "reports" / "run_matrix.md",
        workspace / "events.jsonl",
        round_dir / "run_status.tsv",
    ]
    if launch_path.exists():
        targets.append(launch_path)
    exp_io.validate_managed_output_paths(workspace, targets)
    canonical_rows = read_run_manifest(workspace)
    canonical_by_key = {managed_run_key(row): row for row in canonical_rows}
    transitions = []
    for run in plan["runs"]:
        row = canonical_by_key[managed_run_key(run)]
        if row.get("status") in {"planned", "pending"}:
            transitions.append(row)
    if transitions:
        committed = merge_run_manifest(
            workspace,
            [{"step_id": row["step_id"], "run_id": row["run_id"], "status": "superseded"} for row in transitions],
        )
    else:
        committed = canonical_rows
    committed_by_key = {managed_run_key(row): row for row in committed}
    round_rows = [committed_by_key[managed_run_key(run)] for run in plan["runs"]]
    write_rows(round_dir / "run_status.tsv", round_rows)
    if launch_path.exists():
        write_rows(launch_path, round_rows)
    for row in transitions:
        if committed_by_key[managed_run_key(row)].get("status") != "superseded":
            continue
        _append_event(
            root,
            "supersede_pending_run",
            {"round_dir": str(round_dir), "run_id": row["run_id"], "status": row["status"]},
        )


def _bad_running_run_keys(root: Path, round_dir: Path, recipe: dict[str, Any]) -> set[tuple[str, str]]:
    adaptive = _adaptive(recipe)
    replacement = adaptive.get("replacement") if isinstance(adaptive.get("replacement"), dict) else {}
    if not replacement.get("enabled", True) or not replacement.get("allow_running_stop", False):
        return set()
    objective = _objective(root, recipe)
    incumbent = _latest_incumbent_score(root)
    margin = float(replacement.get("kill_margin") or 0.0)
    plan = artifacts.read_hparam_plan(round_dir)
    plan_recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    workspace = experiment_root(plan_recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    plan_keys = {managed_run_key(run) for run in plan["runs"]}
    bad_keys = set()
    for row in read_run_manifest(workspace):
        key = managed_run_key(row)
        if key not in plan_keys:
            continue
        if row.get("status") != "running":
            continue
        should_stop = evidence.log_has_failure(row.get("log_path"), row)
        data = {}
        if not should_stop:
            observed_artifacts = evidence.runtime_artifacts(row)
            if observed_artifacts is not None:
                _manifest_path, data, _checkpoint_names = observed_artifacts
        score = artifacts.metric_value(data, objective["metric"])
        if (
            not should_stop
            and incumbent is not None
            and score not in ("", None)
            and _grace_satisfied(row, data, replacement)
        ):
            try:
                value = float(score)
                should_stop = value < incumbent - margin if objective["mode"] == "max" else value > incumbent + margin
            except (TypeError, ValueError):
                should_stop = False
        if should_stop:
            bad_keys.add(key)
    return bad_keys


def _stop_bad_running_runs(
    root: Path,
    round_dir: Path,
    recipe: dict[str, Any],
    *,
    run_keys: set[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    keys = _bad_running_run_keys(root, round_dir, recipe) if run_keys is None else run_keys
    if not keys:
        return []
    plan = artifacts.read_hparam_plan(round_dir)
    plan_recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    workspace = experiment_root(plan_recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    canonical_by_key = {managed_run_key(row): row for row in read_run_manifest(workspace)}
    stopped = []
    for run in plan["runs"]:
        key = managed_run_key(run)
        row = canonical_by_key[key]
        if key not in keys or row.get("status") != "running":
            continue
        stop_hparam_run(round_dir, str(row["run_id"]), reason="adaptive replacement")
        _append_event(root, "stop_bad_running_run", {"round_dir": str(round_dir), "run_id": row["run_id"]})
        stopped.append(key)
    return stopped


def _grace_satisfied(row: dict[str, Any], manifest: dict[str, Any], replacement: dict[str, Any]) -> bool:
    grace_epochs = replacement.get("grace_epochs")
    if grace_epochs is not None:
        try:
            if float(manifest.get("epoch", "")) < float(grace_epochs):
                return False
        except (TypeError, ValueError):
            return False
    grace_minutes = replacement.get("grace_minutes")
    if grace_minutes is not None:
        minutes = _minutes_since(row.get("launched_at", ""))
        if minutes is None or minutes < float(grace_minutes):
            return False
    return True


def _minutes_since(timestamp: str) -> float | None:
    try:
        start = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return (datetime.now(timezone.utc) - start).total_seconds() / 60


def _latest_incumbent_score(root: Path) -> float | None:
    rows = read_rows(root / "adaptive" / "incumbents.tsv")
    if not rows:
        return None
    try:
        return float(rows[-1]["objective_score"])
    except (KeyError, TypeError, ValueError):
        return None


def _budget_exhausted(root: Path, recipe: dict[str, Any], *, prospective_runs: int = 0) -> bool:
    adaptive = _adaptive(recipe)
    max_rounds = int(adaptive.get("max_rounds") or 1)
    max_runs = int(adaptive.get("max_runs_total") or 10**9)
    current_runs = len(read_rows(root / "adaptive" / "run_registry.tsv", require_managed_identity=True))
    return (
        _latest_round_index(root) + 1 >= max_rounds
        or current_runs >= max_runs
        or current_runs + prospective_runs > max_runs
    )


def _adaptive_readme(workflow: dict[str, Any]) -> str:
    return (
        "# Adaptive Hparam Workflow\n\n"
        "This workflow is external-optimized and may use test/external feedback for selection.\n\n"
        f"Objective: `{workflow['objective_metric']}` ({workflow['objective_mode']}).\n"
    )
