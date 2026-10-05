"""Digest, preflight, registration, launch, and lifecycle of adaptive hparam search.

Domain-free kernel orchestration surrounding the pure contract in
``adaptive_proposals`` and the durable round, registry, and event state in
``adaptive_state``. Digests a finished round, preflights and publishes the
accepted proposal as the next round's suggestion, registers and commits that
round, and drives ``adaptive_step`` / ``adaptive_loop``.
``adaptive_evidence`` owns round evidence reading and ranking,
``adaptive_handshake`` owns the proposal request, submission binding, and replay
of an accepted proposal, and ``adaptive_replacement`` launches a replacement
round that retires trailing current-round runs; this module owns monitoring and
digest, suggestion, receipt, incumbent, and event publication.

Round publication is recoverable by construction: staging, commit, and launch
are distinct steps, so a crash between them is reconciled on the next step
rather than leaving a half-registered round or a double launch.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any, Literal, overload

import yaml

from . import (
    adaptive_evidence,
    adaptive_handshake,
    adaptive_proposals,
    adaptive_replacement,
    adaptive_state,
    experiment_io as exp_io,
    plan_contract,
    plan_hparam,
    run_artifacts as artifacts,
)
from .experiment_workspace import (
    AdaptiveInitEvent,
    AdaptiveProposalAcceptedEvent,
    canonical_local_experiment_root,
    experiment_root,
    file_sha256,
    managed_run_key,
    plan_registration_lock,
    plan_registration_rows_state,
    read_experiment_events,
    read_run_manifest,
    validate_step_registration,
)
from .hparam_runtime import monitor_hparam_runs
from .manifests import read_json, read_rows, utc_now, write_rows, write_text
from .models import is_full_git_object_id, recipe_name, resolve_repo_path
from .plans import build_plan, plan_publication_lock, preflight_plan, publish_staged_plan_locked
from .recipes import load_recipe_with_base, strip_internal_recipe_keys


def _preflight_details(report) -> str:
    """Every blocking issue in a preflight report, as one message line."""
    return "; ".join(f"{issue.field}: {issue.message}" for issue in report.blocking_issues())


def _require_preflight_pass(report, subject: str) -> None:
    """Raise with every blocking issue when ``subject``'s preflight did not pass."""
    if report.exit_code != 0:
        raise RuntimeError(
            f"{subject} failed preflight with exit code {report.exit_code}: {_preflight_details(report)}"
        )


def _preflight_candidate(candidate_bytes: bytes, next_dir: Path, subject: str) -> dict[str, Any]:
    with TemporaryDirectory(prefix="agent-tools-adaptive-") as temp_dir:
        candidate_path = Path(temp_dir) / "suggested.yaml"
        candidate_path.write_bytes(candidate_bytes)
        recipe, _, report = preflight_plan(
            recipe_path=candidate_path, output_dir=next_dir, allow_adaptive_workflow=True
        )
    _require_preflight_pass(report, subject)
    return recipe


class AdaptivePreflightError(RuntimeError):
    def __init__(self, report):
        self.report = report
        details = _preflight_details(report)
        super().__init__(f"Round 000 plan failed preflight with exit code {report.exit_code}: {details}")


def _validate_adaptive_step_registration(workspace: Path, round_dir: Path, plan: Mapping[str, Any]) -> None:
    recipe_value = plan.get("recipe")
    recipe = recipe_value if isinstance(recipe_value, dict) else {}
    validate_step_registration(
        workspace,
        {
            "step": recipe["step"],
            "experiment_id": recipe["experiment"]["id"],
            "plan_controller": "adaptive",
            "recipe_path": recipe.get("_recipe_path", ""),
            "plans": [str(round_dir.resolve())],
        },
    )


def init_adaptive_workflow(recipe_path: str | Path, output_dir: str | Path) -> Path:
    root = canonical_local_experiment_root(output_dir, Path.cwd())
    workspace = experiment_root(load_recipe_with_base(recipe_path))
    registration_root = workspace if workspace is not None else root
    with plan_registration_lock(registration_root):
        return _init_adaptive_workflow_locked(recipe_path, root, locked_workspace=registration_root)


@dataclass(frozen=True)
class _InitialRoundInputs:
    recipe_path: Path
    recipe: dict[str, Any]
    adaptive_dir: Path
    round_dir: Path
    workflow_path: Path
    workspace: Path
    source_config_bytes: bytes
    source_config_sha256: str
    round_recipe_payload: dict[str, Any]


def _prepare_initial_round(recipe_path: str | Path, root: Path, *, locked_workspace: Path) -> _InitialRoundInputs:
    resolved_recipe_path = resolve_repo_path(recipe_path)
    if resolved_recipe_path is None:
        raise FileNotFoundError("Path is required.")
    recipe_path = resolved_recipe_path.resolve()
    adaptive_dir = root / "adaptive"
    round_dir = adaptive_dir / "rounds" / "round_000"
    workflow_path = adaptive_dir / "workflow.json"
    recipe, config, preflight = preflight_plan(
        recipe_path=recipe_path,
        output_dir=round_dir,
        allow_existing_output_artifacts=True,
        allow_adaptive_workflow=True,
    )
    if preflight.exit_code != 0:
        raise AdaptivePreflightError(preflight)
    source_config_bytes = config.get("_source_config_bytes") if isinstance(config, dict) else None
    source_config_sha256 = config.get("_source_config_sha256") if isinstance(config, dict) else None
    if not isinstance(source_config_bytes, bytes) or not isinstance(source_config_sha256, str):
        raise ValueError("Successful adaptive preflight did not bind the source config bytes.")
    if hashlib.sha256(source_config_bytes).hexdigest() != source_config_sha256:
        raise ValueError("Adaptive source config bytes do not match their SHA-256.")
    _validate_adaptive_recipe(recipe)
    recipe = plan_hparam.freeze_hparam_execution(recipe)
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Adaptive workflow is not bound to an experiment workspace.")
    if workspace != locked_workspace:
        raise ValueError("Adaptive experiment.root changed while acquiring the registration lock.")
    initial_run_count = _round_run_count(recipe)
    if initial_run_count > int(adaptive_state.adaptive_settings(recipe).get("max_runs_total") or 10**9):
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
    round_recipe_payload = _materialized_round_recipe(recipe, recipe_path, 0)
    return _InitialRoundInputs(
        recipe_path=recipe_path,
        recipe=recipe,
        adaptive_dir=adaptive_dir,
        round_dir=round_dir,
        workflow_path=workflow_path,
        workspace=workspace,
        source_config_bytes=source_config_bytes,
        source_config_sha256=source_config_sha256,
        round_recipe_payload=round_recipe_payload,
    )


def _init_adaptive_workflow_locked(recipe_path: str | Path, root: Path, *, locked_workspace: Path) -> Path:
    inputs = _prepare_initial_round(recipe_path, root, locked_workspace=locked_workspace)
    recipe_path = inputs.recipe_path
    recipe = inputs.recipe
    adaptive_dir = inputs.adaptive_dir
    round_dir = inputs.round_dir
    workflow_path = inputs.workflow_path
    workspace = inputs.workspace
    source_config_bytes = inputs.source_config_bytes
    source_config_sha256 = inputs.source_config_sha256
    round_recipe_payload = inputs.round_recipe_payload
    workflow: adaptive_state.InitialAdaptiveWorkflow = {
        "recipe_path": str(recipe_path),
        "execution_identity": {field: recipe["execution"][field] for field in adaptive_state.EXECUTION_IDENTITY_FIELDS},
        "root": str(root),
        "external_optimized": True,
        "objective_metric": str(adaptive_state.adaptive_settings(recipe).get("objective_metric") or "test_auroc"),
        "objective_mode": str(adaptive_state.adaptive_settings(recipe).get("objective_mode") or "max"),
    }
    readme_text = _adaptive_readme(workflow)
    adaptive_event: AdaptiveInitEvent = {"round": 0, "recipe_path": str(recipe_path), "round_dir": str(round_dir)}

    staging_dir = None
    cleanup_staging = False
    if not os.path.lexists(round_dir):
        staging_dir = _stage_round(round_dir, recipe, recipe_path, 0, source_config_sha256)
        cleanup_staging = True
    try:
        with plan_publication_lock(round_dir):
            # Recovery decisions must use state reread after the publication lock is acquired.
            if os.path.lexists(workflow_path):
                committed_plan = _validate_initial_round(round_dir, round_recipe_payload, source_config_bytes)
                _validate_adaptive_step_registration(workspace, round_dir, committed_plan)
                if (
                    plan_registration_rows_state(
                        workspace,
                        plan_hparam.hparam_manifest_rows(committed_plan),
                        source="Canonical adaptive round",
                    )
                    != "present"
                ):
                    raise ValueError("Adaptive workflow exists before canonical round registration.")
                adaptive_state.validate_public_initial_workflow(root, workflow, readme_text)
                plan_event = adaptive_state.plan_created_payload(round_dir, committed_plan)
                adaptive_state.validate_initial_event_order(
                    workspace, plan_event, adaptive_event, allow_ready_event=True
                )
                adaptive_state.reconcile_plan_event(workspace, round_dir, committed_plan)
                adaptive_state.reconcile_event(
                    workspace,
                    "adaptive_init",
                    adaptive_event,
                    identity_field="round_dir",
                )
                adaptive_state.validate_initial_event_order(
                    workspace, plan_event, adaptive_event, allow_ready_event=True
                )
                return root
            round_exists = os.path.lexists(round_dir)
            registered_keys = set()
            plan: Mapping[str, Any]
            if round_exists:
                plan = _validate_initial_round(round_dir, round_recipe_payload, source_config_bytes)
                expected_keys = {managed_run_key(run) for run in plan["runs"]}
            else:
                if staging_dir is None:
                    staging_dir = _stage_round(round_dir, recipe, recipe_path, 0, source_config_sha256)
                    cleanup_staging = True
                plan = read_json(staging_dir / "plan.json")
                expected_keys = {managed_run_key(run) for run in plan["runs"]}
            _validate_adaptive_step_registration(workspace, round_dir, plan)
            if (
                plan_registration_rows_state(
                    workspace,
                    plan_hparam.hparam_manifest_rows(plan),
                    source="Canonical adaptive round",
                )
                == "present"
            ):
                registered_keys = expected_keys
            plan_event = adaptive_state.plan_created_payload(round_dir, plan)
            plan_event_exists = adaptive_state.validate_event_history(
                workspace,
                "plan_created",
                plan_event,
                identity_field="plan_dir",
            )
            if plan_event_exists and not registered_keys:
                raise ValueError("Adaptive plan-created event exists before canonical registration.")
            adaptive_state.validate_initial_event_order(workspace, plan_event, adaptive_event, allow_ready_event=False)
            if round_exists:
                if not registered_keys:
                    candidate_dir = staging_dir or _stage_round(
                        round_dir,
                        recipe,
                        recipe_path,
                        0,
                        source_config_sha256,
                    )
                    owns_candidate = staging_dir is None
                    try:
                        if artifacts.plan_tree_sha256(round_dir) != artifacts.plan_tree_sha256(candidate_dir):
                            raise ValueError(
                                f"Uncommitted adaptive round differs from deterministic regeneration: {round_dir}"
                            )
                    finally:
                        if owns_candidate:
                            shutil.rmtree(candidate_dir)
                committed_plan = plan_hparam.commit_hparam_plan(
                    round_dir,
                    emit_event=False,
                    preflight_validated=not registered_keys,
                )
                adaptive_state.ensure_initial_registry(root, round_dir, committed_plan)
            elif staging_dir is not None:
                staged_plan_sha256 = artifacts.plan_tree_sha256(staging_dir)
                placeholder_backup = _publish_staged_round_locked(staging_dir, round_dir)
                try:
                    _validate_initial_round(round_dir, round_recipe_payload, source_config_bytes)
                except BaseException:
                    cleanup_staging = False
                    cleanup_staging = _restore_uncommitted_round(
                        staging_dir,
                        round_dir,
                        placeholder_backup,
                        staged_plan_sha256,
                    )
                    raise
                try:
                    committed_plan = plan_hparam.commit_hparam_plan(
                        round_dir,
                        emit_event=False,
                        preflight_validated=True,
                    )
                except plan_hparam.HparamRegistrationPreflightError:
                    cleanup_staging = False
                    cleanup_staging = _restore_uncommitted_round(
                        staging_dir,
                        round_dir,
                        placeholder_backup,
                        staged_plan_sha256,
                    )
                    raise
                if placeholder_backup is not None:
                    shutil.rmtree(placeholder_backup)
                adaptive_state.ensure_initial_registry(root, round_dir, committed_plan)
            adaptive_state.reconcile_plan_event(workspace, round_dir, committed_plan)
            _ensure_initial_readme(root, readme_text)
            registry_path = adaptive_dir / "run_registry.tsv"
            readme_path = adaptive_dir / "README.md"
            support_snapshots = exp_io.read_managed_files_at(root, [registry_path, readme_path])
            adaptive_state.validate_initial_support_snapshots(root, workflow, readme_text, support_snapshots)
            workflow_text = json.dumps(workflow, indent=2, sort_keys=True) + "\n"
            created_workflow = exp_io.conditional_atomic_replace_text_at(
                workflow_path,
                workflow_text,
                None,
                managed_root=root,
                dependency_path=registry_path,
                expected_dependency_sha256=support_snapshots[str(registry_path)]["sha256"],
                guard_path=readme_path,
                expected_guard_sha256=support_snapshots[str(readme_path)]["sha256"],
            )
            if not created_workflow:
                raise RuntimeError("Adaptive workflow inputs changed before readiness publication.")
            adaptive_state.validate_public_initial_workflow(root, workflow, readme_text)
            adaptive_state.reconcile_event(
                workspace,
                "adaptive_init",
                adaptive_event,
                identity_field="round_dir",
            )
            adaptive_state.validate_initial_event_order(workspace, plan_event, adaptive_event, allow_ready_event=True)
            return root
    finally:
        if cleanup_staging and staging_dir is not None and staging_dir.exists() and not staging_dir.is_symlink():
            shutil.rmtree(staging_dir)


def digest_hparam_run(run_dir: str | Path) -> Path:
    root = canonical_local_experiment_root(run_dir, Path.cwd())
    workflow_root, round_dir, round_index = _resolve_workflow_round(root)
    if (workflow_root / "adaptive" / "workflow.json").exists():
        adaptive_state.read_workflow(workflow_root)
    plan = artifacts.read_hparam_plan(round_dir)
    recipe_value = plan.get("recipe")
    recipe = recipe_value if isinstance(recipe_value, dict) else {}
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
    objective = adaptive_state.workflow_objective(workflow_root, recipe)
    rows = adaptive_evidence.digest_rows(
        round_dir, round_index, workspace, objective, read_run_manifest=read_run_manifest
    )
    write_rows(out, rows)
    write_text(out_dir / f"round_{round_index:03d}.md", _digest_markdown(rows, objective))
    adaptive_state.append_event(workflow_root, "digest", {"round": round_index, "path": str(out), "rows": len(rows)})
    _write_incumbent(workflow_root, rows, objective, round_index)
    return out


def suggest_next_round(workflow_dir: str | Path, *, digest_path: str | Path | None = None) -> Path:
    root = canonical_local_experiment_root(workflow_dir, Path.cwd())
    workflow = adaptive_state.read_workflow(root)
    next_round = adaptive_state.next_round_index(root)
    next_dir = adaptive_state.round_path(root, next_round)
    recipe, _, source_preflight = preflight_plan(
        recipe_path=workflow["recipe_path"], output_dir=next_dir, allow_adaptive_workflow=True
    )
    _require_preflight_pass(source_preflight, "Adaptive source recipe")
    _validate_adaptive_recipe(recipe)
    recipe = _with_workflow_execution(recipe, workflow)
    strategy = adaptive_state.suggest_strategy(recipe)
    if strategy == "agent_proposal" and digest_path is None:
        raise ValueError("agent_proposal requests must be generated by hparam-adaptive-step.")
    digest = Path(digest_path) if digest_path is not None else _latest_digest(root)
    rows = read_rows(digest)
    objective = adaptive_state.workflow_objective(root, recipe)
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Adaptive workflow is not bound to an experiment workspace.")
    if workspace != adaptive_state.workflow_workspace(root):
        raise ValueError("Adaptive source experiment.root differs from the frozen workflow workspace.")
    if strategy == "agent_proposal":
        current_round = adaptive_state.latest_round_index(root)
        round_dir = adaptive_state.round_path(root, current_round)
        if not adaptive_state.round_is_terminal(round_dir, workspace, read_run_manifest=read_run_manifest):
            raise ValueError("agent_proposal requires the current adaptive round to be terminal.")
        return adaptive_handshake.write_agent_proposal_input(
            root,
            workflow,
            recipe,
            digest,
            adaptive_handshake.proposal_digest_rows(root, workspace, read_run_manifest=read_run_manifest),
        )

    ranked = adaptive_evidence.rank_rows(rows, objective)
    out_dir = root / "adaptive" / "suggestions"
    out = out_dir / f"round_{next_round:03d}.yaml"
    exp_io.validate_managed_output_paths(
        workspace,
        [out, out_dir / f"round_{next_round:03d}.md", workspace / "events.jsonl"],
    )
    if not ranked:
        adaptive_state.append_event(root, "suggest_blocked", {"round": next_round, "reason": "no_scored_runs"})
        raise ValueError(f"No digest rows with finite {objective['metric']} are available for suggestion.")
    best = ranked[0]
    source_value = recipe.get("_local_recipe")
    source = source_value if isinstance(source_value, dict) else recipe
    suggested = copy.deepcopy(source)
    suggested_root = experiment_root(suggested)
    if suggested_root is not None:
        suggested["experiment"]["root"] = str(suggested_root)
    suggested["name"] = f"{recipe_name(recipe)}_adaptive_round_{next_round:03d}"
    suggested.setdefault("search", {})["parameters"] = _suggest_parameters(recipe, ranked)
    suggested["search"]["max_runs"] = int(
        adaptive_state.adaptive_settings(recipe).get("round_size") or _hparam_count(suggested)
    )
    if suggested.get("base_recipe"):
        suggested["base_recipe"] = str(_resolve_base_recipe(workflow["recipe_path"], suggested["base_recipe"]))
    candidate_payload = strip_internal_recipe_keys(suggested)
    candidate_bytes = yaml.safe_dump(candidate_payload, sort_keys=False).encode()
    _preflight_candidate(candidate_bytes, next_dir, "Adaptive suggestion")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_bytes(candidate_bytes)
    rationale = _suggestion_rationale(next_round, objective, best, suggested["search"]["parameters"])
    write_text(out_dir / f"round_{next_round:03d}.md", rationale)
    adaptive_state.append_event(
        root, "suggest", {"round": next_round, "path": str(out), "best_run": best.get("run_id")}
    )
    return out


def _agent_suggestion_payload(
    recipe: dict[str, Any], workflow: dict[str, Any], target_round: int, validated: adaptive_proposals.ValidatedProposal
) -> dict[str, Any]:
    source_value = recipe.get("_local_recipe")
    source = source_value if isinstance(source_value, dict) else recipe
    suggested = copy.deepcopy(source)
    suggested_root = experiment_root(suggested)
    if suggested_root is not None:
        suggested["experiment"]["root"] = str(suggested_root)
    suggested["name"] = f"{recipe_name(recipe)}_adaptive_round_{target_round:03d}"
    search = suggested.setdefault("search", {})
    if "configurations" in validated:
        search.pop("parameters", None)
        search["configurations"] = validated["configurations"]
    else:
        search.pop("configurations", None)
        search["parameters"] = validated["parameters"]
    search["max_runs"] = validated["max_runs"]
    decisions = suggested.get("decisions")
    if isinstance(decisions, dict):
        # The accepted proposal owns this round's points and budget.
        decisions.pop("hparam_search_space", None)
        decisions.pop("hparam_budget", None)
    if suggested.get("base_recipe"):
        suggested["base_recipe"] = str(_resolve_base_recipe(workflow["recipe_path"], suggested["base_recipe"]))
    return strip_internal_recipe_keys(suggested)


def _agent_suggestion_rationale(validated: adaptive_proposals.ValidatedProposal) -> str:
    lines = [
        f"# Agent Proposal Round {validated['target_round']:03d}",
        "",
        f"request_id: {validated['request_id']}",
        f"evidence_runs: {', '.join(validated['evidence_run_ids'])}",
        "",
        "## Rationale",
        "",
        validated["rationale"],
        "",
    ]
    if "configurations" in validated:
        lines.extend(["## Configurations", ""])
        lines.extend(f"- point {index}: {point}" for index, point in enumerate(validated["configurations"]))
    else:
        lines.extend(["## Parameters", ""])
        lines.extend(f"- {key}: {value}" for key, value in validated["parameters"].items())
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class _AcceptedProposalArtifacts:
    accepted_path: Path
    accepted_bytes: bytes
    suggestion_path: Path
    suggestion_bytes: bytes
    rationale_path: Path
    rationale_bytes: bytes


def _recover_published_agent_suggestion(
    candidate_payload: dict[str, Any],
    *,
    suggestion: Path,
    workspace: Path,
    next_dir: Path,
) -> dict[str, Any]:
    if os.path.lexists(suggestion):
        try:
            published = exp_io.read_managed_files_at(workspace, [suggestion], allow_invalid_utf8=True)[str(suggestion)]
            published_payload = yaml.safe_load(published["text"])
        except (ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"Existing adaptive suggestion is invalid: {suggestion}") from exc
        published_execution = published_payload.get("execution") if isinstance(published_payload, dict) else None
        published_runtime_commit = (
            published_execution.get("runtime_commit") if isinstance(published_execution, dict) else None
        )
        if not is_full_git_object_id(published_runtime_commit):
            raise ValueError(f"Existing adaptive suggestion has an invalid runtime commit: {suggestion}")
        recovered_payload = copy.deepcopy(candidate_payload)
        recovered_execution = recovered_payload.get("execution")
        if not isinstance(recovered_execution, dict):
            raise ValueError("Agent proposal candidate lacks execution identity.")
        recovered_execution["runtime_commit"] = published_runtime_commit
        recovered_bytes = yaml.safe_dump(recovered_payload, sort_keys=False).encode()
        if published_payload != recovered_payload or published["sha256"] != hashlib.sha256(recovered_bytes).hexdigest():
            raise ValueError(f"Existing adaptive projection differs from the accepted proposal: {suggestion}")
        candidate_payload = recovered_payload
        _preflight_candidate(recovered_bytes, next_dir, "Published agent suggestion")
    return candidate_payload


def _write_exact_bytes(path: Path, content: bytes, *, managed_root: Path) -> None:
    expected_sha256 = hashlib.sha256(content).hexdigest()
    if os.path.lexists(path):
        try:
            snapshot = exp_io.read_managed_files_at(managed_root, [path], allow_invalid_utf8=True)[str(path)]
        except ValueError as exc:
            raise ValueError(f"Existing adaptive projection differs from the accepted proposal: {path}") from exc
        if snapshot["sha256"] != expected_sha256:
            raise ValueError(f"Existing adaptive projection differs from the accepted proposal: {path}")
        return
    created = exp_io.conditional_atomic_replace_text_at(
        path,
        content.decode(),
        None,
        managed_root=managed_root,
    )
    try:
        snapshot = exp_io.read_managed_files_at(managed_root, [path], allow_invalid_utf8=True)[str(path)]
    except ValueError as exc:
        if created:
            raise RuntimeError(f"Adaptive projection changed after publication: {path}") from exc
        raise ValueError(f"Existing adaptive projection differs from the accepted proposal: {path}") from exc
    if snapshot["sha256"] != expected_sha256:
        if created:
            raise RuntimeError(f"Adaptive projection changed after publication: {path}")
        raise ValueError(f"Existing adaptive projection differs from the accepted proposal: {path}")


@overload
def adaptive_step(
    workflow_dir: str | Path,
    *,
    proposal_path: str | Path | None = None,
    execute: Literal[True],
) -> Path: ...


@overload
def adaptive_step(
    workflow_dir: str | Path,
    *,
    proposal_path: str | Path,
    execute: bool = False,
) -> Path: ...


@overload
def adaptive_step(
    workflow_dir: str | Path,
    *,
    proposal_path: str | Path | None = None,
    execute: bool = False,
) -> Path | None: ...


def adaptive_step(
    workflow_dir: str | Path,
    *,
    proposal_path: str | Path | None = None,
    execute: bool = False,
) -> Path | None:
    root = canonical_local_experiment_root(workflow_dir, Path.cwd())
    if not execute:
        return _adaptive_step(root, proposal_path=proposal_path, execute=False)
    workspace = adaptive_state.workflow_workspace(root)
    with plan_registration_lock(workspace):
        if proposal_path is not None:
            applied = adaptive_handshake.applied_agent_proposal(
                root, workspace, proposal_path, read_run_manifest=read_run_manifest
            )
            if applied is not None:
                return applied
        return _adaptive_step(root, proposal_path=proposal_path, execute=True)


def _stage_and_publish_round(
    *,
    root: Path,
    workspace: Path,
    next_dir: Path,
    next_round: int,
    recipe_payload: dict[str, Any],
    recipe_source: str | Path,
    bound_config_sha256: str | None,
    expected_recipe: dict[str, Any] | None,
    expected_base_recipe: dict[str, Any] | None,
    bound_config_path: Path | None,
) -> None:
    try:
        staging_dir = _stage_round(
            next_dir,
            recipe_payload,
            recipe_source,
            next_round,
            bound_config_sha256,
            expected_recipe=expected_recipe,
            expected_base_recipe=expected_base_recipe,
            bound_config_path=bound_config_path,
        )
    except BaseException:
        if (
            bound_config_path is not None
            and next_dir.is_dir()
            and not next_dir.is_symlink()
            and _is_bound_config_placeholder(next_dir, bound_config_path)
            and not bound_config_path.is_symlink()
        ):
            bound_config_path.unlink()
            lock_path = bound_config_path.with_name(f".{bound_config_path.name}.cas.lock")
            if os.path.lexists(lock_path):
                lock_path.unlink()
            next_dir.rmdir()
        raise
    cleanup_staging = True
    try:
        if bound_config_path is not None and file_sha256(bound_config_path) != bound_config_sha256:
            raise ValueError("Agent proposal frozen source config changed during plan materialization.")
        with plan_publication_lock(next_dir):
            staged_plan = read_json(staging_dir / "plan.json")
            _validate_adaptive_step_registration(workspace, next_dir, staged_plan)
            plan_registration_rows_state(
                workspace,
                plan_hparam.hparam_manifest_rows(staged_plan),
                source="Canonical adaptive round",
            )
            staged_plan_sha256 = artifacts.plan_tree_sha256(staging_dir)
            published_now = not (next_dir / "plan.json").exists()
            if published_now:
                placeholder_backup = _publish_staged_round_locked(
                    staging_dir,
                    next_dir,
                    bound_config_path=bound_config_path,
                    bound_config_sha256=bound_config_sha256,
                )
            else:
                if artifacts.plan_tree_sha256(next_dir) != staged_plan_sha256:
                    raise ValueError(f"Published adaptive round differs from deterministic regeneration: {next_dir}")
                placeholder_backup = None
            try:
                committed_plan = plan_hparam.commit_hparam_plan(
                    next_dir,
                    emit_event=False,
                    preflight_validated=True,
                )
            except plan_hparam.HparamRegistrationPreflightError:
                if published_now:
                    cleanup_staging = False
                    cleanup_staging = _restore_uncommitted_round(
                        staging_dir,
                        next_dir,
                        placeholder_backup,
                        staged_plan_sha256,
                    )
                raise
            adaptive_state.reconcile_plan_event(workspace, next_dir, committed_plan)
            if placeholder_backup is not None:
                shutil.rmtree(placeholder_backup)
            adaptive_state.append_registry_rows(root, next_round, next_dir)
            if staging_dir.exists() and not staging_dir.is_symlink():
                shutil.rmtree(staging_dir)
    except BaseException:
        if cleanup_staging and staging_dir.exists() and not staging_dir.is_symlink():
            shutil.rmtree(staging_dir)
        raise


def _preflight_adaptive_source(workflow: dict[str, Any], next_dir: Path) -> dict[str, Any]:
    recipe, _, source_preflight = preflight_plan(
        recipe_path=workflow["recipe_path"], output_dir=next_dir, allow_adaptive_workflow=True
    )
    _require_preflight_pass(source_preflight, "Adaptive source recipe")
    _validate_adaptive_recipe(recipe)
    recipe = _with_workflow_execution(recipe, workflow)
    return recipe


def _publish_proposal_receipt(
    workspace: Path,
    next_round: int,
    validated: adaptive_proposals.ValidatedProposal,
    proposal_file: Path,
    proposal_sha256: str,
    accepted: _AcceptedProposalArtifacts,
) -> AdaptiveProposalAcceptedEvent:
    _write_exact_bytes(accepted.accepted_path, accepted.accepted_bytes, managed_root=workspace)
    _write_exact_bytes(accepted.suggestion_path, accepted.suggestion_bytes, managed_root=workspace)
    _write_exact_bytes(accepted.rationale_path, accepted.rationale_bytes, managed_root=workspace)
    agent_proposal_event: AdaptiveProposalAcceptedEvent = {
        "round": next_round,
        "request_id": validated["request_id"],
        "proposal_path": str(proposal_file),
        "proposal_sha256": proposal_sha256,
        "suggestion": str(accepted.suggestion_path),
        "suggestion_sha256": hashlib.sha256(accepted.suggestion_bytes).hexdigest(),
    }
    adaptive_state.reconcile_event(
        workspace,
        "agent_proposal_accepted",
        agent_proposal_event,
        identity_field="request_id",
    )
    return agent_proposal_event


def _adaptive_step(
    root: Path,
    *,
    proposal_path: str | Path | None,
    execute: bool,
) -> Path | None:
    workflow = adaptive_state.read_workflow(root)
    next_round = adaptive_state.next_round_index(root)
    next_dir = adaptive_state.round_path(root, next_round)
    recipe = _preflight_adaptive_source(workflow, next_dir)
    strategy = adaptive_state.suggest_strategy(recipe)
    if strategy == "agent_proposal" and execute and proposal_path is None:
        raise ValueError("agent_proposal execute requires --proposal.")
    if strategy != "agent_proposal" and proposal_path is not None:
        raise ValueError("--proposal requires adaptive.suggest.strategy=agent_proposal.")
    current_round = adaptive_state.latest_round_index(root)
    round_dir = adaptive_state.round_path(root, current_round)
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Adaptive workflow is not bound to an experiment workspace.")
    if workspace != adaptive_state.workflow_workspace(root):
        raise ValueError("Adaptive source experiment.root differs from the frozen workflow workspace.")
    bound_config_path: Path | None = None
    bound_config_sha256: str | None = None
    round_recipe_payload: dict[str, Any] | None = None
    agent_proposal_event: AdaptiveProposalAcceptedEvent | None = None
    accepted: _AcceptedProposalArtifacts | None = None

    if strategy == "agent_proposal" and proposal_path is None:
        digest = digest_hparam_run(round_dir)
        if not adaptive_state.round_is_terminal(round_dir, workspace, read_run_manifest=read_run_manifest):
            return None
        return suggest_next_round(root, digest_path=digest)

    if proposal_path is not None:
        proposal_file, input_path, validated, proposal_sha256, input_sha256 = adaptive_handshake.load_agent_proposal(
            root, workflow, recipe, workspace, proposal_path, read_run_manifest=read_run_manifest
        )
        candidate_payload = _agent_suggestion_payload(recipe, workflow, next_round, validated)
        next_recipe = _preflight_candidate(
            yaml.safe_dump(candidate_payload, sort_keys=False).encode(), next_dir, "Agent proposal"
        )
        workflow = adaptive_state.read_workflow(root)
        recipe = _preflight_adaptive_source(workflow, next_dir)
        workspace = experiment_root(recipe)
        if workspace is None:
            raise ValueError("Adaptive workflow is not bound to an experiment workspace.")
        if workspace != adaptive_state.workflow_workspace(root):
            raise ValueError("Adaptive source experiment.root differs from the frozen workflow workspace.")
        refreshed_candidate_payload = _agent_suggestion_payload(recipe, workflow, next_round, validated)
        if refreshed_candidate_payload != candidate_payload:
            # Effective semantics can stay constant across offsetting base/local edits, so use the refreshed pair.
            candidate_payload = refreshed_candidate_payload
            next_recipe = _preflight_candidate(
                yaml.safe_dump(candidate_payload, sort_keys=False).encode(), next_dir, "Agent proposal"
            )
        proposal_input, _ = adaptive_handshake.validated_agent_proposal_input(
            root,
            workflow,
            recipe,
            workspace,
            input_path,
            proposal_file,
            expected_sha256=input_sha256,
            read_run_manifest=read_run_manifest,
        )
        if file_sha256(proposal_file) != proposal_sha256:
            raise ValueError("Agent proposal submission changed during validation.")
        if not execute:
            return proposal_file

        if adaptive_state.budget_exhausted(root, recipe, prospective_runs=_round_run_count(next_recipe)):
            raise ValueError("Agent proposal no longer fits the remaining adaptive budget.")
        bound_config_sha256 = proposal_input["input"]["source_config_sha256"]
        bound_config_bytes = adaptive_handshake.bound_source_config_bytes(recipe, bound_config_sha256)
        bound_config_path = next_dir / "source_config.yaml"
        candidate_payload.setdefault("inputs", {})["config"] = str(bound_config_path)
        if file_sha256(proposal_file) != proposal_sha256 or file_sha256(input_path) != input_sha256:
            raise ValueError("Agent proposal submission or input snapshot changed during validation.")
        accepted_path = root / "adaptive" / "proposals" / f"round_{next_round:03d}.json"
        suggestion_dir = root / "adaptive" / "suggestions"
        suggestion = suggestion_dir / f"round_{next_round:03d}.yaml"
        rationale_path = suggestion_dir / f"round_{next_round:03d}.md"
        candidate_payload = _recover_published_agent_suggestion(
            candidate_payload, suggestion=suggestion, workspace=workspace, next_dir=next_dir
        )
        round_recipe_payload = candidate_payload
        exp_io.validate_managed_output_paths(
            workspace,
            [
                input_path,
                proposal_file,
                accepted_path,
                suggestion,
                rationale_path,
                workspace / "events.jsonl",
                root / "adaptive" / "run_registry.tsv",
                bound_config_path,
                next_dir / "round_recipe.yaml",
            ],
        )
        adaptive_state.reject_unresolved_launch_attempts(root, workspace)
        accepted_payload: adaptive_handshake.AcceptedProposalPayload = {
            "schema_version": 1,
            **validated,
            "input_path": str(input_path),
            "input_sha256": input_sha256,
            "proposal_path": str(proposal_file),
            "proposal_sha256": proposal_sha256,
        }
        accepted_bytes = (json.dumps(accepted_payload, indent=2, sort_keys=True) + "\n").encode()
        suggestion_bytes = yaml.safe_dump(candidate_payload, sort_keys=False).encode()
        rationale_bytes = _agent_suggestion_rationale(validated).encode()
        accepted = _AcceptedProposalArtifacts(
            accepted_path=accepted_path,
            accepted_bytes=accepted_bytes,
            suggestion_path=suggestion,
            suggestion_bytes=suggestion_bytes,
            rationale_path=rationale_path,
            rationale_bytes=rationale_bytes,
        )
        _write_exact_bytes(bound_config_path, bound_config_bytes, managed_root=workspace)
        agent_proposal_event = _publish_proposal_receipt(
            workspace, next_round, validated, proposal_file, proposal_sha256, accepted
        )
    else:
        if execute:
            adaptive_state.reject_unresolved_launch_attempts(root, workspace)
        targets = [workspace / "events.jsonl"]
        if execute:
            targets.extend([root / "adaptive" / "run_registry.tsv", next_dir / "round_recipe.yaml"])
        exp_io.validate_managed_output_paths(workspace, targets)
        digest = digest_hparam_run(round_dir)
        suggestion = suggest_next_round(root)
        next_recipe, _, preflight = preflight_plan(
            recipe_path=suggestion, output_dir=next_dir, allow_adaptive_workflow=True
        )
        if preflight.exit_code != 0:
            raise RuntimeError(f"Round {next_round:03d} plan failed preflight with exit code {preflight.exit_code}.")
        next_run_count = _round_run_count(next_recipe)
        # Retiring current runs is allowed only when the complete replacement round fits the remaining budget.
        if execute and adaptive_state.budget_exhausted(root, recipe, prospective_runs=next_run_count):
            adaptive_state.append_event(
                root,
                "adaptive_budget_exhausted",
                {"round": current_round, "digest": str(digest), "suggestion": str(suggestion)},
            )
            return suggestion
        if not execute:
            adaptive_state.append_event(
                root,
                "adaptive_step_dry_run",
                {"round": current_round, "digest": str(digest), "suggestion": str(suggestion)},
            )
            return suggestion
    if bound_config_path is not None and file_sha256(bound_config_path) != bound_config_sha256:
        raise ValueError("Agent proposal frozen source config changed before round materialization.")
    if accepted is not None:
        _write_exact_bytes(accepted.accepted_path, accepted.accepted_bytes, managed_root=workspace)
        _write_exact_bytes(accepted.suggestion_path, accepted.suggestion_bytes, managed_root=workspace)
        _write_exact_bytes(accepted.rationale_path, accepted.rationale_bytes, managed_root=workspace)
    recipe_payload = round_recipe_payload if round_recipe_payload is not None else load_recipe_with_base(suggestion)
    recipe_source = workflow["recipe_path"] if round_recipe_payload is not None else suggestion
    expected_recipe = (
        _materialized_round_recipe(recipe_payload, recipe_source, next_round)
        if round_recipe_payload is not None
        else None
    )
    expected_base_recipe = (
        strip_internal_recipe_keys(copy.deepcopy(recipe["_base_recipe"]))
        if round_recipe_payload is not None and isinstance(recipe.get("_base_recipe"), dict)
        else None
    )
    _stage_and_publish_round(
        root=root,
        workspace=workspace,
        next_dir=next_dir,
        next_round=next_round,
        recipe_payload=recipe_payload,
        recipe_source=recipe_source,
        bound_config_sha256=bound_config_sha256,
        expected_recipe=expected_recipe,
        expected_base_recipe=expected_base_recipe,
        bound_config_path=bound_config_path,
    )
    adaptive_replacement.launch_replacement_round(root, workspace, round_dir, recipe, next_round, next_dir)
    if agent_proposal_event is not None:
        # The replay receipt is terminal: any earlier launch or replacement failure must remain failed.
        adaptive_state.reconcile_event(
            workspace,
            "agent_proposal_execute_completed",
            agent_proposal_event,
            identity_field="request_id",
        )
        adaptive_state.validate_agent_proposal_execute_events(
            read_experiment_events(workspace),
            agent_proposal_event,
            next_dir,
        )
    return suggestion


def adaptive_loop(workflow_dir: str | Path, *, execute: bool = False) -> Path:
    root = canonical_local_experiment_root(workflow_dir, Path.cwd())
    workflow = adaptive_state.read_workflow(root)
    next_dir = adaptive_state.round_path(root, adaptive_state.next_round_index(root))
    recipe, _, source_preflight = preflight_plan(
        recipe_path=workflow["recipe_path"], output_dir=next_dir, allow_adaptive_workflow=True
    )
    _require_preflight_pass(source_preflight, "Adaptive source recipe")
    _validate_adaptive_recipe(recipe)
    recipe = _with_workflow_execution(recipe, workflow)
    if adaptive_state.suggest_strategy(recipe) == "agent_proposal":
        raise ValueError("hparam-adaptive-loop does not support agent_proposal; use the two-phase adaptive step.")
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Adaptive workflow is not bound to an experiment workspace.")
    if workspace != adaptive_state.workflow_workspace(root):
        raise ValueError("Adaptive source experiment.root differs from the frozen workflow workspace.")
    exp_io.validate_managed_output_paths(workspace, [workspace / "events.jsonl"])
    last = root
    while not adaptive_state.budget_exhausted(root, recipe):
        previous_round = adaptive_state.latest_round_index(root)
        step = adaptive_step(root, execute=execute)
        # Only agent_proposal can return None, and this loop rejects that strategy.
        assert step is not None
        last = step
        if not execute:
            break
        if adaptive_state.latest_round_index(root) == previous_round:
            break
        time.sleep(float(adaptive_state.adaptive_settings(recipe).get("poll_seconds") or 60))
    adaptive_state.append_event(root, "adaptive_loop_done", {"path": str(last)})
    return Path(last)


def _validate_adaptive_recipe(recipe: dict[str, Any]) -> None:
    adaptive = adaptive_state.adaptive_settings(recipe)
    if adaptive.get("enabled") is not True:
        raise ValueError("adaptive.enabled must be true for adaptive workflow.")
    objective = str(adaptive.get("objective_metric") or "test_auroc")
    uses_external = objective.startswith("test_") or objective.startswith("external_")
    if uses_external and adaptive.get("test_feedback_for_selection") is not True:
        raise ValueError("adaptive.test_feedback_for_selection=true is required for test/external objectives.")
    # Agent proposals may seed exact points independently of their frozen domain.
    search_value = recipe.get("search")
    search = search_value if isinstance(search_value, dict) else {}
    if "configurations" in search and (
        adaptive_state.suggest_strategy(recipe) != "agent_proposal" or not search.get("parameters")
    ):
        raise ValueError("Adaptive source recipes must declare search.parameters, not search.configurations.")
    if not search.get("parameters"):
        raise ValueError("Adaptive source recipes must declare a non-empty search.parameters mapping.")


def _validate_initial_round(
    round_dir: Path,
    expected_recipe: dict[str, Any],
    source_config_bytes: bytes,
) -> plan_contract.HparamPlan:
    if round_dir.is_symlink() or not round_dir.is_dir():
        raise ValueError(f"Adaptive round 000 is missing or aliased: {round_dir}")
    round_recipe = round_dir / "round_recipe.yaml"
    if round_recipe.is_symlink() or not round_recipe.is_file():
        raise FileNotFoundError(f"Missing frozen adaptive round recipe: {round_recipe}")
    actual_recipe = yaml.safe_load(round_recipe.read_text())
    if adaptive_proposals.canonical_sha256(actual_recipe) != adaptive_proposals.canonical_sha256(expected_recipe):
        raise ValueError(f"Frozen adaptive round recipe differs from the requested initialization: {round_recipe}")
    source_config = round_dir / "config.source.yaml"
    if source_config.is_symlink() or not source_config.is_file():
        raise FileNotFoundError(f"Missing frozen hparam source config: {source_config}")
    if source_config.read_bytes() != source_config_bytes:
        raise ValueError(f"Frozen hparam source config differs from the requested initialization: {source_config}")
    for path in (round_dir / "plan.md", round_dir / "run_all.sh", round_dir / "validation.sh"):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Missing frozen adaptive plan artifact: {path}")
    plan = artifacts.read_hparam_plan(
        round_dir,
        require_workspace_state=False,
        require_adaptive_commit=False,
    )
    plan_recipe_value = plan.get("recipe")
    plan_recipe = plan_recipe_value if isinstance(plan_recipe_value, dict) else {}
    if str(plan_recipe.get("_recipe_path") or "") != str(round_recipe):
        raise ValueError(f"Hparam plan is not bound to its frozen adaptive round recipe: {round_dir}")
    for run in plan["runs"]:
        for path in (Path(run["run_dir"]) / "run.json", Path(run["artifacts"])):
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"Missing frozen adaptive run artifact: {path}")
    return plan


def _stage_round(
    round_dir: Path,
    recipe: dict[str, Any],
    source_recipe_path: str | Path,
    round_index: int,
    source_config_sha256: str | None,
    *,
    expected_recipe: dict[str, Any] | None = None,
    expected_base_recipe: dict[str, Any] | None = None,
    bound_config_path: Path | None = None,
) -> Path:
    staging_dir = round_dir.parent / f".{round_dir.name}.{os.getpid()}.{time.time_ns()}.staging"
    frozen_recipe = expected_recipe or _materialized_round_recipe(recipe, source_recipe_path, round_index)
    if expected_base_recipe is None and isinstance(recipe.get("_base_recipe"), dict):
        expected_base_recipe = strip_internal_recipe_keys(copy.deepcopy(recipe["_base_recipe"]))
    try:
        with TemporaryDirectory(prefix="agent-tools-adaptive-init-") as temp_dir:
            staged_recipe = _write_round_recipe(recipe, source_recipe_path, Path(temp_dir), round_index)
            report = build_plan(
                recipe_path=staged_recipe,
                output_dir=round_dir,
                source_config_sha256=source_config_sha256,
                expected_recipe=frozen_recipe,
                expected_base_recipe=expected_base_recipe,
                staging_dir=staging_dir,
                defer_commit=True,
                registered_recipe_path=round_dir / "round_recipe.yaml",
                allow_adaptive_workflow=True,
            )
            if report.exit_code == 0:
                (staging_dir / "round_recipe.yaml").write_bytes(staged_recipe.read_bytes())
        if report.exit_code != 0:
            raise RuntimeError(f"Round {round_index:03d} plan failed with exit code {report.exit_code}.")
        if bound_config_path is not None:
            if source_config_sha256 is None:
                raise ValueError("Agent proposal lacks a frozen source config SHA-256.")
            if file_sha256(bound_config_path) != source_config_sha256:
                raise ValueError("Agent proposal frozen source config changed during round materialization.")
            (staging_dir / bound_config_path.name).write_bytes(bound_config_path.read_bytes())
        return staging_dir
    except BaseException:
        if staging_dir.exists() and not staging_dir.is_symlink():
            shutil.rmtree(staging_dir)
        raise


def _publish_staged_round_locked(
    staging_dir: Path,
    round_dir: Path,
    *,
    bound_config_path: Path | None = None,
    bound_config_sha256: str | None = None,
) -> Path | None:
    placeholder_backup = None
    if os.path.lexists(round_dir):
        if bound_config_path is None or bound_config_sha256 is None:
            raise ValueError(f"Adaptive round output already exists: {round_dir}")
        if round_dir.is_symlink() or not round_dir.is_dir():
            raise ValueError(f"Adaptive round output is not a physical directory: {round_dir}")
        if (
            not _is_bound_config_placeholder(round_dir, bound_config_path)
            or file_sha256(bound_config_path) != bound_config_sha256
        ):
            raise ValueError(f"Adaptive round output changed before publication: {round_dir}")
        placeholder_backup = round_dir.parent / f".{round_dir.name}.{os.getpid()}.{time.time_ns()}.backup"
        round_dir.replace(placeholder_backup)
    try:
        publish_staged_plan_locked(staging_dir, round_dir, out_preexisted=False)
    except BaseException:
        if placeholder_backup is not None and not os.path.lexists(round_dir):
            placeholder_backup.replace(round_dir)
        raise
    return placeholder_backup


def _is_bound_config_placeholder(round_dir: Path, bound_config_path: Path) -> bool:
    lock_path = bound_config_path.with_name(f".{bound_config_path.name}.cas.lock")
    contents = set(round_dir.iterdir())
    return contents == {bound_config_path} or contents == {bound_config_path, lock_path}


def _restore_uncommitted_round(
    staging_dir: Path,
    round_dir: Path,
    placeholder_backup: Path | None,
    staged_plan_sha256: str,
) -> bool:
    # Registration preflight has not mutated the workspace, so the physical publication can still be undone.
    if os.path.lexists(round_dir) and not os.path.lexists(staging_dir):
        round_dir.replace(staging_dir)
        try:
            unchanged = artifacts.plan_tree_sha256(staging_dir) == staged_plan_sha256
        except (OSError, ValueError):
            unchanged = False
        if not unchanged:
            if not os.path.lexists(round_dir):
                staging_dir.replace(round_dir)
            return False
    elif os.path.lexists(staging_dir):
        return False
    if placeholder_backup is not None and not os.path.lexists(round_dir):
        placeholder_backup.replace(round_dir)
    return True


def _ensure_initial_readme(root: Path, expected: str) -> None:
    readme_path = root / "adaptive" / "README.md"
    if not os.path.lexists(readme_path):
        exp_io.conditional_atomic_replace_text_at(
            readme_path,
            expected,
            None,
            managed_root=root,
        )
    snapshot = exp_io.read_managed_files_at(root, [readme_path])[str(readme_path)]
    if snapshot["text"] != expected:
        raise ValueError(f"Existing adaptive README differs from requested initialization: {readme_path}")


def _write_round_recipe(
    recipe: dict[str, Any], source_recipe_path: str | Path, round_dir: Path, round_index: int
) -> Path:
    round_dir.mkdir(parents=True, exist_ok=True)
    copied = _materialized_round_recipe(recipe, source_recipe_path, round_index)
    target = round_dir / "round_recipe.yaml"
    target.write_text(yaml.safe_dump(copied, sort_keys=False))
    return target


def _materialized_round_recipe(
    recipe: dict[str, Any], source_recipe_path: str | Path, round_index: int
) -> dict[str, Any]:
    source_value = recipe.get("_local_recipe")
    source = source_value if isinstance(source_value, dict) else recipe
    copied = strip_internal_recipe_keys(copy.deepcopy(source))
    if isinstance(recipe.get("execution"), dict):
        execution_value = copied.get("execution")
        execution = dict(execution_value) if isinstance(execution_value, dict) else {}
        execution.update({field: recipe["execution"][field] for field in adaptive_state.EXECUTION_IDENTITY_FIELDS})
        copied["execution"] = execution
    copied_root = experiment_root(copied)
    if copied_root is not None:
        copied["experiment"]["root"] = str(copied_root)
    if copied.get("base_recipe"):
        copied["base_recipe"] = str(_resolve_base_recipe(source_recipe_path, copied["base_recipe"]))
    copied["name"] = f"{recipe_name(recipe)}-round-{round_index:03d}"
    return copied


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


def _workflow_execution_route(workflow: dict[str, Any]) -> dict[str, str]:
    root = Path(str(workflow.get("root") or ""))
    if not root.is_absolute():
        raise ValueError("Adaptive workflow lacks a frozen execution route.")
    initial_plan = artifacts.read_hparam_plan(adaptive_state.round_path(root, 0))
    initial_recipe_value = initial_plan.get("recipe")
    initial_recipe = initial_recipe_value if isinstance(initial_recipe_value, dict) else {}
    initial_execution_value = initial_recipe.get("execution")
    initial_execution = initial_execution_value if isinstance(initial_execution_value, dict) else {}
    return adaptive_state.execution_route(initial_execution)


def _validate_workflow_scientific_contract(recipe: dict[str, Any], workflow: dict[str, Any]) -> None:
    root = Path(str(workflow.get("root") or ""))
    initial_plan = artifacts.read_hparam_plan(adaptive_state.round_path(root, 0))
    initial_recipe_value = initial_plan.get("recipe")
    initial_recipe = initial_recipe_value if isinstance(initial_recipe_value, dict) else {}
    fields = ("task", "variant", "step", "inputs", "evaluation_policy")
    changed = [field for field in fields if recipe.get(field) != initial_recipe.get(field)]
    search_value = recipe.get("search")
    search = search_value if isinstance(search_value, dict) else {}
    initial_search_value = initial_recipe.get("search")
    initial_search = initial_search_value if isinstance(initial_search_value, dict) else {}
    searched_runtime_fields = {
        field.removeprefix("runtime.")
        for field in initial_search.get("parameters", {})
        if isinstance(field, str) and field.startswith("runtime.")
    }
    runtime_value = recipe.get("runtime")
    runtime = runtime_value if isinstance(runtime_value, dict) else {}
    initial_runtime_value = initial_recipe.get("runtime")
    initial_runtime = initial_runtime_value if isinstance(initial_runtime_value, dict) else {}
    adaptive = adaptive_state.adaptive_settings(recipe)
    initial_adaptive = adaptive_state.adaptive_settings(initial_recipe)
    suggest_value = adaptive.get("suggest")
    suggest = suggest_value if isinstance(suggest_value, dict) else {}
    initial_suggest_value = initial_adaptive.get("suggest")
    initial_suggest = initial_suggest_value if isinstance(initial_suggest_value, dict) else {}
    replacement_value = adaptive.get("replacement")
    replacement = replacement_value if isinstance(replacement_value, dict) else {}
    initial_replacement_value = initial_adaptive.get("replacement")
    initial_replacement = initial_replacement_value if isinstance(initial_replacement_value, dict) else {}
    replacement_defaults = {
        "enabled": True,
        "allow_running_stop": False,
        "grace_epochs": None,
        "grace_minutes": None,
        "kill_margin": 0.0,
    }
    frozen_values = {
        "runtime.fixed": (
            {field: value for field, value in runtime.items() if field not in searched_runtime_fields},
            {field: value for field, value in initial_runtime.items() if field not in searched_runtime_fields},
        ),
        "search.parameters": (
            adaptive_proposals.canonical_sha256(search.get("parameters")),
            adaptive_proposals.canonical_sha256(initial_search.get("parameters")),
        ),
        "search.configurations": (
            adaptive_proposals.canonical_sha256(search.get("configurations")),
            adaptive_proposals.canonical_sha256(initial_search.get("configurations")),
        ),
        "adaptive.suggest.bounds": (suggest.get("bounds"), initial_suggest.get("bounds")),
        "adaptive.suggest.strategy": (
            adaptive_state.suggest_strategy(recipe),
            adaptive_state.suggest_strategy(initial_recipe),
        ),
        "adaptive.round_size": (
            int(adaptive.get("round_size") or _hparam_count(recipe)),
            int(initial_adaptive.get("round_size") or _hparam_count(initial_recipe)),
        ),
        "adaptive.max_rounds": (adaptive.get("max_rounds"), initial_adaptive.get("max_rounds")),
        "adaptive.max_runs_total": (
            adaptive.get("max_runs_total"),
            initial_adaptive.get("max_runs_total"),
        ),
        "adaptive.replacement": (
            {**replacement_defaults, **replacement},
            {**replacement_defaults, **initial_replacement},
        ),
    }
    changed.extend(field for field, values in frozen_values.items() if values[0] != values[1])
    if changed:
        raise ValueError(
            f"Adaptive source recipe changed its scientific contract from frozen round 000: {', '.join(changed)}"
        )
    frozen_config = adaptive_state.round_path(root, 0) / "config.source.yaml"
    if adaptive_handshake.source_config_sha256(recipe) != file_sha256(frozen_config):
        raise ValueError("Adaptive source config changed from frozen round 000.")


def _with_workflow_execution(recipe: dict[str, Any], workflow: dict[str, Any]) -> dict[str, Any]:
    frozen = workflow.get("execution_identity")
    if (
        not isinstance(frozen, dict)
        or set(frozen) != set(adaptive_state.EXECUTION_IDENTITY_FIELDS)
        or any(frozen.get(field) in (None, "") for field in adaptive_state.EXECUTION_IDENTITY_FIELDS)
    ):
        raise ValueError("Adaptive workflow lacks frozen execution identity.")
    adaptive = adaptive_state.adaptive_settings(recipe)
    for field, default in (("objective_metric", "test_auroc"), ("objective_mode", "max")):
        if str(adaptive.get(field) or default) != str(workflow.get(field) or default):
            raise ValueError(f"Adaptive source adaptive.{field} differs from the frozen workflow.")
    _validate_workflow_scientific_contract(recipe, workflow)
    current_value = recipe.get("execution")
    current = current_value if isinstance(current_value, dict) else {}
    frozen_route = _workflow_execution_route(workflow)
    current_route = adaptive_state.execution_route(current)
    for field in adaptive_state.EXECUTION_ROUTE_FIELDS:
        if current_route[field] != frozen_route[field]:
            raise ValueError(f"Adaptive source execution.{field} differs from the frozen workflow route.")
    for field in adaptive_state.FROZEN_EXECUTION_IDENTITY_FIELDS:
        value = current.get(field)
        expected = frozen[field]
        if value in (None, "", "ASK_USER"):
            continue
        if value != expected:
            raise ValueError(f"Adaptive source execution.{field} differs from the frozen workflow.")
    execution = dict(current)
    execution.update({field: copy.deepcopy(frozen[field]) for field in adaptive_state.FROZEN_EXECUTION_IDENTITY_FIELDS})
    if execution.get("runtime_commit") in (None, "", "ASK_USER"):
        execution["runtime_commit"] = copy.deepcopy(frozen["runtime_commit"])
    recipe["execution"] = execution
    if isinstance(recipe.get("_local_recipe"), dict):
        local = copy.deepcopy(recipe["_local_recipe"])
        local_execution = dict(local.get("execution")) if isinstance(local.get("execution"), dict) else {}
        local_execution.update(
            {field: copy.deepcopy(execution[field]) for field in adaptive_state.EXECUTION_IDENTITY_FIELDS}
        )
        local["execution"] = local_execution
        recipe["_local_recipe"] = local
    return recipe


def _resolve_workflow_round(path: Path) -> tuple[Path, Path, int]:
    if (path / "adaptive" / "workflow.json").exists():
        idx = adaptive_state.latest_round_index(path)
        return path, adaptive_state.round_path(path, idx), idx
    parts = path.parts
    if "rounds" in parts:
        idx = int(path.name.split("_")[-1])
        workflow_root = Path(*parts[: parts.index("adaptive")]) if "adaptive" in parts else path
        return workflow_root, path, idx
    return path, path, 0


def _digest_markdown(
    rows: list[dict[str, Any]], objective: adaptive_proposals.ProposalObjective | dict[str, str]
) -> str:
    ranked = adaptive_evidence.rank_rows(rows, objective)
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


def _write_incumbent(
    root: Path,
    rows: list[dict[str, Any]],
    objective: adaptive_proposals.ProposalObjective | dict[str, str],
    round_index: int,
) -> None:
    ranked = adaptive_evidence.rank_rows(rows, objective)
    if not ranked:
        return
    best = ranked[0]
    path = root / "adaptive" / "incumbents.tsv"
    incumbents: list[dict[str, Any]] = list(read_rows(path)) if path.exists() else []
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
            "epoch": best.get("epoch", ""),
            "external_optimized": True,
            "selected_at": utc_now(),
        }
    )
    write_rows(path, incumbents)


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
        value = (
            float(best[key])
            if key == "runtime.lr_decay_floor"
            else _coerce_like(best[key], values[0] if values else best[key])
        )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            suggested[key] = _numeric_neighbors(value)
            if key == "runtime.lr_decay_floor":
                suggested[key] = sorted({min(1.0, max(0.0, item)) for item in suggested[key]})
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
    search = recipe.get("search") or {}
    configurations = search.get("configurations") or []
    if configurations:
        return len(configurations)
    params = search.get("parameters") or {}
    count = 1
    for choices in params.values():
        count *= len(choices)
    return count


def _round_run_count(recipe: dict[str, Any]) -> int:
    count = _hparam_count(recipe)
    max_runs = (recipe.get("search") or {}).get("max_runs")
    if max_runs is not None and max_runs != "":
        count = min(count, int(max_runs))
    return count


def _suggestion_rationale(
    round_index: int,
    objective: adaptive_proposals.ProposalObjective | dict[str, str],
    best: dict[str, Any],
    params: dict[str, list[Any]],
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


def _adaptive_readme(workflow: Mapping[str, Any]) -> str:
    return (
        "# Adaptive Hparam Workflow\n\n"
        "This workflow is external-optimized and may use test/external feedback for selection.\n\n"
        f"Objective: `{workflow['objective_metric']}` ({workflow['objective_mode']}).\n"
    )
