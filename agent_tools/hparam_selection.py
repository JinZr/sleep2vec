from __future__ import annotations

from collections.abc import Iterator
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import (
    checkpoint_test_results,
    experiment_io as exp_io,
    experiment_tracking as tracking,
    run_artifacts as artifacts,
    run_evidence as evidence,
)
from .experiment_workspace import (
    SUCCESS_STATUSES,
    TERMINAL_STATUSES,
    append_event,
    experiment_metadata_issues,
    experiment_root,
    managed_run_key,
    managed_run_parameters,
    merge_run_manifest,
    read_managed_yaml_mapping,
    read_run_manifest,
    resolve_run_row,
    validate_existing_experiment_manifest,
    validate_frozen_run_update,
    validate_managed_run_rows,
    validated_run_key,
)
from .manifests import read_json, read_rows, write_rows


@dataclass(frozen=True)
class _HparamSelectionBuild:
    workspace: Path
    step_id: str
    metric: str
    mode: str
    selection_split: str
    out: Path
    selection_report_out: Path
    report_run_keys: set[tuple[str, str] | None]
    step_ranked: list[dict[str, Any]]
    all_ranked: list[dict[str, Any]]
    unscored_rows: list[dict[str, Any]]
    checkpoint_audits_to_write: list[tuple[Path, list[dict[str, Any]]]]
    current_registered: list[tuple[Path, dict[str, Any]]]
    plan_root_by_key: dict[tuple[str, str] | None, Path]


@dataclass(frozen=True)
class _HparamSelectionInputs:
    plan_run_keys: set[tuple[str, str] | None]
    workspace: Path
    step_id: str
    metric: str
    mode: str
    selection_split: str
    out: Path
    selection_report_out: Path
    checkpoint_out: Path
    canonical_rows: list[dict[str, Any]]
    canonical_by_key: dict[tuple[str, str] | None, dict[str, Any]]
    existing_report_steps: list[dict[str, Any]]
    step_runs: list[dict[str, Any]]
    evidence_runs_by_key: dict[tuple[str, str] | None, dict[str, Any]]
    report_run_keys: set[tuple[str, str] | None]
    current_registered: list[tuple[Path, dict[str, Any]]]
    plan_root_by_key: dict[tuple[str, str] | None, Path]


def select_hparam_candidates(
    run_dir: str | Path,
    metric: str | None = None,
    mode: str | None = None,
) -> Path:
    """Select from terminal registered hparam runs and return workspace reports/ranking.csv.

    Requires an active experiment and terminal runs across the current step's
    registered plans. metric/mode may be omitted or match the frozen policy;
    selection_split always comes from that policy. Reads successful-run evidence
    locally or remotely, ranking saved checkpoint test results for test selection.

    Writes ranking/audit tables, canonical selection fields, the selection report
    and its hash bindings, and a deduplicated selection event. Reentry validates
    and preserves frozen selection evidence rather than freely reranking it;
    inconsistent bindings fail. No training or evaluation is launched. Invalid
    prerequisites raise ValueError and unhandled evidence/I/O errors propagate;
    publication spans multiple writes and is not an all-or-nothing transaction."""
    selection = _build_hparam_selection(run_dir, metric=metric, mode=mode)
    return _commit_hparam_selection(selection)


def resolve_hparam_candidates(  # noqa: C901
    run_dir: str | Path,
    candidate_rows: list[dict[str, Any]],
    *,
    top_k: int = 1,
    all_candidates: bool = False,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    if not all_candidates and (type(top_k) is not int or top_k <= 0):
        raise ValueError("top_k must be a positive integer.")
    validate_managed_run_rows(candidate_rows, source="selected candidates", cardinality="one_per_run")
    root = Path(run_dir)
    plan = artifacts.read_hparam_plan(root)
    recipe_value = plan.get("recipe")
    recipe = recipe_value if isinstance(recipe_value, dict) else {}
    evaluation_value = recipe.get("evaluation_policy")
    evaluation = evaluation_value if isinstance(evaluation_value, dict) else {}
    selection_metric = str(evaluation.get("selection_metric") or "")
    selection_mode = str(evaluation.get("selection_mode") or "")
    selection_split = str(evaluation.get("selection_split") or "")
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Selected candidates require a managed experiment workspace.")
    step_value = recipe.get("step")
    step = step_value if isinstance(step_value, dict) else {}
    step_id = str(step.get("id") or "")
    workspace_rows = read_run_manifest(workspace)
    workspace_by_key = {validated_run_key(run): run for run in workspace_rows}

    owner_runs_by_key = {}
    owner_plans_by_key = {}
    for _registered_root, owner_plan in artifacts.iter_registered_hparam_plans(
        workspace,
        step_id,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        selection_split=selection_split,
    ):
        for run in owner_plan["runs"]:
            key = validated_run_key(run)
            if key in owner_runs_by_key:
                raise ValueError(f"Managed run is owned by multiple registered hparam plans: {key[0]} / {key[1]}")
            owner_runs_by_key[key] = run
            owner_plans_by_key[key] = owner_plan

    active_runs = []
    for key, run in owner_runs_by_key.items():
        canonical = workspace_by_key.get(key)
        if canonical is None:
            raise ValueError(f"Managed run is missing from run_manifest.tsv: {run['step_id']} / {run['run_id']}")
        status = str(canonical.get("status") or "")
        if status not in TERMINAL_STATUSES:
            active_runs.append(f"{run['step_id']} / {run['run_id']} ({status})")
    selectors_by_key = {}
    matched_current_step = False
    for row in candidate_rows:
        key = validated_run_key(row)
        managed = workspace_by_key.get(key)
        if managed is None:
            if str(row.get("step_id") or "") == step_id:
                raise ValueError(
                    f"Selected candidate is not managed by a registered hparam plan for the current step: "
                    f"{row['step_id']} / {row['run_id']}"
                )
            raise ValueError(
                f"Selected candidate is not managed by the experiment workspace: {row['step_id']} / {row['run_id']}"
            )
        candidate_parameters = managed_run_parameters(row)
        ownership_evidence = (
            {field: value for field, value in row.items() if field not in candidate_parameters}
            if key in owner_runs_by_key
            else row
        )
        if key not in owner_runs_by_key:
            validate_frozen_run_update(managed, ownership_evidence, require_checkpoint_ownership=True)
        if str(row.get("step_id") or "") != step_id:
            continue
        matched_current_step = True
        run = owner_runs_by_key.get(key)
        if run is None:
            raise ValueError(
                f"Selected candidate is not managed by a registered hparam plan for the current step: "
                f"{key[0]} / {key[1]}"
            )
        if str(managed.get("status") or "") not in SUCCESS_STATUSES:
            continue
        plan_parameters = managed_run_parameters(run)
        extra_parameters = sorted(set(candidate_parameters) - set(plan_parameters))
        if extra_parameters:
            raise ValueError(
                f"Selected candidate defines parameters outside the managed plan: {', '.join(extra_parameters)}"
            )
        for field, value in candidate_parameters.items():
            expected = "" if plan_parameters[field] is None else str(plan_parameters[field])
            actual = "" if value is None else str(value)
            if actual != expected:
                raise ValueError(f"Selected candidate parameter differs from the managed plan: {field}")
        derived = {field: value for field, value in row.items() if field not in candidate_parameters}
        validate_frozen_run_update(
            run,
            derived,
            require_checkpoint_ownership=True,
            allow_execution_identity_fill=True,
        )
        selectors_by_key[key] = (derived, run)
    if active_runs:
        raise ValueError(
            "Hparam candidate resolution requires every registered run to be terminal: " + ", ".join(active_runs)
        )
    if not matched_current_step:
        raise ValueError(f"No selected candidates match the current hparam step: {step_id}")
    if not selectors_by_key:
        raise ValueError("No successful selected candidates remain after canonical status filtering.")

    step_rows = [workspace_by_key[key] for key in owner_runs_by_key]
    canonical_ranked = tracking.validated_hparam_ranking(
        {
            "step_id": step_id,
            "selection": {"metric": selection_metric, "mode": selection_mode, "split": selection_split},
            "rows": step_rows,
        }
    )
    ranking_rows = tracking.hparam_ranking_projection(canonical_ranked) if canonical_ranked is not None else []
    ranking_by_key = {validated_run_key(row): row for row in ranking_rows}
    if selection_split == "test" and not ranking_by_key:
        raise ValueError("Test-selected candidate resolution requires canonical hparam selection.")
    if ranking_by_key:
        ranking_path = workspace / "reports" / "ranking.csv"
        frozen_ranking = read_rows(ranking_path, require_managed_identity=True)
        validate_managed_run_rows(frozen_ranking, source=str(ranking_path), cardinality="one_per_run")
        frozen_by_key = {
            validated_run_key(row): row for row in frozen_ranking if str(row.get("step_id") or "") == step_id
        }
        if set(frozen_by_key) != set(ranking_by_key):
            raise ValueError(f"Frozen hparam ranking candidates differ from canonical selection: {step_id}")
        for key, expected_row in ranking_by_key.items():
            frozen_row = frozen_by_key[key]
            for field, expected in expected_row.items():
                actual = "" if frozen_row.get(field) is None else str(frozen_row.get(field))
                expected_value = "" if expected is None else str(expected)
                if actual != expected_value:
                    raise ValueError(
                        f"Frozen hparam ranking {field} differs from canonical selection: {key[0]} / {key[1]}"
                    )

    resolved = []
    selection_fields = ("rank", "checkpoint_path", "checkpoint_sha256")
    if ranking_by_key:
        for key, ranking_row in ranking_by_key.items():
            selector = selectors_by_key.get(key)
            if selector is None:
                continue
            derived, run = selector
            if (
                selection_split == "test"
                and any(
                    derived.get(field) not in (None, "") for field in ("rank", "checkpoint_path", "checkpoint_sha256")
                )
                and derived.get("checkpoint_sha256") in (None, "")
            ):
                raise ValueError(f"Test-selected candidate is missing frozen checkpoint_sha256: {key[0]} / {key[1]}")
            for field in selection_fields:
                value = derived.get(field)
                if value in (None, ""):
                    continue
                expected = "" if ranking_row.get(field) is None else str(ranking_row.get(field))
                if str(value) != expected:
                    raise ValueError(
                        f"Selected candidate {field} differs from frozen hparam selection: {key[0]} / {key[1]}"
                    )
            resolved.append({**derived, **run, **ranking_row, "status": workspace_by_key[key].get("status", "")})
        if not resolved:
            raise ValueError("No successful selected candidates remain after canonical selection filtering.")
        selected = resolved if all_candidates else resolved[:top_k]
    else:
        ranked_rows = []
        seen_ranks = set()
        for key, (derived, run) in selectors_by_key.items():
            row = {**derived, **run, "status": workspace_by_key[key].get("status", "")}
            rank: Any = row.get("rank")
            try:
                numeric_rank = float(rank)
            except (TypeError, ValueError):
                numeric_rank = math.nan
            if (
                isinstance(rank, bool)
                or not math.isfinite(numeric_rank)
                or not numeric_rank.is_integer()
                or numeric_rank <= 0
            ):
                raise ValueError(
                    f"Selected candidate rank must be a positive integer: {row['step_id']} / {row['run_id']}"
                )
            integer_rank = int(numeric_rank)
            if integer_rank in seen_ranks:
                raise ValueError(f"Selected candidate ranks must be unique: {integer_rank}")
            seen_ranks.add(integer_rank)
            ranked_rows.append((integer_rank, row))
        ranked_rows.sort(key=lambda item: item[0])
        selected = [row for _rank, row in ranked_rows]
        if not all_candidates:
            selected = selected[:top_k]

    if selection_split == "test" and ranking_by_key:
        # Physical I/O follows selection so an unused lower-rank checkpoint cannot block top-k postprocessing.
        for row in selected:
            key = validated_run_key(row)
            canonical = workspace_by_key[key]
            checkpoint_path = str(canonical.get("checkpoint_path") or "")
            checkpoint_sha256 = str(canonical.get("checkpoint_sha256") or "")
            evidence_row = {**owner_runs_by_key[key], **canonical}
            owner_recipe_value = owner_plans_by_key[key].get("recipe")
            owner_recipe = owner_recipe_value if isinstance(owner_recipe_value, dict) else {}
            execution_value = owner_recipe.get("execution")
            execution = execution_value if isinstance(execution_value, dict) else {}
            for field in ("target", "host"):
                if evidence_row.get(field) in (None, ""):
                    evidence_row[field] = execution.get(field, "")
            if evidence.checkpoint_file_sha256(evidence_row, checkpoint_path) != checkpoint_sha256:
                raise ValueError(f"Frozen checkpoint SHA-256 differs: {checkpoint_path}")
    return selected, owner_plans_by_key


def _build_hparam_selection(
    run_dir: str | Path,
    metric: str | None = None,
    mode: str | None = None,
) -> _HparamSelectionBuild:
    inputs = _preflight_hparam_selection(run_dir, metric=metric, mode=mode)
    preserved, existing_checkpoint_ranked = _validate_existing_hparam_selection(inputs)
    return _rank_hparam_selection_candidates(inputs, preserved, existing_checkpoint_ranked)


def _preflight_hparam_selection(
    run_dir: str | Path,
    metric: str | None = None,
    mode: str | None = None,
) -> _HparamSelectionInputs:
    root = Path(run_dir)
    plan = artifacts.read_hparam_plan(root)
    plan_run_keys = {managed_run_key(run) for run in plan["runs"]}
    recipe_value = plan.get("recipe")
    recipe = recipe_value if isinstance(recipe_value, dict) else {}
    evaluation_value = recipe.get("evaluation_policy")
    evaluation = evaluation_value if isinstance(evaluation_value, dict) else {}
    frozen_metric = evaluation.get("selection_metric")
    frozen_mode = evaluation.get("selection_mode")
    selection_split = str(evaluation.get("selection_split") or "")
    if metric not in (None, frozen_metric) or mode not in (None, frozen_mode):
        raise ValueError("hparam-select must use the selection metric and mode frozen in the recipe.")
    metric = str(frozen_metric or "")
    mode = str(frozen_mode or "")
    if not metric or mode not in {"min", "max"}:
        raise ValueError("Recipe must define evaluation_policy.selection_metric and selection_mode.")
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    experiment_manifest_path = workspace / "experiment.yaml"
    experiment_manifest_text = exp_io.read_managed_files_at(workspace, [experiment_manifest_path])[
        str(experiment_manifest_path)
    ]["text"]
    experiment_manifest = read_managed_yaml_mapping(
        experiment_manifest_text, source=f"Managed experiment manifest {experiment_manifest_path}"
    )
    if set(experiment_manifest) != {"experiment"} or not isinstance(experiment_manifest["experiment"], dict):
        raise ValueError("Invalid active experiment owner: experiment.yaml must contain only experiment metadata.")
    active_experiment = experiment_manifest["experiment"]
    if active_experiment.get("status") == "completed":
        raise ValueError(f"Experiment is completed and cannot select hparam candidates: {workspace}")
    active_issues = experiment_metadata_issues({"experiment": active_experiment, "step": recipe["step"]})
    if active_issues:
        raise ValueError("Invalid active experiment owner: " + "; ".join(issue["message"] for issue in active_issues))
    validate_existing_experiment_manifest(experiment_manifest_text, recipe["experiment"], workspace)
    # Keep the canonical snapshot invocation-local; publication rereads it after the first manifest merge.
    canonical_rows = read_run_manifest(workspace)
    canonical_by_key = {managed_run_key(row): row for row in canonical_rows}
    step_id = str((recipe.get("step") or {}).get("id") or "")
    out = workspace / "reports" / "ranking.csv"
    selection_report_out = workspace / "reports" / "hparam_selection.md"
    checkpoint_out = root / "checkpoint_test_ranking.csv"
    # Other selected steps feed the shared report, so reject their canonical drift before touching projections.
    existing_selected_step_ids = sorted(
        {
            str(row["step_id"])
            for row in canonical_rows
            if str(row.get("step_id") or "") != step_id
            and any(row.get(field) not in (None, "") for field in tracking.HPARAM_SELECTION_METADATA_FIELDS)
        }
    )
    existing_report_run_keys: set[tuple[str, str]] = set()
    existing_test_selections = []
    for selected_step_id in existing_selected_step_ids:
        policy_rows = [
            row
            for row in canonical_rows
            if str(row.get("step_id") or "") == selected_step_id
            and any(row.get(field) not in (None, "") for field in tracking.HPARAM_SELECTION_METADATA_FIELDS)
        ]
        policies = {
            (
                str(row.get("metric") or ""),
                str(row.get("selection_mode") or ""),
                str(row.get("selection_split") or ""),
            )
            for row in policy_rows
        }
        if len(policies) != 1:
            raise ValueError(f"Canonical hparam rows disagree on selection policy: {selected_step_id}")
        selected_metric, selected_mode, selected_split = next(iter(policies))
        registered = list(
            artifacts.iter_registered_hparam_plans(
                workspace,
                selected_step_id,
                selection_metric=selected_metric,
                selection_mode=selected_mode,
                selection_split=selected_split,
            )
        )
        if not registered:
            raise ValueError(f"Selected hparam step has no registered plan: {selected_step_id}")
        if selected_split == "test":
            existing_test_selections.append((selected_step_id, selected_metric, selected_mode, registered))
        existing_report_run_keys.update(
            validated_run_key(run)
            for _registered_root, registered_plan in registered
            for run in registered_plan["runs"]
        )
    existing_report_steps = _selection_report_steps([canonical_by_key[key] for key in sorted(existing_report_run_keys)])
    exp_io.validate_managed_output_paths(
        workspace,
        [
            out,
            checkpoint_out,
            workspace / "run_manifest.tsv",
            workspace / "run_matrix.csv",
            workspace / "reports" / "run_matrix.md",
            workspace / "events.jsonl",
            selection_report_out,
        ],
    )
    step_runs = []
    evidence_runs_by_key = {}
    plan_root_by_key = {}
    current_registered = list(
        artifacts.iter_registered_hparam_plans(
            workspace,
            step_id,
            selection_metric=metric,
            selection_mode=mode,
            selection_split=selection_split,
        )
    )
    for registered_root, registered_plan in current_registered:
        registered_recipe_value = registered_plan.get("recipe")
        registered_recipe = registered_recipe_value if isinstance(registered_recipe_value, dict) else {}
        execution_value = registered_recipe.get("execution")
        execution = execution_value if isinstance(execution_value, dict) else {}
        for run in registered_plan["runs"]:
            key = managed_run_key(run)
            step_runs.append(run)
            plan_root_by_key[key] = registered_root
            evidence_run = {**run, **(canonical_by_key.get(key) or {})}
            # A registered plan's frozen execution target remains authoritative before lifecycle rows fill it.
            for field in ("target", "host"):
                if evidence_run.get(field) in (None, ""):
                    evidence_run[field] = execution.get(field, "")
            evidence_runs_by_key[key] = evidence_run
    report_run_keys = existing_report_run_keys | {managed_run_key(run) for run in step_runs}
    # A non-hparam run may share the current step id, so ownership must be checked by the full managed key.
    misowned_selection = sorted(
        f"{row.get('step_id', '')} / {row.get('run_id', '')}"
        for row in canonical_rows
        if managed_run_key(row) not in report_run_keys
        and any(row.get(field) not in (None, "") for field in tracking.HPARAM_SELECTION_METADATA_FIELDS)
    )
    if misowned_selection:
        raise ValueError(
            "Canonical hparam selection metadata is not owned by a registered hparam plan: "
            + ", ".join(misowned_selection)
        )
    for selected_step_id, selected_metric, selected_mode, registered in existing_test_selections:
        _validate_test_selection_events(
            workspace,
            out,
            selected_step_id,
            selected_metric,
            selected_mode,
            registered,
            canonical_rows,
            canonical_by_key,
        )
    current_has_selection = any(
        str(row.get("step_id") or "") == step_id
        and any(row.get(field) not in (None, "") for field in tracking.HPARAM_SELECTION_METADATA_FIELDS)
        for row in canonical_rows
    )
    if selection_split == "test" and current_has_selection:
        _validate_test_selection_events(
            workspace,
            out,
            step_id,
            metric,
            mode,
            current_registered,
            canonical_rows,
            canonical_by_key,
            skip_signature_root=root,
        )
    return _HparamSelectionInputs(
        plan_run_keys=plan_run_keys,
        workspace=workspace,
        step_id=step_id,
        metric=metric,
        mode=mode,
        selection_split=selection_split,
        out=out,
        selection_report_out=selection_report_out,
        checkpoint_out=checkpoint_out,
        canonical_rows=canonical_rows,
        canonical_by_key=canonical_by_key,
        existing_report_steps=existing_report_steps,
        step_runs=step_runs,
        evidence_runs_by_key=evidence_runs_by_key,
        report_run_keys=report_run_keys,
        current_registered=current_registered,
        plan_root_by_key=plan_root_by_key,
    )


def _validate_existing_hparam_selection(
    inputs: _HparamSelectionInputs,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_evidence_runs = [
        inputs.evidence_runs_by_key.get(managed_run_key(row), row) for row in inputs.canonical_rows
    ]
    existing_ranked = read_rows(inputs.out, require_managed_identity=True)
    validate_managed_run_rows(existing_ranked, source=str(inputs.out), cardinality="one_per_run")
    for row in existing_ranked:
        canonical = resolve_run_row(inputs.canonical_rows, row)
        if canonical is None:
            raise ValueError(
                f"Existing ranking row is outside the canonical manifest: "
                f"{row.get('step_id', '')} / {row.get('run_id', '')}"
            )
        validate_frozen_run_update(canonical, row, require_checkpoint_ownership=True)
    tracking.validate_checkpoint_evidence_rows(checkpoint_evidence_runs, existing_ranked)
    if inputs.selection_split == "test":
        _validate_stored_checkpoint_hashes(
            existing_ranked,
            inputs.evidence_runs_by_key,
            step_id=inputs.step_id,
        )
    existing_checkpoint_ranked = []
    if inputs.selection_split == "test":
        existing_checkpoint_ranked = read_rows(inputs.checkpoint_out, require_managed_identity=True)
        validate_managed_run_rows(
            existing_checkpoint_ranked,
            source=str(inputs.checkpoint_out),
            cardinality="many_per_run",
        )
        for row in existing_checkpoint_ranked:
            canonical = resolve_run_row(inputs.canonical_rows, row)
            if canonical is None:
                raise ValueError(
                    f"Existing checkpoint test ranking row is outside the canonical manifest: "
                    f"{row.get('step_id', '')} / {row.get('run_id', '')}"
                )
            validate_frozen_run_update(canonical, row, require_checkpoint_ownership=True)
        tracking.validate_checkpoint_evidence_rows(checkpoint_evidence_runs, existing_checkpoint_ranked)
        _validate_stored_checkpoint_hashes(
            existing_checkpoint_ranked,
            inputs.evidence_runs_by_key,
            step_id=inputs.step_id,
            required=True,
        )
    for row in existing_ranked:
        score = row.get("score")
        score_is_finite = not isinstance(score, bool) and artifacts.float_or_none(score) is not None
        if score_is_finite and row.get("checkpoint_path") in (None, ""):
            raise ValueError(
                f"Existing ranking row with a finite score lacks checkpoint evidence: "
                f"{row.get('step_id', '')} / {row.get('run_id', '')}"
            )
        if row.get("step_id") != inputs.step_id and not score_is_finite:
            raise ValueError(
                f"Existing ranking row for another step has an invalid score: "
                f"{row.get('step_id', '')} / {row.get('run_id', '')}"
            )
    preserved = [row for selected_step in inputs.existing_report_steps for row in selected_step["ranked"]]
    prior_step_rows = [
        row
        for row in existing_ranked
        if row.get("step_id") == inputs.step_id and artifacts.float_or_none(row.get("score")) is not None
    ]
    for row in prior_step_rows:
        if row.get("metric") != inputs.metric:
            raise ValueError("Existing ranking selection metric differs from the current recipe.")
    remaining_prior_keys = {managed_run_key(row) for row in prior_step_rows}
    remaining_prior_keys -= {managed_run_key(run) for run in inputs.step_runs}
    if remaining_prior_keys:
        raise ValueError("Existing ranking rows are not owned by a registered plan for this step.")
    return preserved, existing_checkpoint_ranked


def _rank_hparam_selection_candidates(
    inputs: _HparamSelectionInputs,
    preserved: list[dict[str, Any]],
    existing_checkpoint_ranked: list[dict[str, Any]],
) -> _HparamSelectionBuild:
    rows = []
    unscored_rows = []
    active_runs = []
    for run in inputs.step_runs:
        canonical = resolve_run_row(inputs.canonical_rows, run)
        if canonical is None:
            raise ValueError(f"Managed run is missing from run_manifest.tsv: {run['step_id']} / {run['run_id']}")
        status = str(canonical.get("status") or "")
        artifact_row = inputs.evidence_runs_by_key[managed_run_key(run)]
        if status not in TERMINAL_STATUSES:
            active_runs.append(f"{run['step_id']} / {run['run_id']} ({status})")
            continue
        if status not in SUCCESS_STATUSES:
            unscored_rows.append(
                {
                    "step_id": run["step_id"],
                    "run_id": run["run_id"],
                    "run_name": run["run_name"],
                    "status": status,
                }
            )
            continue
        # The execution target owns runtime evidence; never interpret a same-named manager-local tree for SSH runs.
        observed_artifacts = evidence.runtime_artifacts(artifact_row)
        if observed_artifacts is None:
            if evidence.is_remote_row(artifact_row) and status in {"completed", "finished"}:
                raise ValueError(
                    f"Successful SSH hparam run has unavailable runtime artifacts: "
                    f"{run['step_id']} / {run['run_id']}"
                )
            manifest_path = ""
            manifest: dict[str, Any] = {}
            checkpoint_names: list[str] = []
        else:
            manifest_path, manifest, checkpoint_names = observed_artifacts
        if inputs.selection_split == "test":
            test_rows = _checkpoint_test_result_rows(
                artifact_row,
                inputs.metric,
                manifest_path,
                manifest,
                checkpoint_names,
            )
            for row in test_rows:
                row["status"] = status
            rows.extend(test_rows)
            continue
        score = artifacts.metric_value(manifest, inputs.metric)
        ckpt = artifacts.fixed_checkpoint_path_from_names(
            manifest,
            str(run["checkpoint_dir"]),
            checkpoint_names,
        )
        row = {
            "step_id": run["step_id"],
            "run_id": run["run_id"],
            "run_name": run["run_name"],
            "parameter_summary": run.get("parameter_summary", ""),
            "version": run["version"],
            "metric": inputs.metric,
            "score": score,
            "config": run.get("config"),
            "checkpoint_path": ckpt,
            "run_manifest": str(manifest_path or ""),
            "status": canonical.get("status", ""),
            **managed_run_parameters(run),
        }
        valid_score = not isinstance(score, bool) and artifacts.float_or_none(score) is not None
        if valid_score and ckpt:
            tracking.validate_checkpoint_evidence_rows([artifact_row], [row])
            row["checkpoint_sha256"] = evidence.checkpoint_file_sha256(artifact_row, ckpt)
        if not valid_score or not ckpt:
            unscored_rows.append(row)
        else:
            rows.append(row)
    if active_runs:
        raise ValueError("Hparam selection requires every managed hparam run to be terminal: " + ", ".join(active_runs))
    reverse = inputs.mode == "max"
    if inputs.selection_split == "test":
        rows.sort(
            key=lambda row: (
                str(row.get("step_id") or ""),
                str(row.get("run_id") or ""),
                int(row["epoch"]),
                str(row.get("checkpoint_path") or ""),
            )
        )
    ranked = artifacts.assign_ranks(rows, key="score", reverse=reverse)
    if not ranked:
        raise ValueError(f"No valid {inputs.metric} scores are available for hparam selection.")
    checkpoint_audits_to_write = []
    if inputs.selection_split == "test":
        validate_managed_run_rows(ranked, source="checkpoint test ranking", cardinality="many_per_run")
        audit_paths = [
            registered_root / "checkpoint_test_ranking.csv" for registered_root, _plan in inputs.current_registered
        ]
        exp_io.validate_managed_output_paths(inputs.workspace, audit_paths)
        for registered_root, registered_plan in inputs.current_registered:
            registered_keys = {managed_run_key(run) for run in registered_plan["runs"]}
            plan_rows = [dict(row) for row in rows if managed_run_key(row) in registered_keys]
            expected_audit = artifacts.assign_ranks(plan_rows, key="score", reverse=reverse)
            if not expected_audit:
                continue
            audit_path = registered_root / "checkpoint_test_ranking.csv"
            # Reuse the pre-read audit only when this registered plan owns the invoking plan's exact run set.
            stored_audit = (
                existing_checkpoint_ranked
                if registered_keys == inputs.plan_run_keys
                else read_rows(audit_path, require_managed_identity=True)
            )
            if stored_audit:
                validate_managed_run_rows(stored_audit, source=str(audit_path), cardinality="many_per_run")
                if _checkpoint_ranking_signature(stored_audit) != _checkpoint_ranking_signature(expected_audit):
                    raise ValueError("Frozen checkpoint test ranking differs from current checkpoint test evidence.")
                continue
            plan_has_selection = any(
                any(
                    (inputs.canonical_by_key.get(key) or {}).get(field) not in (None, "")
                    for field in tracking.HPARAM_SELECTION_METADATA_FIELDS
                )
                for key in registered_keys
            )
            if plan_has_selection:
                raise ValueError("Frozen checkpoint test ranking referenced by candidate_selected event is missing.")
            checkpoint_audits_to_write.append((audit_path, expected_audit))
        candidates_by_run: dict[tuple[str, str] | None, list[dict[str, Any]]] = {}
        for row in ranked:
            candidates_by_run.setdefault(managed_run_key(row), []).append(row)
        best_by_run = {}
        for key, candidates in candidates_by_run.items():
            candidate = dict(checkpoint_test_results.best_checkpoint_test_result(candidates, inputs.mode))
            candidate["checkpoint_rank"] = candidate["rank"]
            best_by_run[key] = candidate
        step_ranked = artifacts.assign_ranks(list(best_by_run.values()), key="score", reverse=reverse)
    else:
        step_ranked = ranked
    validate_managed_run_rows(step_ranked, source="current ranking", cardinality="one_per_run")
    canonical_step_ranked = [
        row
        for row in inputs.canonical_rows
        if str(row.get("step_id") or "") == inputs.step_id
        and row.get("selection_task") == "hparam_tune"
        and any(row.get(field) not in (None, "") for field in ("score", "rank", "checkpoint_path", "checkpoint_sha256"))
    ]
    if canonical_step_ranked:
        if any(row.get("checkpoint_sha256") in (None, "") for row in canonical_step_ranked) or not (
            _existing_checkpoint_ranking_is_consistent(canonical_step_ranked, step_ranked)
        ):
            raise ValueError("Frozen canonical hparam selection differs from current runtime evidence.")
    all_ranked = tracking.hparam_ranking_projection([*preserved, *step_ranked])
    return _HparamSelectionBuild(
        workspace=inputs.workspace,
        step_id=inputs.step_id,
        metric=inputs.metric,
        mode=inputs.mode,
        selection_split=inputs.selection_split,
        out=inputs.out,
        selection_report_out=inputs.selection_report_out,
        report_run_keys=inputs.report_run_keys,
        step_ranked=step_ranked,
        all_ranked=all_ranked,
        unscored_rows=unscored_rows,
        checkpoint_audits_to_write=checkpoint_audits_to_write,
        current_registered=inputs.current_registered,
        plan_root_by_key=inputs.plan_root_by_key,
    )


def _commit_hparam_selection(selection: _HparamSelectionBuild) -> Path:
    # Freeze every contributing plan audit before the shared ranking makes the selection finalizable.
    for audit_path, audit_rows in selection.checkpoint_audits_to_write:
        write_rows(audit_path, audit_rows)
    audit_bindings = {}
    if selection.selection_split == "test":
        for registered_root, registered_plan in selection.current_registered:
            audit_path = registered_root / "checkpoint_test_ranking.csv"
            if not audit_path.is_file():
                continue
            binding = {
                "checkpoint_ranking": str(audit_path),
                "checkpoint_ranking_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            }
            for run in registered_plan["runs"]:
                audit_bindings[managed_run_key(run)] = binding
    write_rows(selection.out, selection.all_ranked)
    merge_run_manifest(
        selection.workspace,
        [
            {
                "step_id": row.get("step_id"),
                "run_id": row.get("run_id"),
                "run_name": row.get("run_name"),
                "metric": selection.metric,
                "selection_task": "hparam_tune",
                "selection_mode": selection.mode,
                "selection_split": selection.selection_split,
                "score": row.get("score"),
                "rank": row.get("rank"),
                "checkpoint_path": row.get("checkpoint_path"),
                "checkpoint_sha256": row.get("checkpoint_sha256"),
                "run_manifest": row.get("run_manifest"),
                **audit_bindings.get(managed_run_key(row), {}),
                **(
                    {
                        "epoch": row.get("epoch"),
                        "checkpoint_rank": row.get("checkpoint_rank"),
                        "source": row.get("source"),
                    }
                    if selection.selection_split == "test"
                    else {}
                ),
            }
            for row in selection.step_ranked
        ]
        + [
            {
                "step_id": row.get("step_id"),
                "run_id": row.get("run_id"),
                "run_name": row.get("run_name"),
                "metric": selection.metric,
                "selection_task": "hparam_tune",
                "selection_mode": selection.mode,
                "selection_split": selection.selection_split,
                **{field: "" for field in tracking.HPARAM_SELECTION_RESULT_FIELDS},
                **({"run_manifest": row["run_manifest"]} if row.get("run_manifest") else {}),
            }
            for row in selection.unscored_rows
        ],
    )
    selected_rows = read_run_manifest(selection.workspace)
    report_steps = _selection_report_steps(
        [row for row in selected_rows if managed_run_key(row) in selection.report_run_keys]
    )
    report_text = tracking.hparam_selection_report_text(report_steps, root=selection.workspace)
    exp_io.write_text_at(selection.selection_report_out, report_text)
    report_sha256 = hashlib.sha256(report_text.encode()).hexdigest()
    merge_run_manifest(
        selection.workspace,
        [
            {
                "step_id": row["step_id"],
                "run_id": row["run_id"],
                "run_name": row.get("run_name"),
                "selection_report": str(selection.selection_report_out),
                "selection_report_sha256": report_sha256,
            }
            for row in selected_rows
            if managed_run_key(row) in selection.report_run_keys
        ],
    )
    selected_checkpoint_ranking = (
        selection.plan_root_by_key[managed_run_key(selection.step_ranked[0])] / "checkpoint_test_ranking.csv"
        if selection.selection_split == "test"
        else None
    )
    selection_event = {
        "step_id": selection.step_id,
        "metric": selection.metric,
        "mode": selection.mode,
        "ranking": str(selection.out),
        "selected_run_id": selection.step_ranked[0].get("run_id"),
        "selected_checkpoint_path": selection.step_ranked[0].get("checkpoint_path"),
        "selected_checkpoint_sha256": selection.step_ranked[0].get("checkpoint_sha256"),
        "checkpoint_ranking": str(selected_checkpoint_ranking) if selected_checkpoint_ranking is not None else None,
        "selection_report": str(selection.selection_report_out),
        "selection_report_sha256": report_sha256,
    }
    if not _selection_event_exists(selection.workspace, selection_event):
        append_event(selection.workspace, "candidate_selected", selection_event)
    return selection.out


def _selection_report_steps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_steps = []
    selected_step_ids = sorted(
        {
            str(row["step_id"])
            for row in rows
            if any(row.get(field) not in (None, "") for field in tracking.HPARAM_SELECTION_METADATA_FIELDS)
        }
    )
    for step_id in selected_step_ids:
        step_rows = sorted(
            [row for row in rows if str(row["step_id"]) == step_id],
            key=lambda row: str(row["run_id"]),
        )
        if any(row.get("selection_task") != "hparam_tune" for row in step_rows):
            raise ValueError(f"Canonical hparam selection metadata is only partially materialized for step {step_id}")
        policies = {
            (
                str(row.get("metric") or ""),
                str(row.get("selection_mode") or ""),
                str(row.get("selection_split") or ""),
            )
            for row in step_rows
        }
        if len(policies) != 1:
            raise ValueError(f"Canonical hparam rows disagree on selection policy: {step_id}")
        metric, mode, split = next(iter(policies))
        step = {
            "step_id": step_id,
            "selection": {"metric": metric, "mode": mode, "split": split},
            "rows": step_rows,
        }
        ranked = tracking.validated_hparam_ranking(step)
        if ranked is None:
            raise ValueError(f"Canonical hparam ranks are incomplete for step {step_id}")
        step["ranked"] = ranked
        selected_steps.append(step)
    return selected_steps


def _checkpoint_test_result_rows(
    run: dict[str, Any],
    metric: str,
    manifest_path: str | Path | None,
    manifest: dict[str, Any],
    checkpoint_names: list[str],
) -> list[dict[str, Any]]:
    if not manifest_path:
        raise ValueError(
            f"Completed test-selected run is missing run_manifest.json: {run['step_id']} / {run['run_id']}"
        )
    if manifest.get("test_all_checkpoints_after_fit") is not True:
        raise ValueError(
            f"Completed test-selected run did not enable test_all_checkpoints_after_fit: "
            f"{run['step_id']} / {run['run_id']}"
        )
    results = manifest.get("checkpoint_test_results")
    if not isinstance(results, list):
        raise ValueError(
            f"Completed test-selected run is missing checkpoint_test_results: {run['step_id']} / {run['run_id']}"
        )
    checkpoint_dir = str(run["checkpoint_dir"])
    expected_epochs = checkpoint_test_results.expected_epoch_checkpoints(
        checkpoint_dir,
        checkpoint_names,
        step_id=run["step_id"],
        run_id=run["run_id"],
    )
    tracking.validate_checkpoint_evidence_rows(
        [run],
        [{"step_id": run["step_id"], "run_id": run["run_id"], "checkpoint_path": path} for path in expected_epochs],
    )
    validated_rows = checkpoint_test_results.validate_checkpoint_test_results(
        results,
        metric,
        expected_epochs,
        step_id=run["step_id"],
        run_id=run["run_id"],
    )
    rows = []
    for validated in validated_rows:
        rows.append(
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "run_name": run["run_name"],
                "parameter_summary": run.get("parameter_summary", ""),
                "version": run["version"],
                "metric": metric,
                "score": validated["score"],
                "epoch": validated["epoch"],
                "config": run.get("config"),
                "checkpoint_path": validated["checkpoint_path"],
                "run_manifest": str(manifest_path),
                "source": "checkpoint_test_results",
                **managed_run_parameters(run),
            }
        )
    tracking.validate_checkpoint_evidence_rows([run], rows)
    for row in rows:
        row["checkpoint_sha256"] = evidence.checkpoint_file_sha256(run, row["checkpoint_path"])
    return rows


def _validate_test_selection_events(
    workspace: Path,
    shared_ranking: Path,
    step_id: str,
    metric: str,
    mode: str,
    registered: list[tuple[Path, dict[str, Any]]],
    canonical_rows: list[dict[str, Any]],
    canonical_by_key: dict[tuple[str, str] | None, dict[str, Any]],
    *,
    skip_signature_root: Path | None = None,
) -> None:
    owner_by_key = {}
    runs_by_key = {}
    for registered_root, registered_plan in registered:
        registered_recipe_value = registered_plan.get("recipe")
        registered_recipe = registered_recipe_value if isinstance(registered_recipe_value, dict) else {}
        execution_value = registered_recipe.get("execution")
        execution = execution_value if isinstance(execution_value, dict) else {}
        for run in registered_plan["runs"]:
            key = managed_run_key(run)
            canonical = canonical_by_key.get(key)
            if canonical is None:
                raise ValueError(f"Managed run is missing from run_manifest.tsv: {run['step_id']} / {run['run_id']}")
            evidence_run = {**run, **canonical}
            for field in ("target", "host"):
                if evidence_run.get(field) in (None, ""):
                    evidence_run[field] = execution.get(field, "")
            owner_by_key[key] = registered_root
            runs_by_key[key] = evidence_run

    events = [
        event
        for event in _candidate_selected_events(workspace)
        if event.get("step_id") == step_id and event.get("metric") == metric and event.get("mode") == mode
    ]

    event_bindings = []
    for event in events:
        key = (step_id, str(event.get("selected_run_id") or ""))
        selected_run = runs_by_key.get(key)
        event_plan_root = owner_by_key.get(key)
        selected_path = str(event.get("selected_checkpoint_path") or "")
        selected_sha256 = str(event.get("selected_checkpoint_sha256") or "")
        if not selected_path or not selected_sha256 or selected_run is None or event_plan_root is None:
            raise ValueError(f"Frozen checkpoint SHA-256 differs from candidate_selected event: {selected_path}")
        if evidence.checkpoint_file_sha256(selected_run, selected_path) != selected_sha256:
            raise ValueError(f"Frozen checkpoint SHA-256 differs from candidate_selected event: {selected_path}")
        checkpoint_ranking = event_plan_root / "checkpoint_test_ranking.csv"
        if str(event.get("checkpoint_ranking") or "") != str(checkpoint_ranking):
            raise ValueError(
                "Frozen checkpoint test ranking referenced by candidate_selected event is missing or differs."
            )
        if str(event.get("ranking") or "") != str(shared_ranking):
            raise ValueError("Frozen hparam ranking referenced by candidate_selected event is missing or differs.")
        event_bindings.append((key, selected_path, selected_sha256, event_plan_root))

    audits = {}
    for registered_root, registered_plan in registered:
        plan_runs = [runs_by_key[managed_run_key(run)] for run in registered_plan["runs"]]
        has_selection = any(
            any(run.get(field) not in (None, "") for field in tracking.HPARAM_SELECTION_METADATA_FIELDS)
            for run in plan_runs
        )
        if not has_selection or not any(str(run.get("status") or "") in SUCCESS_STATUSES for run in plan_runs):
            continue
        checkpoint_ranking = registered_root / "checkpoint_test_ranking.csv"
        if not checkpoint_ranking.is_file():
            raise ValueError(
                "Frozen checkpoint test ranking referenced by candidate_selected event is missing or differs."
            )
        exp_io.validate_managed_output_paths(workspace, [checkpoint_ranking])
        stored = read_rows(checkpoint_ranking, require_managed_identity=True)
        validate_managed_run_rows(stored, source=str(checkpoint_ranking), cardinality="many_per_run")
        stored_sha256 = hashlib.sha256(checkpoint_ranking.read_bytes()).hexdigest()
        bindings = {
            (str(run.get("checkpoint_ranking") or ""), str(run.get("checkpoint_ranking_sha256") or ""))
            for run in plan_runs
            if str(run.get("status") or "") in SUCCESS_STATUSES
        }
        if bindings != {(str(checkpoint_ranking), stored_sha256)}:
            raise ValueError("Canonical hparam selection differs from frozen checkpoint test ranking.")
        for row in stored:
            canonical = resolve_run_row(canonical_rows, row)
            if canonical is None:
                raise ValueError(
                    f"Existing checkpoint test ranking row is outside the canonical manifest: "
                    f"{row.get('step_id', '')} / {row.get('run_id', '')}"
                )
            validate_frozen_run_update(canonical, row, require_checkpoint_ownership=True)
        tracking.validate_checkpoint_evidence_rows(plan_runs, stored)
        audits[registered_root] = stored
        if registered_root != skip_signature_root:
            expected = _registered_test_checkpoint_ranking(
                registered_plan,
                runs_by_key,
                metric,
                mode,
            )
            if _checkpoint_ranking_signature(stored) != _checkpoint_ranking_signature(expected):
                raise ValueError("Frozen checkpoint test ranking differs from current checkpoint test evidence.")

    for key, selected_path, selected_sha256, registered_root in event_bindings:
        if registered_root not in audits:
            raise ValueError(
                "Frozen checkpoint test ranking referenced by candidate_selected event is missing or differs."
            )
        if not any(
            managed_run_key(row) == key
            and str(row.get("checkpoint_path") or "") == selected_path
            and str(row.get("checkpoint_sha256") or "") == selected_sha256
            for row in audits[registered_root]
        ):
            raise ValueError(f"Frozen checkpoint SHA-256 differs from candidate_selected event: {selected_path}")


def _registered_test_checkpoint_ranking(
    registered_plan: dict[str, Any],
    runs_by_key: dict[tuple[str, str] | None, dict[str, Any]],
    metric: str,
    mode: str,
) -> list[dict[str, Any]]:
    rows = []
    active_runs = []
    for run in registered_plan["runs"]:
        artifact_row = runs_by_key[managed_run_key(run)]
        status = str(artifact_row.get("status") or "")
        if status not in TERMINAL_STATUSES:
            active_runs.append(f"{run['step_id']} / {run['run_id']} ({status})")
            continue
        if status not in SUCCESS_STATUSES:
            continue
        observed_artifacts = evidence.runtime_artifacts(artifact_row)
        if observed_artifacts is None:
            if evidence.is_remote_row(artifact_row) and status in {"completed", "finished"}:
                raise ValueError(
                    f"Successful SSH hparam run has unavailable runtime artifacts: "
                    f"{run['step_id']} / {run['run_id']}"
                )
            manifest_path = ""
            manifest: dict[str, Any] = {}
            checkpoint_names: list[str] = []
        else:
            manifest_path, manifest, checkpoint_names = observed_artifacts
        test_rows = _checkpoint_test_result_rows(
            artifact_row,
            metric,
            manifest_path,
            manifest,
            checkpoint_names,
        )
        for row in test_rows:
            row["status"] = status
        rows.extend(test_rows)
    if active_runs:
        raise ValueError("Hparam selection requires every managed hparam run to be terminal: " + ", ".join(active_runs))
    rows.sort(
        key=lambda row: (
            str(row.get("step_id") or ""),
            str(row.get("run_id") or ""),
            int(row["epoch"]),
            str(row.get("checkpoint_path") or ""),
        )
    )
    ranked = artifacts.assign_ranks(rows, key="score", reverse=mode == "max")
    validate_managed_run_rows(ranked, source="checkpoint test ranking", cardinality="many_per_run")
    return ranked


def _validate_stored_checkpoint_hashes(
    rows: list[dict[str, Any]],
    runs_by_key: dict[tuple[str, str] | None, dict[str, Any]],
    *,
    step_id: str,
    required: bool = False,
) -> None:
    for row in rows:
        if str(row.get("step_id") or "") != step_id:
            continue
        stored = str(row.get("checkpoint_sha256") or "")
        if not stored:
            if required:
                raise ValueError("Frozen checkpoint test ranking is missing checkpoint_sha256.")
            continue
        run = runs_by_key.get(managed_run_key(row))
        if run is None:
            raise ValueError(
                f"Frozen checkpoint test ranking is outside the canonical manifest: "
                f"{row.get('step_id', '')} / {row.get('run_id', '')}"
            )
        if evidence.checkpoint_file_sha256(run, str(row["checkpoint_path"])) != stored:
            raise ValueError(f"Frozen checkpoint SHA-256 differs: {row['checkpoint_path']}")


def _checkpoint_ranking_signature(rows: list[dict[str, Any]]) -> tuple[tuple[str, ...], ...]:
    fields = (
        "step_id",
        "run_id",
        "epoch",
        "checkpoint_path",
        "checkpoint_sha256",
        "metric",
        "score",
        "rank",
        "run_manifest",
        "source",
        "status",
    )
    return tuple(tuple("" if row.get(field) is None else str(row.get(field)) for field in fields) for row in rows)


def _existing_checkpoint_ranking_is_consistent(
    existing_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> bool:
    current_by_run = {managed_run_key(row): row for row in current_rows}
    existing_keys = {managed_run_key(row) for row in existing_rows}
    fields = ["checkpoint_path", "checkpoint_sha256", "metric", "score"]
    if any(row.get("epoch") not in (None, "") for row in current_rows):
        fields.insert(0, "epoch")
    if existing_keys == set(current_by_run):
        fields.append("rank")
    for existing in existing_rows:
        current = current_by_run.get(managed_run_key(existing))
        if current is None:
            return False
        if any(
            ("" if existing.get(field) is None else str(existing.get(field)))
            != ("" if current.get(field) is None else str(current.get(field)))
            for field in fields
        ):
            return False
    return True


def _selection_event_exists(workspace: Path, payload: dict[str, Any]) -> bool:
    return any(
        all(event.get(key) == value for key, value in payload.items())
        for event in _candidate_selected_events(workspace)
    )


def _candidate_selected_events(workspace: Path) -> Iterator[dict[str, Any]]:
    path = workspace / "events.jsonl"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("event_type") == "candidate_selected":
            yield event


def scan_hparam_checkpoints(run_dir: str | Path, metric: str, mode: str, *, top_k: int | None = None) -> Path:
    """Rebuild local checkpoint_ranking.csv for a validated registered hparam plan.

    Checks existing ranking rows against canonical run identities and frozen
    fields before scanning. For runs with a runtime manifest, prefers usable
    history scores paired with physical epoch checkpoints; if none yield rows,
    tries a manifest score and resolved checkpoint instead. Missing evidence,
    unusable scores/epochs and unresolved or unusable checkpoint candidates are
    excluded rather than necessarily raising. A successful empty or partial
    ranking does not certify all inputs as valid. Malformed runtime manifests,
    history JSON parse failures and unhandled read errors propagate.

    Uses the supplied metric/mode rather than binding them to the selection
    policy: 'max' sorts descending, other modes ascending. top_k slices the
    sorted checkpoint rows before assigning ranks. Overwrites the ranking and
    returns its path, writing a step_id,run_id header when empty. Does not require
    terminal runs, refresh their state or commit the formal selection performed
    by select_hparam_candidates. Scanning/validation errors precede output writes;
    a write failure may leave a partial ranking.
    """
    root = Path(run_dir)
    plan = artifacts.read_hparam_plan(root)
    recipe_value = plan.get("recipe")
    recipe = recipe_value if isinstance(recipe_value, dict) else {}
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    canonical_rows = read_run_manifest(workspace)
    out = root / "checkpoint_ranking.csv"
    exp_io.validate_managed_output_paths(root, [out])
    existing_ranked = read_rows(out, require_managed_identity=True)
    validate_managed_run_rows(existing_ranked, source=str(out), cardinality="many_per_run")
    for row in existing_ranked:
        canonical = resolve_run_row(canonical_rows, row)
        if canonical is None:
            raise ValueError(
                f"Existing checkpoint ranking row is outside the canonical manifest: "
                f"{row.get('step_id', '')} / {row.get('run_id', '')}"
            )
        validate_frozen_run_update(canonical, row, require_checkpoint_ownership=True)
    rows = []
    for run in plan["runs"]:
        manifest_path = artifacts.find_run_manifest(run)
        manifest = read_json(manifest_path) if manifest_path else {}
        rows.extend(_checkpoint_scan_rows(run, metric, manifest_path, manifest))
    reverse = mode == "max"
    ranked = artifacts.assign_ranks(rows, key="score", reverse=reverse, top_k=top_k)
    validate_managed_run_rows(ranked, source="checkpoint ranking", cardinality="many_per_run")
    if ranked:
        write_rows(out, ranked)
    else:
        out.write_text("step_id,run_id\n")
    return out


def _checkpoint_scan_rows(
    run: dict[str, Any],
    metric: str,
    manifest_path: Path | None,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    if manifest_path:
        runtime_dir = Path(str(run["runtime_dir"]))
        checkpoint_dir = Path(str(run["checkpoint_dir"]))
        for epoch, history_score in _history_metric_rows(runtime_dir, metric):
            history_checkpoint = artifacts.checkpoint_for_epoch_in_dir(checkpoint_dir, epoch)
            if history_checkpoint:
                rows.append(
                    {
                        "step_id": run["step_id"],
                        "run_id": run["run_id"],
                        "version": run["version"],
                        "config": run.get("config"),
                        "metric": metric,
                        "score": history_score,
                        "epoch": epoch,
                        "checkpoint_path": str(history_checkpoint),
                        "run_manifest": str(manifest_path),
                        "source": "history",
                        **managed_run_parameters(run),
                    }
                )
    if rows:
        return rows
    score = artifacts.metric_value(manifest, metric)
    checkpoint = artifacts.fixed_checkpoint_path(manifest, Path(str(run["checkpoint_dir"])))
    valid_score = None if isinstance(score, bool) else artifacts.float_or_none(score)
    if valid_score is not None and checkpoint:
        rows.append(
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "version": run["version"],
                "config": run.get("config"),
                "metric": metric,
                "score": valid_score,
                "epoch": manifest.get("epoch") or artifacts.epoch_from_checkpoint_name(Path(checkpoint).name),
                "checkpoint_path": checkpoint,
                "run_manifest": str(manifest_path or ""),
                "source": "manifest",
                **managed_run_parameters(run),
            }
        )
    return rows


def _history_metric_rows(run_dir: Path, metric: str) -> list[tuple[int, float]]:
    by_epoch: dict[int, float] = {}
    for record in _history_records(run_dir):
        if metric not in record:
            continue
        epoch = _history_epoch(record)
        raw_score = record.get(metric)
        score = None if isinstance(raw_score, bool) else artifacts.float_or_none(raw_score)
        if epoch is not None and score is not None:
            by_epoch[epoch] = score
    return sorted(by_epoch.items())


def _history_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("wandb/**/wandb-history*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            if line.strip():
                records.append(json.loads(line))
    for path in sorted(run_dir.glob("wandb/**/wandb-history*.csv")):
        with path.open(newline="") as file_obj:
            records.extend(csv.DictReader(file_obj))
    history = (
        read_json(run_dir / "run_manifest.json").get("history") if (run_dir / "run_manifest.json").exists() else None
    )
    if isinstance(history, list):
        records.extend(row for row in history if isinstance(row, dict))
    return records


def _history_epoch(record: dict[str, Any]) -> int | None:
    for key in ("epoch", "trainer/epoch", "current_epoch", "global_epoch"):
        epoch = artifacts.epoch_number(record.get(key))
        if epoch is not None:
            return epoch
    return None
