"""Hyper-parameter selection.

Preflights the registered plan and ranks candidates on the frozen validation
``selection_split``. Reads frozen artifacts and observed results only; it never
launches work or authors a new search space. Registered-plan integrity
validation recompiles run contracts from the frozen configurations or parameter
product through the hparam adapter and ``plan_hparam``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from . import (
    experiment_io as exp_io,
    experiment_tracking as tracking,
    experiment_workspace,
    hparam_runtime,
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
from .manifests import read_rows, write_rows
from .models import JsonValue


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


@dataclass(frozen=True)
class _HparamSelectionInputs:
    workspace: Path
    step_id: str
    metric: str
    mode: str
    selection_split: str
    out: Path
    selection_report_out: Path
    canonical_rows: list[dict[str, str]]
    canonical_by_key: dict[tuple[str, str] | None, dict[str, str]]
    existing_report_steps: list[tracking.HparamSelectionReportStep]
    step_runs: list[dict[str, Any]]
    evidence_runs_by_key: dict[tuple[str, str] | None, dict[str, Any]]
    report_run_keys: set[tuple[str, str] | None]


def select_hparam_candidates(
    run_dir: str | Path,
    metric: str | None = None,
    mode: str | None = None,
) -> Path:
    """Select from terminal registered hparam runs and return workspace reports/ranking.csv.

    Requires an active experiment and terminal runs across the current step's
    registered plans. metric/mode may be omitted or match the frozen policy;
    selection_split always comes from that policy. Reads successful-run evidence
    locally.

    Writes ranking/audit tables, canonical selection fields, the selection report
    and its hash bindings, and a deduplicated selection event. Reentry validates
    and preserves frozen selection evidence rather than freely reranking it;
    inconsistent bindings fail. No training or evaluation is launched. Invalid
    prerequisites raise ValueError and unhandled evidence/I/O errors propagate;
    publication spans multiple writes and is not an all-or-nothing transaction."""
    selection = _build_hparam_selection(run_dir, metric=metric, mode=mode)
    return _commit_hparam_selection(selection)


def _build_hparam_selection(
    run_dir: str | Path,
    metric: str | None = None,
    mode: str | None = None,
) -> _HparamSelectionBuild:
    inputs = _preflight_hparam_selection(run_dir, metric=metric, mode=mode)
    preserved = _validate_existing_hparam_selection(inputs)
    return _rank_hparam_selection_candidates(inputs, preserved)


def _preflight_hparam_selection(
    run_dir: str | Path,
    metric: str | None = None,
    mode: str | None = None,
) -> _HparamSelectionInputs:
    root = Path(run_dir)
    plan = hparam_runtime.read_hparam_plan_under_run_lock(root)
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
    with experiment_workspace.managed_run_lock(workspace):
        canonical_rows = read_run_manifest(workspace)
    canonical_by_key = {managed_run_key(row): row for row in canonical_rows}
    step_id = str((recipe.get("step") or {}).get("id") or "")
    out = workspace / "reports" / "ranking.csv"
    selection_report_out = workspace / "reports" / "hparam_selection.md"
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
        with experiment_workspace.managed_run_lock(workspace):
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
            selection_report_out,
        ],
    )
    step_runs = []
    evidence_runs_by_key = {}
    with experiment_workspace.managed_run_lock(workspace):
        current_registered = list(
            artifacts.iter_registered_hparam_plans(
                workspace,
                step_id,
                selection_metric=metric,
                selection_mode=mode,
                selection_split=selection_split,
            )
        )
    for _registered_root, registered_plan in current_registered:
        for run in registered_plan["runs"]:
            key = managed_run_key(run)
            step_runs.append(run)
            evidence_runs_by_key[key] = {**run, **(canonical_by_key.get(key) or {})}
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
    return _HparamSelectionInputs(
        workspace=workspace,
        step_id=step_id,
        metric=metric,
        mode=mode,
        selection_split=selection_split,
        out=out,
        selection_report_out=selection_report_out,
        canonical_rows=canonical_rows,
        canonical_by_key=canonical_by_key,
        existing_report_steps=existing_report_steps,
        step_runs=step_runs,
        evidence_runs_by_key=evidence_runs_by_key,
        report_run_keys=report_run_keys,
    )


def _validate_existing_hparam_selection(
    inputs: _HparamSelectionInputs,
) -> list[dict[str, Any]]:
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
    return preserved


def _rank_hparam_selection_candidates(
    inputs: _HparamSelectionInputs,
    preserved: list[dict[str, Any]],
) -> _HparamSelectionBuild:
    rows = []
    unscored_rows = []
    checkpoint_errors = []
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
        manifest_path, manifest, checkpoint_names = evidence.runtime_artifacts(artifact_row)
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
            row["checkpoint_sha256"] = evidence.checkpoint_file_sha256(ckpt)
        if valid_score and not ckpt:
            checkpoint_errors.append(
                f"Valid {inputs.metric} score lacks resolvable checkpoint evidence for "
                f"{run['step_id']} / {run['run_id']}: "
                f"best_model_path={manifest.get('best_model_path')!r}, "
                f"checkpoint_path={manifest.get('checkpoint_path')!r}, epoch={manifest.get('epoch')!r}."
            )
        if not valid_score or not ckpt:
            unscored_rows.append(row)
        else:
            rows.append(row)
    if active_runs:
        raise ValueError("Hparam selection requires every managed hparam run to be terminal: " + ", ".join(active_runs))
    reverse = inputs.mode == "max"
    step_ranked = artifacts.assign_ranks(rows, key="score", reverse=reverse)
    if not step_ranked:
        raise ValueError(
            "; ".join(checkpoint_errors) or f"No valid {inputs.metric} scores are available for hparam selection."
        )
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
    )


def _commit_hparam_selection(selection: _HparamSelectionBuild) -> Path:
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
    with experiment_workspace.managed_run_lock(selection.workspace):
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
    selection_event = {
        "step_id": selection.step_id,
        "metric": selection.metric,
        "mode": selection.mode,
        "ranking": str(selection.out),
        "selected_run_id": selection.step_ranked[0].get("run_id"),
        "selected_checkpoint_path": selection.step_ranked[0].get("checkpoint_path"),
        "selected_checkpoint_sha256": selection.step_ranked[0].get("checkpoint_sha256"),
        "selection_report": str(selection.selection_report_out),
        "selection_report_sha256": report_sha256,
    }
    if not _selection_event_exists(selection.workspace, selection_event):
        append_event(selection.workspace, "candidate_selected", selection_event)
    return selection.out


def _selection_report_steps(rows: list[dict[str, str]]) -> list[tracking.HparamSelectionReportStep]:
    selected_steps: list[tracking.HparamSelectionReportStep] = []
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
        step: tracking.HparamSelectionReportStep = {
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


def _existing_checkpoint_ranking_is_consistent(
    existing_rows: Sequence[Mapping[str, str]],
    current_rows: Sequence[Mapping[str, JsonValue]],
) -> bool:
    current_by_run = {managed_run_key(row): row for row in current_rows}
    existing_keys = {managed_run_key(row) for row in existing_rows}
    fields = ["checkpoint_path", "checkpoint_sha256", "metric", "score"]
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
