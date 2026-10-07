from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from agent_tool_test_helpers import (
    FakePipelineRuntime,
    PipelineInterrupted,
    complete_experiment,
    prepare_pipeline_sources,
    run_pipeline,
)
import pytest
import yaml

from agent_tools import (
    experiment_pipeline,
    experiment_pipeline_attempts as pipeline_attempts,
    experiment_pipeline_cohort_selection as cohort_selection,
    experiment_pipeline_spec as pipeline_spec,
)
from agent_tools.experiment_workspace import file_sha256


def _spec(root: Path) -> dict:
    return {
        "pipeline": {
            "id": "cohort-gate",
            "kind": "cohort_selection",
            "experiment_id": "unit",
            "step": {
                "id": "cohort-evaluate",
                "phase": "evaluate",
                "purpose": "Select one candidate before report-only evaluation.",
            },
            "finalize": True,
        },
        "runtime": {
            "workdir": "/runtime/snapshot",
            "python": "/runtime/python",
            "runtime_commit": "a" * 40,
            "accelerator": "gpu",
            "device": "cuda",
            "precision": "32-true",
            "batch_size": 128,
            "seed": 4523,
        },
        "execution": {
            "gpu_pool": [0, 1],
            "gpus_per_run": 1,
            "max_concurrent": 2,
            "max_attempts": 2,
        },
        "evaluation_policy": {
            "external_test_locked": False,
            "final_test_unlocked": True,
        },
        "checkpoint_policy": {
            "avg_ckpts": 1,
            "require_no_model_averaging": True,
            "forbidden_state_dict_prefixes": ["ema_model.", "running_mean_model."],
            "require_ahi_eval_threshold": True,
        },
        "checkpoint_sources": {
            "age": {
                "plan": str(root / "plans" / "train-age"),
                "selection_metric": "val_mae",
                "selection_mode": "min",
                "task": "age",
                "variant": "sleep2vec2",
                "label_name": "age",
            }
        },
        "candidates": {"kind": "top_k", "count": 2},
        "selector": {
            "strategy": "target_gate",
            "gates": [
                {
                    "job": "selection-internal",
                    "metric": "mae",
                    "mode": "min",
                    "threshold": 5.0,
                }
            ],
            "tie_breaker": "internal_rank",
            "on_no_feasible": "no_winner",
        },
        "jobs": [
            {
                "id": "selection-internal",
                "role": "selection",
                "provenance": "internal",
                "cohort": "internal_holdout",
                "modality": "psg",
                "inference_preset_path": str(root / "presets" / "internal.pickle"),
                "num_workers": 8,
                "task": "age",
                "variant": "sleep2vec2",
                "label_name": "age",
            },
            {
                "id": "report-external",
                "role": "report_only",
                "provenance": "external",
                "cohort": "external_test",
                "modality": "psg",
                "inference_preset_path": str(root / "presets" / "external.pickle"),
                "num_workers": 8,
                "task": "age",
                "variant": "sleep2vec2",
                "label_name": "age",
            },
        ],
    }


def _candidates(tmp_path: Path) -> dict[str, dict]:
    candidates = {}
    for rank, score in ((1, 4.0), (2, 4.2)):
        candidate_id = f"age-rank-{rank:03d}"
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "source_id": "age",
            "source_rank": rank,
            "step_id": "train-age",
            "run_id": f"run-{rank:03d}",
            "selection_metric": "val_mae",
            "variant": "sleep2vec2",
            "label_name": "age",
            "score": score,
            "checkpoint": str(tmp_path / f"rank-{rank}.ckpt"),
            "checkpoint_sha256": str(rank) * 64,
            "config": str(tmp_path / f"rank-{rank}.yaml"),
            "config_sha256": str(rank) * 64,
        }
    return candidates


def _evidence(tmp_path: Path, values: dict[str, float]) -> list[dict]:
    rows = []
    for candidate_id, value in values.items():
        manifest = tmp_path / f"{candidate_id}.json"
        manifest.write_text(json.dumps({"metrics": {"mae": value}}) + "\n")
        rows.append(
            {
                "candidate_id": candidate_id,
                "job_template_id": "selection-internal",
                "cohort": "internal_holdout",
                "metrics": {"mae": value},
                "result_manifest": str(manifest),
                "result_manifest_sha256": file_sha256(manifest),
            }
        )
    return rows


@pytest.mark.parametrize("job", [{"candidate_id": "age-rank-001"}, {"checkpoint_source": "age-rank-001"}])
def test_candidate_for_job_preserves_candidate_identity(tmp_path: Path, job: dict):
    candidates = _candidates(tmp_path)
    candidate = candidates["age-rank-001"]
    candidate["extra_evidence"] = {"notes": ["frozen"]}

    selected = cohort_selection.candidate_for_job(job, candidates)

    assert selected is candidate
    selected["extra_evidence"]["notes"].append("reviewed")
    assert candidate["extra_evidence"]["notes"] == ["frozen", "reviewed"]


def test_cohort_selection_uses_kind_without_schema_marker(tmp_path: Path):
    root = tmp_path / "workspace"
    spec = _spec(root)

    pipeline_spec.validate_spec(spec, root, unlock_final_test=True)

    marked = copy.deepcopy(spec)
    marked["schema_version"] = 2
    with pytest.raises(ValueError, match="Unknown spec field.*schema_version"):
        pipeline_spec.validate_spec(marked, root, unlock_final_test=True)


def test_cohort_selection_frozen_state_has_no_schema_marker(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec, scores=(4.0, 4.2))

    def stop(*_args, **_kwargs):
        raise PipelineInterrupted

    with pytest.raises(PipelineInterrupted):
        run_pipeline(spec_path, FakePipelineRuntime(spec).hooks(inspect_target=stop))

    assert "schema_version" not in json.loads((root / "pipelines" / "cohort-gate" / "pipeline.json").read_text())


def test_cohort_selection_rejects_cross_role_cohort_and_preset_reuse(tmp_path: Path):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec["jobs"][1]["cohort"] = spec["jobs"][0]["cohort"]
    with pytest.raises(ValueError, match="same cohort"):
        pipeline_spec.validate_spec(spec, root, unlock_final_test=True)

    spec = _spec(root)
    spec["jobs"][1]["inference_preset_path"] = spec["jobs"][0]["inference_preset_path"]
    with pytest.raises(ValueError, match="same preset"):
        pipeline_spec.validate_spec(spec, root, unlock_final_test=True)

    spec = _spec(root)
    spec["jobs"][1]["provenance"] = "internal"
    with pytest.raises(ValueError, match="must be external for report_only"):
        pipeline_spec.validate_spec(spec, root, unlock_final_test=True)


def test_cohort_selection_rejects_identical_preset_bytes_across_roles(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec, scores=(4.0, 4.2))
    for job in spec["jobs"]:
        Path(job["inference_preset_path"]).write_bytes(b"same cohort payload")

    with pytest.raises(ValueError, match="same preset bytes"):
        run_pipeline(spec_path, FakePipelineRuntime(spec).hooks())

    assert not (root / "pipelines" / "cohort-gate").exists()


def test_cohort_selection_expands_selection_matrix_then_only_the_winner(tmp_path: Path):
    spec = _spec(tmp_path)
    candidates = _candidates(tmp_path)

    selection = cohort_selection.build_phase_jobs(spec, candidates, role="selection")
    report = cohort_selection.build_phase_jobs(
        spec,
        candidates,
        role="report_only",
        winner_id="age-rank-002",
    )

    assert [(job["candidate_id"], job["job_template_id"]) for job in selection] == [
        ("age-rank-001", "selection-internal"),
        ("age-rank-002", "selection-internal"),
    ]
    assert len(report) == 1
    assert report[0]["candidate_id"] == "age-rank-002"
    assert report[0]["checkpoint_source"] == "age"


def test_cohort_selection_freezes_the_requested_ranked_candidates(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec, scores=(4.0, 4.2, 4.4))
    resolve = experiment_pipeline.resolve_hparam_candidates
    resolver_calls = []

    def record_resolve(*args, **kwargs):
        resolver_calls.append(kwargs)
        return resolve(*args, **kwargs)

    def stop(*_args, **_kwargs):
        raise PipelineInterrupted

    monkeypatch.setattr(experiment_pipeline, "resolve_hparam_candidates", record_resolve)
    with pytest.raises(PipelineInterrupted):
        run_pipeline(spec_path, FakePipelineRuntime(spec).hooks(inspect_target=stop))

    selected = json.loads((root / "pipelines" / "cohort-gate" / "candidates.json").read_text())["candidates"]
    assert resolver_calls == [{"top_k": 2}]
    assert [(row["candidate_id"], row["source_rank"], row["run_id"]) for row in selected] == [
        ("age-rank-001", 1, "run-001"),
        ("age-rank-002", 2, "run-002"),
    ]


def test_target_gate_chooses_the_best_frozen_internal_rank_among_feasible_candidates(tmp_path: Path):
    spec = _spec(tmp_path)
    candidates = _candidates(tmp_path)
    evidence = _evidence(tmp_path, {"age-rank-001": 4.9, "age-rank-002": 4.1})

    ranking, decision = cohort_selection.rank_candidates(spec, candidates, evidence)

    assert [row["feasible"] for row in ranking] == [True, True]
    assert decision["winner"]["candidate_id"] == "age-rank-001"
    winner = next(row for row in decision["candidates"] if row["candidate_id"] == "age-rank-001")
    assert decision["winner"] is winner


def test_target_gate_has_no_hidden_fallback_and_requires_a_complete_matrix(tmp_path: Path):
    spec = _spec(tmp_path)
    candidates = _candidates(tmp_path)

    _ranking, decision = cohort_selection.rank_candidates(
        spec,
        candidates,
        _evidence(tmp_path, {"age-rank-001": 5.1, "age-rank-002": 5.2}),
    )
    assert decision["winner"] is None

    with pytest.raises(ValueError, match="complete frozen candidate-by-cohort matrix"):
        cohort_selection.rank_candidates(
            spec,
            candidates,
            _evidence(tmp_path, {"age-rank-001": 4.9}),
        )


def test_frozen_winner_is_hash_bound_and_tamper_evident(tmp_path: Path, monkeypatch):
    pipeline_dir = tmp_path / "pipelines" / "cohort-gate"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "pipeline.json").write_text("{}\n")
    spec = _spec(tmp_path)
    candidates = _candidates(tmp_path)
    evidence = _evidence(tmp_path, {"age-rank-001": 4.9, "age-rank-002": 4.1})
    monkeypatch.setattr(pipeline_attempts, "reconcile_pipeline_event", lambda *_args, **_kwargs: None)

    _ranking, decision = experiment_pipeline._load_or_freeze_cohort_decision(
        tmp_path,
        pipeline_dir,
        spec,
        candidates,
        evidence,
    )

    state = json.loads((pipeline_dir / "pipeline.json").read_text())
    assert decision["winner"]["candidate_id"] == "age-rank-001"
    assert state["cohort_winner_sha256"] == file_sha256(pipeline_dir / "cohort_selection_winner.json")
    assert b"\r\n" in (pipeline_dir / "cohort_selection_ranking.csv").read_bytes()
    _ranking, reloaded = experiment_pipeline._load_or_freeze_cohort_decision(
        tmp_path,
        pipeline_dir,
        spec,
        candidates,
        evidence,
    )
    assert reloaded == decision
    _ranking, validated = experiment_pipeline._validate_cohort_decision(
        pipeline_dir,
        spec,
        candidates,
        evidence,
    )
    assert validated == decision

    changed_policy = copy.deepcopy(spec)
    changed_policy["selector"]["gates"][0]["strict"] = True
    with pytest.raises(ValueError, match="decision changed"):
        experiment_pipeline._validate_cohort_decision(pipeline_dir, changed_policy, candidates, evidence)

    (pipeline_dir / "cohort_selection_winner.json").write_text("{}\n")
    with pytest.raises(ValueError, match="decision changed"):
        experiment_pipeline._validate_cohort_decision(
            pipeline_dir,
            spec,
            candidates,
            evidence,
        )


def _candidate_metrics(values: dict[str, float]):
    """Report each selection-phase run's MAE by its frozen candidate; report-only runs report 3.0."""

    def metrics(run: dict) -> dict:
        role, candidate_id, _template_id = run["job_id"].split("--")
        return {"mae": values[candidate_id] if role == "selection" else 3.0}

    return metrics


def _selection_manifest(pipeline_dir: Path, candidate_id: str) -> Path:
    attempt_dir = pipeline_dir / "phases" / "selection" / "results" / f"selection--{candidate_id}--selection-internal"
    return attempt_dir / "attempt-001" / "infer" / "run_manifest.json"


@pytest.mark.parametrize("with_report", [True, False])
def test_selection_evidence_is_reread_before_completion(tmp_path: Path, monkeypatch, with_report: bool):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "cohort-gate"
    spec = _spec(root)
    if not with_report:
        spec["jobs"] = spec["jobs"][:1]
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec, scores=(4.0, 4.2))
    runtime = FakePipelineRuntime(spec, metrics=_candidate_metrics({"age-rank-001": 4.9, "age-rank-002": 4.1}))
    finalized = []
    evidence_reads = []
    read_evidence = experiment_pipeline.pipeline_results.selection_evidence

    def change_winner_evidence():
        manifest_path = _selection_manifest(pipeline_dir, "age-rank-001")
        manifest = json.loads(manifest_path.read_text())
        manifest_path.write_text(json.dumps({**manifest, "metrics": {"mae": 5.9}}) + "\n")

    def current_evidence(*args):
        evidence_reads.append(True)
        if not with_report and len(evidence_reads) == 2:
            change_winner_evidence()
        return read_evidence(*args)

    def launch_and_change_evidence(*args, **kwargs):
        result = runtime.launch_runs(*args, **kwargs)
        if args[2][0]["job_id"].startswith("report_only--"):
            change_winner_evidence()
        return result

    monkeypatch.setattr(experiment_pipeline.pipeline_results, "selection_evidence", current_evidence)
    monkeypatch.setattr(
        experiment_pipeline.pipeline_results,
        "write_cohort_result_summary",
        lambda *_args: pytest.fail("changed selection evidence must block final reporting"),
    )

    with pytest.raises(ValueError, match="Frozen cohort-selection decision changed"):
        run_pipeline(spec_path, runtime.hooks(launch_runs=launch_and_change_evidence), finalized=finalized)

    assert finalized == []
    assert len(evidence_reads) == 2
    assert json.loads((pipeline_dir / "pipeline.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("with_report", [True, False])
def test_no_winner_stops_before_report_only_materialization(tmp_path: Path, monkeypatch, with_report: bool):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "cohort-gate"
    spec = _spec(root)
    spec["jobs"][0]["provenance"] = "external"
    spec["selector"]["gates"][0]["strict"] = True
    if not with_report:
        spec["jobs"] = spec["jobs"][:1]
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec, scores=(4.0, 4.2))
    runtime = FakePipelineRuntime(spec, metrics=_candidate_metrics({"age-rank-001": 5.0, "age-rank-002": 5.1}))
    finalized = []

    result = run_pipeline(spec_path, runtime.hooks(), finalized=finalized)

    assert result["status"] == "failed"
    assert finalized == []
    assert [run_ids for effect, run_ids in runtime.calls if effect == "launch"] == [["run-000", "run-001"]]
    assert json.loads((pipeline_dir / "pipeline.json").read_text())["failure"] == "no_feasible_candidate"
    report = (pipeline_dir / "selection_failure.md").read_text()
    assert "| age-rank-001 | internal_holdout | external | mae | 5.0 | `<` | 5.0 | False |" in report
    assert not (pipeline_dir / "phases" / "report_only").exists()


@pytest.mark.parametrize("with_report", [True, False])
def test_report_only_phase_is_built_from_the_frozen_winner(tmp_path: Path, monkeypatch, with_report: bool):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "cohort-gate"
    spec = _spec(root)
    if not with_report:
        spec["jobs"] = spec["jobs"][:1]
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec, scores=(4.0, 4.2))
    runtime = FakePipelineRuntime(spec, metrics=_candidate_metrics({"age-rank-001": 5.5, "age-rank-002": 4.1}))
    launched_jobs = []
    # Completion events and finalization calls are recorded in one ordered history.
    completion_order: list = []
    append_event = pipeline_attempts.append_event

    def launch(*args, **kwargs):
        job_ids = [run["job_id"] for run in args[2]]
        if job_ids[0].startswith("report_only--"):
            assert (pipeline_dir / "cohort_selection_winner.json").is_file()
        launched_jobs.append(job_ids)
        return runtime.launch_runs(*args, **kwargs)

    def append(root_path, event_type, payload):
        if event_type == "pipeline_completed":
            completion_order.append(event_type)
        append_event(root_path, event_type, payload)

    monkeypatch.setattr(pipeline_attempts, "append_event", append)

    result = run_pipeline(spec_path, runtime.hooks(launch_runs=launch), finalized=completion_order)

    assert result["status"] == "completed"
    selection_jobs = [f"selection--age-rank-00{rank}--selection-internal" for rank in (1, 2)]
    report_jobs = [["report_only--age-rank-002--report-external"]] if with_report else []
    assert launched_jobs == [selection_jobs, *report_jobs]
    if not with_report:
        assert not (pipeline_dir / "phases" / "report_only").exists()
    assert json.loads((pipeline_dir / "pipeline.json").read_text())["status"] == "completed"
    assert completion_order == ["pipeline_completed", (root, pipeline_dir / "final.md")]


@pytest.mark.parametrize("with_report", [True, False])
def test_completed_cohort_pipeline_reconciles_completion_event_before_finalization(
    tmp_path: Path, monkeypatch, with_report: bool
):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "cohort-gate"
    report = pipeline_dir / "final.md"
    spec = _spec(root)
    if not with_report:
        spec["jobs"] = spec["jobs"][:1]
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec, scores=(4.0, 4.2))
    runtime = FakePipelineRuntime(spec, metrics=_candidate_metrics({"age-rank-001": 4.9, "age-rank-002": 4.1}))
    # Completion events and finalization calls are recorded in one ordered history.
    order: list = []
    append_event = pipeline_attempts.append_event
    interrupted = {"value": True}

    def append(root_path, event_type, payload):
        if event_type == "pipeline_completed":
            if interrupted["value"]:
                raise RuntimeError("event append interrupted")
            order.append(event_type)
        append_event(root_path, event_type, payload)

    monkeypatch.setattr(pipeline_attempts, "append_event", append)
    with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        run_pipeline(spec_path, runtime.hooks(), finalized=order)
    assert json.loads((pipeline_dir / "pipeline.json").read_text())["status"] == "completed"
    assert order == []
    calls = list(runtime.calls)

    interrupted["value"] = False
    result = run_pipeline(spec_path, runtime.hooks(), resume=True, finalized=order)

    assert result["status"] == "completed"
    assert order == ["pipeline_completed", (root, report)]
    assert runtime.calls == calls
    events_before = (root / "events.jsonl").read_bytes()

    complete_experiment(root)
    order.clear()
    result = run_pipeline(spec_path, runtime.hooks(), resume=True, finalized=order)

    assert result["status"] == "completed"
    assert order == []
    assert (root / "events.jsonl").read_bytes() == events_before

    report.write_text("changed after completion\n")
    with pytest.raises(ValueError, match="Completed pipeline artifact changed"):
        run_pipeline(spec_path, runtime.hooks(), resume=True, finalized=order)
    assert order == []
    assert (root / "events.jsonl").read_bytes() == events_before


@pytest.mark.parametrize("with_report", [True, False])
def test_external_selection_is_accepted_with_explicit_test_unlock(tmp_path: Path, with_report: bool):
    spec = _spec(tmp_path)
    spec["jobs"][0]["provenance"] = "external"
    if not with_report:
        spec["jobs"] = spec["jobs"][:1]
    pipeline_spec.validate_spec(spec, tmp_path, unlock_final_test=True)
    with pytest.raises(ValueError, match="unlock"):
        pipeline_spec.validate_spec(spec, tmp_path, unlock_final_test=False)


def test_cohort_selection_requires_selection_jobs(tmp_path: Path):
    spec = _spec(tmp_path)
    spec["jobs"] = spec["jobs"][1:]
    with pytest.raises(ValueError, match="selection"):
        pipeline_spec.validate_spec(spec, tmp_path, unlock_final_test=True)


@pytest.mark.parametrize("strict", [None, False, True])
@pytest.mark.parametrize("mode", ["min", "max"])
def test_target_gate_comparison_uses_raw_values_and_explicit_strictness(tmp_path: Path, strict, mode: str):
    spec = _spec(tmp_path)
    gate = spec["selector"]["gates"][0]
    gate["mode"] = mode
    if strict is not None:
        gate["strict"] = strict
    pipeline_spec.validate_spec(spec, tmp_path, unlock_final_test=True)
    adjacent = math.nextafter(5.0, -math.inf if mode == "min" else math.inf)
    ranking, decision = cohort_selection.rank_candidates(
        spec, _candidates(tmp_path), _evidence(tmp_path, {"age-rank-001": 5.0, "age-rank-002": adjacent})
    )
    assert [row["feasible"] for row in ranking] == [not strict, True]
    assert decision["winner"]["candidate_id"] == ("age-rank-002" if strict else "age-rank-001")
    assert decision["selector"] == spec["selector"]


@pytest.mark.parametrize("strict", ["true", "false", 0, 1, None])
def test_gate_strict_rejects_nonboolean_values(tmp_path: Path, strict):
    spec = _spec(tmp_path)
    spec["selector"]["gates"][0]["strict"] = strict
    with pytest.raises(ValueError, match="strict.*bool"):
        pipeline_spec.validate_spec(spec, tmp_path, unlock_final_test=True)


@pytest.mark.parametrize("with_report", [True, False])
def test_real_cohort_summary_preserves_selection_provenance_and_gate_outcomes(tmp_path: Path, with_report: bool):
    results = experiment_pipeline.pipeline_results
    spec = _spec(tmp_path)
    spec["jobs"][0]["provenance"] = "external"
    spec["selector"]["gates"][0]["strict"] = True
    if not with_report:
        spec["jobs"] = spec["jobs"][:1]
    candidates = _candidates(tmp_path)
    for candidate in candidates.values():
        candidate["selection_metric"] = "test_mae"
    evidence = _evidence(tmp_path, {"age-rank-001": 5.0, "age-rank-002": 4.9})
    for row in evidence:
        manifest = Path(row["result_manifest"])
        payload = json.loads(manifest.read_text())
        payload["prediction_row_count"] = 10
        manifest.write_text(json.dumps(payload))
    _ranking, decision = cohort_selection.rank_candidates(spec, candidates, evidence)
    winner = decision["winner"]
    (tmp_path / "cohort_selection_winner.json").write_text(json.dumps(decision))
    source_recipe = tmp_path / "resolved_recipe.yaml"
    source_recipe.write_text(yaml.safe_dump({"evaluation_policy": {"selection_split": "test"}}))
    (tmp_path / "pipeline.json").write_text(
        json.dumps({"source_plans": [{"resolved_recipe_path": str(source_recipe)}]})
    )
    phases = {}
    for role in ("selection", "report_only"):
        jobs = cohort_selection.build_phase_jobs(spec, candidates, role=role, winner_id=winner["candidate_id"])
        phases[role] = {**spec, "jobs": jobs}
        if not jobs:
            continue
        attempts = [
            {
                "job_id": job["id"],
                "step_id": "evaluate",
                "run_id": f"run-{index:03d}",
                "attempt": 1,
                "verified": True,
                "runtime_commit": "a" * 40,
                "result_root": str(tmp_path),
                "result_manifest": str(tmp_path / f"{job['candidate_id']}.json"),
            }
            for index, job in enumerate(jobs, 1)
        ]
        results.write_rows_atomic(tmp_path / "phases" / role / "jobs.tsv", attempts)
    report = results.write_cohort_result_summary(
        tmp_path, spec, candidates, winner, phases["selection"], phases["report_only"]
    )
    text = report.read_text()
    assert "internal-test + external-test selected" in text
    assert "Winner: `age-rank-002` (internal rank 2)" in text
    assert "| age-rank-001 | 1 | test | test_mae | 4.0 |" in text
    assert "| age-rank-001 | internal_holdout | external | mae | 5.0 | `<` | 5.0 | False |" in text
    assert "| age-rank-002 | internal_holdout | external | mae | 4.9 | `<` | 5.0 | True |" in text
    assert "| internal_holdout | external | mae |" in text
    assert ("No report-only cohort" in text) is not with_report
    assert (tmp_path / "summary.md").read_text() == text
    rows = results.read_rows(tmp_path / "results.csv")
    assert len(rows) == (3 if with_report else 2)
    assert all(row["selection_split"] == "test" for row in rows)
    if not with_report:
        assert not (tmp_path / "phases" / "report_only").exists()
