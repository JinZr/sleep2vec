from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from agent_tools import experiment_pipeline, experiment_pipeline_cohort_selection as cohort_selection
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


def test_cohort_selection_uses_kind_without_schema_marker(tmp_path: Path):
    root = tmp_path / "workspace"
    spec = _spec(root)

    experiment_pipeline._validate_spec(spec, root, unlock_final_test=True)

    marked = copy.deepcopy(spec)
    marked["schema_version"] = 2
    with pytest.raises(ValueError, match="Unknown spec field.*schema_version"):
        experiment_pipeline._validate_spec(marked, root, unlock_final_test=True)


def test_cohort_selection_frozen_state_has_no_schema_marker(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "experiment.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "id": "unit",
                    "title": "Unit",
                    "objective": "Exercise cohort selection.",
                    "root": str(root),
                    "baseline": {"type": "none"},
                    "status": "active",
                }
            },
            sort_keys=False,
        )
    )
    (root / "run_manifest.tsv").write_text("step_id\trun_id\n")
    spec = _spec(root)
    for job in spec["jobs"]:
        preset = Path(job["inference_preset_path"])
        preset.parent.mkdir(parents=True, exist_ok=True)
        preset.write_bytes(job["id"].encode())
    spec_file = tmp_path / "cohort.yaml"
    source_text = yaml.safe_dump(spec, sort_keys=False)
    spec_file.write_text(source_text)
    pipeline_dir = root / "pipelines" / ".cohort-gate.staging"
    pipeline_dir.mkdir(parents=True)
    monkeypatch.setattr(experiment_pipeline, "_source_plan_snapshots", lambda *_args: [])

    experiment_pipeline._freeze_pipeline(root, pipeline_dir, spec_file, source_text, spec)

    assert "schema_version" not in json.loads((pipeline_dir / "pipeline.json").read_text())


def test_cohort_selection_rejects_cross_role_cohort_and_preset_reuse(tmp_path: Path):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec["jobs"][1]["cohort"] = spec["jobs"][0]["cohort"]
    with pytest.raises(ValueError, match="same cohort"):
        experiment_pipeline._validate_spec(spec, root, unlock_final_test=True)

    spec = _spec(root)
    spec["jobs"][1]["inference_preset_path"] = spec["jobs"][0]["inference_preset_path"]
    with pytest.raises(ValueError, match="same preset"):
        experiment_pipeline._validate_spec(spec, root, unlock_final_test=True)

    spec = _spec(root)
    spec["jobs"][0]["provenance"] = "external"
    with pytest.raises(ValueError, match="must be internal for selection"):
        experiment_pipeline._validate_spec(spec, root, unlock_final_test=True)


def test_cohort_selection_rejects_identical_preset_bytes_across_roles(tmp_path: Path):
    spec = _spec(tmp_path)
    for job in spec["jobs"]:
        preset = Path(job["inference_preset_path"])
        preset.parent.mkdir(parents=True, exist_ok=True)
        preset.write_bytes(b"same cohort payload")

    with pytest.raises(ValueError, match="same preset bytes"):
        experiment_pipeline._preset_snapshots(spec)


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
    spec = _spec(tmp_path)
    source_plan = {
        "recipe": {"step": {"id": "train-age"}},
        "runs": [{"step_id": "train-age", "run_id": "run-001"}],
    }
    rows = [
        {"rank": "1", "run_id": "run-001"},
        {"rank": "2", "run_id": "run-002"},
    ]
    resolver_calls = []
    monkeypatch.setattr(experiment_pipeline.artifacts, "read_hparam_plan", lambda *_args: source_plan)
    monkeypatch.setattr(experiment_pipeline, "select_hparam_candidates", lambda *_args: None)

    def resolve(*args, **kwargs):
        resolver_calls.append((args, kwargs))
        return rows, {}

    monkeypatch.setattr(experiment_pipeline, "resolve_hparam_candidates", resolve)
    monkeypatch.setattr(
        experiment_pipeline,
        "_freeze_checkpoint_candidate",
        lambda _spec, source_id, _source, _plan_dir, _step_id, _recipe, row, _policy: {
            "source_id": source_id,
            "run_id": row["run_id"],
        },
    )

    selected = experiment_pipeline._select_checkpoint_sources(tmp_path, spec)

    assert resolver_calls[0][1] == {"top_k": 2}
    assert [(row["candidate_id"], row["source_rank"]) for row in selected] == [
        ("age-rank-001", 1),
        ("age-rank-002", 2),
    ]


def test_target_gate_chooses_the_best_frozen_internal_rank_among_feasible_candidates(tmp_path: Path):
    spec = _spec(tmp_path)
    candidates = _candidates(tmp_path)
    evidence = _evidence(tmp_path, {"age-rank-001": 4.9, "age-rank-002": 4.1})

    ranking, decision = cohort_selection.rank_candidates(spec, candidates, evidence)

    assert [row["feasible"] for row in ranking] == [True, True]
    assert decision["winner"]["candidate_id"] == "age-rank-001"


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
    monkeypatch.setattr(experiment_pipeline, "_reconcile_pipeline_event", lambda *_args, **_kwargs: None)

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

    (pipeline_dir / "cohort_selection_winner.json").write_text("{}\n")
    with pytest.raises(ValueError, match="decision changed"):
        experiment_pipeline._validate_cohort_decision(
            pipeline_dir,
            spec,
            candidates,
            evidence,
        )


def test_no_winner_stops_before_report_only_materialization(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "cohort-gate"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("pipeline: frozen\n")
    spec = _spec(root)
    candidates = _candidates(tmp_path)
    calls = []
    states = []

    monkeypatch.setattr(experiment_pipeline, "_load_or_freeze_selections", lambda *_args: candidates)
    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(
        experiment_pipeline,
        "_execute_cohort_phase",
        lambda *_args, **kwargs: calls.append(_args[2]["_execution_stage"])
        or {"status": "completed", "jobs": [{"status": "completed"}]},
    )
    monkeypatch.setattr(experiment_pipeline.pipeline_results, "selection_evidence", lambda *_args: [])
    monkeypatch.setattr(
        experiment_pipeline,
        "_load_or_freeze_cohort_decision",
        lambda *_args: ([], {"winner": None, "candidates": []}),
    )
    monkeypatch.setattr(experiment_pipeline, "_update_state", lambda *_args, **kwargs: states.append(kwargs))
    monkeypatch.setattr(experiment_pipeline, "append_event", lambda *_args, **_kwargs: None)

    result = experiment_pipeline._execute_cohort_selection(
        root,
        pipeline_dir,
        spec,
        poll_seconds=0,
        finalize_callback=lambda *_args: pytest.fail("no-winner must not finalize"),
    )

    assert result["status"] == "failed"
    assert calls == ["selection"]
    assert states[-1]["failure"] == "no_feasible_candidate"
    assert not (pipeline_dir / "phases" / "report_only").exists()


def test_report_only_phase_is_built_from_the_frozen_winner(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "cohort-gate"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("pipeline: frozen\n")
    (pipeline_dir / "candidates.json").write_text("{}\n")
    spec = _spec(root)
    candidates = _candidates(tmp_path)
    winner = candidates["age-rank-002"]
    phases = []
    states = []
    finalized = []

    def execute_phase(_root, _pipeline_dir, phase_spec, _candidates, **_kwargs):
        if phase_spec["_execution_stage"] == "report_only":
            assert (pipeline_dir / "cohort_selection_winner.json").is_file()
        phases.append((phase_spec["_execution_stage"], phase_spec["jobs"]))
        return {
            "status": "completed",
            "jobs": [{"job_id": job["id"], "status": "completed"} for job in phase_spec["jobs"]],
        }

    def freeze_decision(*_args):
        (pipeline_dir / "cohort_selection_ranking.csv").write_text("candidate_id\n")
        (pipeline_dir / "cohort_selection_winner.json").write_text("{}\n")
        return [], {"winner": winner, "candidates": list(candidates.values())}

    def write_summary(*_args):
        for name in ("results.csv", "metrics.csv", "summary.md", "final.md"):
            (pipeline_dir / name).write_text(f"{name}\n")
        return pipeline_dir / "final.md"

    monkeypatch.setattr(experiment_pipeline, "_load_or_freeze_selections", lambda *_args: candidates)
    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(experiment_pipeline, "_execute_cohort_phase", execute_phase)
    monkeypatch.setattr(experiment_pipeline.pipeline_results, "selection_evidence", lambda *_args: [])
    monkeypatch.setattr(experiment_pipeline, "_load_or_freeze_cohort_decision", freeze_decision)
    monkeypatch.setattr(experiment_pipeline, "_validate_cohort_decision", lambda *_args: ([], {}))
    monkeypatch.setattr(experiment_pipeline.pipeline_results, "write_cohort_result_summary", write_summary)
    monkeypatch.setattr(experiment_pipeline, "_update_state", lambda *_args, **kwargs: states.append(kwargs))
    monkeypatch.setattr(experiment_pipeline, "append_event", lambda *_args, **_kwargs: None)

    result = experiment_pipeline._execute_cohort_selection(
        root,
        pipeline_dir,
        spec,
        poll_seconds=0,
        finalize_callback=lambda *_args: finalized.append(True),
    )

    assert result["status"] == "completed"
    assert [phase for phase, _jobs in phases] == ["selection", "report_only"]
    assert {job["candidate_id"] for job in phases[0][1]} == set(candidates)
    assert {job["candidate_id"] for job in phases[1][1]} == {winner["candidate_id"]}
    assert states[-1]["status"] == "completed"
    assert finalized == [True]
