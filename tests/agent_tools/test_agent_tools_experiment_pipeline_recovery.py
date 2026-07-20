from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from agent_tool_test_helpers import write_finetune_recipe
import pytest
import yaml

from agent_tools import experiment_pipeline, managed_scheduler, plans
from agent_tools.experiment_workspace import file_sha256, read_run_manifest
from agent_tools.manifests import write_rows


def _spec(root: Path) -> dict:
    return {
        "schema_version": 1,
        "pipeline": {
            "id": "external-v1",
            "kind": "external_matrix",
            "experiment_id": "unit",
            "step": {
                "id": "external-evaluate",
                "phase": "evaluate",
                "purpose": "Run the frozen external matrix.",
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
            "gpu_pool": list(range(8)),
            "gpus_per_run": 1,
            "max_concurrent": 8,
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
        "jobs": [
            {
                "id": "age-hsp-i2-psg",
                "checkpoint_source": "age",
                "cohort": "hsp_i2",
                "modality": "psg",
                "inference_preset_path": str(root / "presets" / "hsp_i2_age.pickle"),
                "num_workers": 8,
                "task": "age",
                "variant": "sleep2vec2",
                "label_name": "age",
            }
        ],
    }


@pytest.mark.parametrize("prefixes", [["ema_model."], ["running_mean_model."]])
def test_schema_requires_both_model_averaging_prefixes(tmp_path: Path, prefixes: list[str]):
    spec = _spec(tmp_path)
    spec["checkpoint_policy"]["forbidden_state_dict_prefixes"] = prefixes

    with pytest.raises(ValueError, match="forbidden_state_dict_prefixes"):
        experiment_pipeline._validate_spec(spec, tmp_path, unlock_final_test=True)


def test_successful_source_accepts_no_test_after_fit_manifest(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    run = {"step_id": "train-age", "run_id": "run-000"}
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps({"status": "skipped_test", "metrics": {"val_mae": 4.5}}) + "\n")

    monkeypatch.setattr(
        experiment_pipeline.artifacts,
        "read_hparam_plan",
        lambda _plan_dir: {"runs": [run]},
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "read_run_manifest",
        lambda _root: [{**run, "status": "finished"}],
    )
    monkeypatch.setattr(experiment_pipeline.artifacts, "find_run_manifest", lambda _run: manifest_path)

    states = experiment_pipeline._inspect_sources(root, spec, refresh=False)

    assert states[0]["complete"] is True
    assert states[0]["failed_runs"] == []


def test_retry_preflight_failure_does_not_block_independent_retry(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    spec = _spec(root)
    second_job = dict(spec["jobs"][0], id="age-hsp-i2-bcg", modality="bcg", num_workers=16)
    spec["jobs"].append(second_job)
    attempts = [{"job_id": job["id"], "attempt": 1, "status": "failed", "verified": "false"} for job in spec["jobs"]]
    order = []

    def attempt_recipe(_pipeline_dir, _spec, job, _selection, attempt):
        base = pipeline_dir / job["id"] / f"attempt-{attempt:03d}"
        return {"job": job["id"]}, base.with_suffix(".yaml"), base / "plan", base / "results"

    def retry_preflight(_pipeline_dir, job_id, _attempt, _recipe_path, _plan_dir):
        order.append(f"preflight:{job_id}")
        if job_id == "age-hsp-i2-psg":
            raise experiment_pipeline.RetryPreparationError("preflight failed")

    def materialize(_root, _spec, job, _selection, attempt, **_paths):
        order.append(f"materialize:{job['id']}")
        return {"job_id": job["id"], "attempt": attempt, "status": "planned", "verified": "false"}

    monkeypatch.setattr(experiment_pipeline, "_attempt_recipe", attempt_recipe)
    monkeypatch.setattr(experiment_pipeline, "_ensure_retry_preflight", retry_preflight)
    monkeypatch.setattr(experiment_pipeline, "_materialize_attempt", materialize)
    monkeypatch.setattr(experiment_pipeline, "append_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [])

    updated, created = experiment_pipeline._create_needed_retries(
        root,
        pipeline_dir,
        spec,
        {"age": {}},
        attempts,
    )

    assert created is True
    assert order == [
        "preflight:age-hsp-i2-psg",
        "preflight:age-hsp-i2-bcg",
        "materialize:age-hsp-i2-bcg",
    ]
    assert updated[0]["retry_preparation_error"] == "preflight failed"
    assert [row["attempt"] for row in updated if row["job_id"] == "age-hsp-i2-bcg"] == [1, 2]
    assert [job["status"] for job in experiment_pipeline._logical_job_states(spec, updated)] == [
        "failed",
        "running",
    ]


@pytest.mark.parametrize(
    "identity_error",
    [
        "PID 123 was reused by a different process.",
        "Canonical run has partial process identity; missing: process_start_token",
    ],
)
def test_unsafe_process_identity_is_blocked_and_never_retried(
    tmp_path: Path,
    monkeypatch,
    identity_error: str,
):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    attempt = {
        "experiment_id": "unit",
        "step_id": "external-evaluate",
        "run_id": "run-001",
        "job_id": "age-hsp-i2-psg",
        "attempt": 1,
        "status": "failed",
        "verified": "false",
    }
    write_rows(root / "run_manifest.tsv", [{**attempt, "process_identity_error": identity_error}])
    monkeypatch.setattr(
        experiment_pipeline,
        "_attempt_recipe",
        lambda *_args, **_kwargs: pytest.fail("unsafe process identity must not be retried"),
    )
    monkeypatch.setattr(experiment_pipeline, "append_event", lambda *_args, **_kwargs: None)

    updated, created = experiment_pipeline._create_needed_retries(
        root,
        pipeline_dir,
        _spec(root),
        {"age": {}},
        [attempt],
    )

    assert created is False
    assert updated[0]["retry_blocker"] == f"unsafe process identity: {identity_error}"
    assert experiment_pipeline._logical_job_states(_spec(root), updated)[0]["status"] == "blocked"


def test_atomic_generic_plan_freezes_single_runtime_command(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    recipe_path = write_finetune_recipe(source, variant="sleep2vec2")
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = tmp_path / "workspace"
    recipe["task"] = "infer"
    recipe["experiment"]["root"] = str(workspace)
    recipe["step"] = {
        "id": "external-evaluate",
        "phase": "evaluate",
        "purpose": "Exercise atomic external planning.",
    }
    recipe["execution"] = {
        "target": "local",
        "workdir": "/runtime/snapshot",
        "python": "/runtime/python",
        "runtime_commit": "a" * 40,
    }
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    config_bytes = Path(recipe["inputs"]["config"]).read_bytes()
    bound_config = {
        "_source_config_bytes": config_bytes,
        "_source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS, issues=[], decisions={})
    command = "/runtime/python -m sleep2vec2.infer --config frozen.yaml"
    monkeypatch.setattr(plans, "preflight_plan", lambda **_kwargs: (recipe, bound_config, report))
    monkeypatch.setattr(plans, "config_summary", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(plans, "_commands_for_recipe", lambda *_args, **_kwargs: [command])
    plan_dir = workspace / "plans" / "attempt-001"
    staging_dir = workspace / "plans" / ".attempt-001.staging"

    result = plans.build_plan(
        recipe_path=recipe_path,
        output_dir=plan_dir,
        staging_dir=staging_dir,
    )

    assert result.exit_code == 0
    assert plan_dir.is_dir()
    assert not staging_dir.exists()
    plan = json.loads((plan_dir / "plan.json").read_text())
    planned = plan["runs"][0]
    assert planned["command"] == command
    script_lines = Path(planned["script"]).read_text().splitlines()
    assert command in script_lines
    helper_index = script_lines.index("_agent_commit_status() {")
    running_index = script_lines.index("_agent_commit_status running")
    command_index = script_lines.index(command)
    assert script_lines[helper_index + 1].startswith("  /runtime/python -c ")
    assert any(line.startswith("/runtime/python -c ") and "a" * 40 in line for line in script_lines)
    assert helper_index < running_index < command_index
    assert plan["recipe"]["execution"] == recipe["execution"]
    canonical = read_run_manifest(workspace)[0]
    assert canonical.get("command") in (None, "")
    experiment_pipeline._validate_attempt_plan(
        {
            "step_id": planned["step_id"],
            "run_id": planned["run_id"],
            "recipe": str(recipe_path),
            "plan_dir": str(plan_dir),
        },
        canonical,
    )

    def inspect_command(_execution, probe):
        if "runtime_hostname" in probe[2]:
            payload = {
                "python": "/runtime/python",
                "python_version": "3.12",
                "runtime_commit": "a" * 40,
                "runtime_repo_root": "/runtime/snapshot",
                "runtime_hostname": "unit-host",
                "module": "sleep2vec2.infer",
                "module_origin": "/runtime/snapshot/sleep2vec2/infer.py",
            }
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
        evidence = {"supported_options": ["--config"], "cli_options_sha256": "cli-digest"}
        return SimpleNamespace(
            returncode=0,
            stdout="AGENT_CLI_PREFLIGHT=" + json.dumps(evidence) + "\n",
            stderr="",
        )

    snapshot = managed_scheduler.inspect_execution_target(
        {
            "target": "local",
            "workdir": "/runtime/snapshot",
            "python": "/runtime/python",
            "runtime_commit": "a" * 40,
        },
        [planned],
        command_runner=inspect_command,
    )
    assert snapshot["module"] == "sleep2vec2.infer"
    assert snapshot["required_options"] == ["--config"]


@pytest.mark.parametrize("tamper", [False, True])
def test_uncommitted_attempt_plan_is_deterministically_validated(tmp_path: Path, tamper: bool):
    root = tmp_path / "workspace"
    root.mkdir()
    experiment = {
        "id": "unit",
        "title": "Unit",
        "objective": "Exercise crash-safe external planning.",
        "root": str(root),
        "baseline": {"type": "none"},
        "status": "active",
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    (root / "run_manifest.tsv").write_text("step_id\trun_id\n")

    source_recipe = yaml.safe_load(write_finetune_recipe(tmp_path / "source", variant="sleep2vec2").read_text())
    config = Path(source_recipe["inputs"]["config"])
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    spec = _spec(root)
    preset = Path(spec["jobs"][0]["inference_preset_path"])
    preset.parent.mkdir(parents=True)
    preset.write_bytes(b"preset")
    selection = {
        "source_id": "age",
        "config": str(config),
        "config_sha256": file_sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "variant": "sleep2vec2",
        "label_name": "age",
    }
    pipeline_dir = root / "pipelines" / "external-v1"
    recipe, recipe_path, plan_dir, result_root = experiment_pipeline._attempt_recipe(
        pipeline_dir,
        spec,
        spec["jobs"][0],
        selection,
        1,
    )
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    staging_dir = plan_dir.parent / ".attempt-001.crash-window"

    report = plans.build_plan(
        recipe_path=recipe_path,
        output_dir=plan_dir,
        unlock_final_test=True,
        staging_dir=staging_dir,
        defer_commit=True,
    )
    assert report.exit_code == 0
    plan_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.replace(plan_dir)
    assert read_run_manifest(root) == []
    frozen_plan = json.loads((plan_dir / "plan.json").read_text())
    frozen_identity = {
        "target": "local",
        "workdir": spec["runtime"]["workdir"],
        "python": spec["runtime"]["python"],
        "runtime_commit": spec["runtime"]["runtime_commit"],
    }
    assert recipe["execution"] == frozen_identity
    assert frozen_plan["recipe"]["execution"] == frozen_identity
    assert yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())["execution"] == frozen_identity
    launch_path = Path(frozen_plan["runs"][0]["script"])
    launch_before = launch_path.read_bytes()
    launch_lines = launch_before.decode().splitlines()
    helper_index = launch_lines.index("_agent_commit_status() {")
    assert launch_lines[helper_index + 1].startswith(f"  {spec['runtime']['python']} -c ")

    if tamper:
        (plan_dir / "plan.md").write_text("tampered\n")
        with pytest.raises(ValueError, match="differs from deterministic regeneration"):
            experiment_pipeline._materialize_attempt(
                root,
                spec,
                spec["jobs"][0],
                selection,
                1,
                recipe_path=recipe_path,
                plan_dir=plan_dir,
                result_root=result_root,
            )
        assert read_run_manifest(root) == []
        return

    row = experiment_pipeline._materialize_attempt(
        root,
        spec,
        spec["jobs"][0],
        selection,
        1,
        recipe_path=recipe_path,
        plan_dir=plan_dir,
        result_root=result_root,
    )

    canonical = read_run_manifest(root)
    assert len(canonical) == 1
    assert row["job_id"] == "age-hsp-i2-psg"
    assert canonical[0]["pipeline_id"] == "external-v1"
    assert canonical[0]["terminal_status_owner"] == "script"
    assert launch_path.read_bytes() == launch_before
    assert not list(plan_dir.parent.glob(".attempt-001.*.staging"))


def test_attempt_config_drift_fails_before_plan_publication(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    experiment = {
        "id": "unit",
        "title": "Unit",
        "objective": "Reject attempt config drift before publication.",
        "root": str(root),
        "baseline": {"type": "none"},
        "status": "active",
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    (root / "run_manifest.tsv").write_text("step_id\trun_id\n")

    source_recipe = yaml.safe_load(write_finetune_recipe(tmp_path / "source", variant="sleep2vec2").read_text())
    config = Path(source_recipe["inputs"]["config"])
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    spec = _spec(root)
    preset = Path(spec["jobs"][0]["inference_preset_path"])
    preset.parent.mkdir(parents=True)
    preset.write_bytes(b"preset")
    selection = {
        "source_id": "age",
        "config": str(config),
        "config_sha256": file_sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "variant": "sleep2vec2",
        "label_name": "age",
    }
    pipeline_dir = root / "pipelines" / "external-v1"
    recipe, recipe_path, plan_dir, result_root = experiment_pipeline._attempt_recipe(
        pipeline_dir,
        spec,
        spec["jobs"][0],
        selection,
        1,
    )
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    config.write_text(config.read_text() + "\n# drifted after checkpoint selection\n")

    with pytest.raises(ValueError, match="externally bound SHA-256"):
        experiment_pipeline._materialize_attempt(
            root,
            spec,
            spec["jobs"][0],
            selection,
            1,
            recipe_path=recipe_path,
            plan_dir=plan_dir,
            result_root=result_root,
        )

    assert not plan_dir.exists()
    assert not list(plan_dir.parent.glob(f".{plan_dir.name}.*.staging"))
    assert not result_root.exists()
    assert read_run_manifest(root) == []


def test_jobs_exceeding_capacity_launch_only_available_gpu_slots(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    experiment = {
        "id": "unit",
        "title": "Unit",
        "objective": "Exercise external scheduler capacity.",
        "root": str(root),
        "baseline": {"type": "none"},
        "status": "active",
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    owner_dir = root / "pipelines" / "external-v1"
    owner_dir.mkdir(parents=True)
    runs = []
    for index in range(9):
        run_id = f"run-{index:03d}"
        job_id = f"job-{index:02d}"
        run_dir = owner_dir / "plans" / job_id / "attempt-001" / "runs" / f"{run_id}--{job_id}"
        run_dir.mkdir(parents=True)
        config = run_dir / "config.yaml"
        script = run_dir / "launch.sh"
        artifacts_path = run_dir / "artifacts.json"
        config.write_text("model: unit\n")
        script.write_text("#!/usr/bin/env bash\ntrue\n")
        script.chmod(0o755)
        artifacts_path.write_text("{}\n")
        runs.append(
            {
                "experiment_id": "unit",
                "step_id": "external-evaluate",
                "run_id": run_id,
                "run_name": job_id,
                "version": job_id,
                "status": "planned",
                "parameter_summary": "single resolved recipe",
                "config": str(config),
                "config_sha256": file_sha256(config),
                "script": str(script),
                "script_sha256": file_sha256(script),
                "run_dir": str(run_dir),
                "artifacts": str(artifacts_path),
                "runtime_dir": "",
                "checkpoint_dir": "",
                "pipeline_id": "external-v1",
                "job_id": job_id,
                "attempt": 1,
                "result_root": str(owner_dir / "results" / job_id / "attempt-001"),
                "terminal_status_owner": "script",
            }
        )
    write_rows(root / "run_manifest.tsv", runs)
    built = []
    started = []

    def build_command(_execution, _script, _log_path, _pid_path, gpus, **_kwargs):
        command = f"gpu={','.join(str(gpu) for gpu in gpus)}"
        built.append(command)
        return command

    hooks = managed_scheduler.SchedulerHooks(
        validated_snapshot=lambda *_args, **_kwargs: (None, False),
        build_command=build_command,
        start_process=lambda _execution, command: started.append(command) or "launched",
    )
    result = managed_scheduler.launch_managed_runs(
        root,
        owner_dir,
        runs,
        {
            "target": "local",
            "workdir": str(root),
            "gpu_pool": list(range(8)),
            "gpus_per_run": 1,
            "max_concurrent": 8,
        },
        {"devices": [0]},
        dry_run=False,
        default_script_commits_terminal_status=True,
        runtime_output_fields=("result_root",),
        runtime_output_root=root,
        hooks=hooks,
    )

    assert started == [f"gpu={index}" for index in range(8)]
    assert built == started
    assert [row["status"] for row in result.committed_rows].count("launched") == 8
    assert [row["status"] for row in result.committed_rows].count("pending") == 1
    assert sorted(row["gpus"] for row in result.committed_rows if row["status"] == "launched") == [
        str(index) for index in range(8)
    ]


def test_run_attempts_waits_when_capacity_blocks_before_execution_snapshot(tmp_path: Path, monkeypatch):
    class WaitObserved(Exception):
        pass

    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    plan_dir = pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"
    attempt = {
        "step_id": "external-evaluate",
        "run_id": "run-001",
        "pipeline_id": "external-v1",
        "job_id": "age-hsp-i2-psg",
        "variant": "sleep2vec2",
        "attempt": 1,
        "status": "pending",
        "verified": "false",
        "plan_dir": str(plan_dir),
        "runtime_commit": "",
    }
    write_rows(pipeline_dir / "jobs.tsv", [attempt])

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(experiment_pipeline, "_validate_attempt_rows", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_planned_runs",
        lambda _rows: [{"step_id": "external-evaluate", "run_id": "run-001"}],
    )
    launches = []

    def capacity_blocked(*_args, **_kwargs):
        launches.append(True)
        return SimpleNamespace(committed_rows=[dict(attempt)])

    monkeypatch.setattr(experiment_pipeline.managed_scheduler, "launch_managed_runs", capacity_blocked)
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [dict(attempt)])
    monkeypatch.setattr(experiment_pipeline.time, "sleep", lambda _seconds: (_ for _ in ()).throw(WaitObserved()))

    with pytest.raises(WaitObserved):
        experiment_pipeline._run_attempts(
            root,
            pipeline_dir,
            _spec(root),
            {"age": {}},
            [attempt],
            poll_seconds=1,
        )

    assert launches == [True]
    assert not (pipeline_dir / "execution_snapshot.json").exists()


def test_run_attempts_terminal_attempt_skips_live_snapshot_probe_and_verifies_result(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    snapshot_path = pipeline_dir / managed_scheduler.EXECUTION_SNAPSHOT_NAME
    snapshot_path.write_text(json.dumps({"runtime_commit": "a" * 40}) + "\n")
    attempt = {
        "step_id": "external-evaluate",
        "run_id": "run-001",
        "pipeline_id": "external-v1",
        "job_id": "age-hsp-i2-psg",
        "variant": "sleep2vec2",
        "attempt": 1,
        "status": "completed",
        "verified": "false",
        "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"),
        "runtime_commit": "",
    }
    write_rows(pipeline_dir / "jobs.tsv", [attempt])
    validations = []
    launches = []

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_attempt_rows",
        lambda *_args: validations.append("attempts"),
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_planned_runs",
        lambda _rows: [{"step_id": "external-evaluate", "run_id": "run-001"}],
    )
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [dict(attempt)])
    monkeypatch.setattr(
        experiment_pipeline.managed_scheduler,
        "validated_execution_snapshot",
        lambda *_args, **_kwargs: pytest.fail("terminal attempts must not probe the live runtime"),
    )
    monkeypatch.setattr(
        experiment_pipeline.managed_scheduler,
        "launch_managed_runs",
        lambda *_args, **_kwargs: launches.append(True) or SimpleNamespace(committed_rows=[dict(attempt)]),
    )
    result_manifest = pipeline_dir / "result_manifest.json"
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_result_manifest",
        lambda *_args: validations.append("result") or result_manifest,
    )

    result = experiment_pipeline._run_attempts(
        root,
        pipeline_dir,
        _spec(root),
        {"age": {}},
        [attempt],
        poll_seconds=0,
    )

    persisted = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")[0]
    assert result["status"] == "completed"
    assert launches == [True]
    assert validations == ["attempts", "result"]
    assert persisted["verified"] == "true"
    assert persisted["runtime_commit"] == "a" * 40


def test_run_attempts_mixed_group_validates_live_snapshot_before_launch(tmp_path: Path, monkeypatch):
    class LaunchObserved(Exception):
        pass

    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    (pipeline_dir / managed_scheduler.EXECUTION_SNAPSHOT_NAME).write_text("{}\n")
    spec = _spec(root)
    spec["jobs"].append(
        {
            **spec["jobs"][0],
            "id": "age-hsp-i2-bcg",
            "modality": "bcg",
            "num_workers": 16,
        }
    )
    attempts = [
        {
            "step_id": "external-evaluate",
            "run_id": "run-001",
            "pipeline_id": "external-v1",
            "job_id": "age-hsp-i2-psg",
            "variant": "sleep2vec2",
            "attempt": 1,
            "status": "completed",
            "verified": "true",
            "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"),
            "runtime_commit": "a" * 40,
        },
        {
            "step_id": "external-evaluate",
            "run_id": "run-002",
            "pipeline_id": "external-v1",
            "job_id": "age-hsp-i2-bcg",
            "variant": "sleep2vec2",
            "attempt": 1,
            "status": "pending",
            "verified": "false",
            "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-bcg" / "attempt-001"),
            "runtime_commit": "",
        },
    ]
    write_rows(pipeline_dir / "jobs.tsv", attempts)
    calls = []

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(experiment_pipeline, "_validate_attempt_rows", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_planned_runs",
        lambda rows: [{"step_id": row["step_id"], "run_id": row["run_id"]} for row in rows],
    )
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [dict(row) for row in attempts])

    def validate_snapshot(owner_dir, _execution, runs, _canonical):
        assert owner_dir == pipeline_dir
        assert [run["run_id"] for run in runs] == ["run-001", "run-002"]
        calls.append("snapshot")

    def observe_launch(_root, owner_dir, runs, *_args, **_kwargs):
        assert owner_dir == pipeline_dir
        assert [run["run_id"] for run in runs] == ["run-001", "run-002"]
        calls.append("launch")
        raise LaunchObserved

    monkeypatch.setattr(experiment_pipeline.managed_scheduler, "validated_execution_snapshot", validate_snapshot)
    monkeypatch.setattr(experiment_pipeline.managed_scheduler, "launch_managed_runs", observe_launch)

    with pytest.raises(LaunchObserved):
        experiment_pipeline._run_attempts(
            root,
            pipeline_dir,
            spec,
            {"age": {}},
            attempts,
            poll_seconds=0,
        )

    assert calls == ["snapshot", "launch"]


def test_run_attempts_blocks_on_external_missing_pid_capacity_blocker(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    spec = _spec(root)
    spec["jobs"].append(
        {
            **spec["jobs"][0],
            "id": "age-hsp-i2-bcg",
            "modality": "bcg",
            "num_workers": 16,
        }
    )
    attempts = [
        {
            "step_id": "external-evaluate",
            "run_id": "run-001",
            "pipeline_id": "external-v1",
            "job_id": "age-hsp-i2-psg",
            "variant": "sleep2vec",
            "attempt": 1,
            "status": "pending",
            "verified": "false",
            "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"),
            "runtime_commit": "",
        },
        {
            "step_id": "external-evaluate",
            "run_id": "run-002",
            "pipeline_id": "external-v1",
            "job_id": "age-hsp-i2-bcg",
            "variant": "sleep2vec2",
            "attempt": 1,
            "status": "pending",
            "verified": "false",
            "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-bcg" / "attempt-001"),
            "runtime_commit": "",
        },
    ]
    write_rows(pipeline_dir / "jobs.tsv", attempts)
    blocker = {
        "step_id": "train-age",
        "run_id": "run-099",
        "status": "missing_pid",
        "target": "local",
        "gpus": "0",
    }

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(experiment_pipeline, "_validate_attempt_rows", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_planned_runs",
        lambda rows: [{"step_id": row["step_id"], "run_id": row["run_id"]} for row in rows],
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "read_run_manifest",
        lambda _root: [*[dict(row) for row in attempts], dict(blocker)],
    )
    launches = []

    def blocked_launch(*_args, **kwargs):
        launches.append(kwargs["fail_on_missing_pid_blocker"])
        raise managed_scheduler.MissingPidCapacityError(blocker["step_id"], blocker["run_id"])

    monkeypatch.setattr(experiment_pipeline.managed_scheduler, "launch_managed_runs", blocked_launch)
    monkeypatch.setattr(
        experiment_pipeline,
        "_create_needed_retries",
        lambda *_args, **_kwargs: pytest.fail("a missing_pid capacity blocker must not create retries"),
    )
    monkeypatch.setattr(
        experiment_pipeline.time,
        "sleep",
        lambda *_args: pytest.fail("a missing_pid capacity blocker must not sleep"),
    )

    result = experiment_pipeline._run_attempts(
        root,
        pipeline_dir,
        spec,
        {"age": {}},
        attempts,
        poll_seconds=1,
    )

    assert result["status"] == "blocked"
    assert result["missing_pid_blocker"] == {
        "status": "missing_pid",
        "step_id": blocker["step_id"],
        "run_id": blocker["run_id"],
    }
    assert launches == [True]
    assert [row["status"] for row in experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")] == [
        "pending",
        "pending",
    ]
    assert not list(pipeline_dir.rglob(managed_scheduler.EXECUTION_SNAPSHOT_NAME))


def test_run_attempts_blocks_before_retry_when_external_run_has_missing_pid(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    attempt = {
        "step_id": "external-evaluate",
        "run_id": "run-001",
        "pipeline_id": "external-v1",
        "job_id": "age-hsp-i2-psg",
        "variant": "sleep2vec2",
        "attempt": 1,
        "status": "failed",
        "verified": "false",
        "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"),
        "runtime_commit": "",
    }
    blocker = {
        "step_id": "train-age",
        "run_id": "run-099",
        "status": "missing_pid",
        "target": "local",
        "gpus": "0",
    }
    write_rows(pipeline_dir / "jobs.tsv", [attempt])

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(experiment_pipeline, "_validate_attempt_rows", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_planned_runs",
        lambda rows: [{"step_id": row["step_id"], "run_id": row["run_id"]} for row in rows],
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "read_run_manifest",
        lambda _root: [dict(attempt), dict(blocker)],
    )
    monkeypatch.setattr(
        experiment_pipeline.managed_scheduler,
        "launch_managed_runs",
        lambda *_args, **_kwargs: SimpleNamespace(committed_rows=[dict(attempt)]),
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_create_needed_retries",
        lambda *_args, **_kwargs: pytest.fail("capacity blocker must be handled before retry creation"),
    )
    monkeypatch.setattr(
        experiment_pipeline.time,
        "sleep",
        lambda *_args: pytest.fail("capacity blocker must not sleep"),
    )

    result = experiment_pipeline._run_attempts(
        root,
        pipeline_dir,
        _spec(root),
        {"age": {}},
        [attempt],
        poll_seconds=1,
    )

    assert result["status"] == "blocked"
    assert result["missing_pid_blocker"] == {
        "status": "missing_pid",
        "step_id": blocker["step_id"],
        "run_id": blocker["run_id"],
    }
    persisted = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")
    assert [(row["attempt"], row["status"]) for row in persisted] == [("1", "failed")]


def test_run_attempts_syncs_owned_missing_pid_and_blocks_pending_sibling(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    spec = _spec(root)
    spec["jobs"].append(
        {
            **spec["jobs"][0],
            "id": "age-hsp-i2-bcg",
            "modality": "bcg",
            "num_workers": 16,
        }
    )
    attempts = [
        {
            "step_id": "external-evaluate",
            "run_id": "run-001",
            "pipeline_id": "external-v1",
            "job_id": "age-hsp-i2-psg",
            "variant": "sleep2vec2",
            "attempt": 1,
            "status": "running",
            "verified": "false",
            "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"),
            "runtime_commit": "",
        },
        {
            "step_id": "external-evaluate",
            "run_id": "run-002",
            "pipeline_id": "external-v1",
            "job_id": "age-hsp-i2-bcg",
            "variant": "sleep2vec2",
            "attempt": 1,
            "status": "pending",
            "verified": "false",
            "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-bcg" / "attempt-001"),
            "runtime_commit": "",
        },
    ]
    write_rows(pipeline_dir / "jobs.tsv", attempts)
    canonical = [{**attempts[0], "status": "missing_pid"}, dict(attempts[1])]

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(experiment_pipeline, "_validate_attempt_rows", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_planned_runs",
        lambda rows: [{"step_id": row["step_id"], "run_id": row["run_id"]} for row in rows],
    )
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [dict(row) for row in canonical])
    launches = []

    def blocked_launch(*_args, **kwargs):
        launches.append(kwargs["fail_on_missing_pid_blocker"])
        raise managed_scheduler.MissingPidCapacityError(canonical[0]["step_id"], canonical[0]["run_id"])

    monkeypatch.setattr(experiment_pipeline.managed_scheduler, "launch_managed_runs", blocked_launch)
    monkeypatch.setattr(
        experiment_pipeline,
        "_create_needed_retries",
        lambda *_args, **_kwargs: pytest.fail("an owned missing_pid attempt must not create retries"),
    )
    monkeypatch.setattr(
        experiment_pipeline.time,
        "sleep",
        lambda *_args: pytest.fail("an owned missing_pid attempt must not sleep"),
    )

    result = experiment_pipeline._run_attempts(
        root,
        pipeline_dir,
        spec,
        {"age": {}},
        attempts,
        poll_seconds=1,
    )

    assert result["status"] == "blocked"
    assert result["missing_pid_blocker"] == {
        "status": "missing_pid",
        "step_id": canonical[0]["step_id"],
        "run_id": canonical[0]["run_id"],
    }
    assert launches == [True]
    persisted = {row["run_id"]: row for row in experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")}
    assert persisted["run-001"]["status"] == "missing_pid"
    assert persisted["run-002"]["status"] == "pending"
    assert not list(pipeline_dir.rglob(managed_scheduler.EXECUTION_SNAPSHOT_NAME))


def test_execute_pipeline_persists_and_clears_missing_pid_blocker_on_resume(tmp_path: Path, monkeypatch):
    class ResumeObserved(Exception):
        pass

    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    (pipeline_dir / "pipeline.json").write_text(json.dumps({"status": "running_external"}) + "\n")
    blocker = {"status": "missing_pid", "step_id": "train-age", "run_id": "run-099"}
    blocked_result = {
        "status": "blocked",
        "jobs": [{"job_id": "age-hsp-i2-psg", "status": "running"}],
        "missing_pid_blocker": blocker,
    }

    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_frozen_pipeline",
        lambda *_args: json.loads((pipeline_dir / "pipeline.json").read_text()),
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_inspect_sources",
        lambda *_args, **_kwargs: [{"failed_runs": [], "uncertain_runs": [], "complete": True}],
    )
    monkeypatch.setattr(experiment_pipeline, "_load_or_freeze_selections", lambda *_args: {})
    monkeypatch.setattr(experiment_pipeline, "_load_or_create_initial_attempts", lambda *_args: [])
    monkeypatch.setattr(experiment_pipeline, "_run_attempts", lambda *_args, **_kwargs: blocked_result)

    result = experiment_pipeline._execute_pipeline(
        root,
        pipeline_dir,
        _spec(root),
        poll_seconds=1,
        finalize_callback=None,
    )

    assert result == blocked_result
    state = json.loads((pipeline_dir / "pipeline.json").read_text())
    assert state["status"] == "blocked"
    assert state["missing_pid_blocker"] == blocker

    def observe_resume(*_args, **_kwargs):
        resumed_state = json.loads((pipeline_dir / "pipeline.json").read_text())
        assert resumed_state["status"] == "running_external"
        assert resumed_state["missing_pid_blocker"] is None
        raise ResumeObserved

    monkeypatch.setattr(experiment_pipeline, "_run_attempts", observe_resume)

    with pytest.raises(ResumeObserved):
        experiment_pipeline._execute_pipeline(
            root,
            pipeline_dir,
            _spec(root),
            poll_seconds=1,
            finalize_callback=None,
        )


@pytest.mark.parametrize(
    ("field", "drifted"),
    [
        ("step_id", "foreign-step"),
        ("run_id", "run-999"),
    ],
)
def test_planned_runs_rejects_managed_key_drift(tmp_path: Path, field: str, drifted: str):
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    result_root = tmp_path / "results" / "attempt-001"
    expected = {
        "pipeline_id": "external-v1",
        "job_id": "age-hsp-i2-psg",
        "attempt": 1,
        "result_root": str(result_root),
        "terminal_status_owner": "script",
    }
    planned = {
        "step_id": "external-evaluate",
        "run_id": "run-001",
        field: drifted,
    }
    (plan_dir / "plan.json").write_text(json.dumps({"runs": [planned]}) + "\n")
    attempt = {
        "step_id": "external-evaluate",
        "run_id": "run-001",
        "plan_dir": str(plan_dir),
        **expected,
    }

    with pytest.raises(ValueError, match="drift"):
        experiment_pipeline._planned_runs([attempt])


def test_frozen_pipeline_rejects_external_preset_byte_drift(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    plan_dir = Path(spec["checkpoint_sources"]["age"]["plan"])
    plan_dir.mkdir(parents=True)
    plan_path = plan_dir / "plan.json"
    recipe_path = plan_dir / "recipe.resolved.yaml"
    plan_path.write_text("{}\n")
    recipe_path.write_text("task: hparam_tune\n")
    preset = Path(spec["jobs"][0]["inference_preset_path"])
    preset.parent.mkdir(parents=True)
    preset.write_bytes(b"frozen-preset")

    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    source_text = yaml.safe_dump(spec, sort_keys=False)
    resolved_text = yaml.safe_dump(spec, sort_keys=False)
    original_spec = tmp_path / "external.yaml"
    original_spec.write_text(source_text)
    (pipeline_dir / "spec.source.yaml").write_text(source_text)
    (pipeline_dir / "spec.resolved.yaml").write_text(resolved_text)
    state = {
        "schema_version": 1,
        "pipeline_id": "external-v1",
        "experiment_id": "unit",
        "status": "waiting_for_sources",
        "spec_path": str(original_spec),
        "spec_source_sha256": experiment_pipeline._text_sha256(source_text),
        "spec_resolved_sha256": experiment_pipeline._text_sha256(resolved_text),
        "runtime_commit": "a" * 40,
        "source_plans": [
            {
                "source_id": "age",
                "plan_dir": str(plan_dir),
                "plan_path": str(plan_path),
                "plan_sha256": file_sha256(plan_path),
                "resolved_recipe_path": str(recipe_path),
                "resolved_recipe_sha256": file_sha256(recipe_path),
            }
        ],
        "external_presets": [
            {
                "job_id": "age-hsp-i2-psg",
                "path": str(preset),
                "sha256": file_sha256(preset),
            }
        ],
    }
    (pipeline_dir / "pipeline.json").write_text(json.dumps(state) + "\n")
    monkeypatch.setattr(experiment_pipeline.artifacts, "read_hparam_plan", lambda _plan_dir: {})

    preset.write_bytes(b"changed-preset")

    with pytest.raises(ValueError, match="Frozen external preset changed"):
        experiment_pipeline._validate_frozen_pipeline(pipeline_dir, source_text, spec)


@pytest.mark.parametrize("tamper", [False, True])
def test_orphan_checkpoint_selection_is_rederived_before_state_commit(tmp_path: Path, monkeypatch, tamper: bool):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    spec = _spec(root)
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "rank-1.ckpt"
    alternate = tmp_path / "alternate.ckpt"
    config.write_text("model: unit\n")
    checkpoint.write_bytes(b"rank-1")
    alternate.write_bytes(b"alternate")
    derived = {
        "source_id": "age",
        "plan": spec["checkpoint_sources"]["age"]["plan"],
        "selection_metric": "val_mae",
        "selection_mode": "min",
        "score": 4.5,
        "config": str(config),
        "config_sha256": file_sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "variant": "sleep2vec2",
        "label_name": "age",
        "source_task": "age",
    }
    orphan = copy.deepcopy(derived)
    if tamper:
        orphan["checkpoint"] = str(alternate)
        orphan["checkpoint_sha256"] = file_sha256(alternate)
    checkpoints_path = pipeline_dir / "checkpoints.json"
    checkpoints_path.write_text(
        json.dumps(
            {
                "pipeline_id": "external-v1",
                "created_at": "2026-07-20T00:00:00Z",
                "sources": [orphan],
            }
        )
        + "\n"
    )
    state_path = pipeline_dir / "pipeline.json"
    state_path.write_text(json.dumps({"status": "waiting_for_sources"}) + "\n")
    monkeypatch.setattr(experiment_pipeline, "_select_checkpoint_sources", lambda *_args: [derived])

    if tamper:
        with pytest.raises(ValueError, match="differs from validation-derived selection"):
            experiment_pipeline._load_or_freeze_selections(root, pipeline_dir, spec)
        assert "checkpoint_selection_sha256" not in json.loads(state_path.read_text())
    else:
        selections = experiment_pipeline._load_or_freeze_selections(root, pipeline_dir, spec)
        assert selections == {"age": derived}
        assert json.loads(state_path.read_text())["checkpoint_selection_sha256"] == file_sha256(checkpoints_path)


def test_completed_pipeline_resume_validates_and_finalizes_without_reexecution(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    report = pipeline_dir / "final.md"
    report.write_text("completed\n")
    state = {
        "status": "completed",
        "final_report": str(report),
        "result_artifacts": {str(report): file_sha256(report)},
    }
    (pipeline_dir / "pipeline.json").write_text(json.dumps(state) + "\n")
    attempt = {"status": "completed"}

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: state)
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_experiment",
        lambda *_args, **_kwargs: {"status": "active"},
    )
    monkeypatch.setattr(experiment_pipeline, "_load_or_freeze_selections", lambda *_args: {"age": {}})
    monkeypatch.setattr(experiment_pipeline, "read_rows", lambda *_args, **_kwargs: [attempt])
    monkeypatch.setattr(experiment_pipeline, "_validate_attempt_rows", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_logical_job_states",
        lambda *_args: [{"job_id": "age-hsp-i2-psg", "status": "completed"}],
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_inspect_sources",
        lambda *_args, **_kwargs: pytest.fail("completed pipelines must not recheck training sources"),
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_run_attempts",
        lambda *_args, **_kwargs: pytest.fail("completed pipelines must not rerun external attempts"),
    )
    finalized = []

    result = experiment_pipeline._execute_pipeline(
        root,
        pipeline_dir,
        _spec(root),
        poll_seconds=0,
        finalize_callback=lambda finalized_root, finalized_report: finalized.append((finalized_root, finalized_report)),
    )

    assert result["status"] == "completed"
    assert finalized == [(root, report)]

    finalized.clear()
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_experiment",
        lambda *_args, **_kwargs: {"status": "completed"},
    )
    result = experiment_pipeline._execute_pipeline(
        root,
        pipeline_dir,
        _spec(root),
        poll_seconds=0,
        finalize_callback=lambda finalized_root, finalized_report: finalized.append((finalized_root, finalized_report)),
    )
    assert result["status"] == "completed"
    assert finalized == []
