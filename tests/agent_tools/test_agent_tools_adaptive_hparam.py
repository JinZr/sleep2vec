from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import adaptive_hparam, hparam_runtime, manifests, run_evidence
from agent_tools.experiment_workspace import merge_run_manifest


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


def _adaptive_recipe(
    tmp_path: Path, *, test_feedback: bool = True, max_rounds: int = 2, relative_base: bool = False
) -> Path:
    base = write_finetune_recipe(tmp_path)
    return write_yaml(
        tmp_path / "adaptive_tune.yaml",
        {
            "name": "unit_adaptive",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": base.name if relative_base else str(base),
            "execution": {"workdir": str(tmp_path / "runtime")},
            "search": {
                "method": "grid",
                "max_runs": 1,
                "parameters": {"runtime.lr": [1e-6], "yaml:/model/head/name": ["classification"]},
            },
            "adaptive": {
                "enabled": True,
                "objective_metric": "test_auroc",
                "objective_mode": "max",
                "test_feedback_for_selection": test_feedback,
                "max_rounds": max_rounds,
                "max_runs_total": 4,
                "round_size": 1,
                "poll_seconds": 1,
                "replacement": {
                    "enabled": True,
                    "allow_running_stop": True,
                    "grace_epochs": 1,
                    "grace_minutes": 1,
                    "kill_margin": 0.05,
                },
                "suggest": {"strategy": "best_neighborhood"},
            },
            "evaluation_policy": {
                "selection_metric": "val_ahi_pearson",
                "selection_mode": "max",
                "selection_split": "val",
                "external_test_locked": True,
                "test_after_fit": False,
                "final_eval_split": "test",
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "external optimized adaptive", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def _write_fake_manifest(workflow_dir: Path, *, score: float = 0.7) -> None:
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    launched = _run("hparam-launch", "--plan-dir", str(round_dir))
    assert launched.returncode == 0, launched.stderr
    plan = json.loads((round_dir / "plan.json").read_text())
    run = plan["runs"][0]
    run_dir = Path(run["runtime_dir"])
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "epoch=3.ckpt").write_text("checkpoint")
    (ckpt_dir / "best-epoch=3.ckpt").write_text("alias")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": run["version"],
                "monitor": "val_ahi_pearson",
                "monitor_mode": "max",
                "best_model_score": 0.5,
                "best_model_path": str(ckpt_dir / "best-epoch=3.ckpt"),
                "epoch": 3,
                "status": "finished",
                "metrics": {"val_ahi_pearson": 0.5, "test_auroc": score},
            }
        )
    )


def test_adaptive_recipe_requires_explicit_test_feedback_flag(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path, test_feedback=False)

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 1
    assert "adaptive.test_feedback_for_selection" in result.stdout


def test_adaptive_rejects_removed_run_budget_and_gpu_fields(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_trials"] = payload["search"].pop("max_runs")
    payload["adaptive"]["max_trials_total"] = payload["adaptive"].pop("max_runs_total")
    payload["execution"]["gpus_per_trial"] = 1
    recipe.write_text(yaml.safe_dump(payload))

    result = _run("doctor", "--recipe", str(recipe))

    assert result.returncode == 1
    assert "search.max_trials is no longer supported" in result.stdout
    assert "adaptive.max_trials_total is no longer supported" in result.stdout
    assert "execution.gpus_per_trial is no longer supported" in result.stdout


def test_adaptive_init_preflight_leaves_blocked_root_untouched_then_retries(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _adaptive_recipe(source)
    workspace = tmp_path / "adaptive-workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    payload["decisions"]["label_name"] = {"value": "ASK_USER", "source": "explicit_recipe"}
    recipe.write_text(yaml.safe_dump(payload))

    blocked = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workspace))

    assert blocked.returncode != 0
    assert not workspace.exists()
    payload["decisions"]["label_name"] = {"value": "ahi", "source": "explicit_recipe"}
    recipe.write_text(yaml.safe_dump(payload))

    retry = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workspace))

    assert retry.returncode == 0, retry.stderr
    assert (workspace / "adaptive" / "rounds" / "round_000" / "plan.json").exists()


def test_adaptive_init_creates_round_zero_without_modifying_original_recipe(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    before = recipe.read_text()
    workflow_dir = tmp_path / "workflow"

    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))

    assert result.returncode == 0, result.stderr
    assert recipe.read_text() == before
    assert (workflow_dir / "adaptive" / "workflow.json").exists()
    assert (workflow_dir / "adaptive" / "rounds" / "round_000" / "plan.json").exists()
    assert (workflow_dir / "adaptive" / "run_registry.tsv").exists()
    assert "adaptive_init" in (tmp_path / "events.jsonl").read_text()


def test_adaptive_init_initializes_fresh_experiment_root(tmp_path: Path):
    source_dir = tmp_path / "source"
    recipe = _adaptive_recipe(source_dir)
    payload = yaml.safe_load(recipe.read_text())
    workflow_dir = tmp_path / "fresh-workflow"
    payload["experiment"]["root"] = str(workflow_dir)
    recipe.write_text(yaml.safe_dump(payload))

    assert not workflow_dir.exists()
    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))

    assert result.returncode == 0, result.stderr
    assert (workflow_dir / "experiment.yaml").exists()
    assert (workflow_dir / "adaptive" / "workflow.json").exists()
    assert (workflow_dir / "adaptive" / "rounds" / "round_000" / "plan.json").exists()
    assert _read_table(workflow_dir / "run_manifest.tsv")


def test_adaptive_init_preflights_round_recipe_before_workspace_mutation(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    round_recipe = workflow_dir / "adaptive" / "rounds" / "round_000" / "round_recipe.yaml"
    round_recipe.parent.mkdir(parents=True)
    round_recipe.symlink_to(tmp_path / "run_manifest.tsv")
    manifest_before = (tmp_path / "run_manifest.tsv").read_bytes()
    events_path = tmp_path / "events.jsonl"

    with pytest.raises(ValueError, match="Managed output"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow_dir)

    assert (tmp_path / "run_manifest.tsv").read_bytes() == manifest_before
    assert not events_path.exists()
    assert not (workflow_dir / "adaptive" / "workflow.json").exists()


def test_adaptive_management_paths_are_canonical_across_alias_entries(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _adaptive_recipe(source)
    real_root = tmp_path / "real-workflow"
    alias_root = tmp_path / "workflow-alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(real_root)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    initialized = adaptive_hparam.init_adaptive_workflow(recipe, alias_root)

    assert initialized == real_root
    workflow = json.loads((real_root / "adaptive" / "workflow.json").read_text())
    assert workflow["root"] == str(real_root)
    assert workflow["recipe_path"] == str(recipe.resolve())
    registry = _read_table(real_root / "adaptive" / "run_registry.tsv")
    for field in ("round_dir", "config", "script"):
        assert Path(registry[0][field]).is_absolute()
        assert Path(registry[0][field]).resolve().is_relative_to(real_root)
    round_recipe = yaml.safe_load((real_root / "adaptive" / "rounds" / "round_000" / "round_recipe.yaml").read_text())
    assert round_recipe["experiment"]["root"] == str(real_root)
    assert Path(round_recipe["base_recipe"]).is_absolute()

    _write_fake_manifest(real_root)
    digest = adaptive_hparam.digest_hparam_run(alias_root)
    suggestion = adaptive_hparam.suggest_next_round(alias_root)
    stepped = adaptive_hparam.adaptive_step(alias_root)
    looped = adaptive_hparam.adaptive_loop(alias_root)

    for path in (digest, suggestion, stepped, looped):
        assert path.is_absolute()
        assert path.resolve().is_relative_to(real_root)
    suggested = yaml.safe_load(suggestion.read_text())
    assert suggested["experiment"]["root"] == str(real_root)
    assert Path(suggested["base_recipe"]).is_absolute()
    events = [json.loads(line) for line in (real_root / "events.jsonl").read_text().splitlines()]
    for event in events:
        for field in ("path", "recipe_path", "round_dir", "digest", "suggestion"):
            if event.get(field):
                assert Path(event[field]).is_absolute()


def test_adaptive_workflow_root_drift_fails_before_suggestion_write(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    workflow["root"] = str(tmp_path / "other-workflow")
    workflow_path.write_text(json.dumps(workflow))
    events = (tmp_path / "events.jsonl").read_bytes()

    result = _run("hparam-suggest", "--workflow-dir", str(workflow_dir))

    assert result.returncode == 1
    assert "workflow root differs" in result.stderr
    assert not (workflow_dir / "adaptive" / "suggestions").exists()
    assert (tmp_path / "events.jsonl").read_bytes() == events


def test_adaptive_relative_recipe_locator_fails_before_suggestion_write(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    workflow_path = workflow_dir / "adaptive" / "workflow.json"
    workflow = json.loads(workflow_path.read_text())
    workflow["recipe_path"] = recipe.name
    workflow_path.write_text(json.dumps(workflow))
    before = {path.relative_to(workflow_dir): path.read_bytes() for path in workflow_dir.rglob("*") if path.is_file()}
    monkeypatch.chdir(recipe.parent)

    with pytest.raises(ValueError, match="recipe_path must be absolute"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert {
        path.relative_to(workflow_dir): path.read_bytes() for path in workflow_dir.rglob("*") if path.is_file()
    } == before


def test_adaptive_legacy_registry_fails_before_monitor_or_digest_write(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    (workflow_dir / "adaptive" / "trial_registry.tsv").write_text("trial_id\ntrial_000\n")
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"

    result = _run("hparam-digest", "--run-dir", str(workflow_dir))

    assert result.returncode == 1
    assert "Legacy adaptive registry is read-only" in result.stderr
    assert not (round_dir / "run_status.tsv").exists()
    assert not (workflow_dir / "adaptive" / "digests").exists()


def test_adaptive_registry_must_bind_current_round_before_monitor(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    registry = workflow_dir / "adaptive" / "run_registry.tsv"
    registry.write_text("step_id\trun_id\tround\tround_dir\n")
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"

    result = _run("hparam-digest", "--run-dir", str(workflow_dir))

    assert result.returncode == 1
    assert "registry is missing the current plan run" in result.stderr
    assert not (round_dir / "run_status.tsv").exists()
    assert not (workflow_dir / "adaptive" / "digests").exists()


@pytest.mark.parametrize("registry_fault", ["foreign", "unmanaged", "config_drift"])
def test_adaptive_registry_ownership_fails_before_workflow_mutation(tmp_path: Path, registry_fault: str):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    registry_path = workflow_dir / "adaptive" / "run_registry.tsv"
    registry = _read_table(registry_path)
    if registry_fault == "foreign":
        registry[0]["experiment_id"] = "foreign-experiment"
    elif registry_fault == "unmanaged":
        registry.append(
            {
                **registry[0],
                "experiment_id": "foreign-experiment",
                "step_id": "foreign-step",
                "run_id": "run-999",
                "version": "foreign-version",
            }
        )
    else:
        registry[0]["config"] = str(tmp_path / "other-config.yaml")
    manifests.write_rows(registry_path, registry)
    before = {path.relative_to(workflow_dir): path.read_bytes() for path in workflow_dir.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="canonical manifest|Frozen run field differs"):
        adaptive_hparam._workflow(workflow_dir)

    assert {
        path.relative_to(workflow_dir): path.read_bytes() for path in workflow_dir.rglob("*") if path.is_file()
    } == before


def test_adaptive_registry_rejects_header_only_legacy_identity(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    registry_path = workflow_dir / "adaptive" / "run_registry.tsv"
    registry_path.write_text("trial_id\tround\n")

    with pytest.raises(ValueError, match="Historical trial_id fields"):
        adaptive_hparam._workflow(workflow_dir)

    assert registry_path.read_text() == "trial_id\tround\n"


def test_adaptive_stop_scan_rejects_header_only_legacy_status(tmp_path: Path):
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    status_path = round_dir / "run_status.tsv"
    status_path.write_text("trial_id\tstatus\n")
    recipe = {"adaptive": {"replacement": {"enabled": True, "allow_running_stop": True}}}

    with pytest.raises(ValueError, match="Historical trial_id fields"):
        adaptive_hparam._stop_bad_running_runs(tmp_path, round_dir, recipe)

    assert status_path.read_text() == "trial_id\tstatus\n"


def test_adaptive_digest_and_suggest_use_external_objective_and_fixed_checkpoint(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)

    digest = _run("hparam-digest", "--run-dir", str(workflow_dir))
    suggest = _run("hparam-suggest", "--workflow-dir", str(workflow_dir))

    assert digest.returncode == 0, digest.stderr
    assert suggest.returncode == 0, suggest.stderr
    rows = _read_table(workflow_dir / "adaptive" / "digests" / "round_000.csv")
    assert rows[0]["test_auroc"] == "0.73"
    assert rows[0]["checkpoint_path"].endswith("epoch=3.ckpt")
    assert "best-epoch" not in rows[0]["checkpoint_path"]
    suggestion = yaml.safe_load((workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").read_text())
    assert suggestion["search"]["parameters"]["runtime.lr"] == [5e-07, 1e-06, 1.5e-06]
    assert "external_optimized: true" in (workflow_dir / "adaptive" / "digests" / "round_000.md").read_text()
    incumbents = _read_table(workflow_dir / "adaptive" / "incumbents.tsv")
    assert incumbents[-1]["objective_score"] == "0.73"


def test_adaptive_digest_uses_canonical_status_not_runtime_manifest(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    runtime_manifest = Path(run["runtime_dir"]) / "run_manifest.json"
    runtime = json.loads(runtime_manifest.read_text())
    runtime["status"] = "completed"
    runtime["metrics"]["status"] = "finished"
    runtime_manifest.write_text(json.dumps(runtime))
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "failed"}],
    )

    digest = adaptive_hparam.digest_hparam_run(round_dir)

    assert _read_table(digest)[0]["status"] == "failed"
    assert _read_table(round_dir / "run_status.tsv")[0]["status"] == "failed"
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "run_status_changed" not in [event["event_type"] for event in events]


def test_adaptive_digest_preflights_outputs_before_monitor(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    digest.parent.mkdir(parents=True)
    digest.symlink_to(tmp_path / "run_manifest.tsv")
    manifest_before = (tmp_path / "run_manifest.tsv").read_bytes()
    events_before = (tmp_path / "events.jsonl").read_bytes()
    monitor_calls = []
    monkeypatch.setattr(
        adaptive_hparam,
        "monitor_hparam_runs",
        lambda path: monitor_calls.append(Path(path)) or round_dir / "run_status.tsv",
    )

    with pytest.raises(ValueError, match="Managed output"):
        adaptive_hparam.digest_hparam_run(workflow_dir)

    assert monitor_calls == []
    assert (tmp_path / "run_manifest.tsv").read_bytes() == manifest_before
    assert (tmp_path / "events.jsonl").read_bytes() == events_before


@pytest.mark.parametrize("filename", ["round_001.yaml", "round_001.md"])
def test_adaptive_suggest_preflights_outputs_before_writing(tmp_path: Path, filename: str):
    recipe = _adaptive_recipe(tmp_path)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    suggestion = workflow_dir / "adaptive" / "suggestions" / filename
    suggestion.parent.mkdir(parents=True)
    suggestion.hardlink_to(tmp_path / "run_manifest.tsv")
    manifest_before = (tmp_path / "run_manifest.tsv").read_bytes()
    events_before = (tmp_path / "events.jsonl").read_bytes()

    with pytest.raises(ValueError, match="Managed output"):
        adaptive_hparam.suggest_next_round(workflow_dir)

    assert (tmp_path / "run_manifest.tsv").read_bytes() == manifest_before
    assert (tmp_path / "events.jsonl").read_bytes() == events_before


def test_adaptive_step_dry_run_writes_suggestion_without_superseding_current_round(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    launch = _read_table(round_dir / "launch_manifest.tsv")[0]
    manifests.write_rows(
        round_dir / "launch_manifest.tsv",
        [{**launch, "status": "planned"}],
    )

    result = _run("hparam-adaptive-step", "--workflow-dir", str(workflow_dir))

    assert result.returncode == 0, result.stderr
    assert (workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").exists()
    assert not (workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").exists()
    events = (tmp_path / "events.jsonl").read_text()
    assert "supersede_pending_run" not in events
    assert "adaptive_step_dry_run" in events
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "planned"
    assert _read_table(round_dir / "run_status.tsv")[0]["status"] == "planned"
    assert _read_table(round_dir / "launch_manifest.tsv")[0]["status"] == "planned"


def test_execute_supersedes_canonical_pending_run_and_prevents_old_round_launch(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    launch_before = _read_table(round_dir / "launch_manifest.tsv")[0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "pending"}],
    )
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    launched_rounds = []

    def fake_launch(run_dir, *, dry_run=True):
        launched_rounds.append((Path(run_dir), dry_run))
        return Path(run_dir) / "launch_manifest.tsv"

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: digest)

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "superseded"
    assert _read_table(round_dir / "run_status.tsv")[0]["status"] == "superseded"
    launch_after = _read_table(round_dir / "launch_manifest.tsv")[0]
    assert launch_after["status"] == "superseded"
    for field in ("target", "host", "workdir", "gpus", "log_path", "pid_path", "command"):
        assert launch_after[field] == launch_before[field]
    assert launched_rounds == [(workflow_dir / "adaptive" / "rounds" / "round_001", False)]
    events_path = tmp_path / "events.jsonl"
    before = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [event["event_type"] for event in before].count("supersede_pending_run") == 1

    adaptive_hparam._supersede_pending_runs(workflow_dir, round_dir)

    after = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [event["event_type"] for event in after].count("supersede_pending_run") == 1
    started = []
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: started.append(True) or "launched")

    hparam_runtime.launch_hparam_runs(round_dir, dry_run=False)

    assert started == []


def test_supersede_uses_canonical_status_and_repairs_stale_round_mirrors(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "failed"}],
    )
    stale = [{**run, "status": "planned", "target": "local", "pid_path": "", "log_path": ""}]
    manifests.write_rows(round_dir / "run_status.tsv", stale)
    manifests.write_rows(round_dir / "launch_manifest.tsv", stale)
    events_path = tmp_path / "events.jsonl"
    before = events_path.read_bytes()

    adaptive_hparam._supersede_pending_runs(workflow_dir, round_dir)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"
    assert _read_table(round_dir / "run_status.tsv")[0]["status"] == "failed"
    assert _read_table(round_dir / "launch_manifest.tsv")[0]["status"] == "failed"
    assert events_path.read_bytes() == before


def test_supersede_event_uses_the_status_committed_by_the_canonical_owner(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    real_merge = merge_run_manifest

    def merge_after_wandb_update(root, rows):
        real_merge(root, [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "failed"}])
        return real_merge(root, rows)

    monkeypatch.setattr(adaptive_hparam, "merge_run_manifest", merge_after_wandb_update)
    events_path = tmp_path / "events.jsonl"
    before = events_path.read_bytes()

    adaptive_hparam._supersede_pending_runs(workflow_dir, round_dir)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"
    assert _read_table(round_dir / "run_status.tsv")[0]["status"] == "failed"
    assert _read_table(round_dir / "launch_manifest.tsv")[0]["status"] == "failed"
    assert events_path.read_bytes() == before


def test_supersede_does_not_override_run_launched_after_eligibility_check(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    real_merge = merge_run_manifest

    def merge_after_launch(root, rows):
        real_merge(root, [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "running"}])
        return real_merge(root, rows)

    monkeypatch.setattr(adaptive_hparam, "merge_run_manifest", merge_after_launch)
    events_path = tmp_path / "events.jsonl"
    before = events_path.read_bytes()

    adaptive_hparam._supersede_pending_runs(workflow_dir, round_dir)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "running"
    assert _read_table(round_dir / "run_status.tsv")[0]["status"] == "running"
    assert _read_table(round_dir / "launch_manifest.tsv")[0]["status"] == "running"
    assert events_path.read_bytes() == before


@pytest.mark.parametrize("target_name", ["run_status.tsv", "launch_manifest.tsv"])
@pytest.mark.parametrize("target_kind", ["directory", "hardlink"])
def test_supersede_preflights_round_mirrors_before_canonical_commit(tmp_path: Path, target_name: str, target_kind: str):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    mirrors = [{**run, "status": "planned", "target": "local", "pid_path": "", "log_path": ""}]
    manifests.write_rows(round_dir / "run_status.tsv", mirrors)
    manifests.write_rows(round_dir / "launch_manifest.tsv", mirrors)
    target = round_dir / target_name
    target.unlink()
    if target_kind == "directory":
        target.mkdir()
    else:
        target.hardlink_to(tmp_path / "run_manifest.tsv")
    manifest_path = tmp_path / "run_manifest.tsv"
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="Managed output"):
        adaptive_hparam._supersede_pending_runs(workflow_dir, round_dir)

    assert manifest_path.read_bytes() == before


def test_adaptive_step_blocks_suggestion_without_scored_objective(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    assert _run("hparam-launch", "--plan-dir", str(round_dir)).returncode == 0

    result = _run("hparam-adaptive-step", "--workflow-dir", str(workflow_dir))

    assert result.returncode != 0
    assert "No digest rows with finite test_auroc" in result.stderr
    assert "suggest_blocked" in (tmp_path / "events.jsonl").read_text()
    assert not (workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").exists()


def test_adaptive_step_execute_resolves_relative_base_recipe_for_next_round(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3, relative_base=True)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    launched = []

    def fake_launch(run_dir, *, dry_run=True):
        launched.append((Path(run_dir), dry_run))
        return Path(run_dir) / "launch_manifest.tsv"

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    suggestion = yaml.safe_load((workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").read_text())
    assert Path(suggestion["base_recipe"]).is_absolute()
    assert (workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").exists()
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "superseded"
    assert (
        _read_table(workflow_dir / "adaptive" / "rounds" / "round_000" / "run_status.tsv")[0]["status"] == "superseded"
    )
    assert (
        _read_table(workflow_dir / "adaptive" / "rounds" / "round_000" / "launch_manifest.tsv")[0]["status"]
        == "superseded"
    )
    assert launched == [(workflow_dir / "adaptive" / "rounds" / "round_001", False)]


def test_adaptive_step_preflights_next_round_before_stop_or_supersede(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    invalid = tmp_path / "invalid-next-round.yaml"
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 0
    invalid.write_text(yaml.safe_dump(payload))
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    digest.parent.mkdir(parents=True)
    digest.write_text("run_id,test_auroc\nrun-000,0.7\n")
    calls = []

    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: digest)
    monkeypatch.setattr(adaptive_hparam, "suggest_next_round", lambda _root: invalid)
    monkeypatch.setattr(adaptive_hparam, "_stop_bad_running_runs", lambda *_args: calls.append("stop"))
    monkeypatch.setattr(adaptive_hparam, "_supersede_pending_runs", lambda *_args: calls.append("supersede"))

    for execute in (False, True):
        try:
            adaptive_hparam.adaptive_step(workflow_dir, execute=execute)
        except RuntimeError as exc:
            assert "failed preflight" in str(exc)
        else:
            raise AssertionError("adaptive_step should fail before mutating the active round")

        assert calls == []
        assert not (workflow_dir / "adaptive" / "rounds" / "round_001").exists()


def test_hparam_count_does_not_materialize_search_values():
    recipe = {
        "search": {
            "parameters": {
                "runtime.lr": range(1000),
                "runtime.batch_size": range(1000),
            }
        }
    }

    assert adaptive_hparam._hparam_count(recipe) == 1_000_000


def test_adaptive_step_execute_stops_bad_running_run_through_recorded_manifest(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=2)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    launch = _read_table(round_dir / "launch_manifest.tsv")[0]
    pid_path = Path(launch["pid_path"])
    log_path = Path(launch["log_path"])
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))
    log_path.write_text("Traceback\nRuntimeError: failed\n")
    manifests.write_rows(
        round_dir / "launch_manifest.tsv",
        [{**launch, "status": "launched"}],
    )
    stopped = []

    def fake_stop(run_dir, run_id, *, reason):
        stopped.append((Path(run_dir), run_id))
        return Path(run_dir) / "run_status.tsv"

    monkeypatch.setattr(adaptive_hparam, "stop_hparam_run", fake_stop)

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert stopped == [(round_dir, "run-000")]
    assert "stop_bad_running_run" in (tmp_path / "events.jsonl").read_text()


def test_adaptive_step_execute_at_budget_keeps_current_runs_unchanged(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=1)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    calls = []

    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: digest)
    monkeypatch.setattr(adaptive_hparam, "_stop_bad_running_runs", lambda *_args: calls.append("stop"))
    monkeypatch.setattr(adaptive_hparam, "_supersede_pending_runs", lambda *_args: calls.append("supersede"))
    before = (tmp_path / "run_manifest.tsv").read_bytes()

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert calls == []
    assert (tmp_path / "run_manifest.tsv").read_bytes() == before
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "planned"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    event_types = [event["event_type"] for event in events]
    assert "adaptive_budget_exhausted" in event_types
    assert "adaptive_step_dry_run" not in event_types
    assert not (workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").exists()


def test_adaptive_step_checks_prospective_round_size_against_run_budget(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"]["max_runs_total"] = 2
    payload["adaptive"]["round_size"] = 2
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    digest = workflow_dir / "adaptive" / "digests" / "round_000.csv"
    manifests.write_rows(digest, [{**run, "test_auroc": 0.73}])
    calls = []
    monkeypatch.setattr(adaptive_hparam, "digest_hparam_run", lambda _round_dir: digest)
    monkeypatch.setattr(adaptive_hparam, "_stop_bad_running_runs", lambda *_args: calls.append("stop"))
    monkeypatch.setattr(adaptive_hparam, "_supersede_pending_runs", lambda *_args: calls.append("supersede"))
    registry = workflow_dir / "adaptive" / "run_registry.tsv"
    registry_before = registry.read_bytes()

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert calls == []
    assert registry.read_bytes() == registry_before
    assert len(_read_table(registry)) == 1
    assert not (workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").exists()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event_type"] == "adaptive_budget_exhausted"


def test_adaptive_loop_stops_when_step_cannot_create_a_budgeted_round(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    suggestion = workflow_dir / "adaptive" / "suggestions" / "round_001.yaml"
    calls = []
    monkeypatch.setattr(
        adaptive_hparam,
        "adaptive_step",
        lambda path, *, execute=False: calls.append((Path(path), execute)) or suggestion,
    )
    monkeypatch.setattr(
        adaptive_hparam.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("loop should stop before polling")),
    )

    result = adaptive_hparam.adaptive_loop(workflow_dir, execute=True)

    assert result == suggestion
    assert calls == [(workflow_dir, True)]


def test_running_stop_passes_remote_status_row_to_failure_log_check(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=1)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    manifests.write_rows(
        round_dir / "run_status.tsv",
        [{**run, "status": "running", "target": "ssh", "host": "baichuan3", "log_path": "/remote/run.log"}],
    )
    seen_rows = []
    stopped = []

    def fake_log_has_failure(path, row=None):
        seen_rows.append((path, row))
        return True

    def fake_stop(run_dir, run_id, *, reason):
        stopped.append((Path(run_dir), run_id))
        return Path(run_dir) / "run_status.tsv"

    monkeypatch.setattr(run_evidence, "log_has_failure", fake_log_has_failure)
    monkeypatch.setattr(adaptive_hparam, "stop_hparam_run", fake_stop)

    adaptive_hparam._stop_bad_running_runs(workflow_dir, round_dir, adaptive_hparam.load_recipe_with_base(recipe))

    assert seen_rows[0][0] == "/remote/run.log"
    assert seen_rows[0][1]["target"] == "ssh"
    assert seen_rows[0][1]["host"] == "baichuan3"
    assert stopped == [(round_dir, "run-000")]


def test_metric_based_running_stop_honors_grace(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=1)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    plan = json.loads((round_dir / "plan.json").read_text())
    run = plan["runs"][0]
    version = run["version"]
    run_dir = Path(run["runtime_dir"])
    run_dir.mkdir(parents=True)
    log_path = round_dir / "logs" / "run-000.log"
    log_path.parent.mkdir()
    log_path.write_text("still training\n")
    (workflow_dir / "adaptive" / "incumbents.tsv").write_text("objective_score\n0.73\n")
    stopped = []

    def fake_stop(run_dir, run_id, *, reason):
        stopped.append((Path(run_dir), run_id))
        return Path(run_dir) / "run_status.tsv"

    monkeypatch.setattr(adaptive_hparam, "stop_hparam_run", fake_stop)

    (run_dir / "run_manifest.json").write_text(json.dumps({"epoch": 0, "metrics": {"test_auroc": 0.6}}))
    (round_dir / "run_status.tsv").write_text(
        "step_id\trun_id\tversion\tstatus\tlog_path\tlaunched_at\truntime_dir\n"
        f"unit-hparam-tune\trun-000\t{version}\trunning\t{log_path}\t{manifests.utc_now()}\t{run['runtime_dir']}\n"
    )
    recipe_data = adaptive_hparam.load_recipe_with_base(recipe)

    adaptive_hparam._stop_bad_running_runs(workflow_dir, round_dir, recipe_data)

    assert stopped == []
    (run_dir / "run_manifest.json").write_text(json.dumps({"epoch": 2, "metrics": {"test_auroc": 0.6}}))
    (round_dir / "run_status.tsv").write_text(
        "step_id\trun_id\tversion\tstatus\tlog_path\tlaunched_at\truntime_dir\n"
        f"unit-hparam-tune\trun-000\t{version}\trunning\t{log_path}\t2000-01-01T00:00:00Z\t{run['runtime_dir']}\n"
    )

    adaptive_hparam._stop_bad_running_runs(workflow_dir, round_dir, recipe_data)

    assert stopped == [(round_dir, "run-000")]
