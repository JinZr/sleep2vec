from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import yaml

from agent_tools import adaptive_hparam


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
            "schema_version": 1,
            "name": "unit_adaptive",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": base.name if relative_base else str(base),
            "search": {
                "method": "grid",
                "max_trials": 1,
                "parameters": {"runtime.lr": [1e-6], "yaml:/model/head/name": ["classification"]},
            },
            "adaptive": {
                "enabled": True,
                "objective_metric": "test_auroc",
                "objective_mode": "max",
                "test_feedback_for_selection": test_feedback,
                "max_rounds": max_rounds,
                "max_trials_total": 4,
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
    plan = json.loads((round_dir / "plan.json").read_text())
    trial = plan["trials"][0]
    run_dir = round_dir / "log-finetune" / trial["version"]
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "epoch=3.ckpt").write_text("checkpoint")
    (ckpt_dir / "best-epoch=3.ckpt").write_text("alias")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": trial["version"],
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


def test_adaptive_init_creates_round_zero_without_modifying_original_recipe(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    before = recipe.read_text()
    workflow_dir = tmp_path / "workflow"

    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir))

    assert result.returncode == 0, result.stderr
    assert recipe.read_text() == before
    assert (workflow_dir / "adaptive" / "workflow.json").exists()
    assert (workflow_dir / "adaptive" / "rounds" / "round_000" / "plan.json").exists()
    assert (workflow_dir / "adaptive" / "trial_registry.tsv").exists()
    assert "adaptive_init" in (workflow_dir / "adaptive" / "events.jsonl").read_text()


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


def test_adaptive_step_dry_run_writes_suggestion_and_supersede_event_without_round_one_plan(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    (round_dir / "launch_manifest.tsv").write_text(
        "trial_id\tversion\tstatus\ttarget\tpid_path\tlog_path\n"
        "trial_000\tunit_adaptive-round-000-trial_000\tplanned\tlocal\t\t\n"
    )

    result = _run("hparam-adaptive-step", "--workflow-dir", str(workflow_dir))

    assert result.returncode == 0, result.stderr
    assert (workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").exists()
    assert not (workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").exists()
    events = (workflow_dir / "adaptive" / "events.jsonl").read_text()
    assert "supersede_pending_trial" in events
    assert "adaptive_step_dry_run" in events


def test_adaptive_step_blocks_suggestion_without_scored_objective(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path, max_rounds=3)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0

    result = _run("hparam-adaptive-step", "--workflow-dir", str(workflow_dir))

    assert result.returncode != 0
    assert "No digest rows with finite test_auroc" in result.stderr
    assert "suggest_blocked" in (workflow_dir / "adaptive" / "events.jsonl").read_text()
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

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_trials", fake_launch)

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    suggestion = yaml.safe_load((workflow_dir / "adaptive" / "suggestions" / "round_001.yaml").read_text())
    assert Path(suggestion["base_recipe"]).is_absolute()
    assert (workflow_dir / "adaptive" / "rounds" / "round_001" / "plan.json").exists()
    assert launched == [(workflow_dir / "adaptive" / "rounds" / "round_001", False)]


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


def test_adaptive_step_execute_stops_bad_running_trial_through_recorded_manifest(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=1)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    _write_fake_manifest(workflow_dir, score=0.73)
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    pid_path = round_dir / "pids" / "trial_000.pid"
    log_path = round_dir / "logs" / "trial_000.log"
    pid_path.parent.mkdir()
    log_path.parent.mkdir()
    pid_path.write_text(str(os.getpid()))
    log_path.write_text("Traceback\nRuntimeError: failed\n")
    (round_dir / "launch_manifest.tsv").write_text(
        "trial_id\tversion\tstatus\ttarget\tpid_path\tlog_path\n"
        f"trial_000\tunit_adaptive-round-000-trial_000\tlaunched\tlocal\t{pid_path}\t{log_path}\n"
    )
    stopped = []

    def fake_stop(run_dir, trial_id):
        stopped.append((Path(run_dir), trial_id))
        return Path(run_dir) / "trial_status.tsv"

    monkeypatch.setattr(adaptive_hparam, "stop_hparam_trial", fake_stop)

    adaptive_hparam.adaptive_step(workflow_dir, execute=True)

    assert stopped == [(round_dir, "trial_000")]
    assert "stop_bad_running_trial" in (workflow_dir / "adaptive" / "events.jsonl").read_text()


def test_running_stop_passes_remote_status_row_to_failure_log_check(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=1)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    (round_dir / "trial_status.tsv").write_text(
        "trial_id\tversion\tstatus\ttarget\thost\tlog_path\n"
        "trial_000\tunit_adaptive-round-000-trial_000\trunning\tssh\tbaichuan3\t/remote/trial.log\n"
    )
    seen_rows = []
    stopped = []

    def fake_log_has_failure(path, row=None):
        seen_rows.append((path, row))
        return True

    def fake_stop(run_dir, trial_id):
        stopped.append((Path(run_dir), trial_id))
        return Path(run_dir) / "trial_status.tsv"

    monkeypatch.setattr(adaptive_hparam, "_log_has_failure", fake_log_has_failure)
    monkeypatch.setattr(adaptive_hparam, "stop_hparam_trial", fake_stop)

    adaptive_hparam._stop_bad_running_trials(workflow_dir, round_dir, adaptive_hparam.load_recipe_with_base(recipe))

    assert seen_rows[0][0] == "/remote/trial.log"
    assert seen_rows[0][1]["target"] == "ssh"
    assert seen_rows[0][1]["host"] == "baichuan3"
    assert stopped == [(round_dir, "trial_000")]


def test_metric_based_running_stop_honors_grace(tmp_path: Path, monkeypatch):
    recipe = _adaptive_recipe(tmp_path, max_rounds=1)
    workflow_dir = tmp_path / "workflow"
    assert _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow_dir)).returncode == 0
    round_dir = workflow_dir / "adaptive" / "rounds" / "round_000"
    version = "unit_adaptive-round-000-trial_000"
    run_dir = round_dir / "log-finetune" / version
    run_dir.mkdir(parents=True)
    log_path = round_dir / "logs" / "trial_000.log"
    log_path.parent.mkdir()
    log_path.write_text("still training\n")
    (workflow_dir / "adaptive" / "incumbents.tsv").write_text("objective_score\n0.73\n")
    stopped = []

    def fake_stop(run_dir, trial_id):
        stopped.append((Path(run_dir), trial_id))
        return Path(run_dir) / "trial_status.tsv"

    monkeypatch.setattr(adaptive_hparam, "stop_hparam_trial", fake_stop)

    (run_dir / "run_manifest.json").write_text(json.dumps({"epoch": 0, "metrics": {"test_auroc": 0.6}}))
    (round_dir / "trial_status.tsv").write_text(
        "trial_id\tversion\tstatus\tlog_path\tlaunched_at\n"
        f"trial_000\t{version}\trunning\t{log_path}\t{adaptive_hparam._now()}\n"
    )
    recipe_data = adaptive_hparam.load_recipe_with_base(recipe)

    adaptive_hparam._stop_bad_running_trials(workflow_dir, round_dir, recipe_data)

    assert stopped == []
    (run_dir / "run_manifest.json").write_text(json.dumps({"epoch": 2, "metrics": {"test_auroc": 0.6}}))
    (round_dir / "trial_status.tsv").write_text(
        "trial_id\tversion\tstatus\tlog_path\tlaunched_at\n"
        f"trial_000\t{version}\trunning\t{log_path}\t2000-01-01T00:00:00Z\n"
    )

    adaptive_hparam._stop_bad_running_trials(workflow_dir, round_dir, recipe_data)

    assert stopped == [(round_dir, "trial_000")]
