from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pandas as pd
import pytest
import yaml

from agent_tools import hparam
from agent_tools.hparam import monitor_hparam_trials


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _hparam_recipe(tmp_path: Path, *, execution: dict | None = None) -> Path:
    base = write_finetune_recipe(tmp_path)
    return write_yaml(
        tmp_path / "tune.yaml",
        {
            "schema_version": 1,
            "name": "unit_hparam",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base),
            "search": {
                "method": "grid",
                "max_trials": 1,
                "parameters": {"runtime.lr": [1e-6]},
            },
            "execution": execution or {},
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
                "train_val_test_policy": {
                    "value": "select on val",
                    "source": "explicit_recipe",
                },
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


def test_hparam_launch_dry_run_renders_ssh_conda_gpu_wandb_and_pid_paths(
    tmp_path: Path,
):
    recipe = _hparam_recipe(
        tmp_path,
        execution={
            "target": "ssh",
            "host": "baichuan3",
            "workdir": str(tmp_path / "plan"),
            "conda_env": "ywx",
            "gpu_pool": [6, 7],
            "gpus_per_trial": 2,
            "max_concurrent": 1,
            "wandb_project": "sleep2vec-unit-hparam",
            "wandb_group": "unit",
        },
    )
    plan_dir = tmp_path / "plan"

    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    result = _run("hparam-launch", "--plan-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert rows[0]["status"] == "planned"
    assert rows[0]["target"] == "ssh"
    assert rows[0]["host"] == "baichuan3"
    assert rows[0]["gpus"] == "6,7"
    assert "ssh baichuan3" in rows[0]["command"]
    assert "mkdir -p" in rows[0]["command"]
    assert "(nohup" in rows[0]["command"]
    assert "conda run --no-capture-output -n ywx" in rows[0]["command"]
    assert "CUDA_VISIBLE_DEVICES=6,7" in rows[0]["command"]
    assert "WANDB_PROJECT=sleep2vec-unit-hparam" in rows[0]["command"]
    assert rows[0]["log_path"].endswith("logs/trial_000.log")
    assert rows[0]["pid_path"].endswith("pids/trial_000.pid")
    assert not (plan_dir / "logs").exists()
    assert not (plan_dir / "pids").exists()


def test_hparam_launch_accepts_scalar_runtime_devices(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"]["devices"] = 2
    write_yaml(base_recipe, base_payload)
    plan_dir = tmp_path / "plan"

    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    result = _run("hparam-launch", "--plan-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert rows[0]["gpus"] == "2"
    assert "CUDA_VISIBLE_DEVICES=2" in rows[0]["command"]


def test_hparam_launch_resolves_relative_plan_dir_before_cd(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "relative_plan"

    plan = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_tools",
            "plan",
            "--recipe",
            str(recipe),
            "--output-dir",
            "relative_plan",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    launch = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_tools",
            "hparam-launch",
            "--plan-dir",
            "relative_plan",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )

    assert plan.returncode == 0, plan.stderr
    assert launch.returncode == 0, launch.stderr
    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert rows[0]["script"] == str(plan_dir / "trial_000.sh")
    assert rows[0]["log_path"] == str(plan_dir / "logs" / "trial_000.log")
    assert "relative_plan/relative_plan" not in rows[0]["command"]


def test_hparam_doctor_rejects_invalid_execution_target(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, execution={"target": "cluster"})

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 1
    assert "execution.target" in result.stdout


def test_hparam_monitor_handles_running_finished_and_failed_rows(tmp_path: Path):
    pid_path = tmp_path / "running.pid"
    pid_path.write_text(str(os.getpid()))
    missing_pid = tmp_path / "missing.pid"
    fail_pid = tmp_path / "fail.pid"
    fail_pid.write_text("999999999")
    fail_log = tmp_path / "fail.log"
    fail_log.write_text("Traceback\nRuntimeError: boom\n")
    with (tmp_path / "launch_manifest.tsv").open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            delimiter="\t",
            fieldnames=[
                "trial_id",
                "version",
                "target",
                "pid_path",
                "log_path",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trial_id": "running",
                "version": "v1",
                "target": "local",
                "pid_path": pid_path,
                "status": "launched",
            }
        )
        writer.writerow(
            {
                "trial_id": "missing",
                "version": "v2",
                "target": "local",
                "pid_path": missing_pid,
                "status": "launched",
            }
        )
        writer.writerow(
            {
                "trial_id": "failed",
                "version": "v3",
                "target": "local",
                "pid_path": fail_pid,
                "log_path": fail_log,
                "status": "launched",
            }
        )

    monitor_hparam_trials(tmp_path)

    status = {row["trial_id"]: row["status"] for row in _read_table(tmp_path / "trial_status.tsv")}
    assert status["running"] == "running"
    assert status["missing"] == "missing_pid"
    assert status["failed"] == "failed"


def test_hparam_monitor_launches_pending_trials_when_slots_free(tmp_path: Path, monkeypatch):
    (tmp_path / "plan.json").write_text(json.dumps({"recipe": {"execution": {"max_concurrent": 1}}}))
    dead_pid = tmp_path / "dead.pid"
    dead_pid.write_text("999999999")
    with (tmp_path / "launch_manifest.tsv").open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            delimiter="\t",
            fieldnames=[
                "trial_id",
                "version",
                "target",
                "pid_path",
                "command",
                "status",
                "launched_at",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trial_id": "trial_000",
                "version": "v0",
                "target": "local",
                "pid_path": dead_pid,
                "command": "run first",
                "status": "launched",
                "launched_at": "2026-01-01T00:00:00Z",
            }
        )
        writer.writerow(
            {
                "trial_id": "trial_001",
                "version": "v1",
                "target": "local",
                "command": "run pending",
                "status": "pending",
                "launched_at": "",
            }
        )
    started = []

    def fake_start(_execution, command):
        started.append(command)
        return "launched"

    monkeypatch.setattr(hparam, "_start_process", fake_start)

    monitor_hparam_trials(tmp_path)

    status = {row["trial_id"]: row for row in _read_table(tmp_path / "trial_status.tsv")}
    manifest = {row["trial_id"]: row for row in _read_table(tmp_path / "launch_manifest.tsv")}
    assert started == ["run pending"]
    assert status["trial_000"]["status"] == "finished"
    assert status["trial_001"]["status"] == "launched"
    assert manifest["trial_001"]["status"] == "launched"
    assert manifest["trial_001"]["launched_at"]


def test_hparam_monitor_health_is_opt_in(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "running.pid"
    pid_path.write_text("123")
    with (tmp_path / "launch_manifest.tsv").open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            delimiter="\t",
            fieldnames=["trial_id", "version", "target", "pid_path", "status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trial_id": "running",
                "version": "v1",
                "target": "local",
                "pid_path": pid_path,
                "status": "launched",
            }
        )
    monkeypatch.setattr(hparam, "_process_running", lambda row, pid: True)

    monitor_hparam_trials(tmp_path)

    row = _read_table(tmp_path / "trial_status.tsv")[0]
    assert "health_status" not in row


def test_hparam_monitor_health_classifies_compute_active(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "running.pid"
    pid_path.write_text("123")
    with (tmp_path / "launch_manifest.tsv").open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            delimiter="\t",
            fieldnames=[
                "trial_id",
                "version",
                "target",
                "pid_path",
                "log_path",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trial_id": "running",
                "version": "v1",
                "target": "local",
                "pid_path": pid_path,
                "status": "launched",
            }
        )
    monkeypatch.setattr(hparam, "_process_running", lambda row, pid: True)
    monkeypatch.setattr(hparam, "_gpu_summary", lambda row, pid: "123, GPU-1, 1024")
    monkeypatch.setattr(hparam, "_proc_io", lambda row, pid: {})
    monkeypatch.setattr(hparam, "_log_age_seconds", lambda path, row: None)
    monkeypatch.setattr(hparam, "_read_trial_progress", lambda run_dir, row: {"status": "missing"})

    monitor_hparam_trials(tmp_path, health=True)

    row = _read_table(tmp_path / "trial_status.tsv")[0]
    assert row["health_status"] == "compute_active"
    assert row["gpu_summary"] == "123, GPU-1, 1024"


def test_hparam_monitor_health_classifies_data_loading_from_io_delta(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "running.pid"
    pid_path.write_text("123")
    with (tmp_path / "launch_manifest.tsv").open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            delimiter="\t",
            fieldnames=["trial_id", "version", "target", "pid_path", "status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trial_id": "running",
                "version": "v1",
                "target": "local",
                "pid_path": pid_path,
                "status": "launched",
            }
        )
    (tmp_path / "trial_status.tsv").write_text(
        "trial_id\tstatus\tio_read_bytes\tio_write_bytes\tcheckpoint_count\nrunning\trunning\t100\t50\t0\n"
    )
    monkeypatch.setattr(hparam, "_process_running", lambda row, pid: True)
    monkeypatch.setattr(hparam, "_gpu_summary", lambda row, pid: "")
    monkeypatch.setattr(hparam, "_proc_io", lambda row, pid: {"read_bytes": 250, "write_bytes": 50})
    monkeypatch.setattr(hparam, "_log_age_seconds", lambda path, row: None)
    monkeypatch.setattr(hparam, "_read_trial_progress", lambda run_dir, row: {"status": "missing"})

    monitor_hparam_trials(tmp_path, health=True)

    row = _read_table(tmp_path / "trial_status.tsv")[0]
    assert row["health_status"] == "data_loading"
    assert row["io_read_delta_bytes"] == "150"


def test_hparam_monitor_health_classifies_stalled_and_unknown_remote(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "running.pid"
    pid_path.write_text("123")
    with (tmp_path / "launch_manifest.tsv").open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            delimiter="\t",
            fieldnames=["trial_id", "version", "target", "host", "pid_path", "status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trial_id": "stalled",
                "version": "v1",
                "target": "local",
                "pid_path": pid_path,
                "status": "launched",
            }
        )
        writer.writerow(
            {
                "trial_id": "remote",
                "version": "v2",
                "target": "ssh",
                "host": "baichuan3",
                "pid_path": pid_path,
                "status": "launched",
            }
        )

    def fake_running(row, pid):
        return None if row["trial_id"] == "remote" else True

    monkeypatch.setattr(hparam, "_process_running", fake_running)
    monkeypatch.setattr(hparam, "_gpu_summary", lambda row, pid: "")
    monkeypatch.setattr(hparam, "_proc_io", lambda row, pid: {"read_bytes": 100, "write_bytes": 50})
    monkeypatch.setattr(hparam, "_log_age_seconds", lambda path, row: 500)
    monkeypatch.setattr(hparam, "_read_trial_progress", lambda run_dir, row: {"status": "missing"})

    monitor_hparam_trials(tmp_path, health=True)

    status = {row["trial_id"]: row["health_status"] for row in _read_table(tmp_path / "trial_status.tsv")}
    assert status["stalled"] == "possibly_stalled"
    assert status["remote"] == "unknown_remote"


def test_hparam_monitor_health_requires_fresh_progress(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "running.pid"
    pid_path.write_text("123")
    with (tmp_path / "launch_manifest.tsv").open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            delimiter="\t",
            fieldnames=["trial_id", "version", "target", "pid_path", "status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trial_id": "running",
                "version": "v1",
                "target": "local",
                "pid_path": pid_path,
                "status": "launched",
            }
        )
    (tmp_path / "trial_status.tsv").write_text(
        "trial_id\tstatus\tprogress_processed\tprogress_updated_at\tcheckpoint_count\n"
        "running\trunning\t5\t2000-01-01T00:00:00Z\t0\n"
    )
    monkeypatch.setattr(hparam, "_process_running", lambda row, pid: True)
    monkeypatch.setattr(hparam, "_gpu_summary", lambda row, pid: "")
    monkeypatch.setattr(hparam, "_proc_io", lambda row, pid: {})
    monkeypatch.setattr(hparam, "_log_age_seconds", lambda path, row: 500)
    monkeypatch.setattr(
        hparam,
        "_read_trial_progress",
        lambda run_dir, row: {
            "status": "running",
            "processed": 5,
            "updated_at": "2000-01-01T00:00:00Z",
        },
    )

    monitor_hparam_trials(tmp_path, health=True)

    row = _read_table(tmp_path / "trial_status.tsv")[0]
    assert row["health_status"] == "possibly_stalled"


def test_hparam_remote_command_timeout_returns_unknown_remote(monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["ssh", "baichuan3", "ps"], 10)

    monkeypatch.setattr(hparam.subprocess, "run", fake_run)

    result = hparam._run_row_command({"target": "ssh", "host": "baichuan3"}, "ps")

    assert result.returncode == 124


def test_hparam_start_process_timeout_returns_launch_failed(monkeypatch):
    def fake_run(*_args, **kwargs):
        assert kwargs["timeout"] == hparam.LAUNCH_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(["bash", "-lc", "cmd"], kwargs["timeout"])

    monkeypatch.setattr(hparam.subprocess, "run", fake_run)

    assert hparam._start_process({}, "cmd") == "launch_failed"


def test_hparam_select_uses_fixed_epoch_checkpoint_not_best_alias(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run_dir = plan_dir / "log-finetune" / "unit_hparam-trial_000"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "epoch=11.ckpt").write_text("fixed")
    (ckpt_dir / "best-epoch=11.ckpt").write_text("alias")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": "unit_hparam-trial_000",
                "monitor": "val_ahi_pearson",
                "best_model_score": 0.71,
                "best_model_path": str(ckpt_dir / "best-epoch=11.ckpt"),
                "epoch": 11,
                "metrics": {"val_ahi_pearson": 0.71},
            }
        )
    )

    result = _run(
        "hparam-select",
        "--run-dir",
        str(plan_dir),
        "--metric",
        "val_ahi_pearson",
        "--mode",
        "max",
    )

    assert result.returncode == 0, result.stderr
    rows = _read_table(plan_dir / "candidate_ranking.csv")
    assert rows[0]["checkpoint_path"].endswith("epoch=11.ckpt")
    assert "best-epoch" not in rows[0]["checkpoint_path"]


def test_hparam_select_preserves_zero_padded_epoch_checkpoint(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run_dir = plan_dir / "log-finetune" / "unit_hparam-trial_000"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    fixed = ckpt_dir / "epoch=09-step=90.ckpt"
    fixed.write_text("fixed")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": "unit_hparam-trial_000",
                "monitor": "val_ahi_pearson",
                "best_model_score": 0.72,
                "best_model_path": str(ckpt_dir / "best-epoch=09-step=90.ckpt"),
                "epoch": 9,
                "metrics": {"val_ahi_pearson": 0.72},
            }
        )
    )

    result = _run(
        "hparam-select",
        "--run-dir",
        str(plan_dir),
        "--metric",
        "val_ahi_pearson",
        "--mode",
        "max",
    )

    assert result.returncode == 0, result.stderr
    rows = _read_table(plan_dir / "candidate_ranking.csv")
    assert rows[0]["checkpoint_path"] == str(fixed)


def test_hparam_external_eval_uses_trial_runtime_from_candidate_ranking(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["parameters"] = {"runtime.batch_size": [48]}
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"]["batch_size"] = 32
    write_yaml(base_recipe, base_payload)
    write_yaml(recipe, payload)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run_dir = plan_dir / "log-finetune" / "unit_hparam-trial_000"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    fixed = ckpt_dir / "epoch=11.ckpt"
    fixed.write_text("fixed")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": "unit_hparam-trial_000",
                "monitor": "val_ahi_pearson",
                "best_model_score": 0.71,
                "best_model_path": str(ckpt_dir / "best-epoch=11.ckpt"),
                "epoch": 11,
                "metrics": {"val_ahi_pearson": 0.71},
            }
        )
    )

    selected = _run(
        "hparam-select",
        "--run-dir",
        str(plan_dir),
        "--metric",
        "val_ahi_pearson",
        "--mode",
        "max",
    )
    assert selected.returncode == 0, selected.stderr
    rows = _read_table(plan_dir / "candidate_ranking.csv")
    assert "runtime.batch_size" not in rows[0]
    unlocked = _run(
        "hparam-external-eval",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(plan_dir / "candidate_ranking.csv"),
        "--unlock-final-test",
    )

    assert unlocked.returncode == 0, unlocked.stderr
    external_script = (plan_dir / "external_eval.sh").read_text()
    assert "--batch-size 48" in external_script


def test_hparam_external_eval_requires_unlock_and_only_replaces_data_fields(
    tmp_path: Path,
):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"].update(
        {
            "devices": [6, 7],
            "accelerator": "cpu",
            "device": "cpu",
            "batch_size": 32,
            "num_workers": 2,
            "precision": 32,
        }
    )
    write_yaml(base_recipe, base_payload)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    trial_config = plan_dir / "configs" / "trial_000.yaml"
    payload = yaml.safe_load(trial_config.read_text())
    payload["data"]["finetune_preset_path"] = "stale_preset.pkl"
    trial_config.write_text(yaml.safe_dump(payload))
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "trial_id,rank,config,checkpoint_path,runtime.batch_size\n"
        f"trial_000,1,{plan_dir / 'configs' / 'trial_000.yaml'},{tmp_path / 'epoch=1.ckpt'},48\n"  # noqa: E501
        f"trial_001,2,{plan_dir / 'configs' / 'trial_000.yaml'},{tmp_path / 'epoch=2.ckpt'},48\n"  # noqa: E501
        f"trial_002,3,{plan_dir / 'configs' / 'trial_000.yaml'},{tmp_path / 'epoch=3.ckpt'},48\n"  # noqa: E501
    )

    locked = _run("hparam-external-eval", "--run-dir", str(plan_dir), "--selected", str(selected))
    assert locked.returncode != 0
    unlocked = _run(
        "hparam-external-eval",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(selected),
        "--unlock-final-test",
        "--finetune-data-index",
        "external_index.csv",
    )

    assert unlocked.returncode == 0, unlocked.stderr
    original = yaml.safe_load((plan_dir / "configs" / "trial_000.yaml").read_text())
    external = yaml.safe_load((plan_dir / "external_eval_configs" / "trial_000_001_external.yaml").read_text())
    assert external["data"]["finetune_data_index"] == "external_index.csv"
    assert external["data"]["finetune_preset_path"] is None
    assert external["model"] == original["model"]
    external_script = (plan_dir / "external_eval.sh").read_text()
    assert f"cd {hparam._sh(hparam.REPO_ROOT)}" in external_script
    assert f"export PYTHONPATH={hparam._sh(hparam.REPO_ROOT)}${{PYTHONPATH:+:$PYTHONPATH}}" in external_script
    assert external_script.count("python -m sleep2vec.infer") == 1
    assert "--devices 6 7" in external_script
    assert "--accelerator cpu" in external_script
    assert "--device cpu" in external_script
    assert "--batch-size 48" in external_script
    assert "--num-workers 2" in external_script
    assert "--precision 32" in external_script

    kaldi_eval = _run(
        "hparam-external-eval",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(selected),
        "--unlock-final-test",
        "--kaldi-data-root",
        "/kaldi/root",
        "--kaldi-manifest",
        "test.jsonl",
    )
    assert kaldi_eval.returncode == 0, kaldi_eval.stderr
    kaldi_external = yaml.safe_load((plan_dir / "external_eval_configs" / "trial_000_001_external.yaml").read_text())
    assert kaldi_external["data"]["backend"] == "kaldi"
    assert kaldi_external["data"]["kaldi_data_root"] == "/kaldi/root"
    assert kaldi_external["data"]["kaldi_manifest"] == "test.jsonl"
    assert kaldi_external["data"]["finetune_data_index"] is None
    assert kaldi_external["data"]["finetune_preset_path"] is None

    top_two = _run(
        "hparam-external-eval",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(selected),
        "--unlock-final-test",
        "--top-k",
        "2",
    )
    assert top_two.returncode == 0, top_two.stderr
    assert len(_read_table(plan_dir / "external_eval_manifest.tsv")) == 2

    all_candidates = _run(
        "hparam-external-eval",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(selected),
        "--unlock-final-test",
        "--all-candidates",
    )
    assert all_candidates.returncode == 0, all_candidates.stderr
    assert len(_read_table(plan_dir / "external_eval_manifest.tsv")) == 3


def test_hparam_export_logits_requires_unlock_and_writes_stable_paths(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    trial_config = plan_dir / "configs" / "trial_000.yaml"
    payload = yaml.safe_load(trial_config.read_text())
    payload["data"]["finetune_preset_path"] = "stale_preset.pkl"
    trial_config.write_text(yaml.safe_dump(payload))
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "trial_id,rank,config,checkpoint_path\n"
        f"trial_000,1,{plan_dir / 'configs' / 'trial_000.yaml'},{tmp_path / 'epoch=1.ckpt'}\n"  # noqa: E501
        f"trial_001,2,{plan_dir / 'configs' / 'trial_000.yaml'},{tmp_path / 'epoch=2.ckpt'}\n"  # noqa: E501
    )

    locked = _run("hparam-export-logits", "--run-dir", str(plan_dir), "--selected", str(selected))
    assert locked.returncode != 0
    unlocked = _run(
        "hparam-export-logits",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(selected),
        "--unlock-final-test",
        "--top-k",
        "2",
        "--test-finetune-data-index",
        "external_test.csv",
    )

    assert unlocked.returncode == 0, unlocked.stderr
    rows = _read_table(plan_dir / "logits_export_manifest.tsv")
    assert len(rows) == 2
    assert rows[0]["val_logits_path"].endswith("logits_exports/trial_000_001_val_logits.csv")
    assert rows[0]["test_logits_path"].endswith("logits_exports/trial_000_001_test_logits.csv")
    assert "python -m sleep2vec.infer" in rows[0]["val_infer_command"]
    assert "--eval-split val" in rows[0]["val_infer_command"]
    assert "--eval-split test" in rows[0]["test_infer_command"]
    test_config = yaml.safe_load(Path(rows[0]["test_config"]).read_text())
    assert test_config["data"]["finetune_data_index"] == "external_test.csv"
    assert test_config["data"]["finetune_preset_path"] is None
    script = (plan_dir / "logits_export.sh").read_text()
    assert "hparam-export-logits" in script
    assert "--execute" in script
    assert "--unlock-final-test" in script

    kaldi_logits = _run(
        "hparam-export-logits",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(selected),
        "--unlock-final-test",
        "--val-kaldi-data-root",
        "/kaldi/val",
        "--val-kaldi-manifest",
        "val.jsonl",
        "--test-kaldi-data-root",
        "/kaldi/test",
        "--test-kaldi-manifest",
        "test.jsonl",
    )
    assert kaldi_logits.returncode == 0, kaldi_logits.stderr
    rows = _read_table(plan_dir / "logits_export_manifest.tsv")
    val_config = yaml.safe_load(Path(rows[0]["val_config"]).read_text())
    test_config = yaml.safe_load(Path(rows[0]["test_config"]).read_text())
    assert val_config["data"]["backend"] == "kaldi"
    assert val_config["data"]["kaldi_data_root"] == "/kaldi/val"
    assert val_config["data"]["kaldi_manifest"] == "val.jsonl"
    assert val_config["data"]["finetune_data_index"] is None
    assert val_config["data"]["finetune_preset_path"] is None
    assert test_config["data"]["backend"] == "kaldi"
    assert test_config["data"]["kaldi_data_root"] == "/kaldi/test"
    assert test_config["data"]["kaldi_manifest"] == "test.jsonl"
    assert test_config["data"]["finetune_data_index"] is None
    assert test_config["data"]["finetune_preset_path"] is None


def test_hparam_export_logits_execute_uses_manifest_paths(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "trial_id,rank,config,checkpoint_path\n"
        f"trial_000,1,{plan_dir / 'configs' / 'trial_000.yaml'},{tmp_path / 'epoch=1.ckpt'}\n"  # noqa: E501
    )
    calls = []

    def _fake_run_logit_export(recipe, **kwargs):
        calls.append(kwargs)
        Path(kwargs["output_path"]).write_text("label,logit\n0,-1.0\n1,1.0\n")

    monkeypatch.setattr(hparam, "_run_logit_export", _fake_run_logit_export)

    manifest = hparam.export_hparam_logits(
        plan_dir,
        selected,
        unlock_final_test=True,
        execute=True,
        batch_size=4,
        devices=[0],
    )

    rows = _read_table(manifest)
    assert len(calls) == 2
    assert calls[0]["eval_split"] == "val"
    assert calls[0]["batch_size"] == 4
    assert calls[0]["devices"] == [0]
    assert calls[1]["eval_split"] == "test"
    assert Path(rows[0]["val_logits_path"]).exists()
    assert Path(rows[0]["test_logits_path"]).exists()


def test_hparam_checkpoint_scan_ranks_history_fixed_epoch_checkpoints(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run_dir = plan_dir / "log-finetune" / "unit_hparam-trial_000"
    ckpt_dir = run_dir / "checkpoints"
    history_dir = run_dir / "wandb" / "run-1" / "files"
    ckpt_dir.mkdir(parents=True)
    history_dir.mkdir(parents=True)
    (ckpt_dir / "epoch=13.ckpt").write_text("fixed13")
    (ckpt_dir / "epoch=20.ckpt").write_text("fixed20")
    (ckpt_dir / "best-epoch=20.ckpt").write_text("alias")
    (history_dir / "wandb-history.jsonl").write_text(
        json.dumps({"epoch": 2, "val_auroc": 0.99})
        + "\n"
        + json.dumps({"epoch": 13, "val_auroc": 0.72})
        + "\n"
        + json.dumps({"epoch": 20, "val_auroc": 0.81})
        + "\n"
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": "unit_hparam-trial_000",
                "best_model_path": str(ckpt_dir / "best-epoch=20.ckpt"),
                "epoch": 20,
                "metrics": {"val_auroc": 0.5},
            }
        )
    )

    result = _run(
        "hparam-checkpoint-scan",
        "--run-dir",
        str(plan_dir),
        "--metric",
        "val_auroc",
        "--mode",
        "max",
    )

    assert result.returncode == 0, result.stderr
    rows = _read_table(plan_dir / "checkpoint_ranking.csv")
    assert rows[0]["epoch"] == "20"
    assert rows[0]["score"] == "0.81"
    assert rows[0]["checkpoint_path"].endswith("epoch=20.ckpt")
    assert "best-epoch" not in rows[0]["checkpoint_path"]
    assert rows[0]["source"] == "history"
    assert {row["epoch"] for row in rows} == {"13", "20"}


def test_hparam_threshold_and_ensemble_compute_binary_metrics(tmp_path: Path):
    val_a = tmp_path / "val_a.csv"
    test_a = tmp_path / "test_a.csv"
    val_b = tmp_path / "val_b.csv"
    test_b = tmp_path / "test_b.csv"
    val_c = tmp_path / "val_c.csv"
    test_c = tmp_path / "test_c.csv"
    pd.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]}).to_csv(val_a, index=False)
    pd.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.2, 0.8, 0.9]}).to_csv(test_a, index=False)
    pd.DataFrame({"label": [0, 0, 1, 1], "prob": [0.2, 0.8, 0.7, 0.6]}).to_csv(val_b, index=False)
    pd.DataFrame({"label": [0, 0, 1, 1], "prob": [0.2, 0.8, 0.7, 0.6]}).to_csv(test_b, index=False)
    pd.DataFrame({"label": [0, 0, 1, 1], "prob": [0.8, 0.7, 0.2, 0.1]}).to_csv(val_c, index=False)
    pd.DataFrame({"label": [0, 0, 1, 1], "prob": [0.8, 0.7, 0.2, 0.1]}).to_csv(test_c, index=False)
    selected = tmp_path / "selected.csv"
    selected.write_text(
        "trial_id,val_predictions_path,test_predictions_path\n"
        f"trial_000,{val_a},{test_a}\n"
        f"trial_001,{val_b},{test_b}\n"
        f"trial_002,{val_c},{test_c}\n"
    )

    threshold = _run("hparam-threshold", "--run-dir", str(tmp_path), "--selected", str(selected))
    ensemble = _run("hparam-ensemble", "--run-dir", str(tmp_path), "--candidates", str(selected))

    assert threshold.returncode == 0, threshold.stderr
    assert ensemble.returncode == 0, ensemble.stderr
    threshold_rows = _read_table(tmp_path / "threshold_summary.csv")
    assert float(threshold_rows[0]["test_auroc"]) == 1.0
    assert "test_f1" in threshold_rows[0]
    ensemble_rows = _read_table(tmp_path / "ensemble_summary.csv")
    assert ensemble_rows[0]["n_models"] == "3"

    ensemble_search = _run(
        "hparam-ensemble",
        "--run-dir",
        str(tmp_path),
        "--candidates",
        str(selected),
        "--search-combinations",
        "--max-size",
        "2",
        "--metric",
        "exploratory_test_auroc",
        "--top-k",
        "6",
    )

    assert ensemble_search.returncode == 0, ensemble_search.stderr
    search_rows = _read_table(tmp_path / "ensemble_summary.csv")
    assert len(search_rows) == 6
    assert search_rows[0]["rank"] == "1"
    assert any(row["n_models"] == "2" for row in search_rows)


def test_hparam_threshold_and_ensemble_read_repo_prediction_csv_lists(tmp_path: Path):
    val_seq = tmp_path / "val_seq.csv"
    test_seq = tmp_path / "test_seq.csv"
    val_ahi = tmp_path / "val_ahi.csv"
    test_ahi = tmp_path / "test_ahi.csv"
    val_logit = tmp_path / "val_logit.csv"
    test_logit = tmp_path / "test_logit.csv"
    val_custom = tmp_path / "val_custom.csv"
    test_custom = tmp_path / "test_custom.csv"
    pd.DataFrame(
        {
            "path": ["a.npz", "b.npz"],
            "groundtruth": [json.dumps([0, 0]), json.dumps([1, 1])],
            "prob_1": [json.dumps([0.1, 0.2]), json.dumps([0.8, 0.9])],
        }
    ).to_csv(val_seq, index=False)
    pd.DataFrame(
        {
            "path": ["a.npz", "b.npz"],
            "groundtruth": [json.dumps([0, 0]), json.dumps([1, 1])],
            "prob_1": [json.dumps([0.1, 0.2]), json.dumps([0.8, 0.9])],
        }
    ).to_csv(test_seq, index=False)
    pd.DataFrame(
        {
            "path": ["a.npz", "b.npz"],
            "groundtruth": [json.dumps([0, 0]), json.dumps([1, 1])],
            "prob": [json.dumps([0.1, 0.2]), json.dumps([0.8, 0.9])],
        }
    ).to_csv(val_ahi, index=False)
    pd.DataFrame(
        {
            "path": ["a.npz", "b.npz"],
            "groundtruth": [json.dumps([0, 0]), json.dumps([1, 1])],
            "prob": [json.dumps([0.1, 0.2]), json.dumps([0.8, 0.9])],
        }
    ).to_csv(test_ahi, index=False)
    pd.DataFrame(
        {
            "path": ["a.npz", "b.npz"],
            "groundtruth": [json.dumps([0, 0]), json.dumps([1, 1])],
            "logit": [json.dumps([-2.0, -1.0]), json.dumps([1.0, 2.0])],
        }
    ).to_csv(val_logit, index=False)
    pd.DataFrame(
        {
            "path": ["a.npz", "b.npz"],
            "groundtruth": [json.dumps([0, 0]), json.dumps([1, 1])],
            "logit": [json.dumps([-2.0, -1.0]), json.dumps([1.0, 2.0])],
        }
    ).to_csv(test_logit, index=False)
    pd.DataFrame(
        {
            "path": ["a.npz", "b.npz"],
            "custom_label": [json.dumps([0, 0]), json.dumps([1, 1])],
            "prob": [json.dumps([0.1, 0.2]), json.dumps([0.8, 0.9])],
        }
    ).to_csv(val_custom, index=False)
    pd.DataFrame(
        {
            "path": ["a.npz", "b.npz"],
            "custom_label": [json.dumps([0, 0]), json.dumps([1, 1])],
            "prob": [json.dumps([0.1, 0.2]), json.dumps([0.8, 0.9])],
        }
    ).to_csv(test_custom, index=False)
    selected = tmp_path / "selected_repo_predictions.csv"
    selected.write_text(
        "trial_id,label_name,val_predictions_path,test_predictions_path\n"
        f"trial_seq,,{val_seq},{test_seq}\n"
        f"trial_ahi,,{val_ahi},{test_ahi}\n"
        f"trial_logit,,{val_logit},{test_logit}\n"
        f"trial_custom,custom_label,{val_custom},{test_custom}\n"
    )

    threshold = _run("hparam-threshold", "--run-dir", str(tmp_path), "--selected", str(selected))
    ensemble = _run("hparam-ensemble", "--run-dir", str(tmp_path), "--candidates", str(selected))

    assert threshold.returncode == 0, threshold.stderr
    threshold_rows = _read_table(tmp_path / "threshold_summary.csv")
    assert len(threshold_rows) == 4
    assert all(float(row["test_auroc"]) == 1.0 for row in threshold_rows)
    assert all(float(row["test_accuracy"]) == 1.0 for row in threshold_rows)
    assert ensemble.returncode == 0, ensemble.stderr
    ensemble_rows = _read_table(tmp_path / "ensemble_summary.csv")
    assert float(ensemble_rows[0]["exploratory_test_auroc"]) == 1.0


def test_hparam_ensemble_aligns_predictions_by_sample_identity(tmp_path: Path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame(
        {
            "path": ["a.npz", "b.npz", "c.npz", "d.npz"],
            "token_start": [0, 0, 0, 0],
            "groundtruth": [0, 0, 1, 1],
            "prob": [0.1, 0.2, 0.8, 0.9],
        }
    ).to_csv(first, index=False)
    pd.DataFrame(
        {
            "path": ["b.npz", "a.npz", "d.npz", "c.npz"],
            "token_start": [0, 0, 0, 0],
            "groundtruth": [0, 0, 1, 1],
            "prob": [0.6, 0.1, 0.8, 0.9],
        }
    ).to_csv(second, index=False)

    y, p = hparam._average_binary_predictions(
        [
            hparam._read_binary_predictions(first),
            hparam._read_binary_predictions(second),
        ]
    )

    assert y == [0, 0, 1, 1]
    assert p == pytest.approx([0.1, 0.4, 0.85, 0.85])


def test_hparam_export_logits_copy_accepts_probability_prediction_csv(tmp_path: Path):
    prediction = tmp_path / "predictions.csv"
    output = tmp_path / "copied.csv"
    pd.DataFrame({"path": ["a.npz", "b.npz"], "groundtruth": [0, 1], "prob": [0.2, 0.8]}).to_csv(
        prediction, index=False
    )

    hparam._copy_logits_csv(prediction, output)

    copied = pd.read_csv(output)
    assert list(copied.columns) == ["path", "groundtruth", "prob"]
