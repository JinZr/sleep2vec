from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pandas as pd
import yaml

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
            "search": {"method": "grid", "max_trials": 1, "parameters": {"runtime.lr": [1e-6]}},
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
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


def test_hparam_launch_dry_run_renders_ssh_conda_gpu_wandb_and_pid_paths(tmp_path: Path):
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
            "wandb_project": "sleep2vec-depression-matched",
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
    assert "conda run --no-capture-output -n ywx" in rows[0]["command"]
    assert "CUDA_VISIBLE_DEVICES=6,7" in rows[0]["command"]
    assert "WANDB_PROJECT=sleep2vec-depression-matched" in rows[0]["command"]
    assert rows[0]["log_path"].endswith("logs/trial_000.log")
    assert rows[0]["pid_path"].endswith("pids/trial_000.pid")


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
            fieldnames=["trial_id", "version", "target", "pid_path", "log_path", "status"],
        )
        writer.writeheader()
        writer.writerow(
            {"trial_id": "running", "version": "v1", "target": "local", "pid_path": pid_path, "status": "launched"}
        )
        writer.writerow(
            {"trial_id": "missing", "version": "v2", "target": "local", "pid_path": missing_pid, "status": "launched"}
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

    result = _run("hparam-select", "--run-dir", str(plan_dir), "--metric", "val_ahi_pearson", "--mode", "max")

    assert result.returncode == 0, result.stderr
    rows = _read_table(plan_dir / "candidate_ranking.csv")
    assert rows[0]["checkpoint_path"].endswith("epoch=11.ckpt")
    assert "best-epoch" not in rows[0]["checkpoint_path"]


def test_hparam_external_eval_requires_unlock_and_only_replaces_data_fields(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "trial_id,rank,config,checkpoint_path\n"
        f"trial_000,1,{plan_dir / 'configs' / 'trial_000.yaml'},{tmp_path / 'epoch=1.ckpt'}\n"
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
    external = yaml.safe_load((plan_dir / "external_eval_configs" / "trial_000_external.yaml").read_text())
    assert external["data"]["finetune_data_index"] == "external_index.csv"
    assert external["model"] == original["model"]
    assert "python -m sleep2vec.infer" in (plan_dir / "external_eval.sh").read_text()


def test_hparam_threshold_and_ensemble_compute_binary_metrics(tmp_path: Path):
    val = tmp_path / "val.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.4, 0.6, 0.9]}).to_csv(val, index=False)
    pd.DataFrame({"label": [0, 1, 1, 0], "prob": [0.2, 0.8, 0.7, 0.3]}).to_csv(test, index=False)
    selected = tmp_path / "selected.csv"
    selected.write_text(
        "trial_id,val_predictions_path,test_predictions_path\n" f"trial_000,{val},{test}\n" f"trial_001,{val},{test}\n"
    )

    threshold = _run("hparam-threshold", "--run-dir", str(tmp_path), "--selected", str(selected))
    ensemble = _run("hparam-ensemble", "--run-dir", str(tmp_path), "--candidates", str(selected))

    assert threshold.returncode == 0, threshold.stderr
    assert ensemble.returncode == 0, ensemble.stderr
    threshold_rows = _read_table(tmp_path / "threshold_summary.csv")
    assert float(threshold_rows[0]["test_auroc"]) == 1.0
    assert "test_f1" in threshold_rows[0]
    ensemble_rows = _read_table(tmp_path / "ensemble_summary.csv")
    assert ensemble_rows[0]["n_models"] == "2"
