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
        f"trial_001,2,{plan_dir / 'configs' / 'trial_000.yaml'},{tmp_path / 'epoch=2.ckpt'}\n"
        f"trial_002,3,{plan_dir / 'configs' / 'trial_000.yaml'},{tmp_path / 'epoch=3.ckpt'}\n"
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
    assert external["model"] == original["model"]
    assert (plan_dir / "external_eval.sh").read_text().count("python -m sleep2vec.infer") == 1

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
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "trial_id,rank,config,checkpoint_path\n"
        f"trial_000,1,{plan_dir / 'configs' / 'trial_000.yaml'},{tmp_path / 'epoch=1.ckpt'}\n"
        f"trial_001,2,{plan_dir / 'configs' / 'trial_000.yaml'},{tmp_path / 'epoch=2.ckpt'}\n"
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
    script = (plan_dir / "logits_export.sh").read_text()
    assert "hparam-export-logits" in script
    assert "--execute" in script
    assert "--unlock-final-test" in script


def test_hparam_export_logits_execute_uses_manifest_paths(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "trial_id,rank,config,checkpoint_path\n"
        f"trial_000,1,{plan_dir / 'configs' / 'trial_000.yaml'},{tmp_path / 'epoch=1.ckpt'}\n"
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

    result = _run("hparam-checkpoint-scan", "--run-dir", str(plan_dir), "--metric", "val_auroc", "--mode", "max")

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
