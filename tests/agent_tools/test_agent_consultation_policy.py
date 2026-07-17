from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import config_payload, survival_config_payload, write_finetune_recipe, write_yaml
import yaml

from agent_tools.configs import config_summary
from agent_tools.decisions import DecisionStatus, evaluate_consultation_gates
from agent_tools.plans import evaluate_recipe
from agent_tools.recipes import load_consultation_policy


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], cwd=cwd, text=True, capture_output=True)


def test_doctor_returns_exit_2_when_label_name_missing_for_finetune(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"), cwd=Path.cwd())

    assert result.returncode == 2
    assert "Status: NEEDS_USER_INPUT" in result.stdout
    assert "label_name" in result.stdout


def test_doctor_without_output_dir_is_read_only(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    payload = yaml.safe_load(recipe.read_text())
    payload["name"] = f"doctor_read_only_{tmp_path.name}"
    write_yaml(recipe, payload)
    implicit_output = Path.cwd() / "artifacts" / "agent_context" / payload["name"]
    assert not implicit_output.exists()


def test_remote_deferred_config_path_warns_without_local_dummy_config(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["config"] = "/wujidata/example/config.yaml"
    payload["inputs"]["data_backend"] = "npz"
    payload["decisions"]["required_channels"] = {"value": ["ppg", "ahi", "stage5"], "source": "explicit_recipe"}
    payload["execution"] = {"target": "ssh", "host": "baichuan3", "path_context": "remote", "path_validation": "defer"}
    write_yaml(recipe, payload)

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"), cwd=Path.cwd())

    assert result.returncode == 0
    assert "Status: WARN" in result.stdout
    assert "path validation deferred for remote path" in result.stdout


def test_remote_deferred_survival_sidecars_do_not_require_local_files(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\nx.npz,train,60,001,1\n")
    config_payload_data = survival_config_payload(
        index,
        {
            "disease_columns_index": "/wujidata/survival/disease_columns.txt",
            "event_time_index": "/wujidata/survival/event_time.csv",
            "is_event_index": "/wujidata/survival/is_event.csv",
            "has_label_index": "/wujidata/survival/has_label.csv",
        },
    )
    config_payload_data["data"]["finetune_data_index"] = "/wujidata/survival/index.csv"
    config = write_yaml(tmp_path / "survival.yaml", config_payload_data)
    recipe = write_yaml(
        tmp_path / "recipe.yaml",
        {
            "name": "remote_survival",
            "task": "finetune",
            "variant": "sleep2vec",
            "inputs": {"config": str(config), "label_name": "incident_cox", "pretrained_backbone_path": None},
            "runtime": {"devices": [0]},
            "artifacts": {"results_csv_path": str(tmp_path / "results.csv"), "version_name": "remote"},
            "evaluation_policy": {
                "selection_metric": "val_loss",
                "selection_mode": "min",
                "selection_split": "val",
                "external_test_locked": True,
                "test_after_fit": False,
            },
            "execution": {"target": "ssh", "host": "baichuan3"},
            "decisions": {
                "task": {"value": "finetune", "source": "explicit_recipe"},
                "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
                "pretrained_backbone_path": {
                    "value": None,
                    "source": "explicit_recipe",
                    "meaning": "train from scratch",
                },
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"), cwd=Path.cwd())

    assert result.returncode == 0
    assert "Status: WARN" in result.stdout
    assert "survival_sidecars" not in result.stdout
    assert "path validation deferred for remote path" in result.stdout


def test_survival_preset_does_not_require_sidecar_files(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\nx.npz,train,60,001,1\n")
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(b"preset")
    config_payload_data = survival_config_payload(
        index,
        {
            "disease_columns_index": "/path/to/disease_columns.txt",
            "event_time_index": "/path/to/event_time.csv",
            "is_event_index": "/path/to/is_event.csv",
            "has_label_index": "/path/to/has_label.csv",
        },
    )
    config_payload_data["data"]["finetune_preset_path"] = str(preset)
    config = write_yaml(tmp_path / "survival_preset.yaml", config_payload_data)
    recipe = write_yaml(
        tmp_path / "survival_preset_recipe.yaml",
        {
            "name": "survival_preset",
            "task": "finetune",
            "variant": "sleep2vec",
            "inputs": {"config": str(config), "label_name": "incident_cox", "pretrained_backbone_path": None},
            "runtime": {"devices": [0]},
            "artifacts": {"results_csv_path": str(tmp_path / "results.csv"), "version_name": "preset"},
            "evaluation_policy": {
                "selection_metric": "val_loss",
                "selection_mode": "min",
                "selection_split": "val",
                "external_test_locked": True,
                "test_after_fit": False,
            },
            "decisions": {
                "task": {"value": "finetune", "source": "explicit_recipe"},
                "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
                "pretrained_backbone_path": {
                    "value": None,
                    "source": "explicit_recipe",
                    "meaning": "train from scratch",
                },
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"), cwd=Path.cwd())

    assert result.returncode == 0
    assert "survival_sidecars" not in result.stdout


def test_local_missing_config_path_still_fails(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["config"] = str(tmp_path / "missing.yaml")
    write_yaml(recipe, payload)

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"), cwd=Path.cwd())

    assert result.returncode == 1
    assert "Required input path does not exist" in result.stdout


def test_local_config_path_expands_home_before_validation(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration,ppg_mask,ah_event_mask,stage_mask\nx.npz,train,60,1,1,1\n")
    write_yaml(home / "config.yaml", config_payload(index))
    monkeypatch.setenv("HOME", str(home))
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["config"] = "~/config.yaml"
    policy = load_consultation_policy()

    report = evaluate_consultation_gates("finetune", payload, config_summary("~/config.yaml"), {}, policy)

    assert report.exit_code == 0


def test_remote_ssh_path_validation_uses_short_test_command(tmp_path: Path, monkeypatch):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["config"] = "/wujidata/example/config.yaml"
    payload["inputs"]["data_backend"] = "npz"
    payload["decisions"]["required_channels"] = {"value": ["ppg", "ahi", "stage5"], "source": "explicit_recipe"}
    payload["execution"] = {"target": "ssh", "host": "baichuan3", "path_context": "remote", "path_validation": "ssh"}
    policy = load_consultation_policy()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("agent_tools.decision_paths.subprocess.run", fake_run)

    report = evaluate_consultation_gates("finetune", payload, None, {}, policy)

    assert report.exit_code == 0
    assert calls == [["ssh", "baichuan3", "test -e /wujidata/example/config.yaml"]]


def test_remote_ssh_survival_checks_do_not_read_local_sidecars_or_index(tmp_path: Path, monkeypatch):
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\nx.npz,train,60,001,1\n")
    config_payload_data = survival_config_payload(
        index,
        {
            "disease_columns_index": "/wujidata/survival/disease_columns.txt",
            "event_time_index": "/wujidata/survival/event_time.csv",
            "is_event_index": "/wujidata/survival/is_event.csv",
            "has_label_index": "/wujidata/survival/has_label.csv",
        },
    )
    config_payload_data["data"]["finetune_data_index"] = "/wujidata/survival/index.csv"
    config = write_yaml(tmp_path / "survival_ssh.yaml", config_payload_data)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("agent_tools.decision_paths.subprocess.run", fake_run)

    for validation in ("ssh", "remote"):
        recipe = write_yaml(
            tmp_path / f"recipe_{validation}.yaml",
            {
                "name": f"remote_survival_{validation}",
                "task": "finetune",
                "variant": "sleep2vec",
                "inputs": {"config": str(config), "label_name": "incident_cox", "pretrained_backbone_path": None},
                "runtime": {"devices": [0]},
                "artifacts": {"results_csv_path": str(tmp_path / f"{validation}.csv"), "version_name": validation},
                "evaluation_policy": {
                    "selection_metric": "val_loss",
                    "selection_mode": "min",
                    "selection_split": "val",
                    "external_test_locked": True,
                    "test_after_fit": False,
                },
                "execution": {
                    "target": "ssh",
                    "host": "baichuan3",
                    "path_context": "remote",
                    "path_validation": validation,
                },
                "decisions": {
                    "task": {"value": "finetune", "source": "explicit_recipe"},
                    "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
                    "pretrained_backbone_path": {
                        "value": None,
                        "source": "explicit_recipe",
                        "meaning": "train from scratch",
                    },
                    "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                    "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                },
            },
        )

        _recipe, _cfg, report = evaluate_recipe(recipe)

        assert report.exit_code == 0

    assert calls
    call_scripts = [command[2] for command in calls]
    for path in (
        "/wujidata/survival/index.csv",
        "/wujidata/survival/disease_columns.txt",
        "/wujidata/survival/event_time.csv",
        "/wujidata/survival/is_event.csv",
        "/wujidata/survival/has_label.csv",
    ):
        assert any(path in script for script in call_scripts)

    calls.clear()

    def fail_event_time(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, int("event_time.csv" in command[2]), "", "")

    monkeypatch.setattr("agent_tools.decision_paths.subprocess.run", fail_event_time)
    missing_recipe = write_yaml(
        tmp_path / "recipe_missing_event_time.yaml",
        {
            "name": "remote_survival_missing_event_time",
            "task": "finetune",
            "variant": "sleep2vec",
            "inputs": {"config": str(config), "label_name": "incident_cox", "pretrained_backbone_path": None},
            "runtime": {"devices": [0]},
            "artifacts": {"results_csv_path": str(tmp_path / "missing.csv"), "version_name": "missing"},
            "evaluation_policy": {
                "selection_metric": "val_loss",
                "selection_mode": "min",
                "selection_split": "val",
                "external_test_locked": True,
                "test_after_fit": False,
            },
            "execution": {
                "target": "ssh",
                "host": "baichuan3",
                "path_context": "remote",
                "path_validation": "ssh",
            },
            "decisions": {
                "task": {"value": "finetune", "source": "explicit_recipe"},
                "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
                "pretrained_backbone_path": {
                    "value": None,
                    "source": "explicit_recipe",
                    "meaning": "train from scratch",
                },
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )

    _recipe, _cfg, report = evaluate_recipe(missing_recipe)

    assert report.exit_code == 1
    assert any(issue.field == "finetune.survival.event_time_index" for issue in report.issues)


def test_high_impact_decision_with_unresolved_source_blocks(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["decisions"]["label_name"] = {"value": "ahi", "source": "unresolved"}
    write_yaml(recipe, payload)

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"), cwd=Path.cwd())

    assert result.returncode == 2
    assert "label_name is not explicitly resolved" in result.stdout


def test_hparam_tune_missing_external_test_locked_needs_user_input(tmp_path: Path):
    recipe = {
        "name": "unit_tune",
        "task": "hparam_tune",
        "variant": "sleep2vec",
        "base_recipe": str(write_finetune_recipe(tmp_path)),
        "search": {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1e-6]}},
        "evaluation_policy": {
            "selection_metric": "val_ahi_pearson",
            "selection_mode": "max",
            "selection_split": "val",
            "test_after_fit": False,
            "final_eval_split": "validation",
            "final_test_unlocked": False,
            "require_manual_unlock_for_final_test": True,
        },
        "decisions": {"task": {"value": "hparam_tune", "source": "explicit_recipe"}},
    }
    recipe_path = write_yaml(tmp_path / "tune.yaml", recipe)

    result = _run("doctor", "--recipe", str(recipe_path), "--output-dir", str(tmp_path / "doctor"), cwd=Path.cwd())

    assert result.returncode == 2
    assert "external_test_locked" in result.stdout


def test_hparam_tune_selection_split_test_needs_user_input(tmp_path: Path):
    recipe = {
        "name": "unit_tune",
        "task": "hparam_tune",
        "variant": "sleep2vec",
        "base_recipe": str(write_finetune_recipe(tmp_path)),
        "search": {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1e-6]}},
        "evaluation_policy": {
            "selection_metric": "val_ahi_pearson",
            "selection_mode": "max",
            "selection_split": "test",
            "external_test_locked": True,
            "test_after_fit": False,
            "final_eval_split": "validation",
            "final_test_unlocked": False,
            "require_manual_unlock_for_final_test": True,
        },
        "decisions": {
            "task": {"value": "hparam_tune", "source": "explicit_recipe"},
            "label_name": {"value": "ahi", "source": "explicit_recipe"},
            "external_test_locked": {"value": True, "source": "explicit_recipe"},
            "train_val_test_policy": {"value": "bad", "source": "explicit_recipe"},
            "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
        },
    }

    result = _run(
        "doctor",
        "--recipe",
        str(write_yaml(tmp_path / "tune.yaml", recipe)),
        "--output-dir",
        str(tmp_path / "doctor"),
        cwd=Path.cwd(),
    )

    assert result.returncode == 2
    assert "selection_split=test" in result.stdout


def test_hparam_tune_blocks_on_base_config_blocking_issues(tmp_path: Path):
    base = write_finetune_recipe(tmp_path)
    base_payload = yaml.safe_load(base.read_text())
    config = Path(base_payload["inputs"]["config"])
    config_payload = yaml.safe_load(config.read_text())
    config_payload["data"]["finetune_data_index"] = None
    config_payload["data"]["finetune_preset_path"] = None
    write_yaml(config, config_payload)
    recipe = {
        "name": "unit_tune",
        "task": "hparam_tune",
        "variant": "sleep2vec",
        "base_recipe": str(base),
        "search": {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1e-6]}},
        "evaluation_policy": {
            "selection_metric": "val_ahi_pearson",
            "selection_mode": "max",
            "selection_split": "val",
            "external_test_locked": True,
            "test_after_fit": False,
            "final_eval_split": "validation",
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
    }

    result = _run(
        "doctor",
        "--recipe",
        str(write_yaml(tmp_path / "tune.yaml", recipe)),
        "--output-dir",
        str(tmp_path / "doctor"),
        cwd=Path.cwd(),
    )

    assert result.returncode == 2
    assert "data.backend=npz" in result.stdout


def test_hparam_tune_requires_local_selection_policy_even_when_base_has_it(tmp_path: Path):
    recipe = {
        "name": "unit_tune",
        "task": "hparam_tune",
        "variant": "sleep2vec",
        "base_recipe": str(write_finetune_recipe(tmp_path)),
        "search": {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1e-6]}},
        "evaluation_policy": {
            "external_test_locked": True,
            "test_after_fit": False,
            "final_eval_split": "validation",
            "final_test_unlocked": False,
            "require_manual_unlock_for_final_test": True,
        },
        "decisions": {
            "task": {"value": "hparam_tune", "source": "explicit_recipe"},
            "label_name": {"value": "ahi", "source": "explicit_recipe"},
            "external_test_locked": {"value": True, "source": "explicit_recipe"},
            "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
        },
    }

    result = _run(
        "doctor",
        "--recipe",
        str(write_yaml(tmp_path / "tune.yaml", recipe)),
        "--output-dir",
        str(tmp_path / "doctor"),
        cwd=Path.cwd(),
    )

    assert result.returncode == 2
    assert "selection_metric" in result.stdout


def test_hparam_tune_blocks_when_selection_metric_conflicts_with_config(tmp_path: Path):
    recipe = {
        "name": "unit_tune",
        "task": "hparam_tune",
        "variant": "sleep2vec",
        "base_recipe": str(write_finetune_recipe(tmp_path)),
        "search": {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1e-6]}},
        "evaluation_policy": {
            "selection_metric": "val_loss",
            "selection_mode": "max",
            "selection_split": "val",
            "external_test_locked": True,
            "test_after_fit": False,
            "final_eval_split": "validation",
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
    }

    result = _run(
        "doctor",
        "--recipe",
        str(write_yaml(tmp_path / "tune.yaml", recipe)),
        "--output-dir",
        str(tmp_path / "doctor"),
        cwd=Path.cwd(),
    )

    assert result.returncode == 1
    assert "differs from config" in result.stdout


def test_missing_high_impact_label_requires_user_input():
    policy = load_consultation_policy()
    report = evaluate_consultation_gates(
        "finetune",
        {"task": "finetune", "evaluation_policy": {"test_after_fit": False, "external_test_locked": True}},
        None,
        {},
        policy,
    )

    assert report.status == DecisionStatus.NEEDS_USER_INPUT
    assert any(issue.field == "label_name" for issue in report.issues)
