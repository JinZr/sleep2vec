from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import yaml

from agent_tools.decisions import DecisionStatus, evaluate_consultation_gates
from agent_tools.recipes import load_policy_files


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

    result = _run("doctor", "--recipe", str(recipe), cwd=Path.cwd())

    assert result.returncode == 2
    assert not implicit_output.exists()


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
        "schema_version": 1,
        "name": "unit_tune",
        "task": "hparam_tune",
        "variant": "sleep2vec",
        "base_recipe": str(write_finetune_recipe(tmp_path)),
        "search": {"method": "grid", "max_trials": 1, "parameters": {"runtime.lr": [1e-6]}},
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
        "schema_version": 1,
        "name": "unit_tune",
        "task": "hparam_tune",
        "variant": "sleep2vec",
        "base_recipe": str(write_finetune_recipe(tmp_path)),
        "search": {"method": "grid", "max_trials": 1, "parameters": {"runtime.lr": [1e-6]}},
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
        "schema_version": 1,
        "name": "unit_tune",
        "task": "hparam_tune",
        "variant": "sleep2vec",
        "base_recipe": str(base),
        "search": {"method": "grid", "max_trials": 1, "parameters": {"runtime.lr": [1e-6]}},
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
        "schema_version": 1,
        "name": "unit_tune",
        "task": "hparam_tune",
        "variant": "sleep2vec",
        "base_recipe": str(write_finetune_recipe(tmp_path)),
        "search": {"method": "grid", "max_trials": 1, "parameters": {"runtime.lr": [1e-6]}},
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
        "schema_version": 1,
        "name": "unit_tune",
        "task": "hparam_tune",
        "variant": "sleep2vec",
        "base_recipe": str(write_finetune_recipe(tmp_path)),
        "search": {"method": "grid", "max_trials": 1, "parameters": {"runtime.lr": [1e-6]}},
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

    assert result.returncode == 2
    assert "differs from config" in result.stdout


def test_approved_defaults_do_not_resolve_high_impact_label():
    policy, defaults = load_policy_files()
    report = evaluate_consultation_gates(
        "finetune",
        {"task": "finetune", "evaluation_policy": {"test_after_fit": False, "external_test_locked": True}},
        None,
        {},
        policy,
        defaults,
    )

    assert report.status == DecisionStatus.NEEDS_USER_INPUT
    assert any(issue.field == "label_name" for issue in report.issues)


def test_approved_defaults_can_resolve_low_impact_wandb_mode():
    _policy, defaults = load_policy_files()
    assert defaults["approved_defaults"]["wandb_mode"]["value"] == "offline"
