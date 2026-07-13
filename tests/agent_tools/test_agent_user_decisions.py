from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import yaml

from agent_tools.plans import evaluate_recipe


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def test_user_decision_yaml_resolves_missing_label_name(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"label_name": {"value": "ahi", "source": "explicit_user"}}},
    )

    result = _run(
        "doctor",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(tmp_path / "doctor"),
    )

    assert result.returncode == 0
    assert "Status: PASS" in result.stdout


def test_user_decision_yaml_resolves_external_test_locked(tmp_path: Path):
    base = write_finetune_recipe(tmp_path)
    recipe = write_yaml(
        tmp_path / "tune.yaml",
        {
            "name": "unit_tune",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base),
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
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"external_test_locked": {"value": True, "source": "explicit_user"}}},
    )

    result = _run(
        "doctor",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(tmp_path / "doctor"),
    )

    assert result.returncode == 0
    assert "Status: PASS" in result.stdout


def test_recipe_ask_user_always_blocks(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["decisions"]["label_name"] = {
        "value": "ASK_USER",
        "source": "unresolved",
        "question": "Which label?",
    }
    write_yaml(recipe, payload)

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 2
    assert "ASK_USER" in result.stdout


def test_user_decision_file_requires_decisions_mapping_before_output(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    decisions = tmp_path / "decisions.yaml"
    decisions.write_text("label_name: ahi\n")
    output_dir = tmp_path / "doctor"

    result = _run(
        "doctor",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 1
    assert "must contain a decisions mapping" in result.stderr
    assert not output_dir.exists()


def test_user_task_cannot_override_explicit_recipe_task(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"task": {"value": "evaluate", "source": "explicit_user"}}},
    )

    effective, _cfg, report = evaluate_recipe(recipe, decisions)

    assert effective["task"] == "finetune"
    assert report.exit_code == 1
    assert any(issue.field == "task" and "conflicts" in issue.message for issue in report.blocking_issues())


def test_user_task_fills_missing_recipe_task(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.pop("task")
    recipe.write_text(yaml.safe_dump(payload))
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"task": {"value": "finetune", "source": "explicit_user"}}},
    )

    effective, _cfg, report = evaluate_recipe(recipe, decisions)

    assert report.exit_code == 0
    assert effective["task"] == "finetune"


def test_user_split_decision_requires_concrete_split(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"train_val_test_policy": {"value": "select on val", "source": "explicit_user"}}},
    )

    effective, _cfg, report = evaluate_recipe(recipe, decisions)

    assert effective["evaluation_policy"]["selection_split"] == "val"
    assert report.exit_code == 1
    assert any(issue.field == "train_val_test_policy" for issue in report.blocking_issues())


def test_user_split_decision_materializes_selection_split(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"train_val_test_policy": {"value": "train", "source": "explicit_user"}}},
    )

    effective, _cfg, report = evaluate_recipe(recipe, decisions)

    assert report.exit_code == 0
    assert effective["evaluation_policy"]["selection_split"] == "train"
