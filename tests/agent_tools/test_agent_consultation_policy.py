from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import config_payload, survival_config_payload, write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools.configs import config_summary
from agent_tools.decision_hparam import hparam_recipe_contract_issues
from agent_tools.decision_paths import path_issues, validate_input_path
from agent_tools.decisions import DecisionStatus, evaluate_consultation_gates
from agent_tools.plans import evaluate_recipe
from agent_tools.recipes import load_consultation_policy


def test_selection_split_and_test_access_questions_are_distinct():
    policy = load_consultation_policy()
    fields = {field["id"]: field for field in policy["high_impact_fields"]}

    assert fields["train_val_test_policy"]["question"] == (
        "Which split should be used for model selection? Only val is supported; the test split is never a "
        "selection split."
    )
    assert fields["train_val_test_policy"]["allowed_values"] == ["val"]
    assert fields["external_test_locked"]["question"] == (
        "Should test data remain locked during this task, or is test access explicitly authorized?"
    )


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], cwd=cwd, text=True, capture_output=True)


@pytest.mark.parametrize(("task", "warns"), [("finetune", True), ("preset_prepare", False)])
def test_missing_artifacts_warning_respects_task_output_ownership(task, warns):
    report = evaluate_consultation_gates(
        task, {"task": task, "variant": "sleep2vec"}, None, {}, load_consultation_policy()
    )
    issues = [issue for issue in report.issues if issue.field == "output_dir"]
    assert bool(issues) is warns
    if warns:
        assert issues[0].status == DecisionStatus.WARN
        assert "artifacts/" in issues[0].message


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
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    result = _run("doctor", "--recipe", str(recipe), cwd=Path.cwd())

    assert result.returncode == 2
    assert "Status: NEEDS_USER_INPUT" in result.stdout
    assert not implicit_output.exists()
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


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
                "train_val_test_policy": {"value": "val", "source": "explicit_recipe"},
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
    assert "Required input file does not exist" in result.stdout


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


def test_runtime_paths_reject_tilde_but_source_locators_expand_it(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "source.yaml").write_text("source")
    monkeypatch.setenv("HOME", str(home))
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    raw_path = "~/source.yaml"
    literal_runtime_path = runtime / raw_path
    literal_runtime_path.parent.mkdir(parents=True)
    literal_runtime_path.write_text("literal runtime path")
    recipe = {"execution": {"workdir": str(runtime)}}

    runtime_issue = validate_input_path(recipe, "ckpt_path", raw_path, configured=False)
    source_issue = validate_input_path(
        recipe,
        "config",
        raw_path,
        configured=False,
        relative_to_workdir=False,
    )

    assert runtime_issue is not None
    assert runtime_issue.field == "ckpt_path"
    assert "must not use ~" in runtime_issue.message
    assert source_issue is None


@pytest.mark.parametrize(
    ("task_type", "section", "fields", "path_kwargs"),
    [
        (
            "survival",
            "survival",
            ("disease_columns_index", "event_time_index", "is_event_index", "has_label_index"),
            {"requires_survival_sidecars": True},
        ),
        (
            "multilabel_classification",
            "multilabel",
            ("disease_columns_index", "label_index", "has_label_index"),
            {"requires_multilabel_sidecars": True},
        ),
    ],
)
def test_local_runtime_sidecars_reject_tilde(task_type, section, fields, path_kwargs):
    config = {
        "finetune": {
            "task": {"type": task_type},
            section: {field: f"~/sidecars/{field}.csv" for field in fields},
        }
    }

    issues = path_issues("infer", {}, config, **path_kwargs)

    assert {issue.field for issue in issues} == {f"finetune.{section}.{field}" for field in fields}
    assert all("must not use ~" in issue.message for issue in issues)


def test_removed_ssh_execution_fields_are_rejected_as_unknown(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = {"target": "ssh"}
    write_yaml(recipe, payload)

    _recipe, _cfg, report = evaluate_recipe(recipe)

    assert report.exit_code == 1
    assert any(
        issue.field == "execution.target" and issue.message == "Unknown execution field for this task: target."
        for issue in report.issues
    )
    hparam_issues = hparam_recipe_contract_issues({"execution": {"target": "ssh"}}, source_layer="recipe")
    assert [(issue.field, issue.message) for issue in hparam_issues] == [
        ("execution.target", "Unknown hparam execution field: target.")
    ]


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
            "train_val_test_policy": {"value": "val", "source": "explicit_recipe"},
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
            "train_val_test_policy": {"value": "val", "source": "explicit_recipe"},
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


def test_direct_finetune_uses_task_neutral_test_access_question():
    policy = load_consultation_policy()
    report = evaluate_consultation_gates(
        "finetune",
        {"task": "finetune", "evaluation_policy": {"test_after_fit": False}},
        None,
        {},
        policy,
    )

    issue = next(issue for issue in report.issues if issue.field == "external_test_locked")
    assert issue.question == "Should test data remain locked during this task, or is test access explicitly authorized?"


def test_consultation_policy_rejects_tasks_without_registered_adapters():
    policy = yaml.safe_load(yaml.safe_dump(load_consultation_policy()))
    policy["high_impact_fields"][0]["required_for_tasks"].append("ghost")

    report = evaluate_consultation_gates("finetune", {"task": "finetune"}, None, {}, policy)

    assert report.exit_code == 1
    issue = next(issue for issue in report.issues if issue.field == "consultation_policy")
    assert issue.evidence["unknown_tasks"] == ["ghost"]
    assert issue.evidence["missing_tasks"] == []


def test_consultation_policy_rejects_registered_tasks_without_policy_coverage():
    policy = yaml.safe_load(yaml.safe_dump(load_consultation_policy()))
    for field in policy["high_impact_fields"]:
        field["required_for_tasks"] = [task for task in field["required_for_tasks"] if task != "sleep2stat"]

    report = evaluate_consultation_gates("finetune", {"task": "finetune"}, None, {}, policy)

    assert report.exit_code == 1
    issue = next(issue for issue in report.issues if issue.field == "consultation_policy")
    assert issue.evidence["unknown_tasks"] == []
    assert issue.evidence["missing_tasks"] == ["sleep2stat"]
