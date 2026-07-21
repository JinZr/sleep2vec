from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import plans
from agent_tools.models import REPO_ROOT
from agent_tools.plans import evaluate_recipe
from agent_tools.recipes import load_recipe_with_base, load_yaml_file


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _snapshot(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _set_path(payload: dict, path: tuple[str, ...], value: object) -> None:
    owner = payload
    for part in path[:-1]:
        owner = owner.setdefault(part, {})
    owner[path[-1]] = value


def _write_hparam_recipe(tmp_path: Path) -> Path:
    base = write_finetune_recipe(tmp_path / "base")
    return write_yaml(
        tmp_path / "tune.yaml",
        {
            "name": "unit_recipe_closure_hparam",
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


def _write_infer_recipe(tmp_path: Path) -> Path:
    finetune = write_finetune_recipe(tmp_path)
    config = yaml.safe_load(finetune.read_text())["inputs"]["config"]
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("checkpoint")
    return write_yaml(
        tmp_path / "infer.yaml",
        {
            "name": "unit_recipe_closure_infer",
            "task": "infer",
            "variant": "sleep2vec",
            "inputs": {
                "config": config,
                "ckpt_path": str(checkpoint),
                "label_name": "ahi",
                "eval_split": "val",
            },
            "runtime": {"devices": [0]},
            "artifacts": {"overwrite": False},
            "evaluation_policy": {"external_test_locked": False},
            "decisions": {
                "task": {"value": "infer", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "ckpt_path": {"value": str(checkpoint), "source": "explicit_recipe"},
                "eval_split": {"value": "val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def _infer_recipe_payload(tmp_path: Path) -> tuple[Path, dict]:
    recipe = _write_infer_recipe(tmp_path)
    return recipe, yaml.safe_load(recipe.read_text())


def _evaluate_payload(recipe: Path, payload: dict):
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    return evaluate_recipe(recipe)[2]


def _set_explicit_input(payload: dict, field: str, value: object) -> None:
    payload["inputs"][field] = value
    payload["decisions"][field] = {"value": value, "source": "explicit_recipe"}


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        (("naem",), "typo", "naem"),
        (("experiment", "titel"), "typo", "experiment.titel"),
        (("step", "purpsoe"), "typo", "step.purpsoe"),
        (("inputs", "confg"), "typo", "inputs.confg"),
        (("runtime", "lrr"), 1e-6, "runtime.lrr"),
        (("artifacts", "versoin_name"), "typo", "artifacts.versoin_name"),
        (("evaluation_policy", "selection_metic"), "val_loss", "evaluation_policy.selection_metic"),
        (
            ("decisions", "overwrite_polciy"),
            {"value": False, "source": "explicit_recipe"},
            "decisions.overwrite_polciy",
        ),
        (("execution", "path_contex"), "local", "execution.path_contex"),
    ],
)
def test_recipe_rejects_unknown_fields_in_existing_owner_sections(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    field: str,
):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    _set_path(payload, path, value)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, _cfg, report = evaluate_recipe(recipe)

    issue = next(issue for issue in report.blocking_issues() if issue.field == field)
    assert report.exit_code == 1
    assert issue.evidence["source_layer"] == "effective"
    assert issue.evidence["preflight_before_workspace"] is True


def test_recipe_rejects_unregistered_task_at_the_recipe_boundary(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["task"] = "not_registered"
    payload["decisions"]["task"] = {"value": "not_registered", "source": "explicit_recipe"}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, _cfg, report = evaluate_recipe(recipe)

    issue = next(issue for issue in report.blocking_issues() if issue.field == "task")
    assert report.exit_code == 1
    assert issue.message == "Unsupported task: not_registered"
    assert issue.evidence["source_layer"] == "effective"
    assert issue.evidence["preflight_before_workspace"] is True


@pytest.mark.parametrize("section", ["inputs", "runtime", "decisions"])
def test_recipe_rejects_non_mapping_sections_before_consumers_run(tmp_path: Path, section: str):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload[section] = []
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, _cfg, report = evaluate_recipe(recipe)

    issue = next(issue for issue in report.blocking_issues() if issue.field == section)
    assert report.exit_code == 1
    assert issue.evidence["source_layer"] == "effective"
    assert issue.evidence["preflight_before_workspace"] is True


@pytest.mark.parametrize(
    ("task", "field", "value"),
    [
        ("finetune", "avg_ckpts", 2),
        ("infer", "epochs", 2),
        ("sleep2stat", "lr", 1e-6),
        ("sex_age_baseline", "wandb_mode", "offline"),
    ],
)
def test_runtime_fields_are_rejected_when_the_task_or_variant_does_not_consume_them(
    tmp_path: Path,
    task: str,
    field: str,
    value: object,
):
    case = tmp_path / task
    if task == "infer":
        recipe = _write_infer_recipe(case)
    elif task == "sleep2stat":
        payload = load_yaml_file("recipes/examples/tiny_fixture_sleep2stat.yaml")
        recipe = write_yaml(case / "sleep2stat.yaml", payload)
    else:
        variant = "sex_age_baseline" if task == "sex_age_baseline" else "sleep2vec"
        recipe = write_finetune_recipe(case, variant=variant)
    payload = yaml.safe_load(recipe.read_text())
    payload.setdefault("runtime", {})[field] = value
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, _cfg, report = evaluate_recipe(recipe)

    issue = next(issue for issue in report.blocking_issues() if issue.field == f"runtime.{field}")
    assert report.exit_code == 1
    assert issue.evidence["source_layer"] == "effective"


@pytest.mark.parametrize("task", ["finetune", "preset_prepare", "sleep2stat"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python", "/runtime/python"),
        ("runtime_commit", "a" * 40),
    ],
)
def test_execution_identity_fields_are_rejected_when_task_does_not_consume_them(
    tmp_path: Path, task: str, field: str, value: object
):
    case = tmp_path / task
    if task == "finetune":
        recipe = write_finetune_recipe(case)
    elif task == "preset_prepare":
        recipe = write_yaml(
            case / "preset.yaml",
            {"name": "unit_preset", "task": "preset_prepare", "variant": "sleep2vec"},
        )
    else:
        recipe = write_yaml(case / "sleep2stat.yaml", load_yaml_file("recipes/examples/tiny_fixture_sleep2stat.yaml"))
    payload = yaml.safe_load(recipe.read_text())
    payload.setdefault("execution", {})[field] = value
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, _cfg, report = evaluate_recipe(recipe)

    issue = next(issue for issue in report.blocking_issues() if issue.field == f"execution.{field}")
    assert report.exit_code == 1
    assert issue.evidence["source_layer"] == "effective"
    assert issue.evidence["preflight_before_workspace"] is True


@pytest.mark.parametrize("task", ["finetune", "preset_prepare", "sleep2stat"])
def test_non_identity_tasks_accept_absolute_execution_workdir(tmp_path: Path, task: str):
    case = tmp_path / task
    if task == "finetune":
        recipe = write_finetune_recipe(case)
    elif task == "preset_prepare":
        recipe = write_yaml(
            case / "preset.yaml",
            {"name": "unit_preset", "task": "preset_prepare", "variant": "sleep2vec"},
        )
    else:
        recipe = write_yaml(case / "sleep2stat.yaml", load_yaml_file("recipes/examples/tiny_fixture_sleep2stat.yaml"))
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = {"workdir": str(tmp_path / "runtime")}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, _cfg, report = evaluate_recipe(recipe)

    assert all(issue.field != "execution.workdir" for issue in report.blocking_issues())


@pytest.mark.parametrize("task", ["infer", "evaluate"])
def test_infer_evaluate_accepts_complete_execution_identity(tmp_path: Path, task: str):
    recipe = _write_infer_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["task"] = task
    payload["decisions"]["task"] = {"value": task, "source": "explicit_recipe"}
    if task == "evaluate":
        payload["evaluation_policy"]["final_test_unlocked"] = True
        payload["decisions"]["final_eval_unlock"] = {"value": True, "source": "explicit_recipe"}
    payload["execution"] = {
        "target": "local",
        "workdir": str(tmp_path / "runtime"),
        "python": "/runtime/python",
        "runtime_commit": "a" * 40,
    }
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, _cfg, report = evaluate_recipe(recipe)

    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]


@pytest.mark.parametrize("task", ["infer", "evaluate"])
def test_infer_evaluate_accepts_workdir_without_runtime_identity(tmp_path: Path, task: str):
    recipe = _write_infer_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["task"] = task
    payload["decisions"]["task"] = {"value": task, "source": "explicit_recipe"}
    if task == "evaluate":
        payload["evaluation_policy"]["final_test_unlocked"] = True
        payload["decisions"]["final_eval_unlock"] = {"value": True, "source": "explicit_recipe"}
    payload["execution"] = {"workdir": str(tmp_path / "runtime")}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, _cfg, report = evaluate_recipe(recipe)

    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]


def test_infer_relative_checkpoint_is_resolved_from_execution_workdir(tmp_path: Path):
    recipe, payload = _infer_recipe_payload(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "runtime-only.ckpt").write_text("checkpoint")
    _set_explicit_input(payload, "ckpt_path", "runtime-only.ckpt")
    payload["execution"] = {"workdir": str(runtime)}

    report = _evaluate_payload(recipe, payload)
    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]


@pytest.mark.parametrize("input_field", ["ckpt_path", "pretrained_backbone_path"])
def test_infer_relative_model_input_does_not_fall_back_to_repo_root(tmp_path: Path, input_field: str):
    recipe, payload = _infer_recipe_payload(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    _set_explicit_input(payload, input_field, "AGENTS.md")
    payload["execution"] = {"workdir": str(runtime)}

    report = _evaluate_payload(recipe, payload)
    assert report.exit_code == 1
    assert any(issue.field == input_field for issue in report.blocking_issues())


@pytest.mark.parametrize("input_field", ["ckpt_path", "pretrained_backbone_path"])
def test_infer_checkpoint_inputs_must_be_files(tmp_path: Path, input_field: str):
    recipe, payload = _infer_recipe_payload(tmp_path)
    directory = tmp_path / f"{input_field}-directory"
    directory.mkdir()
    _set_explicit_input(payload, input_field, str(directory))

    report = _evaluate_payload(recipe, payload)
    issue = next(issue for issue in report.blocking_issues() if issue.field == input_field)
    assert report.exit_code == 1
    assert "file does not exist" in issue.message


def test_infer_relative_average_checkpoint_dir_is_resolved_from_execution_workdir(tmp_path: Path):
    recipe, payload = _infer_recipe_payload(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    relative_dir = f"{tmp_path.name}-averages"
    (runtime / relative_dir).mkdir()
    assert not (REPO_ROOT / relative_dir).exists()
    _set_explicit_input(payload, "label_name", "stage5")
    payload["runtime"].update({"avg_ckpts": 2, "avg_ckpt_dir": relative_dir})
    payload["execution"] = {"workdir": str(runtime)}

    report = _evaluate_payload(recipe, payload)
    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]


def test_infer_average_checkpoint_dir_must_be_a_directory(tmp_path: Path):
    recipe, payload = _infer_recipe_payload(tmp_path)
    averages = tmp_path / "averages"
    averages.write_text("not a directory")
    _set_explicit_input(payload, "label_name", "stage5")
    payload["runtime"].update({"avg_ckpts": 2, "avg_ckpt_dir": str(averages)})

    report = _evaluate_payload(recipe, payload)
    issue = next(issue for issue in report.blocking_issues() if issue.field == "avg_ckpt_dir")
    assert report.exit_code == 1
    assert "directory does not exist" in issue.message


def test_infer_rejects_ahi_checkpoint_averaging(tmp_path: Path):
    recipe, payload = _infer_recipe_payload(tmp_path)
    averages = tmp_path / "averages"
    averages.mkdir()
    payload["runtime"].update({"avg_ckpts": 2, "avg_ckpt_dir": str(averages)})

    report = _evaluate_payload(recipe, payload)
    assert report.exit_code == 1
    assert any(issue.field == "runtime.avg_ckpts" for issue in report.blocking_issues())


@pytest.mark.parametrize("value", [True, "2", 0])
def test_infer_rejects_invalid_avg_ckpts(tmp_path: Path, value: object):
    recipe, payload = _infer_recipe_payload(tmp_path)
    payload["runtime"]["avg_ckpts"] = value

    report = _evaluate_payload(recipe, payload)
    issue = next(issue for issue in report.blocking_issues() if issue.field == "runtime.avg_ckpts")
    assert report.exit_code == 1
    assert issue.message == "runtime.avg_ckpts must be a positive integer."
    assert issue.evidence["preflight_before_workspace"] is True


def test_infer_checkpoint_alias_is_valid_with_average_checkpoint_dir(tmp_path: Path):
    recipe, payload = _infer_recipe_payload(tmp_path)
    runtime = tmp_path / "runtime"
    averages = runtime / "averages"
    averages.mkdir(parents=True)
    _set_explicit_input(payload, "ckpt_path", "best")
    _set_explicit_input(payload, "label_name", "stage5")
    payload["runtime"].update({"avg_ckpts": 2, "avg_ckpt_dir": "averages"})
    payload["execution"] = {"workdir": str(runtime)}

    report = _evaluate_payload(recipe, payload)
    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]


def test_infer_multilabel_preset_still_requires_runtime_sidecars(tmp_path: Path):
    recipe, payload = _infer_recipe_payload(tmp_path)
    config_path = Path(payload["inputs"]["config"])
    config = yaml.safe_load(config_path.read_text())
    config["finetune"]["task"].update({"type": "multilabel_classification", "output_dim": 2, "is_seq": False})
    config["finetune"]["multilabel"] = {
        "key_column": "eid",
        "disease_columns_index": "missing-diseases.txt",
        "label_index": "missing-labels.csv",
        "has_label_index": "missing-has-label.csv",
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    preset = tmp_path / "multilabel.pickle"
    preset.write_bytes(b"preset")
    payload["inputs"]["inference_preset_path"] = str(preset)

    report = _evaluate_payload(recipe, payload)
    assert report.exit_code == 2
    assert any(issue.field == "multilabel_sidecars" for issue in report.blocking_issues())


def test_infer_relative_checkpoint_defaults_to_repo_root_without_workdir(tmp_path: Path):
    recipe, payload = _infer_recipe_payload(tmp_path)
    _set_explicit_input(payload, "ckpt_path", "AGENTS.md")

    report = _evaluate_payload(recipe, payload)
    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]


def test_infer_source_config_remains_repo_relative_with_execution_workdir(tmp_path: Path):
    recipe, payload = _infer_recipe_payload(tmp_path)
    payload["inputs"]["config"] = os.path.relpath(payload["inputs"]["config"], REPO_ROOT)
    payload["execution"] = {"workdir": str(tmp_path / "runtime")}

    report = _evaluate_payload(recipe, payload)
    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]


def test_infer_configured_relative_index_is_summarized_from_execution_workdir(tmp_path: Path):
    recipe, payload = _infer_recipe_payload(tmp_path)
    config_path = Path(payload["inputs"]["config"])
    config = yaml.safe_load(config_path.read_text())
    source_index = Path(config["data"]["finetune_data_index"])
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "runtime-index.csv").write_bytes(source_index.read_bytes())
    config["data"]["finetune_data_index"] = "runtime-index.csv"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    payload["execution"] = {"workdir": str(runtime)}

    report = _evaluate_payload(recipe, payload)
    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]


@pytest.mark.parametrize("task", ["infer", "evaluate"])
@pytest.mark.parametrize("missing_field", ["python", "runtime_commit", "workdir"])
def test_infer_evaluate_execution_identity_is_all_or_none(tmp_path: Path, task: str, missing_field: str):
    recipe = _write_infer_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["task"] = task
    payload["decisions"]["task"] = {"value": task, "source": "explicit_recipe"}
    payload["execution"] = {
        "workdir": str(tmp_path / "runtime"),
        "python": "/runtime/python",
        "runtime_commit": "a" * 40,
    }
    payload["execution"].pop(missing_field)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, _cfg, report = evaluate_recipe(recipe)

    issue = next(issue for issue in report.blocking_issues() if issue.field == f"execution.{missing_field}")
    assert report.exit_code == 1
    assert issue.evidence["preflight_before_workspace"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workdir", "relative/runtime"),
        ("workdir", []),
        ("python", "ASK_USER"),
        ("python", []),
        ("python", "conda run -n exp python"),
        ("python", "~/miniconda/bin/python"),
        ("runtime_commit", "A" * 40),
        ("runtime_commit", []),
        ("target", "ssh"),
    ],
)
def test_infer_rejects_invalid_execution_identity(tmp_path: Path, field: str, value: object):
    recipe = _write_infer_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = {
        "target": "local",
        "workdir": str(tmp_path / "runtime"),
        "python": "/runtime/python",
        "runtime_commit": "a" * 40,
    }
    payload["execution"][field] = value
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, _cfg, report = evaluate_recipe(recipe)

    issue = next(issue for issue in report.blocking_issues() if issue.field == f"execution.{field}")
    assert report.exit_code == 1
    assert issue.evidence["preflight_before_workspace"] is True


@pytest.mark.parametrize("field", ["_recipe_path", "_base_recipe", "_local_recipe", "_private"])
def test_raw_recipe_cannot_inject_reserved_internal_fields(tmp_path: Path, field: str):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({"name": "reserved", "task": "finetune", field: {}}))

    with pytest.raises(ValueError, match=field):
        load_recipe_with_base(path)


def test_base_source_type_error_is_visible_after_local_section_replacement(tmp_path: Path):
    recipe = _write_hparam_recipe(tmp_path)
    local = yaml.safe_load(recipe.read_text())
    base = Path(local["base_recipe"])
    base_payload = yaml.safe_load(base.read_text())
    base_payload["runtime"] = []
    base.write_text(yaml.safe_dump(base_payload, sort_keys=False))
    local["runtime"] = {"devices": [1]}
    recipe.write_text(yaml.safe_dump(local, sort_keys=False))

    effective, _cfg, report = evaluate_recipe(recipe)

    assert effective["runtime"] == {"devices": [1]}
    issue = next(issue for issue in report.blocking_issues() if issue.field == "runtime")
    assert report.exit_code == 1
    assert issue.evidence["source_layer"] == "base"


def test_local_hparam_typo_reports_local_source_layer(tmp_path: Path):
    recipe = _write_hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_run"] = payload["search"]["max_runs"]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, _cfg, report = evaluate_recipe(recipe)

    issue = next(issue for issue in report.blocking_issues() if issue.field == "search.max_run")
    assert report.exit_code == 1
    assert issue.evidence["source_layer"] == "local"


def test_recipe_decision_task_closes_shape_when_top_level_task_is_missing(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.pop("task")
    payload["top_typo"] = True
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, cfg, report = evaluate_recipe(recipe)

    issue = next(issue for issue in report.blocking_issues() if issue.field == "top_typo")
    assert report.exit_code == 1
    assert cfg is None
    assert issue.evidence["source_layer"] == "effective"


def test_hparam_local_ask_user_task_uses_resolved_task_for_closure(tmp_path: Path):
    recipe = _write_hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["task"] = "ASK_USER"
    payload["top_typo"] = True
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, cfg, report = evaluate_recipe(recipe)

    issue = next(issue for issue in report.blocking_issues() if issue.field == "top_typo")
    assert report.exit_code == 1
    assert cfg is None
    assert issue.evidence["source_layer"] == "local"


def test_hparam_unresolved_ask_user_task_still_closes_local_shape(tmp_path: Path):
    recipe = _write_hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["task"] = "ASK_USER"
    payload["decisions"]["task"]["value"] = "ASK_USER"
    payload["top_typo"] = True
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, cfg, report = evaluate_recipe(recipe)

    issues = {issue.field: issue for issue in report.blocking_issues()}
    assert report.exit_code == 1
    assert cfg is None
    assert issues["task"].evidence["source_layer"] == "local"
    assert issues["top_typo"].evidence["source_layer"] == "local"


def test_hparam_missing_local_task_does_not_inherit_base_task(tmp_path: Path):
    recipe = _write_hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.pop("task")
    payload["decisions"].pop("task")
    payload["top_typo"] = True
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, cfg, report = evaluate_recipe(recipe)

    issues = {issue.field: issue for issue in report.blocking_issues()}
    assert report.exit_code == 1
    assert cfg is None
    assert issues["task"].evidence["source_layer"] == "local"
    assert issues["top_typo"].evidence["source_layer"] == "local"


def test_hparam_user_task_fills_missing_local_task_without_base_conflict(tmp_path: Path):
    recipe = _write_hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.pop("task")
    payload["decisions"].pop("task")
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"task": {"value": "hparam_tune", "source": "explicit_user"}}},
    )

    effective, _cfg, report = evaluate_recipe(recipe, decisions)

    assert report.exit_code == 0
    assert effective["task"] == "hparam_tune"
    assert effective["decisions"]["task"]["value"] == "hparam_tune"


def test_hparam_base_is_closed_as_finetune_even_when_it_declares_another_task(tmp_path: Path):
    recipe = _write_hparam_recipe(tmp_path)
    local = yaml.safe_load(recipe.read_text())
    base = Path(local["base_recipe"])
    payload = yaml.safe_load(base.read_text())
    payload["task"] = "infer"
    payload["runtime"]["accelerator"] = "cpu"
    base.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, cfg, report = evaluate_recipe(recipe)

    issues = {issue.field: issue for issue in report.blocking_issues()}
    assert report.exit_code == 1
    assert cfg is None
    assert issues["task"].evidence["source_layer"] == "base"
    assert issues["runtime.accelerator"].evidence["source_layer"] == "base"


@pytest.mark.parametrize("task", ["finetune", "hparam_tune"])
@pytest.mark.parametrize(
    ("required_channels", "expected_exit_code"),
    [
        (["ppg", "ahi", "stage5"], 0),
        (["ecg"], 1),
    ],
)
def test_non_preset_required_channels_decision_is_checked_without_creating_preset_section(
    tmp_path: Path,
    task: str,
    required_channels: list[str],
    expected_exit_code: int,
):
    source = tmp_path / task
    recipe = write_finetune_recipe(source) if task == "finetune" else _write_hparam_recipe(source)
    decisions = tmp_path / f"{task}-decisions.yaml"
    decisions.write_text(
        yaml.safe_dump(
            {
                "decisions": {
                    "required_channels": {"value": required_channels, "source": "explicit_user"},
                }
            },
            sort_keys=False,
        )
    )

    effective, _cfg, report = evaluate_recipe(recipe, decisions)

    assert report.exit_code == expected_exit_code
    assert "preset" not in effective
    assert effective["decisions"]["required_channels"]["value"] == required_channels
    if expected_exit_code:
        assert any(issue.field == "required_channels" for issue in report.blocking_issues())


def test_preset_regenerate_is_not_an_authored_preset_field(tmp_path: Path):
    recipe = write_yaml(
        tmp_path / "preset.yaml",
        {
            "name": "unit_preset_regenerate",
            "task": "preset_prepare",
            "variant": "sleep2vec",
            "preset": {"regenerate": True},
        },
    )

    _recipe, cfg, report = evaluate_recipe(recipe)

    issue = next(issue for issue in report.blocking_issues() if issue.field == "preset.regenerate")
    assert report.exit_code == 1
    assert cfg is None
    assert issue.evidence["source_layer"] == "effective"


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"decisions": {}, "metadata": {}}, "metadata"),
        ({"decisions": {"unknown_decision": {"value": True, "source": "explicit_user"}}}, "unknown_decision"),
        ({"decisions": {"label_name": {"value": "ahi", "soucre": "explicit_user"}}}, "soucre"),
        ({"decisions": {"hparam_budget": {"value": 1, "source": "explicit_user"}}}, "hparam_budget"),
    ],
)
def test_user_decision_file_has_a_closed_task_aware_contract(tmp_path: Path, payload: dict, field: str):
    recipe = write_finetune_recipe(tmp_path)
    decisions = tmp_path / "decisions.yaml"
    decisions.write_text(yaml.safe_dump(payload, sort_keys=False))

    result = _run("doctor", "--recipe", str(recipe), "--user-decisions", str(decisions))

    assert result.returncode == 1
    assert field in result.stdout + result.stderr


def test_context_rejects_invalid_user_decisions_before_writing_bundle(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path / "source")
    config = yaml.safe_load(recipe.read_text())["inputs"]["config"]
    decisions = tmp_path / "decisions.yaml"
    decisions.write_text(
        yaml.safe_dump(
            {"decisions": {"unknown_decision": {"value": True, "source": "explicit_user"}}},
            sort_keys=False,
        )
    )
    output_dir = tmp_path / "context"

    result = _run(
        "context",
        "--task",
        "finetune",
        "--variant",
        "sleep2vec",
        "--config",
        str(config),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 1
    assert "unknown_decision" in result.stdout + result.stderr
    assert not output_dir.exists()


def test_plan_closure_failure_does_not_create_or_mutate_workspace(tmp_path: Path):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source)
    payload = yaml.safe_load(recipe.read_text())
    workspace = tmp_path / "workspace"
    payload["experiment"]["root"] = str(workspace)
    payload["inputs"]["confg"] = payload["inputs"]["config"]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = _snapshot(source)

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(workspace / "plans" / "closure"))

    assert result.returncode == 1
    assert "inputs.confg" in result.stdout + result.stderr
    assert _snapshot(source) == before
    assert not workspace.exists()


def test_hparam_base_closure_failure_does_not_mutate_existing_workspace(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _write_hparam_recipe(source)
    payload = yaml.safe_load(recipe.read_text())
    base = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base.read_text())
    base_payload["runtime"]["lrr"] = 1e-6
    base.write_text(yaml.safe_dump(base_payload, sort_keys=False))
    before = _snapshot(source)

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(source / "plans" / "closure"))

    assert result.returncode == 1
    assert "runtime.lrr" in result.stdout + result.stderr
    assert _snapshot(source) == before
    assert not (source / "plans").exists()


def test_adaptive_init_closure_failure_leaves_target_and_source_untouched(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _write_hparam_recipe(source)
    payload = yaml.safe_load(recipe.read_text())
    workflow = tmp_path / "workflow"
    payload["experiment"]["root"] = str(workflow)
    payload["search"]["max_run"] = payload["search"]["max_runs"]
    payload["adaptive"] = {
        "enabled": True,
        "objective_metric": "test_auroc",
        "objective_mode": "max",
        "test_feedback_for_selection": True,
        "max_rounds": 2,
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
    }
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = _snapshot(source)

    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow))

    assert result.returncode == 1
    assert "search.max_run" in result.stdout + result.stderr
    assert _snapshot(source) == before
    assert not workflow.exists()


def test_adaptive_init_consults_before_runtime_validation(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _write_hparam_recipe(source)
    payload = yaml.safe_load(recipe.read_text())
    workflow = tmp_path / "workflow"
    payload["experiment"]["root"] = str(workflow)
    payload["adaptive"] = {
        "enabled": True,
        "objective_mode": "max",
        "max_rounds": 2,
        "max_runs_total": 4,
        "round_size": 1,
        "suggest": {"strategy": "agent_proposal"},
    }
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = _snapshot(source)

    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow))

    assert result.returncode == 2
    assert "Status: NEEDS_USER_INPUT" in result.stdout
    assert "adaptive.objective_metric" in result.stdout
    assert "test_feedback_for_selection" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert _snapshot(source) == before
    assert not workflow.exists()


@pytest.mark.parametrize(
    ("suggest", "replacement", "field"),
    [
        ({"strategy": "unknown"}, None, "adaptive.suggest.strategy"),
        ({"strategy": "agent_proposal"}, {"enabled": True}, "adaptive.replacement"),
        (
            {"strategy": "agent_proposal", "bounds": {"runtime.weight_decay": [0.0, 1.0]}},
            None,
            "adaptive.suggest.bounds",
        ),
    ],
)
def test_agent_proposal_contract_failure_precedes_workspace_writes(
    tmp_path: Path,
    suggest: dict,
    replacement: dict | None,
    field: str,
):
    source = tmp_path / "source"
    recipe = _write_hparam_recipe(source)
    payload = yaml.safe_load(recipe.read_text())
    workflow = tmp_path / "workflow"
    payload["experiment"]["root"] = str(workflow)
    payload["adaptive"] = {
        "enabled": True,
        "objective_metric": "val_ahi_pearson",
        "objective_mode": "max",
        "max_rounds": 2,
        "max_runs_total": 4,
        "round_size": 1,
        "suggest": suggest,
    }
    if replacement is not None:
        payload["adaptive"]["replacement"] = replacement
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = _snapshot(source)

    result = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow))

    assert result.returncode == 1
    assert field in result.stdout + result.stderr
    assert _snapshot(source) == before
    assert not workflow.exists()


@pytest.mark.parametrize(
    "field",
    ["objective_metric", "objective_mode", "round_size", "max_rounds", "max_runs_total"],
)
@pytest.mark.parametrize("unresolved", ["missing", "null", "empty"])
def test_agent_proposal_explicit_control_fields_block_before_workspace_writes(
    tmp_path: Path,
    field: str,
    unresolved: str,
):
    source = tmp_path / "source"
    recipe = _write_hparam_recipe(source)
    payload = yaml.safe_load(recipe.read_text())
    workflow = tmp_path / "workflow"
    payload["experiment"]["root"] = str(workflow)
    payload["adaptive"] = {
        "enabled": True,
        "objective_metric": "val_ahi_pearson",
        "objective_mode": "max",
        "max_rounds": 2,
        "max_runs_total": 4,
        "round_size": 1,
        "suggest": {"strategy": "agent_proposal"},
    }
    if unresolved == "missing":
        payload["adaptive"].pop(field)
    else:
        payload["adaptive"][field] = None if unresolved == "null" else ""
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = _snapshot(source)

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--output-dir",
        str(workflow / "plans" / "agent-proposal-explicit-fields"),
    )

    assert result.returncode == 2
    assert "Status: NEEDS_USER_INPUT" in result.stdout
    assert f"adaptive.{field}" in result.stdout + result.stderr
    assert _snapshot(source) == before
    assert not workflow.exists()


@pytest.mark.parametrize(
    ("objective_metric", "expected_exit_code", "expected_status"),
    [
        pytest.param("   ", 2, "NEEDS_USER_INPUT", id="blank"),
        pytest.param(0, 1, "FAIL", id="zero"),
        pytest.param(False, 1, "FAIL", id="false"),
        pytest.param([], 1, "FAIL", id="list"),
        pytest.param({}, 1, "FAIL", id="mapping"),
        pytest.param(1, 1, "FAIL", id="integer"),
    ],
)
def test_agent_proposal_invalid_objective_blocks_before_workspace_writes(
    tmp_path: Path,
    objective_metric,
    expected_exit_code: int,
    expected_status: str,
):
    source = tmp_path / "source"
    recipe = _write_hparam_recipe(source)
    payload = yaml.safe_load(recipe.read_text())
    workflow = tmp_path / "workflow"
    payload["experiment"]["root"] = str(workflow)
    payload["adaptive"] = {
        "enabled": True,
        "objective_metric": objective_metric,
        "objective_mode": "max",
        "max_rounds": 2,
        "max_runs_total": 4,
        "round_size": 1,
        "test_feedback_for_selection": True,
        "suggest": {"strategy": "agent_proposal"},
    }
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = _snapshot(source)

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--output-dir",
        str(workflow / "plans" / "agent-proposal-objective-type"),
    )

    assert result.returncode == expected_exit_code
    assert f"Status: {expected_status}" in result.stdout
    assert "adaptive.objective_metric" in result.stdout + result.stderr
    assert _snapshot(source) == before
    assert not workflow.exists()


def test_infer_runtime_fields_have_observable_command_effects():
    command = plans._commands_for_recipe(
        {
            "name": "infer-runtime-contract",
            "task": "infer",
            "variant": "sleep2vec",
            "inputs": {
                "config": "config.yaml",
                "ckpt_path": "model.ckpt",
                "label_name": "ahi",
                "eval_split": "val",
            },
            "runtime": {
                "devices": [2, 3],
                "precision": 32,
                "batch_size": 4,
                "num_workers": 5,
                "lr": 2e-6,
                "weight_decay": 3e-5,
                "accelerator": "cpu",
                "device": "cpu",
                "avg_ckpts": 2,
                "avg_ckpt_dir": "averaged",
                "seed": 7,
                "wandb_mode": "offline",
            },
        }
    )[0]

    for expected in (
        "--devices 2 3",
        "--precision 32",
        "--batch-size 4",
        "--num-workers 5",
        "--lr 2e-06",
        "--weight-decay 3e-05",
        "--accelerator cpu",
        "--device cpu",
        "--avg-ckpts 2",
        "--avg-ckpt-dir averaged",
        "--seed 7",
        "--wandb-mode offline",
    ):
        assert expected in command


def test_sleep2stat_postprocess_fields_have_observable_command_effects():
    commands = plans._commands_for_recipe(
        {
            "name": "sleep2stat-runtime-contract",
            "task": "sleep2stat",
            "inputs": {"config": "sleep2stat.yaml", "split": ["test"]},
            "runtime": {
                "dry_run": False,
                "summarize_after_run": False,
                "plot_cohort_after_run": True,
                "plot_group_column": "site",
                "plot_stage_source": "yasa",
                "plot_adjust_covariates": ["age", "sex"],
            },
        },
        {"is_sleep2stat": True, "sleep2stat": {"run": {"output_dir": "runs/unit"}}},
    )

    assert not any(command.startswith("python -m sleep2stat summarize ") for command in commands)
    plot = next(command for command in commands if command.startswith("python -m sleep2stat plot-cohort "))
    assert "--group-column site" in plot
    assert "--stage-source yasa" in plot
    assert "--adjust-covariates age sex" in plot
