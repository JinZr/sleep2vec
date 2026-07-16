from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import plans
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
