from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import survival_config_payload, write_finetune_recipe, write_survival_sidecars, write_yaml
import pytest
import yaml

from agent_tools import experiments, plans
from agent_tools.experiment_workspace import file_sha256, merge_run_manifest, read_run_manifest
from agent_tools.models import REPO_ROOT
from agent_tools.plans import collect_runs


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _first_run(plan_dir: Path) -> dict:
    return json.loads((plan_dir / "plan.json").read_text())["runs"][0]


def _hparam_recipe(
    tmp_path: Path,
    *,
    variant: str = "sleep2vec",
    parameters: dict | None = None,
    max_runs: int | str = 1,
    ckpt_path: Path | None = None,
    final_config_path: Path | None = None,
    selection_metric: str = "val_ahi_pearson",
    selection_mode: str = "max",
) -> Path:
    base = write_finetune_recipe(tmp_path, variant=variant)
    inputs = {"ckpt_path": str(ckpt_path)} if ckpt_path else {}
    if final_config_path is not None:
        inputs["final_eval_config_path"] = str(final_config_path)
    return write_yaml(
        tmp_path / f"tune_{variant}.yaml",
        {
            "name": f"unit_tune_{variant}",
            "task": "hparam_tune",
            "variant": variant,
            "base_recipe": str(base),
            "inputs": inputs,
            "search": {"method": "grid", "max_runs": max_runs, "parameters": parameters or {"runtime.lr": [1e-6]}},
            "evaluation_policy": {
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
                "selection_split": "val",
                "final_eval_split": "test",
                "external_test_locked": True,
                "test_after_fit": False,
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


def _survival_recipe_with_missing_sidecar_key(tmp_path: Path) -> tuple[Path, Path]:
    index = tmp_path / "survival_index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\na.npz,train,60,001,1\nb.npz,val,60,003,1\n")
    config = write_yaml(
        tmp_path / "survival_config.yaml",
        survival_config_payload(index, write_survival_sidecars(tmp_path)),
    )
    recipe = {
        "name": "unit_survival_missing_sidecar_key",
        "task": "finetune",
        "variant": "sleep2vec",
        "inputs": {"config": str(config), "label_name": "incident_cox", "pretrained_backbone_path": None},
        "runtime": {"devices": [0]},
        "artifacts": {"results_csv_path": str(tmp_path / "results.csv"), "version_name": "unit"},
        "evaluation_policy": {
            "selection_metric": "val_loss",
            "selection_mode": "min",
            "selection_split": "val",
            "final_eval_split": "test",
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
    }
    return write_yaml(tmp_path / "survival_recipe.yaml", recipe), config


def _bad_survival_sidecars() -> dict[str, str]:
    return {
        "disease_columns_index": "/path/to/disease_columns.txt",
        "event_time_index": "/path/to/event_time.csv",
        "is_event_index": "/path/to/is_event.csv",
        "has_label_index": "/path/to/has_label.csv",
    }


def _write_survival_config_with_bad_sidecars(tmp_path: Path, *, preset_path: Path | None = None) -> Path:
    index = tmp_path / "survival_runtime_index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\na.npz,test,60,001,1\n")
    payload = survival_config_payload(index, _bad_survival_sidecars())
    if preset_path is not None:
        payload["data"]["finetune_preset_path"] = str(preset_path)
    return write_yaml(tmp_path / "bad_survival_config.yaml", payload)


def _write_infer_recipe(
    tmp_path: Path,
    config: Path,
    *,
    inference_preset_path: Path | None = None,
    eval_split: str = "test",
) -> Path:
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("checkpoint")
    inputs = {
        "config": str(config),
        "label_name": "incident_cox",
        "ckpt_path": str(ckpt),
        "eval_split": eval_split,
    }
    if inference_preset_path is not None:
        inputs["inference_preset_path"] = str(inference_preset_path)
    return write_yaml(
        tmp_path / "infer_survival.yaml",
        {
            "name": "unit_infer_survival",
            "task": "infer",
            "variant": "sleep2vec",
            "inputs": inputs,
            "evaluation_policy": {"external_test_locked": False, "final_test_unlocked": True},
            "artifacts": {"overwrite": True},
            "decisions": {
                "task": {"value": "infer", "source": "explicit_recipe"},
                "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
                "ckpt_path": {"value": str(ckpt), "source": "explicit_recipe"},
                "external_test_locked": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": True, "source": "explicit_recipe"},
                "overwrite_policy": {"value": True, "source": "explicit_recipe"},
            },
        },
    )


def _write_preset_recipe(
    tmp_path: Path,
    *,
    config: str | Path,
    index: str | Path,
    variant: str = "sleep2vec",
    preset: dict | None = None,
    execution: dict | None = None,
    name: str = "unit_preset",
) -> Path:
    preset = preset or {"n_tokens": 128, "split": ["train"], "allow_missing_channels": False}
    payload = {
        "name": name,
        "task": "preset_prepare",
        "variant": variant,
        "inputs": {"config": str(config), "index": [str(index)], "dataset_name": "unit"},
        "preset": preset,
        "decisions": {
            "task": {"value": "preset_prepare", "source": "explicit_recipe"},
            "preset_regeneration": {"value": True, "source": "explicit_recipe"},
            "overwrite_policy": {"value": bool(preset.get("overwrite", False)), "source": "explicit_recipe"},
            "required_channels": {"value": preset.get("channels", ["ppg"]), "source": "explicit_recipe"},
            "min_channels": {"value": preset.get("min_channels", 1), "source": "explicit_recipe"},
        },
    }
    if execution is not None:
        payload["execution"] = execution
    return write_yaml(tmp_path / f"{name}.yaml", payload)


def test_plan_does_not_create_run_all_when_consultation_required(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "Questions for user" in result.stdout
    assert "label_name" in result.stdout
    assert (output_dir / "plan.blocked.md").exists()
    assert not (output_dir / "run_all.sh").exists()


def test_doctor_blocks_survival_index_keys_missing_from_sidecars(tmp_path: Path):
    recipe, _config = _survival_recipe_with_missing_sidecar_key(tmp_path)

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 1
    assert "Status: FAIL" in result.stdout
    assert "survival key values missing from sidecars" in result.stdout


def test_plan_blocks_survival_index_keys_missing_from_sidecars(tmp_path: Path):
    recipe, _config = _survival_recipe_with_missing_sidecar_key(tmp_path)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "survival key values missing from sidecars" in result.stdout
    assert (output_dir / "plan.blocked.md").exists()
    assert not (output_dir / "run.sh").exists()


def test_plan_with_unresolved_experiment_metadata_does_not_write_output(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["step"]["purpose"] = "ASK_USER"
    recipe.write_text(yaml.safe_dump(payload))
    output_dir = tmp_path / "unresolved-plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert not output_dir.exists()


def test_blocked_plan_initializes_workspace_and_retry_uses_new_plan_dir(tmp_path: Path):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source, include_label=False)
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload))
    blocked_dir = workspace / "plans" / "blocked"

    blocked = _run("plan", "--recipe", str(recipe), "--output-dir", str(blocked_dir))

    assert blocked.returncode == 2
    assert (workspace / "experiment.yaml").exists()
    assert (blocked_dir / "plan.blocked.md").exists()
    decisions = tmp_path / "decisions.yaml"
    decisions.write_text(yaml.safe_dump({"decisions": {"label_name": {"value": "ahi", "source": "explicit_user"}}}))
    retry_dir = workspace / "plans" / "retry"

    retry = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(retry_dir),
    )

    assert retry.returncode == 0, retry.stderr
    assert (retry_dir / "run.sh").exists()


def test_hparam_recipe_cannot_inherit_experiment_and_step_from_base(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.pop("experiment")
    payload.pop("step")
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    doctor = _run("doctor", "--recipe", str(recipe))
    plan = _run("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert doctor.returncode == 2
    assert plan.returncode == 2
    assert "experiment" in doctor.stdout
    assert "experiment" in plan.stdout
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_effective_user_config_fails_before_workspace_mutation(tmp_path: Path):
    for case, config_text in (("missing", None), ("invalid", "model: not-a-mapping\n")):
        root = tmp_path / case
        recipe = write_finetune_recipe(root)
        selected_config = root / "selected.yaml"
        if config_text is not None:
            selected_config.write_text(config_text)
        decisions = root / "decisions.yaml"
        decisions.write_text(
            yaml.safe_dump({"decisions": {"config": {"value": str(selected_config), "source": "explicit_user"}}})
        )
        before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

        result = _run(
            "plan",
            "--recipe",
            str(recipe),
            "--user-decisions",
            str(decisions),
            "--output-dir",
            str(root / "plan"),
        )

        assert result.returncode == 1
        assert "config" in result.stdout.lower()
        assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_unresolved_effective_user_config_fails_before_workspace_mutation(tmp_path: Path):
    for case, value in (("null", None), ("empty", ""), ("ask", "ASK_USER")):
        root = tmp_path / case
        recipe = write_finetune_recipe(root)
        decisions = root / "decisions.yaml"
        decisions.write_text(yaml.safe_dump({"decisions": {"config": {"value": value, "source": "explicit_user"}}}))
        before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

        result = _run(
            "plan",
            "--recipe",
            str(recipe),
            "--user-decisions",
            str(decisions),
            "--output-dir",
            str(root / "plan"),
        )

        assert result.returncode == 2
        assert "config" in result.stdout.lower()
        assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_unresolved_hparam_user_config_fails_before_workspace_mutation(tmp_path: Path):
    for case, value in (("null", None), ("empty", ""), ("ask", "ASK_USER")):
        root = tmp_path / case
        recipe = _hparam_recipe(root)
        decisions = root / "decisions.yaml"
        decisions.write_text(yaml.safe_dump({"decisions": {"config": {"value": value, "source": "explicit_user"}}}))
        before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

        result = _run(
            "plan",
            "--recipe",
            str(recipe),
            "--user-decisions",
            str(decisions),
            "--output-dir",
            str(root / "plan"),
        )

        assert result.returncode == 2
        assert "config" in result.stdout.lower()
        assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_resolved_hparam_user_config_owns_consultation_and_snapshot(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    recipe_payload = yaml.safe_load(recipe.read_text())
    base_recipe = yaml.safe_load(Path(recipe_payload["base_recipe"]).read_text())
    base_config = yaml.safe_load(Path(base_recipe["inputs"]["config"]).read_text())
    selected_config = tmp_path / "selected.yaml"
    base_config["data"]["max_tokens"] = 5
    selected_config.write_text(yaml.safe_dump(base_config, sort_keys=False))
    decisions = tmp_path / "decisions.yaml"
    decisions.write_text(
        yaml.safe_dump({"decisions": {"config": {"value": str(selected_config), "source": "explicit_user"}}})
    )

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(tmp_path / "plan"),
    )

    assert result.returncode == 0, result.stderr or result.stdout
    run = _first_run(tmp_path / "plan")
    assert yaml.safe_load(Path(run["config"]).read_text())["data"]["max_tokens"] == 5
    plan = json.loads((tmp_path / "plan" / "plan.json").read_text())
    assert plan["recipe"]["inputs"]["config"] == str(selected_config)
    assert plan["recipe"]["_base_recipe"]["inputs"]["config"] != str(selected_config)


def test_hparam_user_decisions_freeze_one_effective_recipe(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"]["selection_metric"] = "val_wrong"
    write_yaml(recipe, payload)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {
            "decisions": {
                "selection_metric": {"value": "val_ahi_pearson", "source": "explicit_user"},
                "selection_mode": {"value": "max", "source": "explicit_user"},
                "train_val_test_policy": {"value": "val", "source": "explicit_user"},
                "hparam_search_space": {
                    "value": {"runtime.lr": [2e-6]},
                    "source": "explicit_user",
                },
                "hparam_budget": {"value": 1, "source": "explicit_user"},
            }
        },
    )
    plan_dir = tmp_path / "plan"

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(plan_dir),
    )

    assert result.returncode == 0, result.stderr or result.stdout
    plan = json.loads((plan_dir / "plan.json").read_text())
    effective = plan["recipe"]
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    assert effective["evaluation_policy"]["selection_metric"] == "val_ahi_pearson"
    assert effective["evaluation_policy"]["selection_split"] == "val"
    assert effective["_local_recipe"]["evaluation_policy"]["selection_metric"] == "val_wrong"
    assert effective["search"] == {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [2e-6]}}
    assert effective["_local_recipe"]["search"] != effective["search"]
    assert "--lr 2e-06" in plan["runs"][0]["command"]
    assert resolved == {key: value for key, value in effective.items() if key != "_recipe_path"}


def test_hparam_user_selection_metric_rechecks_config_monitor(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {
            "decisions": {
                "selection_metric": {"value": "val_other", "source": "explicit_user"},
            }
        },
    )
    plan_dir = tmp_path / "plan"

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(plan_dir),
    )

    assert result.returncode == 2
    assert "selection_metric differs" in result.stdout
    assert not (plan_dir / "runs").exists()


def test_resolved_hparam_user_config_rechecks_base_consultation(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    recipe_payload = yaml.safe_load(recipe.read_text())
    base_recipe = yaml.safe_load(Path(recipe_payload["base_recipe"]).read_text())
    base_config = yaml.safe_load(Path(base_recipe["inputs"]["config"]).read_text())
    selected_config = tmp_path / "selected-mismatch.yaml"
    base_config["finetune"]["task"]["monitor"] = "val_other"
    selected_config.write_text(yaml.safe_dump(base_config, sort_keys=False))
    decisions = tmp_path / "decisions.yaml"
    decisions.write_text(
        yaml.safe_dump({"decisions": {"config": {"value": str(selected_config), "source": "explicit_user"}}})
    )

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(tmp_path / "plan"),
    )

    assert result.returncode == 2
    assert "selection_metric differs" in result.stdout
    assert (tmp_path / "plan" / "plan.blocked.md").exists()
    assert not (tmp_path / "plan" / "runs").exists()


def test_missing_or_unsupported_task_without_workspace_returns_report(tmp_path: Path):
    for name, task, expected_returncode in (("missing", None, 2), ("unsupported", "unknown", 1)):
        root = tmp_path / name
        root.mkdir()
        payload = {"name": name, "variant": "sleep2vec", "inputs": {}}
        if task is not None:
            payload["task"] = task
        recipe = root / "recipe.yaml"
        recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
        before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

        result = _run("plan", "--recipe", str(recipe), "--output-dir", str(root / "plan"))

        assert result.returncode == expected_returncode
        assert "task" in result.stdout.lower()
        assert not result.stderr
        assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_plan_skips_survival_index_gate_when_finetune_preset_is_configured(tmp_path: Path):
    recipe, config = _survival_recipe_with_missing_sidecar_key(tmp_path)
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(b"preset")
    payload = yaml.safe_load(config.read_text())
    payload["data"]["finetune_preset_path"] = str(preset)
    write_yaml(config, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "survival key values missing from sidecars" not in result.stdout
    assert (output_dir / "run.sh").exists()


def test_plan_skips_missing_index_path_when_finetune_preset_is_configured(tmp_path: Path):
    recipe, config = _survival_recipe_with_missing_sidecar_key(tmp_path)
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(b"preset")
    payload = yaml.safe_load(config.read_text())
    payload["data"]["finetune_data_index"] = str(tmp_path / "missing_index.csv")
    payload["data"]["finetune_preset_path"] = str(preset)
    write_yaml(config, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "finetune_data_index" not in result.stdout
    assert (output_dir / "run.sh").exists()


def test_plan_blocks_missing_finetune_preset_path(tmp_path: Path):
    recipe, config = _survival_recipe_with_missing_sidecar_key(tmp_path)
    payload = yaml.safe_load(config.read_text())
    payload["data"]["finetune_preset_path"] = str(tmp_path / "missing_preset.pkl")
    write_yaml(config, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "finetune_preset_path" in result.stdout
    assert "missing_preset.pkl" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_hparam_plan_skips_survival_index_gate_when_base_preset_is_configured(tmp_path: Path):
    base, config = _survival_recipe_with_missing_sidecar_key(tmp_path)
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(b"preset")
    payload = yaml.safe_load(config.read_text())
    payload["data"]["finetune_preset_path"] = str(preset)
    write_yaml(config, payload)
    recipe = write_yaml(
        tmp_path / "tune_survival.yaml",
        {
            "name": "unit_tune_survival",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base),
            "search": {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1e-6]}},
            "evaluation_policy": {
                "selection_metric": "val_loss",
                "selection_mode": "min",
                "selection_split": "val",
                "final_eval_split": "test",
                "external_test_locked": True,
                "test_after_fit": False,
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "survival key values missing from sidecars" not in result.stdout
    assert (output_dir / "run_all.sh").exists()


def test_hparam_plan_skips_remote_deferred_survival_index_summary(tmp_path: Path):
    index = tmp_path / "local_index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\nx.npz,train,60,001,1\n")
    payload = survival_config_payload(
        index,
        {
            "disease_columns_index": "/wujidata/survival/disease_columns.txt",
            "event_time_index": "/wujidata/survival/event_time.csv",
            "is_event_index": "/wujidata/survival/is_event.csv",
            "has_label_index": "/wujidata/survival/has_label.csv",
        },
    )
    payload["data"]["finetune_data_index"] = "/wujidata/survival/index.csv"
    config = write_yaml(tmp_path / "remote_survival_config.yaml", payload)
    base = _survival_recipe_with_missing_sidecar_key(tmp_path)[0]
    base_payload = yaml.safe_load(base.read_text())
    base_payload["inputs"]["config"] = str(config)
    write_yaml(base, base_payload)
    recipe = write_yaml(
        tmp_path / "tune_remote_survival.yaml",
        {
            "name": "unit_tune_remote_survival",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base),
            "search": {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1e-6]}},
            "evaluation_policy": {
                "selection_metric": "val_loss",
                "selection_mode": "min",
                "selection_split": "val",
                "final_eval_split": "test",
                "external_test_locked": True,
                "test_after_fit": False,
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "execution": {
                "target": "ssh",
                "host": "baichuan3",
                "path_context": "remote",
                "path_validation": "defer",
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "Index CSV not found" not in result.stdout
    assert (output_dir / "run_all.sh").exists()


def test_context_blocks_survival_index_keys_missing_from_sidecars(tmp_path: Path):
    _recipe, config = _survival_recipe_with_missing_sidecar_key(tmp_path)
    output_dir = tmp_path / "context"

    result = _run(
        "context",
        "--task",
        "finetune",
        "--variant",
        "sleep2vec",
        "--label-name",
        "incident_cox",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode in {1, 2}
    assert (output_dir / "commands.blocked.sh").exists()
    assert not (output_dir / "commands.sh").exists()
    context = json.loads((output_dir / "context.json").read_text())
    assert any("survival key values missing from sidecars" in issue for issue in context["blocking_issues"])


def test_context_writes_questions_and_blocked_script(tmp_path: Path):
    config = yaml.safe_load(write_finetune_recipe(tmp_path).read_text())["inputs"]["config"]
    output_dir = tmp_path / "context"

    result = _run("context", "--task", "finetune", "--config", config, "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert (output_dir / "questions.md").exists()
    assert (output_dir / "commands.blocked.sh").exists()
    assert not (output_dir / "commands.sh").exists()
    context = json.loads((output_dir / "context.json").read_text())
    assert context["skill"]["name"] == "finetuning"
    assert "runtime-orchestrator" in context["owners"]
    assert context["relevant_docs"]
    assert context["index_summary"]["rows"] == 1
    assert context["preset_summary"] is None
    assert context["expected_artifacts"]


def test_context_writes_questions_for_mixed_fail_and_user_input(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    config = Path(yaml.safe_load(recipe.read_text())["inputs"]["config"])
    payload = yaml.safe_load(config.read_text())
    payload["data"]["finetune_data_index"] = None
    payload["data"]["finetune_preset_path"] = str(tmp_path / "missing preset.pkl")
    write_yaml(config, payload)
    output_dir = tmp_path / "context"

    result = _run(
        "context",
        "--task",
        "finetune",
        "--variant",
        "sleep2vec",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 1
    context = json.loads((output_dir / "context.json").read_text())
    assert context["consultation_required"] is True
    assert (output_dir / "questions.md").exists()
    assert (output_dir / "commands.blocked.sh").exists()
    assert not (output_dir / "commands.sh").exists()


def test_unlock_final_test_required_for_final_external_script(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    blocked_output = tmp_path / "blocked"
    unlocked_output = tmp_path / "unlocked"

    blocked = _run("plan", "--recipe", str(recipe), "--output-dir", str(blocked_output))
    unlocked = _run("plan", "--recipe", str(recipe), "--output-dir", str(unlocked_output), "--unlock-final-test")

    assert blocked.returncode == 0
    assert not (blocked_output / "final_external_test.sh").exists()
    assert unlocked.returncode == 2
    assert not (unlocked_output / "final_external_test.sh").exists()


def test_unlock_final_test_with_explicit_ckpt_generates_final_script(tmp_path: Path):
    ckpt = tmp_path / "best model.ckpt"
    ckpt.write_text("checkpoint")
    recipe = _hparam_recipe(tmp_path, ckpt_path=ckpt)
    output_dir = tmp_path / "unlocked"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 0
    script = (output_dir / "final_external_test.sh").read_text()
    assert "python -m sleep2vec.infer" in script
    assert shlex_quote(str(ckpt)) in script
    assert "This script evaluates the configured final test split." in script
    assert "Run commands do not evaluate the external test split." not in script


def test_plan_uses_user_decision_label_name_in_command(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"label_name": {"value": "ahi", "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--user-decisions", str(decisions), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = (output_dir / "run.sh").read_text()
    assert "--label-name ahi" in script
    assert "--label-name --version-name" not in script


def test_plan_uses_user_decision_test_after_fit_in_command(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].pop("test_after_fit")
    write_yaml(recipe, payload)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"test_after_fit": {"value": False, "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--user-decisions", str(decisions), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "--no-test-after-fit" in (output_dir / "run.sh").read_text()


def test_plan_blocks_user_decision_test_after_fit_when_finetune_lock_stays_resolved(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"test_after_fit": {"value": True, "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--user-decisions", str(decisions), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "test_after_fit=true would evaluate test" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_plan_normalizes_scalar_runtime_devices(tmp_path: Path):
    for value, expected in [(0, "--devices 0"), ("10", "--devices 10")]:
        recipe = write_finetune_recipe(tmp_path / str(value))
        payload = yaml.safe_load(recipe.read_text())
        payload["runtime"]["devices"] = value
        write_yaml(recipe, payload)
        output_dir = recipe.parent / "plan"

        result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

        assert result.returncode == 0, result.stderr
        script = (output_dir / "run.sh").read_text()
        assert expected in script
        assert "--devices 1 0" not in script


def test_finetune_plan_honors_runtime_wandb_mode_env(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["runtime"]["wandb_mode"] = "offline"
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "WANDB_MODE=offline python -m sleep2vec.finetune" in (output_dir / "run.sh").read_text()


def test_hparam_run_script_honors_base_runtime_wandb_mode_env(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"]["wandb_mode"] = "offline"
    write_yaml(base_recipe, base_payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "WANDB_MODE=offline python -m sleep2vec.finetune" in Path(_first_run(output_dir)["script"]).read_text()


def test_hparam_plan_and_launch_use_merged_effective_recipe(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["label_name"] = "effective-label"
    payload["runtime"] = {"devices": [4], "batch_size": 7}
    payload["artifacts"] = {"results_csv_path": "effective/results.csv"}
    payload["decisions"]["label_name"]["value"] = "effective-label"
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    planned = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))
    launched = _run("hparam-launch", "--plan-dir", str(output_dir))

    assert planned.returncode == 0, planned.stderr or planned.stdout
    assert launched.returncode == 0, launched.stderr or launched.stdout
    plan = json.loads((output_dir / "plan.json").read_text())
    command = plan["runs"][0]["command"]
    assert "--label-name effective-label" in command
    assert "--batch-size 7" in command
    assert f"--results-csv-path {output_dir / 'effective/results.csv'}" in command
    assert plan["recipe"]["runtime"]["devices"] == [4]
    assert plan["recipe"]["_base_recipe"]["runtime"]["devices"] != [4]
    with (output_dir / "launch_manifest.tsv").open(newline="") as file_obj:
        launch_rows = list(csv.DictReader(file_obj, delimiter="\t"))
    assert launch_rows[0]["gpus"] == "4"


def test_infer_eval_split_ask_user_blocks_command_generation(tmp_path: Path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("checkpoint")
    config = yaml.safe_load(write_finetune_recipe(tmp_path).read_text())["inputs"]["config"]
    recipe = write_yaml(
        tmp_path / "infer.yaml",
        {
            "name": "unit_infer",
            "task": "infer",
            "variant": "sleep2vec",
            "inputs": {
                "config": config,
                "label_name": "ahi",
                "ckpt_path": str(ckpt),
                "eval_split": "ASK_USER",
            },
            "evaluation_policy": {"external_test_locked": True, "final_test_unlocked": False},
            "decisions": {
                "task": {"value": "infer", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "ckpt_path": {"value": str(ckpt), "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "eval_split" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_infer_invalid_eval_split_blocks_command_generation(tmp_path: Path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("checkpoint")
    config = yaml.safe_load(write_finetune_recipe(tmp_path).read_text())["inputs"]["config"]
    recipe = write_yaml(
        tmp_path / "infer.yaml",
        {
            "name": "unit_infer",
            "task": "infer",
            "variant": "sleep2vec",
            "inputs": {
                "config": config,
                "label_name": "ahi",
                "ckpt_path": str(ckpt),
                "eval_split": "validation",
            },
            "evaluation_policy": {"external_test_locked": False, "final_test_unlocked": False},
            "decisions": {
                "task": {"value": "infer", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "ckpt_path": {"value": str(ckpt), "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "eval_split must be one of" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_infer_blocks_missing_inference_preset_path(tmp_path: Path):
    config = _write_survival_config_with_bad_sidecars(tmp_path)
    recipe = _write_infer_recipe(tmp_path, config, inference_preset_path=tmp_path / "missing_override.pkl")
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "inference_preset_path" in result.stdout
    assert "missing_override.pkl" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_infer_blocks_missing_config_finetune_preset_path(tmp_path: Path):
    config = _write_survival_config_with_bad_sidecars(tmp_path, preset_path=tmp_path / "missing_config_preset.pkl")
    recipe = _write_infer_recipe(tmp_path, config)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "finetune_preset_path" in result.stdout
    assert "missing_config_preset.pkl" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_infer_survival_blocks_invalid_sidecars_without_preset(tmp_path: Path):
    config = _write_survival_config_with_bad_sidecars(tmp_path)
    recipe = _write_infer_recipe(tmp_path, config)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "survival_sidecars" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_infer_survival_allows_invalid_sidecars_with_preset(tmp_path: Path):
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(b"preset")
    config = _write_survival_config_with_bad_sidecars(tmp_path, preset_path=preset)
    recipe = _write_infer_recipe(tmp_path, config)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "survival_sidecars" not in result.stdout
    assert (output_dir / "run.sh").exists()


def test_infer_preset_path_does_not_skip_survival_sidecar_checks(tmp_path: Path):
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(b"preset")
    config = _write_survival_config_with_bad_sidecars(tmp_path)
    recipe = _write_infer_recipe(tmp_path, config)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["preset_path"] = str(preset)
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "survival_sidecars" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_infer_checks_survival_sidecar_keys_only_for_eval_split(tmp_path: Path):
    index = tmp_path / "survival_infer_index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\n" "val.npz,val,60,001,1\n" "test.npz,test,60,003,1\n")
    config = write_yaml(
        tmp_path / "survival_infer_config.yaml",
        survival_config_payload(index, write_survival_sidecars(tmp_path)),
    )
    recipe = _write_infer_recipe(tmp_path, config, eval_split="val")
    output_dir = tmp_path / "plan_infer_val"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "survival key values missing from sidecars" not in result.stdout
    assert (output_dir / "run.sh").exists()


def test_unlock_final_test_with_yaml_search_requires_explicit_final_config(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    recipe = _hparam_recipe(tmp_path, parameters={"yaml:/finetune/task/output_dim": [31]}, ckpt_path=ckpt)
    output_dir = tmp_path / "unlocked"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 2
    assert not (output_dir / "final_external_test.sh").exists()
    assert "final_eval_config_path" in (output_dir / "questions.md").read_text()


def test_unlock_final_test_with_yaml_search_uses_explicit_final_config(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    selected_config = tmp_path / "selected_run.yaml"
    recipe = _hparam_recipe(
        tmp_path,
        parameters={"yaml:/finetune/task/output_dim": [31]},
        ckpt_path=ckpt,
        final_config_path=selected_config,
    )
    selected_config.write_text((tmp_path / "config.yaml").read_text())
    output_dir = tmp_path / "unlocked"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 0
    script = (output_dir / "final_external_test.sh").read_text()
    assert shlex_quote(str(selected_config)) in script
    assert "runs/run-000" not in script


def test_infer_user_decision_ckpt_path_must_exist(tmp_path: Path):
    config = yaml.safe_load(write_finetune_recipe(tmp_path).read_text())["inputs"]["config"]
    recipe = write_yaml(
        tmp_path / "infer.yaml",
        {
            "name": "unit_infer",
            "task": "infer",
            "variant": "sleep2vec",
            "inputs": {
                "config": config,
                "label_name": "ahi",
                "ckpt_path": "ASK_USER",
                "eval_split": "validation",
            },
            "evaluation_policy": {"final_test_unlocked": False},
            "artifacts": {"overwrite": True},
            "decisions": {
                "task": {"value": "infer", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "overwrite_policy": {"value": True, "source": "explicit_recipe"},
            },
        },
    )
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {
            "decisions": {"ckpt_path": {"value": str(tmp_path / "missing.ckpt"), "source": "explicit_user"}},
        },
    )
    output_dir = tmp_path / "plan"

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 1
    assert "ckpt_path" in (output_dir / "questions.md").read_text()
    assert not (output_dir / "run.sh").exists()


def test_variant_controls_generated_finetune_module(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, variant="sleep2vec2")
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = (output_dir / "run.sh").read_text()
    assert "python -m sleep2vec2.finetune" in script
    assert "--no-test-after-fit" in script


@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_model_variant_controls_generated_hparam_module(tmp_path: Path, variant: str):
    recipe = _hparam_recipe(tmp_path, variant=variant)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = Path(_first_run(output_dir)["script"]).read_text()
    assert f"cd {shlex_quote(str(REPO_ROOT))}" in script
    assert f"export PYTHONPATH={shlex_quote(str(REPO_ROOT))}" in script
    assert f"python -m {variant}.finetune" in script
    assert "--no-test-after-fit" in script
    assert f"--results-csv-path {shlex_quote(str(tmp_path / 'results.csv'))}" in script


def test_hparam_plan_allows_test_after_fit_when_explicitly_unlocked(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, variant="sleep2vec2")
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].update(
        {
            "external_test_locked": False,
            "test_after_fit": True,
            "final_test_unlocked": True,
            "require_manual_unlock_for_final_test": False,
        }
    )
    payload["decisions"].update(
        {
            "external_test_locked": {"value": False, "source": "explicit_user"},
            "test_after_fit": {"value": True, "source": "explicit_user"},
            "final_eval_unlock": {"value": True, "source": "explicit_user"},
        }
    )
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = Path(_first_run(output_dir)["script"]).read_text()
    assert "python -m sleep2vec2.finetune" in script
    assert "--no-test-after-fit" not in script
    assert "Run commands evaluate the configured test split after fit." in script
    assert "Run commands do not evaluate the external test split." not in script
    assert "Run commands evaluate the configured test split after fit." in (output_dir / "run_all.sh").read_text()
    assert not (output_dir / "final_external_test.sh").exists()
    plan = (output_dir / "plan.md").read_text()
    assert "Run commands evaluate the configured test split" in plan
    assert "explicit checkpoint path is required" in plan
    assert "explicit unlock is required" not in plan


def test_hparam_plan_guards_stale_final_script_when_unlocked_without_ckpt(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, variant="sleep2vec2")
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].update(
        {
            "external_test_locked": False,
            "test_after_fit": True,
            "final_test_unlocked": True,
            "require_manual_unlock_for_final_test": False,
        }
    )
    payload["decisions"].update(
        {
            "external_test_locked": {"value": False, "source": "explicit_user"},
            "test_after_fit": {"value": True, "source": "explicit_user"},
            "final_eval_unlock": {"value": True, "source": "explicit_user"},
        }
    )
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"
    output_dir.mkdir()
    stale_final_script = output_dir / "final_external_test.sh"
    stale_final_script.write_text("# stale final test script\n")

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "Output artifacts already exist" in result.stdout
    assert str(stale_final_script) in result.stdout
    assert not (output_dir / "plan.md").exists()


def test_hparam_plan_removes_stale_final_script_when_overwrite_allowed_without_ckpt(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, variant="sleep2vec2")
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].update(
        {
            "external_test_locked": False,
            "test_after_fit": True,
            "final_test_unlocked": True,
            "require_manual_unlock_for_final_test": False,
        }
    )
    payload["decisions"].update(
        {
            "external_test_locked": {"value": False, "source": "explicit_user"},
            "test_after_fit": {"value": True, "source": "explicit_user"},
            "final_eval_unlock": {"value": True, "source": "explicit_user"},
            "overwrite_policy": {"value": True, "source": "explicit_user"},
        }
    )
    payload["artifacts"] = {**payload.get("artifacts", {}), "overwrite": True}
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"
    output_dir.mkdir()
    stale_final_script = output_dir / "final_external_test.sh"
    stale_final_script.write_text("# stale final test script\n")

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert not stale_final_script.exists()
    plan = (output_dir / "plan.md").read_text()
    assert "explicit checkpoint path is required" in plan
    assert "Final external-test script generated" not in plan


def test_hparam_plan_blocks_user_test_after_fit_when_lock_stays_resolved(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, variant="sleep2vec2")
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"test_after_fit": {"value": True, "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 2
    assert "test_after_fit" in (output_dir / "questions.md").read_text()
    assert not (output_dir / "runs").exists()


def test_pretrain_and_adapt_are_not_runnable_recipe_tasks(tmp_path: Path):
    pretrained = tmp_path / "pretrained.ckpt"
    pretrained.write_text("checkpoint")
    recipes = []
    for task in ("pretrain", "adapt"):
        recipe = {
            "name": f"unit_{task}",
            "task": task,
            "variant": "sleep2vec",
            "inputs": {},
            "artifacts": {"output_dir": str(tmp_path / task)},
            "decisions": {"task": {"value": task, "source": "explicit_recipe"}},
        }
        if task == "adapt":
            recipe["inputs"]["pretrained_backbone_path"] = str(pretrained)
            recipe["decisions"]["pretrained_backbone_path"] = {
                "value": str(pretrained),
                "source": "explicit_recipe",
            }
        recipes.append((task, write_yaml(tmp_path / f"{task}.yaml", recipe)))

    for task, recipe in recipes:
        output_dir = tmp_path / f"{task}_plan"
        result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

        assert result.returncode == 1
        assert f"Unsupported task: {task}" in result.stdout
        assert (output_dir / "plan.blocked.md").exists()
        assert not (output_dir / "run.sh").exists()


def test_context_without_workspace_writes_blocked_script(tmp_path: Path):
    output_dir = tmp_path / "context"

    result = _run("context", "--task", "pretrain", "--variant", "sleep2vec", "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "Unsupported task: pretrain" in json.loads((output_dir / "context.json").read_text())["blocking_issues"]
    assert (output_dir / "commands.blocked.sh").exists()
    assert not (output_dir / "commands.sh").exists()
    assert not (output_dir / "validation.sh").exists()


def test_missing_variant_blocks_command_generation(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.pop("variant")
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert not (output_dir / "run.sh").exists()


def test_plan_refuses_existing_artifact_when_overwrite_false(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    output_dir = tmp_path / "plan"
    output_dir.mkdir()
    run_script = output_dir / "run.sh"
    run_script.write_text("keep me")

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert run_script.read_text() == "keep me"


def test_plan_overwrite_rejects_output_alias_to_canonical_without_writing(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path / "source")
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    payload["decisions"]["overwrite_policy"]["value"] = True
    payload["artifacts"]["overwrite"] = True
    recipe.write_text(yaml.safe_dump(payload))
    initial = _run("plan", "--recipe", str(recipe), "--output-dir", str(workspace / "plan-1"))
    assert initial.returncode == 0, initial.stderr or initial.stdout
    output_dir = workspace / "plan-2"
    output_dir.mkdir()
    canonical_before = (workspace / "run_manifest.tsv").read_bytes()
    step_manifest = workspace / "steps" / payload["step"]["id"] / "step.yaml"
    step_before = step_manifest.read_bytes()
    (output_dir / "plan.json").hardlink_to(workspace / "run_manifest.tsv")

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert (workspace / "run_manifest.tsv").read_bytes() == canonical_before
    assert step_manifest.read_bytes() == step_before
    assert not (output_dir / "runs").exists()


def test_plan_allows_existing_workspace_matrix_and_events_for_new_plan(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path / "source")
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload))

    first = _run("plan", "--recipe", str(recipe), "--output-dir", str(workspace / "plan-1"))
    second = _run("plan", "--recipe", str(recipe), "--output-dir", str(workspace / "plan-2"))

    assert first.returncode == 0, first.stderr or first.stdout
    assert second.returncode == 0, second.stderr or second.stdout
    assert (workspace / "plan-2" / "plan.json").exists()


def test_plan_refuses_existing_blocked_artifact_when_overwrite_missing(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    payload = yaml.safe_load(recipe.read_text())
    payload["decisions"].pop("overwrite_policy")
    payload["artifacts"].pop("overwrite")
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"
    output_dir.mkdir()
    blocked_plan = output_dir / "plan.blocked.md"
    blocked_plan.write_text("keep me")

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert blocked_plan.read_text() == "keep me"


def test_generated_commands_quote_paths_with_spaces(tmp_path: Path):
    root = tmp_path / "path with space"
    root.mkdir()
    recipe = write_finetune_recipe(root)
    output_dir = root / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = (output_dir / "run.sh").read_text()
    frozen_config = output_dir / "runs" / "run-000--unit" / "config.yaml"
    assert shlex_quote(str(frozen_config)) in script
    assert shlex_quote(str(root / "config.yaml")) not in script


def test_preset_plan_includes_explicit_preset_args(tmp_path: Path):
    base = write_finetune_recipe(tmp_path)
    config = yaml.safe_load(base.read_text())["inputs"]["config"]
    index = tmp_path / "preset_index.csv"
    index.write_text("path,split,duration,ppg_mask,ah_event_mask\nx.npz,train,60,1,1\n")
    output_template = tmp_path / "{dataset}_{split}_{tokens}.pkl"
    manifest_output = tmp_path / "manifest.json"
    recipe = _write_preset_recipe(
        tmp_path,
        config=config,
        index=index,
        preset={
            "n_tokens": 128,
            "stride_tokens": 64,
            "split": ["train"],
            "channels": ["ppg", "ahi"],
            "meta_data_names": ["age"],
            "include_no_metadata": True,
            "allow_missing_channels": True,
            "min_channels": 2,
            "output_template": str(output_template),
            "overwrite": True,
            "batch_size": 4,
            "shuffle": False,
            "mask_rate": 0.1,
            "dry_run": True,
            "manifest_output": str(manifest_output),
            "write_sidecar_manifest": False,
        },
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = (output_dir / "run.sh").read_text()
    assert "--stride-tokens 64" in script
    assert "--channels ppg ahi" in script
    assert "--meta-data-names age" in script
    assert "--include-no-metadata" in script
    assert "--allow-missing-channels" in script
    assert "--min-channels 2" in script
    assert f"--output-template {shlex_quote(str(output_template))}" in script
    assert "--overwrite" in script
    assert "--batch-size 4" in script
    assert "--no-shuffle" in script
    assert "--mask-rate 0.1" in script
    assert "--dry-run" in script
    assert f"--manifest-output {shlex_quote(str(manifest_output))}" in script
    assert "--no-write-sidecar-manifest" in script


@pytest.mark.parametrize(
    ("variant", "expected_script"),
    [
        ("sleep2vec", "preprocess/save_dataset_presets.py"),
        ("sleep2vec2", "sleep2vec2/preprocess/save_dataset_presets.py"),
        ("sleep2expert", "sleep2expert/preprocess/save_dataset_presets.py"),
    ],
)
def test_preset_plan_routes_to_variant_local_script(tmp_path: Path, variant: str, expected_script: str):
    base = write_finetune_recipe(tmp_path, variant=variant)
    config = yaml.safe_load(base.read_text())["inputs"]["config"]
    index = tmp_path / "preset_index.csv"
    index.write_text("path,split,duration,ppg_mask\nx.npz,train,60,1\n")
    recipe = _write_preset_recipe(tmp_path, config=config, index=index, variant=variant)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    assert expected_script in (output_dir / "run.sh").read_text()


@pytest.mark.parametrize(
    ("field", "value"),
    [("manifest_output", "manifest.json"), ("write_sidecar_manifest", False)],
)
@pytest.mark.parametrize("variant", ["sleep2vec2", "sleep2expert"])
def test_variant_preset_rejects_root_only_manifest_flags_before_writing(
    tmp_path: Path,
    variant: str,
    field: str,
    value: str | bool,
):
    base = write_finetune_recipe(tmp_path, variant=variant)
    config = yaml.safe_load(base.read_text())["inputs"]["config"]
    index = tmp_path / "preset_index.csv"
    index.write_text("path,split,duration,ppg_mask\nx.npz,train,60,1\n")
    recipe = _write_preset_recipe(
        tmp_path,
        config=config,
        index=index,
        variant=variant,
        preset={"n_tokens": 128, "split": ["train"], "allow_missing_channels": False, field: value},
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert f"does not support {field}" in result.stdout
    assert not (output_dir / "run.sh").exists()
    assert not (output_dir / "plan.json").exists()


def test_preset_plan_checks_inputs_index_even_when_config_has_finetune_preset(tmp_path: Path):
    base = write_finetune_recipe(tmp_path)
    config = Path(yaml.safe_load(base.read_text())["inputs"]["config"])
    payload = yaml.safe_load(config.read_text())
    preset = tmp_path / "existing_preset.pkl"
    preset.write_bytes(b"preset")
    payload["data"]["finetune_preset_path"] = str(preset)
    write_yaml(config, payload)
    bad_index = tmp_path / "bad_preset_index.csv"
    bad_index.write_text("eid\n001\n")
    recipe = write_yaml(
        tmp_path / "preset_bad_index.yaml",
        {
            "name": "unit_preset_bad_index",
            "task": "preset_prepare",
            "variant": "sleep2vec",
            "inputs": {"config": str(config), "index": [str(bad_index)], "dataset_name": "unit"},
            "preset": {"n_tokens": 128, "split": ["train"]},
            "decisions": {
                "task": {"value": "preset_prepare", "source": "explicit_recipe"},
                "preset_regeneration": {"value": True, "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "required_channels": {"value": ["ppg"], "source": "explicit_recipe"},
                "min_channels": {"value": 1, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan_bad_index"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "Index CSV missing required column: path" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_preset_plan_blocks_survival_config_with_invalid_sidecars(tmp_path: Path):
    config = _write_survival_config_with_bad_sidecars(tmp_path)
    index = tmp_path / "preset_index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\nx.npz,train,60,001,1\n")
    recipe = write_yaml(
        tmp_path / "preset_survival_bad_sidecars.yaml",
        {
            "name": "unit_preset_survival_bad_sidecars",
            "task": "preset_prepare",
            "variant": "sleep2vec",
            "inputs": {"config": str(config), "index": [str(index)], "dataset_name": "unit"},
            "preset": {"n_tokens": 128, "split": ["train"], "allow_missing_channels": False},
            "decisions": {
                "task": {"value": "preset_prepare", "source": "explicit_recipe"},
                "preset_regeneration": {"value": True, "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "required_channels": {"value": ["ppg"], "source": "explicit_recipe"},
                "min_channels": {"value": 1, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan_bad_sidecars"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "survival_sidecars" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_preset_plan_checks_survival_sidecar_keys_only_for_requested_split(tmp_path: Path):
    index = tmp_path / "preset_survival_index.csv"
    index.write_text("path,split,duration,eid,ppg_mask\n" "train.npz,train,60,001,1\n" "test.npz,test,60,003,1\n")
    config = write_yaml(
        tmp_path / "survival_config.yaml",
        survival_config_payload(index, write_survival_sidecars(tmp_path)),
    )
    recipe = write_yaml(
        tmp_path / "preset_survival_train.yaml",
        {
            "name": "unit_preset_survival_train",
            "task": "preset_prepare",
            "variant": "sleep2vec",
            "inputs": {"config": str(config), "index": [str(index)], "dataset_name": "unit"},
            "preset": {"n_tokens": 128, "split": ["train"], "allow_missing_channels": False},
            "decisions": {
                "task": {"value": "preset_prepare", "source": "explicit_recipe"},
                "preset_regeneration": {"value": True, "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "required_channels": {"value": ["ppg"], "source": "explicit_recipe"},
                "min_channels": {"value": 1, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan_survival_train"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "survival key values missing from sidecars" not in result.stdout
    assert (output_dir / "run.sh").exists()


def test_preset_plan_skips_local_index_summary_for_remote_deferred_index(tmp_path: Path):
    config = yaml.safe_load(write_finetune_recipe(tmp_path).read_text())["inputs"]["config"]
    recipe = write_yaml(
        tmp_path / "preset_remote_index.yaml",
        {
            "name": "unit_preset_remote_index",
            "task": "preset_prepare",
            "variant": "sleep2vec",
            "inputs": {"config": config, "index": ["/wujidata/index.csv"], "dataset_name": "unit"},
            "preset": {"n_tokens": 128, "split": ["train"], "allow_missing_channels": False},
            "execution": {
                "target": "ssh",
                "host": "baichuan3",
                "path_context": "remote",
                "path_validation": "defer",
            },
            "decisions": {
                "task": {"value": "preset_prepare", "source": "explicit_recipe"},
                "preset_regeneration": {"value": True, "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "required_channels": {"value": ["ppg"], "source": "explicit_recipe"},
                "min_channels": {"value": 1, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan_remote_index"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "Index CSV not found" not in result.stdout
    assert (output_dir / "run.sh").exists()


def test_finetune_plan_includes_explicit_input_and_runtime_args(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    pretrained = tmp_path / "pretrained model.ckpt"
    resume = tmp_path / "resume checkpoint.ckpt"
    pretrained.write_text("checkpoint")
    resume.write_text("checkpoint")
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["pretrained_backbone_path"] = str(pretrained)
    payload["inputs"]["ckpt_path"] = str(resume)
    payload["runtime"].update(
        {
            "device": "cuda",
            "warmup_steps": 11,
            "gradient_clip_val": 0.5,
            "accumulate_grad_batches": 2,
            "patience": 4,
            "check_val_every_n_epoch": 2,
            "ckpt_every_n_epochs": 3,
        }
    )
    payload["decisions"]["pretrained_backbone_path"] = {
        "value": str(pretrained),
        "source": "explicit_recipe",
    }
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = (output_dir / "run.sh").read_text()
    assert f"--pretrained-backbone-path {shlex_quote(str(pretrained))}" in script
    assert f"--ckpt-path {shlex_quote(str(resume))}" in script
    assert "--device cuda" in script
    assert "--warmup-steps 11" in script
    assert "--gradient-clip-val 0.5" in script
    assert "--accumulate-grad-batches 2" in script
    assert "--patience 4" in script
    assert "--check-val-every-n-epoch 2" in script
    assert "--ckpt-every-n-epochs 3" in script


def test_finetune_blocks_missing_pretrained_backbone_path(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    missing = tmp_path / "missing_pretrained.ckpt"
    payload["inputs"]["pretrained_backbone_path"] = str(missing)
    payload["decisions"]["pretrained_backbone_path"] = {
        "value": str(missing),
        "source": "explicit_recipe",
    }
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "pretrained_backbone_path" in result.stdout
    assert "missing_pretrained.ckpt" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_finetune_blocks_missing_resume_ckpt_path(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["ckpt_path"] = str(tmp_path / "missing_resume.ckpt")
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "ckpt_path" in result.stdout
    assert "missing_resume.ckpt" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_hparam_rejects_bare_search_parameter(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, parameters={"lr": [1e-6]})
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert not (output_dir / "run_all.sh").exists()


def test_hparam_rejects_non_positive_max_runs(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, max_runs=0)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "hparam_budget" in (output_dir / "questions.md").read_text()
    assert not (output_dir / "run_all.sh").exists()


def test_hparam_rejects_removed_max_trials_field(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_trials"] = payload["search"].pop("max_runs")
    recipe.write_text(yaml.safe_dump(payload))

    result = _run("doctor", "--recipe", str(recipe))

    assert result.returncode == 1
    assert "search.max_trials is no longer supported" in result.stdout


def test_hparam_rejects_removed_adaptive_field_when_adaptive_is_disabled(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"] = {"enabled": False, "max_trials_total": 4}
    recipe.write_text(yaml.safe_dump(payload))

    result = _run("doctor", "--recipe", str(recipe))

    assert result.returncode == 1
    assert "adaptive.max_trials_total is no longer supported" in result.stdout


@pytest.mark.parametrize(
    ("section", "field"),
    [("execution", "max_concurent"), ("evaluation_policy", "selection_metic")],
)
def test_hparam_rejects_unknown_execution_and_evaluation_fields(
    tmp_path: Path,
    section: str,
    field: str,
):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.setdefault(section, {})[field] = 1
    recipe.write_text(yaml.safe_dump(payload))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert f"{section}.{field}" in result.stdout
    assert not output_dir.exists()


def test_hparam_runtime_parameter_reaches_run_script(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, parameters={"runtime.lr": [2e-6]})
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "--lr 2e-06" in Path(_first_run(output_dir)["script"]).read_text()


def test_hparam_runtime_training_knobs_reach_run_script(tmp_path: Path):
    recipe = _hparam_recipe(
        tmp_path,
        parameters={
            "runtime.gradient_clip_val": [0.5],
            "runtime.accumulate_grad_batches": [2],
            "runtime.warmup_steps": [500],
            "runtime.patience": [4],
            "runtime.check_val_every_n_epoch": [2],
            "runtime.ckpt_every_n_epochs": [3],
        },
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = Path(_first_run(output_dir)["script"]).read_text()
    assert "--gradient-clip-val 0.5" in script
    assert "--accumulate-grad-batches 2" in script
    assert "--warmup-steps 500" in script
    assert "--patience 4" in script
    assert "--check-val-every-n-epoch 2" in script
    assert "--ckpt-every-n-epochs 3" in script


def test_hparam_run_includes_base_input_and_runtime_args(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    pretrained = tmp_path / "base pretrained.ckpt"
    pretrained.write_text("checkpoint")
    base_payload["inputs"]["pretrained_backbone_path"] = str(pretrained)
    base_payload["runtime"].update({"warmup_steps": 7, "gradient_clip_val": 0.75, "accumulate_grad_batches": 3})
    base_payload["decisions"]["pretrained_backbone_path"] = {
        "value": str(pretrained),
        "source": "explicit_recipe",
    }
    write_yaml(base_recipe, base_payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = Path(_first_run(output_dir)["script"]).read_text()
    assert f"--pretrained-backbone-path {shlex_quote(str(pretrained))}" in script
    assert "--warmup-steps 7" in script
    assert "--gradient-clip-val 0.75" in script
    assert "--accumulate-grad-batches 3" in script


def test_hparam_blocks_when_base_finetune_pretrained_decision_is_missing(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["decisions"].pop("pretrained_backbone_path")
    write_yaml(base_recipe, base_payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "base_finetune.pretrained_backbone_path" in result.stdout
    assert not (output_dir / "run_all.sh").exists()


def test_hparam_outer_workspace_owns_base_finetune_metadata(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    for field in ("id", "title", "objective", "root", "baseline"):
        base_payload["experiment"][field] = "ASK_USER"
    for field in ("id", "phase", "purpose"):
        base_payload["step"][field] = "ASK_USER"
    base_recipe.write_text(yaml.safe_dump(base_payload))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stdout
    assert "base_finetune.experiment" not in result.stdout


def test_single_finetune_freezes_runtime_and_checkpoint_paths(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr
    run = _first_run(output_dir)
    expected_runtime = REPO_ROOT / "log-finetune" / run["version"]
    assert run["runtime_dir"] == str(expected_runtime)
    assert run["checkpoint_dir"] == str(expected_runtime / "checkpoints")
    assert json.loads(Path(run["artifacts"]).read_text())["runtime_dir"] == str(expected_runtime)
    script = Path(run["script"])
    script_text = script.read_text()
    assert f"cd {shlex_quote(str(REPO_ROOT))}" in script_text
    assert f"export PYTHONPATH={shlex_quote(str(REPO_ROOT))}${{PYTHONPATH:+:$PYTHONPATH}}" in script_text
    frozen_hash = file_sha256(script)
    assert run["script_sha256"] == frozen_hash
    assert json.loads((Path(run["run_dir"]) / "run.json").read_text())["script_sha256"] == frozen_hash
    with (tmp_path / "run_manifest.tsv").open(newline="") as file_obj:
        manifest = next(csv.DictReader(file_obj, delimiter="\t"))
    assert manifest["script_sha256"] == frozen_hash


def test_hparam_workdir_is_verbatim_run_cwd_for_all_generated_scripts(tmp_path: Path):
    checkpoint = tmp_path / "selected.ckpt"
    checkpoint.write_text("checkpoint")
    recipe = _hparam_recipe(tmp_path, ckpt_path=checkpoint)
    payload = yaml.safe_load(recipe.read_text())
    run_cwd = tmp_path / "runtime cwd"
    payload["execution"] = {"workdir": str(run_cwd)}
    recipe.write_text(yaml.safe_dump(payload))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 0, result.stderr
    run = _first_run(output_dir)
    expected_runtime = run_cwd / "log-finetune" / run["version"]
    assert run["runtime_dir"] == str(expected_runtime)
    expected_cwd = f"cd {shlex_quote(str(run_cwd))}"
    expected_pythonpath = f"export PYTHONPATH={shlex_quote(str(run_cwd))}${{PYTHONPATH:+:$PYTHONPATH}}"
    for script in (Path(run["script"]), output_dir / "run_all.sh", output_dir / "final_external_test.sh"):
        text = script.read_text()
        assert expected_cwd in text
        assert expected_pythonpath in text


def test_hparam_relative_workdir_fails_before_workspace_creation(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = {"workdir": "relative/runtime"}
    recipe.write_text(yaml.safe_dump(payload))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "execution.workdir must be an absolute path" in result.stdout
    assert not output_dir.exists()


def test_collect_runs_only_reads_managed_manifest_paths(tmp_path: Path):
    runtime_dir = tmp_path / "runtime" / "run-000"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": "runtime-version",
                "config_path": "/runtime/config.yaml",
                "status": "finished",
                "command": "runtime command",
                "epoch": 4,
                "metrics": {"score": 0.8},
            }
        )
    )
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tconfig\tcommand\truntime_dir\tstatus\n"
        f"unit\ttrain\trun-000\tmanaged-version\t/managed/config.yaml\tmanaged command\t{runtime_dir}\tfailed\n"
    )
    historical = tmp_path / "historical"
    historical.mkdir()
    (historical / "run_manifest.json").write_text(json.dumps({"version": "historical-version"}))
    (historical / "trial_status.tsv").write_text("trial_id\tstatus\ntrial_000\tfinished\n")
    output = tmp_path / "collected.csv"

    collect_runs(tmp_path, "score", output)

    with output.open(newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-000"
    assert rows[0]["version"] == "managed-version"
    assert rows[0]["config"] == "/managed/config.yaml"
    assert rows[0]["command"] == "managed command"
    assert rows[0]["status"] == "failed"
    assert rows[0]["epoch"] == "4"
    assert rows[0]["score"] == "0.8"
    assert "runtime-version" not in output.read_text()
    assert "historical-version" not in output.read_text()


def test_collect_runs_rejects_invalid_runtime_manifest_without_overwriting_output(tmp_path: Path):
    runtime_dir = tmp_path / "runtime" / "run-000"
    runtime_dir.mkdir(parents=True)
    manifest = runtime_dir / "run_manifest.json"
    manifest.write_text("{")
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\truntime_dir\tstatus\n" f"unit\ttrain\trun-000\t{runtime_dir}\tfailed\n"
    )
    output = tmp_path / "collected.csv"
    output.write_bytes(b"existing output\n")

    with pytest.raises(ValueError, match="run manifest"):
        collect_runs(tmp_path, "score", output)

    assert output.read_bytes() == b"existing output\n"


def test_collect_runs_rejects_missing_canonical_manifest_without_overwriting_output(tmp_path: Path):
    output = tmp_path / "collected.csv"
    output.write_bytes(b"existing output\n")

    try:
        collect_runs(tmp_path, None, output)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("collect_runs must require run_manifest.tsv")

    assert output.read_bytes() == b"existing output\n"


def test_collect_runs_rejects_canonical_manifest_output_alias_without_writing(tmp_path: Path):
    manifest = tmp_path / "run_manifest.tsv"
    original = b"step_id\trun_id\n"
    manifest.write_bytes(original)

    with pytest.raises(ValueError, match="cannot overwrite canonical run_manifest.tsv"):
        collect_runs(tmp_path, None, manifest)

    assert manifest.read_bytes() == original


def test_collect_runs_rejects_unsafe_output_topology_without_writing(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = workspace / "run_manifest.tsv"
    original_manifest = b"step_id\trun_id\n"
    manifest.write_bytes(original_manifest)
    sentinel = tmp_path / "sentinel.csv"
    sentinel.write_bytes(b"keep me\n")
    output = workspace / "collected.csv"
    output.hardlink_to(sentinel)

    with pytest.raises(ValueError, match="Managed output paths"):
        collect_runs(workspace, None, output)

    assert manifest.read_bytes() == original_manifest
    assert sentinel.read_bytes() == b"keep me\n"


def test_collect_runs_allows_header_only_canonical_manifest(tmp_path: Path):
    (tmp_path / "run_manifest.tsv").write_text("step_id\trun_id\n")
    output = tmp_path / "collected.csv"

    collect_runs(tmp_path, None, output)

    assert output.read_text() == "version\n"


def test_hparam_run_all_rejects_tampered_leaf_before_execution(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    output_dir = tmp_path / "plan"
    marker = tmp_path / "ran.txt"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))
    assert result.returncode == 0
    run_all_text = (output_dir / "run_all.sh").read_text()
    assert (
        f"python -m agent_tools hparam-launch --plan-dir {shlex_quote(str(output_dir.resolve()))} --execute"
        in run_all_text
    )
    Path(_first_run(output_dir)["script"]).write_text(
        f"#!/usr/bin/env bash\nset -euo pipefail\nprintf ok > {shlex_quote(str(marker))}\n"
    )

    run_all = subprocess.run(["bash", str(output_dir / "run_all.sh")], cwd=tmp_path, text=True, capture_output=True)

    assert run_all.returncode != 0
    assert "snapshot hash changed" in run_all.stderr
    assert not marker.exists()


@pytest.mark.parametrize("task", ["finetune", "infer", "evaluate", "preset_prepare", "sleep2stat"])
@pytest.mark.parametrize("cwd_kind", ["plan", "outside"])
def test_non_hparam_run_script_commits_lifecycle_from_any_cwd(
    tmp_path: Path,
    monkeypatch,
    task: str,
    cwd_kind: str,
):
    source = tmp_path / "source"
    recipe_path = write_finetune_recipe(source)
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = tmp_path / "workspace"
    recipe["task"] = task
    recipe["name"] = f"unit_{task}"
    recipe["experiment"]["root"] = str(workspace)
    recipe["step"] = {
        "id": f"unit-{task.replace('_', '-')}",
        "phase": "train" if task == "finetune" else "analyze",
        "purpose": "Exercise managed non-hparam lifecycle.",
    }
    recipe["decisions"]["task"] = {"value": task, "source": "explicit_recipe"}
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS, issues=[], decisions={})
    monkeypatch.setattr(plans, "preflight_plan", lambda **_kwargs: (recipe, None, report))
    marker = tmp_path / "runtime.txt"
    runtime_code = (
        "import sys; from pathlib import Path; "
        "from agent_tools.experiment_workspace import read_run_manifest; "
        "rows = read_run_manifest(sys.argv[1]); "
        "Path(sys.argv[2]).write_text(rows[0]['status'] + '\\n' + str(Path.cwd()))"
    )
    command = " ".join(shlex_quote(str(value)) for value in (sys.executable, "-c", runtime_code, workspace, marker))
    monkeypatch.setattr(plans, "_commands_for_recipe", lambda *_args, **_kwargs: [command])
    plan_dir = workspace / "plan"

    assert plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir).exit_code == 0
    outside = tmp_path / "outside"
    outside.mkdir()
    cwd = plan_dir if cwd_kind == "plan" else outside
    result = subprocess.run(["bash", str(plan_dir / "run.sh")], cwd=cwd, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert marker.read_text().splitlines() == ["running", str(REPO_ROOT)]
    assert read_run_manifest(workspace)[0]["status"] == "completed"
    script = (plan_dir / "run.sh").read_text()
    assert f"cd {shlex_quote(str(REPO_ROOT))}" in script
    assert f"export PYTHONPATH={shlex_quote(str(REPO_ROOT))}${{PYTHONPATH:+:$PYTHONPATH}}" in script
    assert f"  {shlex_quote(sys.executable)} -c " in script
    final_report = tmp_path / "final.md"
    final_report.write_text("# Final\n\nManaged run completed.\n")
    assert experiments.finalize_experiment(workspace, final_report) == workspace / "reports" / "final.md"


def test_non_hparam_run_script_records_failure_and_preserves_runtime_exit_code(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    recipe_path = write_finetune_recipe(source)
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = tmp_path / "workspace"
    recipe["experiment"]["root"] = str(workspace)
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS, issues=[], decisions={})
    monkeypatch.setattr(plans, "preflight_plan", lambda **_kwargs: (recipe, None, report))
    command = " ".join(shlex_quote(str(value)) for value in (sys.executable, "-c", "import sys; sys.exit(7)"))
    monkeypatch.setattr(plans, "_commands_for_recipe", lambda *_args, **_kwargs: [command])
    plan_dir = workspace / "plan"
    plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)

    result = subprocess.run(["bash", str(plan_dir / "run.sh")], cwd=plan_dir, text=True, capture_output=True)

    assert result.returncode == 7
    assert read_run_manifest(workspace)[0]["status"] == "failed"


def test_non_hparam_run_script_propagates_terminal_commit_failure(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    recipe_path = write_finetune_recipe(source)
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = tmp_path / "workspace"
    recipe["experiment"]["root"] = str(workspace)
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS, issues=[], decisions={})
    monkeypatch.setattr(plans, "preflight_plan", lambda **_kwargs: (recipe, None, report))
    runtime_code = "import sys; from pathlib import Path; (Path(sys.argv[1]) / 'run_manifest.tsv').unlink()"
    command = " ".join(shlex_quote(str(value)) for value in (sys.executable, "-c", runtime_code, workspace))
    monkeypatch.setattr(plans, "_commands_for_recipe", lambda *_args, **_kwargs: [command])
    plan_dir = workspace / "plan"
    plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)

    result = subprocess.run(["bash", str(plan_dir / "run.sh")], cwd=plan_dir, text=True, capture_output=True)

    assert result.returncode != 0
    assert "run_manifest.tsv" in result.stderr


def test_non_hparam_run_script_refuses_to_execute_terminal_run(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    recipe_path = write_finetune_recipe(source)
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = tmp_path / "workspace"
    recipe["experiment"]["root"] = str(workspace)
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS, issues=[], decisions={})
    monkeypatch.setattr(plans, "preflight_plan", lambda **_kwargs: (recipe, None, report))
    marker = tmp_path / "runtime.txt"
    command = " ".join(
        shlex_quote(str(value))
        for value in (sys.executable, "-c", "import sys; from pathlib import Path; Path(sys.argv[1]).touch()", marker)
    )
    monkeypatch.setattr(plans, "_commands_for_recipe", lambda *_args, **_kwargs: [command])
    plan_dir = workspace / "plan"
    plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)
    run = _first_run(plan_dir)
    merge_run_manifest(
        workspace,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "completed"}],
    )

    result = subprocess.run(["bash", str(plan_dir / "run.sh")], cwd=plan_dir, text=True, capture_output=True)

    assert result.returncode != 0
    assert not marker.exists()
    assert read_run_manifest(workspace)[0]["status"] == "completed"


def test_hparam_yaml_parameter_updates_run_config(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, parameters={"yaml:/finetune/task/output_dim": [31]})
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    run_config = yaml.safe_load(Path(_first_run(output_dir)["config"]).read_text())
    assert run_config["finetune"]["task"]["output_dim"] == 31


def test_hparam_yaml_parameter_rejects_negative_list_index(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, parameters={"yaml:/model/channels/-1/input_dim": [9]})
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert not (output_dir / "run_all.sh").exists()


def test_importing_decisions_does_not_import_torch_or_lightning(monkeypatch):
    sys.modules.pop("torch", None)
    sys.modules.pop("pytorch_lightning", None)

    import agent_tools.decisions  # noqa: F401

    assert "torch" not in sys.modules
    assert "pytorch_lightning" not in sys.modules


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)
