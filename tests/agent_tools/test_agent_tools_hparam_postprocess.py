from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pandas as pd
import pytest
import yaml

from agent_tools import cli, hparam_postprocess
from agent_tools.experiment_workspace import merge_run_manifest
from agent_tools.models import REPO_ROOT


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _hparam_recipe(tmp_path: Path, *, execution: dict | None = None, run_count: int = 1) -> Path:
    base = write_finetune_recipe(tmp_path)
    return write_yaml(
        tmp_path / "tune.yaml",
        {
            "name": "unit_hparam",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base),
            "search": {
                "method": "grid",
                "max_runs": run_count,
                "parameters": {"runtime.lr": [1e-6 * (index + 1) for index in range(run_count)]},
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


def _first_run(plan_dir: Path) -> dict:
    return json.loads((plan_dir / "plan.json").read_text())["runs"][0]


def _ranking_path(plan_dir: Path) -> Path:
    recipe = json.loads((plan_dir / "plan.json").read_text())["recipe"]
    return Path(recipe["experiment"]["root"]) / "reports" / "ranking.csv"


def test_hparam_external_eval_uses_run_runtime_from_candidate_ranking(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["parameters"] = {"runtime.batch_size": [48]}
    payload["execution"] = {"workdir": str(tmp_path)}
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"]["batch_size"] = 32
    write_yaml(base_recipe, base_payload)
    write_yaml(recipe, payload)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    version = run["version"]
    run_dir = Path(run["runtime_dir"])
    ckpt_dir = Path(run["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True)
    fixed = ckpt_dir / "epoch=11.ckpt"
    fixed.write_text("fixed")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": version,
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
    rows = _read_table(_ranking_path(plan_dir))
    assert rows[0]["runtime.batch_size"] == "48"
    unlocked = _run(
        "hparam-external-eval",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(_ranking_path(plan_dir)),
        "--unlock-final-test",
    )

    assert unlocked.returncode == 0, unlocked.stderr
    external_script = (plan_dir / "external_eval.sh").read_text()
    assert "--batch-size 48" in external_script


def test_hparam_external_eval_uses_frozen_plan_fields_and_workdir(tmp_path: Path):
    run_cwd = tmp_path / "runtime cwd"
    recipe = _hparam_recipe(tmp_path, execution={"workdir": str(run_cwd)})
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["parameters"] = {"runtime.batch_size": [48]}
    write_yaml(recipe, payload)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    selected = plan_dir / "selected.csv"
    with selected.open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "step_id",
                "run_id",
                "rank",
                "version",
                "config",
                "runtime_dir",
                "runtime.batch_size",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "rank": 1,
                "version": run["version"],
                "config": run["config"],
                "runtime_dir": run["runtime_dir"],
                "runtime.batch_size": 48,
                "checkpoint_path": Path(run["checkpoint_dir"]) / "epoch=1.ckpt",
            }
        )

    result = _run(
        "hparam-external-eval",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(selected),
        "--unlock-final-test",
    )

    assert result.returncode == 0, result.stderr
    row = _read_table(plan_dir / "external_eval_manifest.tsv")[0]
    assert row["version"] == run["version"]
    assert row["config"] == run["config"]
    assert row["runtime_dir"] == run["runtime_dir"]
    script = (plan_dir / "external_eval.sh").read_text()
    assert "--batch-size 48" in script
    assert "--batch-size 999" not in script
    assert f"cd {shlex.quote(str(run_cwd))}" in script
    assert f"export PYTHONPATH={shlex.quote(str(run_cwd))}${{PYTHONPATH:+:$PYTHONPATH}}" in script


@pytest.mark.parametrize("snapshot_field", ["config", "script"])
def test_hparam_external_eval_rejects_changed_snapshot_before_writing(tmp_path: Path, snapshot_field: str):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "step_id,run_id,rank,checkpoint_path\n"
        f"{run['step_id']},{run['run_id']},1,{Path(run['checkpoint_dir']) / 'epoch=1.ckpt'}\n"
    )
    snapshot_path = Path(run[snapshot_field])
    snapshot_path.write_bytes(snapshot_path.read_bytes() + b"\n# changed after planning\n")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="snapshot hash changed"):
        hparam_postprocess.generate_external_eval(
            plan_dir,
            selected,
            unlock_final_test=True,
        )

    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_hparam_external_eval_rejects_base_recipe_drift_before_writing(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, execution={"workdir": str(tmp_path)})
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint")
    selected = plan_dir / "selected.csv"
    selected.write_text("step_id,run_id,rank,checkpoint_path\n" f"{run['step_id']},{run['run_id']},1,{checkpoint}\n")
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["_base_recipe"]["inputs"]["label_name"] = "drifted-label"
    plan_path.write_text(json.dumps(plan))
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="recipe.resolved.yaml"):
        hparam_postprocess.generate_external_eval(plan_dir, selected, unlock_final_test=True)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize(
    ("candidate_parameters", "error"),
    [
        ({"runtime.batch_size": 999}, "parameter differs"),
        ({"runtime.batch_size": 48, "runtime.lr": 1e-6}, "parameters outside"),
        ({"runtime.batch_size": 48, "param.runtime.lr": 1e-6}, "Historical parameter fields"),
    ],
)
def test_hparam_external_eval_rejects_candidate_parameter_sources_before_writing(
    tmp_path: Path, candidate_parameters: dict[str, object], error: str
):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["parameters"] = {"runtime.batch_size": [48]}
    write_yaml(recipe, payload)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    selected = plan_dir / "selected.csv"
    row = {
        "step_id": run["step_id"],
        "run_id": run["run_id"],
        "rank": 1,
        "checkpoint_path": Path(run["checkpoint_dir"]) / "epoch=1.ckpt",
        **candidate_parameters,
    }
    with selected.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    result = _run(
        "hparam-external-eval",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(selected),
        "--unlock-final-test",
    )

    assert result.returncode == 1
    assert error in result.stderr
    assert not (plan_dir / "external_eval_configs").exists()
    assert not (plan_dir / "external_eval.sh").exists()


def test_hparam_external_eval_filters_workspace_ranking_to_current_step(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    other_checkpoint_dir = tmp_path / "other-checkpoints"
    merge_run_manifest(
        tmp_path,
        [
            {
                "experiment_id": run["experiment_id"],
                "step_id": "other-step",
                "run_id": "run-000",
                "config": run["config"],
                "checkpoint_dir": str(other_checkpoint_dir),
                "status": "completed",
            }
        ],
    )
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    ranking = _ranking_path(plan_dir)
    ranking.parent.mkdir(parents=True, exist_ok=True)
    with ranking.open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["step_id", "run_id", "rank", "config", "checkpoint_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "step_id": "other-step",
                "run_id": "run-000",
                "rank": 1,
                "config": run["config"],
                "checkpoint_path": other_checkpoint_dir / "epoch=1.ckpt",
            }
        )
        writer.writerow(
            {
                "step_id": "unit-hparam-tune",
                "run_id": run["run_id"],
                "rank": 1,
                "config": run["config"],
                "checkpoint_path": checkpoint,
            }
        )

    result = _run(
        "hparam-external-eval",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(ranking),
        "--unlock-final-test",
    )

    assert result.returncode == 0, result.stderr
    rows = _read_table(plan_dir / "external_eval_manifest.tsv")
    assert len(rows) == 1
    assert rows[0]["step_id"] == "unit-hparam-tune"


def test_postprocess_relative_plan_dir_persists_absolute_management_paths(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint")
    val_predictions = tmp_path / "val_predictions.csv"
    test_predictions = tmp_path / "test_predictions.csv"
    pd.DataFrame({"label": [0, 1], "prob": [0.1, 0.9]}).to_csv(val_predictions, index=False)
    pd.DataFrame({"label": [0, 1], "prob": [0.2, 0.8]}).to_csv(test_predictions, index=False)
    selected = tmp_path / "selected.csv"
    with selected.open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "experiment_id",
                "step_id",
                "run_id",
                "rank",
                "config",
                "checkpoint_path",
                "val_predictions_path",
                "test_predictions_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": run["experiment_id"],
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "rank": 1,
                "config": run["config"],
                "checkpoint_path": checkpoint,
                "val_predictions_path": val_predictions,
                "test_predictions_path": test_predictions,
            }
        )
    monkeypatch.chdir(tmp_path)

    external = hparam_postprocess.generate_external_eval("plan", selected, unlock_final_test=True)
    logits = hparam_postprocess.export_hparam_logits("plan", selected, unlock_final_test=True)
    threshold = hparam_postprocess.threshold_hparam_outputs("plan", selected)
    ensemble = hparam_postprocess.ensemble_hparam_outputs("plan", selected)

    assert all(path.is_absolute() for path in (external, logits, threshold, ensemble))
    external_row = _read_table(plan_dir / "external_eval_manifest.tsv")[0]
    logits_row = _read_table(plan_dir / "logits_export_manifest.tsv")[0]
    assert Path(external_row["external_config"]).is_absolute()
    for field in ("val_config", "val_logits_path", "test_config", "test_logits_path"):
        assert Path(logits_row[field]).is_absolute()
    logits_script = plan_dir / "logits_export.sh"
    assert logits_script.exists()
    replay_command = shlex.split(logits_script.read_text().splitlines()[-1])
    assert replay_command[replay_command.index("--run-dir") + 1] == str(plan_dir.resolve())
    assert replay_command[replay_command.index("--selected") + 1] == str(selected.resolve())


@pytest.mark.parametrize(
    "mutation",
    ["external_ancestor", "logits_ancestor", "logits_script", "threshold_leaf", "ensemble_leaf"],
)
def test_hparam_postprocess_preflights_outputs_before_side_effects(tmp_path: Path, monkeypatch, mutation: str):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("checkpoint")
    val_predictions = tmp_path / "val.csv"
    test_predictions = tmp_path / "test.csv"
    pd.DataFrame({"label": [0, 1], "prob": [0.1, 0.9]}).to_csv(val_predictions, index=False)
    pd.DataFrame({"label": [0, 1], "prob": [0.2, 0.8]}).to_csv(test_predictions, index=False)
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "step_id,run_id,rank,checkpoint_path,val_predictions_path,test_predictions_path\n"
        f"{run['step_id']},{run['run_id']},1,{checkpoint},{val_predictions},{test_predictions}\n"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = plan_dir / "plan.json"
    if mutation == "external_ancestor":
        (plan_dir / "external_eval_configs").symlink_to(outside, target_is_directory=True)
    elif mutation == "logits_ancestor":
        (plan_dir / "logits_exports").symlink_to(outside, target_is_directory=True)
    elif mutation == "logits_script":
        (plan_dir / "logits_export.sh").symlink_to(protected)
    elif mutation == "threshold_leaf":
        (plan_dir / "threshold_summary.csv").symlink_to(protected)
    else:
        (plan_dir / "ensemble_summary.csv").symlink_to(protected)
    inference_calls = []
    monkeypatch.setattr(
        hparam_postprocess,
        "_run_logit_export",
        lambda *_args, **_kwargs: inference_calls.append(_kwargs),
    )
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file() and not path.is_symlink()
    }

    with pytest.raises(ValueError, match="Managed output"):
        if mutation == "external_ancestor":
            hparam_postprocess.generate_external_eval(plan_dir, selected, unlock_final_test=True)
        elif mutation in {"logits_ancestor", "logits_script"}:
            hparam_postprocess.export_hparam_logits(
                plan_dir,
                selected,
                unlock_final_test=True,
                execute=mutation == "logits_ancestor",
            )
        elif mutation == "threshold_leaf":
            hparam_postprocess.threshold_hparam_outputs(plan_dir, selected)
        else:
            hparam_postprocess.ensemble_hparam_outputs(plan_dir, selected)

    assert inference_calls == []
    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file() and not path.is_symlink()
    } == before


def test_hparam_external_eval_rejects_checkpoint_outside_frozen_directory_before_writing(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    foreign_checkpoint = tmp_path / "foreign" / "epoch=1.ckpt"
    foreign_checkpoint.parent.mkdir()
    foreign_checkpoint.write_text("foreign")
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "step_id,run_id,rank,checkpoint_path\n" f"{run['step_id']},{run['run_id']},1,{foreign_checkpoint}\n"
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="checkpoint_path is outside the frozen checkpoint_dir"):
        hparam_postprocess.generate_external_eval(plan_dir, selected, unlock_final_test=True)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_external_eval_rejects_checkpoint_symlink_inside_frozen_directory_before_writing(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, execution={"workdir": str(tmp_path)})
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    foreign = tmp_path / "foreign.ckpt"
    foreign.write_text("foreign")
    checkpoint = Path(run["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() or checkpoint.is_symlink():
        checkpoint.unlink()
    checkpoint.symlink_to(foreign)
    selected = plan_dir / "selected.csv"
    selected.write_text("step_id,run_id,rank,checkpoint_path\n" f"{run['step_id']},{run['run_id']},1,{checkpoint}\n")
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file() and not path.is_symlink()
    }

    with pytest.raises(ValueError, match="checkpoint_path is not a regular managed checkpoint"):
        hparam_postprocess.generate_external_eval(plan_dir, selected, unlock_final_test=True)

    assert {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file() and not path.is_symlink()
    } == before


def test_hparam_external_eval_validates_other_step_rows_before_filtering(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    ranking = _ranking_path(plan_dir)
    ranking.parent.mkdir(parents=True, exist_ok=True)
    ranking.write_text(
        "step_id,run_id,rank,config,checkpoint_path\n"
        f"other-step,,1,{run['config']},{tmp_path / 'epoch=1.ckpt'}\n"
        f"{run['step_id']},{run['run_id']},1,{run['config']},{tmp_path / 'epoch=1.ckpt'}\n"
    )

    result = _run(
        "hparam-external-eval",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(ranking),
        "--unlock-final-test",
    )

    assert result.returncode == 1
    assert "must define step_id and run_id" in result.stderr
    assert not (plan_dir / "external_eval_configs").exists()
    assert not (plan_dir / "external_eval.sh").exists()


def test_selected_candidates_reject_legacy_other_step_before_filtering():
    plan = {"runs": [{"step_id": "current-step", "run_id": "run-000"}]}
    rows = [
        {"step_id": "other-step", "run_id": "run-000", "trial_id": "trial_000"},
        {"step_id": "current-step", "run_id": "run-000"},
    ]

    with pytest.raises(ValueError, match="Historical trial_id"):
        hparam_postprocess._selected_candidate_rows(rows, plan=plan)


def test_hparam_external_eval_rejects_header_only_removed_candidate_table(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    selected = plan_dir / "selected.csv"
    selected.write_text("trial_id\n")
    before = {path.relative_to(plan_dir): path.read_bytes() for path in plan_dir.rglob("*") if path.is_file()}

    result = _run(
        "hparam-external-eval",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(selected),
        "--unlock-final-test",
    )

    assert result.returncode == 1
    assert "Historical managed table fields" in result.stderr
    assert {path.relative_to(plan_dir): path.read_bytes() for path in plan_dir.rglob("*") if path.is_file()} == before


def test_selected_candidates_reject_foreign_experiment_before_plan_fields_replace_it(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    run = plan["runs"][0]
    rows = [
        {
            "experiment_id": "other",
            "step_id": run["step_id"],
            "run_id": run["run_id"],
            "version": run["version"],
            "config": run["config"],
            "rank": "1",
        }
    ]

    with pytest.raises(ValueError, match="Frozen run field differs.*experiment_id"):
        hparam_postprocess._selected_candidate_rows(rows, plan=plan)


def test_selected_candidates_reject_foreign_other_step_before_filtering(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    run = plan["runs"][0]
    merge_run_manifest(
        tmp_path,
        [
            {
                "experiment_id": run["experiment_id"],
                "step_id": "other-step",
                "run_id": "run-000",
                "status": "completed",
            }
        ],
    )
    rows = [
        {"experiment_id": "other", "step_id": "other-step", "run_id": "run-000"},
        {"experiment_id": run["experiment_id"], "step_id": run["step_id"], "run_id": run["run_id"]},
    ]

    with pytest.raises(ValueError, match="Frozen run field differs.*experiment_id"):
        hparam_postprocess._selected_candidate_rows(rows, plan=plan)


def test_selected_candidates_filters_managed_same_step_rows_from_previous_plans(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    first_plan_dir = tmp_path / "plan-1"
    second_plan_dir = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(first_plan_dir)).returncode == 0
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(second_plan_dir)).returncode == 0
    first_run = _first_run(first_plan_dir)
    second_plan = json.loads((second_plan_dir / "plan.json").read_text())
    second_run = second_plan["runs"][0]

    rows = [
        {"step_id": first_run["step_id"], "run_id": first_run["run_id"], "rank": "1"},
        {"step_id": second_run["step_id"], "run_id": second_run["run_id"], "rank": "2"},
    ]
    selected = hparam_postprocess._selected_candidate_rows(
        rows,
        plan=second_plan,
        all_candidates=True,
    )
    top = hparam_postprocess._selected_candidate_rows(rows, plan=second_plan)

    assert [row["run_id"] for row in selected] == [second_run["run_id"]]
    assert [row["run_id"] for row in top] == [second_run["run_id"]]


def test_selected_candidates_ranks_current_plan_rows_after_filtering_previous_plan(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, run_count=2)
    first_plan_dir = tmp_path / "plan-1"
    second_plan_dir = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(first_plan_dir)).returncode == 0
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(second_plan_dir)).returncode == 0
    first_run = _first_run(first_plan_dir)
    second_plan = json.loads((second_plan_dir / "plan.json").read_text())
    better, worse = second_plan["runs"]

    selected = hparam_postprocess._selected_candidate_rows(
        [
            {"step_id": first_run["step_id"], "run_id": first_run["run_id"], "rank": "1"},
            {"step_id": worse["step_id"], "run_id": worse["run_id"], "rank": "4"},
            {"step_id": better["step_id"], "run_id": better["run_id"], "rank": "3"},
        ],
        plan=second_plan,
        top_k=1,
    )

    assert [row["run_id"] for row in selected] == [better["run_id"]]


@pytest.mark.parametrize("rank", [None, "", 0, -1, 1.5, "nan", "invalid", True])
def test_selected_candidates_require_positive_integer_rank(tmp_path: Path, rank):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    run = plan["runs"][0]

    with pytest.raises(ValueError, match="rank must be a positive integer"):
        hparam_postprocess._selected_candidate_rows(
            [{"step_id": run["step_id"], "run_id": run["run_id"], "rank": rank}],
            plan=plan,
        )


def test_selected_candidates_enforce_top_k_as_hard_limit(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, run_count=2)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    rows = [{"step_id": run["step_id"], "run_id": run["run_id"], "rank": "1"} for run in plan["runs"]]

    selected = hparam_postprocess._selected_candidate_rows(rows, plan=plan, top_k=1)

    assert [row["run_id"] for row in selected] == [plan["runs"][0]["run_id"]]


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_selected_candidates_require_positive_integer_top_k(tmp_path: Path, top_k):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    run = plan["runs"][0]

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        hparam_postprocess._selected_candidate_rows(
            [{"step_id": run["step_id"], "run_id": run["run_id"], "rank": 1}],
            plan=plan,
            top_k=top_k,
        )


def test_hparam_external_eval_rejects_workspace_ranking_without_current_step(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    other_checkpoint_dir = tmp_path / "other-checkpoints"
    merge_run_manifest(
        tmp_path,
        [
            {
                "experiment_id": run["experiment_id"],
                "step_id": "other-step",
                "run_id": "run-000",
                "config": run["config"],
                "checkpoint_dir": str(other_checkpoint_dir),
                "status": "completed",
            }
        ],
    )
    ranking = _ranking_path(plan_dir)
    ranking.parent.mkdir(parents=True, exist_ok=True)
    ranking.write_text(
        "step_id,run_id,rank,config,checkpoint_path\n"
        f"other-step,run-000,1,{run['config']},{other_checkpoint_dir / 'epoch=1.ckpt'}\n"
    )

    result = _run(
        "hparam-external-eval",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(ranking),
        "--unlock-final-test",
    )

    assert result.returncode == 1
    assert "No selected candidates match the current hparam step" in result.stderr
    assert not (plan_dir / "external_eval.sh").exists()


def test_hparam_external_eval_requires_unlock_and_only_replaces_data_fields(
    tmp_path: Path,
):
    recipe = _hparam_recipe(tmp_path, run_count=3)
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
    base_config = Path(base_payload["inputs"]["config"])
    base_config_payload = yaml.safe_load(base_config.read_text())
    stale_preset = tmp_path / "stale_preset.pkl"
    stale_preset.write_bytes(b"fixture")
    base_config_payload["data"]["finetune_preset_path"] = str(stale_preset)
    write_yaml(base_config, base_config_payload)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    runs = json.loads((plan_dir / "plan.json").read_text())["runs"]
    run_config = Path(runs[0]["config"])
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "step_id,run_id,rank,config,checkpoint_path\n"
        f"unit-hparam-tune,run-000,1,{runs[0]['config']},{Path(runs[0]['checkpoint_dir']) / 'epoch=1.ckpt'}\n"
        f"unit-hparam-tune,run-001,2,{runs[1]['config']},{Path(runs[1]['checkpoint_dir']) / 'epoch=2.ckpt'}\n"
        f"unit-hparam-tune,run-002,3,{runs[2]['config']},{Path(runs[2]['checkpoint_dir']) / 'epoch=3.ckpt'}\n"
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
    original = yaml.safe_load(run_config.read_text())
    external = yaml.safe_load((plan_dir / "external_eval_configs" / "run-000_001_external.yaml").read_text())
    assert external["data"]["finetune_data_index"] == "external_index.csv"
    assert external["data"]["finetune_preset_path"] is None
    assert external["model"] == original["model"]
    external_script = (plan_dir / "external_eval.sh").read_text()
    assert f"cd {shlex.quote(str(REPO_ROOT))}" in external_script
    assert f"export PYTHONPATH={shlex.quote(str(REPO_ROOT))}${{PYTHONPATH:+:$PYTHONPATH}}" in external_script
    assert external_script.count("python -m sleep2vec.infer") == 1
    assert "--devices 6 7" in external_script
    assert "--accelerator cpu" in external_script
    assert "--device cpu" in external_script
    assert "--batch-size 32" in external_script
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
    kaldi_external = yaml.safe_load((plan_dir / "external_eval_configs" / "run-000_001_external.yaml").read_text())
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
    recipe = _hparam_recipe(tmp_path, run_count=2)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_config = Path(base_payload["inputs"]["config"])
    base_config_payload = yaml.safe_load(base_config.read_text())
    stale_preset = tmp_path / "stale_preset.pkl"
    stale_preset.write_bytes(b"fixture")
    base_config_payload["data"]["finetune_preset_path"] = str(stale_preset)
    write_yaml(base_config, base_config_payload)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    runs = json.loads((plan_dir / "plan.json").read_text())["runs"]
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "step_id,run_id,rank,config,checkpoint_path\n"
        f"unit-hparam-tune,run-000,1,{runs[0]['config']},{Path(runs[0]['checkpoint_dir']) / 'epoch=1.ckpt'}\n"
        f"unit-hparam-tune,run-001,2,{runs[1]['config']},{Path(runs[1]['checkpoint_dir']) / 'epoch=2.ckpt'}\n"
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
    assert rows[0]["val_logits_path"].endswith("logits_exports/run-000_001_val_logits.csv")
    assert rows[0]["test_logits_path"].endswith("logits_exports/run-000_001_test_logits.csv")
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


def test_hparam_export_logits_dry_run_script_freezes_absolute_inputs(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "step_id,run_id,rank,config,checkpoint_path\n"
        f"{run['step_id']},{run['run_id']},1,{run['config']},{Path(run['checkpoint_dir']) / 'epoch=1.ckpt'}\n"
    )
    monkeypatch.chdir(tmp_path)

    result = cli.main(
        [
            "hparam-export-logits",
            "--run-dir",
            "plan",
            "--selected",
            "plan/selected.csv",
            "--skip-test",
        ]
    )

    assert result == 0
    script = plan_dir / "logits_export.sh"
    command = shlex.split(script.read_text().splitlines()[-1])
    assert command[command.index("--run-dir") + 1] == str(plan_dir.resolve())
    assert command[command.index("--selected") + 1] == str(selected.resolve())

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\n" f"exec {shlex.quote(sys.executable)} -c 'import agent_tools'\n")
    fake_python.chmod(0o755)
    replay_cwd = tmp_path / "replay-cwd"
    replay_cwd.mkdir()
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env.pop("PYTHONPATH", None)

    replay = subprocess.run(["bash", str(script)], cwd=replay_cwd, env=env, text=True, capture_output=True)

    assert replay.returncode == 0, replay.stderr


def test_hparam_export_logits_uses_effective_recipe_label(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"] = {"label_name": "effective-label"}
    payload["decisions"]["label_name"]["value"] = "effective-label"
    write_yaml(recipe, payload)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "step_id,run_id,rank,config,checkpoint_path\n"
        f"{run['step_id']},{run['run_id']},1,{run['config']},{Path(run['checkpoint_dir']) / 'epoch=1.ckpt'}\n"
    )

    result = _run(
        "hparam-export-logits",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(selected),
        "--skip-test",
    )

    assert result.returncode == 0, result.stderr
    row = _read_table(plan_dir / "logits_export_manifest.tsv")[0]
    assert row["label_name"] == "effective-label"
    assert "--label-name effective-label" in row["val_infer_command"]


@pytest.mark.parametrize(
    ("command", "rank", "extra_args", "unexpected_paths"),
    [
        (
            "hparam-external-eval",
            1,
            ["--unlock-final-test", "--top-k", "0"],
            ["external_eval_configs", "external_eval_manifest.tsv", "external_eval.sh"],
        ),
        (
            "hparam-export-logits",
            2,
            ["--skip-test", "--top-k", "0"],
            ["logits_export_configs", "logits_exports", "logits_export_manifest.tsv", "logits_export.sh"],
        ),
    ],
)
def test_hparam_postprocess_rejects_nonpositive_top_k_before_writing(
    tmp_path: Path,
    command: str,
    rank: int,
    extra_args: list[str],
    unexpected_paths: list[str],
):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "step_id,run_id,rank,config,checkpoint_path\n"
        f"{run['step_id']},{run['run_id']},{rank},{run['config']},"
        f"{Path(run['checkpoint_dir']) / 'epoch=1.ckpt'}\n"
    )

    result = _run(command, "--run-dir", str(plan_dir), "--selected", str(selected), *extra_args)

    assert result.returncode == 1
    assert "top_k must be a positive integer" in result.stderr
    assert all(not (plan_dir / path).exists() for path in unexpected_paths)


def test_hparam_export_logits_rejects_workspace_ranking_without_current_step(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    other_checkpoint_dir = tmp_path / "other-checkpoints"
    merge_run_manifest(
        tmp_path,
        [
            {
                "experiment_id": run["experiment_id"],
                "step_id": "other-step",
                "run_id": "run-000",
                "config": run["config"],
                "checkpoint_dir": str(other_checkpoint_dir),
                "status": "completed",
            }
        ],
    )
    ranking = _ranking_path(plan_dir)
    ranking.parent.mkdir(parents=True, exist_ok=True)
    ranking.write_text(
        "step_id,run_id,rank,config,checkpoint_path\n"
        f"other-step,run-000,1,{run['config']},{other_checkpoint_dir / 'epoch=1.ckpt'}\n"
    )

    result = _run(
        "hparam-export-logits",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(ranking),
        "--unlock-final-test",
    )

    assert result.returncode == 1
    assert "No selected candidates match the current hparam step" in result.stderr
    assert not (plan_dir / "logits_export.sh").exists()


def test_hparam_postprocess_rejects_unknown_run_in_current_step(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    selected = plan_dir / "selected.csv"
    selected.write_text(
        "step_id,run_id,rank,checkpoint_path\n" f"unit-hparam-tune,run-999,1,{tmp_path / 'epoch=1.ckpt'}\n"
    )

    result = _run(
        "hparam-external-eval",
        "--run-dir",
        str(plan_dir),
        "--selected",
        str(selected),
        "--unlock-final-test",
    )

    assert result.returncode == 1
    assert "not managed by the current hparam plan" in result.stderr
    assert not (plan_dir / "external_eval.sh").exists()


def test_threshold_and_ensemble_require_managed_plan_before_writing(tmp_path: Path):
    run_dir = tmp_path / "not-a-plan"
    run_dir.mkdir()
    selected = run_dir / "selected.csv"
    selected.write_text("step_id,run_id\ntune,run-000\n")

    threshold = _run("hparam-threshold", "--run-dir", str(run_dir), "--selected", str(selected))
    ensemble = _run("hparam-ensemble", "--run-dir", str(run_dir), "--candidates", str(selected))

    assert threshold.returncode == 1
    assert ensemble.returncode == 1
    assert not (run_dir / "threshold_summary.csv").exists()
    assert not (run_dir / "ensemble_summary.csv").exists()


def test_hparam_export_logits_execute_uses_manifest_paths(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    selected = plan_dir / "selected.csv"
    run = _first_run(plan_dir)
    selected.write_text(
        "step_id,run_id,rank,config,checkpoint_path\n"
        f"unit-hparam-tune,run-000,1,{Path(run['config'])},{Path(run['checkpoint_dir']) / 'epoch=1.ckpt'}\n"
    )
    calls = []

    def _fake_run_logit_export(recipe, **kwargs):
        calls.append(kwargs)
        Path(kwargs["output_path"]).write_text("label,logit\n0,-1.0\n1,1.0\n")

    monkeypatch.setattr(hparam_postprocess, "_run_logit_export", _fake_run_logit_export)

    manifest = hparam_postprocess.export_hparam_logits(
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


def test_hparam_export_logits_does_not_commit_manifest_after_execution_failure(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    selected = plan_dir / "selected.csv"
    run = _first_run(plan_dir)
    selected.write_text(
        "step_id,run_id,rank,config,checkpoint_path\n"
        f"unit-hparam-tune,run-000,1,{Path(run['config'])},{Path(run['checkpoint_dir']) / 'epoch=1.ckpt'}\n"
    )
    calls = []

    def _fake_run_logit_export(recipe, **kwargs):
        calls.append(kwargs)
        if kwargs["eval_split"] == "test":
            raise RuntimeError("test export failed")
        Path(kwargs["output_path"]).write_text("label,logit\n0,-1.0\n1,1.0\n")

    monkeypatch.setattr(hparam_postprocess, "_run_logit_export", _fake_run_logit_export)

    with pytest.raises(RuntimeError, match="test export failed"):
        hparam_postprocess.export_hparam_logits(plan_dir, selected, unlock_final_test=True, execute=True)

    assert [call["eval_split"] for call in calls] == ["val", "test"]
    assert Path(calls[0]["output_path"]).exists()
    assert not (plan_dir / "logits_export_manifest.tsv").exists()


@pytest.mark.parametrize(
    ("header", "value"),
    [
        ("val_predictions_path", "val.csv"),
        ("test_predictions_path", "test.csv"),
    ],
)
def test_hparam_threshold_requires_validation_and_test_inputs(tmp_path: Path, header: str, value: str):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    selected = tmp_path / "selected.csv"
    selected.write_text(f"step_id,run_id,{header}\n" f"{run['step_id']},{run['run_id']},{tmp_path / value}\n")

    result = _run("hparam-threshold", "--run-dir", str(plan_dir), "--selected", str(selected))

    assert result.returncode == 1
    assert "must define validation and test predictions/logits" in result.stderr
    assert not (plan_dir / "threshold_summary.csv").exists()


@pytest.mark.parametrize("empty_split", ["val", "test"])
def test_hparam_threshold_rejects_prediction_files_without_samples(tmp_path: Path, empty_split: str):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = _first_run(plan_dir)
    val = tmp_path / "val.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame({"label": [], "prob": []}).to_csv(val, index=False)
    pd.DataFrame({"label": [0, 1], "prob": [0.1, 0.9]}).to_csv(test, index=False)
    if empty_split == "test":
        val, test = test, val
    selected = tmp_path / "selected.csv"
    selected.write_text(
        "step_id,run_id,val_predictions_path,test_predictions_path\n" f"{run['step_id']},{run['run_id']},{val},{test}\n"
    )

    result = _run("hparam-threshold", "--run-dir", str(plan_dir), "--selected", str(selected))

    assert result.returncode == 1
    assert "must contain samples" in result.stderr
    assert not (plan_dir / "threshold_summary.csv").exists()


def test_hparam_threshold_and_ensemble_compute_binary_metrics(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, run_count=3)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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
        "step_id,run_id,val_predictions_path,test_predictions_path\n"
        f"unit-hparam-tune,run-000,{val_a},{test_a}\n"
        f"unit-hparam-tune,run-001,{val_b},{test_b}\n"
        f"unit-hparam-tune,run-002,{val_c},{test_c}\n"
    )

    threshold = _run("hparam-threshold", "--run-dir", str(plan_dir), "--selected", str(selected))
    ensemble = _run("hparam-ensemble", "--run-dir", str(plan_dir), "--candidates", str(selected))

    assert threshold.returncode == 0, threshold.stderr
    assert ensemble.returncode == 0, ensemble.stderr
    threshold_rows = _read_table(plan_dir / "threshold_summary.csv")
    assert float(threshold_rows[0]["test_auroc"]) == 1.0
    assert "test_f1" in threshold_rows[0]
    ensemble_rows = _read_table(plan_dir / "ensemble_summary.csv")
    assert ensemble_rows[0]["n_models"] == "3"

    ensemble_search = _run(
        "hparam-ensemble",
        "--run-dir",
        str(plan_dir),
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
    search_rows = _read_table(plan_dir / "ensemble_summary.csv")
    assert len(search_rows) == 6
    assert search_rows[0]["rank"] == "1"
    assert any(row["n_models"] == "2" for row in search_rows)


def test_hparam_threshold_and_ensemble_read_repo_prediction_csv_lists(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, run_count=4)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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
        "step_id,run_id,label_name,val_predictions_path,test_predictions_path\n"
        f"unit-hparam-tune,run-000,,{val_seq},{test_seq}\n"
        f"unit-hparam-tune,run-001,,{val_ahi},{test_ahi}\n"
        f"unit-hparam-tune,run-002,,{val_logit},{test_logit}\n"
        f"unit-hparam-tune,run-003,custom_label,{val_custom},{test_custom}\n"
    )

    threshold = _run("hparam-threshold", "--run-dir", str(plan_dir), "--selected", str(selected))
    ensemble = _run("hparam-ensemble", "--run-dir", str(plan_dir), "--candidates", str(selected))

    assert threshold.returncode == 0, threshold.stderr
    threshold_rows = _read_table(plan_dir / "threshold_summary.csv")
    assert len(threshold_rows) == 4
    assert all(float(row["test_auroc"]) == 1.0 for row in threshold_rows)
    assert all(float(row["test_accuracy"]) == 1.0 for row in threshold_rows)
    assert ensemble.returncode == 0, ensemble.stderr
    ensemble_rows = _read_table(plan_dir / "ensemble_summary.csv")
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

    y, p = hparam_postprocess._average_binary_predictions(
        [
            hparam_postprocess._read_binary_predictions(first),
            hparam_postprocess._read_binary_predictions(second),
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

    hparam_postprocess._copy_logits_csv(prediction, output)

    copied = pd.read_csv(output)
    assert list(copied.columns) == ["path", "groundtruth", "prob"]
