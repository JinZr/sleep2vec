from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_tools.models import REPO_ROOT
from agent_tools.plans import build_plan

VARIANTS = ("sleep2vec", "sleep2vec2", "sleep2expert")
PRETRAIN_CONFIG = "configs/sleep2vec_dense_pretrain.yaml"
FINETUNE_CONFIG = "configs/ppg_ahi_finetune.yaml"


def _write_recipe(
    tmp_path: Path,
    *,
    variant: str = "sleep2vec",
    config: str = PRETRAIN_CONFIG,
    eval_split: str = "val",
) -> tuple[Path, Path, dict]:
    sample = tmp_path / "night.npz"
    sample.touch()
    index = tmp_path / "index.csv"
    index.write_text(f"path,split,duration\n{sample},{eval_split},60\n")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    experiment_root = tmp_path / "experiment"
    payload = {
        "name": "whole-night-extraction",
        "task": "embedding_extraction",
        "variant": variant,
        "experiment": {
            "id": "embedding-extraction",
            "title": "Embedding extraction",
            "objective": "Export whole-night embeddings.",
            "root": str(experiment_root),
            "baseline": "none",
        },
        "step": {"id": "extract", "phase": "evaluate", "purpose": "Export embeddings."},
        "inputs": {
            "config": config,
            "ckpt_path": str(checkpoint),
            "data_index": [str(index)],
            "eval_split": eval_split,
        },
        "extraction": {
            "channels": ["ppg"] if "ppg_" in config else ["heartbeat", "breath"],
            "embedding_kind": "both",
            "layer_index": -1,
            "max_source_tokens": 4095,
            "output_format": "npz",
            "sequence_mode": "whole-night",
        },
        "runtime": {"device": "cpu", "num_workers": 0},
        "artifacts": {"embedding_dir": str(tmp_path / "embeddings"), "overwrite": False},
        "evaluation_policy": {},
    }
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    return recipe, experiment_root / "plans" / "extract", payload


def _rewrite(recipe: Path, payload: dict) -> None:
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))


@pytest.mark.parametrize("variant", VARIANTS)
def test_plan_routes_each_variant_and_freezes_config(tmp_path: Path, variant: str):
    recipe, plan_dir, _payload = _write_recipe(tmp_path, variant=variant)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    command = plan["commands"][0]
    frozen_config = plan_dir / "runs" / "run-000--whole-night-extraction" / "config.yaml"
    assert command.startswith(f"python -m {variant}.extract_embeddings ")
    assert f"--config {frozen_config}" in command
    assert frozen_config.read_bytes() == (REPO_ROOT / PRETRAIN_CONFIG).read_bytes()
    assert "--sequence-mode whole-night" in command
    assert "--batch-size 1" in command
    assert "--data-backend npz" in command


def test_plan_rejects_sex_age_baseline_variant(tmp_path: Path):
    recipe, plan_dir, payload = _write_recipe(tmp_path, variant="sex_age_baseline")

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert not Path(payload["experiment"]["root"]).exists()


@pytest.mark.parametrize("config", [PRETRAIN_CONFIG, FINETUNE_CONFIG])
def test_plan_accepts_pretrain_and_finetune_configs(tmp_path: Path, config: str):
    recipe, plan_dir, _payload = _write_recipe(tmp_path, config=config)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0


def test_repo_relative_config_loads_outside_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    recipe, plan_dir, _payload = _write_recipe(tmp_path)
    monkeypatch.chdir(tmp_path)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0


def test_user_decisions_materialize_pretrain_config_and_channels(tmp_path: Path):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    payload["inputs"]["config"] = "ASK_USER"
    payload["extraction"]["channels"] = "ASK_USER"
    _rewrite(recipe, payload)
    decisions = tmp_path / "decisions.yaml"
    decisions.write_text(
        yaml.safe_dump(
            {
                "decisions": {
                    "config": {"value": PRETRAIN_CONFIG, "source": "explicit_user"},
                    "embedding_channels": {"value": ["heartbeat", "breath"], "source": "explicit_user"},
                }
            }
        )
    )

    report = build_plan(recipe_path=recipe, output_dir=plan_dir, user_decisions_path=decisions)

    assert report.exit_code == 0
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    assert resolved["inputs"]["config"] == PRETRAIN_CONFIG
    assert resolved["extraction"]["channels"] == ["heartbeat", "breath"]


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("inputs", "config"),
        ("inputs", "ckpt_path"),
        ("inputs", "data_index"),
        ("inputs", "eval_split"),
        ("extraction", "channels"),
        ("extraction", "embedding_kind"),
        ("extraction", "layer_index"),
        ("extraction", "max_source_tokens"),
        ("extraction", "output_format"),
        ("extraction", "sequence_mode"),
        ("runtime", "device"),
        ("runtime", "num_workers"),
        ("artifacts", "embedding_dir"),
        ("artifacts", "overwrite"),
    ],
)
def test_unresolved_required_choices_return_consultation_status(tmp_path: Path, section: str, field: str):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    payload[section][field] = "ASK_USER"
    _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 2
    assert not (plan_dir / "run.sh").exists()
    assert not (plan_dir / "plan.json").exists()


def test_test_split_requires_both_unlocks(tmp_path: Path):
    recipe, plan_dir, _payload = _write_recipe(tmp_path, eval_split="test")

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 2
    assert {issue.field for issue in report.blocking_issues()} >= {"external_test_locked", "final_eval_unlock"}


def test_test_split_plans_after_both_unlocks(tmp_path: Path):
    recipe, plan_dir, payload = _write_recipe(tmp_path, eval_split="test")
    payload["evaluation_policy"] = {"external_test_locked": False, "final_test_unlocked": True}
    _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("inputs", "data_backend", "npz"),
        ("inputs", "override_dataset_names", ["cohort"]),
        ("inputs", "kaldi_manifest", "manifest.json"),
        ("runtime", "batch_size", 1),
        ("extraction", "preset_path", "preset.pkl"),
        ("root", "execution", {"target": "local"}),
    ],
)
def test_removed_contract_fields_fail_before_workspace(tmp_path: Path, section: str, field: str, value: object):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    if section == "root":
        payload[field] = value
    else:
        payload[section][field] = value
    _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert not payload["experiment"]["root"] or not Path(payload["experiment"]["root"]).exists()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "No whole-night rows"),
        (["{sample},val,30", "{sample},val,30"], "Duplicate whole-night path"),
        (["{sample},val,45"], "aligned to 30-second tokens"),
        (["{sample},val,90"], "expected [1, 2]"),
        (["{missing},val,30"], "Whole-night NPZ path not found"),
    ],
)
def test_plan_rejects_invalid_whole_night_index(tmp_path: Path, rows: list[str], message: str):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    sample = tmp_path / "night.npz"
    index = Path(payload["inputs"]["data_index"][0])
    rendered = [row.format(sample=sample, missing=tmp_path / "missing.npz") for row in rows]
    index.write_text("path,split,duration\n" + "\n".join(rendered) + ("\n" if rendered else ""))
    if "expected" in message:
        payload["extraction"]["max_source_tokens"] = 2
        _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any(message in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


def test_nonempty_embedding_output_fails_before_workspace(tmp_path: Path):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    embedding_dir = Path(payload["artifacts"]["embedding_dir"])
    embedding_dir.mkdir()
    (embedding_dir / "old.npz").touch()

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert not Path(payload["experiment"]["root"]).exists()


@pytest.mark.parametrize("embedding_position", ["ancestor", "descendant"])
def test_plan_and_embedding_directories_cannot_contain_one_another_while_unresolved(
    tmp_path: Path, embedding_position: str
):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    payload["extraction"]["channels"] = "ASK_USER"
    payload["artifacts"]["embedding_dir"] = (
        payload["experiment"]["root"] if embedding_position == "ancestor" else str(plan_dir / "embeddings")
    )
    _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("must not contain one another" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


def test_strict_runtime_loader_rejects_non_model_yaml(tmp_path: Path):
    config = tmp_path / "not-model.yaml"
    config.write_text("sleep2stat: {}\n")
    recipe, plan_dir, payload = _write_recipe(tmp_path, config=str(config))
    _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("strict runtime loading" in issue.message for issue in report.blocking_issues())


@pytest.mark.parametrize("variant", VARIANTS)
def test_plan_rejects_non_roformer_config(tmp_path: Path, variant: str):
    config_data = yaml.safe_load((REPO_ROOT / PRETRAIN_CONFIG).read_text())
    config_data["model"]["backbone"]["name"] = "hf_bert"
    config = tmp_path / "hf_bert_pretrain.yaml"
    config.write_text(yaml.safe_dump(config_data, sort_keys=False))
    recipe, plan_dir, payload = _write_recipe(tmp_path, variant=variant, config=str(config))

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("requires a RoFormer config" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


def test_plan_rejects_finetune_config_owned_preset(tmp_path: Path):
    config_data = yaml.safe_load((REPO_ROOT / FINETUNE_CONFIG).read_text())
    config_data["data"]["finetune_preset_path"] = "preset.pkl"
    config = tmp_path / "ppg_finetune.yaml"
    config.write_text(yaml.safe_dump(config_data, sort_keys=False))
    recipe, plan_dir, payload = _write_recipe(tmp_path, config=str(config))

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("finetune_preset_path must be null" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


def test_plan_rejects_finetune_data_model_channel_mismatch(tmp_path: Path):
    config_data = yaml.safe_load((REPO_ROOT / FINETUNE_CONFIG).read_text())
    config_data["data"]["data_channel_names"] = ["heartbeat"]
    config = tmp_path / "ppg_finetune.yaml"
    config.write_text(yaml.safe_dump(config_data, sort_keys=False))
    recipe, plan_dir, payload = _write_recipe(tmp_path, config=str(config))

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("data.data_channel_names must match" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


@pytest.mark.parametrize(
    ("eval_split", "source_field"), [("val", "train_dataset_names"), ("test", "test_dataset_names")]
)
def test_plan_rejects_rows_excluded_by_config_source_filter(tmp_path: Path, eval_split: str, source_field: str):
    config_data = yaml.safe_load((REPO_ROOT / FINETUNE_CONFIG).read_text())
    config_data["data"][source_field] = ["shhs"]
    config = tmp_path / "ppg_finetune.yaml"
    config.write_text(yaml.safe_dump(config_data, sort_keys=False))
    recipe, plan_dir, payload = _write_recipe(tmp_path, config=str(config), eval_split=eval_split)
    sample = tmp_path / "night.npz"
    Path(payload["inputs"]["data_index"][0]).write_text(f"path,split,duration,source\n{sample},{eval_split},60,mesa\n")
    if eval_split == "test":
        payload["evaluation_policy"] = {"external_test_locked": False, "final_test_unlocked": True}
        _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("does not match configured sources" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


def test_plan_accepts_rows_retained_by_config_source_filter(tmp_path: Path):
    config_data = yaml.safe_load((REPO_ROOT / FINETUNE_CONFIG).read_text())
    config_data["data"]["train_dataset_names"] = ["shhs"]
    config = tmp_path / "ppg_finetune.yaml"
    config.write_text(yaml.safe_dump(config_data, sort_keys=False))
    recipe, plan_dir, payload = _write_recipe(tmp_path, config=str(config))
    sample = tmp_path / "night.npz"
    Path(payload["inputs"]["data_index"][0]).write_text(f"path,split,duration,source\n{sample},val,60,shhs-v1\n")

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0


@pytest.mark.parametrize("task", ["finetune", "infer", "evaluate"])
def test_other_model_tasks_keep_unresolved_config_in_consultation(tmp_path: Path, task: str):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    payload.update({"task": task, "inputs": {"config": "ASK_USER"}})
    payload.pop("extraction")
    payload.pop("runtime")
    payload["artifacts"] = {"overwrite": False}
    payload["evaluation_policy"] = {}
    _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 2
    assert any(issue.field == "config" for issue in report.blocking_issues())
    assert not (plan_dir / "run.sh").exists()
