from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from agent_tools import experiments, plan_contract
from agent_tools.experiment_workspace import file_sha256, read_run_manifest
from agent_tools.manifests import write_rows
from agent_tools.models import REPO_ROOT
from agent_tools.plans import build_plan

VARIANTS = ("sleep2vec", "sleep2vec2", "sleep2expert")
PRETRAIN_CONFIG = "configs/sleep2vec_dense_pretrain_cls.yaml"
NON_CLS_PRETRAIN_CONFIG = "configs/sleep2vec_dense_pretrain.yaml"
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
    run = plan["runs"][0]
    frozen_config = plan_dir / "runs" / "run-000--whole-night-extraction" / "config.yaml"
    assert command.startswith(f"python -m {variant}.extract_embeddings ")
    assert f"--config {frozen_config}" in command
    assert frozen_config.read_bytes() == (REPO_ROOT / PRETRAIN_CONFIG).read_bytes()
    assert "--sequence-mode whole-night" in command
    assert "--batch-size 1" in command
    assert "--data-backend npz" in command
    snapshots = {item["field"]: item for item in run["input_snapshots"]}
    checkpoint = Path(_payload["inputs"]["ckpt_path"])
    index = Path(_payload["inputs"]["data_index"][0])
    assert snapshots == {
        "inputs.ckpt_path": {
            "field": "inputs.ckpt_path",
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "inputs.data_index[0]": {
            "field": "inputs.data_index[0]",
            "path": str(index),
            "sha256": file_sha256(index),
        },
    }
    run_json = json.loads((Path(run["run_dir"]) / "run.json").read_text())
    assert run_json["input_snapshots"] == run["input_snapshots"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_launch_rejects_checkpoint_changed_after_planning(tmp_path: Path, variant: str):
    recipe, plan_dir, payload = _write_recipe(tmp_path, variant=variant)
    assert build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    Path(payload["inputs"]["ckpt_path"]).write_bytes(b"replacement checkpoint")

    result = subprocess.run(["bash", str(plan_dir / "run.sh")], text=True, capture_output=True)

    assert result.returncode != 0
    assert "inputs.ckpt_path" in result.stderr
    assert not Path(payload["artifacts"]["embedding_dir"]).exists()
    assert read_run_manifest(Path(payload["experiment"]["root"]))[0]["status"] == "planned"


def test_launch_rejects_index_changed_after_planning(tmp_path: Path):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    assert build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    sample = tmp_path / "night.npz"
    Path(payload["inputs"]["data_index"][0]).write_text(f"path,split,duration\n{sample},val,30\n")

    result = subprocess.run(["bash", str(plan_dir / "run.sh")], text=True, capture_output=True)

    assert result.returncode != 0
    assert "inputs.data_index[0]" in result.stderr
    assert not Path(payload["artifacts"]["embedding_dir"]).exists()
    assert read_run_manifest(Path(payload["experiment"]["root"]))[0]["status"] == "planned"


def test_status_rejects_frozen_input_snapshot_omission(tmp_path: Path):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    assert build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["input_snapshots"] = [
        snapshot for snapshot in plan["recipe"]["input_snapshots"] if snapshot["field"] != "inputs.ckpt_path"
    ]
    plan["runs"][0]["input_snapshots"] = [
        snapshot for snapshot in plan["runs"][0]["input_snapshots"] if snapshot["field"] != "inputs.ckpt_path"
    ]
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved["input_snapshots"] = plan["recipe"]["input_snapshots"]
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))

    root = Path(payload["experiment"]["root"])
    rows = read_run_manifest(root)
    rows[0]["input_snapshots"] = plan["runs"][0]["input_snapshots"]
    write_rows(root / "run_manifest.tsv", rows)

    with pytest.raises(ValueError, match="input snapshots differ from required recipe inputs"):
        experiments.experiment_status(root)


def test_status_does_not_rehash_frozen_external_inputs(tmp_path: Path):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    assert build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    Path(payload["inputs"]["ckpt_path"]).write_bytes(b"changed after planning")

    snapshot = experiments.experiment_status(Path(payload["experiment"]["root"]))

    assert snapshot["summary"]["state"] == "ready_to_launch"


def test_status_recompiles_relative_inputs_with_frozen_creator_root(tmp_path: Path, monkeypatch):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    payload["inputs"]["ckpt_path"] = os.path.relpath(tmp_path / "model.ckpt", REPO_ROOT)
    payload["inputs"]["data_index"] = [os.path.relpath(tmp_path / "index.csv", REPO_ROOT)]
    _rewrite(recipe, payload)
    assert build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    monkeypatch.setattr(plan_contract, "REPO_ROOT", Path("/controller/repo"))

    snapshot = experiments.experiment_status(Path(payload["experiment"]["root"]))

    assert snapshot["summary"]["state"] == "ready_to_launch"


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


@pytest.mark.parametrize("variant", VARIANTS)
def test_plan_rejects_config_without_cls_embedding(tmp_path: Path, variant: str):
    recipe, plan_dir, payload = _write_recipe(tmp_path, variant=variant, config=NON_CLS_PRETRAIN_CONFIG)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("model.cls.embedding_type=bert" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


def test_repo_relative_config_loads_outside_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    recipe, plan_dir, _payload = _write_recipe(tmp_path)
    monkeypatch.chdir(tmp_path)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0


@pytest.mark.parametrize(
    ("config", "channels"),
    [(PRETRAIN_CONFIG, ["heartbeat", "breath"]), (FINETUNE_CONFIG, ["ppg"])],
)
def test_user_decisions_materialize_config_and_channels(tmp_path: Path, config: str, channels: list[str]):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    payload["inputs"]["config"] = "ASK_USER"
    payload["extraction"]["channels"] = "ASK_USER"
    _rewrite(recipe, payload)
    decisions = tmp_path / "decisions.yaml"
    decisions.write_text(
        yaml.safe_dump(
            {
                "decisions": {
                    "config": {"value": config, "source": "explicit_user"},
                    "embedding_channels": {"value": channels, "source": "explicit_user"},
                }
            }
        )
    )

    report = build_plan(recipe_path=recipe, output_dir=plan_dir, user_decisions_path=decisions)

    assert report.exit_code == 0
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    assert resolved["inputs"]["config"] == config
    assert resolved["extraction"]["channels"] == channels


def test_strict_runtime_loader_uses_bound_config_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sleep2vec import config as runtime_config

    source_config = tmp_path / "ppg_pretrain.yaml"
    original_bytes = (REPO_ROOT / PRETRAIN_CONFIG).read_bytes()
    source_config.write_bytes(original_bytes)
    recipe, plan_dir, _payload = _write_recipe(tmp_path, config=str(source_config))
    original_loader = runtime_config.load_pretrain_config
    loaded_paths: list[Path] = []

    def load_pretrain_config(path: str | Path):
        loaded_paths.append(Path(path))
        source_config.write_text("not: a valid model config\n")
        return original_loader(path)

    monkeypatch.setattr(runtime_config, "load_pretrain_config", load_pretrain_config)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0
    assert loaded_paths and all(path != source_config for path in loaded_paths)
    frozen_config = plan_dir / "runs" / "run-000--whole-night-extraction" / "config.yaml"
    assert frozen_config.read_bytes() == original_bytes


def test_plan_rejects_index_changed_during_final_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from agent_tools.adapters.embedding_extraction import EmbeddingExtractionAdapter

    recipe, plan_dir, payload = _write_recipe(tmp_path)
    index = Path(payload["inputs"]["data_index"][0])
    original_check = EmbeddingExtractionAdapter.configured_input_issues
    checks = 0

    def configured_input_issues(self, recipe_payload, config_summary):
        nonlocal checks
        checks += 1
        issues = original_check(self, recipe_payload, config_summary)
        if checks == 2:
            index.write_text(index.read_text().replace(",60", ",30"))
        return issues

    monkeypatch.setattr(EmbeddingExtractionAdapter, "configured_input_issues", configured_input_issues)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("changed while the final plan snapshot" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


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


@pytest.mark.parametrize(
    ("section", "field", "decision_field"),
    [
        ("inputs", "config", "config"),
        ("inputs", "ckpt_path", "ckpt_path"),
        ("inputs", "eval_split", "eval_split"),
        ("extraction", "channels", "embedding_channels"),
        ("extraction", "embedding_kind", "embedding_kind"),
        ("extraction", "layer_index", "layer_index"),
        ("extraction", "max_source_tokens", "max_source_tokens"),
        ("extraction", "output_format", "output_format"),
        ("extraction", "sequence_mode", "sequence_mode"),
        ("artifacts", "overwrite", "overwrite_policy"),
    ],
)
@pytest.mark.parametrize("value", [None, ""])
def test_blank_required_recipe_choices_return_consultation_status(
    tmp_path: Path, section: str, field: str, decision_field: str, value: object
):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    payload[section][field] = value
    _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 2
    assert any(issue.field == decision_field for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


@pytest.mark.parametrize("channels", [["heartbeat", 1], ["heartbeat", None], ["heartbeat", ""]])
def test_invalid_channel_values_fail_without_exception(tmp_path: Path, channels: list[object]):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    payload["extraction"]["channels"] = channels
    _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("non-empty strings" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


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


def test_plan_rejects_duplicate_whole_night_index_headers(tmp_path: Path):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    first = tmp_path / "night.npz"
    second = tmp_path / "other.npz"
    second.touch()
    Path(payload["inputs"]["data_index"][0]).write_text(f"path,split,duration,path\n{first},val,60,{second}\n")

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("duplicate columns" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


def test_plan_rejects_duplicate_whole_night_sample_keys(tmp_path: Path):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    first = tmp_path / "first" / "set" / "night.npz"
    second = tmp_path / "second" / "set" / "night.npz"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    Path(payload["inputs"]["data_index"][0]).write_text(
        "path,split,duration,source\n" f"{first},val,60,mesa\n" f"{second},val,60,mesa\n"
    )

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("Duplicate embedding sample_key" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


def test_plan_rejects_surplus_whole_night_index_values(tmp_path: Path):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    sample = tmp_path / "night.npz"
    Path(payload["inputs"]["data_index"][0]).write_text(f"path,split,duration\n{sample},val,60,unexpected\n")

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("unexpected extra values" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


def test_nonempty_embedding_output_fails_before_workspace(tmp_path: Path):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    embedding_dir = Path(payload["artifacts"]["embedding_dir"])
    embedding_dir.mkdir()
    (embedding_dir / "old.npz").touch()

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert not Path(payload["experiment"]["root"]).exists()


@pytest.mark.parametrize("symlink_position", ["output", "ancestor"])
def test_dangling_embedding_output_symlink_fails_before_workspace(tmp_path: Path, symlink_position: str):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    embedding_dir = Path(payload["artifacts"]["embedding_dir"])
    if symlink_position == "output":
        embedding_dir.symlink_to(tmp_path / "missing", target_is_directory=True)
    else:
        parent = tmp_path / "link"
        parent.symlink_to(tmp_path / "missing", target_is_directory=True)
        payload["artifacts"]["embedding_dir"] = str(parent / "embeddings")
        _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("must not be a symlink" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


def test_existing_embedding_output_symlink_ancestor_fails_before_workspace(tmp_path: Path):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    parent = tmp_path / "link"
    parent.symlink_to(target, target_is_directory=True)
    payload["artifacts"]["embedding_dir"] = str(parent / "embeddings")
    _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("must not be a symlink" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


def test_embedding_output_parent_component_fails_before_workspace(tmp_path: Path):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "link").symlink_to(target, target_is_directory=True)
    payload["artifacts"]["embedding_dir"] = str(tmp_path / "new" / ".." / "link" / "embeddings")
    _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("must not contain '..'" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


@pytest.mark.parametrize("channel", ["/tmp/export", "../escape"])
def test_plan_rejects_unsafe_extraction_channels(tmp_path: Path, channel: str):
    config = yaml.safe_load((REPO_ROOT / PRETRAIN_CONFIG).read_text())
    config["model"]["channels"][0]["name"] = channel
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    recipe, plan_dir, payload = _write_recipe(tmp_path, config=str(config_path))
    payload["extraction"]["channels"] = [channel]
    _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("single safe path components" in issue.message for issue in report.blocking_issues())
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


@pytest.mark.parametrize("managed_dir", ["plans", "reports", "steps"])
def test_embedding_output_rejects_experiment_managed_directories(tmp_path: Path, managed_dir: str):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    payload["artifacts"]["embedding_dir"] = str(Path(payload["experiment"]["root"]) / managed_dir / "embeddings")
    _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert any("experiment-managed" in issue.message for issue in report.blocking_issues())
    assert not Path(payload["experiment"]["root"]).exists()


def test_embedding_output_allows_dedicated_experiment_directory(tmp_path: Path):
    recipe, plan_dir, payload = _write_recipe(tmp_path)
    payload["artifacts"]["embedding_dir"] = str(Path(payload["experiment"]["root"]) / "embeddings")
    _rewrite(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0


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


def test_plan_uses_index_path_as_missing_source_fallback(tmp_path: Path):
    config_data = yaml.safe_load((REPO_ROOT / FINETUNE_CONFIG).read_text())
    config_data["data"]["train_dataset_names"] = ["shhs"]
    config = tmp_path / "ppg_finetune.yaml"
    config.write_text(yaml.safe_dump(config_data, sort_keys=False))
    recipe, plan_dir, payload = _write_recipe(tmp_path, config=str(config))
    sample = tmp_path / "night.npz"
    index = tmp_path / "shhs-index.csv"
    index.write_text(f"path,split,duration\n{sample},val,60\n")
    payload["inputs"]["data_index"] = [str(index)]
    _rewrite(recipe, payload)

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
