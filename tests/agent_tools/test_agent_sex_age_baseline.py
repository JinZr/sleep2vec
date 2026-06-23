from __future__ import annotations

import json
from pathlib import Path
import pickle
import subprocess

import yaml

from agent_tools.configs import config_summary
from agent_tools.plans import build_plan, evaluate_recipe
from data.default_dataset import SampleIndex


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload))
    return path


def _write_survival_config(tmp_path: Path) -> Path:
    index = tmp_path / "index.csv"
    disease_columns = tmp_path / "disease_columns.txt"
    event_time = tmp_path / "event_time.csv"
    is_event = tmp_path / "is_event.csv"
    has_label = tmp_path / "has_label.csv"
    index.write_text("eid,split,age,sex\n" "001,train,50,0\n" "002,val,60,1\n" "003,test,55,0\n")
    disease_columns.write_text("d1\nd2\n")
    header = "eid,d1,d2\n"
    event_time.write_text(header + "001,10,20\n002,30,40\n003,50,60\n")
    is_event.write_text(header + "001,1,0\n002,0,1\n003,1,1\n")
    has_label.write_text(header + "001,1,1\n002,1,1\n003,1,1\n")
    return _write_yaml(
        tmp_path / "cox.yaml",
        {
            "model": {
                "name": "sex_age_mlp",
                "features": ["age", "sex"],
                "age": {"transform": "divide", "scale": 100.0, "embedding_dim": 4},
                "sex": {"encoding": "binary", "embedding_dim": 4},
                "head": {"hidden_dim": 8, "dropout": 0.1, "activation": "elu"},
            },
            "data": {
                "backend": "npz",
                "finetune_data_index": str(index),
                "finetune_preset_path": None,
                "kaldi_data_root": None,
                "kaldi_manifest": None,
                "split_column": "split",
                "key_column": "eid",
                "deduplicate_by_key": True,
            },
            "finetune": {
                "task": {
                    "type": "survival",
                    "output_dim": 2,
                    "is_seq": False,
                    "monitor": "val_c_index",
                    "monitor_mod": "max",
                },
                "survival": {
                    "key_column": "eid",
                    "disease_columns_index": str(disease_columns),
                    "event_time_index": str(event_time),
                    "is_event_index": str(is_event),
                    "has_label_index": str(has_label),
                },
            },
            "outputs": {"prediction_csv": True, "per_disease_metrics_csv": True},
        },
    )


def _write_kaldi_survival_config(tmp_path: Path, *, split_key: str = "003") -> Path:
    config = _write_survival_config(tmp_path)
    kaldi_root = tmp_path / "kaldi"
    kaldi_root.mkdir()
    (kaldi_root / "val.csv").write_text(f"eid,age,sex\n{split_key},55,0\n")
    manifest = tmp_path / "kaldi_manifest.json"
    manifest.write_text(json.dumps({"splits": {"val": {"manifest": "val.csv"}}}))
    payload = yaml.safe_load(config.read_text())
    payload["data"].update(
        {
            "backend": "kaldi",
            "finetune_data_index": None,
            "finetune_preset_path": None,
            "kaldi_data_root": str(kaldi_root),
            "kaldi_manifest": str(manifest),
        }
    )
    return _write_yaml(config, payload)


def _write_metadata_preset(path: Path, rows: list[dict]) -> Path:
    with path.open("wb") as file_obj:
        pickle.dump(
            [
                SampleIndex(
                    id=str(row["eid"]),
                    path="ignored.npz",
                    start=0,
                    end=1,
                    metadata=row,
                )
                for row in rows
            ],
            file_obj,
        )
    return path


def _write_multilabel_config(
    tmp_path: Path,
    *,
    index_path: str | Path | None = None,
    sidecars: dict[str, str] | None = None,
) -> Path:
    index = tmp_path / "index.csv"
    disease_columns = tmp_path / "disease_columns.txt"
    label = tmp_path / "label.csv"
    has_label = tmp_path / "has_label.csv"
    if index_path is None:
        index.write_text("eid,split,age,sex\n" "001,train,50,0\n" "002,val,60,1\n" "003,test,55,0\n")
        index_path = index
    if sidecars is None:
        disease_columns.write_text("d1\nd2\n")
        header = "eid,d1,d2\n"
        label.write_text(header + "001,1,0\n002,0,1\n003,1,1\n")
        has_label.write_text(header + "001,1,1\n002,1,1\n003,1,1\n")
        sidecars = {
            "disease_columns_index": str(disease_columns),
            "label_index": str(label),
            "has_label_index": str(has_label),
        }
    return _write_yaml(
        tmp_path / "multilabel.yaml",
        {
            "model": {
                "name": "sex_age_mlp",
                "features": ["age", "sex"],
                "age": {"transform": "divide", "scale": 100.0, "embedding_dim": 4},
                "sex": {"encoding": "binary", "embedding_dim": 4},
                "head": {"hidden_dim": 8, "dropout": 0.1, "activation": "elu"},
            },
            "data": {
                "backend": "npz",
                "finetune_data_index": str(index_path),
                "finetune_preset_path": None,
                "kaldi_data_root": None,
                "kaldi_manifest": None,
                "split_column": "split",
                "key_column": "eid",
                "deduplicate_by_key": True,
            },
            "finetune": {
                "task": {
                    "type": "multilabel_classification",
                    "output_dim": 2,
                    "is_seq": False,
                    "monitor": "val_loss",
                    "monitor_mod": "min",
                },
                "multilabel": {"key_column": "eid", **sidecars},
            },
            "outputs": {"prediction_csv": True, "per_disease_metrics_csv": True},
        },
    )


def _finetune_recipe(tmp_path: Path, config: Path) -> Path:
    return _write_yaml(
        tmp_path / "finetune.yaml",
        {
            "name": "unit_sex_age",
            "task": "finetune",
            "variant": "sex_age_baseline",
            "inputs": {"config": str(config), "label_name": "incident_cox"},
            "runtime": {"devices": [0], "device": "cpu", "epochs": 1, "batch_size": 2, "num_workers": 0},
            "artifacts": {"version_name": "unit-sex-age", "results_csv_path": str(tmp_path / "results.csv")},
            "evaluation_policy": {
                "selection_metric": "val_c_index",
                "selection_mode": "max",
                "selection_split": "val",
                "external_test_locked": True,
                "test_after_fit": False,
            },
            "decisions": {
                "task": {"value": "finetune", "source": "explicit_recipe"},
                "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
                "pretrained_backbone_path": {"value": None, "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "required_channels": {"value": [], "source": "explicit_config"},
            },
        },
    )


def _infer_recipe(tmp_path: Path, config: Path, ckpt: Path) -> Path:
    return _write_yaml(
        tmp_path / "infer.yaml",
        {
            "name": "unit_sex_age_infer",
            "task": "infer",
            "variant": "sex_age_baseline",
            "inputs": {
                "config": str(config),
                "ckpt_path": str(ckpt),
                "label_name": "incident_cox",
                "eval_split": "val",
            },
            "runtime": {"devices": [0], "accelerator": "cpu", "device": "cpu", "batch_size": 2, "num_workers": 0},
            "artifacts": {"overwrite": False},
            "evaluation_policy": {"external_test_locked": True},
            "decisions": {
                "task": {"value": "infer", "source": "explicit_recipe"},
                "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def test_sex_age_baseline_config_summary_reports_backend_and_variant(tmp_path: Path):
    config = _write_survival_config(tmp_path)

    summary = config_summary(config)

    assert summary["variant_guess"] == "sex_age_baseline"
    assert summary["data_backend"] == "npz"
    assert summary["data"]["finetune_data_index"]


def test_sex_age_baseline_finetune_plan_renders_standalone_module(tmp_path: Path):
    config = _write_survival_config(tmp_path)
    recipe = _finetune_recipe(tmp_path, config)

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan")

    assert report.exit_code == 0
    script = (tmp_path / "plan" / "run.sh").read_text()
    assert "python -m sex_age_baseline.finetune" in script
    assert "--pretrained-backbone-path" not in script
    assert "--inference-preset-path" not in script


def test_sex_age_baseline_variant_routes_invalid_config_to_strict_loader(tmp_path: Path):
    config = _write_survival_config(tmp_path)
    payload = yaml.safe_load(config.read_text())
    payload["model"]["name"] = "sex_age_mlp_typo"
    _write_yaml(config, payload)
    recipe = _finetune_recipe(tmp_path, config)

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan-invalid-config")

    assert report.exit_code == 2
    assert any(
        issue.field == "config" and "model.name must be 'sex_age_mlp'" in issue.message for issue in report.issues
    )
    assert not (tmp_path / "plan-invalid-config" / "run.sh").exists()


def test_sex_age_baseline_kaldi_finetune_blocks_survival_keys_missing_from_sidecars(tmp_path: Path):
    config = _write_kaldi_survival_config(tmp_path, split_key="004")
    recipe = _finetune_recipe(tmp_path, config)

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan-kaldi-missing-sidecar-key")

    assert report.exit_code == 1
    assert any(
        issue.field == "data_input" and "survival key values missing from sidecars" in issue.message
        for issue in report.issues
    )
    assert not (tmp_path / "plan-kaldi-missing-sidecar-key" / "run.sh").exists()


def test_sex_age_baseline_kaldi_infer_accepts_manifest_split_without_split_column(tmp_path: Path):
    config = _write_kaldi_survival_config(tmp_path, split_key="003")
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("placeholder")
    recipe = _infer_recipe(tmp_path, config, ckpt)

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan-kaldi-valid")

    assert report.exit_code == 0
    script = (tmp_path / "plan-kaldi-valid" / "run.sh").read_text()
    assert "python -m sex_age_baseline.infer" in script


def test_sex_age_baseline_finetune_blocks_pretrained_backbone_path(tmp_path: Path):
    config = _write_survival_config(tmp_path)
    recipe = _finetune_recipe(tmp_path, config)
    pretrained = tmp_path / "pretrained.ckpt"
    pretrained.write_text("checkpoint")
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["pretrained_backbone_path"] = str(pretrained)
    payload["decisions"]["pretrained_backbone_path"] = {
        "value": str(pretrained),
        "source": "explicit_recipe",
    }
    _write_yaml(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan-pretrained")

    assert report.exit_code == 1
    assert any(issue.field == "pretrained_backbone_path" for issue in report.issues)
    assert not (tmp_path / "plan-pretrained" / "run.sh").exists()


def test_sex_age_baseline_remote_ssh_multilabel_checks_sidecar_paths(tmp_path: Path, monkeypatch):
    config = _write_multilabel_config(
        tmp_path,
        index_path="/wujidata/multilabel/index.csv",
        sidecars={
            "disease_columns_index": "/wujidata/multilabel/disease_columns.txt",
            "label_index": "/wujidata/multilabel/label.csv",
            "has_label_index": "/wujidata/multilabel/has_label.csv",
        },
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("agent_tools.decisions.subprocess.run", fake_run)

    for validation in ("ssh", "remote"):
        recipe = _finetune_recipe(tmp_path, config)
        payload = yaml.safe_load(recipe.read_text())
        payload["name"] = f"unit_sex_age_multilabel_{validation}"
        payload["inputs"]["label_name"] = "diagnosis"
        payload["decisions"]["label_name"] = {"value": "diagnosis", "source": "explicit_recipe"}
        payload["evaluation_policy"].update({"selection_metric": "val_loss", "selection_mode": "min"})
        payload["artifacts"]["results_csv_path"] = str(tmp_path / f"{validation}.csv")
        payload["artifacts"]["version_name"] = validation
        payload["execution"] = {
            "target": "ssh",
            "host": "baichuan3",
            "path_context": "remote",
            "path_validation": validation,
        }
        _write_yaml(recipe, payload)

        _recipe, _cfg, report = evaluate_recipe(recipe)

        assert report.exit_code == 0

    call_scripts = [command[2] for command in calls]
    for path in (
        "/wujidata/multilabel/index.csv",
        "/wujidata/multilabel/disease_columns.txt",
        "/wujidata/multilabel/label.csv",
        "/wujidata/multilabel/has_label.csv",
    ):
        assert any(path in script for script in call_scripts)

    calls.clear()

    def fail_label(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, int("label.csv" in command[2]), "", "")

    monkeypatch.setattr("agent_tools.decisions.subprocess.run", fail_label)
    missing_recipe = _finetune_recipe(tmp_path, config)
    missing_payload = yaml.safe_load(missing_recipe.read_text())
    missing_payload["name"] = "unit_sex_age_multilabel_missing_label"
    missing_payload["inputs"]["label_name"] = "diagnosis"
    missing_payload["decisions"]["label_name"] = {"value": "diagnosis", "source": "explicit_recipe"}
    missing_payload["evaluation_policy"].update({"selection_metric": "val_loss", "selection_mode": "min"})
    missing_payload["execution"] = {
        "target": "ssh",
        "host": "baichuan3",
        "path_context": "remote",
        "path_validation": "ssh",
    }
    _write_yaml(missing_recipe, missing_payload)

    _recipe, _cfg, report = evaluate_recipe(missing_recipe)

    assert report.exit_code == 1
    assert any(issue.field == "finetune.multilabel.label_index" for issue in report.issues)


def test_sex_age_baseline_hparam_blocks_local_multilabel_sidecar_issues(tmp_path: Path):
    config = _write_multilabel_config(tmp_path)
    config_payload = yaml.safe_load(config.read_text())
    config_payload["finetune"]["multilabel"]["label_index"] = str(tmp_path / "missing_label.csv")
    _write_yaml(config, config_payload)
    base = _finetune_recipe(tmp_path, config)
    base_payload = yaml.safe_load(base.read_text())
    base_payload["inputs"]["label_name"] = "diagnosis"
    base_payload["decisions"]["label_name"] = {"value": "diagnosis", "source": "explicit_recipe"}
    base_payload["evaluation_policy"].update({"selection_metric": "val_loss", "selection_mode": "min"})
    base_payload["artifacts"]["version_name"] = "unit-sex-age-multilabel"
    _write_yaml(base, base_payload)
    recipe = _write_yaml(
        tmp_path / "hparam_multilabel.yaml",
        {
            "name": "unit_sex_age_multilabel_hparam",
            "task": "hparam_tune",
            "variant": "sex_age_baseline",
            "base_recipe": str(base),
            "search": {"method": "grid", "max_trials": 1, "parameters": {"runtime.lr": [1e-3]}},
            "evaluation_policy": {
                "selection_metric": "val_loss",
                "selection_mode": "min",
                "selection_split": "val",
                "external_test_locked": True,
                "test_after_fit": False,
                "final_eval_split": "validation",
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "diagnosis", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )

    _recipe, _cfg, report = evaluate_recipe(recipe)

    assert report.exit_code == 2
    assert any(issue.field == "multilabel_sidecars" for issue in report.issues)


def test_sex_age_baseline_hparam_blocks_base_pretrained_backbone_path(tmp_path: Path):
    config = _write_survival_config(tmp_path)
    base = _finetune_recipe(tmp_path, config)
    pretrained = tmp_path / "pretrained.ckpt"
    pretrained.write_text("checkpoint")
    base_payload = yaml.safe_load(base.read_text())
    base_payload["inputs"]["pretrained_backbone_path"] = str(pretrained)
    base_payload["decisions"]["pretrained_backbone_path"] = {
        "value": str(pretrained),
        "source": "explicit_recipe",
    }
    _write_yaml(base, base_payload)
    recipe = _write_yaml(
        tmp_path / "hparam_pretrained.yaml",
        {
            "name": "unit_sex_age_hparam_pretrained",
            "task": "hparam_tune",
            "variant": "sex_age_baseline",
            "base_recipe": str(base),
            "search": {"method": "grid", "max_trials": 1, "parameters": {"runtime.lr": [1e-3]}},
            "evaluation_policy": {
                "selection_metric": "val_c_index",
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
                "label_name": {"value": "incident_cox", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan-hparam-pretrained")

    assert report.exit_code == 1
    assert any(issue.field == "base_finetune.pretrained_backbone_path" for issue in report.issues)
    assert not (tmp_path / "plan-hparam-pretrained" / "trial_000.sh").exists()


def test_sex_age_baseline_finetune_preset_keeps_survival_sidecar_checks(tmp_path: Path):
    config = _write_survival_config(tmp_path)
    config_payload = yaml.safe_load(config.read_text())
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(b"preset")
    config_payload["data"]["finetune_data_index"] = None
    config_payload["data"]["finetune_preset_path"] = str(preset)
    config_payload["finetune"]["survival"]["event_time_index"] = str(tmp_path / "missing_event_time.csv")
    _write_yaml(config, config_payload)
    recipe = _finetune_recipe(tmp_path, config)

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan-finetune-preset-bad-sidecars")

    assert report.exit_code == 2
    assert any(issue.field == "survival_sidecars" for issue in report.issues)
    assert not (tmp_path / "plan-finetune-preset-bad-sidecars" / "run.sh").exists()


def test_sex_age_baseline_finetune_preset_blocks_survival_keys_missing_from_sidecars(tmp_path: Path):
    config = _write_survival_config(tmp_path)
    config_payload = yaml.safe_load(config.read_text())
    preset = _write_metadata_preset(
        tmp_path / "preset_missing_sidecar_key.pkl",
        [{"eid": "004", "split": "val", "age": 55, "sex": 0}],
    )
    config_payload["data"]["finetune_data_index"] = None
    config_payload["data"]["finetune_preset_path"] = str(preset)
    _write_yaml(config, config_payload)
    recipe = _finetune_recipe(tmp_path, config)

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan-preset-missing-sidecar-key")

    assert report.exit_code == 1
    assert any(
        issue.field == "data_input" and "survival key values missing from sidecars" in issue.message
        for issue in report.issues
    )
    assert not (tmp_path / "plan-preset-missing-sidecar-key" / "run.sh").exists()


def test_sex_age_baseline_inference_preset_keeps_survival_sidecar_checks(tmp_path: Path):
    config = _write_survival_config(tmp_path)
    config_payload = yaml.safe_load(config.read_text())
    config_payload["finetune"]["survival"]["event_time_index"] = str(tmp_path / "missing_event_time.csv")
    _write_yaml(config, config_payload)
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("placeholder")
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(b"preset")
    recipe = _infer_recipe(tmp_path, config, ckpt)
    recipe_payload = yaml.safe_load(recipe.read_text())
    recipe_payload["inputs"]["inference_preset_path"] = str(preset)
    _write_yaml(recipe, recipe_payload)

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan-infer-preset-bad-sidecars")

    assert report.exit_code == 2
    assert any(issue.field == "survival_sidecars" for issue in report.issues)
    assert not (tmp_path / "plan-infer-preset-bad-sidecars" / "run.sh").exists()


def test_sex_age_baseline_infer_plan_can_render_inference_preset_path(tmp_path: Path):
    config = _write_survival_config(tmp_path)
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("placeholder")
    preset = _write_metadata_preset(tmp_path / "preset.pkl", [{"eid": "002", "split": "val", "age": 60, "sex": 1}])
    recipe = _infer_recipe(tmp_path, config, ckpt)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["inference_preset_path"] = str(preset)
    _write_yaml(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan-infer-preset")

    assert report.exit_code == 0
    script = (tmp_path / "plan-infer-preset" / "run.sh").read_text()
    assert "python -m sex_age_baseline.infer" in script
    assert "--inference-preset-path" in script


def test_sex_age_baseline_infer_preset_checks_only_eval_split_keys(tmp_path: Path):
    config = _write_survival_config(tmp_path)
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("placeholder")
    preset = _write_metadata_preset(
        tmp_path / "preset_split_filter.pkl",
        [
            {"eid": "002", "split": "val", "age": 60, "sex": 1},
            {"eid": "004", "split": "test", "age": 55, "sex": 0},
        ],
    )
    recipe = _infer_recipe(tmp_path, config, ckpt)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["inference_preset_path"] = str(preset)
    _write_yaml(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan-infer-preset-split-filter")

    assert report.exit_code == 0
    assert (tmp_path / "plan-infer-preset-split-filter" / "run.sh").exists()


def test_sex_age_baseline_infer_blocks_pretrained_backbone_path(tmp_path: Path):
    config = _write_survival_config(tmp_path)
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("placeholder")
    pretrained = tmp_path / "pretrained.ckpt"
    pretrained.write_text("checkpoint")
    recipe = _infer_recipe(tmp_path, config, ckpt)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["pretrained_backbone_path"] = str(pretrained)
    _write_yaml(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan-infer-pretrained")

    assert report.exit_code == 1
    assert any(issue.field == "pretrained_backbone_path" for issue in report.issues)
    assert not (tmp_path / "plan-infer-pretrained" / "run.sh").exists()


def test_sex_age_baseline_infer_blocks_override_dataset_names(tmp_path: Path):
    config = _write_survival_config(tmp_path)
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("placeholder")
    recipe = _infer_recipe(tmp_path, config, ckpt)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["override_dataset_names"] = ["ukb"]
    _write_yaml(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan-infer-override-datasets")

    assert report.exit_code == 1
    assert any(issue.field == "override_dataset_names" for issue in report.issues)
    assert not (tmp_path / "plan-infer-override-datasets" / "run.sh").exists()


def test_sex_age_baseline_infer_plan_renders_standalone_module(tmp_path: Path):
    config = _write_survival_config(tmp_path)
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("placeholder")
    recipe = _infer_recipe(tmp_path, config, ckpt)

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan-infer")

    assert report.exit_code == 0
    script = (tmp_path / "plan-infer" / "run.sh").read_text()
    assert "python -m sex_age_baseline.infer" in script
    assert "--pretrained-backbone-path" not in script
    assert "--inference-preset-path" not in script
