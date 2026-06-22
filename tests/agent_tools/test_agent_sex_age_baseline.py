from __future__ import annotations

from pathlib import Path
import pickle

import yaml

from agent_tools.configs import config_summary
from agent_tools.plans import build_plan
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


def test_sex_age_baseline_infer_plan_can_render_inference_preset_path(tmp_path: Path):
    config = _write_survival_config(tmp_path)
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("placeholder")
    preset = tmp_path / "preset.pkl"
    with preset.open("wb") as file_obj:
        pickle.dump(
            [
                SampleIndex(
                    id="002",
                    path="ignored.npz",
                    start=0,
                    end=1,
                    metadata={"eid": "002", "split": "val", "age": 60, "sex": 1},
                )
            ],
            file_obj,
        )
    recipe = _infer_recipe(tmp_path, config, ckpt)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["inference_preset_path"] = str(preset)
    _write_yaml(recipe, payload)

    report = build_plan(recipe_path=recipe, output_dir=tmp_path / "plan-infer-preset")

    assert report.exit_code == 0
    script = (tmp_path / "plan-infer-preset" / "run.sh").read_text()
    assert "python -m sex_age_baseline.infer" in script
    assert "--inference-preset-path" in script


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
