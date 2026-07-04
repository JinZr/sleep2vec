from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pytest
import yaml

from sleep2vec.common import apply_finetune_config, apply_task_flags, dump_cli_args_yaml
from sleep2vec.config import (
    ConfusionMatrixVisualizationConfig,
    EvalVisualizationPlotConfig,
    EvalVisualizationsConfig,
    TaskConfig,
)


def _write_yaml(tmp_path: Path, payload: dict, name: str = "finetune.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return path


def _finetune_payload() -> dict:
    return {
        "model": {
            "backbone": {
                "name": "roformer",
                "hidden_size": 8,
                "num_hidden_layers": 3,
                "num_attention_heads": 2,
                "vocab_size": 1,
            },
            "projection": {
                "name": "simclr",
                "enabled": True,
                "hidden_dim": 8,
                "out_dim": 4,
            },
            "cls": {
                "embedding_type": None,
                "downstream": "tokens",
            },
            "channels": [
                {"name": "eeg", "input_dim": 4, "tokenizer": {"name": "linear", "out_dim": 8}},
                {"name": "ecg", "input_dim": 4, "tokenizer": {"name": "linear", "out_dim": 8}},
            ],
            "head": {
                "name": "classification",
                "dropout": 0.1,
                "hidden_dim": None,
                "channel_agg": {"name": "mean", "kwargs": {}},
                "temporal_agg": {"name": "mean", "kwargs": {}},
            },
        },
        "data": {
            "max_tokens": 4,
            "data_channel_names": ["ecg", "eeg"],
            "finetune_data_index": "index/custom.csv",
            "finetune_preset_path": "preset/custom.pkl",
            "train_dataset_names": ["train_a"],
            "test_dataset_names": ["test_a"],
            "n_few_shot": 32,
        },
        "finetune": {
            "freeze_tokenizer": False,
            "loss": {
                "class_weights": None,
                "pos_weight": None,
            },
            "sampler": {
                "weighted_random": False,
            },
            "lora": {
                "freeze_backbone_and_insert_lora": True,
                "insert_lora": True,
                "separate_adapters": True,
                "r": 4,
                "alpha": 12,
                "dropout": 0.15,
                "target_modules": ["query", "dense"],
                "use_dora": True,
            },
            "task": {
                "type": "classification",
                "output_dim": 2,
                "is_seq": False,
                "monitor": "val_accuracy",
                "monitor_mod": "max",
            },
        },
    }


def _multilabel_payload() -> dict:
    payload = _finetune_payload()
    payload["finetune"]["task"] = {
        "type": "multilabel_classification",
        "output_dim": 3,
        "is_seq": False,
        "monitor": "val_macro_auroc",
        "monitor_mod": "max",
    }
    payload["finetune"]["multilabel"] = {
        "key_column": "eid",
        "disease_columns_index": "disease_columns.txt",
        "label_index": "label.csv",
        "has_label_index": "has_label.csv",
    }
    return payload


@pytest.mark.parametrize(
    ("label_name", "expected"),
    [
        (
            "stage3",
            {
                "output_dim": 3,
                "is_classification": True,
                "is_seq": True,
                "is_multilabel": False,
                "monitor": "val_accuracy",
                "monitor_mod": "max",
                "label_source_name": "stage5",
                "auxiliary_label_source_names": [],
                "stage_names": ["W", "NREM", "REM"],
                "class_labels": ["W", "NREM", "REM"],
            },
        ),
        (
            "stage4",
            {
                "output_dim": 4,
                "is_classification": True,
                "is_seq": True,
                "is_multilabel": False,
                "monitor": "val_accuracy",
                "monitor_mod": "max",
                "label_source_name": "stage5",
                "auxiliary_label_source_names": [],
                "stage_names": ["W", "N1N2", "N3", "REM"],
                "class_labels": ["W", "N1N2", "N3", "REM"],
            },
        ),
        (
            "stage5",
            {
                "output_dim": 5,
                "is_classification": True,
                "is_seq": True,
                "is_multilabel": False,
                "monitor": "val_accuracy",
                "monitor_mod": "max",
                "label_source_name": "stage5",
                "auxiliary_label_source_names": [],
                "stage_names": ["W", "N1", "N2", "N3", "REM"],
                "class_labels": ["W", "N1", "N2", "N3", "REM"],
            },
        ),
        (
            "ahi",
            {
                "output_dim": 30,
                "is_classification": True,
                "is_seq": True,
                "is_multilabel": True,
                "monitor": "val_ahi_pearson",
                "monitor_mod": "max",
                "label_source_name": "ahi",
                "auxiliary_label_source_names": ["stage5"],
                "stage_names": None,
                "class_labels": None,
            },
        ),
        (
            "sex",
            {
                "output_dim": 2,
                "is_classification": True,
                "is_seq": False,
                "is_multilabel": False,
                "monitor": "val_accuracy",
                "monitor_mod": "max",
                "auxiliary_label_source_names": [],
                "class_labels": ["female", "male"],
            },
        ),
        (
            "age",
            {
                "output_dim": 1,
                "is_classification": False,
                "is_seq": False,
                "is_multilabel": False,
                "monitor": "val_mae",
                "monitor_mod": "min",
                "auxiliary_label_source_names": [],
                "class_labels": None,
            },
        ),
    ],
)
def test_apply_task_flags_builtin_labels(label_name: str, expected: dict):
    args = argparse.Namespace(label_name=label_name)
    apply_task_flags(args)

    for key, expected_value in expected.items():
        assert getattr(args, key) == expected_value


def test_apply_task_flags_unknown_label_requires_task_config():
    args = argparse.Namespace(label_name="custom_target")

    with pytest.raises(ValueError, match="Unknown label_name 'custom_target'"):
        apply_task_flags(args)


def test_apply_task_flags_sets_multilabel_classification_task_attrs():
    args = argparse.Namespace(label_name="disease_detection")
    task_cfg = TaskConfig(
        type="multilabel_classification",
        output_dim=3,
        is_seq=False,
        monitor="val_macro_auroc",
        monitor_mod="max",
    )

    apply_task_flags(args, task_cfg)

    assert args.output_dim == 3
    assert args.is_classification is True
    assert args.is_multilabel is True
    assert args.is_survival is False
    assert args.is_seq is False
    assert args.monitor == "val_macro_auroc"
    assert args.monitor_mod == "max"


def test_apply_task_flags_rejects_builtin_conflict_from_yaml_task():
    args = argparse.Namespace(label_name="stage5")
    task_cfg = TaskConfig(
        type="classification",
        output_dim=2,
        is_seq=True,
        monitor="val_accuracy",
        monitor_mod="max",
    )

    with pytest.raises(ValueError, match="output_dim must be 5 when --label-name is 'stage5'"):
        apply_task_flags(args, task_cfg)


def test_apply_task_flags_rejects_stage4_builtin_conflict_from_yaml_task():
    args = argparse.Namespace(label_name="stage4")
    task_cfg = TaskConfig(
        type="classification",
        output_dim=5,
        is_seq=True,
        monitor="val_accuracy",
        monitor_mod="max",
    )

    with pytest.raises(ValueError, match="output_dim must be 4 when --label-name is 'stage4'"):
        apply_task_flags(args, task_cfg)


def test_apply_task_flags_rejects_ahi_builtin_conflict_from_yaml_task():
    args = argparse.Namespace(label_name="ahi")
    task_cfg = TaskConfig(
        type="classification",
        output_dim=29,
        is_seq=True,
        monitor="val_ahi_pearson",
        monitor_mod="max",
    )

    with pytest.raises(ValueError, match="output_dim must be 30 when --label-name is 'ahi'"):
        apply_task_flags(args, task_cfg)


def test_apply_task_flags_rejects_ahi_val_loss_monitor():
    args = argparse.Namespace(label_name="ahi")
    task_cfg = TaskConfig(
        type="classification",
        output_dim=30,
        is_seq=True,
        monitor="val_loss",
        monitor_mod="min",
    )

    with pytest.raises(ValueError, match="finetune.task.monitor must be one of"):
        apply_task_flags(args, task_cfg)


def test_apply_task_flags_rejects_ahi_monitor_that_validation_never_logs():
    args = argparse.Namespace(label_name="ahi")
    task_cfg = TaskConfig(
        type="classification",
        output_dim=30,
        is_seq=True,
        monitor="val_f1",
        monitor_mod="max",
    )

    with pytest.raises(ValueError, match="finetune.task.monitor must be one of"):
        apply_task_flags(args, task_cfg)


def test_apply_task_flags_rejects_ahi_pointwise_monitor():
    args = argparse.Namespace(label_name="ahi")
    task_cfg = TaskConfig(
        type="classification",
        output_dim=30,
        is_seq=True,
        monitor="val_ahi_pointwise_f1",
        monitor_mod="max",
    )

    with pytest.raises(ValueError, match="finetune.task.monitor must be one of"):
        apply_task_flags(args, task_cfg)


def test_apply_task_flags_rejects_ahi_pointwise_roc_auc_monitor():
    args = argparse.Namespace(label_name="ahi")
    task_cfg = TaskConfig(
        type="classification",
        output_dim=30,
        is_seq=True,
        monitor="val_ahi_pointwise_roc_auc",
        monitor_mod="max",
    )

    with pytest.raises(ValueError, match="finetune.task.monitor must be one of"):
        apply_task_flags(args, task_cfg)


@pytest.mark.parametrize(
    ("label_name", "output_dim", "is_seq", "monitor"),
    [
        ("sex", 2, False, "val_roc_auc"),
        ("stage5", 5, True, "val_f1_macro"),
        ("stage4", 4, True, "val_cohen_kappa"),
    ],
)
def test_apply_task_flags_allows_builtin_classification_imbalance_monitors(
    label_name: str,
    output_dim: int,
    is_seq: bool,
    monitor: str,
):
    args = argparse.Namespace(label_name=label_name)
    task_cfg = TaskConfig(
        type="classification",
        output_dim=output_dim,
        is_seq=is_seq,
        monitor=monitor,
        monitor_mod="max",
    )

    apply_task_flags(args, task_cfg)

    assert args.monitor == monitor


def test_apply_finetune_config_populates_namespace(tmp_path: Path):
    config_path = _write_yaml(tmp_path, _finetune_payload())
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    config_bundle, model_cfg = apply_finetune_config(args)

    assert [c.name for c in model_cfg.channels] == ["eeg", "ecg"]
    assert args.channel_names == ["eeg", "ecg"]
    assert args.channel_input_dims == {"eeg": 4, "ecg": 4}
    assert args.channel_aliases == {}
    assert set(args.data_channel_names) == {"eeg", "ecg"}
    assert args.max_tokens == 4
    assert args.finetune_data_index == Path("index/custom.csv")
    assert args.finetune_preset_path == Path("preset/custom.pkl")
    assert args.data_backend == "npz"
    assert args.kaldi_data_root is None
    assert args.kaldi_manifest is None
    assert args.train_dataset_names == ["train_a"]
    assert args.test_dataset_names == ["test_a"]
    assert args.n_few_shot == 32
    assert args.freeze_backbone_and_insert_lora is True
    assert args.insert_lora is True
    assert args.separate_adapters is True
    assert args.lora_r == 4
    assert args.lora_alpha == 12
    assert args.lora_dropout == 0.15
    assert args.lora_target_modules == ["query", "dense"]
    assert args.lora_use_dora is True
    assert args.freeze_tokenizer is False
    assert args.eval_visualizations is None
    assert args.output_dim == 2
    assert args.is_classification is True
    assert args.is_seq is False
    assert args.class_weights is None
    assert args.pos_weight is None
    assert args.weighted_random_sampler is False
    assert config_bundle.finetune.task is not None


def test_apply_finetune_config_populates_channel_aliases(tmp_path: Path):
    payload = _finetune_payload()
    payload["model"]["channels"][0]["aliases"] = ["psg_eeg", "bcg_eeg"]
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    apply_finetune_config(args)

    assert args.channel_aliases == {"eeg": ["psg_eeg", "bcg_eeg"]}


@pytest.mark.parametrize(
    "module_name",
    [
        "sleep2vec2.common",
        "sleep2expert.common",
    ],
)
def test_variant_apply_finetune_config_populates_lora_namespace(tmp_path: Path, module_name: str):
    apply_config = importlib.import_module(module_name).apply_finetune_config
    config_path = _write_yaml(tmp_path, _finetune_payload())
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    apply_config(args)

    assert args.freeze_backbone_and_insert_lora is True
    assert args.insert_lora is True
    assert args.separate_adapters is True
    assert args.lora_r == 4
    assert args.lora_alpha == 12
    assert args.lora_dropout == 0.15
    assert args.lora_target_modules == ["query", "dense"]
    assert args.lora_use_dora is True


def test_apply_finetune_config_applies_binary_imbalance_knobs(tmp_path: Path):
    payload = _finetune_payload()
    payload["finetune"]["loss"] = {"class_weights": [1.0, 2.436], "pos_weight": None}
    payload["finetune"]["sampler"] = {"weighted_random": True}
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="src_isDep")

    apply_finetune_config(args)

    assert args.class_weights == [1.0, 2.436]
    assert args.pos_weight is None
    assert args.weighted_random_sampler is True


def test_apply_finetune_config_expands_scalar_ahi_pos_weight(tmp_path: Path):
    payload = _finetune_payload()
    payload["finetune"]["task"] = {
        "type": "classification",
        "output_dim": 30,
        "is_seq": True,
        "monitor": "val_ahi_pearson",
        "monitor_mod": "max",
    }
    payload["finetune"]["loss"] = {"class_weights": None, "pos_weight": 2.5}
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="ahi")

    apply_finetune_config(args)

    assert args.class_weights is None
    assert args.pos_weight == [2.5] * 30
    assert args.weighted_random_sampler is False


def test_apply_finetune_config_expands_scalar_multilabel_pos_weight(tmp_path: Path):
    payload = _multilabel_payload()
    payload["finetune"]["loss"] = {"class_weights": None, "pos_weight": 2.5}
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="disease_detection")

    apply_finetune_config(args)

    assert args.class_weights is None
    assert args.pos_weight == [2.5, 2.5, 2.5]
    assert args.multilabel.key_column == "eid"


def test_apply_finetune_config_accepts_multilabel_pos_weight_list(tmp_path: Path):
    payload = _multilabel_payload()
    payload["finetune"]["loss"] = {"class_weights": None, "pos_weight": [1.0, 2.0, 3.0]}
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="disease_detection")

    apply_finetune_config(args)

    assert args.pos_weight == [1.0, 2.0, 3.0]


def test_apply_finetune_config_rejects_class_weights_for_regression(tmp_path: Path):
    payload = _finetune_payload()
    payload["finetune"]["task"] = {
        "type": "regression",
        "output_dim": 1,
        "is_seq": False,
        "monitor": "val_mae",
        "monitor_mod": "min",
    }
    payload["finetune"]["loss"] = {"class_weights": [1.0], "pos_weight": None}
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="class_weights is only supported"):
        apply_finetune_config(args)


def test_apply_finetune_config_rejects_class_weights_length_mismatch(tmp_path: Path):
    payload = _finetune_payload()
    payload["finetune"]["loss"] = {"class_weights": [1.0], "pos_weight": None}
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="class_weights length must match"):
        apply_finetune_config(args)


def test_apply_finetune_config_rejects_pos_weight_for_single_label_classification(tmp_path: Path):
    payload = _finetune_payload()
    payload["finetune"]["loss"] = {"class_weights": None, "pos_weight": 2.0}
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="pos_weight is only supported"):
        apply_finetune_config(args)


def test_apply_finetune_config_rejects_pos_weight_length_mismatch(tmp_path: Path):
    payload = _finetune_payload()
    payload["finetune"]["task"] = {
        "type": "classification",
        "output_dim": 30,
        "is_seq": True,
        "monitor": "val_ahi_pearson",
        "monitor_mod": "max",
    }
    payload["finetune"]["loss"] = {"class_weights": None, "pos_weight": [1.0, 2.0]}
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="ahi")

    with pytest.raises(ValueError, match="pos_weight length must match"):
        apply_finetune_config(args)


def test_apply_finetune_config_rejects_weighted_random_for_sequence_task(tmp_path: Path):
    payload = _finetune_payload()
    payload["finetune"]["task"] = {
        "type": "classification",
        "output_dim": 5,
        "is_seq": True,
        "monitor": "val_accuracy",
        "monitor_mod": "max",
    }
    payload["finetune"]["sampler"] = {"weighted_random": True}
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="stage5")

    with pytest.raises(ValueError, match="weighted_random is only supported"):
        apply_finetune_config(args)


def test_apply_finetune_config_rejects_weighted_random_for_regression(tmp_path: Path):
    payload = _finetune_payload()
    payload["finetune"]["task"] = {
        "type": "regression",
        "output_dim": 1,
        "is_seq": False,
        "monitor": "val_mae",
        "monitor_mod": "min",
    }
    payload["finetune"]["sampler"] = {"weighted_random": True}
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="weighted_random is only supported"):
        apply_finetune_config(args)


def test_apply_finetune_config_populates_kaldi_backend(tmp_path: Path):
    payload = _finetune_payload()
    payload["data"]["finetune_preset_path"] = None
    payload["data"].update(
        {
            "backend": "kaldi",
            "kaldi_data_root": "kaldi/root",
            "kaldi_manifest": "kaldi/root/manifest.json",
        }
    )
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    apply_finetune_config(args)

    assert args.data_backend == "kaldi"
    assert args.kaldi_data_root == Path("kaldi/root")
    assert args.kaldi_manifest == Path("kaldi/root/manifest.json")
    assert args.finetune_preset_path is None


def test_apply_finetune_config_rejects_kaldi_missing_manifest(tmp_path: Path):
    payload = _finetune_payload()
    payload["data"]["finetune_preset_path"] = None
    payload["data"].update({"backend": "kaldi", "kaldi_data_root": "kaldi/root"})
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="Kaldi backend requires explicit kaldi_data_root and kaldi_manifest"):
        apply_finetune_config(args)


def test_apply_finetune_config_rejects_kaldi_preset_path(tmp_path: Path):
    payload = _finetune_payload()
    payload["data"].update(
        {
            "backend": "kaldi",
            "kaldi_data_root": "kaldi/root",
            "kaldi_manifest": "kaldi/root/manifest.json",
        }
    )
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="legacy NPZ preset pickles are unsupported"):
        apply_finetune_config(args)


def test_apply_finetune_config_populates_eval_visualizations(tmp_path: Path):
    payload = _finetune_payload()
    payload["finetune"]["eval_visualizations"] = {
        "enabled": True,
        "stages": ["val", "test"],
        "confusion_matrix": {"enabled": True, "show_raw_counts": True},
        "roc_curve": {"enabled": True},
        "regression_scatter": {"enabled": False},
    }
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    config_bundle, _ = apply_finetune_config(args)

    assert config_bundle.finetune.eval_visualizations is not None
    assert args.eval_visualizations is not None
    assert args.eval_visualizations.enabled is True
    assert args.eval_visualizations.stages == ["val", "test"]
    assert args.eval_visualizations.confusion_matrix.enabled is True
    assert args.eval_visualizations.confusion_matrix.show_raw_counts is True
    assert args.eval_visualizations.roc_curve.enabled is True
    assert args.eval_visualizations.regression_scatter.enabled is False


def test_apply_finetune_config_rejects_data_channel_mismatch(tmp_path: Path):
    payload = _finetune_payload()
    payload["data"]["data_channel_names"] = ["eeg"]
    config_path = _write_yaml(tmp_path, payload)
    args = argparse.Namespace(config=config_path, label_name="custom_target")

    with pytest.raises(ValueError, match="data.data_channel_names in YAML must match model.channels"):
        apply_finetune_config(args)


def test_dump_cli_args_yaml_converts_namespace_and_paths(tmp_path: Path):
    args = argparse.Namespace(
        alpha=1,
        output_path=Path("outputs/run_a"),
        nested={"artifact": Path("artifacts/model.ckpt")},
        values=[Path("x.txt"), 5],
        pair_probs={("breath", "ppg"): 0.4},
    )
    dest = tmp_path / "logs" / "cli_args.yaml"

    written = dump_cli_args_yaml(args, dest)

    assert written == dest
    loaded = yaml.safe_load(dest.read_text())
    assert loaded["alpha"] == 1
    assert loaded["output_path"] == "outputs/run_a"
    assert loaded["nested"]["artifact"] == "artifacts/model.ckpt"
    assert loaded["values"] == ["x.txt", 5]
    assert loaded["pair_probs"]["['breath', 'ppg']"] == 0.4


def test_dump_cli_args_yaml_serializes_eval_visualizations_dataclass(tmp_path: Path):
    args = argparse.Namespace(
        eval_visualizations=EvalVisualizationsConfig(
            enabled=True,
            stages=["val", "test"],
            confusion_matrix=ConfusionMatrixVisualizationConfig(enabled=True, show_raw_counts=True),
            roc_curve=EvalVisualizationPlotConfig(enabled=True),
            regression_scatter=EvalVisualizationPlotConfig(enabled=False),
        )
    )
    dest = tmp_path / "logs" / "cli_args.yaml"

    dump_cli_args_yaml(args, dest)

    loaded = yaml.safe_load(dest.read_text())
    assert loaded["eval_visualizations"]["enabled"] is True
    assert loaded["eval_visualizations"]["stages"] == ["val", "test"]
    assert loaded["eval_visualizations"]["confusion_matrix"]["enabled"] is True
    assert loaded["eval_visualizations"]["confusion_matrix"]["show_raw_counts"] is True
    assert loaded["eval_visualizations"]["roc_curve"]["enabled"] is True
    assert loaded["eval_visualizations"]["regression_scatter"]["enabled"] is False
