from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sex_age_baseline.config import load_config


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload))
    return path


def _cox_payload(tmp_path: Path) -> dict:
    sidecars = _write_survival_sidecars(tmp_path)
    return {
        "model": {
            "name": "sex_age_mlp",
            "features": ["age", "sex"],
            "age": {"transform": "divide", "scale": 100.0, "embedding_dim": 4},
            "sex": {"encoding": "binary", "embedding_dim": 4},
            "head": {"hidden_dim": 8, "dropout": 0.1, "activation": "elu"},
        },
        "data": {
            "backend": "npz",
            "finetune_data_index": str(tmp_path / "index.csv"),
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
            "survival": {"key_column": "eid", **sidecars},
        },
        "outputs": {"prediction_csv": True, "per_disease_metrics_csv": True},
    }


def _multilabel_payload(tmp_path: Path) -> dict:
    sidecars = _write_multilabel_sidecars(tmp_path)
    payload = _cox_payload(tmp_path)
    payload["finetune"] = {
        "task": {
            "type": "multilabel_classification",
            "output_dim": 2,
            "is_seq": False,
            "monitor": "val_macro_auroc",
            "monitor_mod": "max",
        },
        "multilabel": {"key_column": "eid", **sidecars},
        "loss": {"pos_weight": None},
    }
    return payload


def _write_survival_sidecars(tmp_path: Path) -> dict[str, str]:
    disease_columns = tmp_path / "disease_columns.txt"
    event_time = tmp_path / "event_time.csv"
    is_event = tmp_path / "is_event.csv"
    has_label = tmp_path / "has_label.csv"
    disease_columns.write_text("d1\nd2\n")
    header = "eid,d1,d2\n"
    event_time.write_text(header + "001,5,6\n002,3,4\n")
    is_event.write_text(header + "001,1,0\n002,0,1\n")
    has_label.write_text(header + "001,1,1\n002,1,1\n")
    return {
        "disease_columns_index": str(disease_columns),
        "event_time_index": str(event_time),
        "is_event_index": str(is_event),
        "has_label_index": str(has_label),
    }


def _write_multilabel_sidecars(tmp_path: Path) -> dict[str, str]:
    disease_columns = tmp_path / "disease_columns.txt"
    label_index = tmp_path / "disease_label.csv"
    has_label = tmp_path / "has_label.csv"
    disease_columns.write_text("d1\nd2\n")
    header = "eid,d1,d2\n"
    label_index.write_text(header + "001,1,0\n002,0,1\n")
    has_label.write_text(header + "001,1,1\n002,1,1\n")
    return {
        "disease_columns_index": str(disease_columns),
        "label_index": str(label_index),
        "has_label_index": str(has_label),
    }


@pytest.mark.parametrize("path", ["configs/sex_age_baseline/cox.yaml", "configs/sex_age_baseline/multilabel.yaml"])
def test_checked_in_configs_load(path: str):
    cfg = load_config(path)

    assert cfg.model.features == ["age", "sex"]
    assert cfg.data.key_column == "eid"
    assert cfg.data.backend == "npz"


def test_validates_sidecar_output_dim(tmp_path: Path):
    payload = _cox_payload(tmp_path)
    config = _write_yaml(tmp_path / "cox.yaml", payload)

    cfg = load_config(config, validate_sidecars=True)

    assert cfg.finetune.task.output_dim == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["model"].update({"features": ["age"]}),
        lambda payload: payload["finetune"]["task"].update({"type": "regression"}),
        lambda payload: payload["finetune"]["task"].update({"is_seq": True}),
    ],
)
def test_invalid_semantics_fail(tmp_path: Path, mutate):
    payload = _cox_payload(tmp_path)
    mutate(payload)
    config = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError):
        load_config(config)


def test_bad_output_dim_sidecar_mismatch_fails(tmp_path: Path):
    payload = _cox_payload(tmp_path)
    payload["finetune"]["task"]["output_dim"] = 3
    config = _write_yaml(tmp_path / "bad_dim.yaml", payload)

    with pytest.raises(ValueError, match="output_dim"):
        load_config(config, validate_sidecars=True)


def test_multilabel_config_validates_sidecars(tmp_path: Path):
    payload = _multilabel_payload(tmp_path)
    config = _write_yaml(tmp_path / "multilabel.yaml", payload)

    cfg = load_config(config, validate_sidecars=True)

    assert cfg.finetune.multilabel.label_index.endswith("disease_label.csv")


def test_npz_preset_config_loads(tmp_path: Path):
    payload = _cox_payload(tmp_path)
    payload["data"]["finetune_data_index"] = None
    payload["data"]["finetune_preset_path"] = str(tmp_path / "preset.pkl")
    config = _write_yaml(tmp_path / "preset.yaml", payload)

    cfg = load_config(config)

    assert cfg.data.backend == "npz"
    assert cfg.data.finetune_preset_path.endswith("preset.pkl")


def test_kaldi_config_loads(tmp_path: Path):
    payload = _cox_payload(tmp_path)
    payload["data"].update(
        {
            "backend": "kaldi",
            "finetune_data_index": None,
            "finetune_preset_path": None,
            "kaldi_data_root": str(tmp_path / "kaldi"),
            "kaldi_manifest": str(tmp_path / "kaldi" / "manifest.json"),
        }
    )
    config = _write_yaml(tmp_path / "kaldi.yaml", payload)

    cfg = load_config(config)

    assert cfg.data.backend == "kaldi"
    assert cfg.data.kaldi_manifest.endswith("manifest.json")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda payload: payload["data"].update({"finetune_preset_path": "preset.pkl"}), "exactly one"),
        (lambda payload: payload["data"].update({"finetune_data_index": None}), "exactly one"),
        (lambda payload: payload["data"].update({"backend": "bad"}), "data.backend"),
        (
            lambda payload: payload["data"].update(
                {"backend": "kaldi", "finetune_data_index": None, "kaldi_data_root": None}
            ),
            "kaldi_data_root",
        ),
        (
            lambda payload: payload["data"].update(
                {
                    "backend": "kaldi",
                    "finetune_data_index": None,
                    "finetune_preset_path": "preset.pkl",
                    "kaldi_data_root": "/kaldi",
                    "kaldi_manifest": "/kaldi/manifest.json",
                }
            ),
            "must not set",
        ),
    ],
)
def test_backend_input_validation_fails(tmp_path: Path, mutate, match: str):
    payload = _cox_payload(tmp_path)
    mutate(payload)
    config = _write_yaml(tmp_path / "bad_backend.yaml", payload)

    with pytest.raises(ValueError, match=match):
        load_config(config)
