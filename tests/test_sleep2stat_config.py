from pathlib import Path

import pytest
import yaml

from sleep2stat.config import load_config


def _minimal_payload() -> dict:
    return {
        "run": {
            "name": "unit",
            "output_dir": "results/sleep2stat/unit",
        },
        "data": {
            "backend": "npz",
            "index": "index.csv",
            "split": ["test"],
            "record_id_columns": ["source", "patient_id"],
        },
        "signals": {
            "channels": {
                "ppg": {
                    "source": "ppg",
                    "sfreq": 100,
                    "kind": "ppg",
                    "input_dim": 3000,
                }
            }
        },
        "analyzers": [
            {
                "name": "stage5_model",
                "type": "sleep2vec_downstream",
                "namespace": "sleep2vec2",
                "label_name": "stage5",
                "config": "configs/sleep2vec2/ppg_stage5_finetune_large.yaml",
                "ckpt_path": "/path/to/stage5.ckpt",
                "input_channels": ["ppg"],
            }
        ],
        "reducers": [
            {
                "name": "stage5_stats",
                "type": "hypnogram_stats",
                "source": "stage5_model",
            }
        ],
        "outputs": {
            "write_global_tables": True,
            "write_per_record": True,
            "compression": "gzip",
        },
    }


def _write_yaml(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "sleep2stat.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_load_config_accepts_minimal_model_first_yaml(tmp_path: Path):
    config = load_config(_write_yaml(tmp_path, _minimal_payload()))

    assert config.run.name == "unit"
    assert config.data.backend == "npz"
    assert config.outputs.global_tables["epoch_alignment"] is False
    assert config.outputs.global_tables["second_alignment"] is False
    assert config.outputs.global_tables["event_alignment"] is True
    assert config.outputs.global_tables["night_stats"] is True
    assert config.signals.channels["ppg"].input_dim == 3000
    assert config.analyzers[0].name == "stage5_model"


def test_load_config_rejects_duplicate_record_id_override(tmp_path: Path):
    payload = _minimal_payload()
    payload["data"]["allow_duplicate_record_ids"] = True

    with pytest.raises(ValueError, match="Unknown sleep2stat config field"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_global_tables_without_per_record_sidecars(tmp_path: Path):
    payload = _minimal_payload()
    payload["outputs"]["write_global_tables"] = True
    payload["outputs"]["write_per_record"] = False

    with pytest.raises(ValueError, match="requires outputs.write_per_record=true"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_accepts_stage_reference_stage_key(tmp_path: Path):
    payload = _minimal_payload()
    payload["analyzers"] = [
        {
            "name": "reference_stage5",
            "type": "npz_stage_reference",
            "label_name": "stage5",
            "stage_key": "stage5",
        }
    ]
    payload["reducers"][0]["source"] = "reference_stage5"

    config = load_config(_write_yaml(tmp_path, payload))

    assert config.analyzers[0].stage_key == "stage5"


def test_load_config_rejects_schema_version(tmp_path: Path):
    payload = _minimal_payload()
    payload["schema_version"] = 1

    with pytest.raises(ValueError, match="Unknown sleep2stat top-level"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_requires_top_level_blocks(tmp_path: Path):
    payload = _minimal_payload()
    del payload["outputs"]

    with pytest.raises(ValueError, match="Missing required sleep2stat config block"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_requires_data_split(tmp_path: Path):
    payload = _minimal_payload()
    del payload["data"]["split"]

    with pytest.raises(ValueError, match="data.split is required"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_accepts_kaldi_backend(tmp_path: Path):
    payload = _minimal_payload()
    payload["data"]["backend"] = "kaldi"
    payload["data"].pop("index")
    payload["data"]["kaldi_data_root"] = "index/kaldi_shhs"
    payload["data"]["kaldi_manifest"] = "manifest.json"

    config = load_config(_write_yaml(tmp_path, payload))

    assert config.data.backend == "kaldi"
    assert config.data.kaldi_data_root == Path("index/kaldi_shhs")
    assert config.data.kaldi_manifest == Path("manifest.json")


def test_load_config_requires_kaldi_paths(tmp_path: Path):
    payload = _minimal_payload()
    payload["data"]["backend"] = "kaldi"
    payload["data"].pop("index")

    with pytest.raises(ValueError, match="data.backend=kaldi requires"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_accepts_yasa_analyzers_and_reducer_alias(tmp_path: Path):
    payload = _minimal_payload()
    payload["signals"]["channels"] = {
        "eeg": {
            "source": "eeg",
            "sfreq": 100,
            "kind": "eeg",
            "input_dim": 3000,
            "mne_name": "EEG",
        }
    }
    payload["analyzers"] = [
        {
            "name": "yasa_stage",
            "type": "yasa_stage",
            "input_channels": ["eeg"],
        },
        {
            "name": "yasa_bandpower",
            "type": "yasa_bandpower",
            "input_channels": ["eeg"],
        },
    ]
    payload["reducers"] = [{"name": "yasa_stats", "type": "yasa_hypnogram_stats", "source": "yasa_stage"}]

    config = load_config(_write_yaml(tmp_path, payload))

    assert config.signals.channels["eeg"].mne_name == "EEG"
    assert [analyzer.type for analyzer in config.analyzers] == ["yasa_stage", "yasa_bandpower"]
    assert config.reducers[0].type == "yasa_hypnogram_stats"


def test_load_config_accepts_v02_path_metadata_and_global_table_controls(tmp_path: Path):
    payload = _minimal_payload()
    payload["data"]["path_base"] = "index_dir"
    payload["data"]["metadata_columns"] = ["age", "sex"]
    payload["outputs"]["global_tables"] = {
        "epoch_alignment": True,
        "second_alignment": False,
        "event_alignment": True,
        "night_stats": True,
    }

    config = load_config(_write_yaml(tmp_path, payload))

    assert config.data.path_base == "index_dir"
    assert config.data.metadata_columns == ["age", "sex"]
    assert config.outputs.global_tables["epoch_alignment"] is True
    assert config.outputs.global_tables["second_alignment"] is False


def test_load_config_rejects_unknown_global_table_name(tmp_path: Path):
    payload = _minimal_payload()
    payload["outputs"]["global_tables"] = {"epoch_predictions": True}

    with pytest.raises(ValueError, match="outputs.global_tables"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_accepts_v02_analyzer_and_reducer_types(tmp_path: Path):
    payload = _minimal_payload()
    payload["signals"]["channels"] = {
        "spo2": {"source": "spo2", "sfreq": 1, "kind": "spo2", "input_dim": 30},
        "eeg": {"source": "eeg", "sfreq": 100, "kind": "eeg", "input_dim": 3000},
    }
    payload["analyzers"] = [
        {
            "name": "spo2_desaturation",
            "type": "spo2_desaturation",
            "input_channels": ["spo2"],
            "drop_thresholds": [3, 4],
            "min_duration_sec": 10,
        },
        {
            "name": "yasa_spindles",
            "type": "yasa_spindles",
            "input_channels": ["eeg"],
            "stage_source": "yasa_stage",
            "stages": ["N2"],
        },
    ]
    payload["reducers"] = [{"name": "spo2_density", "type": "event_density", "source": "spo2_desaturation"}]

    config = load_config(_write_yaml(tmp_path, payload))

    assert [analyzer.type for analyzer in config.analyzers] == ["spo2_desaturation", "yasa_spindles"]
    assert config.analyzers[0].drop_thresholds == [3.0, 4.0]
    assert config.reducers[0].type == "event_density"


def test_load_config_rejects_yasa_with_kaldi_backend(tmp_path: Path):
    payload = _minimal_payload()
    payload["data"]["backend"] = "kaldi"
    payload["data"].pop("index")
    payload["data"]["kaldi_data_root"] = "index/kaldi_shhs"
    payload["data"]["kaldi_manifest"] = "manifest.json"
    payload["signals"]["channels"] = {"eeg": {"source": "eeg", "sfreq": 100, "kind": "eeg", "input_dim": 3000}}
    payload["analyzers"] = [{"name": "yasa_stage", "type": "yasa_stage", "input_channels": ["eeg"]}]
    payload["reducers"] = [{"name": "yasa_stats", "type": "yasa_hypnogram_stats", "source": "yasa_stage"}]

    with pytest.raises(ValueError, match="YASA analyzers require data.backend=npz"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_unknown_nested_fields(tmp_path: Path):
    payload = _minimal_payload()
    payload["analyzers"][0]["batch_siz"] = 4

    with pytest.raises(ValueError, match="Unknown sleep2stat config field"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_unknown_reducer_reference(tmp_path: Path):
    payload = _minimal_payload()
    payload["reducers"][0]["source"] = "missing_model"

    with pytest.raises(ValueError, match="references unknown analyzer"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_unknown_analyzer_and_reducer(tmp_path: Path):
    payload = _minimal_payload()
    payload["analyzers"][0]["type"] = "yasa_sleep_staging"
    with pytest.raises(ValueError, match="Unknown sleep2stat analyzer type"):
        load_config(_write_yaml(tmp_path, payload))

    payload = _minimal_payload()
    payload["reducers"][0]["type"] = "new_stats"
    with pytest.raises(ValueError, match="Unknown sleep2stat reducer type"):
        load_config(_write_yaml(tmp_path, payload))
