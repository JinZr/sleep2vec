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
            "path_column": "path",
            "duration_column": "duration",
            "split_column": "split",
            "record_id_columns": ["source", "patient_id"],
            "token_sec": 30,
            "max_tokens": 1535,
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


@pytest.mark.parametrize("field", ["path_base", "custom_path_base"])
def test_load_config_rejects_legacy_path_base_fields(tmp_path: Path, field: str):
    payload = _minimal_payload()
    payload["data"][field] = "index_dir"

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
            "stage_key": "stage5",
        }
    ]
    payload["reducers"][0]["source"] = "reference_stage5"

    config = load_config(_write_yaml(tmp_path, payload))

    assert config.analyzers[0].stage_key == "stage5"


def test_load_config_allows_npz_stage_reference_label_name_as_is(tmp_path: Path):
    payload = _minimal_payload()
    payload["analyzers"] = [
        {
            "name": "reference_stage5",
            "type": "npz_stage_reference",
            "stage_key": "stage5",
            "label_name": "unused_stage",
        }
    ]
    payload["reducers"][0]["source"] = "reference_stage5"

    config = load_config(_write_yaml(tmp_path, payload))

    assert config.analyzers[0].stage_key == "stage5"
    assert config.analyzers[0].label_name == "unused_stage"


def test_load_config_rejects_npz_stage_reference_unknown_stage_field(tmp_path: Path):
    payload = _minimal_payload()
    payload["analyzers"] = [
        {
            "name": "reference_stage5",
            "type": "npz_stage_reference",
            "stage_key": "stage5",
            "npz_key": "stage5",
        }
    ]
    payload["reducers"][0]["source"] = "reference_stage5"

    with pytest.raises(ValueError, match="Unknown sleep2stat config field"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_requires_npz_stage_reference_stage_key(tmp_path: Path):
    payload = _minimal_payload()
    payload["analyzers"] = [
        {
            "name": "reference_stage5",
            "type": "npz_stage_reference",
        }
    ]
    payload["reducers"][0]["source"] = "reference_stage5"

    with pytest.raises(ValueError, match="requires stage_key"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_unknown_top_level_field(tmp_path: Path):
    payload = _minimal_payload()
    payload["legacy_field"] = 1

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


@pytest.mark.parametrize(
    "field",
    ["backend", "path_column", "duration_column", "split_column", "token_sec", "max_tokens"],
)
def test_load_config_requires_explicit_data_semantics(tmp_path: Path, field: str):
    payload = _minimal_payload()
    del payload["data"][field]

    with pytest.raises(ValueError, match=f"data missing required field\\(s\\).*{field}"):
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


def test_load_config_accepts_yasa_analyzers_and_hypnogram_stats(tmp_path: Path):
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
            "stage_source": "yasa_stage",
            "outputs": {"by_epoch": True, "by_stage": True, "by_night": True, "relative": True},
        },
    ]
    payload["reducers"] = [{"name": "yasa_stats", "type": "hypnogram_stats", "source": "yasa_stage"}]

    config = load_config(_write_yaml(tmp_path, payload))

    assert config.signals.channels["eeg"].mne_name == "EEG"
    assert [analyzer.type for analyzer in config.analyzers] == ["yasa_stage", "yasa_bandpower"]
    assert config.reducers[0].type == "hypnogram_stats"


def test_load_config_rejects_yasa_hypnogram_stats_alias(tmp_path: Path):
    payload = _minimal_payload()
    payload["analyzers"] = [{"name": "yasa_stage", "type": "npz_stage_reference", "stage_key": "stage5"}]
    payload["reducers"] = [{"name": "yasa_stats", "type": "yasa_hypnogram_stats", "source": "yasa_stage"}]

    with pytest.raises(ValueError, match="Unknown sleep2stat reducer type"):
        load_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize("input_channels", [["eog_loc"], ["eog_loc", "eog_roc", "eog_extra"]])
def test_load_config_rejects_yasa_rem_without_exactly_two_channels(tmp_path: Path, input_channels: list[str]):
    payload = _minimal_payload()
    payload["signals"]["channels"] = {
        "eog_loc": {"source": "eog_loc", "sfreq": 100, "kind": "eog", "input_dim": 3000},
        "eog_roc": {"source": "eog_roc", "sfreq": 100, "kind": "eog", "input_dim": 3000},
        "eog_extra": {"source": "eog_extra", "sfreq": 100, "kind": "eog", "input_dim": 3000},
    }
    payload["analyzers"] = [{"name": "yasa_rem", "type": "yasa_rem", "input_channels": input_channels}]
    payload["reducers"] = []

    with pytest.raises(ValueError, match="requires exactly two EOG"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_yasa_rem_with_non_eog_channel(tmp_path: Path):
    payload = _minimal_payload()
    payload["signals"]["channels"] = {
        "eog_loc": {"source": "eog_loc", "sfreq": 100, "kind": "eog", "input_dim": 3000},
        "emg": {"source": "emg", "sfreq": 100, "kind": "emg", "input_dim": 3000},
    }
    payload["analyzers"] = [{"name": "yasa_rem", "type": "yasa_rem", "input_channels": ["eog_loc", "emg"]}]
    payload["reducers"] = []

    with pytest.raises(ValueError, match="requires exactly two EOG"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_yasa_bandpower_by_stage_without_stage_source(tmp_path: Path):
    payload = _minimal_payload()
    payload["signals"]["channels"]["eeg"] = {"source": "eeg", "sfreq": 100, "kind": "eeg", "input_dim": 3000}
    payload["analyzers"].append(
        {
            "name": "yasa_bandpower",
            "type": "yasa_bandpower",
            "input_channels": ["eeg"],
            "outputs": {"by_epoch": True, "by_stage": True, "by_night": True, "relative": True},
        }
    )

    with pytest.raises(ValueError, match="requires stage_source"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_yasa_bandpower_outputs_stage_source(tmp_path: Path):
    payload = _minimal_payload()
    payload["signals"]["channels"]["eeg"] = {"source": "eeg", "sfreq": 100, "kind": "eeg", "input_dim": 3000}
    payload["analyzers"] = [
        {"name": "yasa_stage", "type": "yasa_stage", "input_channels": ["eeg"]},
        {"name": "other_stage", "type": "yasa_stage", "input_channels": ["eeg"]},
        {
            "name": "yasa_bandpower",
            "type": "yasa_bandpower",
            "input_channels": ["eeg"],
            "stage_source": "yasa_stage",
            "outputs": {
                "stage_source": "other_stage",
                "by_epoch": True,
                "by_stage": True,
                "by_night": True,
                "relative": True,
            },
        },
    ]
    payload["reducers"] = []

    with pytest.raises(ValueError, match="legacy outputs.stage_source"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_accepts_yasa_bandpower_without_stage_source_when_by_stage_false(tmp_path: Path):
    payload = _minimal_payload()
    payload["signals"]["channels"]["eeg"] = {"source": "eeg", "sfreq": 100, "kind": "eeg", "input_dim": 3000}
    payload["analyzers"].append(
        {
            "name": "yasa_bandpower",
            "type": "yasa_bandpower",
            "input_channels": ["eeg"],
            "outputs": {"by_epoch": True, "by_stage": False, "by_night": True, "relative": True},
        }
    )

    config = load_config(_write_yaml(tmp_path, payload))

    assert config.analyzers[-1].outputs["by_stage"] is False


@pytest.mark.parametrize("field", ["by_epoch", "by_stage", "by_night", "relative"])
def test_load_config_rejects_yasa_bandpower_missing_output_mode(tmp_path: Path, field: str):
    payload = _minimal_payload()
    payload["signals"]["channels"]["eeg"] = {"source": "eeg", "sfreq": 100, "kind": "eeg", "input_dim": 3000}
    outputs = {"by_epoch": True, "by_stage": True, "by_night": True, "relative": True}
    del outputs[field]
    payload["analyzers"] = [
        {"name": "yasa_stage", "type": "yasa_stage", "input_channels": ["eeg"]},
        {
            "name": "yasa_bandpower",
            "type": "yasa_bandpower",
            "input_channels": ["eeg"],
            "stage_source": "yasa_stage",
            "outputs": outputs,
        },
    ]
    payload["reducers"] = []

    with pytest.raises(ValueError, match=f"outputs missing required field\\(s\\).*{field}"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_accepts_v02_path_metadata_and_global_table_controls(tmp_path: Path):
    payload = _minimal_payload()
    payload["data"]["metadata_columns"] = ["age", "sex"]
    payload["outputs"]["global_tables"] = {
        "epoch_alignment": True,
        "second_alignment": False,
        "event_alignment": True,
        "night_stats": True,
    }

    config = load_config(_write_yaml(tmp_path, payload))

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
            "name": "yasa_stage",
            "type": "yasa_stage",
            "input_channels": ["eeg"],
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

    assert [analyzer.type for analyzer in config.analyzers] == ["spo2_desaturation", "yasa_stage", "yasa_spindles"]
    assert config.analyzers[0].drop_thresholds == [3.0, 4.0]
    assert config.reducers[0].type == "event_density"


def test_load_config_rejects_spo2_source_field(tmp_path: Path):
    payload = _minimal_payload()
    payload["signals"]["channels"] = {"spo2": {"source": "spo2", "sfreq": 1, "kind": "spo2", "input_dim": 30}}
    payload["analyzers"] = [{"name": "spo2_summary", "type": "spo2_summary", "spo2_source": "spo2"}]
    payload["reducers"] = []

    with pytest.raises(ValueError, match="Unknown sleep2stat config field"):
        load_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize("field", ["drop_thresholds", "min_duration_sec"])
def test_load_config_rejects_spo2_desaturation_missing_required_field(tmp_path: Path, field: str):
    payload = _minimal_payload()
    payload["signals"]["channels"] = {"spo2": {"source": "spo2", "sfreq": 1, "kind": "spo2", "input_dim": 30}}
    analyzer = {
        "name": "spo2_desaturation",
        "type": "spo2_desaturation",
        "input_channels": ["spo2"],
        "drop_thresholds": [3, 4],
        "min_duration_sec": 10,
    }
    del analyzer[field]
    payload["analyzers"] = [analyzer]
    payload["reducers"] = []

    with pytest.raises(ValueError, match=f"missing required field\\(s\\).*{field}"):
        load_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize("field", ["min_value", "max_value", "max_drop_per_sec"])
def test_load_config_rejects_spo2_artifact_legacy_aliases(tmp_path: Path, field: str):
    payload = _minimal_payload()
    payload["signals"]["channels"] = {"spo2": {"source": "spo2", "sfreq": 1, "kind": "spo2", "input_dim": 30}}
    payload["analyzers"] = [
        {
            "name": "spo2_summary",
            "type": "spo2_summary",
            "input_channels": ["spo2"],
            "artifact": {field: 90},
        }
    ]
    payload["reducers"] = []

    with pytest.raises(ValueError, match="legacy artifact field"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_yasa_with_kaldi_backend(tmp_path: Path):
    payload = _minimal_payload()
    payload["data"]["backend"] = "kaldi"
    payload["data"].pop("index")
    payload["data"]["kaldi_data_root"] = "index/kaldi_shhs"
    payload["data"]["kaldi_manifest"] = "manifest.json"
    payload["signals"]["channels"] = {"eeg": {"source": "eeg", "sfreq": 100, "kind": "eeg", "input_dim": 3000}}
    payload["analyzers"] = [{"name": "yasa_stage", "type": "yasa_stage", "input_channels": ["eeg"]}]
    payload["reducers"] = [{"name": "yasa_stats", "type": "hypnogram_stats", "source": "yasa_stage"}]

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


def test_load_config_rejects_duplicate_analyzer_names(tmp_path: Path):
    payload = _minimal_payload()
    payload["analyzers"].append(dict(payload["analyzers"][0]))

    with pytest.raises(ValueError, match="duplicate analyzer name"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_duplicate_reducer_names(tmp_path: Path):
    payload = _minimal_payload()
    payload["reducers"].append(dict(payload["reducers"][0]))

    with pytest.raises(ValueError, match="duplicate reducer name"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_unknown_analyzer_stage_source(tmp_path: Path):
    payload = _minimal_payload()
    payload["signals"]["channels"]["eeg"] = {"source": "eeg", "sfreq": 100, "kind": "eeg", "input_dim": 3000}
    payload["analyzers"].append(
        {
            "name": "yasa_spindles",
            "type": "yasa_spindles",
            "input_channels": ["eeg"],
            "stage_source": "missing_stage",
            "stages": ["N2"],
        }
    )

    with pytest.raises(ValueError, match="enabled earlier analyzer"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_disabled_analyzer_stage_source(tmp_path: Path):
    payload = _minimal_payload()
    payload["signals"]["channels"]["eeg"] = {"source": "eeg", "sfreq": 100, "kind": "eeg", "input_dim": 3000}
    payload["analyzers"] = [
        {"name": "yasa_stage", "type": "yasa_stage", "input_channels": ["eeg"], "enabled": False},
        {
            "name": "yasa_spindles",
            "type": "yasa_spindles",
            "input_channels": ["eeg"],
            "stage_source": "yasa_stage",
            "stages": ["N2"],
        },
    ]
    payload["reducers"] = []

    with pytest.raises(ValueError, match="enabled earlier analyzer"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_later_analyzer_stage_source(tmp_path: Path):
    payload = _minimal_payload()
    payload["signals"]["channels"]["eeg"] = {"source": "eeg", "sfreq": 100, "kind": "eeg", "input_dim": 3000}
    payload["analyzers"] = [
        {
            "name": "yasa_spindles",
            "type": "yasa_spindles",
            "input_channels": ["eeg"],
            "stage_source": "yasa_stage",
            "stages": ["N2"],
        },
        {"name": "yasa_stage", "type": "yasa_stage", "input_channels": ["eeg"]},
    ]
    payload["reducers"] = []

    with pytest.raises(ValueError, match="enabled earlier analyzer"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_self_referencing_analyzer_stage_source(tmp_path: Path):
    payload = _minimal_payload()
    payload["signals"]["channels"]["eeg"] = {"source": "eeg", "sfreq": 100, "kind": "eeg", "input_dim": 3000}
    payload["analyzers"] = [
        {
            "name": "yasa_spindles",
            "type": "yasa_spindles",
            "input_channels": ["eeg"],
            "stage_source": "yasa_spindles",
            "stages": ["N2"],
        }
    ]
    payload["reducers"] = []

    with pytest.raises(ValueError, match="enabled earlier analyzer"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_later_yasa_bandpower_stage_source(tmp_path: Path):
    payload = _minimal_payload()
    payload["signals"]["channels"]["eeg"] = {"source": "eeg", "sfreq": 100, "kind": "eeg", "input_dim": 3000}
    payload["analyzers"] = [
        {
            "name": "yasa_bandpower",
            "type": "yasa_bandpower",
            "input_channels": ["eeg"],
            "stage_source": "yasa_stage",
            "outputs": {"by_epoch": True, "by_stage": True, "by_night": True, "relative": True},
        },
        {"name": "yasa_stage", "type": "yasa_stage", "input_channels": ["eeg"]},
    ]
    payload["reducers"] = []

    with pytest.raises(ValueError, match="enabled earlier analyzer"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_unknown_yasa_stage_filter(tmp_path: Path):
    payload = _minimal_payload()
    payload["signals"]["channels"]["eeg"] = {"source": "eeg", "sfreq": 100, "kind": "eeg", "input_dim": 3000}
    payload["analyzers"] = [
        {"name": "yasa_stage", "type": "yasa_stage", "input_channels": ["eeg"]},
        {
            "name": "yasa_spindles",
            "type": "yasa_spindles",
            "input_channels": ["eeg"],
            "stage_source": "yasa_stage",
            "stages": ["N22", "REMM"],
        },
    ]
    payload["reducers"] = []

    with pytest.raises(ValueError, match="unsupported YASA stage filter"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_downstream_postprocess_threshold(tmp_path: Path):
    payload = _minimal_payload()
    payload["analyzers"][0]["label_name"] = "ahi"
    payload["analyzers"][0]["postprocess"] = {"threshold": {"source": "checkpoint", "value": None}}

    with pytest.raises(ValueError, match="legacy postprocess.threshold"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_allows_runtime_threshold_value_as_is(tmp_path: Path):
    payload = _minimal_payload()
    payload["analyzers"][0]["label_name"] = "ahi"
    payload["analyzers"][0]["threshold"] = {"value": 0.5}
    payload["analyzers"][0]["postprocess"] = {
        "min_event_duration_sec": 10,
        "merge_tolerance_sec": 3,
        "output_second_alignment": True,
        "output_event_alignment": True,
    }

    config = load_config(_write_yaml(tmp_path, payload))

    assert config.analyzers[0].threshold == {"value": 0.5}


def test_load_config_rejects_downstream_thresholds_field(tmp_path: Path):
    payload = _minimal_payload()
    payload["analyzers"][0]["label_name"] = "ahi"
    payload["analyzers"][0]["thresholds"] = [0.3, 0.5]

    with pytest.raises(ValueError, match="Unknown sleep2stat config field"):
        load_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    "field",
    ["min_event_duration_sec", "merge_tolerance_sec", "output_second_alignment", "output_event_alignment"],
)
def test_load_config_rejects_ahi_missing_postprocess_field(tmp_path: Path, field: str):
    payload = _minimal_payload()
    payload["analyzers"][0]["label_name"] = "ahi"
    postprocess = {
        "min_event_duration_sec": 10,
        "merge_tolerance_sec": 3,
        "output_second_alignment": True,
        "output_event_alignment": True,
    }
    del postprocess[field]
    payload["analyzers"][0]["postprocess"] = postprocess

    with pytest.raises(ValueError, match=f"AHI postprocess missing required field\\(s\\).*{field}"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_later_ahi_denominator_stage_source(tmp_path: Path):
    payload = _minimal_payload()
    stage_analyzer = dict(payload["analyzers"][0])
    payload["analyzers"] = [
        {
            "name": "ahi_model",
            "type": "sleep2vec_downstream",
            "namespace": "sleep2vec2",
            "label_name": "ahi",
            "config": "configs/sleep2vec2/ppg_ahi_finetune_large.yaml",
            "ckpt_path": "/path/to/ahi.ckpt",
            "input_channels": ["ppg"],
            "postprocess": {
                "min_event_duration_sec": 10,
                "merge_tolerance_sec": 3,
                "denominator_stage_source": "stage5_model",
                "output_second_alignment": True,
                "output_event_alignment": True,
            },
        },
        stage_analyzer,
    ]

    with pytest.raises(
        ValueError, match="postprocess.denominator_stage_source must reference an enabled earlier analyzer"
    ):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_enabled_reducer_targeting_disabled_analyzer(tmp_path: Path):
    payload = _minimal_payload()
    payload["analyzers"][0]["enabled"] = False

    with pytest.raises(ValueError, match="references disabled analyzer"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_allows_disabled_reducer_targeting_disabled_analyzer(tmp_path: Path):
    payload = _minimal_payload()
    payload["analyzers"][0]["enabled"] = False
    payload["reducers"][0]["enabled"] = False

    config = load_config(_write_yaml(tmp_path, payload))

    assert config.analyzers[0].enabled is False
    assert config.reducers[0].enabled is False


def test_load_config_rejects_unknown_analyzer_and_reducer(tmp_path: Path):
    payload = _minimal_payload()
    payload["analyzers"][0]["type"] = "yasa_sleep_staging"
    with pytest.raises(ValueError, match="Unknown sleep2stat analyzer type"):
        load_config(_write_yaml(tmp_path, payload))

    payload = _minimal_payload()
    payload["reducers"][0]["type"] = "new_stats"
    with pytest.raises(ValueError, match="Unknown sleep2stat reducer type"):
        load_config(_write_yaml(tmp_path, payload))
