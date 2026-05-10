from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

kaldi_native_io = pytest.importorskip("kaldi_native_io")

from data.psg_pretrain_dataset import _build_channel_registry
from data.utils import load_npz
from preprocess.convert_npz_to_kaldi import convert, parse_args


def _write_config(tmp_path: Path, channel_dims: dict[str, int], preset_build: dict | None = None) -> Path:
    path = tmp_path / "config.yaml"
    payload = {
        "model": {"channels": [{"name": name, "input_dim": input_dim} for name, input_dim in channel_dims.items()]}
    }
    if preset_build is not None:
        payload["preset_build"] = preset_build
    path.write_text(yaml.safe_dump(payload))
    return path


def _scp_keys(path: Path) -> list[str]:
    return [line.split(maxsplit=1)[0] for line in path.read_text().splitlines() if line.strip()]


def _read_matrix(scp_path: Path, key: str) -> np.ndarray:
    with kaldi_native_io.RandomAccessFloatMatrixReader(f"scp:{scp_path}") as reader:
        return np.asarray(reader[key], dtype=np.float32)


def test_converter_roundtrip_writes_manifest_and_matching_scp(tmp_path: Path):
    config_path = _write_config(tmp_path, {"eeg": 4, "ppg": 8})
    actual_root = tmp_path / "actual"
    original_root = tmp_path / "original"
    actual_root.mkdir()
    npz_path = actual_root / "sample.npz"
    original_npz_path = original_root / "sample.npz"
    np.savez(
        npz_path,
        eeg=np.arange(16, dtype=np.float32),
        ppg=np.arange(32, dtype=np.float32),
        stage5=np.arange(4, dtype=np.float32),
        ah_event=np.arange(120, dtype=np.float32),
        ahi=np.asarray(7.0, dtype=np.float32),
        tst=np.asarray(33.0, dtype=np.float32),
    )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(original_npz_path),
                "source": "original_source",
                "dataset": "mesa",
                "split": "train",
                "duration": 120,
                "session_id": "night 1",
                "age": 50,
                "sex": 1,
                "eeg_mask": 1,
                "ppg_mask": 1,
                "stage_mask": 1,
                "ah_event_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    output_dir = tmp_path / "kaldi"
    convert(
        parse_args(
            [
                "--index",
                str(index_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--max-tokens",
                "2",
                "--stride-tokens",
                "2",
                "--token-sec",
                "30",
                "--channels-from-config",
                "--extra-channels",
                "stage5",
                "ahi",
                "--source-field",
                "dataset",
                "--path-prefix-map",
                f"{original_root}={actual_root}",
            ]
        )
    )

    manifest = pd.read_csv(output_dir / "manifests" / "train.csv", low_memory=False)
    assert manifest["sample_key"].tolist() == [
        "mesa_night_1_000000_000002",
        "mesa_night_1_000002_000004",
    ]
    assert manifest.loc[0, "path"] == str(original_npz_path)
    assert manifest.loc[0, "source"] == "original_source"
    assert manifest.loc[0, "sample_source"] == "mesa"
    assert manifest.loc[0, "ahi"] == 7.0
    assert manifest.loc[0, "tst"] == 33.0
    assert json.loads(manifest.loc[0, "available_channels"]) == ["eeg", "ppg", "stage5", "ahi"]

    for channel in ["eeg", "ppg", "stage5", "ahi"]:
        assert _scp_keys(output_dir / "channels" / "train" / f"{channel}.scp") == manifest["sample_key"].tolist()

    registry = _build_channel_registry(
        channel_names=["eeg", "ppg", "stage5", "ahi"],
        channel_input_dims={"eeg": 4, "ppg": 8, "stage5": 1, "ahi": 30},
        mask_rate=0.0,
    )
    with load_npz(str(npz_path)) as npz:
        expected_eeg = registry["eeg"][1](registry["eeg"][0](npz, 0, 2)).numpy()
        expected_ppg = registry["ppg"][1](registry["ppg"][0](npz, 0, 2)).numpy()
        expected_stage5 = registry["stage5"][1](registry["stage5"][0](npz, 0, 2)).numpy()
        expected_ahi = registry["ahi"][1](registry["ahi"][0](npz, 0, 2)).numpy()

    key = "mesa_night_1_000000_000002"
    np.testing.assert_array_equal(_read_matrix(output_dir / "channels" / "train" / "eeg.scp", key), expected_eeg)
    np.testing.assert_array_equal(_read_matrix(output_dir / "channels" / "train" / "ppg.scp", key), expected_ppg)
    np.testing.assert_array_equal(_read_matrix(output_dir / "channels" / "train" / "stage5.scp", key), expected_stage5)
    np.testing.assert_array_equal(_read_matrix(output_dir / "channels" / "train" / "ahi.scp", key), expected_ahi)

    manifest_json = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_json["format_version"] == 2
    assert manifest_json["backend"] == "kaldi_native_io"
    assert manifest_json["token_sec"] == 30
    assert manifest_json["max_tokens"] == 2
    assert manifest_json["stride_tokens"] == 2
    assert manifest_json["splits"]["train"]["manifest"] == "manifests/train.csv"
    assert manifest_json["splits"]["train"]["channels"]["eeg"] == {"input_dim": 4, "scp": "channels/train/eeg.scp"}
    assert manifest_json["source_index"] == [str(index_path)]


def test_converter_writes_split_specific_manifests_and_sorted_scps(tmp_path: Path):
    config_path = _write_config(tmp_path, {"eeg": 4})
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, eeg=np.arange(8, dtype=np.float32))
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 60,
                "session_id": "z",
                "eeg_mask": 1,
            },
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "val",
                "duration": 60,
                "session_id": "b",
                "eeg_mask": 1,
            },
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 60,
                "session_id": "a",
                "eeg_mask": 1,
            },
        ]
    ).to_csv(index_path, index=False)

    output_dir = tmp_path / "kaldi"
    convert(
        parse_args(
            [
                "--index",
                str(index_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--max-tokens",
                "2",
                "--channels-from-config",
            ]
        )
    )

    train_manifest_path = output_dir / "manifests" / "train.csv"
    val_manifest_path = output_dir / "manifests" / "val.csv"
    train_scp_path = output_dir / "channels" / "train" / "eeg.scp"
    val_scp_path = output_dir / "channels" / "val" / "eeg.scp"
    assert train_manifest_path.exists()
    assert val_manifest_path.exists()
    assert train_scp_path.exists()
    assert val_scp_path.exists()
    assert (output_dir / "manifest.json").exists()
    assert not (output_dir / "manifest.csv").exists()

    train_manifest = pd.read_csv(train_manifest_path, low_memory=False)
    val_manifest = pd.read_csv(val_manifest_path, low_memory=False)
    assert train_manifest["split"].tolist() == ["train", "train"]
    assert val_manifest["split"].tolist() == ["val"]
    assert _scp_keys(train_scp_path) == sorted(train_manifest["sample_key"].tolist())
    assert _scp_keys(val_scp_path) == sorted(val_manifest["sample_key"].tolist())

    manifest_json = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_json["format_version"] == 2
    assert manifest_json["splits"]["train"]["manifest"] == "manifests/train.csv"
    assert manifest_json["splits"]["train"]["channels"]["eeg"] == {
        "input_dim": 4,
        "scp": "channels/train/eeg.scp",
    }
    assert manifest_json["splits"]["val"]["manifest"] == "manifests/val.csv"
    assert manifest_json["splits"]["val"]["channels"]["eeg"] == {
        "input_dim": 4,
        "scp": "channels/val/eeg.scp",
    }


def test_converter_rejects_sanitized_split_directory_collisions(tmp_path: Path):
    config_path = _write_config(tmp_path, {"eeg": 4})
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, eeg=np.arange(8, dtype=np.float32))
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "fold/1",
                "duration": 60,
                "session_id": "s1",
                "eeg_mask": 1,
            },
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "fold_1",
                "duration": 60,
                "session_id": "s2",
                "eeg_mask": 1,
            },
        ]
    ).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="both map to directory 'fold_1'"):
        convert(
            parse_args(
                [
                    "--index",
                    str(index_path),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(tmp_path / "kaldi"),
                    "--max-tokens",
                    "2",
                    "--channels-from-config",
                ]
            )
        )


def test_converter_allow_missing_channels_keeps_partial_samples(tmp_path: Path):
    config_path = _write_config(tmp_path, {"eeg": 4, "ppg": 4})
    npz_path = tmp_path / "sample.npz"
    np.savez(
        npz_path,
        eeg=np.arange(8, dtype=np.float32),
        stage5=np.arange(2, dtype=np.float32),
    )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "source": np.nan,
                "dataset": "mesa",
                "split": "train",
                "duration": 60,
                "session_id": "s1",
                "eeg_mask": 1,
                "ppg_mask": 0,
                "stage_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    strict_args = parse_args(
        [
            "--index",
            str(index_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "strict"),
            "--max-tokens",
            "2",
            "--channels-from-config",
            "--extra-channels",
            "stage5",
        ]
    )
    with pytest.raises(ValueError, match="No samples satisfied"):
        convert(strict_args)

    output_dir = tmp_path / "allow_missing"
    convert(
        parse_args(
            [
                "--index",
                str(index_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--max-tokens",
                "2",
                "--channels-from-config",
                "--extra-channels",
                "stage5",
                "--allow-missing-channels",
                "--min-channels",
                "2",
            ]
        )
    )

    manifest = pd.read_csv(output_dir / "manifests" / "train.csv", low_memory=False)
    assert manifest.loc[0, "source"] == "mesa"
    assert manifest.loc[0, "sample_source"] == "mesa"
    assert json.loads(manifest.loc[0, "available_channels"]) == ["eeg", "stage5"]
    assert _scp_keys(output_dir / "channels" / "train" / "eeg.scp") == ["mesa_s1_000000_000002"]
    assert _scp_keys(output_dir / "channels" / "train" / "stage5.scp") == ["mesa_s1_000000_000002"]
    assert _scp_keys(output_dir / "channels" / "train" / "ppg.scp") == []


def test_converter_trims_one_token_channel_length_difference(tmp_path: Path):
    config_path = _write_config(tmp_path, {"eeg": 4, "ppg": 4})
    npz_path = tmp_path / "sample.npz"
    np.savez(
        npz_path,
        eeg=np.arange(12, dtype=np.float32),
        ppg=np.arange(8, dtype=np.float32),
    )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 90,
                "session_id": "s1",
                "eeg_mask": 1,
                "ppg_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    output_dir = tmp_path / "kaldi"
    convert(
        parse_args(
            [
                "--index",
                str(index_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--max-tokens",
                "3",
                "--channels-from-config",
            ]
        )
    )

    manifest = pd.read_csv(output_dir / "manifests" / "train.csv", low_memory=False)
    key = "mesa_s1_000000_000003"
    assert manifest.loc[0, "sample_key"] == key
    assert manifest.loc[0, "token_start"] == 0
    assert manifest.loc[0, "token_end"] == 2
    assert manifest.loc[0, "num_tokens"] == 2
    assert _read_matrix(output_dir / "channels" / "train" / "eeg.scp", key).shape == (2, 4)
    assert _read_matrix(output_dir / "channels" / "train" / "ppg.scp", key).shape == (2, 4)


def test_converter_skips_zero_length_trimmed_window(tmp_path: Path):
    config_path = _write_config(tmp_path, {"eeg": 4, "ppg": 4})
    npz_path = tmp_path / "sample.npz"
    np.savez(
        npz_path,
        eeg=np.arange(8, dtype=np.float32),
        ppg=np.arange(4, dtype=np.float32),
    )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 60,
                "session_id": "s1",
                "eeg_mask": 1,
                "ppg_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    output_dir = tmp_path / "kaldi"
    convert(
        parse_args(
            [
                "--index",
                str(index_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--max-tokens",
                "1",
                "--channels-from-config",
            ]
        )
    )

    manifest = pd.read_csv(output_dir / "manifests" / "train.csv", low_memory=False)
    key = "mesa_s1_000000_000001"
    assert manifest["sample_key"].tolist() == [key]
    assert manifest["num_tokens"].tolist() == [1]
    assert _scp_keys(output_dir / "channels" / "train" / "eeg.scp") == [key]
    assert _scp_keys(output_dir / "channels" / "train" / "ppg.scp") == [key]


def test_converter_rejects_channel_length_difference_greater_than_one(tmp_path: Path):
    config_path = _write_config(tmp_path, {"eeg": 4, "ppg": 4})
    npz_path = tmp_path / "sample.npz"
    np.savez(
        npz_path,
        eeg=np.arange(16, dtype=np.float32),
        ppg=np.arange(8, dtype=np.float32),
    )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 120,
                "session_id": "s1",
                "eeg_mask": 1,
                "ppg_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="differing by more than one"):
        convert(
            parse_args(
                [
                    "--index",
                    str(index_path),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(tmp_path / "kaldi"),
                    "--max-tokens",
                    "4",
                    "--channels-from-config",
                ]
            )
        )


def test_converter_honors_preset_build_required_channels(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {"ppg": 8},
        preset_build={"required_channels": ["ppg", "stage5"], "min_channels": 2},
    )
    npz_path = tmp_path / "sample.npz"
    np.savez(
        npz_path,
        ppg=np.arange(16, dtype=np.float32),
        stage5=np.arange(2, dtype=np.float32),
    )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 60,
                "session_id": "s1",
                "ppg_mask": 1,
                "stage_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    output_dir = tmp_path / "kaldi"
    convert(
        parse_args(
            [
                "--index",
                str(index_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--max-tokens",
                "2",
            ]
        )
    )

    manifest = pd.read_csv(output_dir / "manifests" / "train.csv", low_memory=False)
    assert json.loads(manifest.loc[0, "available_channels"]) == ["ppg", "stage5"]
    manifest_json = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_json["splits"]["train"]["channels"]["stage5"] == {
        "input_dim": 1,
        "scp": "channels/train/stage5.scp",
    }
    assert _scp_keys(output_dir / "channels" / "train" / "stage5.scp") == ["mesa_s1_000000_000002"]


def test_converter_auto_adds_stage5_when_preset_build_requires_ahi(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {"ppg": 8},
        preset_build={"required_channels": ["ppg", "ahi"], "min_channels": 2},
    )
    npz_path = tmp_path / "sample.npz"
    np.savez(
        npz_path,
        ppg=np.arange(16, dtype=np.float32),
        stage5=np.arange(2, dtype=np.float32),
        ah_event=np.arange(60, dtype=np.float32),
        ahi=np.asarray(7.0, dtype=np.float32),
        tst=np.asarray(33.0, dtype=np.float32),
    )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 60,
                "session_id": "s1",
                "ppg_mask": 1,
                "stage_mask": 1,
                "ah_event_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    output_dir = tmp_path / "kaldi"
    convert(
        parse_args(
            [
                "--index",
                str(index_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--max-tokens",
                "2",
            ]
        )
    )

    manifest = pd.read_csv(output_dir / "manifests" / "train.csv", low_memory=False)
    assert json.loads(manifest.loc[0, "available_channels"]) == ["ppg", "ahi", "stage5"]
    manifest_json = json.loads((output_dir / "manifest.json").read_text())
    assert manifest_json["splits"]["train"]["channels"]["ahi"] == {
        "input_dim": 30,
        "scp": "channels/train/ahi.scp",
    }
    assert manifest_json["splits"]["train"]["channels"]["stage5"] == {
        "input_dim": 1,
        "scp": "channels/train/stage5.scp",
    }
    assert _scp_keys(output_dir / "channels" / "train" / "stage5.scp") == ["mesa_s1_000000_000002"]


def test_converter_uses_preset_build_min_channels_for_partial_samples(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        {"ppg": 8},
        preset_build={"required_channels": ["ppg", "stage5"], "min_channels": 2},
    )
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, ppg=np.arange(16, dtype=np.float32))
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 60,
                "session_id": "s1",
                "ppg_mask": 1,
                "stage_mask": 0,
            }
        ]
    ).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="No samples satisfied"):
        convert(
            parse_args(
                [
                    "--index",
                    str(index_path),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(tmp_path / "kaldi"),
                    "--max-tokens",
                    "2",
                    "--allow-missing-channels",
                    "--min-channels",
                    "1",
                ]
            )
        )


def test_converter_rejects_missing_split_column(tmp_path: Path):
    config_path = _write_config(tmp_path, {"eeg": 4})
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, eeg=np.arange(8, dtype=np.float32))
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "duration": 60,
                "session_id": "s1",
                "eeg_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="split"):
        convert(
            parse_args(
                [
                    "--index",
                    str(index_path),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(tmp_path / "kaldi"),
                    "--max-tokens",
                    "2",
                    "--channels-from-config",
                ]
            )
        )


def test_converter_rejects_duplicate_sample_keys(tmp_path: Path):
    config_path = _write_config(tmp_path, {"eeg": 4})
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, eeg=np.arange(8, dtype=np.float32))
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 60,
                "session_id": "s1",
                "eeg_mask": 1,
            },
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 60,
                "session_id": "s1",
                "eeg_mask": 1,
            },
        ]
    ).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="Duplicate Kaldi sample_key"):
        convert(
            parse_args(
                [
                    "--index",
                    str(index_path),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(tmp_path / "kaldi"),
                    "--max-tokens",
                    "2",
                    "--channels-from-config",
                ]
            )
        )


def test_converter_rejects_rank3_tokenized_channel(tmp_path: Path):
    config_path = _write_config(tmp_path, {"eeg": 4})
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, eeg=np.arange(16, dtype=np.float32).reshape(2, 8))
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "mesa",
                "split": "train",
                "duration": 60,
                "session_id": "s1",
                "eeg_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="tokenized to rank 3"):
        convert(
            parse_args(
                [
                    "--index",
                    str(index_path),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(tmp_path / "kaldi"),
                    "--max-tokens",
                    "2",
                    "--channels-from-config",
                ]
            )
        )
