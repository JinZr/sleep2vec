from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

kaldi_native_io = pytest.importorskip("kaldi_native_io")

from data.kaldi_psg_dataset import KaldiPSGDataset
from data.psg_pretrain_dataset import PSGPretrainDataset
from data.utils import AROUSAL_INDEX_KEYS, AROUSAL_METADATA_KEYS
from preprocess.convert_npz_to_kaldi import convert, parse_args


def _write_source(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    events = np.zeros((60, 4), dtype=np.float32)
    events[0, [0, 1]] = 1.0
    events[29, 2] = 1.0
    events[30, 3] = 1.0
    npz_path = tmp_path / "sample.npz"
    payload = {
        "arousal_event": events,
        "stage5": np.asarray([1.0, 2.0], dtype=np.float32),
        "tst": np.asarray(4.0, dtype=np.float32),
    }
    payload.update({key: np.asarray(index + 2.0, dtype=np.float32) for index, key in enumerate(AROUSAL_INDEX_KEYS)})
    np.savez(npz_path, **payload)

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(npz_path),
                "dataset": "center-a",
                "source": "center-a",
                "split": "train",
                "duration": 60,
                "session_id": "record-1",
                "arousal_event_mask": 1,
                "stage_mask": 1,
            }
        ]
    ).to_csv(index_path, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"model": {"channels": [{"name": "ppg", "input_dim": 8}]}}))
    return config_path, index_path, events


def _convert(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    config_path, index_path, events = _write_source(tmp_path)
    output_dir = tmp_path / "kaldi"
    args = parse_args(
        [
            "--index",
            str(index_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--max-tokens",
            "2",
            "--num-workers",
            "1",
            "--extra-channels",
            "arousal",
        ]
    )
    assert args.compress_ark is True
    manifest_path = convert(args)
    return output_dir, manifest_path, events


def _dataset_kwargs() -> dict:
    return {
        "channel_names": ["arousal", "stage5"],
        "channel_input_dims": {"arousal": 120, "stage5": 1},
        "split": ["train"],
        "max_tokens": 2,
        "mask_rate": 0.0,
        "meta_data_names": list(AROUSAL_METADATA_KEYS),
        "meta_data_regression_names": list(AROUSAL_METADATA_KEYS),
        "randomly_select_channels": False,
        "allow_missing_channels": False,
        "is_train_set": False,
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 0,
    }


def test_arousal_kaldi_roundtrip_matches_npz_dataset(tmp_path: Path) -> None:
    output_dir, manifest_path, events = _convert(tmp_path)
    manifest_json = json.loads(manifest_path.read_text())
    arousal_spec = manifest_json["splits"]["train"]["channels"]["arousal"]
    assert manifest_json["token_sec"] == 30
    assert arousal_spec["input_dim"] == 120
    assert arousal_spec["ark_storage"] == "float_matrix"

    sample_key = "center-a_record-1_000000_000002"
    with kaldi_native_io.RandomAccessFloatMatrixReader(
        f"scp:{output_dir / 'channels' / 'train' / 'arousal.scp'}"
    ) as reader:
        matrix = np.asarray(reader[sample_key], dtype=np.float32).copy()
    np.testing.assert_array_equal(matrix, events.reshape(2, 120))

    manifest_frame = pd.read_csv(output_dir / "manifests" / "train.csv")
    for index, key in enumerate(AROUSAL_INDEX_KEYS):
        assert manifest_frame.loc[0, key] == index + 2.0
    assert manifest_frame.loc[0, "tst"] == 4.0

    npz_dataset = PSGPretrainDataset(
        save_preset_path=None,
        load_preset_path=None,
        index=str(tmp_path / "index.csv"),
        **_dataset_kwargs(),
    )
    kaldi_dataset = KaldiPSGDataset(
        kaldi_data_root=output_dir,
        manifest=manifest_path,
        **_dataset_kwargs(),
    )

    npz_batch = next(iter(npz_dataset.dataloader()))
    kaldi_batch = next(iter(kaldi_dataset.dataloader()))

    assert torch.equal(npz_batch["tokens"]["arousal"], kaldi_batch["tokens"]["arousal"])
    assert torch.equal(npz_batch["tokens"]["stage5"], kaldi_batch["tokens"]["stage5"])
    assert torch.equal(npz_batch["token_start"], kaldi_batch["token_start"])
    assert npz_batch["metadata"]["path"] == kaldi_batch["metadata"]["path"]
    for key in AROUSAL_METADATA_KEYS:
        assert torch.equal(npz_batch["metadata"][key], kaldi_batch["metadata"][key])


def test_arousal_kaldi_loader_rejects_invalid_values_and_scalars(tmp_path: Path) -> None:
    output_dir, manifest_path, _ = _convert(tmp_path)
    manifest_csv = output_dir / "manifests" / "train.csv"
    manifest_frame = pd.read_csv(manifest_csv)
    manifest_frame.loc[0, "tst"] = 0.0
    manifest_frame.to_csv(manifest_csv, index=False)

    with pytest.raises(ValueError, match="requires tst > 0"):
        KaldiPSGDataset(kaldi_data_root=output_dir, manifest=manifest_path, **_dataset_kwargs())

    manifest_frame.loc[0, "tst"] = 4.0
    manifest_frame.to_csv(manifest_csv, index=False)
    sample_key = "center-a_record-1_000000_000002"
    ark_path = output_dir / "channels" / "train" / "arousal.ark"
    scp_path = output_dir / "channels" / "train" / "arousal.scp"
    with kaldi_native_io.FloatMatrixWriter(f"ark,scp:{ark_path},{scp_path}") as writer:
        writer.write(sample_key, np.full((2, 120), 0.5, dtype=np.float32))

    dataset = KaldiPSGDataset(kaldi_data_root=output_dir, manifest=manifest_path, **_dataset_kwargs())
    with pytest.raises(ValueError, match="only 0/1"):
        next(iter(dataset.dataloader()))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "missing arousal metadata columns"),
        ("nonfinite", "non-finite arousal metadata"),
    ],
)
def test_arousal_kaldi_loader_rejects_missing_or_nonfinite_scalars(tmp_path: Path, mutation: str, match: str) -> None:
    output_dir, manifest_path, _ = _convert(tmp_path)
    manifest_csv = output_dir / "manifests" / "train.csv"
    manifest_frame = pd.read_csv(manifest_csv)
    if mutation == "missing":
        manifest_frame = manifest_frame.drop(columns=["arousal_res_index_per_hour"])
    else:
        manifest_frame.loc[0, "arousal_res_index_per_hour"] = np.inf
    manifest_frame.to_csv(manifest_csv, index=False)

    with pytest.raises(ValueError, match=match):
        KaldiPSGDataset(kaldi_data_root=output_dir, manifest=manifest_path, **_dataset_kwargs())


def test_arousal_kaldi_loader_rejects_invalid_input_dim_and_matrix_width(tmp_path: Path) -> None:
    output_dir, manifest_path, _ = _convert(tmp_path)
    manifest_json = json.loads(manifest_path.read_text())
    manifest_json["splits"]["train"]["channels"]["arousal"]["input_dim"] = 120.9
    manifest_path.write_text(json.dumps(manifest_json) + "\n")

    with pytest.raises(ValueError, match="requires input_dim=120"):
        KaldiPSGDataset(kaldi_data_root=output_dir, manifest=manifest_path, **_dataset_kwargs())

    manifest_json["splits"]["train"]["channels"]["arousal"]["input_dim"] = 120
    manifest_path.write_text(json.dumps(manifest_json) + "\n")
    sample_key = "center-a_record-1_000000_000002"
    ark_path = output_dir / "channels" / "train" / "arousal.ark"
    scp_path = output_dir / "channels" / "train" / "arousal.scp"
    with kaldi_native_io.FloatMatrixWriter(f"ark,scp:{ark_path},{scp_path}") as writer:
        writer.write(sample_key, np.zeros((2, 119), dtype=np.float32))

    dataset = KaldiPSGDataset(kaldi_data_root=output_dir, manifest=manifest_path, **_dataset_kwargs())
    with pytest.raises(ValueError, match="has input_dim=119, expected input_dim=120"):
        next(iter(dataset.dataloader()))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("fractional_token_sec", "token_sec=30"),
        ("compressed_storage", "ark_storage='float_matrix'"),
    ],
)
def test_arousal_kaldi_loader_rejects_inexact_manifest_contract(tmp_path: Path, mutation: str, match: str) -> None:
    output_dir, manifest_path, _ = _convert(tmp_path)
    manifest_json = json.loads(manifest_path.read_text())
    if mutation == "fractional_token_sec":
        manifest_json["token_sec"] = 30.9
    else:
        manifest_json["splits"]["train"]["channels"]["arousal"]["ark_storage"] = "compressed_matrix"
    manifest_path.write_text(json.dumps(manifest_json) + "\n")

    with pytest.raises(ValueError, match=match):
        KaldiPSGDataset(kaldi_data_root=output_dir, manifest=manifest_path, **_dataset_kwargs())


def test_arousal_kaldi_conversion_requires_thirty_second_tokens(tmp_path: Path) -> None:
    config_path, index_path, _ = _write_source(tmp_path)
    args = parse_args(
        [
            "--index",
            str(index_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "kaldi"),
            "--max-tokens",
            "2",
            "--token-sec",
            "10",
            "--extra-channels",
            "arousal",
        ]
    )

    with pytest.raises(ValueError, match="--token-sec 30"):
        convert(args)


def test_arousal_kaldi_conversion_rejects_overlapping_eval_windows(tmp_path: Path) -> None:
    config_path, index_path, _ = _write_source(tmp_path)
    frame = pd.read_csv(index_path)
    frame["split"] = "val"
    frame.to_csv(index_path, index=False)
    args = parse_args(
        [
            "--index",
            str(index_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "kaldi"),
            "--max-tokens",
            "2",
            "--stride-tokens",
            "1",
            "--include-overlap-eval-splits",
            "--extra-channels",
            "arousal",
        ]
    )

    with pytest.raises(ValueError, match="contiguous non-overlapping windows"):
        convert(args)


def test_arousal_kaldi_conversion_supports_ark_shards(tmp_path: Path) -> None:
    config_path, index_path, events = _write_source(tmp_path)
    output_dir = tmp_path / "kaldi"
    args = parse_args(
        [
            "--index",
            str(index_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--max-tokens",
            "1",
            "--ark-shards",
            "2",
            "--num-workers",
            "1",
            "--extra-channels",
            "arousal",
        ]
    )

    convert(args)

    channel_dir = output_dir / "channels" / "train"
    keys = ["center-a_record-1_000000_000001", "center-a_record-1_000001_000002"]
    assert [line.split()[0] for line in (channel_dir / "arousal.1.scp").read_text().splitlines()] == [keys[0]]
    assert [line.split()[0] for line in (channel_dir / "arousal.2.scp").read_text().splitlines()] == [keys[1]]
    aggregate_scp = channel_dir / "arousal.scp"
    assert [line.split()[0] for line in aggregate_scp.read_text().splitlines()] == keys
    with kaldi_native_io.RandomAccessFloatMatrixReader(f"scp:{aggregate_scp}") as reader:
        np.testing.assert_array_equal(np.asarray(reader[keys[0]]), events[:30].reshape(1, 120))
        np.testing.assert_array_equal(np.asarray(reader[keys[1]]), events[30:].reshape(1, 120))
