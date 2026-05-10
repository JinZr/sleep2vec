from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

kaldi_native_io = pytest.importorskip("kaldi_native_io")

from data.kaldi_psg_dataset import KaldiPSGDataset
from data.samplers import PairFirstBatchSampler
from sleep2vec.utils import _build_finetune_loader


def _write_kaldi_root(
    root: Path,
    channel_input_dims: dict[str, int],
    matrices: dict[str, dict[str, np.ndarray]],
) -> None:
    channels_dir = root / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    manifest_channels = {}
    for channel, input_dim in channel_input_dims.items():
        ark_path = channels_dir / f"{channel}.ark"
        scp_path = channels_dir / f"{channel}.scp"
        with kaldi_native_io.FloatMatrixWriter(f"ark,scp:{ark_path},{scp_path}") as writer:
            for key, matrix in matrices.get(channel, {}).items():
                writer.write(key, np.asarray(matrix, dtype=np.float32))
        manifest_channels[channel] = {"input_dim": input_dim, "scp": f"channels/{channel}.scp"}

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "backend": "kaldi_native_io",
                "channels": manifest_channels,
            }
        )
        + "\n"
    )


def _write_manifest(root: Path, rows: list[dict]) -> Path:
    path = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _row(sample_key: str, channels: list[str], *, start: int = 0, end: int = 2, **metadata):
    row = {
        "sample_key": sample_key,
        "record_key": sample_key.rsplit("_", 2)[0],
        "path": f"/original/{sample_key}.npz",
        "source": "center-a",
        "dataset": "mesa",
        "split": "train",
        "token_start": start,
        "token_end": end,
        "num_tokens": end - start,
        "age": 40,
        "sex": 1,
        "available_channels": json.dumps(channels),
    }
    row.update(metadata)
    return row


def _finetune_args(
    root: Path,
    label_name: str,
    *,
    label_source_name: str,
    output_dim: int,
    auxiliary_label_source_names: list[str] | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        data_backend="kaldi",
        kaldi_data_root=root,
        kaldi_manifest=root / "manifest.csv",
        label_name=label_name,
        label_source_name=label_source_name,
        auxiliary_label_source_names=auxiliary_label_source_names or [],
        data_channel_names=["ppg"],
        channel_input_dims={"ppg": 4},
        finetune_preset_path=None,
        finetune_data_index=Path("unused.csv"),
        max_tokens=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        is_classification=True,
        output_dim=output_dim,
    )


def test_kaldi_dataset_batch_contract_without_npz_reads(tmp_path: Path, monkeypatch) -> None:
    keys = ["mesa_s1_000000_000002", "mesa_s2_000002_000005"]
    _write_kaldi_root(
        tmp_path,
        {"eeg": 3, "ppg": 2},
        {
            "eeg": {
                keys[0]: np.ones((2, 3), dtype=np.float32),
                keys[1]: np.full((3, 3), 2.0, dtype=np.float32),
            },
            "ppg": {
                keys[0]: np.full((2, 2), 3.0, dtype=np.float32),
                keys[1]: np.full((3, 2), 4.0, dtype=np.float32),
            },
        },
    )
    manifest = _write_manifest(
        tmp_path,
        [
            _row(keys[0], ["eeg", "ppg"], age=40, sex=1),
            _row(keys[1], ["eeg", "ppg"], start=2, end=5, age=41, sex=0),
        ],
    )

    def fail_load_npz(path):
        raise AssertionError(f"Unexpected NPZ read: {path}")

    monkeypatch.setattr("data.default_dataset.load_npz", fail_load_npz)

    dataset = KaldiPSGDataset(
        channel_names=["eeg", "ppg"],
        channel_input_dims={"eeg": 3, "ppg": 2},
        kaldi_data_root=tmp_path,
        manifest=manifest,
        split=["train"],
        max_tokens=3,
        mask_rate=0.0,
        randomly_select_channels=False,
        allow_missing_channels=False,
        is_train_set=False,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataset.dataloader(device="cpu")))

    assert set(batch) == {"id", "length", "token_start", "metadata", "pair", "w", "h", "tokens", "mlm_mask"}
    assert batch["id"] == keys
    assert batch["length"].tolist() == [2, 3]
    assert batch["token_start"].tolist() == [0, 2]
    assert batch["pair"] == ("eeg", "ppg")
    assert set(batch["tokens"]) == {"eeg", "ppg"}
    assert set(batch["mlm_mask"]) == {"eeg", "ppg"}
    assert batch["tokens"]["eeg"].shape == (2, 3, 3)
    assert batch["tokens"]["ppg"].shape == (2, 3, 2)
    assert batch["tokens"]["eeg"][0, 2].eq(0.0).all()
    assert batch["tokens"]["ppg"][0, 2].eq(0.0).all()
    assert batch["mlm_mask"]["eeg"].dtype == torch.bool
    assert batch["mlm_mask"]["eeg"].sum().item() == 0
    assert batch["metadata"]["age"].tolist() == [40.0, 41.0]
    assert batch["metadata"]["sex"].tolist() == [1, 0]
    assert batch["metadata"]["source"] == ["center-a", "center-a"]
    assert batch["metadata"]["path"] == [f"/original/{key}.npz" for key in keys]
    assert batch["w"].shape == (2, 2)
    assert batch["h"].shape == (2, 2)


def test_kaldi_dataset_missing_channels_uses_pair_first_sampler(tmp_path: Path, monkeypatch) -> None:
    keys = [
        "mesa_ab_000000_000002",
        "mesa_ac_000000_000002",
        "mesa_bc_000000_000002",
    ]
    available_by_key = {
        keys[0]: {"eeg", "ppg"},
        keys[1]: {"eeg", "ecg"},
        keys[2]: {"ppg", "ecg"},
    }
    matrices = {
        channel: {
            key: np.full((2, 2), float(i + 1), dtype=np.float32)
            for i, key in enumerate(keys)
            if channel in available_by_key[key]
        }
        for channel in ["eeg", "ppg", "ecg"]
    }
    _write_kaldi_root(tmp_path, {"eeg": 2, "ppg": 2, "ecg": 2}, matrices)
    manifest = _write_manifest(
        tmp_path,
        [
            _row(keys[0], ["eeg", "ppg"]),
            _row(keys[1], ["eeg", "ecg"]),
            _row(keys[2], ["ppg", "ecg"]),
        ],
    )

    dataset = KaldiPSGDataset(
        channel_names=["eeg", "ppg", "ecg"],
        channel_input_dims={"eeg": 2, "ppg": 2, "ecg": 2},
        kaldi_data_root=tmp_path,
        manifest=manifest,
        split=["train"],
        max_tokens=2,
        mask_rate=0.0,
        allow_missing_channels=True,
        min_channels=2,
        is_train_set=True,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    read_calls = []
    original_read_matrix = dataset.reader_pool.read_matrix

    def spy_read_matrix(channel: str, key: str):
        read_calls.append((channel, key))
        return original_read_matrix(channel, key)

    monkeypatch.setattr(dataset.reader_pool, "read_matrix", spy_read_matrix)

    loader = dataset.dataloader(device="cpu")
    batches = list(loader)

    assert isinstance(loader.batch_sampler, PairFirstBatchSampler)
    assert set(loader.batch_sampler.pairs) == {("eeg", "ppg"), ("eeg", "ecg"), ("ppg", "ecg")}
    assert batches
    for batch in batches:
        pair = tuple(batch["pair"])
        assert set(batch["tokens"]) == set(pair)
        for sample_id in batch["id"]:
            assert set(pair).issubset(available_by_key[sample_id])
    for channel, key in read_calls:
        assert channel in available_by_key[key]


def test_kaldi_dataset_stage_and_ahi_labels(tmp_path: Path) -> None:
    keys = ["mesa_s1_000000_000002", "mesa_s2_000000_000001"]
    _write_kaldi_root(
        tmp_path,
        {"ppg": 4, "stage5": 1, "ahi": 30},
        {
            "ppg": {
                keys[0]: np.ones((2, 4), dtype=np.float32),
                keys[1]: np.full((1, 4), 2.0, dtype=np.float32),
            },
            "stage5": {
                keys[0]: np.asarray([[0.0], [4.0]], dtype=np.float32),
                keys[1]: np.asarray([[2.0]], dtype=np.float32),
            },
            "ahi": {
                keys[0]: np.ones((2, 30), dtype=np.float32),
                keys[1]: np.full((1, 30), 3.0, dtype=np.float32),
            },
        },
    )
    manifest = _write_manifest(
        tmp_path,
        [
            _row(
                keys[0],
                ["ppg", "stage5", "ahi"],
                ahi=7.5,
                tst=321.0,
            ),
            _row(
                keys[1],
                ["ppg", "stage5", "ahi"],
                end=1,
                ahi=8.5,
                tst=111.0,
            ),
        ],
    )

    dataset = KaldiPSGDataset(
        channel_names=["ppg", "stage5", "ahi"],
        channel_input_dims={"ppg": 4, "stage5": 1, "ahi": 30},
        kaldi_data_root=tmp_path,
        manifest=manifest,
        split=["train"],
        max_tokens=2,
        mask_rate=0.0,
        meta_data_names=["ahi", "tst"],
        meta_data_regression_names=["ahi", "tst"],
        randomly_select_channels=False,
        allow_missing_channels=False,
        is_train_set=False,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataset.dataloader(device="cpu")))

    assert set(batch["tokens"]) == {"ppg", "stage5", "ahi"}
    assert batch["tokens"]["ppg"].shape == (2, 2, 4)
    assert batch["tokens"]["stage5"].shape == (2, 2, 1)
    assert batch["tokens"]["ahi"].shape == (2, 2, 30)
    assert batch["tokens"]["ppg"][1, 1].eq(0.0).all()
    assert batch["tokens"]["stage5"][1, 1].eq(-1.0).all()
    assert batch["tokens"]["ahi"][1, 1].eq(-1.0).all()
    assert batch["metadata"]["ahi"].tolist() == pytest.approx([7.5, 8.5])
    assert batch["metadata"]["tst"].tolist() == pytest.approx([321.0, 111.0])
    assert batch["mlm_mask"]["stage5"].sum().item() == 0
    assert batch["mlm_mask"]["ahi"].sum().item() == 0


def test_kaldi_build_finetune_loader_stage_task_reads_stage5(tmp_path: Path) -> None:
    key = "mesa_s1_000000_000002"
    _write_kaldi_root(
        tmp_path,
        {"ppg": 4, "stage5": 1},
        {
            "ppg": {key: np.ones((2, 4), dtype=np.float32)},
            "stage5": {key: np.asarray([[0.0], [4.0]], dtype=np.float32)},
        },
    )
    _write_manifest(tmp_path, [_row(key, ["ppg", "stage5"])])

    loader = _build_finetune_loader(
        _finetune_args(tmp_path, "stage3", label_source_name="stage5", output_dim=3),
        split=["train"],
        sources=["center-a"],
        shuffle=False,
        is_train_set=False,
    )
    batch = next(iter(loader))

    assert set(batch["tokens"]) == {"ppg", "stage5"}
    assert batch["tokens"]["stage5"].shape == (1, 2, 1)
    assert batch["mlm_mask"]["stage5"].sum().item() == 0


def test_kaldi_build_finetune_loader_ahi_task_reads_labels_and_metadata(tmp_path: Path) -> None:
    key = "mesa_s1_000000_000002"
    _write_kaldi_root(
        tmp_path,
        {"ppg": 4, "ahi": 30, "stage5": 1},
        {
            "ppg": {key: np.ones((2, 4), dtype=np.float32)},
            "ahi": {key: np.ones((2, 30), dtype=np.float32)},
            "stage5": {key: np.asarray([[0.0], [4.0]], dtype=np.float32)},
        },
    )
    _write_manifest(tmp_path, [_row(key, ["ppg", "ahi", "stage5"], ahi=7.5, tst=321.0)])

    loader = _build_finetune_loader(
        _finetune_args(
            tmp_path,
            "ahi",
            label_source_name="ahi",
            output_dim=30,
            auxiliary_label_source_names=["stage5"],
        ),
        split=["train"],
        sources=["center-a"],
        shuffle=False,
        is_train_set=False,
    )
    batch = next(iter(loader))

    assert set(batch["tokens"]) == {"ppg", "ahi", "stage5"}
    assert batch["tokens"]["ahi"].shape == (1, 2, 30)
    assert batch["tokens"]["stage5"].shape == (1, 2, 1)
    assert batch["metadata"]["ahi"].tolist() == pytest.approx([7.5])
    assert batch["metadata"]["tst"].tolist() == pytest.approx([321.0])


def test_kaldi_dataset_reader_pool_works_with_multiple_workers(tmp_path: Path) -> None:
    keys = [f"mesa_s{i}_000000_000002" for i in range(4)]
    _write_kaldi_root(
        tmp_path,
        {"eeg": 2, "ppg": 2},
        {
            "eeg": {key: np.full((2, 2), i, dtype=np.float32) for i, key in enumerate(keys)},
            "ppg": {key: np.full((2, 2), i + 10, dtype=np.float32) for i, key in enumerate(keys)},
        },
    )
    manifest = _write_manifest(tmp_path, [_row(key, ["eeg", "ppg"]) for key in keys])
    dataset = KaldiPSGDataset(
        channel_names=["eeg", "ppg"],
        channel_input_dims={"eeg": 2, "ppg": 2},
        kaldi_data_root=tmp_path,
        manifest=manifest,
        split=["train"],
        max_tokens=2,
        mask_rate=0.0,
        randomly_select_channels=False,
        allow_missing_channels=False,
        is_train_set=False,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    batches = list(dataset.dataloader(device="cpu"))

    assert [batch["id"][0] for batch in batches] == keys


def test_kaldi_dataset_rejects_missing_manifest_channel(tmp_path: Path) -> None:
    _write_kaldi_root(tmp_path, {"eeg": 2}, {"eeg": {}})
    manifest = _write_manifest(tmp_path, [_row("mesa_s1_000000_000002", ["eeg"])])

    with pytest.raises(ValueError, match="missing requested channel"):
        KaldiPSGDataset(
            channel_names=["eeg", "ppg"],
            channel_input_dims={"eeg": 2, "ppg": 2},
            kaldi_data_root=tmp_path,
            manifest=manifest,
            split=["train"],
            max_tokens=2,
            mask_rate=0.0,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )


def test_kaldi_dataset_rejects_matrix_length_mismatch(tmp_path: Path) -> None:
    key = "mesa_s1_000000_000002"
    _write_kaldi_root(
        tmp_path,
        {"eeg": 2},
        {"eeg": {key: np.ones((1, 2), dtype=np.float32)}},
    )
    manifest = _write_manifest(tmp_path, [_row(key, ["eeg"], start=0, end=2)])
    dataset = KaldiPSGDataset(
        channel_names=["eeg"],
        channel_input_dims={"eeg": 2},
        kaldi_data_root=tmp_path,
        manifest=manifest,
        split=["train"],
        max_tokens=2,
        mask_rate=0.0,
        randomly_select_channels=False,
        allow_missing_channels=False,
        is_train_set=False,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    with pytest.raises(ValueError, match="expected 2 from manifest"):
        next(iter(dataset.dataloader(device="cpu")))
