from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from sleep2wave.data.default_dataset import SampleIndex
from sleep2wave.data.generative_dataset import Sleep2WaveGenerativeDataset
from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_ALIASES, MODALITY_SPECS


def _signal(length: int, offset: float) -> np.ndarray:
    return np.arange(length, dtype=np.float32) + offset


def _npz_payload(num_epochs: int, *, aliases: bool = False) -> dict[str, np.ndarray]:
    payload = {}
    alias_by_target = {target: alias for alias, target in MODALITY_ALIASES.items()}
    for idx, modality in enumerate(CANONICAL_MODALITIES):
        spec = MODALITY_SPECS[modality]
        key = alias_by_target[modality] if aliases else modality
        payload[key] = _signal(num_epochs * spec.frames_per_epoch, offset=float(idx))
    return payload


def _index_row(path: Path, *, subject_id: str, night_id: str, split: str = "train") -> dict:
    row = {
        "path": str(path),
        "duration": 60,
        "split": split,
        "subject_id": subject_id,
        "night_id": night_id,
        "source": "synthetic",
    }
    row.update({f"{modality}_mask": 1 for modality in CANONICAL_MODALITIES})
    return row


def test_generative_dataset_returns_30s_epoch_shapes_and_masks(tmp_path: Path):
    first_npz = tmp_path / "first.npz"
    second_npz = tmp_path / "second_alias.npz"
    np.savez(first_npz, **_npz_payload(2, aliases=False))
    np.savez(second_npz, **_npz_payload(2, aliases=True))

    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            _index_row(first_npz, subject_id="s1", night_id="n1"),
            _index_row(second_npz, subject_id="s2", night_id="n2"),
        ]
    ).to_csv(index_path, index=False)

    dataset = Sleep2WaveGenerativeDataset(
        index=index_path,
        split="train",
        context_epochs=2,
        condition_modalities=["ecg"],
        target_modalities=["eeg"],
        task_type="translation",
        corruption_name="gaussian_noise",
        corruption_kwargs={"std": 0.01},
        seed=7,
    )
    batch = next(iter(dataset.dataloader(batch_size=2)))

    assert batch["clean_signals"]["eeg"].shape == (2, 2, 1, 3840)
    assert batch["clean_signals"]["spo2"].shape == (2, 2, 1, 120)
    assert batch["observed_signals"]["eeg"].shape == (2, 2, 1, 3840)
    assert batch["availability_mask"]["eeg"].shape == (2, 2)
    assert batch["quality_mask"]["eeg"].shape == (2, 2)
    assert batch["corruption_mask"]["eeg"].shape == (2, 2, 1, 3840)
    assert batch["availability_mask"]["resp"].all()
    assert batch["quality_mask"]["resp"].eq(1.0).all()
    assert batch["corruption_mask"]["eeg"].any()
    assert batch["epoch_index"].tolist() == [[0, 1], [0, 1]]
    assert batch["metadata"]["subject_id"] == ["s1", "s2"]
    assert batch["metadata"]["night_id"] == ["n1", "n2"]
    assert batch["condition_modalities"] == ["ecg"]
    assert batch["target_modalities"] == ["eeg"]
    assert batch["task_type"] == "translation"


def test_generative_dataset_missing_modalities_are_zero_and_unavailable(tmp_path: Path):
    npz_path = tmp_path / "eeg_only.npz"
    np.savez(npz_path, eeg=_signal(2 * MODALITY_SPECS["eeg"].frames_per_epoch, 0.0))
    preset_path = tmp_path / "preset.pkl"
    sample = SampleIndex(
        id="sample",
        path=str(npz_path),
        start=0,
        end=2,
        payload={
            "available_channels": ["eeg"],
            "canonical_channel_map": {"eeg": "eeg"},
            "quality_mask_keys": {},
            "availability_mask_keys": {},
            "subject_id": "s1",
            "night_id": "n1",
            "night_epoch_count": 2,
        },
        metadata={"split": "train", "source": "synthetic"},
    )
    with preset_path.open("wb") as f:
        pickle.dump([sample], f)

    item = Sleep2WaveGenerativeDataset(preset_path=preset_path, split="train", context_epochs=2)[0]

    assert item["availability_mask"]["eeg"].tolist() == [True, True]
    assert item["availability_mask"]["spo2"].tolist() == [False, False]
    assert item["clean_signals"]["spo2"].shape == (2, 1, 120)
    assert item["clean_signals"]["spo2"].eq(0.0).all()
    assert item["quality_mask"]["spo2"].eq(0.0).all()


def test_generative_dataset_rejects_wrong_context_epoch_span(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, eeg=_signal(2 * MODALITY_SPECS["eeg"].frames_per_epoch, 0.0))
    preset_path = tmp_path / "preset.pkl"
    with preset_path.open("wb") as f:
        pickle.dump(
            [
                SampleIndex(
                    id="bad",
                    path=str(npz_path),
                    start=0,
                    end=1,
                    payload={"available_channels": ["eeg"]},
                    metadata={"split": "train"},
                )
            ],
            f,
        )

    dataset = Sleep2WaveGenerativeDataset(preset_path=preset_path, split="train", context_epochs=2)
    with pytest.raises(ValueError, match="expected context_epochs=2"):
        dataset[0]


def test_generative_dataset_rejects_short_available_signal(tmp_path: Path):
    npz_path = tmp_path / "short.npz"
    np.savez(npz_path, eeg=np.arange(10, dtype=np.float32))
    preset_path = tmp_path / "preset.pkl"
    with preset_path.open("wb") as f:
        pickle.dump(
            [
                SampleIndex(
                    id="short",
                    path=str(npz_path),
                    start=0,
                    end=2,
                    payload={"available_channels": ["eeg"], "canonical_channel_map": {"eeg": "eeg"}},
                    metadata={"split": "train"},
                )
            ],
            f,
        )

    dataset = Sleep2WaveGenerativeDataset(preset_path=preset_path, split="train", context_epochs=2)
    with pytest.raises(ValueError, match="too short"):
        dataset[0]


def test_generative_dataset_rejects_index_rows_with_no_available_modalities(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, eeg=_signal(2 * MODALITY_SPECS["eeg"].frames_per_epoch, 0.0))
    index_path = tmp_path / "index.csv"
    row = _index_row(npz_path, subject_id="s1", night_id="n1")
    for modality in CANONICAL_MODALITIES:
        row[f"{modality}_mask"] = 0
    pd.DataFrame([row]).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="no available Sleep2Wave modalities"):
        Sleep2WaveGenerativeDataset(index=index_path, split="train", context_epochs=2)
