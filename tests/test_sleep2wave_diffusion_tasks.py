from __future__ import annotations

import pytest

from sleep2wave.diffusion.tasks import build_generation_task


def test_generation_task_accepts_valid_restoration_and_imputation():
    restoration = build_generation_task(
        "restoration",
        condition_modalities=["eeg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )
    imputation = build_generation_task(
        "imputation",
        condition_modalities=["spo2"],
        target_modalities=["spo2"],
        auxiliary_restoration_token=True,
    )

    assert restoration.task_type == "restoration"
    assert restoration.condition_modalities == ("eeg",)
    assert restoration.target_modalities == ("eeg",)
    assert restoration.use_auxiliary_token
    assert imputation.task_type == "imputation"
    assert imputation.use_auxiliary_token


def test_generation_task_accepts_valid_translation_and_partial_full():
    translation = build_generation_task(
        "translation",
        condition_modalities=["ecg"],
        target_modalities=["eeg"],
    )
    partial_full = build_generation_task(
        "partial_full",
        condition_modalities=["ecg", "belt"],
        target_modalities=["eeg", "eog", "emg", "airflow", "spo2", "ibi", "resp"],
    )

    assert translation.task_type == "translation"
    assert not translation.use_auxiliary_token
    assert partial_full.task_type == "partial_full"
    assert partial_full.condition_modalities == ("ecg", "belt")


def test_generation_task_rejects_empty_condition_set():
    with pytest.raises(ValueError, match="condition_modalities must be a non-empty sequence"):
        build_generation_task("translation", condition_modalities=[], target_modalities=["eeg"])


def test_generation_task_rejects_empty_target_set():
    with pytest.raises(ValueError, match="target_modalities must be a non-empty sequence"):
        build_generation_task("translation", condition_modalities=["ecg"], target_modalities=[])


def test_generation_task_rejects_unknown_modality():
    with pytest.raises(ValueError, match="canonical modality names"):
        build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["unknown"])


def test_generation_task_rejects_duplicate_modality():
    with pytest.raises(ValueError, match="Duplicate Sleep2Wave modality"):
        build_generation_task("translation", condition_modalities=["ecg", "ecg"], target_modalities=["eeg"])


def test_generation_task_rejects_translation_overlap():
    with pytest.raises(ValueError, match="requires disjoint condition and target modalities"):
        build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["ecg"])


def test_generation_task_rejects_restoration_without_auxiliary_token():
    with pytest.raises(ValueError, match="requires auxiliary_restoration_token=True"):
        build_generation_task("restoration", condition_modalities=["eeg"], target_modalities=["eeg"])


def test_generation_task_rejects_restoration_target_not_condition():
    with pytest.raises(ValueError, match="requires the target modality 'eeg' as a condition"):
        build_generation_task(
            "restoration",
            condition_modalities=["ecg"],
            target_modalities=["eeg"],
            auxiliary_restoration_token=True,
        )
