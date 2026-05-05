from __future__ import annotations

import pytest
import torch

from sleep2wave.data.modalities import CANONICAL_MODALITIES
from sleep2wave.training.task_sampler import Sleep2WaveTaskSampler


def _availability(*, unavailable: str | None = None):
    masks = {modality: torch.ones(2, 2, dtype=torch.bool) for modality in CANONICAL_MODALITIES}
    if unavailable is not None:
        masks[unavailable] = torch.zeros(2, 2, dtype=torch.bool)
    return masks


def test_task_sampler_samples_translation_with_disjoint_sets():
    sampler = Sleep2WaveTaskSampler(
        phase=1,
        task_mix={"translation": 1.0},
        condition_counts=[1],
        seed=2,
    )

    task = sampler.sample(_availability())

    assert task.task_type == "translation"
    assert len(task.condition_modalities) == 1
    assert set(task.condition_modalities).isdisjoint(task.target_modalities)


def test_task_sampler_never_samples_unavailable_target():
    sampler = Sleep2WaveTaskSampler(
        phase=1,
        task_mix={"translation": 1.0},
        condition_counts=[1],
        seed=3,
    )

    for _ in range(20):
        task = sampler.sample(_availability(unavailable="eeg"))
        assert "eeg" not in task.condition_modalities
        assert "eeg" not in task.target_modalities


def test_task_sampler_uses_aux_for_restoration():
    sampler = Sleep2WaveTaskSampler(
        phase=1,
        task_mix={"restoration": 1.0},
        auxiliary_restoration_token=True,
        seed=4,
    )

    task = sampler.sample(_availability())

    assert task.task_type == "restoration"
    assert task.use_auxiliary_token
    assert task.condition_modalities == task.target_modalities


def test_task_sampler_requires_enough_modalities_for_two_condition_task():
    sampler = Sleep2WaveTaskSampler(
        phase=3,
        task_mix={"two_condition": 1.0},
        condition_counts=[2],
        seed=5,
    )
    availability = {modality: torch.zeros(2, 2, dtype=torch.bool) for modality in CANONICAL_MODALITIES}
    availability["eeg"] = torch.ones(2, 2, dtype=torch.bool)
    availability["ecg"] = torch.ones(2, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="Not enough available modalities"):
        sampler.sample(availability)


def test_task_sampler_uses_only_common_modalities_for_mixed_availability_batches():
    sampler = Sleep2WaveTaskSampler(
        phase=1,
        task_mix={"translation": 1.0},
        condition_counts=[1],
        seed=1,
    )
    availability = {modality: torch.zeros(2, 2, dtype=torch.bool) for modality in CANONICAL_MODALITIES}
    availability["eeg"][0] = True
    availability["ecg"][0] = True
    availability["eeg"][1] = True
    availability["ecg"][1] = True
    availability["spo2"][1] = True

    task = sampler.sample(availability)

    assert task.task_type == "translation"
    assert len(task.condition_modalities) == 1
    assert set(task.condition_modalities).isdisjoint(task.target_modalities)
    assert set(task.condition_modalities + task.target_modalities).issubset({"eeg", "ecg"})


def test_task_sampler_accepts_partial_epoch_availability():
    sampler = Sleep2WaveTaskSampler(
        phase=1,
        task_mix={"translation": 1.0},
        condition_counts=[1],
        seed=1,
    )
    availability = {modality: torch.zeros(2, 2, dtype=torch.bool) for modality in CANONICAL_MODALITIES}
    availability["eeg"] = torch.ones(2, 2, dtype=torch.bool)
    availability["ecg"][0, 0] = True
    availability["ecg"][1, 1] = True

    task = sampler.sample(availability)

    assert task.task_type == "translation"
    assert set(task.condition_modalities + task.target_modalities).issubset({"eeg", "ecg"})


def test_task_sampler_partial_full_samples_configured_condition_counts():
    sampler = Sleep2WaveTaskSampler(
        phase=4,
        task_mix={"partial_full": 1.0},
        condition_counts=[1, 2, 3, 4],
        seed=7,
    )

    counts = {len(sampler.sample(_availability()).condition_modalities) for _ in range(200)}

    assert {1, 2, 3, 4}.issubset(counts)


def test_task_sampler_partial_full_bounds_condition_counts_by_availability():
    sampler = Sleep2WaveTaskSampler(
        phase=4,
        task_mix={"partial_full": 1.0},
        condition_counts=[1, 2, 3, 4],
        seed=8,
    )
    availability = {modality: torch.zeros(2, 2, dtype=torch.bool) for modality in CANONICAL_MODALITIES}
    availability["eeg"] = torch.ones(2, 2, dtype=torch.bool)
    availability["ecg"] = torch.ones(2, 2, dtype=torch.bool)
    availability["spo2"] = torch.ones(2, 2, dtype=torch.bool)

    counts = {len(sampler.sample(availability).condition_modalities) for _ in range(100)}

    assert counts == {1, 2}


def test_task_sampler_rejects_batches_without_common_translation_pair():
    sampler = Sleep2WaveTaskSampler(
        phase=1,
        task_mix={"translation": 1.0},
        condition_counts=[1],
        seed=1,
    )
    availability = {modality: torch.zeros(2, 2, dtype=torch.bool) for modality in CANONICAL_MODALITIES}
    availability["eeg"][0] = True
    availability["ecg"][0] = True
    availability["spo2"][1] = True
    availability["ibi"][1] = True

    with pytest.raises(
        ValueError, match="Translation and partial_full tasks require at least two available modalities"
    ):
        sampler.sample(availability)
