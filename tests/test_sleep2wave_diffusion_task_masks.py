from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sleep2wave.data.modalities import CANONICAL_MODALITIES
from sleep2wave.diffusion.task_masks import TokenLayout, build_directional_task_attention_mask
from sleep2wave.diffusion.tasks import AUX_MODALITY, build_generation_task


def _can_attend(mask, query_idx: int, key_idx: int, batch_idx: int = 0) -> bool:
    return not bool(mask.blocked[batch_idx, query_idx, key_idx])


def test_token_layout_uses_deterministic_modality_epoch_order():
    layout = TokenLayout(context_epochs=2)

    assert layout.token_names[:4] == ["eeg_0", "eeg_1", "eog_0", "eog_1"]
    assert layout.token_index("eeg", 0) == 0
    assert layout.token_index("eeg", 1) == 1
    assert layout.token_index("eog", 0) == 2
    assert layout.token_index(AUX_MODALITY, 0) == len(CANONICAL_MODALITIES) * 2
    assert layout.token_count == (len(CANONICAL_MODALITIES) + 1) * 2


def test_translation_mask_allows_target_queries_to_condition_keys():
    layout = TokenLayout(context_epochs=2)
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    mask = build_directional_task_attention_mask(task, layout)

    eeg0 = layout.token_index("eeg", 0)
    ecg0 = layout.token_index("ecg", 0)
    ecg1 = layout.token_index("ecg", 1)

    assert mask.blocked.shape == (1, layout.token_count, layout.token_count)
    assert _can_attend(mask, eeg0, ecg0)
    assert _can_attend(mask, ecg0, ecg1)


def test_translation_mask_blocks_condition_queries_to_target_keys():
    layout = TokenLayout(context_epochs=2)
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    mask = build_directional_task_attention_mask(task, layout)

    eeg0 = layout.token_index("eeg", 0)
    ecg0 = layout.token_index("ecg", 0)

    assert not _can_attend(mask, ecg0, eeg0)


def test_restoration_mask_routes_clean_target_through_auxiliary_tokens():
    layout = TokenLayout(context_epochs=2)
    task = build_generation_task(
        "restoration",
        condition_modalities=["eeg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )
    mask = build_directional_task_attention_mask(task, layout)

    eeg0 = layout.token_index("eeg", 0)
    aux0 = layout.token_index(AUX_MODALITY, 0)

    assert bool(mask.condition_tokens[0, eeg0])
    assert not bool(mask.target_tokens[0, eeg0])
    assert bool(mask.target_tokens[0, aux0])
    assert _can_attend(mask, aux0, eeg0)
    assert not _can_attend(mask, eeg0, aux0)


def test_auxiliary_tokens_are_inactive_for_translation_and_partial_full():
    layout = TokenLayout(context_epochs=2)
    tasks = [
        build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"]),
        build_generation_task("partial_full", condition_modalities=["ecg"], target_modalities=["eeg", "eog"]),
    ]

    for task in tasks:
        mask = build_directional_task_attention_mask(task, layout)
        aux0 = layout.token_index(AUX_MODALITY, 0)
        assert not bool(mask.active_tokens[0, aux0])
        assert mask.blocked[0, aux0, :].all()
        assert mask.blocked[0, :, aux0].all()


def test_unavailable_condition_epochs_are_blocked_but_targets_stay_active():
    layout = TokenLayout(context_epochs=2)
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    mask = build_directional_task_attention_mask(
        task,
        layout,
        availability_mask={"ecg": torch.tensor([[True, False]])},
    )

    eeg1 = layout.token_index("eeg", 1)
    ecg0 = layout.token_index("ecg", 0)
    ecg1 = layout.token_index("ecg", 1)

    assert bool(mask.active_tokens[0, eeg1])
    assert not bool(mask.active_tokens[0, ecg1])
    assert _can_attend(mask, eeg1, ecg0)
    assert not _can_attend(mask, eeg1, ecg1)
    assert mask.blocked[0, ecg1, :].all()
    assert mask.blocked[0, :, ecg1].all()


def test_unavailable_target_epochs_are_inactive():
    layout = TokenLayout(context_epochs=2)
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    mask = build_directional_task_attention_mask(
        task,
        layout,
        availability_mask={
            "ecg": torch.tensor([[True, True]]),
            "eeg": torch.tensor([[True, False]]),
        },
    )

    eeg0 = layout.token_index("eeg", 0)
    eeg1 = layout.token_index("eeg", 1)

    assert bool(mask.target_tokens[0, eeg0])
    assert not bool(mask.target_tokens[0, eeg1])
    assert not bool(mask.active_tokens[0, eeg1])


def test_disabling_target_target_attention_blocks_cross_target_tokens():
    layout = TokenLayout(context_epochs=2)
    task = build_generation_task(
        "translation",
        condition_modalities=["ecg"],
        target_modalities=["eeg", "eog"],
        allow_target_target_attention=False,
    )
    mask = build_directional_task_attention_mask(task, layout)

    eeg0 = layout.token_index("eeg", 0)
    eeg1 = layout.token_index("eeg", 1)
    eog0 = layout.token_index("eog", 0)
    ecg0 = layout.token_index("ecg", 0)

    assert _can_attend(mask, eeg0, ecg0)
    assert _can_attend(mask, eeg0, eeg0)
    assert not _can_attend(mask, eeg0, eog0)
    assert not _can_attend(mask, eeg0, eeg1)
