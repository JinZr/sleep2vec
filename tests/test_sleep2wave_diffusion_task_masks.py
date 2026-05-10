from __future__ import annotations

import torch

from sleep2wave.data.modalities import CANONICAL_MODALITIES
from sleep2wave.diffusion.task_masks import (
    TokenLayout,
    build_directional_task_attention_mask,
    build_patch_condition_availability,
)
from sleep2wave.diffusion.tasks import AUX_MODALITY, build_generation_task


def _can_attend(mask, query_idx: int, key_idx: int, batch_idx: int = 0) -> bool:
    return not bool(mask.blocked[batch_idx, query_idx, key_idx])


def test_token_layout_uses_deterministic_modality_epoch_order():
    layout = TokenLayout(context_epochs=2, channel_count=2, patches_per_epoch=3)

    assert layout.token_names[:4] == ["eeg_0_0_0", "eeg_0_0_1", "eeg_0_0_2", "eeg_0_1_0"]
    assert layout.token_index("eeg", 0, 0, 0) == 0
    assert layout.token_index("eeg", 0, 0, 1) == 1
    assert layout.token_index("eeg", 0, 1, 0) == 3
    assert layout.token_index("eeg", 1, 0, 0) == 6
    assert layout.token_index("eog", 0, 0, 0) == 12
    assert layout.token_index(AUX_MODALITY, 0, 0, 0) == len(CANONICAL_MODALITIES) * 2 * 2 * 3
    assert layout.token_count == (len(CANONICAL_MODALITIES) + 1) * 2 * 2 * 3


def test_translation_mask_allows_target_queries_to_condition_keys():
    layout = TokenLayout(context_epochs=2, patches_per_epoch=3)
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    mask = build_directional_task_attention_mask(task, layout)

    eeg0 = layout.token_index("eeg", 0, 0, 0)
    ecg0 = layout.token_index("ecg", 0, 0, 0)
    ecg1 = layout.token_index("ecg", 1, 0, 2)

    assert mask.blocked.shape == (1, layout.token_count, layout.token_count)
    assert _can_attend(mask, eeg0, ecg0)
    assert _can_attend(mask, ecg0, ecg1)


def test_translation_mask_blocks_condition_queries_to_target_keys():
    layout = TokenLayout(context_epochs=2, patches_per_epoch=3)
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    mask = build_directional_task_attention_mask(task, layout)

    eeg0 = layout.token_index("eeg", 0, 0, 0)
    ecg0 = layout.token_index("ecg", 0, 0, 0)

    assert not _can_attend(mask, ecg0, eeg0)


def test_restoration_mask_routes_clean_target_through_auxiliary_tokens():
    layout = TokenLayout(context_epochs=2, patches_per_epoch=3)
    task = build_generation_task(
        "restoration",
        condition_modalities=["eeg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )
    mask = build_directional_task_attention_mask(task, layout)

    eeg0 = layout.token_index("eeg", 0, 0, 0)
    aux0 = layout.token_index(AUX_MODALITY, 0, 0, 0)

    assert bool(mask.condition_tokens[0, eeg0])
    assert not bool(mask.target_tokens[0, eeg0])
    assert bool(mask.target_tokens[0, aux0])
    assert _can_attend(mask, aux0, eeg0)
    assert not _can_attend(mask, eeg0, aux0)


def test_auxiliary_tokens_are_inactive_for_translation_and_partial_full():
    layout = TokenLayout(context_epochs=2, patches_per_epoch=3)
    tasks = [
        build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"]),
        build_generation_task("partial_full", condition_modalities=["ecg"], target_modalities=["eeg", "eog"]),
    ]

    for task in tasks:
        mask = build_directional_task_attention_mask(task, layout)
        aux0 = layout.token_index(AUX_MODALITY, 0, 0, 0)
        assert not bool(mask.active_tokens[0, aux0])
        assert mask.blocked[0, aux0, :].all()
        assert mask.blocked[0, :, aux0].all()


def test_unavailable_condition_epochs_are_blocked_but_targets_stay_active():
    layout = TokenLayout(context_epochs=2, patches_per_epoch=3)
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    mask = build_directional_task_attention_mask(
        task,
        layout,
        availability_mask={"ecg": torch.tensor([[True, False]])},
    )

    eeg1 = layout.token_index("eeg", 1, 0, 0)
    ecg0 = layout.token_index("ecg", 0, 0, 2)
    ecg1 = layout.token_index("ecg", 1, 0, 0)

    assert bool(mask.active_tokens[0, eeg1])
    assert not bool(mask.active_tokens[0, ecg1])
    assert _can_attend(mask, eeg1, ecg0)
    assert not _can_attend(mask, eeg1, ecg1)
    assert mask.blocked[0, ecg1, :].all()
    assert mask.blocked[0, :, ecg1].all()


def test_patch_availability_blocks_only_selected_condition_patches():
    layout = TokenLayout(context_epochs=2, patches_per_epoch=3)
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    mask = build_directional_task_attention_mask(
        task,
        layout,
        availability_mask={"ecg": torch.tensor([[[True, False, True], [True, True, True]]])},
    )

    eeg0 = layout.token_index("eeg", 0, 0, 0)
    ecg0_patch0 = layout.token_index("ecg", 0, 0, 0)
    ecg0_patch1 = layout.token_index("ecg", 0, 0, 1)

    assert _can_attend(mask, eeg0, ecg0_patch0)
    assert not bool(mask.active_tokens[0, ecg0_patch1])
    assert not _can_attend(mask, eeg0, ecg0_patch1)


def test_condition_patch_mask_does_not_disable_restoration_aux_target():
    layout = TokenLayout(context_epochs=2, patches_per_epoch=3)
    task = build_generation_task(
        "restoration",
        condition_modalities=["eeg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )
    mask = build_directional_task_attention_mask(
        task,
        layout,
        availability_mask={"eeg": torch.ones(1, 2, dtype=torch.bool)},
        condition_availability_mask={"eeg": torch.tensor([[[False, True, True], [True, True, True]]])},
    )

    eeg0 = layout.token_index("eeg", 0, 0, 0)
    aux0 = layout.token_index(AUX_MODALITY, 0, 0, 0)

    assert not bool(mask.condition_tokens[0, eeg0])
    assert bool(mask.target_tokens[0, aux0])
    assert bool(mask.active_tokens[0, aux0])


def test_patch_condition_availability_keeps_target_activation_out_of_conditions():
    task = build_generation_task(
        "imputation",
        condition_modalities=["eeg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )
    availability = {"eeg": torch.tensor([[False, True]])}
    corruption = {"eeg": torch.zeros(1, 2, 1, 6, dtype=torch.bool)}
    corruption["eeg"][0, 1, 0, 0:2] = True

    condition_availability = build_patch_condition_availability(
        availability,
        corruption,
        task,
        patches_per_epoch=3,
    )

    assert condition_availability["eeg"].tolist() == [[[[False, False, False]], [[False, True, True]]]]


def test_channel_mask_blocks_padded_condition_and_target_tokens():
    layout = TokenLayout(context_epochs=2, channel_count=2, patches_per_epoch=3)
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    channel_mask = {
        "ecg": torch.tensor([[[True, False], [True, False]]]),
        "eeg": torch.tensor([[[True, False], [True, False]]]),
    }
    mask = build_directional_task_attention_mask(task, layout, channel_mask=channel_mask)

    eeg_valid = layout.token_index("eeg", 0, 0, 0)
    eeg_padded = layout.token_index("eeg", 0, 1, 0)
    ecg_valid = layout.token_index("ecg", 0, 0, 0)
    ecg_padded = layout.token_index("ecg", 0, 1, 0)

    assert bool(mask.active_tokens[0, eeg_valid])
    assert bool(mask.active_tokens[0, ecg_valid])
    assert not bool(mask.active_tokens[0, eeg_padded])
    assert not bool(mask.active_tokens[0, ecg_padded])
    assert not _can_attend(mask, eeg_valid, ecg_padded)
    assert mask.blocked[0, eeg_padded, :].all()
    assert mask.blocked[0, :, ecg_padded].all()


def test_restoration_aux_tokens_are_channel_specific():
    layout = TokenLayout(context_epochs=2, channel_count=2, patches_per_epoch=3)
    task = build_generation_task(
        "restoration",
        condition_modalities=["eeg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )
    channel_mask = {"eeg": torch.tensor([[[True, False], [True, False]]])}
    mask = build_directional_task_attention_mask(task, layout, channel_mask=channel_mask)

    aux_valid = layout.token_index(AUX_MODALITY, 0, 0, 0)
    aux_padded = layout.token_index(AUX_MODALITY, 0, 1, 0)
    eeg_valid = layout.token_index("eeg", 0, 0, 0)

    assert bool(mask.target_tokens[0, aux_valid])
    assert not bool(mask.target_tokens[0, aux_padded])
    assert _can_attend(mask, aux_valid, eeg_valid)
    assert mask.blocked[0, aux_padded, :].all()


def test_unavailable_target_epochs_are_inactive():
    layout = TokenLayout(context_epochs=2, patches_per_epoch=3)
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])
    mask = build_directional_task_attention_mask(
        task,
        layout,
        availability_mask={
            "ecg": torch.tensor([[True, True]]),
            "eeg": torch.tensor([[True, False]]),
        },
    )

    eeg0 = layout.token_index("eeg", 0, 0, 0)
    eeg1 = layout.token_index("eeg", 1, 0, 2)

    assert bool(mask.target_tokens[0, eeg0])
    assert not bool(mask.target_tokens[0, eeg1])
    assert not bool(mask.active_tokens[0, eeg1])


def test_disabling_target_target_attention_blocks_cross_target_tokens():
    layout = TokenLayout(context_epochs=2, patches_per_epoch=3)
    task = build_generation_task(
        "translation",
        condition_modalities=["ecg"],
        target_modalities=["eeg", "eog"],
        allow_target_target_attention=False,
    )
    mask = build_directional_task_attention_mask(task, layout)

    eeg0 = layout.token_index("eeg", 0, 0, 0)
    eeg1 = layout.token_index("eeg", 1, 0, 0)
    eog0 = layout.token_index("eog", 0, 0, 0)
    ecg0 = layout.token_index("ecg", 0, 0, 0)

    assert _can_attend(mask, eeg0, ecg0)
    assert _can_attend(mask, eeg0, eeg0)
    assert not _can_attend(mask, eeg0, eog0)
    assert not _can_attend(mask, eeg0, eeg1)
