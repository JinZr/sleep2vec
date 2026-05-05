from __future__ import annotations

import typing as t

import torch

SignalDict = dict[str, torch.Tensor]


def _pad_channel_dim(tensor: torch.Tensor, channels: int, *, value: float | bool = 0) -> torch.Tensor:
    if tensor.shape[1] == channels:
        return tensor
    if tensor.shape[1] > channels:
        raise ValueError(f"Tensor has {tensor.shape[1]} channels, expected at most {channels}.")

    pad_shape = list(tensor.shape)
    pad_shape[1] = channels - tensor.shape[1]
    pad = torch.full(pad_shape, value, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=1)


def _stack_signal_dict(samples: list[dict[str, t.Any]], key: str) -> SignalDict:
    modalities = samples[0][key].keys()
    stacked: SignalDict = {}
    for modality in modalities:
        max_channels = max(sample[key][modality].shape[1] for sample in samples)
        tensors = [_pad_channel_dim(sample[key][modality], max_channels) for sample in samples]
        stacked[modality] = torch.stack(tensors, dim=0)
    return stacked


def _stack_mask_dict(samples: list[dict[str, t.Any]], key: str) -> SignalDict:
    modalities = samples[0][key].keys()
    stacked: SignalDict = {}
    for modality in modalities:
        values = [sample[key][modality] for sample in samples]
        if values[0].dim() == 3:
            max_channels = max(value.shape[1] for value in values)
            values = [_pad_channel_dim(value, max_channels, value=False) for value in values]
        stacked[modality] = torch.stack(values, dim=0)
    return stacked


def collate_sleep2wave_generative(samples: list[dict[str, t.Any]]) -> dict[str, t.Any]:
    if not samples:
        raise ValueError("Cannot collate an empty Sleep2Wave generative batch.")

    metadata_keys = samples[0]["metadata"].keys()
    metadata = {key: [sample["metadata"].get(key) for sample in samples] for key in metadata_keys}

    return {
        "clean_signals": _stack_signal_dict(samples, "clean_signals"),
        "observed_signals": _stack_signal_dict(samples, "observed_signals"),
        "availability_mask": _stack_mask_dict(samples, "availability_mask"),
        "quality_mask": _stack_mask_dict(samples, "quality_mask"),
        "corruption_mask": _stack_mask_dict(samples, "corruption_mask"),
        "epoch_index": torch.stack([sample["epoch_index"] for sample in samples], dim=0),
        "night_position": torch.stack([sample["night_position"] for sample in samples], dim=0),
        "metadata": metadata,
        "condition_modalities": list(samples[0]["condition_modalities"]),
        "target_modalities": list(samples[0]["target_modalities"]),
        "task_type": samples[0]["task_type"],
    }


__all__ = ["SignalDict", "collate_sleep2wave_generative"]
