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


def _stack_channel_mask(samples: list[dict[str, t.Any]]) -> SignalDict:
    modalities = samples[0]["clean_signals"].keys()
    stacked: SignalDict = {}
    for modality in modalities:
        if "channel_mask" in samples[0]:
            values = [sample["channel_mask"][modality] for sample in samples]
            max_channels = max(value.shape[1] for value in values)
            masks = [_pad_channel_dim(value, max_channels, value=False) for value in values]
            stacked[modality] = torch.stack(masks, dim=0)
            continue
        max_channels = max(sample["clean_signals"][modality].shape[1] for sample in samples)
        masks = []
        for sample in samples:
            signal = sample["clean_signals"][modality]
            mask = torch.ones(signal.shape[:2], dtype=torch.bool)
            masks.append(_pad_channel_dim(mask, max_channels, value=False))
        stacked[modality] = torch.stack(masks, dim=0)
    return stacked


def collate_sleep2wave_generative(samples: list[dict[str, t.Any]]) -> dict[str, t.Any]:
    if not samples:
        raise ValueError("Cannot collate an empty sleep2wave generative batch.")

    metadata_keys = samples[0]["metadata"].keys()
    metadata = {key: [sample["metadata"].get(key) for sample in samples] for key in metadata_keys}

    batch = {
        "clean_signals": _stack_signal_dict(samples, "clean_signals"),
        "observed_signals": _stack_signal_dict(samples, "observed_signals"),
        "availability_mask": _stack_mask_dict(samples, "availability_mask"),
        "quality_mask": _stack_mask_dict(samples, "quality_mask"),
        "corruption_mask": _stack_mask_dict(samples, "corruption_mask"),
        "channel_mask": _stack_channel_mask(samples),
        "epoch_index": torch.stack([sample["epoch_index"] for sample in samples], dim=0),
        "night_position": torch.stack([sample["night_position"] for sample in samples], dim=0),
        "metadata": metadata,
        "condition_modalities": list(samples[0]["condition_modalities"]),
        "target_modalities": list(samples[0]["target_modalities"]),
        "task_type": samples[0]["task_type"],
    }
    if "clean_latents" in samples[0]:
        batch["clean_latents"] = {
            modality: torch.stack([sample["clean_latents"][modality] for sample in samples], dim=0)
            for modality in samples[0]["clean_latents"]
        }
    if "observed_latents" in samples[0]:
        batch["observed_latents"] = {
            modality: torch.stack([sample["observed_latents"][modality] for sample in samples], dim=0)
            for modality in samples[0]["observed_latents"]
        }
    return batch


__all__ = ["SignalDict", "collate_sleep2wave_generative"]
