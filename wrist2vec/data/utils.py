from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import random
import typing as t

import numpy as np
import torch
from tqdm import tqdm

from wrist2vec.source_routing import (
    make_effective_channel_name,
    normalize_channel_source_names,
    uses_explicit_channel_sources,
)


def load_npz(path: str, mmap_mode: str | None = "r"):
    """
    Try to memory-map NPZ/NPY files when possible to reduce peak RAM.
    Falls back to a regular load if memmap is unsupported (e.g., compressed npz).
    """
    try:
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    except (ValueError, TypeError):
        return np.load(path, allow_pickle=False)


def default_extractor(
    name: str,
    frames_per_token: int,
    dtype: torch.dtype = torch.float32,
    *,
    source_name: str | None = None,
):
    """Slice one NPZ channel between token-aligned frame offsets."""

    def extract(npz, start: int, end: int):
        s = start * frames_per_token
        e = end * frames_per_token

        arr = npz[source_name or name]
        segment = arr[s:e]

        # Collapse trivial second dimension without copying.
        if segment.ndim == 2 and segment.shape[1] == 1:
            segment = segment[:, 0]

        return torch.as_tensor(segment, dtype=dtype)

    return extract


def default_tokenizer(frames_per_token: int):
    """Tokenize 1D or 2D signals into equal-length chunks."""

    def tokenize(data: torch.Tensor):
        if data.dim() == 1:
            total_length = data.shape[0]
            num_tokens = total_length // frames_per_token
            trimmed = data[: num_tokens * frames_per_token]
            return trimmed.view(num_tokens, frames_per_token)

        if data.dim() == 2:
            channels, total_length = data.shape
            num_tokens = total_length // frames_per_token
            trimmed = data[:, : num_tokens * frames_per_token]
            tokens = trimmed.view(channels, num_tokens, frames_per_token)
            return tokens.permute(1, 0, 2)

        raise ValueError(f"Unsupported input dimension: {data.shape}")

    return tokenize


def default_mlm_mask_generator(mask_ratio: float = 0.15):
    """Randomly mask a ratio of tokens for MLM/span masking tasks."""

    def generate_mask(tokens: torch.Tensor):
        num_tokens = tokens.shape[0]
        num_mask = int(num_tokens * mask_ratio)
        mask = torch.zeros(num_tokens, dtype=torch.bool)
        if num_mask > 0:
            mask_indices = torch.randperm(num_tokens)[:num_mask]
            mask[mask_indices] = True
        return mask

    return generate_mask


def _frames_per_token_for_channel(channel_name: str, channel_input_dims: t.Mapping[str, int] | None) -> int:
    if channel_name == "stage5":
        return 1
    if channel_name == "ahi":
        return 30
    if channel_input_dims is None or channel_name not in channel_input_dims:
        raise ValueError(f"Missing channel_input_dims for channel '{channel_name}'.")
    return int(channel_input_dims[channel_name])


def resolve_channel_sources(
    npz,
    channel_name: str,
    channel_source_names: t.Mapping[str, t.Sequence[str]] | None = None,
) -> list[str]:
    normalized = normalize_channel_source_names([channel_name], channel_source_names)
    requested = normalized[str(channel_name)]
    if channel_name == "ahi":
        try:
            load_builtin_ahi_metadata(npz)
        except Exception:
            return []
        return ["ah_event"]
    return [source_name for source_name in requested if source_name in npz]


def compute_channel_matches(
    npz,
    channel_names: t.Sequence[str],
    *,
    channel_source_names: t.Mapping[str, t.Sequence[str]] | None = None,
    expand_source_branches: bool = False,
) -> tuple[list[str], dict[str, list[str]]]:
    normalized = normalize_channel_source_names(channel_names, channel_source_names)
    available_channels: list[str] = []
    channel_sources: dict[str, list[str]] = {}

    for channel_name in channel_names:
        matched_sources = resolve_channel_sources(npz, channel_name, normalized)
        if not matched_sources:
            continue

        required_sources = normalized[str(channel_name)]
        if expand_source_branches and channel_name != "ahi" and matched_sources != required_sources:
            continue

        available_channels.append(str(channel_name))
        channel_sources[str(channel_name)] = list(matched_sources)

    return available_channels, channel_sources


def choose_source_name(
    channel_name: str,
    *,
    channel_sources: t.Mapping[str, t.Sequence[str]] | None = None,
    rng: random.Random | None = None,
) -> str | None:
    matched_sources = list(dict(channel_sources or {}).get(str(channel_name), []))
    if not matched_sources:
        return None if channel_name in {"stage5", "ahi"} else str(channel_name)
    if channel_name == "ahi":
        return "ah_event"
    if len(matched_sources) == 1:
        return matched_sources[0]
    chooser = rng or random
    return chooser.choice(matched_sources)


def load_channel_payload(
    npz,
    *,
    channel_name: str,
    start: int,
    end: int,
    channel_input_dims: t.Mapping[str, int] | None,
    source_name: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    frames_per_token = _frames_per_token_for_channel(channel_name, channel_input_dims)
    if channel_name == "ahi":
        source_name = "ah_event"
    elif channel_name == "stage5":
        source_name = "stage5"
    elif source_name is None:
        source_name = str(channel_name)
    return default_extractor(channel_name, frames_per_token, dtype=dtype, source_name=source_name)(npz, start, end)


def load_channel_tokens(
    npz,
    *,
    channel_name: str,
    start: int,
    end: int,
    channel_input_dims: t.Mapping[str, int] | None,
    source_name: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = load_channel_payload(
        npz,
        channel_name=channel_name,
        start=start,
        end=end,
        channel_input_dims=channel_input_dims,
        source_name=source_name,
        dtype=dtype,
    )
    frames_per_token = _frames_per_token_for_channel(channel_name, channel_input_dims)
    tokens = default_tokenizer(frames_per_token)(payload)
    return payload, tokens


def _load_scalar_npz_value(npz, key: str) -> float:
    if key not in npz:
        raise KeyError(f"Built-in AHI contract requires NPZ key '{key}'.")
    raw = np.asarray(npz[key])
    if raw.ndim != 0:
        raise ValueError(f"Built-in AHI contract requires NPZ key '{key}' to be scalar, got shape {raw.shape}.")
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"Built-in AHI contract requires NPZ key '{key}' to be finite, got {value}.")
    return value


def load_builtin_ahi_metadata(npz) -> tuple[float, float]:
    if "ah_event" not in npz:
        raise KeyError("Built-in AHI contract requires NPZ key 'ah_event'.")
    ahi_value = _load_scalar_npz_value(npz, "ahi")
    tst_value = _load_scalar_npz_value(npz, "tst")
    if ahi_value < 0:
        raise ValueError(f"Built-in AHI contract requires scalar 'ahi' >= 0, got {ahi_value}.")
    if tst_value <= 0:
        raise ValueError(f"Built-in AHI contract requires scalar 'tst' > 0, got {tst_value}.")
    return ahi_value, tst_value


def pad(x, max_len: int, pad_value: torch.types.Number = 0, dim: int = 0) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if x.shape[dim] == max_len:
        return x
    if x.shape[dim] > max_len:
        return x.narrow(dim, 0, max_len)
    pad_shape = list(x.shape)
    pad_shape[dim] = max_len - x.shape[dim]
    padding = torch.full(pad_shape, pad_value, device=x.device, dtype=x.dtype)
    return torch.concat([x, padding], dim)


def pad_batch(
    x: t.List[torch.Tensor],
    max_len: t.Union[int, None] = None,
    pad_value: torch.types.Number = 0,
    dim: int = 0,
) -> torch.Tensor:
    if max_len is None:
        max_len = max(y.shape[dim] for y in x)
    return torch.stack([pad(y, max_len, pad_value, dim) for y in x])


def _default_worker_count() -> int:
    cpu = os.cpu_count() or 8
    return min(64, cpu * 2)


def filter_valid_sample_indices(
    data: t.Sequence[t.Any],
    extractors: t.Mapping[str, t.Callable],
    tokenizers: t.Mapping[str, t.Callable],
    *,
    allow_missing_channels: bool,
    channel_names: t.Sequence[str] | None = None,
    channel_input_dims: t.Mapping[str, int] | None = None,
    channel_source_names: t.Mapping[str, t.Sequence[str]] | None = None,
    expand_source_branches: bool = False,
    min_channels: int = 2,
    tolerance: int = 1,
    max_workers: int | None = None,
) -> list[t.Any]:
    """
    Filter out samples with tokenized channel-length mismatches.
    - allow_missing_channels=True: keep samples with >= min_channels available channels
      and record available channels in SampleIndex.payload.
    - allow_missing_channels=False: require all configured channels to exist.
    """

    worker_count = max_workers or _default_worker_count()
    channel_names = list(channel_names or [])
    requires_builtin_ahi = "ahi" in extractors
    normalized_source_names = normalize_channel_source_names(channel_names, channel_source_names)
    store_channel_sources = bool(expand_source_branches or uses_explicit_channel_sources(normalized_source_names))

    def _available_from_npz(npz):
        return compute_channel_matches(
            npz,
            channel_names,
            channel_source_names=normalized_source_names,
            expand_source_branches=expand_source_branches,
        )

    samples_by_path: dict[t.Any, list[t.Any]] = {}
    for sample_index in data:
        path = getattr(sample_index, "path", None)
        samples_by_path.setdefault(path, []).append(sample_index)

    def process_path(path: str, samples: list[t.Any]) -> list[t.Any]:
        filtered_samples: list[t.Any] = []
        try:
            with load_npz(path) as npz:
                for sample_index in samples:
                    try:
                        if requires_builtin_ahi:
                            ahi_value, tst_value = load_builtin_ahi_metadata(npz)
                            metadata = getattr(sample_index, "metadata", None)
                            if isinstance(metadata, dict):
                                metadata["ahi"] = ahi_value
                                metadata["tst"] = tst_value

                        available, channel_sources = _available_from_npz(npz)
                        if allow_missing_channels:
                            if len(available) < min_channels:
                                logging.info(
                                    "[Skip] Not enough channels at %s: have=%d need>=%d. Meta: %s",
                                    sample_index.id,
                                    len(available),
                                    min_channels,
                                    getattr(sample_index, "metadata", {}),
                                )
                                continue
                            payload = {}
                            tokens = {}
                            for key in available:
                                source_name = choose_source_name(key, channel_sources=channel_sources)
                                payload[key], tokens[key] = load_channel_tokens(
                                    npz,
                                    channel_name=key,
                                    start=sample_index.start,
                                    end=sample_index.end,
                                    channel_input_dims=channel_input_dims,
                                    source_name=source_name,
                                )
                        else:
                            payload = {}
                            tokens = {}
                            for key in channel_names:
                                matched_sources = list(channel_sources.get(key, []))
                                required_sources = list(normalized_source_names[key])
                                explicit_source_names = required_sources != [key]
                                if explicit_source_names and not matched_sources:
                                    raise ValueError(
                                        f"Configured source_names for channel '{key}' were not found in NPZ."
                                    )
                                if (
                                    explicit_source_names
                                    and expand_source_branches
                                    and key != "ahi"
                                    and matched_sources != required_sources
                                ):
                                    raise ValueError(
                                        f"Missing required sources for channel '{key}'. "
                                        f"required={required_sources}, matched={matched_sources}"
                                    )

                                if expand_source_branches and normalized_source_names[key] != [key] and matched_sources:
                                    for source_name in normalized_source_names[key]:
                                        effective_name = make_effective_channel_name(key, source_name)
                                        payload[effective_name], tokens[effective_name] = load_channel_tokens(
                                            npz,
                                            channel_name=key,
                                            start=sample_index.start,
                                            end=sample_index.end,
                                            channel_input_dims=channel_input_dims,
                                            source_name=source_name,
                                        )
                                    continue

                                source_name = choose_source_name(key, channel_sources=channel_sources)
                                if explicit_source_names and source_name is None:
                                    raise ValueError(
                                        f"Configured source_names for channel '{key}' produced no usable source."
                                    )
                                payload[key], tokens[key] = load_channel_tokens(
                                    npz,
                                    channel_name=key,
                                    start=sample_index.start,
                                    end=sample_index.end,
                                    channel_input_dims=channel_input_dims,
                                    source_name=source_name,
                                )

                        if requires_builtin_ahi and not bool((tokens["ahi"].reshape(-1) != -1.0).any().item()):
                            logging.info(
                                "[Skip] Built-in AHI sample %s has no valid ah_event labels. Meta: %s",
                                getattr(sample_index, "id", "?"),
                                getattr(sample_index, "metadata", {}),
                            )
                            continue

                        lengths = [v.shape[0] for v in tokens.values()]
                        max_len, min_len = max(lengths), min(lengths)

                        if max_len - min_len <= tolerance:
                            payload_dict = getattr(sample_index, "payload", None)
                            if isinstance(payload_dict, dict):
                                if allow_missing_channels:
                                    payload_dict["available_channels"] = list(tokens.keys())
                                if store_channel_sources:
                                    payload_dict["channel_sources"] = {
                                        key: list(value) for key, value in channel_sources.items()
                                    }
                            filtered_samples.append(sample_index)
                            continue
                        logging.info(
                            "[Skip] Token length mismatch at %s: %s. Meta: %s",
                            sample_index.id,
                            lengths,
                            getattr(sample_index, "metadata", {}),
                        )
                    except Exception as e:
                        logging.info(f"[Skip] Error loading sample {getattr(sample_index, 'id', '?')}: {e}")
        except Exception as e:
            for sample_index in samples:
                logging.info(f"[Skip] Error loading sample {getattr(sample_index, 'id', '?')}: {e}")
        return filtered_samples

    filtered_data: list[t.Any] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(process_path, path, samples) for path, samples in samples_by_path.items()]
        iterator = as_completed(futures)
        iterator = tqdm(iterator, total=len(futures), desc="Validating samples", leave=False)
        for f in iterator:
            filtered_data.extend(f.result())

    logging.info(f"Loaded {len(filtered_data)} valid samples (from {len(data)} total)")
    return filtered_data


def window(tot_len: int, max_len: int, stride: int) -> np.ndarray:
    """Generate sliding windows on token indices."""
    left = np.arange(0, tot_len, stride) if stride > 0 else np.array([0])
    right = np.clip(left + max_len, 0, tot_len)
    return np.stack([left, right], axis=1)
