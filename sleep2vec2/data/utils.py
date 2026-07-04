from concurrent.futures import ThreadPoolExecutor
import logging
import os
import typing as t

import numpy as np
import torch
from tqdm import tqdm


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
    source_alias: str | None = None,
):
    """Slice one NPZ channel between token-aligned frame offsets."""
    if source_name is not None:
        sources = (str(source_name),)
    elif source_alias is not None:
        sources = (str(name), str(source_alias))
    else:
        sources = (str(name),)

    def extract(npz, start: int, end: int):
        s = start * frames_per_token
        e = end * frames_per_token

        for source in sources:
            if source in npz:
                arr = npz[source]
                break
        else:
            arr = npz[sources[0]]
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
    min_channels: int = 2,
    tolerance: int = 1,
    max_workers: int | None = None,
    channel_aliases: t.Mapping[str, str] | None = None,
) -> list[t.Any]:
    """
    Filter out samples with tokenized channel-length mismatches.
    - allow_missing_channels=True: keep samples with >= min_channels available channels.
    - allow_missing_channels=False: require all configured channels to exist.
    Accepted samples record available channels in SampleIndex.payload.
    """

    worker_count = max_workers or _default_worker_count()
    channel_names = list(channel_names or [])
    channel_aliases = {str(name): str(alias) for name, alias in (channel_aliases or {}).items()}
    requires_builtin_ahi = "ahi" in extractors

    def _available_from_npz(npz):
        available = []
        for ch in channel_names:
            if ch == "ahi":
                try:
                    load_builtin_ahi_metadata(npz)
                except Exception:
                    continue
                available.append(ch)
                continue
            alias = channel_aliases.get(ch)
            if ch in npz or (alias is not None and alias in npz):
                available.append(ch)
        return available

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

                        if allow_missing_channels:
                            available = _available_from_npz(npz)
                            if len(available) < min_channels:
                                logging.info(
                                    "[Skip] Not enough channels at %s: have=%d need>=%d. Meta: %s",
                                    sample_index.id,
                                    len(available),
                                    min_channels,
                                    getattr(sample_index, "metadata", {}),
                                )
                                continue
                            payload = {
                                key: extractors[key](npz, sample_index.start, sample_index.end) for key in available
                            }
                            tokens = {key: tokenizers[key](payload[key]) for key in available}
                        else:
                            payload = {
                                key: fn(npz, sample_index.start, sample_index.end) for key, fn in extractors.items()
                            }
                            tokens = {key: fn(payload[key]) for key, fn in tokenizers.items()}

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
                                payload_dict["available_channels"] = list(tokens.keys())
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
        iterator = tqdm(futures, total=len(futures), desc="Validating samples", leave=False)
        for f in iterator:
            filtered_data.extend(f.result())

    logging.info(f"Loaded {len(filtered_data)} valid samples (from {len(data)} total)")
    return filtered_data


def window(tot_len: int, max_len: int, stride: int) -> np.ndarray:
    """Generate sliding windows on token indices."""
    left = np.arange(0, tot_len, stride) if stride > 0 else np.array([0])
    right = np.clip(left + max_len, 0, tot_len)
    return np.stack([left, right], axis=1)
