from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import pickle
import typing as t

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sleep2wave.data.corruptions import apply_corruption
from sleep2wave.data.default_dataset import SampleIndex
from sleep2wave.data.generative_batch import collate_sleep2wave_generative
from sleep2wave.data.modalities import (
    CANONICAL_MODALITIES,
    EPOCH_SEC,
    MODALITY_ALIASES,
    MODALITY_SPECS,
    normalize_modality_name,
    validate_modality_sequence,
)
from sleep2wave.data.quality import resolve_availability_mask, resolve_quality_mask
from sleep2wave.data.utils import load_npz
from sleep2wave.preprocess.split_index_by_dataset import normalize_mask_frame

SLEEP2WAVE_SCHEMA_VERSION = 1
TASK_TYPES = {"restoration", "imputation", "translation", "partial_full"}


@dataclass(frozen=True)
class IndexColumnConfig:
    path_col: str = "path"
    duration_col: str = "duration"
    split_col: str = "split"
    subject_id_col: str = "subject_id"
    night_id_col: str = "night_id"
    source_col: str = "source"


def normalize_sample_index(item: t.Any) -> SampleIndex:
    if isinstance(item, SampleIndex):
        return item
    missing = [name for name in ("id", "path", "start", "end") if not hasattr(item, name)]
    if missing:
        raise ValueError(f"Preset item is missing SampleIndex fields: {missing}")
    return SampleIndex(
        id=item.id,
        path=item.path,
        start=item.start,
        end=item.end,
        payload=dict(getattr(item, "payload", {}) or {}),
        metadata=dict(getattr(item, "metadata", {}) or {}),
    )


def modality_mask_candidates(modality: str) -> list[str]:
    canonical = normalize_modality_name(modality)
    aliases = [alias for alias, target in MODALITY_ALIASES.items() if target == canonical]
    return [f"{canonical}_mask", *(f"{alias}_mask" for alias in aliases)]


def resolve_modality_mask_columns(df: pd.DataFrame, *, require_all: bool = True) -> dict[str, str]:
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for modality in CANONICAL_MODALITIES:
        match = next((candidate for candidate in modality_mask_candidates(modality) if candidate in df.columns), None)
        if match is None:
            missing.append(modality)
            continue
        resolved[modality] = match
    if require_all and missing:
        raise ValueError(f"Missing sleep2wave modality mask columns for: {missing}")
    return resolved


def resolve_npz_key(npz, modality: str, canonical_channel_map: dict[str, str] | None = None) -> str | None:
    mapped = (canonical_channel_map or {}).get(modality)
    aliases = [alias for alias, target in MODALITY_ALIASES.items() if target == modality]
    candidates = [mapped, modality, *aliases]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in npz:
            return candidate
    return None


def _row_path_value(row: pd.Series, column: str) -> str | None:
    if column not in row or pd.isna(row[column]):
        return None
    value = str(row[column])
    return value if value else None


def _derived_path_for_row(row: pd.Series, modality: str) -> str | None:
    return _row_path_value(row, f"{modality}_derived_path") or _row_path_value(row, "derived_path")


def _resolve_quality_key(npz, modality: str) -> str | None:
    candidates = (
        f"{modality}_quality_mask",
        f"{modality}_quality",
        f"{modality}_valid_mask",
    )
    return next((candidate for candidate in candidates if candidate in npz), None)


def _channel_has_window(npz, key: str, modality: str, end_epoch: int) -> bool:
    raw = np.asarray(npz[key])
    right = end_epoch * MODALITY_SPECS[modality].frames_per_epoch
    if raw.ndim == 1:
        return raw.shape[0] >= right
    if raw.ndim == 2 and raw.shape[1] == 1:
        return raw.shape[0] >= right
    if raw.ndim == 2:
        return raw.shape[1] >= right
    return False


def prepare_sleep2wave_index_frame(
    df: pd.DataFrame,
    *,
    columns: IndexColumnConfig,
) -> tuple[pd.DataFrame, IndexColumnConfig]:
    missing = [col for col in (columns.path_col, columns.duration_col, columns.split_col) if col not in df.columns]
    if missing:
        raise ValueError(f"sleep2wave index missing required columns: {missing}")

    df = df.copy()
    if columns.subject_id_col not in df.columns:
        df[columns.subject_id_col] = df[columns.path_col]

    if columns.night_id_col not in df.columns:
        df[columns.night_id_col] = df[columns.path_col]

    return df, IndexColumnConfig(
        path_col=columns.path_col,
        duration_col=columns.duration_col,
        split_col=columns.split_col,
        subject_id_col=columns.subject_id_col,
        night_id_col=columns.night_id_col,
        source_col=columns.source_col,
    )


def build_sample_indices_from_frame(
    df: pd.DataFrame,
    *,
    index_source: str,
    split: str | t.Sequence[str] | None = None,
    context_epochs: int,
    stride_epochs: int | None = None,
    columns: IndexColumnConfig = IndexColumnConfig(),
    require_all_masks: bool = False,
    num_workers: int = 1,
) -> list[SampleIndex]:
    if context_epochs <= 0:
        raise ValueError("context_epochs must be positive.")
    stride_epochs = context_epochs if stride_epochs is None else int(stride_epochs)
    if stride_epochs <= 0:
        raise ValueError("stride_epochs must be positive.")
    num_workers = int(num_workers)
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1.")

    df, columns = prepare_sleep2wave_index_frame(df, columns=columns)
    split_values = None
    if split is not None:
        split_values = {split} if isinstance(split, str) else set(split)
        df = df[df[columns.split_col].astype("string").isin(split_values)].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    mask_columns = resolve_modality_mask_columns(df, require_all=require_all_masks)
    mask_frame = normalize_mask_frame(df, list(mask_columns.values()))

    def process_row(row_number: int, row: pd.Series) -> list[SampleIndex]:
        duration = float(row[columns.duration_col])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Row {row_number} has invalid duration: {duration!r}")

        night_epoch_count = int(duration // EPOCH_SEC)
        if night_epoch_count < context_epochs:
            return []

        if columns.source_col in row and pd.notna(row[columns.source_col]):
            source = row[columns.source_col]
        else:
            source = index_source
        metadata = {
            "subject_id": row[columns.subject_id_col],
            "night_id": row[columns.night_id_col],
            "source": source,
            "split": row[columns.split_col],
            "path": row[columns.path_col],
        }
        for metadata_name in ("age", "sex"):
            if metadata_name in row and pd.notna(row[metadata_name]):
                metadata[metadata_name] = row[metadata_name]
        sample_id = row["id"] if "id" in row and pd.notna(row["id"]) else row_number

        if mask_columns and mask_frame.loc[row_number, list(mask_columns.values())].sum() == 0:
            raise ValueError(f"Row {row_number} has no available sleep2wave modalities.")

        row_samples: list[SampleIndex] = []
        with ExitStack() as stack:
            npz = stack.enter_context(load_npz(str(row[columns.path_col])))
            derived_npzs: dict[str, t.Any] = {}
            for modality in CANONICAL_MODALITIES:
                derived_path = _derived_path_for_row(row, modality)
                if derived_path and derived_path not in derived_npzs and Path(derived_path).exists():
                    derived_npzs[derived_path] = stack.enter_context(load_npz(derived_path))
            for start in range(0, night_epoch_count - context_epochs + 1, stride_epochs):
                end = start + context_epochs
                available_channels: list[str] = []
                canonical_channel_map: dict[str, str] = {}
                derived_channels: dict[str, str] = {}
                availability_mask_keys: dict[str, str] = {}
                quality_mask_keys: dict[str, str] = {}
                quality_mask_sources: dict[str, str] = {}
                for modality in CANONICAL_MODALITIES:
                    mask_col = mask_columns.get(modality)
                    if mask_col is not None and not bool(mask_frame.loc[row_number, mask_col]):
                        continue
                    npz_key = resolve_npz_key(npz, modality)
                    source_npz = npz
                    source_name = "primary"
                    derived_path = _derived_path_for_row(row, modality)
                    if (npz_key is None or not _channel_has_window(npz, npz_key, modality, end)) and derived_path:
                        derived_npz = derived_npzs.get(derived_path)
                        if derived_npz is not None:
                            derived_key = resolve_npz_key(derived_npz, modality)
                            if derived_key is not None and _channel_has_window(derived_npz, derived_key, modality, end):
                                npz_key = derived_key
                                source_npz = derived_npz
                                source_name = "derived"
                                derived_channels[modality] = derived_path
                    if npz_key is None or not _channel_has_window(source_npz, npz_key, modality, end):
                        continue
                    available_channels.append(modality)
                    canonical_channel_map[modality] = npz_key
                    if mask_col is not None:
                        availability_mask_keys[modality] = mask_col
                    quality_key = _resolve_quality_key(source_npz, modality)
                    if quality_key is not None:
                        quality_mask_keys[modality] = quality_key
                        quality_mask_sources[modality] = source_name
                if not available_channels:
                    continue
                row_samples.append(
                    SampleIndex(
                        id=sample_id,
                        path=str(row[columns.path_col]),
                        start=start,
                        end=end,
                        payload={
                            "sleep2wave_schema_version": SLEEP2WAVE_SCHEMA_VERSION,
                            "available_channels": available_channels,
                            "quality_mask_keys": quality_mask_keys,
                            "quality_mask_sources": quality_mask_sources,
                            "availability_mask_keys": availability_mask_keys,
                            "canonical_channel_map": canonical_channel_map,
                            "derived_channels": derived_channels,
                            "epoch_sec": EPOCH_SEC,
                            "sample_rates": {name: spec.sample_rate_hz for name, spec in MODALITY_SPECS.items()},
                            "frames_per_epoch": {name: spec.frames_per_epoch for name, spec in MODALITY_SPECS.items()},
                            "subject_id": row[columns.subject_id_col],
                            "night_id": row[columns.night_id_col],
                            "night_epoch_count": night_epoch_count,
                        },
                        metadata=metadata,
                    )
                )
        return row_samples

    rows = list(df.reset_index(drop=True).iterrows())
    samples: list[SampleIndex] = []
    progress_desc = "Building sleep2wave preset"

    if num_workers == 1:
        iterator = tqdm(rows, total=len(rows), desc=progress_desc, unit="row")
        for row_number, row in iterator:
            samples.extend(process_row(row_number, row))
    else:
        row_results: list[tuple[int, list[SampleIndex]]] = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_row = {executor.submit(process_row, row_number, row): row_number for row_number, row in rows}
            iterator = tqdm(as_completed(future_to_row), total=len(future_to_row), desc=progress_desc, unit="row")
            for future in iterator:
                row_results.append((future_to_row[future], future.result()))
        for _, row_samples in sorted(row_results, key=lambda item: item[0]):
            samples.extend(row_samples)

    if not samples:
        split_hint = f" for split {sorted(split_values)}" if split_values else ""
        raise ValueError(f"No sleep2wave generative windows were produced{split_hint}.")
    return samples


def build_sample_indices_from_index(
    index_path: str | Path,
    *,
    split: str | t.Sequence[str] | None = None,
    context_epochs: int,
    stride_epochs: int | None = None,
    columns: IndexColumnConfig = IndexColumnConfig(),
    require_all_masks: bool = False,
    num_workers: int = 1,
) -> list[SampleIndex]:
    path = Path(index_path)
    df = pd.read_csv(path, low_memory=False)
    return build_sample_indices_from_frame(
        df,
        index_source=str(path),
        split=split,
        context_epochs=context_epochs,
        stride_epochs=stride_epochs,
        columns=columns,
        require_all_masks=require_all_masks,
        num_workers=num_workers,
    )


def _load_preset(path: str | Path) -> list[SampleIndex]:
    with Path(path).open("rb") as f:
        loaded = pickle.load(f)
    if not isinstance(loaded, list):
        raise ValueError("sleep2wave preset must contain a list of SampleIndex objects.")
    return [normalize_sample_index(item) for item in loaded]


class Sleep2WaveGenerativeDataset(Dataset):
    def __init__(
        self,
        *,
        preset_path: str | Path | None = None,
        index: str | Path | None = None,
        split: str | t.Sequence[str] | None = None,
        context_epochs: int = 15,
        stride_epochs: int | None = None,
        condition_modalities: t.Sequence[str] | None = None,
        target_modalities: t.Sequence[str] | None = None,
        task_type: str = "translation",
        corruption_name: str | None = None,
        corruption_kwargs: dict[str, t.Any] | None = None,
        corruption_specs: dict[str, tuple[str, dict[str, t.Any]]] | None = None,
        condition_mask_npz: str | Path | None = None,
        seed: int = 0,
    ) -> None:
        if context_epochs <= 0:
            raise ValueError("context_epochs must be positive.")
        if task_type not in TASK_TYPES:
            raise ValueError(f"task_type must be one of {sorted(TASK_TYPES)}.")
        if (preset_path is None) == (index is None):
            raise ValueError("Exactly one of preset_path or index must be provided.")

        self.context_epochs = int(context_epochs)
        self.condition_modalities = (
            validate_modality_sequence(list(condition_modalities or []), allow_aliases=True)
            if condition_modalities
            else []
        )
        self.target_modalities = (
            validate_modality_sequence(list(target_modalities or []), allow_aliases=True) if target_modalities else []
        )
        self.task_type = task_type
        self.corruption_name = corruption_name
        self.corruption_kwargs = dict(corruption_kwargs or {})
        self.corruption_specs = {
            validate_modality_sequence([modality], allow_aliases=False)[0]: (name, dict(kwargs))
            for modality, (name, kwargs) in (corruption_specs or {}).items()
        }
        if corruption_name is not None:
            for modality in self.condition_modalities or CANONICAL_MODALITIES:
                self.corruption_specs.setdefault(modality, (corruption_name, dict(self.corruption_kwargs)))
        self.condition_mask_npz = Path(condition_mask_npz) if condition_mask_npz is not None else None
        if self.condition_mask_npz is not None and not self.condition_mask_npz.exists():
            raise FileNotFoundError(f"Condition mask NPZ not found: {self.condition_mask_npz}")
        self.seed = int(seed)

        if preset_path is not None:
            data = _load_preset(preset_path)
            if split is not None:
                split_values = {split} if isinstance(split, str) else set(split)
                data = [item for item in data if item.metadata.get("split") in split_values]
        else:
            data = build_sample_indices_from_index(
                index,
                split=split,
                context_epochs=context_epochs,
                stride_epochs=stride_epochs,
            )

        if not data:
            raise ValueError("No sleep2wave generative samples are available.")
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def dataloader(self, **kwargs: t.Any) -> DataLoader:
        return DataLoader(self, collate_fn=collate_sleep2wave_generative, **kwargs)

    def _declared_available(self, sample: SampleIndex) -> set[str] | None:
        raw = sample.payload.get("available_channels")
        if raw is None:
            return None
        return set(validate_modality_sequence(list(raw), allow_aliases=True))

    def _resolve_npz_key(self, npz, sample: SampleIndex, modality: str) -> str | None:
        return resolve_npz_key(npz, modality, sample.payload.get("canonical_channel_map", {}))

    def _load_modality_signal(
        self,
        primary_npz,
        sample: SampleIndex,
        modality: str,
    ) -> tuple[torch.Tensor, bool, str]:
        key = self._resolve_npz_key(primary_npz, sample, modality)
        if key is not None:
            return self._slice_signal(primary_npz, key, modality, sample.start, sample.end), True, "primary"

        derived_path = sample.payload.get("derived_channels", {}).get(modality)
        if derived_path is None:
            return self._zero_signal(modality), False, "primary"
        with load_npz(derived_path) as derived_npz:
            derived_key = self._resolve_npz_key(derived_npz, sample, modality)
            if derived_key is None:
                return self._zero_signal(modality), False, "primary"
            return self._slice_signal(derived_npz, derived_key, modality, sample.start, sample.end), True, "derived"

    def _zero_signal(self, modality: str) -> torch.Tensor:
        spec = MODALITY_SPECS[modality]
        return torch.zeros((self.context_epochs, 1, spec.frames_per_epoch), dtype=torch.float32)

    def _slice_signal(self, npz, key: str, modality: str, start: int, end: int) -> torch.Tensor:
        spec = MODALITY_SPECS[modality]
        left = start * spec.frames_per_epoch
        right = end * spec.frames_per_epoch
        raw = np.asarray(npz[key])

        if raw.ndim == 1:
            if raw.shape[0] < right:
                raise ValueError(f"Channel '{key}' is too short for epochs {start}:{end}.")
            segment = raw[left:right][None, :]
        elif raw.ndim == 2 and raw.shape[1] == 1:
            if raw.shape[0] < right:
                raise ValueError(f"Channel '{key}' is too short for epochs {start}:{end}.")
            segment = raw[left:right, 0][None, :]
        elif raw.ndim == 2 and raw.shape[1] >= right:
            segment = raw[:, left:right]
        else:
            raise ValueError(f"Channel '{key}' must be 1D, [T, 1], or channel-first [C, T], got {raw.shape}.")

        expected = self.context_epochs * spec.frames_per_epoch
        if segment.shape[-1] != expected:
            raise ValueError(f"Channel '{key}' yielded {segment.shape[-1]} frames, expected {expected}.")
        tensor = torch.as_tensor(segment, dtype=torch.float32)
        return tensor.reshape(tensor.shape[0], self.context_epochs, spec.frames_per_epoch).permute(1, 0, 2)

    def _maybe_corrupt(self, signal: torch.Tensor, modality: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.corruption_specs.get(modality)
        if spec is None:
            return signal.clone(), torch.zeros_like(signal, dtype=torch.bool)

        name, kwargs = spec
        kwargs = dict(kwargs)
        if "window_frames" not in kwargs and name in {
            "contiguous_window_mask",
            "flatline_dropout",
            "spo2_plateau_dropout",
            "rpeak_drop_or_jitter_for_ibi",
            "belt_failure",
        }:
            kwargs["window_frames"] = max(1, signal.shape[-1] // 10)
        return apply_corruption(name, signal, seed=self.seed + seed, **kwargs)

    def _condition_mask_key(self, mask_npz, modality: str) -> str | None:
        candidates = (
            f"{modality}_mask",
            modality,
            f"condition/{modality}_mask",
            f"condition/{modality}",
        )
        return next((candidate for candidate in candidates if candidate in mask_npz), None)

    def _reshape_condition_mask(
        self,
        raw: np.ndarray,
        sample: SampleIndex,
        modality: str,
        clean: torch.Tensor,
    ) -> torch.Tensor:
        frames_per_epoch = MODALITY_SPECS[modality].frames_per_epoch
        frame_left = sample.start * frames_per_epoch
        frame_right = sample.end * frames_per_epoch
        epoch_count, channel_count = clean.shape[:2]
        array = np.asarray(raw)

        if array.ndim == 1:
            if array.shape[0] >= frame_right:
                segment = array[frame_left:frame_right].reshape(epoch_count, 1, frames_per_epoch)
            elif array.shape[0] >= sample.end:
                segment = array[sample.start : sample.end].reshape(epoch_count, 1, 1)
            else:
                raise ValueError(
                    f"Condition mask for '{modality}' is too short for epochs {sample.start}:{sample.end}."
                )
        elif array.ndim == 2:
            if array.shape == (epoch_count, frames_per_epoch):
                segment = array[:, None, :]
            elif array.shape[0] >= frame_right and array.shape[1] == 1:
                segment = array[frame_left:frame_right, 0].reshape(epoch_count, 1, frames_per_epoch)
            elif array.shape[1] >= frame_right:
                segment = array[:, frame_left:frame_right].reshape(array.shape[0], epoch_count, frames_per_epoch)
                segment = segment.transpose(1, 0, 2)
            elif array.shape[0] >= sample.end:
                segment = array[sample.start : sample.end, :, None]
            else:
                raise ValueError(f"Condition mask for '{modality}' has unsupported shape {array.shape}.")
        elif array.ndim == 3 and array.shape[0] >= sample.end:
            segment = array[sample.start : sample.end]
        else:
            raise ValueError(f"Condition mask for '{modality}' has unsupported shape {array.shape}.")

        mask = torch.as_tensor(segment, dtype=torch.bool)
        if mask.shape[0] != epoch_count:
            raise ValueError(f"Condition mask for '{modality}' must span {epoch_count} epochs.")
        if mask.shape[1] == 1 and channel_count != 1:
            mask = mask.expand(epoch_count, channel_count, mask.shape[2])
        elif mask.shape[1] != channel_count:
            raise ValueError(f"Condition mask for '{modality}' must have 1 or {channel_count} channels.")
        if mask.shape[2] == 1 and frames_per_epoch != 1:
            mask = mask.expand(epoch_count, channel_count, frames_per_epoch)
        elif mask.shape[2] != frames_per_epoch:
            raise ValueError(f"Condition mask for '{modality}' must have {frames_per_epoch} frames per epoch.")
        return mask

    def _load_condition_mask(
        self,
        mask_npz,
        sample: SampleIndex,
        modality: str,
        clean: torch.Tensor,
    ) -> torch.Tensor | None:
        key = self._condition_mask_key(mask_npz, modality)
        if key is None:
            return None
        return self._reshape_condition_mask(mask_npz[key], sample, modality, clean)

    def __getitem__(self, idx: int) -> dict[str, t.Any]:
        sample = self.data[idx]
        if sample.end - sample.start != self.context_epochs:
            raise ValueError(
                f"Sample {sample.id} spans {sample.end - sample.start} epochs; "
                f"expected context_epochs={self.context_epochs}."
            )

        declared_available = self._declared_available(sample)
        clean_signals: dict[str, torch.Tensor] = {}
        observed_signals: dict[str, torch.Tensor] = {}
        availability_mask: dict[str, torch.Tensor] = {}
        quality_mask: dict[str, torch.Tensor] = {}
        corruption_mask: dict[str, torch.Tensor] = {}

        with ExitStack() as stack:
            npz = stack.enter_context(load_npz(sample.path))
            condition_mask_npz = (
                stack.enter_context(load_npz(str(self.condition_mask_npz)))
                if self.condition_mask_npz is not None
                else None
            )
            for modality_index, modality in enumerate(CANONICAL_MODALITIES):
                if declared_available is not None and modality not in declared_available:
                    clean = self._zero_signal(modality)
                    available = False
                    source_name = "primary"
                else:
                    clean, available, source_name = self._load_modality_signal(npz, sample, modality)

                availability_key = sample.payload.get("availability_mask_keys", {}).get(modality)
                quality_key = sample.payload.get("quality_mask_keys", {}).get(modality)
                clean_signals[modality] = clean
                availability_mask[modality] = resolve_availability_mask(
                    npz,
                    availability_key if availability_key is not None and availability_key in npz else None,
                    sample.start,
                    sample.end,
                    available=available,
                )
                if source_name != "derived":
                    quality_mask[modality] = resolve_quality_mask(
                        npz,
                        quality_key if quality_key is not None and quality_key in npz else None,
                        sample.start,
                        sample.end,
                        available=available,
                    )
                else:
                    with load_npz(sample.payload.get("derived_channels", {}).get(modality)) as mask_npz:
                        quality_mask[modality] = resolve_quality_mask(
                            mask_npz,
                            quality_key if quality_key is not None and quality_key in mask_npz else None,
                            sample.start,
                            sample.end,
                            available=available,
                        )
                if available:
                    corrupt_seed = idx * len(CANONICAL_MODALITIES) + modality_index
                    observed, corrupt_mask = self._maybe_corrupt(clean, modality, corrupt_seed)
                    if condition_mask_npz is not None and modality in self.condition_modalities:
                        external_mask = self._load_condition_mask(condition_mask_npz, sample, modality, clean)
                        if external_mask is not None:
                            observed = observed.clone()
                            observed[external_mask] = 0.0
                            corrupt_mask = corrupt_mask | external_mask
                else:
                    observed = clean.clone()
                    corrupt_mask = torch.zeros_like(clean, dtype=torch.bool)
                observed_signals[modality] = observed
                corruption_mask[modality] = corrupt_mask

        night_epoch_count = int(sample.payload.get("night_epoch_count") or sample.end)
        denom = max(night_epoch_count - 1, 1)
        epoch_index = torch.arange(sample.start, sample.end, dtype=torch.long)
        night_position = epoch_index.to(torch.float32) / float(denom)

        metadata = {
            "id": sample.id,
            "path": sample.path,
            "subject_id": sample.payload.get("subject_id", sample.metadata.get("subject_id")),
            "night_id": sample.payload.get("night_id", sample.metadata.get("night_id")),
            "source": sample.metadata.get("source"),
            "split": sample.metadata.get("split"),
        }

        return {
            "clean_signals": clean_signals,
            "observed_signals": observed_signals,
            "availability_mask": availability_mask,
            "quality_mask": quality_mask,
            "corruption_mask": corruption_mask,
            "epoch_index": epoch_index,
            "night_position": night_position,
            "metadata": metadata,
            "condition_modalities": list(self.condition_modalities),
            "target_modalities": list(self.target_modalities),
            "task_type": self.task_type,
        }


__all__ = [
    "IndexColumnConfig",
    "SLEEP2WAVE_SCHEMA_VERSION",
    "Sleep2WaveGenerativeDataset",
    "build_sample_indices_from_frame",
    "build_sample_indices_from_index",
    "collate_sleep2wave_generative",
    "modality_mask_candidates",
    "normalize_sample_index",
    "prepare_sleep2wave_index_frame",
    "resolve_modality_mask_columns",
    "resolve_npz_key",
]
