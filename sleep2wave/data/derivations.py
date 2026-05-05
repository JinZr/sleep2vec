from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import typing as t

import numpy as np
import pandas as pd

from sleep2wave.data.generative_dataset import resolve_npz_key
from sleep2wave.data.modalities import MODALITY_SPECS
from sleep2wave.data.utils import load_npz


@dataclass(frozen=True)
class DerivationJob:
    row_id: int | str
    path: str
    split: str
    subject_id: int | str
    night_id: int | str
    output_path: str


def validate_subject_split_boundaries(
    df: pd.DataFrame,
    *,
    subject_id_col: str = "subject_id",
    split_col: str = "split",
) -> None:
    missing = [col for col in (subject_id_col, split_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Derivation index missing required columns: {missing}")
    if df[subject_id_col].isna().any():
        offenders = df.index[df[subject_id_col].isna()].astype(str).tolist()
        raise ValueError(f"Derivation index contains missing {subject_id_col} values: {offenders}")
    if df[split_col].isna().any():
        offenders = df.index[df[split_col].isna()].astype(str).tolist()
        raise ValueError(f"Derivation index contains missing {split_col} values: {offenders}")

    split_counts = df.groupby(subject_id_col)[split_col].nunique(dropna=False)
    offenders = sorted(str(subject_id) for subject_id, count in split_counts.items() if count > 1)
    if offenders:
        raise ValueError(f"Subjects appear in multiple splits: {offenders}")


def plan_derivation_jobs(
    df: pd.DataFrame,
    *,
    output_dir: str | Path,
    path_col: str = "path",
    split_col: str = "split",
    subject_id_col: str = "subject_id",
    night_id_col: str = "night_id",
) -> list[DerivationJob]:
    required = [path_col, split_col, subject_id_col, night_id_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Derivation index missing required columns: {missing}")
    validate_subject_split_boundaries(df, subject_id_col=subject_id_col, split_col=split_col)

    output_root = Path(output_dir)
    jobs: list[DerivationJob] = []
    for row_number, row in df.reset_index(drop=True).iterrows():
        row_id = row["id"] if "id" in row and pd.notna(row["id"]) else row_number
        subject_id = row[subject_id_col]
        night_id = row[night_id_col]
        jobs.append(
            DerivationJob(
                row_id=row_id,
                path=str(row[path_col]),
                split=str(row[split_col]),
                subject_id=subject_id,
                night_id=night_id,
                output_path=str(output_root / f"subject-{subject_id}_night-{night_id}_derived.npz"),
            )
        )
    return jobs


def require_derivation_backend(enabled_derivations: t.Sequence[str]) -> None:
    return None


def _as_channel_time(raw: np.ndarray) -> np.ndarray:
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0][None, :]
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected 1D, [T, 1], or channel-first [C, T] signal, got {array.shape}.")


def _single_series(raw: np.ndarray) -> np.ndarray:
    return np.nanmean(_as_channel_time(raw), axis=0).astype(np.float32)


def _local_peak_indices(signal: np.ndarray, *, sample_rate_hz: int) -> np.ndarray:
    clean = np.nan_to_num(signal.astype(np.float32), nan=0.0)
    if clean.size < 3:
        return np.asarray([], dtype=np.int64)
    centered = clean - float(np.median(clean))
    scale = float(np.std(centered))
    if scale <= 0.0:
        return np.asarray([], dtype=np.int64)
    threshold = float(np.median(centered) + 0.5 * scale)
    candidates = np.flatnonzero((centered[1:-1] > centered[:-2]) & (centered[1:-1] >= centered[2:])) + 1
    candidates = candidates[centered[candidates] > threshold]
    if candidates.size <= 1:
        return candidates.astype(np.int64)
    refractory = max(int(0.3 * sample_rate_hz), 1)
    ordered = sorted(candidates.tolist(), key=lambda idx: float(centered[idx]), reverse=True)
    kept: list[int] = []
    for idx in ordered:
        if all(abs(idx - previous) >= refractory for previous in kept):
            kept.append(idx)
    return np.asarray(sorted(kept), dtype=np.int64)


def derive_ibi_from_ecg(ecg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_rate = MODALITY_SPECS["ecg"].sample_rate_hz
    target_rate = MODALITY_SPECS["ibi"].sample_rate_hz
    series = _single_series(ecg)
    target_frames = int(np.floor(series.size / sample_rate * target_rate))
    if target_frames <= 0:
        raise ValueError("ECG signal is too short to derive IBI.")
    peaks = _local_peak_indices(series, sample_rate_hz=sample_rate)
    if peaks.size < 2:
        return np.zeros(target_frames, dtype=np.float32), np.zeros(target_frames, dtype=bool)
    peak_times = peaks.astype(np.float32) / float(sample_rate)
    rr = np.diff(peak_times).astype(np.float32)
    rr_times = peak_times[1:]
    grid = np.arange(target_frames, dtype=np.float32) / float(target_rate)
    ibi = np.interp(grid, rr_times, rr, left=rr[0], right=rr[-1]).astype(np.float32)
    quality = np.isfinite(ibi) & (ibi > 0.25) & (ibi < 3.0)
    return np.nan_to_num(ibi, nan=0.0).astype(np.float32), quality.astype(bool)


def _resample_to_rate(signal: np.ndarray, *, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return signal.astype(np.float32)
    duration = signal.size / float(source_rate)
    target_frames = int(np.floor(duration * target_rate))
    if target_frames <= 0:
        raise ValueError("Signal is too short to resample.")
    source_grid = np.arange(signal.size, dtype=np.float32) / float(source_rate)
    target_grid = np.arange(target_frames, dtype=np.float32) / float(target_rate)
    return np.interp(target_grid, source_grid, signal).astype(np.float32)


def derive_resp_from_signal(signal: np.ndarray, *, source_rate_hz: int) -> tuple[np.ndarray, np.ndarray]:
    target_rate = MODALITY_SPECS["resp"].sample_rate_hz
    series = _resample_to_rate(_single_series(signal), source_rate=source_rate_hz, target_rate=target_rate)
    finite = np.isfinite(series)
    if not finite.any():
        return np.zeros_like(series, dtype=np.float32), np.zeros_like(series, dtype=bool)
    median = float(np.median(series[finite]))
    p75, p25 = np.percentile(series[finite], [75, 25])
    scale = float(p75 - p25)
    if scale <= 0.0:
        scale = float(np.std(series[finite])) or 1.0
    resp = ((series - median) / scale).astype(np.float32)
    return np.nan_to_num(resp, nan=0.0).astype(np.float32), finite.astype(bool)


def _epoch_quality_mask(frame_quality: np.ndarray, *, modality: str) -> np.ndarray:
    frames_per_epoch = MODALITY_SPECS[modality].frames_per_epoch
    values = np.asarray(frame_quality, dtype=bool).reshape(-1)
    epoch_count = values.size // frames_per_epoch
    if epoch_count == 0:
        return np.zeros((0,), dtype=bool)
    return values[: epoch_count * frames_per_epoch].reshape(epoch_count, frames_per_epoch).all(axis=1)


def derive_record_channels(
    *,
    input_path: str | Path,
    output_path: str | Path,
    derive: t.Sequence[str],
) -> Path:
    requested = set(derive)
    unknown = sorted(requested - {"ibi", "resp"})
    if unknown:
        raise ValueError(f"Unsupported sleep2wave derivations: {unknown}")
    if not requested:
        return Path(output_path)

    payload: dict[str, np.ndarray] = {}
    with load_npz(str(input_path)) as npz:
        if "ibi" in requested:
            ecg_key = resolve_npz_key(npz, "ecg")
            if ecg_key is None:
                raise ValueError(f"Cannot derive IBI without ECG in {input_path}")
            ibi, quality = derive_ibi_from_ecg(npz[ecg_key])
            payload["ibi"] = ibi
            payload["ibi_quality_mask"] = _epoch_quality_mask(quality, modality="ibi")

        if "resp" in requested:
            belt_key = resolve_npz_key(npz, "belt")
            airflow_key = resolve_npz_key(npz, "airflow")
            if belt_key is not None:
                source_key = belt_key
                source_rate = MODALITY_SPECS["belt"].sample_rate_hz
            elif airflow_key is not None:
                source_key = airflow_key
                source_rate = MODALITY_SPECS["airflow"].sample_rate_hz
            else:
                raise ValueError(f"Cannot derive RESP without belt or airflow in {input_path}")
            resp, quality = derive_resp_from_signal(npz[source_key], source_rate_hz=source_rate)
            payload["resp"] = resp
            payload["resp_quality_mask"] = _epoch_quality_mask(quality, modality="resp")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return path


def run_derivation_jobs(jobs: t.Sequence[DerivationJob], *, derive: t.Sequence[str]) -> list[Path]:
    return [derive_record_channels(input_path=job.path, output_path=job.output_path, derive=derive) for job in jobs]


__all__ = [
    "DerivationJob",
    "derive_ibi_from_ecg",
    "derive_record_channels",
    "derive_resp_from_signal",
    "plan_derivation_jobs",
    "require_derivation_backend",
    "run_derivation_jobs",
    "validate_subject_split_boundaries",
]
