#!/usr/bin/env python3
"""Preprocess blood-pressure wearable CSV exports into single-channel NPZ + index CSV.

Run with the project environment, for example:

  conda run -n sleep2vec python preprocess/ppg_formant_one_channel.py

Raw layout (default root):

  {raw_root}/{device}/{ssoid}/{ssoid}_{timestamp}.csv

Outputs:

  {out_root}/{device}/{ssoid}/{ssoid}_{timestamp}.npz

Each NPZ stores 1D float arrays for available modalities: green mean ``GR`` (mean of all raw columns
that start with ``g`` except ``gyro*``), infrared mean ``IR`` (mean of all columns starting with
``ir``), plus ``gyro`` and ``acc`` when present. Scalar ``duration`` is the original recording length
in seconds (from sample count and inferred rate).

CSV files are processed in parallel (``ProcessPoolExecutor``, ``--workers``); index rows are appended
from the main process as workers finish.

Pipeline per channel: treat raw values 0 and 10000 as sentinels (interpolate them away); if any channel
exceeds ``round(inferred_fs_hz)`` such samples (e.g. 100 at 100 Hz, 250 at 250 Hz), the whole CSV is
skipped with a warning. Then: resample to 100 Hz ->
4th-order Chebyshev I bandpass 0.4–12 Hz -> remove flatline runs -> z-score. If flatline segments are
removed, the source CSV path is logged.

"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d

LOG = logging.getLogger("ppg_formant_one_channel")

# Raw CSV sentinel outliers: interpolate away; per-channel skip threshold scales with inferred raw fs (Hz).
_SENTINEL_VALUES: tuple[float, float] = (0.0, 10000.0)

# Nominal (duration_s, fs_hz) -> expected sample count.
_NOMINAL_LAYOUTS: tuple[tuple[float, float], ...] = (
    (30.0, 100.0),
    (30.0, 250.0),
    (60.0, 100.0),
    (60.0, 250.0),
)

_INDEX_COLUMNS: list[str] = [
    "path",
    "dataset",
    "session_id",
    "patient_id",
    "duration",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/home/data/data_cubefs/血压原始数据"),
        help="Root directory containing device/ssoid/*.csv",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/notebook/data/personal/S9063410/bp_data_one_channel"),
        help="Output root for NPZ mirrors",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("/home/notebook/data/personal/S9063410/bp_data_one_channel/index.csv"),
        help="Path to write aggregated metadata CSV",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Process at most 50 CSV files per device (for debugging)",
    )
    p.add_argument(
        "--debug-limit",
        type=int,
        default=50,
        help="Max files per device when --debug is set (default: 50)",
    )
    p.add_argument(
        "--flatline-window-sec",
        type=float,
        default=1,
        help="Rolling window length (seconds at 100 Hz) for flatline detection after filtering",
    )
    p.add_argument(
        "--flatline-rel-tol",
        type=float,
        default=1e-5,
        help="Relative peak-to-peak tolerance vs median(|x|)+1e-12 inside the window",
    )
    p.add_argument(
        "--min-keep-sec",
        type=float,
        default=2.0,
        help="Drop a channel if it is shorter than this many seconds at 100 Hz after flatline removal",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ files",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker processes for CSV processing (default: 16; use 1 to disable parallelism)",
    )
    p.add_argument(
        "--metadata-append",
        action="store_true",
        help="Append metadata rows to an existing CSV (no truncate); omit header if file already has content",
    )
    p.add_argument(
        "--path-queue-size",
        type=int,
        default=0,
        help=(
            "Max queued (device, csv_path) pairs (0 = auto: max(64, 8*workers)); "
            "bounded queue back-pressures the directory walker"
        ),
    )
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args()


def infer_fs_duration(num_samples: int) -> tuple[float, float]:
    """Infer sampling rate (Hz) and recording duration (s) from sample count."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    best_err: int | None = None
    best: tuple[float, float] | None = None
    for dur_s, fs_hz in _NOMINAL_LAYOUTS:
        expected = int(round(dur_s * fs_hz))
        err = abs(num_samples - expected)
        if best_err is None or err < best_err:
            best_err = err
            best = (fs_hz, dur_s)

    assert best is not None and best_err is not None
    expected_n = int(round(best[1] * best[0]))
    tol = max(50, int(0.02 * max(expected_n, 1)))
    if best_err <= tol:
        return best[0], best[1]

    fs_hz, dur_nom = best
    dur_s = float(num_samples) / float(fs_hz)
    LOG.warning(
        "Length %d mismatches nominal %.0fs @ %.0fHz (err=%d, tol=%d); using fs=%.1f Hz and duration=%.3f s",
        num_samples,
        dur_nom,
        fs_hz,
        best_err,
        tol,
        fs_hz,
        dur_s,
    )
    return fs_hz, dur_s


def _resample_to_100hz(x: np.ndarray, fs_in: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if math.isclose(fs_in, 100.0, rel_tol=0.0, abs_tol=0.01):
        return x
    if math.isclose(fs_in, 250.0, rel_tol=0.0, abs_tol=0.5):
        return signal.resample_poly(x, up=2, down=5)
    # General rational approximation via polyphase resampling target.
    from fractions import Fraction

    frac = Fraction(100 / fs_in).limit_denominator(1000)
    return signal.resample_poly(x, up=frac.numerator, down=frac.denominator)


def _design_bandpass_sos(fs: float = 100.0) -> np.ndarray:
    # 4th-order Chebyshev Type I, 0.4–12 Hz, passband ripple 1 dB.
    return signal.cheby1(4, 1, [0.4, 12.0], btype="band", fs=fs, output="sos")


_BANDPASS_SOS = _design_bandpass_sos(100.0)


def _bandpass(x: np.ndarray) -> np.ndarray:
    if x.size < 32:
        return x
    return signal.sosfiltfilt(_BANDPASS_SOS, x)


def _flatline_mask_at_100hz(
    x: np.ndarray,
    window_sec: float,
    rel_tol: float,
) -> np.ndarray:
    """True where center sample belongs to a locally flat stretch (after preprocessing)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    win = max(3, int(round(window_sec * 100.0)))
    if win % 2 == 0:
        win += 1
    win = min(win, n)
    mx = maximum_filter1d(x, size=win, mode="nearest")
    mn = minimum_filter1d(x, size=win, mode="nearest")
    spread = mx - mn
    scale = float(np.nanmedian(np.abs(x))) + 1e-12
    return spread < (rel_tol * scale)


def _strip_flat_segments(x: np.ndarray, flat: np.ndarray) -> tuple[np.ndarray, bool]:
    """Remove samples where ``flat`` is True; returns (shortened_array, any_removed)."""
    if flat.size != x.size:
        raise ValueError("flat mask length mismatch")
    keep = ~flat
    removed = bool(np.any(flat))
    return x[keep], removed


def _zscore(x: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    mu = float(np.nanmean(x))
    sigma = float(np.nanstd(x))
    if not math.isfinite(sigma) or sigma < 1e-12:
        return None
    return (x - mu) / sigma


def _sentinel_outlier_mask(v: np.ndarray) -> np.ndarray:
    """True at finite samples equal to exactly 0 or 10000 (raw device sentinels)."""
    v = np.asarray(v, dtype=np.float64)
    a, b = _SENTINEL_VALUES
    return np.isfinite(v) & ((v == a) | (v == b))


def _sentinel_outlier_count(values: pd.Series) -> int:
    v = pd.to_numeric(values, errors="coerce").astype(np.float64).to_numpy()
    if v.size == 0:
        return 0
    return int(np.sum(_sentinel_outlier_mask(v)))


def _green_raw_column_names(df: pd.DataFrame) -> list[str]:
    """Raw CSV column names for green PPG (``g*``), excluding ``gyro*``."""
    return [c for c in df.columns if isinstance(c, str) and c.startswith("g") and not c.startswith("gyro")]


def _ir_raw_column_names(df: pd.DataFrame) -> list[str]:
    """Raw CSV column names for infrared PPG (``ir*``)."""
    return [c for c in df.columns if isinstance(c, str) and c.startswith("ir")]


def _sentinel_max_count_for_fs_hz(fs_hz: float) -> int:
    """Max allowed sentinel samples per channel: ``round(fs_hz)`` (e.g. 100 @ 100 Hz, 250 @ 250 Hz)."""
    return max(1, int(round(float(fs_hz))))


def _sentinel_check_df_or_skip(df: pd.DataFrame, csv_path: Path, max_sentinel_per_channel: int) -> bool:
    """Return False if any present per-column sentinel count exceeds the limit (caller skips sample)."""
    cols = list(_green_raw_column_names(df)) + list(_ir_raw_column_names(df))
    for c in ("gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"):
        if c in df.columns:
            cols.append(c)
    for col in cols:
        n_bad = _sentinel_outlier_count(df[col])
        if n_bad > max_sentinel_per_channel:
            LOG.warning(
                "Skipping sample %s: channel %s has %d sentinel outliers (%s/%s), max allowed %d",
                csv_path.as_posix(),
                col,
                n_bad,
                _SENTINEL_VALUES[0],
                _SENTINEL_VALUES[1],
                max_sentinel_per_channel,
            )
            return False
    return True


def _clean_series(values: pd.Series) -> np.ndarray:
    v = pd.to_numeric(values, errors="coerce").astype(np.float64)
    v = v.to_numpy()
    if np.all(~np.isfinite(v)):
        return np.array([], dtype=np.float64)
    v = v.copy()
    v[_sentinel_outlier_mask(v)] = np.nan
    s = pd.Series(v)
    s = s.interpolate(limit_direction="both")
    s = s.ffill().bfill()
    out = s.to_numpy(dtype=np.float64)
    out[~np.isfinite(out)] = 0.0
    return out


def _mean_cleaned_columns(df: pd.DataFrame, col_names: list[str]) -> np.ndarray | None:
    """Per-column ``_clean_series``, align to min length, return row-wise mean (float64)."""
    if not col_names:
        return None
    arrs = [_clean_series(df[c]) for c in col_names]
    n = min((a.size for a in arrs), default=0)
    if n == 0:
        return None
    stacked = np.stack([a[:n].astype(np.float64, copy=False) for a in arrs], axis=1)
    return np.nanmean(stacked, axis=1)


def process_channel(
    x: np.ndarray,
    fs_in: float,
    flatline_window_sec: float,
    flatline_rel_tol: float,
    min_keep_samples: int,
    source_path: Path,
    channel_name: str,
) -> tuple[np.ndarray | None, bool]:
    """Returns (processed_1d_or_none, stripped_flatline)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return None, False

    y = _resample_to_100hz(x, fs_in)
    y = _bandpass(y)

    flat = _flatline_mask_at_100hz(y, window_sec=flatline_window_sec, rel_tol=flatline_rel_tol)
    y, stripped = _strip_flat_segments(y, flat)
    if stripped:
        LOG.warning(
            "Stripped flatline segments from channel %s in %s (post-filter length was %d)",
            channel_name,
            source_path.as_posix(),
            flat.size,
        )

    if y.size < min_keep_samples:
        LOG.warning(
            "Dropping channel %s in %s: length %d < min_keep %d after flatline removal",
            channel_name,
            source_path.as_posix(),
            y.size,
            min_keep_samples,
        )
        return None, stripped

    z = _zscore(y)
    if z is None:
        LOG.warning("Skipping channel %s in %s after flatline removal (zero variance)", channel_name, source_path)
        return None, stripped
    return z.astype(np.float32), stripped


def _gyro_mag(df: pd.DataFrame) -> np.ndarray | None:
    cols = ("gyro_x", "gyro_y", "gyro_z")
    if not all(c in df.columns for c in cols):
        return None
    gx = _clean_series(df["gyro_x"])
    gy = _clean_series(df["gyro_y"])
    gz = _clean_series(df["gyro_z"])
    n = min(gx.size, gy.size, gz.size)
    if n == 0:
        return None
    gx, gy, gz = gx[:n], gy[:n], gz[:n]
    return np.sqrt(gx * gx + gy * gy + gz * gz)


def _acc_mag(df: pd.DataFrame) -> np.ndarray | None:
    cols = ("acc_x", "acc_y", "acc_z")
    if not all(c in df.columns for c in cols):
        return None
    ax = _clean_series(df["acc_x"])
    ay = _clean_series(df["acc_y"])
    az = _clean_series(df["acc_z"])
    n = min(ax.size, ay.size, az.size)
    if n == 0:
        return None
    ax, ay, az = ax[:n], ay[:n], az[:n]
    return np.sqrt(ax * ax + ay * ay + az * az)


def _session_stem(path: Path) -> str:
    return path.stem


# One picklable task tuple: csv_path, device, out_root, flatline_window_sec, flatline_rel_tol, min_keep_sec, overwrite
_CsvTask = tuple[Path, str, Path, float, float, float, bool]


def _configure_worker_logging(log_level: str) -> None:
    """Reset logging in child processes (avoids duplicated handlers after fork)."""
    level = getattr(logging, log_level)
    fmt = "%(levelname)s %(message)s"
    try:
        logging.basicConfig(level=level, format=fmt, force=True)
    except TypeError:
        logging.basicConfig(level=level, format=fmt)


def _process_csv_task(task: _CsvTask) -> dict[str, object] | None:
    """Top-level entry for ``ProcessPoolExecutor`` (must stay picklable)."""
    csv_path, device, out_root, flatline_window_sec, flatline_rel_tol, min_keep_sec, overwrite = task
    return process_one_csv(
        csv_path,
        device,
        out_root,
        flatline_window_sec=flatline_window_sec,
        flatline_rel_tol=flatline_rel_tol,
        min_keep_sec=min_keep_sec,
        overwrite=overwrite,
    )


def process_one_csv(
    csv_path: Path,
    device: str,
    out_root: Path,
    *,
    flatline_window_sec: float,
    flatline_rel_tol: float,
    min_keep_sec: float,
    overwrite: bool,
) -> dict[str, object] | None:
    out_path = out_root / device / csv_path.parent.name / f"{csv_path.stem}.npz"
    if out_path.exists() and not overwrite:
        LOG.info("Skip existing %s", out_path)
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        LOG.error("Failed to read %s: %s", csv_path, exc)
        return None

    n = len(df)
    if n < 10:
        LOG.warning("Too few rows (%d) in %s", n, csv_path)
        return None

    fs_in, duration_sec = infer_fs_duration(n)
    max_sentinel = _sentinel_max_count_for_fs_hz(fs_in)
    if not _sentinel_check_df_or_skip(df, csv_path, max_sentinel):
        return None
    min_keep_samples = int(round(min_keep_sec * 100.0))

    payloads: dict[str, np.ndarray] = {}

    gr_cols = _green_raw_column_names(df)
    if gr_cols:
        gr_avg = _mean_cleaned_columns(df, gr_cols)
        if gr_avg is not None and gr_avg.size:
            out, _ = process_channel(
                gr_avg,
                fs_in,
                flatline_window_sec,
                flatline_rel_tol,
                min_keep_samples,
                csv_path,
                "GR",
            )
            if out is not None:
                payloads["GR"] = out

    ir_cols = _ir_raw_column_names(df)
    if ir_cols:
        ir_avg = _mean_cleaned_columns(df, ir_cols)
        if ir_avg is not None and ir_avg.size:
            out, _ = process_channel(
                ir_avg,
                fs_in,
                flatline_window_sec,
                flatline_rel_tol,
                min_keep_samples,
                csv_path,
                "IR",
            )
            if out is not None:
                payloads["IR"] = out

    gm = _gyro_mag(df)
    if gm is not None and gm.size:
        out, _ = process_channel(
            gm,
            fs_in,
            flatline_window_sec,
            flatline_rel_tol,
            min_keep_samples,
            csv_path,
            "gyro",
        )
        if out is not None:
            payloads["gyro"] = out

    am = _acc_mag(df)
    if am is not None and am.size:
        out, _ = process_channel(
            am,
            fs_in,
            flatline_window_sec,
            flatline_rel_tol,
            min_keep_samples,
            csv_path,
            "acc",
        )
        if out is not None:
            payloads["acc"] = out

    if not payloads:
        LOG.warning("No usable channels in %s", csv_path)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_arr = np.array(float(duration_sec), dtype=np.float32)
    np.savez_compressed(out_path, duration=duration_arr, **payloads)

    ssoid = csv_path.parent.name
    session_id = _session_stem(csv_path)

    row: dict[str, object] = {
        "path": out_path.resolve().as_posix(),
        "dataset": device,
        "session_id": session_id,
        "patient_id": ssoid,
        "duration": int(round(float(duration_sec))),
    }
    del df, payloads
    return row


def _max_inflight_tasks(workers: int, path_queue_size: int) -> int:
    if path_queue_size > 0:
        return max(workers, path_queue_size)
    return max(64, 8 * workers)


def _iter_csv_tasks(args: argparse.Namespace):
    """Yield picklable task tuples without materializing the full file list in memory."""
    n_queued = 0
    for device in ["OPPO_Watch4_Pro", "OPPO_Watch_X3", "OPPO_Watch_X2_mini", "OPPO_Watch_X2"]:
        device_dir = args.raw_root / device
        if not device_dir.is_dir():
            continue
        for ssoid in os.listdir(device_dir):
            ssoid_dir = device_dir / ssoid
            if not ssoid_dir.is_dir():
                continue
            LOG.info("Queueing %s in %s.", ssoid, device)
            for csv_name in os.listdir(ssoid_dir):
                if not csv_name.endswith(".csv"):
                    continue
                src_csv_path = ssoid_dir / csv_name
                yield (
                    src_csv_path,
                    device,
                    args.out_root,
                    args.flatline_window_sec,
                    args.flatline_rel_tol,
                    args.min_keep_sec,
                    args.overwrite,
                )
                n_queued += 1
                if args.debug and n_queued >= args.debug_limit:
                    return
            if args.debug and n_queued >= args.debug_limit:
                return
        if args.debug and n_queued >= args.debug_limit:
            return


def _run_pool_bounded(
    executor: ProcessPoolExecutor,
    task_iter,
    max_inflight: int,
    writer: csv.DictWriter,
) -> None:
    """Keep at most ``max_inflight`` submitted futures so the parent queue stays bounded."""
    pending: dict = {}
    it = iter(task_iter)

    def refill() -> None:
        while len(pending) < max_inflight:
            try:
                t = next(it)
            except StopIteration:
                break
            fut = executor.submit(_process_csv_task, t)
            pending[fut] = t[0]

    refill()
    while pending:
        done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
        for fut in done:
            csv_path = pending.pop(fut)
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001
                LOG.error("Worker failed for %s: %s", csv_path, exc)
                continue
            if row:
                writer.writerow(row)
        refill()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    args.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(args.metadata_csv):
        with args.metadata_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_INDEX_COLUMNS)
            writer.writeheader()

    workers = max(1, int(args.workers))
    max_inflight = _max_inflight_tasks(workers, int(args.path_queue_size))
    task_iter = _iter_csv_tasks(args)

    with args.metadata_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_INDEX_COLUMNS)
        if workers == 1:
            for t in task_iter:
                row = _process_csv_task(t)
                if row:
                    writer.writerow(row)
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_configure_worker_logging,
                initargs=(args.log_level,),
            ) as executor:
                _run_pool_bounded(executor, task_iter, max_inflight, writer)


if __name__ == "__main__":
    main()
