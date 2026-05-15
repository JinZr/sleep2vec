#!/usr/bin/env python3
"""Preprocess blood-pressure wearable CSV exports into multi-channel NPZ + index CSV.

Run with the project environment, for example:

  conda run -n sleep2vec python wrist2vec/preprocess/ppg_formant_multilight.py

Raw layout (default root):

  {raw_root}/{device}/{ssoid}/{ssoid}_{timestamp}.csv

Outputs:

  {out_root}/{device}/{ssoid}/{ssoid}_{timestamp}.npz

Each NPZ stores 1D float32 arrays per physical channel:
  - Green PPG:      G0-PD0, G0-PD1, G1-PD0, G1-PD1  (raw: g0_c0, g0_c1, g1_c0, g1_c1)
  - Infrared PPG:   IR0-PD0, IR0-PD1                  (raw: ir_c0, ir_c1)
  - Gyroscope axes: gyro_x, gyro_y, gyro_z
  - Accelerometer:  acc_x, acc_y, acc_z
  - Scalar ``duration`` (recording length in seconds).

Only files whose inferred duration is exactly 30 s or 60 s are kept; all others are skipped.

Pipeline per channel: treat raw values 0 and 10000 as sentinels (interpolate them away); if any
column exceeds ``round(inferred_fs_hz)`` sentinel samples, the whole CSV is skipped. Then:
resample to 100 Hz -> 4th-order Chebyshev I bandpass 0.4-12 Hz -> remove flatline runs ->
z-score. Each physical channel is stored independently (no cross-channel averaging).

Multiprocessing (``ProcessPoolExecutor``, ``--workers``) handles CSV conversion; results are
buffered in memory and flushed to the index CSV once per ``--batch-size`` files (default 10 000).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d

LOG = logging.getLogger("ppg_formant_multilight")

_SENTINEL_VALUES: tuple[float, float] = (0.0, 10000.0)

_NOMINAL_LAYOUTS: tuple[tuple[float, float], ...] = (
    (30.0, 100.0),
    (30.0, 250.0),
    (60.0, 100.0),
    (60.0, 250.0),
)

_VALID_DURATIONS: frozenset[float] = frozenset({30.0, 60.0})

# Raw column name -> output NPZ key.
_GREEN_COL_MAP: dict[str, str] = {
    "g0_c0": "G0-PD0",
    "g0_c1": "G0-PD1",
    "g1_c0": "G1-PD0",
    "g1_c1": "G1-PD1",
}
_IR_COL_MAP: dict[str, str] = {
    "ir_c0": "IR0-PD0",
    "ir_c1": "IR0-PD1",
}
_IMU_COLS: tuple[str, ...] = ("gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z")

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
        default=Path("/home/notebook/data/personal/S9063410/bp_data_multilight"),
        help="Output root for NPZ mirrors",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("/home/notebook/data/personal/S9063410/bp_data_multilight/index.csv"),
        help="Path to write aggregated metadata CSV",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Process at most --debug-limit CSV files in total (for debugging)",
    )
    p.add_argument(
        "--debug-limit",
        type=int,
        default=50,
        help="Max files when --debug is set (default: 50)",
    )
    p.add_argument(
        "--target-fs",
        type=float,
        default=100.0,
        help="Target sampling rate in Hz after resampling (default: 100.0)",
    )
    p.add_argument(
        "--bp-low",
        type=float,
        default=0.4,
        help="Bandpass filter lower cutoff in Hz (default: 0.4)",
    )
    p.add_argument(
        "--bp-high",
        type=float,
        default=12.0,
        help="Bandpass filter upper cutoff in Hz (default: 12.0)",
    )
    p.add_argument(
        "--flatline-window-sec",
        type=float,
        default=1.0,
        help="Rolling window length (seconds) for flatline detection after filtering",
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
        help="Drop a channel if shorter than this many seconds at 100 Hz after flatline removal",
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
        help="Number of worker processes (default: 16; use 1 to disable parallelism)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Flush index CSV after this many completed files (default: 10 000)",
    )
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Sampling-rate / duration inference
# ---------------------------------------------------------------------------


def infer_fs_duration(num_samples: int) -> tuple[float, float]:
    """Infer sampling rate (Hz) and recording duration (s) from sample count.

    Returns the nominal (fs_hz, duration_s) if within tolerance, otherwise the
    best-guess fs with the actual measured duration.
    """
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


# ---------------------------------------------------------------------------
# Signal processing helpers (unchanged from single-channel version)
# ---------------------------------------------------------------------------


def _resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resample from fs_in to fs_out Hz."""
    x = np.asarray(x, dtype=np.float64)
    if math.isclose(fs_in, fs_out, rel_tol=1e-4):
        return x
    from fractions import Fraction

    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(x, up=frac.numerator, down=frac.denominator)


def _design_bandpass_sos(fs: float, lowcut: float, highcut: float) -> np.ndarray:
    """4th-order Chebyshev Type I bandpass, 1 dB passband ripple."""
    return signal.cheby1(4, 1, [lowcut, highcut], btype="band", fs=fs, output="sos")


def _bandpass(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    if x.size < 32:
        return x
    return signal.sosfiltfilt(sos, x)


def _flatline_mask(x: np.ndarray, window_sec: float, rel_tol: float, fs: float) -> np.ndarray:
    """True where the center sample belongs to a locally flat stretch (post-preprocessing)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    win = max(3, int(round(window_sec * fs)))
    if win % 2 == 0:
        win += 1
    win = min(win, n)
    mx = maximum_filter1d(x, size=win, mode="nearest")
    mn = minimum_filter1d(x, size=win, mode="nearest")
    spread = mx - mn
    scale = float(np.nanmedian(np.abs(x))) + 1e-12
    return spread < (rel_tol * scale)


def _strip_flat_segments(x: np.ndarray, flat: np.ndarray) -> tuple[np.ndarray, bool]:
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
    v = np.asarray(v, dtype=np.float64)
    a, b = _SENTINEL_VALUES
    return np.isfinite(v) & ((v == a) | (v == b))


def _sentinel_outlier_count(values: pd.Series) -> int:
    v = pd.to_numeric(values, errors="coerce").astype(np.float64).to_numpy()
    if v.size == 0:
        return 0
    return int(np.sum(_sentinel_outlier_mask(v)))


def _sentinel_max_count_for_fs_hz(fs_hz: float) -> int:
    return max(1, int(round(float(fs_hz))))


def _clean_series(values: pd.Series) -> np.ndarray:
    """Coerce to float64, interpolate sentinel/NaN values, forward/backward fill edges."""
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


# ---------------------------------------------------------------------------
# Per-channel sentinel check
# ---------------------------------------------------------------------------


def _sentinel_check_df_or_skip(df: pd.DataFrame, csv_path: Path, max_sentinel_per_channel: int) -> bool:
    """Return False if any present column's sentinel count exceeds the limit."""
    check_cols: list[str] = []
    for raw_col in list(_GREEN_COL_MAP) + list(_IR_COL_MAP):
        if raw_col in df.columns:
            check_cols.append(raw_col)
    for c in _IMU_COLS:
        if c in df.columns:
            check_cols.append(c)
    for col in check_cols:
        n_bad = _sentinel_outlier_count(df[col])
        if n_bad > max_sentinel_per_channel:
            LOG.warning(
                "Skipping %s: column %s has %d sentinel outliers (%s/%s), max allowed %d",
                csv_path.as_posix(),
                col,
                n_bad,
                _SENTINEL_VALUES[0],
                _SENTINEL_VALUES[1],
                max_sentinel_per_channel,
            )
            return False
    return True


# ---------------------------------------------------------------------------
# Single-channel processing pipeline
# ---------------------------------------------------------------------------


def process_channel(
    x: np.ndarray,
    fs_in: float,
    target_fs: float,
    bp_low: float,
    bp_high: float,
    flatline_window_sec: float,
    flatline_rel_tol: float,
    min_keep_samples: int,
    source_path: Path,
    channel_name: str,
) -> tuple[np.ndarray | None, bool]:
    """Resample -> bandpass -> strip flatlines -> z-score.  Returns (array_or_None, stripped)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return None, False

    y = _resample_to(x, fs_in, target_fs)
    sos = _design_bandpass_sos(target_fs, bp_low, bp_high)
    y = _bandpass(y, sos)

    flat = _flatline_mask(y, window_sec=flatline_window_sec, rel_tol=flatline_rel_tol, fs=target_fs)
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
        LOG.warning("Skipping channel %s in %s: zero variance after flatline removal", channel_name, source_path)
        return None, stripped
    return z.astype(np.float32), stripped


# ---------------------------------------------------------------------------
# Per-CSV processing
# ---------------------------------------------------------------------------


def _session_stem(path: Path) -> str:
    return path.stem


# (csv_path, device, out_root, target_fs, bp_low, bp_high, flatline_window_sec, flatline_rel_tol, min_keep_sec, overwrite)
_CsvTask = tuple[Path, str, Path, float, float, float, float, float, float, bool]


def _configure_worker_logging(log_level: str) -> None:
    level = getattr(logging, log_level)
    fmt = "%(levelname)s %(message)s"
    try:
        logging.basicConfig(level=level, format=fmt, force=True)
    except TypeError:
        logging.basicConfig(level=level, format=fmt)


def process_one_csv(
    csv_path: Path,
    device: str,
    out_root: Path,
    *,
    target_fs: float,
    bp_low: float,
    bp_high: float,
    flatline_window_sec: float,
    flatline_rel_tol: float,
    min_keep_sec: float,
    overwrite: bool,
) -> dict[str, object] | None:
    out_path = out_root / device / csv_path.parent.name / f"{csv_path.stem}.npz"
    if out_path.exists() and not overwrite:
        LOG.debug("Skip existing NPZ %s", out_path)
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

    # Require exactly 30 s or 60 s; skip anything else.
    if duration_sec not in _VALID_DURATIONS:
        LOG.warning(
            "Skipping %s: inferred duration %.3f s is not 30 s or 60 s",
            csv_path.as_posix(),
            duration_sec,
        )
        return None

    max_sentinel = _sentinel_max_count_for_fs_hz(fs_in)
    if not _sentinel_check_df_or_skip(df, csv_path, max_sentinel):
        return None

    min_keep_samples = int(round(min_keep_sec * target_fs))
    payloads: dict[str, np.ndarray] = {}

    # Green PPG channels (independent physical photodetectors).
    for raw_col, npz_key in _GREEN_COL_MAP.items():
        if raw_col not in df.columns:
            continue
        raw = _clean_series(df[raw_col])
        if raw.size == 0:
            continue
        out, _ = process_channel(raw, fs_in, target_fs, bp_low, bp_high, flatline_window_sec, flatline_rel_tol, min_keep_samples, csv_path, npz_key)
        if out is not None:
            payloads[npz_key] = out

    # Infrared PPG channels.
    for raw_col, npz_key in _IR_COL_MAP.items():
        if raw_col not in df.columns:
            continue
        raw = _clean_series(df[raw_col])
        if raw.size == 0:
            continue
        out, _ = process_channel(raw, fs_in, target_fs, bp_low, bp_high, flatline_window_sec, flatline_rel_tol, min_keep_samples, csv_path, npz_key)
        if out is not None:
            payloads[npz_key] = out

    # IMU axes (gyroscope and accelerometer, each axis independent).
    for col in _IMU_COLS:
        if col not in df.columns:
            continue
        raw = _clean_series(df[col])
        if raw.size == 0:
            continue
        out, _ = process_channel(raw, fs_in, target_fs, bp_low, bp_high, flatline_window_sec, flatline_rel_tol, min_keep_samples, csv_path, col)
        if out is not None:
            payloads[col] = out

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


def _process_csv_task(task: _CsvTask) -> dict[str, object] | None:
    csv_path, device, out_root, target_fs, bp_low, bp_high, flatline_window_sec, flatline_rel_tol, min_keep_sec, overwrite = task
    return process_one_csv(
        csv_path,
        device,
        out_root,
        target_fs=target_fs,
        bp_low=bp_low,
        bp_high=bp_high,
        flatline_window_sec=flatline_window_sec,
        flatline_rel_tol=flatline_rel_tol,
        min_keep_sec=min_keep_sec,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# Index helpers and task generator
# ---------------------------------------------------------------------------


def _load_indexed_paths(metadata_csv: Path) -> frozenset[str]:
    """Return the set of NPZ paths already recorded in the index CSV."""
    if not metadata_csv.exists() or metadata_csv.stat().st_size == 0:
        return frozenset()
    try:
        df = pd.read_csv(metadata_csv, usecols=["path"])
        paths = frozenset(df["path"].dropna().astype(str).tolist())
        LOG.info("Loaded %d already-indexed paths from %s.", len(paths), metadata_csv)
        return paths
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not read existing index CSV %s (%s); treating as empty.", metadata_csv, exc)
        return frozenset()


def _iter_csv_tasks(args: argparse.Namespace, indexed_paths: frozenset[str]):
    """Yield picklable task tuples, skipping files already present in the index."""
    n_queued = 0
    n_skipped = 0
    for device in ["OPPO_Watch4_Pro", "OPPO_Watch_X3", "OPPO_Watch_X2_mini", "OPPO_Watch_X2"]:
        device_dir = args.raw_root / device
        if not device_dir.is_dir():
            continue
        for ssoid in os.listdir(device_dir):
            ssoid_dir = device_dir / ssoid
            if not ssoid_dir.is_dir():
                continue
            for csv_name in os.listdir(ssoid_dir):
                if not csv_name.endswith(".csv"):
                    continue
                csv_path = ssoid_dir / csv_name
                expected_npz = (args.out_root / device / ssoid / csv_path.stem).with_suffix(".npz")
                if not args.overwrite and expected_npz.resolve().as_posix() in indexed_paths:
                    n_skipped += 1
                    continue
                yield (
                    csv_path,
                    device,
                    args.out_root,
                    args.target_fs,
                    args.bp_low,
                    args.bp_high,
                    args.flatline_window_sec,
                    args.flatline_rel_tol,
                    args.min_keep_sec,
                    args.overwrite,
                )
                n_queued += 1
                if args.debug and n_queued >= args.debug_limit:
                    LOG.info("Debug limit reached: queued %d, skipped %d.", n_queued, n_skipped)
                    return
            if args.debug and n_queued >= args.debug_limit:
                return
        if args.debug and n_queued >= args.debug_limit:
            return
    LOG.info("Task scan done: queued %d new files, skipped %d already indexed.", n_queued, n_skipped)


# ---------------------------------------------------------------------------
# Pool driver with batched CSV flushing
# ---------------------------------------------------------------------------


def _flush_batch(rows: list[dict[str, object]], writer: csv.DictWriter, batch_idx: int) -> None:
    """Write a batch of rows and log progress."""
    for row in rows:
        writer.writerow(row)
    LOG.info("Flushed batch %d: %d rows written to index CSV.", batch_idx, len(rows))


def _run_pool_batched(
    executor: ProcessPoolExecutor,
    task_iter,
    workers: int,
    writer: csv.DictWriter,
    batch_size: int,
) -> None:
    """Submit tasks with a bounded in-flight window; flush completed rows to CSV every batch_size results."""
    max_inflight = max(64, 8 * workers)
    pending: dict = {}
    it = iter(task_iter)
    buffer: list[dict[str, object]] = []
    batch_idx = 0
    total_written = 0

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
                buffer.append(row)

        # Flush whenever the buffer reaches the batch size.
        if len(buffer) >= batch_size:
            batch_idx += 1
            _flush_batch(buffer, writer, batch_idx)
            total_written += len(buffer)
            buffer.clear()

        refill()

    # Flush any remaining rows after all tasks are done.
    if buffer:
        batch_idx += 1
        _flush_batch(buffer, writer, batch_idx)
        total_written += len(buffer)

    LOG.info("Done. Total rows written: %d.", total_written)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    print(args.__dict__)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    args.metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    # Read existing index once at startup; filter tasks before submitting to the pool.
    indexed_paths = _load_indexed_paths(args.metadata_csv)

    # Always append; write header only when the file is new or empty.
    write_header = not (args.metadata_csv.exists() and args.metadata_csv.stat().st_size > 0)

    workers = max(1, int(args.workers))
    task_iter = _iter_csv_tasks(args, indexed_paths)

    with args.metadata_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_INDEX_COLUMNS)
        if write_header:
            writer.writeheader()

        if workers == 1:
            buffer: list[dict[str, object]] = []
            batch_idx = 0
            total_written = 0
            for t in task_iter:
                row = _process_csv_task(t)
                if row:
                    buffer.append(row)
                if len(buffer) >= args.batch_size:
                    batch_idx += 1
                    _flush_batch(buffer, writer, batch_idx)
                    total_written += len(buffer)
                    buffer.clear()
            if buffer:
                batch_idx += 1
                _flush_batch(buffer, writer, batch_idx)
                total_written += len(buffer)
            LOG.info("Done. Total rows written: %d.", total_written)
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_configure_worker_logging,
                initargs=(args.log_level,),
            ) as executor:
                _run_pool_batched(executor, task_iter, workers, writer, args.batch_size)


if __name__ == "__main__":
    main()
