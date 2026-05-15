#!/usr/bin/env python3
"""Preprocess PWV JSON exports into multi-channel NPZ + index CSV.

Run with the project environment, for example:

  conda run -n sleep2vec python wrist2vec/preprocess/pwv_formant_multilight.py

Raw layout (default root):

  {raw_root}/pwv_raw_data_{ssoid}_{time_s}_{create_ms}.json

Outputs:

  {out_root}/pwv_data_v260408/{ssoid}/{ssoid}-{datetime_utc8}.npz
  {index_csv}

Each NPZ stores 1D float32 arrays per physical channel (output key: JSON key):
  ECG      <- ecg
  G0-PD0   <- ppgPgd1
  G0-PD1   <- ppgPgd2
  G0-PD2   <- ppgPgd3
  G0-PD3   <- ppgPgd4
  IR0-PD0  <- ppgIrpd1
  IR0-PD1  <- ppgIrpd2
  IR0-PD2  <- ppgIrpd3
  IR0-PD3  <- ppgIrpd4

Plus scalar ``duration`` (recording length in seconds).

Only files whose inferred duration is exactly 30 s or 60 s are kept.

Signal pipeline per channel: treat raw values 0 and 10000 as sentinels
(interpolate them away); if any channel exceeds round(fs_hz) such samples the
whole file is skipped. Then: resample to --target-fs Hz (default 250) ->
4th-order Chebyshev I bandpass --bp-low to --bp-high Hz (default 0.5-40) ->
remove flatline runs -> z-score.

Multiprocessing (``ProcessPoolExecutor``, ``--workers``) handles conversion;
results are buffered and flushed to the index CSV every ``--batch-size`` files.
Re-runs are safe: files already present in the index CSV are skipped without
re-reading the source JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d

LOG = logging.getLogger("pwv_formant_multilight")

_FS_HZ: float = 250.0
_DATASET_NAME: str = "pwv_data_v260408"

# UTC+8 timezone offset.
_TZ_UTC8 = timezone(timedelta(hours=8))

_SENTINEL_VALUES: tuple[float, float] = (0.0, 10000.0)

_NOMINAL_LAYOUTS: tuple[tuple[float, float], ...] = (
    (30.0, 250.0),
    (60.0, 250.0),
    (30.0, 100.0),
    (60.0, 100.0),
)
_VALID_DURATIONS: frozenset[float] = frozenset({30.0, 60.0})

# JSON field -> NPZ key mapping.
_CHANNEL_MAP: dict[str, str] = {
    "ecg": "ECG",
    "ppgPgd1": "G0-PD0",
    "ppgPgd2": "G0-PD1",
    "ppgPgd3": "G0-PD2",
    "ppgPgd4": "G0-PD3",
    "ppgIrpd1": "IR0-PD0",
    "ppgIrpd2": "IR0-PD1",
    "ppgIrpd3": "IR0-PD2",
    "ppgIrpd4": "IR0-PD3",
}

_INDEX_COLUMNS: list[str] = [
    "path",
    "dataset",
    "session_id",
    "patient_id",
    "duration",
]


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def _parse_json_filename(name: str) -> tuple[str, int] | None:
    """Return (ssoid, time_s) from ``pwv_raw_data_{ssoid}_{time_s}_{ms}.json``.

    Returns None if the name does not match the expected pattern or if ssoid is
    not a purely numeric string (hex / hash ssoids are rejected).
    """
    stem = name.removesuffix(".json")
    parts = stem.split("_")
    # Expected: ["pwv", "raw", "data", ssoid, time_s, create_ms, ...]
    if len(parts) < 6 or parts[:3] != ["pwv", "raw", "data"]:
        return None
    try:
        ssoid = parts[3]
        if not ssoid.isdigit():
            LOG.debug("Skipping %s: non-numeric ssoid %r", name, ssoid)
            return None
        time_s = int(parts[4])
        # Sanity-check: must be a seconds-level Unix timestamp in 2000–2100.
        # Values outside this range are likely millisecond timestamps or corrupt data.
        if not (946_684_800 <= time_s <= 4_102_444_800):
            LOG.debug("Skipping %s: time_s=%d is out of plausible seconds range", name, time_s)
            return None
        return ssoid, time_s
    except (ValueError, IndexError):
        return None


def _session_id_from(ssoid: str, time_s: int) -> str:
    """Return ``{ssoid}-{YYYYMMDDHHmmss}`` in UTC+8."""
    dt = datetime.fromtimestamp(time_s, tz=_TZ_UTC8)
    return f"{ssoid}-{dt.strftime('%Y%m%d%H%M%S')}"


def _out_path(out_root: Path, ssoid: str, session_id: str) -> Path:
    return out_root / _DATASET_NAME / ssoid / f"{session_id}.npz"


# ---------------------------------------------------------------------------
# Duration inference
# ---------------------------------------------------------------------------


def infer_fs_duration(num_samples: int) -> tuple[float, float]:
    """Infer (fs_hz, duration_s) from sample count; returns nominal if within tolerance."""
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
        "Length %d mismatches nominal %.0fs @ %.0fHz (err=%d, tol=%d); using fs=%.1f and duration=%.3fs",
        num_samples, dur_nom, fs_hz, best_err, tol, fs_hz, dur_s,
    )
    return fs_hz, dur_s


# ---------------------------------------------------------------------------
# Signal processing helpers
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
    removed = bool(np.any(flat))
    return x[~flat], removed


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
    a, b = _SENTINEL_VALUES
    return np.isfinite(v) & ((v == a) | (v == b))


def _clean_array(raw: list) -> np.ndarray:
    """Convert a JSON list to float64, replace sentinel/NaN values via interpolation."""
    v = np.asarray(raw, dtype=np.float64)
    if v.size == 0 or np.all(~np.isfinite(v)):
        return np.array([], dtype=np.float64)
    v = v.copy()
    v[_sentinel_outlier_mask(v)] = np.nan
    s = pd.Series(v).interpolate(limit_direction="both").ffill().bfill()
    out = s.to_numpy(dtype=np.float64)
    out[~np.isfinite(out)] = 0.0
    return out


def _sentinel_count(raw: list) -> int:
    v = np.asarray(raw, dtype=np.float64)
    if v.size == 0:
        return 0
    return int(np.sum(_sentinel_outlier_mask(v)))


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
    """Resample -> bandpass -> strip flatlines -> z-score. Returns (array_or_None, stripped)."""
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
            "Stripped flatline segments from %s in %s (post-filter length was %d)",
            channel_name, source_path.as_posix(), flat.size,
        )
    if y.size < min_keep_samples:
        LOG.warning(
            "Dropping %s in %s: length %d < min_keep %d after flatline removal",
            channel_name, source_path.as_posix(), y.size, min_keep_samples,
        )
        return None, stripped
    z = _zscore(y)
    if z is None:
        LOG.warning("Skipping %s in %s: zero variance after flatline removal", channel_name, source_path)
        return None, stripped
    return z.astype(np.float32), stripped


# ---------------------------------------------------------------------------
# Per-JSON processing
# ---------------------------------------------------------------------------


# (json_path, out_root, target_fs, bp_low, bp_high, flatline_window_sec, flatline_rel_tol, min_keep_sec, overwrite)
_JsonTask = tuple[Path, Path, float, float, float, float, float, float, bool]


def _configure_worker_logging(log_level: str) -> None:
    level = getattr(logging, log_level)
    fmt = "%(levelname)s %(message)s"
    try:
        logging.basicConfig(level=level, format=fmt, force=True)
    except TypeError:
        logging.basicConfig(level=level, format=fmt)


def process_one_json(
    json_path: Path,
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
    parsed = _parse_json_filename(json_path.name)
    if parsed is None:
        LOG.warning("Unrecognised filename pattern, skipping: %s", json_path.name)
        return None
    ssoid, time_s = parsed
    session_id = _session_id_from(ssoid, time_s)
    out_path = _out_path(out_root, ssoid, session_id)

    if out_path.exists() and not overwrite:
        LOG.debug("Skip existing NPZ %s", out_path)
        return None

    try:
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        LOG.error("Failed to read %s: %s", json_path, exc)
        return None

    # Determine sample count from the first available channel.
    n_samples: int | None = None
    for json_key in _CHANNEL_MAP:
        raw = data.get(json_key)
        if isinstance(raw, list) and len(raw) > 0:
            n_samples = len(raw)
            break
    if n_samples is None or n_samples < 10:
        LOG.warning("No usable signal data in %s (n_samples=%s)", json_path, n_samples)
        return None

    fs_in, duration_sec = infer_fs_duration(n_samples)

    if duration_sec not in _VALID_DURATIONS:
        LOG.warning(
            "Skipping %s: inferred duration %.3fs is not 30 s or 60 s", json_path.as_posix(), duration_sec
        )
        return None

    max_sentinel = max(1, int(round(fs_in)))
    for json_key in _CHANNEL_MAP:
        raw = data.get(json_key)
        if not isinstance(raw, list) or len(raw) == 0:
            continue
        n_bad = _sentinel_count(raw)
        if n_bad > max_sentinel:
            LOG.warning(
                "Skipping %s: channel %s has %d sentinel outliers (0/10000), max allowed %d",
                json_path.as_posix(), json_key, n_bad, max_sentinel,
            )
            return None

    min_keep_samples = int(round(min_keep_sec * target_fs))
    payloads: dict[str, np.ndarray] = {}

    for json_key, npz_key in _CHANNEL_MAP.items():
        raw = data.get(json_key)
        if not isinstance(raw, list) or len(raw) == 0:
            continue
        arr = _clean_array(raw)
        if arr.size == 0:
            continue
        out, _ = process_channel(
            arr, fs_in, target_fs, bp_low, bp_high,
            flatline_window_sec, flatline_rel_tol, min_keep_samples, json_path, npz_key,
        )
        if out is not None:
            payloads[npz_key] = out

    if not payloads:
        LOG.warning("No usable channels in %s", json_path)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, duration=np.float32(duration_sec), **payloads)

    row: dict[str, object] = {
        "path": out_path.resolve().as_posix(),
        "dataset": _DATASET_NAME,
        "session_id": session_id,
        "patient_id": ssoid,
        "duration": int(round(float(duration_sec))),
    }
    return row


def _process_json_task(task: _JsonTask) -> dict[str, object] | None:
    json_path, out_root, target_fs, bp_low, bp_high, flatline_window_sec, flatline_rel_tol, min_keep_sec, overwrite = task
    return process_one_json(
        json_path,
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


def _iter_json_tasks(args: argparse.Namespace, indexed_paths: frozenset[str]):
    """Yield picklable task tuples, skipping files already present in the index."""
    n_queued = 0
    n_skipped = 0
    for fname in os.listdir(args.raw_root):
        if not fname.endswith(".json"):
            continue
        json_path = args.raw_root / fname
        parsed = _parse_json_filename(fname)
        if parsed is None:
            LOG.debug("Skipping unrecognised filename: %s", fname)
            continue
        ssoid, time_s = parsed
        session_id = _session_id_from(ssoid, time_s)
        expected_npz = _out_path(args.out_root, ssoid, session_id)
        if not args.overwrite and expected_npz.resolve().as_posix() in indexed_paths:
            n_skipped += 1
            continue
        yield (
            json_path,
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
    LOG.info("Task scan done: queued %d new files, skipped %d already indexed.", n_queued, n_skipped)


# ---------------------------------------------------------------------------
# Pool driver with batched CSV flushing
# ---------------------------------------------------------------------------


def _flush_batch(rows: list[dict[str, object]], writer: csv.DictWriter, batch_idx: int) -> None:
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
            fut = executor.submit(_process_json_task, t)
            pending[fut] = t[0]

    refill()
    while pending:
        done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
        for fut in done:
            json_path = pending.pop(fut)
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001
                LOG.error("Worker failed for %s: %s", json_path, exc)
                continue
            if row:
                buffer.append(row)
        if len(buffer) >= batch_size:
            batch_idx += 1
            _flush_batch(buffer, writer, batch_idx)
            total_written += len(buffer)
            buffer.clear()
        refill()

    if buffer:
        batch_idx += 1
        _flush_batch(buffer, writer, batch_idx)
        total_written += len(buffer)

    LOG.info("Done. Total rows written: %d.", total_written)


# ---------------------------------------------------------------------------
# Argument parsing and entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/home/notebook/data/group/pwv_data_v260408/raw_data"),
        help="Directory containing pwv_raw_data_*.json files",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/notebook/data/personal/S9063410/pwv+bp_data_multilight"),
        help="Output root; NPZs are written to {out_root}/pwv_data_v260408/{ssoid}/",
    )
    p.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("/home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pwv_index.csv"),
        help="Path to write aggregated index CSV",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Process at most --debug-limit JSON files in total",
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
        default=250.0,
        help="Target sampling rate in Hz after resampling (default: 250.0)",
    )
    p.add_argument(
        "--bp-low",
        type=float,
        default=0.5,
        help="Bandpass filter lower cutoff in Hz (default: 0.5)",
    )
    p.add_argument(
        "--bp-high",
        type=float,
        default=40.0,
        help="Bandpass filter upper cutoff in Hz (default: 40.0)",
    )
    p.add_argument(
        "--flatline-window-sec",
        type=float,
        default=1.0,
        help="Rolling window length (seconds) for flatline detection (default: 1.0)",
    )
    p.add_argument(
        "--flatline-rel-tol",
        type=float,
        default=1e-5,
        help="Relative peak-to-peak tolerance for flatline detection (default: 1e-5)",
    )
    p.add_argument(
        "--min-keep-sec",
        type=float,
        default=2.0,
        help="Drop a channel if shorter than this many seconds after flatline removal (default: 2.0)",
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    args.metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    indexed_paths = _load_indexed_paths(args.metadata_csv)
    write_header = not (args.metadata_csv.exists() and args.metadata_csv.stat().st_size > 0)

    workers = max(1, int(args.workers))
    task_iter = _iter_json_tasks(args, indexed_paths)

    with args.metadata_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_INDEX_COLUMNS)
        if write_header:
            writer.writeheader()

        if workers == 1:
            buffer: list[dict[str, object]] = []
            batch_idx = 0
            total_written = 0
            for t in task_iter:
                row = _process_json_task(t)
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
