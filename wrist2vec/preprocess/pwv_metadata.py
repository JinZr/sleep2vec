#!/usr/bin/env python3
"""Augment a multilight PWV index CSV with participant metadata and per-channel mask columns.

Reads:
  --index-csv          Base index CSV produced by pwv_formant_multilight.py
                       (columns: path, dataset, session_id, patient_id, duration)
  --demographics-csv   pwv_label/demographics.csv
                       (columns: ssoid, weight, height, sex, birthday_value, age)

Writes:
  --out-csv            Augmented CSV with age, sex, weight, height, bmi,
                       plus one {channel}_mask column per NPZ key (1=present, 0=absent).

Age is derived from collection time minus birthday_value, not the roster age column.
Collection time is parsed from session_id.split("-")[1] as YYYYMMDDhhmmss.

Rows whose patient_id (ssoid) is absent from demographics are silently dropped.
Rows already present in --out-csv (by path) are skipped on re-runs.

Processing is parallelised with ProcessPoolExecutor; results are flushed to disk
every --batch-size rows.

NPZ channel keys recognised (and their mask columns):
  ECG                                  -> ECG_mask
  G0-PD0 … G0-PD3, IR0-PD0 … IR0-PD3  -> G0-PD0_mask …
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
from datetime import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("pwv_metadata")

_NPZ_CHANNEL_KEYS: tuple[str, ...] = (
    "ECG",
    "G0-PD0",
    "G0-PD1",
    "G0-PD2",
    "G0-PD3",
    "IR0-PD0",
    "IR0-PD1",
    "IR0-PD2",
    "IR0-PD3",
)
_MASK_COLUMNS: tuple[str, ...] = tuple(f"{k}_mask" for k in _NPZ_CHANNEL_KEYS)

_BASE_COLUMNS: tuple[str, ...] = ("path", "dataset", "session_id", "patient_id", "duration")
_META_COLUMNS: tuple[str, ...] = ("age", "sex", "weight", "height", "bmi")
_OUTPUT_COLUMNS: tuple[str, ...] = _BASE_COLUMNS + _META_COLUMNS + _MASK_COLUMNS

_SECONDS_PER_YEAR: float = 365.2425 * 24 * 3600


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--index-csv",
        type=Path,
        default=Path("/home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pwv_index.csv"),
        help="Base index CSV from pwv_formant_multilight.py",
    )
    p.add_argument(
        "--demographics-csv",
        type=Path,
        default=Path("/home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pwv_label/demographics.csv"),
        help="Participant demographics with ssoid/sex/weight/height/birthday_value columns",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/home/notebook/data/personal/S9063410/pwv+bp_data_multilight/pwv_index_mask.csv"),
        help="Output path for augmented index CSV",
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
        help="Flush output CSV after this many completed rows (default: 10 000)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process rows already present in --out-csv",
    )
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args()


def _is_null(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    if isinstance(value, str) and value.strip().upper() in {"", "NULL", "NAN", "NONE"}:
        return True
    return False


def _load_demographics(demographics_csv: Path) -> dict[str, dict]:
    """Return {ssoid_str -> {weight, height, bmi, sex, birthday}} with cleaned units."""
    df = pd.read_csv(demographics_csv, dtype={"ssoid": str})
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        ssoid = str(row["ssoid"]).strip()

        weight_raw = row.get("weight")
        if _is_null(weight_raw):
            weight = None
        else:
            weight_raw = float(weight_raw)
            if weight_raw < 40_000 or weight_raw > 200_000:
                weight = None
            else:
                weight = weight_raw / 1000.0

        height_raw = row.get("height")
        if _is_null(height_raw):
            height = None
        else:
            height_raw = float(height_raw)
            if height_raw < 1400 or height_raw > 2100:
                height = None
            else:
                height = height_raw / 10.0

        bmi = (weight / (height / 100.0) ** 2) if (weight is not None and height is not None) else None

        sex_raw = row.get("sex")
        if _is_null(sex_raw):
            sex = None
        elif str(sex_raw).strip().upper() == "M":
            sex = "male"
        elif str(sex_raw).strip().upper() == "F":
            sex = "female"
        else:
            sex = None

        bday_raw = row.get("birthday_value")
        if _is_null(bday_raw):
            birthday = None
        else:
            try:
                dt = datetime.strptime(str(bday_raw).strip(), "%Y-%m-%d")
                birthday = dt if 1940 <= dt.year <= 2010 else None
            except ValueError:
                birthday = None

        if weight == 60.0 and height == 170.0 and birthday is None:
            weight = height = bmi = sex = None

        out[ssoid] = {
            "weight": weight,
            "height": height,
            "bmi": bmi,
            "sex": sex,
            "birthday": birthday,
        }
    return out


def _compute_age(session_id: str, birthday: datetime | None) -> float | None:
    """Derive collection datetime from session_id and return age in years."""
    if birthday is None:
        return None
    try:
        collect_str = session_id.split("-")[1]
        collect_dt = datetime.strptime(collect_str, "%Y%m%d%H%M%S")
        bday_dt = birthday.replace(hour=0, minute=0, second=0, microsecond=0)
        return round((collect_dt - bday_dt).total_seconds() / _SECONDS_PER_YEAR, 1)
    except (IndexError, ValueError):
        return None


_RowTask = tuple[dict, dict]


def _process_row_task(task: _RowTask) -> dict | None:
    row, meta = task
    npz_path = Path(row["path"])

    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            present_keys = set(npz.files)
    except Exception as exc:  # noqa: BLE001
        LOG.error("Failed to load %s: %s", npz_path, exc)
        return None

    age = _compute_age(str(row["session_id"]), meta["birthday"])

    out: dict = {
        "path": row["path"],
        "dataset": row["dataset"],
        "session_id": row["session_id"],
        "patient_id": row["patient_id"],
        "duration": row["duration"],
        "age": age,
        "sex": meta["sex"],
        "weight": meta["weight"],
        "height": meta["height"],
        "bmi": meta["bmi"],
    }
    for key in _NPZ_CHANNEL_KEYS:
        out[f"{key}_mask"] = 1 if key in present_keys else 0

    return out


def _configure_worker_logging(log_level: str) -> None:
    level = getattr(logging, log_level)
    fmt = "%(levelname)s %(message)s"
    try:
        logging.basicConfig(level=level, format=fmt, force=True)
    except TypeError:
        logging.basicConfig(level=level, format=fmt)


def _load_indexed_paths(out_csv: Path) -> frozenset[str]:
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return frozenset()
    try:
        df = pd.read_csv(out_csv, usecols=["path"])
        paths = frozenset(df["path"].dropna().astype(str).tolist())
        LOG.info("Loaded %d already-indexed paths from %s.", len(paths), out_csv)
        return paths
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not read existing output CSV %s (%s); treating as empty.", out_csv, exc)
        return frozenset()


def _flush_batch(rows: list[dict], writer: csv.DictWriter, batch_idx: int) -> None:
    for row in rows:
        clean = {k: ("" if row.get(k) is None else row[k]) for k in _OUTPUT_COLUMNS}
        writer.writerow(clean)
    LOG.info("Flushed batch %d: %d rows.", batch_idx, len(rows))


def _run_pool_batched(
    executor: ProcessPoolExecutor,
    task_iter,
    workers: int,
    writer: csv.DictWriter,
    batch_size: int,
) -> int:
    max_inflight = max(64, 8 * workers)
    pending: dict = {}
    it = iter(task_iter)
    buffer: list[dict] = []
    batch_idx = 0
    total_written = 0

    def refill() -> None:
        while len(pending) < max_inflight:
            try:
                t = next(it)
            except StopIteration:
                break
            fut = executor.submit(_process_row_task, t)
            pending[fut] = t[0]["path"]

    refill()
    while pending:
        done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
        for fut in done:
            path = pending.pop(fut)
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                LOG.error("Worker failed for %s: %s", path, exc)
                continue
            if result is not None:
                buffer.append(result)

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

    return total_written


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    LOG.info("Loading demographics from %s", args.demographics_csv)
    roster = _load_demographics(args.demographics_csv)
    LOG.info("Demographics loaded: %d entries.", len(roster))

    LOG.info("Loading base index from %s", args.index_csv)
    index_df = pd.read_csv(args.index_csv, dtype=str)
    LOG.info("Base index: %d rows.", len(index_df))

    indexed_paths = frozenset() if args.overwrite else _load_indexed_paths(args.out_csv)

    tasks: list[_RowTask] = []
    n_no_roster = 0
    n_already_indexed = 0
    for _, row in index_df.iterrows():
        ssoid = str(row["patient_id"]).strip()
        if ssoid not in roster:
            LOG.debug("Skipping %s: ssoid %s not in demographics.", row["path"], ssoid)
            n_no_roster += 1
            continue
        if row["path"] in indexed_paths:
            n_already_indexed += 1
            continue
        tasks.append((row.to_dict(), roster[ssoid]))

    LOG.info(
        "Tasks to process: %d  |  skipped (no demographics): %d  |  skipped (already indexed): %d",
        len(tasks),
        n_no_roster,
        n_already_indexed,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (args.out_csv.exists() and args.out_csv.stat().st_size > 0)
    workers = max(1, int(args.workers))

    with args.out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(_OUTPUT_COLUMNS))
        if write_header:
            writer.writeheader()

        if workers == 1:
            buffer: list[dict] = []
            batch_idx = 0
            total_written = 0
            for task in tasks:
                result = _process_row_task(task)
                if result is not None:
                    buffer.append(result)
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
                total_written = _run_pool_batched(executor, iter(tasks), workers, writer, args.batch_size)
            LOG.info("Done. Total rows written: %d.", total_written)


if __name__ == "__main__":
    main()
