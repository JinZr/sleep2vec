#!/usr/bin/env python3
"""Cut nightly UKB accelerometer segments with the standalone asleep package."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm import tqdm

EPOCH_SECONDS = 30
IS_SLEEP_LABEL = 1
NON_WEAR_LABEL = -1
RESAMPLE_HZ = 30
UK_TIMEZONE = "Europe/London"


def configure_asleep_runtime(model_cache_dir):
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        import asleep.get_sleep as get_sleep_module
        import asleep.models as asleep_models
        import asleep.sslmodel as sslmodel
    except ImportError as exc:
        raise SystemExit(
            "Could not import asleep. Install it in this environment first, for example: " "pip install asleep"
        ) from exc

    original_load_model = getattr(get_sleep_module, "_sleep2vec_original_load_model", get_sleep_module.load_model)
    get_sleep_module._sleep2vec_original_load_model = original_load_model

    def load_model_from_cache(model_path, force_download=False):
        source_path = Path(model_path)
        cached_model_path = model_cache_dir / source_path.name
        if not force_download and not cached_model_path.exists() and source_path.exists():
            tmp_model_path = cached_model_path.with_name(f".{cached_model_path.name}.{os.getpid()}.tmp")
            shutil.copyfile(source_path, tmp_model_path)
            tmp_model_path.replace(cached_model_path)
        return original_load_model(str(cached_model_path), force_download=force_download)

    get_sleep_module.load_model = load_model_from_cache
    sslmodel.torch_cache_path = model_cache_dir / "torch_hub_cache"

    original_dataloader = getattr(asleep_models, "_sleep2vec_original_dataloader", asleep_models.DataLoader)
    asleep_models._sleep2vec_original_dataloader = original_dataloader

    def single_worker_dataloader(*args, **kwargs):
        kwargs["num_workers"] = 0
        return original_dataloader(*args, **kwargs)

    asleep_models.DataLoader = single_worker_dataloader


def load_asleep_pipeline(model_cache_dir):
    configure_asleep_runtime(model_cache_dir)
    try:
        from asleep.get_sleep import get_parsed_data, get_sleep_windows, transform_data2model_input
        import asleep.sleep_windows as sleep_windows
    except ImportError as exc:
        raise SystemExit(
            "Could not import asleep. Install it in this environment first, for example: " "pip install asleep"
        ) from exc
    return get_parsed_data, transform_data2model_input, get_sleep_windows, sleep_windows


def warm_asleep_model_cache(model_cache_dir, force_download):
    configure_asleep_runtime(model_cache_dir)
    import asleep.get_sleep as get_sleep_module
    import asleep.sslmodel as sslmodel

    model_path = Path(get_sleep_module.__file__).parent / "ssl.joblib.lzma"
    sleep_window_detector = get_sleep_module.load_model(str(model_path), force_download=force_download)
    sslmodel.get_sslnet(tag=sleep_window_detector.repo_tag, pretrained=False)


def default_device():
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Use asleep's sleep-window detector to cut nightly sleep accelerometer "
            "segments from UKB .cwa files. The script is standalone and does not "
            "import sleep2vec."
        )
    )
    parser.add_argument("input_root", type=Path, help="A .cwa file or a directory containing UKB .cwa files")
    parser.add_argument("output_dir", type=Path, help="Directory for nightly .npz segments and manifest.csv")
    parser.add_argument("--pattern", default="*.cwa", help="Recursive filename pattern under input_root")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many input files")
    parser.add_argument("--pytorch-device", default=None, help="Device passed to asleep, e.g. cpu or cuda:0")
    parser.add_argument(
        "--time-shift",
        default="auto",
        help="Hour shift passed to asleep, e.g. auto, 0, +1, or -1. auto uses dynamic UK timezone conversion.",
    )
    parser.add_argument("--force-run", action="store_true", help="Regenerate asleep intermediate files")
    parser.add_argument("--force-download", action="store_true", help="Force asleep model download")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing nightly .npz files")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of input files to process in parallel")
    parser.add_argument(
        "--remove-cache",
        action="store_true",
        help="Remove per-file asleep cache after writing outputs",
    )
    parser.add_argument("--manifest-name", default="manifest.csv", help="Manifest filename under output_dir")
    args = parser.parse_args(argv)
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    return args


def find_inputs(input_root, pattern, limit):
    if input_root.is_file():
        files = [input_root]
    else:
        files = sorted(input_root.rglob(pattern))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No input files matched {input_root} with pattern {pattern!r}")
    return files


def relative_output_stem(path, input_root):
    base = input_root.parent if input_root.is_file() else input_root
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = Path(path.name)
    return rel.with_suffix("")


def mark_nightly_blocks(all_blocks, longest_blocks):
    if all_blocks.empty or longest_blocks.empty:
        return all_blocks.iloc[0:0].copy()

    selected_rows = []
    longest_pairs = {(pd.Timestamp(row["start"]), pd.Timestamp(row["end"])) for _, row in longest_blocks.iterrows()}
    for _, row in all_blocks.iterrows():
        pair = (pd.Timestamp(row["start"]), pd.Timestamp(row["end"]))
        if pair in longest_pairs:
            selected_rows.append(row)

    nightly = pd.DataFrame(selected_rows)
    if nightly.empty:
        return all_blocks.iloc[0:0].copy()
    nightly = nightly.reset_index(drop=True)
    nightly.insert(0, "night_id", np.arange(len(nightly), dtype=int))
    return nightly


def timestamp_for_filename(value):
    timestamp = pd.Timestamp(value)
    suffix = timestamp.strftime("%z") if timestamp.tz is not None else ""
    return timestamp.strftime("%Y%m%dT%H%M%S") + suffix


def format_timestamp(value):
    return pd.Timestamp(value).isoformat()


def is_auto_time_shift(requested_shift):
    return requested_shift == "auto"


def format_time_shift_arg(time_shift_hours):
    return f"+{time_shift_hours}" if time_shift_hours > 0 else str(time_shift_hours)


def resolve_time_shift_hours(path, requested_shift):
    if is_auto_time_shift(requested_shift):
        raise ValueError("auto time-shift is dynamic and does not resolve to one offset")
    return int(requested_shift)


def device_to_local_times(times):
    index = pd.DatetimeIndex(pd.to_datetime(times))
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    return index.tz_convert(UK_TIMEZONE)


def local_noon(local_date):
    return pd.Timestamp(datetime(local_date.year, local_date.month, local_date.day, 12), tz=UK_TIMEZONE)


def iter_local_noon_intervals(local_times):
    if len(local_times) == 0:
        return
    interval_start = local_noon(local_times[0].date())
    if local_times[0] < interval_start:
        interval_start = local_noon(local_times[0].date() - timedelta(days=1))
    while interval_start <= local_times[-1]:
        next_start = local_noon(interval_start.date() + timedelta(days=1))
        yield interval_start, next_start
        interval_start = next_start


def utc_offset_hours(timestamp):
    return int(pd.Timestamp(timestamp).utcoffset().total_seconds() // 3600)


def find_sleep_runs(indices, labels):
    sleep_mask = np.asarray(labels) == IS_SLEEP_LABEL
    runs = []
    run_start = None
    for pos, is_sleep in enumerate(sleep_mask):
        if is_sleep and run_start is None:
            run_start = pos
        elif not is_sleep and run_start is not None:
            runs.append((indices[run_start], indices[pos - 1], pos - run_start))
            run_start = None
    if run_start is not None:
        runs.append((indices[run_start], indices[len(indices) - 1], len(indices) - run_start))
    return runs


def build_dynamic_nightly_blocks(binary_y, device_times, local_times, sleep_windows):
    device_times = pd.DatetimeIndex(pd.to_datetime(device_times))
    local_times = pd.DatetimeIndex(local_times)
    local_df = pd.DataFrame(
        {
            "time": local_times,
            "label": np.asarray(binary_y),
            "is_wear": np.asarray(binary_y) != NON_WEAR_LABEL,
        }
    )
    counter = sleep_windows.find_sleep_block_duration(local_df)
    valid_sleep_block_idxes = sleep_windows.find_valid_sleep_blocks(counter, EPOCH_SECONDS)
    gap2fill = sleep_windows.find_gaps2fill(valid_sleep_block_idxes, EPOCH_SECONDS, counter)
    local_df = sleep_windows.fill_gaps(local_df, counter, gap2fill)

    rows = []
    for interval_start, next_interval_start in iter_local_noon_intervals(local_times):
        interval_mask = (local_times >= interval_start) & (local_times < next_interval_start)
        indices = np.flatnonzero(interval_mask)
        if len(indices) == 0:
            continue

        interval_labels = local_df.loc[indices, "label"].to_numpy()
        sleep_runs = find_sleep_runs(indices, interval_labels)
        if not sleep_runs:
            continue

        start_idx, end_idx, epochs = max(sleep_runs, key=lambda item: item[2])
        device_start = pd.Timestamp(device_times[start_idx])
        device_end_exclusive = pd.Timestamp(device_times[end_idx]) + pd.Timedelta(seconds=EPOCH_SECONDS)
        local_start = local_times[start_idx]
        local_end = local_times[end_idx]
        local_end_exclusive = device_to_local_times([device_end_exclusive])[0]
        rows.append(
            {
                "start": local_start,
                "end": local_end,
                "end_exclusive": local_end_exclusive,
                "device_start": device_start,
                "device_end_exclusive": device_end_exclusive,
                "interval_start": interval_start,
                "interval_end": next_interval_start - pd.Timedelta(seconds=1),
                "wear_duration_H": float(local_df.loc[indices, "is_wear"].sum()) / (2 * 60),
                "epochs": int(epochs),
                "start_utc_offset_hours": utc_offset_hours(local_start),
                "end_utc_offset_hours": utc_offset_hours(local_end_exclusive),
            }
        )

    nightly = pd.DataFrame(rows)
    if nightly.empty:
        return nightly
    nightly.insert(0, "night_id", np.arange(len(nightly), dtype=int))
    return nightly


def write_nightly_blocks_csv(nightly_blocks, output_path):
    csv_df = nightly_blocks.copy()
    for column in (
        "start",
        "end",
        "end_exclusive",
        "device_start",
        "device_end_exclusive",
        "interval_start",
        "interval_end",
    ):
        if column in csv_df.columns:
            csv_df[column] = csv_df[column].map(format_timestamp)
    csv_df.to_csv(output_path, index=False)


def read_cwa_signal_segment(path, start, end_exclusive):
    try:
        import actipy
    except ImportError as exc:
        raise SystemExit(
            "Could not import actipy. Install asleep with dependencies, or install actipy separately."
        ) from exc

    data, _ = actipy.read_device(
        str(path),
        lowpass_hz=None,
        calibrate_gravity=False,
        detect_nonwear=False,
        resample_hz=None,
        start_time=start,
        end_time=end_exclusive,
        verbose=False,
    )
    data = data[(data.index >= start) & (data.index < end_exclusive)]
    missing = [name for name in ("x", "y", "z") if name not in data.columns]
    if missing:
        raise ValueError(f"CWA segment is missing expected acceleration columns: {missing}")
    return data[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False), data.index


def write_night_npz(output_path, segment, times, device_times):
    np.savez_compressed(
        output_path,
        actigraphy=segment.astype(np.float32, copy=False),
        time=np.asarray([format_timestamp(t) for t in times]),
        device_time=np.asarray([format_timestamp(t) for t in device_times]),
    )


def process_file(path, input_root, output_dir, args, asleep_pipeline):
    get_parsed_data, transform_data2model_input, get_sleep_windows, sleep_windows = asleep_pipeline

    rel_stem = relative_output_stem(path, input_root)
    auto_time_shift = is_auto_time_shift(args.time_shift)
    time_shift_hours = None if auto_time_shift else resolve_time_shift_hours(path, args.time_shift)
    time_shift_arg = "0" if auto_time_shift else format_time_shift_arg(time_shift_hours)
    file_output_dir = output_dir / rel_stem
    cache_name = "timezone_auto_device" if auto_time_shift else f"time_shift_{time_shift_hours:+d}"
    file_cache_dir = output_dir / "_asleep_cache" / rel_stem / cache_name
    file_output_dir.mkdir(parents=True, exist_ok=True)
    file_cache_dir.mkdir(parents=True, exist_ok=True)

    asleep_args = SimpleNamespace(
        filepath=str(path),
        outdir=str(file_cache_dir),
        force_run=args.force_run,
        force_download=args.force_download,
        pytorch_device=args.pytorch_device or default_device(),
        time_shift=time_shift_arg,
    )

    raw_data_path = file_cache_dir / "raw.csv"
    info_data_path = file_cache_dir / "info.json"
    data2model_path = file_cache_dir / "data2model.npy"
    times_path = file_cache_dir / "times.npy"
    non_wear_path = file_cache_dir / "non_wear.npy"

    data, _ = get_parsed_data(str(raw_data_path), str(info_data_path), RESAMPLE_HZ, asleep_args)
    data2model, times, non_wear = transform_data2model_input(
        str(data2model_path),
        str(times_path),
        str(non_wear_path),
        data,
        asleep_args,
    )
    binary_y, all_blocks, longest_blocks, _, _ = get_sleep_windows(data2model, times, non_wear, asleep_args)
    device_times = pd.DatetimeIndex(pd.to_datetime(times))
    if auto_time_shift:
        local_times = device_to_local_times(device_times)
        nightly_blocks = build_dynamic_nightly_blocks(binary_y, device_times, local_times, sleep_windows)
    else:
        nightly_blocks = mark_nightly_blocks(all_blocks, longest_blocks)
    write_nightly_blocks_csv(nightly_blocks, file_output_dir / "night_sleep_blocks.csv")

    rows = []
    times_index = pd.to_datetime(times)
    time_shift = pd.Timedelta(hours=time_shift_hours or 0)
    for _, night_row in nightly_blocks.iterrows():
        if auto_time_shift:
            start = pd.Timestamp(night_row["start"])
            end_exclusive = pd.Timestamp(night_row["end_exclusive"])
            device_start = pd.Timestamp(night_row["device_start"])
            device_end_exclusive = pd.Timestamp(night_row["device_end_exclusive"])
            epochs = int(night_row["epochs"])
            start_utc_offset_hours = int(night_row["start_utc_offset_hours"])
            end_utc_offset_hours = int(night_row["end_utc_offset_hours"])
        else:
            start = pd.Timestamp(night_row["start"])
            last_epoch = pd.Timestamp(night_row["end"])
            end_exclusive = last_epoch + pd.Timedelta(seconds=EPOCH_SECONDS)
            mask = np.asarray((times_index >= start) & (times_index <= last_epoch))
            if not mask.any():
                continue
            device_start = start - time_shift
            device_end_exclusive = end_exclusive - time_shift
            epochs = int(mask.sum())
            start_utc_offset_hours = int(time_shift_hours)
            end_utc_offset_hours = int(time_shift_hours)

        night_id = int(night_row["night_id"])
        out_name = f"night_{night_id:03d}_{timestamp_for_filename(start)}.npz"
        out_path = file_output_dir / out_name
        if args.overwrite or not out_path.exists():
            raw_segment, raw_device_times = read_cwa_signal_segment(path, device_start, device_end_exclusive)
            if auto_time_shift:
                raw_local_times = device_to_local_times(raw_device_times)
            else:
                raw_local_times = raw_device_times + time_shift
            write_night_npz(out_path, raw_segment, raw_local_times, raw_device_times)
        else:
            with np.load(out_path) as existing:
                raw_segment = existing["actigraphy"]

        rows.append(
            {
                "source_path": str(path),
                "relative_source": str(path.relative_to(input_root) if input_root.is_dir() else path.name),
                "night_id": night_id,
                "start_time": format_timestamp(start),
                "end_time_exclusive": format_timestamp(end_exclusive),
                "device_start_time": format_timestamp(device_start),
                "device_end_time_exclusive": format_timestamp(device_end_exclusive),
                "duration_seconds": float((device_end_exclusive - device_start).total_seconds()),
                "epochs": epochs,
                "raw_samples": int(raw_segment.shape[0]),
                "start_utc_offset_hours": start_utc_offset_hours,
                "end_utc_offset_hours": end_utc_offset_hours,
                "timezone_mode": "auto" if auto_time_shift else f"fixed_shift_{time_shift_hours:+d}",
                "output_path": str(out_path),
                "asleep_cache_dir": str(file_cache_dir),
            }
        )

    if args.remove_cache:
        shutil.rmtree(file_cache_dir)
    return rows


def process_indexed_file(index, total, path, input_root, output_dir, args):
    started_at = time.perf_counter()
    tqdm.write(f"[{index}/{total}] {path}")
    asleep_pipeline = load_asleep_pipeline(output_dir / "_asleep_models")
    rows = process_file(path, input_root, output_dir, args, asleep_pipeline)
    elapsed = time.perf_counter() - started_at
    tqdm.write(f"[{index}/{total}] completed {path} in {elapsed:.1f}s")
    return index, rows


def process_files(input_files, args):
    if args.num_workers == 1:
        asleep_pipeline = load_asleep_pipeline(args.output_dir / "_asleep_models")
        manifest_rows = []
        total = len(input_files)
        for index, path in enumerate(tqdm(input_files, total=total, desc="Processing CWA files", unit="file"), start=1):
            path = path.resolve()
            started_at = time.perf_counter()
            tqdm.write(f"[{index}/{total}] {path}")
            manifest_rows.extend(process_file(path, args.input_root, args.output_dir, args, asleep_pipeline))
            elapsed = time.perf_counter() - started_at
            tqdm.write(f"[{index}/{total}] completed {path} in {elapsed:.1f}s")
        return manifest_rows

    indexed_rows = []
    total = len(input_files)
    max_workers = min(args.num_workers, total)
    if args.pytorch_device is None:
        args.pytorch_device = "cpu"
        print("Using pytorch device: cpu (default for parallel processing)")
    warm_asleep_model_cache(args.output_dir / "_asleep_models", args.force_download)
    args.force_download = False
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_indexed_file,
                index,
                total,
                path.resolve(),
                args.input_root,
                args.output_dir,
                args,
            )
            for index, path in enumerate(input_files, start=1)
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing CWA files",
            unit="file",
        ):
            indexed_rows.append(future.result())

    manifest_rows = []
    for _, rows in sorted(indexed_rows, key=lambda item: item[0]):
        manifest_rows.extend(rows)
    return manifest_rows


def main():
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_files = find_inputs(args.input_root, args.pattern, args.limit)
    manifest_rows = process_files(input_files, args)

    manifest_path = args.output_dir / args.manifest_name
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
