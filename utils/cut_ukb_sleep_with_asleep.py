#!/usr/bin/env python3
"""Cut nightly UKB accelerometer segments with the standalone asleep package."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd

EPOCH_SECONDS = 30
RESAMPLE_HZ = 30


def load_asleep_pipeline():
    try:
        from asleep.get_sleep import get_parsed_data, get_sleep_windows, transform_data2model_input
    except ImportError as exc:
        raise SystemExit(
            "Could not import asleep. Install it in this environment first, for example: " "pip install asleep"
        ) from exc
    return get_parsed_data, transform_data2model_input, get_sleep_windows


def default_device():
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def parse_args():
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
    parser.add_argument("--time-shift", default="0", help="Hour shift passed to asleep, e.g. +1 or -1")
    parser.add_argument("--force-run", action="store_true", help="Regenerate asleep intermediate files")
    parser.add_argument("--force-download", action="store_true", help="Force asleep model download")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing nightly .npz files")
    parser.add_argument(
        "--remove-cache",
        action="store_true",
        help="Remove per-file asleep cache after writing outputs",
    )
    parser.add_argument("--manifest-name", default="manifest.csv", help="Manifest filename under output_dir")
    return parser.parse_args()


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
    return pd.Timestamp(value).strftime("%Y%m%dT%H%M%S")


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


def write_night_npz(output_path, segment, times):
    np.savez_compressed(
        output_path,
        actigraphy=segment.astype(np.float32, copy=False),
        time=np.asarray([str(t) for t in times]),
    )


def process_file(path, input_root, output_dir, args, asleep_pipeline):
    get_parsed_data, transform_data2model_input, get_sleep_windows = asleep_pipeline

    rel_stem = relative_output_stem(path, input_root)
    file_output_dir = output_dir / rel_stem
    file_cache_dir = output_dir / "_asleep_cache" / rel_stem
    file_output_dir.mkdir(parents=True, exist_ok=True)
    file_cache_dir.mkdir(parents=True, exist_ok=True)

    asleep_args = SimpleNamespace(
        filepath=str(path),
        outdir=str(file_cache_dir),
        force_run=args.force_run,
        force_download=args.force_download,
        pytorch_device=args.pytorch_device or default_device(),
        time_shift=args.time_shift,
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
    _, all_blocks, longest_blocks, _, _ = get_sleep_windows(data2model, times, non_wear, asleep_args)
    nightly_blocks = mark_nightly_blocks(all_blocks, longest_blocks)
    nightly_blocks.to_csv(file_output_dir / "night_sleep_blocks.csv", index=False)

    rows = []
    times_index = pd.to_datetime(times)
    time_shift = pd.Timedelta(hours=int(args.time_shift))
    for _, night_row in nightly_blocks.iterrows():
        start = pd.Timestamp(night_row["start"])
        last_epoch = pd.Timestamp(night_row["end"])
        end_exclusive = last_epoch + pd.Timedelta(seconds=EPOCH_SECONDS)
        mask = np.asarray((times_index >= start) & (times_index <= last_epoch))
        if not mask.any():
            continue

        night_id = int(night_row["night_id"])
        out_name = f"night_{night_id:03d}_{timestamp_for_filename(start)}.npz"
        out_path = file_output_dir / out_name
        if args.overwrite or not out_path.exists():
            raw_segment, raw_times = read_cwa_signal_segment(path, start - time_shift, end_exclusive - time_shift)
            write_night_npz(out_path, raw_segment, raw_times + time_shift)
        else:
            with np.load(out_path) as existing:
                raw_segment = existing["actigraphy"]

        rows.append(
            {
                "source_path": str(path),
                "relative_source": str(path.relative_to(input_root) if input_root.is_dir() else path.name),
                "night_id": night_id,
                "start_time": str(start),
                "end_time_exclusive": str(end_exclusive),
                "duration_seconds": float((end_exclusive - start).total_seconds()),
                "epochs": int(mask.sum()),
                "raw_samples": int(raw_segment.shape[0]),
                "output_path": str(out_path),
                "asleep_cache_dir": str(file_cache_dir),
            }
        )

    if args.remove_cache:
        shutil.rmtree(file_cache_dir)
    return rows


def main():
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    asleep_pipeline = load_asleep_pipeline()
    input_files = find_inputs(args.input_root, args.pattern, args.limit)

    manifest_rows = []
    for idx, path in enumerate(input_files, start=1):
        print(f"[{idx}/{len(input_files)}] {path}")
        manifest_rows.extend(process_file(path.resolve(), args.input_root, args.output_dir, args, asleep_pipeline))

    manifest_path = args.output_dir / args.manifest_name
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
