#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import sys
import typing as t

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sleep2wave.data.generative_dataset import (
    IndexColumnConfig,
    prepare_sleep2wave_index_frame,
    resolve_modality_mask_columns,
    resolve_npz_key,
)
from sleep2wave.data.modalities import CANONICAL_MODALITIES, EPOCH_SEC, MODALITY_SPECS
from sleep2wave.data.utils import load_npz
from sleep2wave.generative.config import load_sleep2wave_config
from sleep2wave.preprocess.split_index_by_dataset import normalize_mask_frame


def parse_args(argv: t.Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CSV-indexed NPZ sleep2wave windows to channel-separated Kaldi ark/scp files.",
    )
    parser.add_argument("--index", nargs="+", required=True, help="Input index CSV file(s).")
    parser.add_argument("--config", type=Path, required=True, help="Sleep2Wave YAML config.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output Sleep2Wave Kaldi data root.")
    parser.add_argument("--split", nargs="*", default=None, help="Optional split values to keep.")
    parser.add_argument("--stride-epochs", type=int, default=None, help="Stride in 30-second epochs.")
    parser.add_argument("--path-col", default="path", help="Index column containing waveform NPZ paths.")
    parser.add_argument("--duration-col", default="duration", help="Index column containing recording duration.")
    parser.add_argument("--split-col", default="split", help="Index column containing split labels.")
    parser.add_argument("--subject-id-col", default="subject_id", help="Index column containing subject ids.")
    parser.add_argument("--night-id-col", default="night_id", help="Index column containing night ids.")
    parser.add_argument("--source-col", default="source", help="Index column containing source dataset names.")
    parser.add_argument(
        "--path-prefix-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite NPZ paths by replacing OLD prefix with NEW. May be passed multiple times.",
    )
    return parser.parse_args(argv)


def _import_kaldi_native_io():
    try:
        import kaldi_native_io
    except ImportError as exc:
        raise RuntimeError(
            "kaldi_native_io is required to write Sleep2Wave Kaldi ark/scp files. "
            "Install requirements.txt before running this converter."
        ) from exc
    return kaldi_native_io


def _sanitize_key_part(value: t.Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "unknown"


def _parse_prefix_maps(raw_maps: t.Sequence[str]) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    for item in raw_maps:
        if "=" not in item:
            raise ValueError(f"--path-prefix-map must use OLD=NEW format, got {item!r}.")
        old, new = item.split("=", 1)
        if not old:
            raise ValueError("--path-prefix-map OLD prefix must not be empty.")
        mappings.append((str(Path(old).expanduser()), str(Path(new).expanduser())))
    return sorted(mappings, key=lambda pair: len(pair[0]), reverse=True)


def _resolve_npz_path(path_value: t.Any, prefix_maps: t.Sequence[tuple[str, str]]) -> Path:
    raw = str(path_value)
    for old, new in prefix_maps:
        old_prefix = old.rstrip(os.sep)
        if raw == old_prefix:
            return Path(new).expanduser()
        if raw.startswith(old_prefix + os.sep):
            return Path(new + raw[len(old_prefix) :]).expanduser()
    return Path(raw).expanduser()


def _load_index_df(index_paths: t.Sequence[Path], columns: IndexColumnConfig) -> pd.DataFrame:
    frames = []
    for path in index_paths:
        frame = pd.read_csv(path, low_memory=False)
        if columns.source_col not in frame.columns:
            frame[columns.source_col] = str(path)
        else:
            frame[columns.source_col] = frame[columns.source_col].where(frame[columns.source_col].notna(), str(path))
        frames.append(frame)
    if not frames:
        raise ValueError("At least one --index CSV is required.")
    return pd.concat(frames, ignore_index=True)


def _record_key_from_row(row: pd.Series, columns: IndexColumnConfig) -> str:
    for name in ("session_id", columns.night_id_col, columns.subject_id_col):
        if name in row.index and pd.notna(row[name]) and str(row[name]):
            return _sanitize_key_part(row[name])
    return _sanitize_key_part(Path(str(row[columns.path_col])).stem)


def _sample_key(*, source_value: t.Any, record_key: str, start: int, end: int) -> str:
    return f"{_sanitize_key_part(source_value)}_{record_key}_{start:06d}_{end:06d}"


def _resolve_quality_key(npz, modality: str) -> str | None:
    candidates = (
        f"{modality}_quality_mask",
        f"{modality}_quality",
        f"{modality}_valid_mask",
    )
    return next((candidate for candidate in candidates if candidate in npz), None)


def _slice_quality_mask(npz, key: str, start: int, end: int) -> list[float]:
    raw = np.asarray(npz[key])
    if raw.ndim == 0:
        return [float(raw.item())] * (end - start)
    if raw.ndim != 1:
        raise ValueError(f"Quality mask {key!r} must be scalar or 1D, got shape {raw.shape}.")
    if raw.shape[0] < end:
        raise ValueError(f"Quality mask {key!r} is too short for epochs {start}:{end}.")
    return [float(value) for value in raw[start:end]]


def _matrix_from_npz_signal(
    npz,
    key: str,
    modality: str,
    start: int,
    end: int,
) -> np.ndarray:
    spec = MODALITY_SPECS[modality]
    left = start * spec.frames_per_epoch
    right = end * spec.frames_per_epoch
    raw = np.asarray(npz[key])

    if raw.ndim == 1:
        if raw.shape[0] < right:
            raise ValueError(f"Channel {key!r} is too short for epochs {start}:{end}.")
        segment = raw[left:right].reshape(end - start, 1, spec.frames_per_epoch)
    elif raw.ndim == 2 and raw.shape[1] == 1:
        if raw.shape[0] < right:
            raise ValueError(f"Channel {key!r} is too short for epochs {start}:{end}.")
        segment = raw[left:right, 0].reshape(end - start, 1, spec.frames_per_epoch)
    elif raw.ndim == 2 and raw.shape[1] >= right:
        segment = raw[:, left:right].reshape(raw.shape[0], end - start, spec.frames_per_epoch).transpose(1, 0, 2)
    elif raw.ndim == 3 and raw.shape[0] >= end and raw.shape[2] == spec.frames_per_epoch:
        segment = raw[start:end]
    else:
        raise ValueError(f"Channel {key!r} must be 1D, [T, 1], channel-first [C, T], or [E, C, F], got {raw.shape}.")

    if segment.shape[0] != end - start or segment.shape[2] != spec.frames_per_epoch:
        raise ValueError(f"Channel {key!r} yielded invalid window shape {segment.shape}.")
    matrix = segment.reshape((end - start) * segment.shape[1], spec.frames_per_epoch)
    return np.ascontiguousarray(matrix.astype(np.float32, copy=False))


def convert(args: argparse.Namespace) -> Path:
    config = load_sleep2wave_config(args.config)
    if config.data is None:
        raise ValueError("Sleep2Wave conversion requires a config with a data block.")
    context_epochs = config.data.context_epochs
    stride_epochs = context_epochs if args.stride_epochs is None else int(args.stride_epochs)
    if stride_epochs <= 0:
        raise ValueError("--stride-epochs must be positive.")

    index_paths = [Path(path).expanduser() for path in args.index]
    missing = [str(path) for path in index_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Index CSV not found: {missing}")

    columns = IndexColumnConfig(
        path_col=args.path_col,
        duration_col=args.duration_col,
        split_col=args.split_col,
        subject_id_col=args.subject_id_col,
        night_id_col=args.night_id_col,
        source_col=args.source_col,
    )
    df, columns = prepare_sleep2wave_index_frame(_load_index_df(index_paths, columns), columns=columns)
    if args.split is not None:
        df = df[df[columns.split_col].astype("string").isin(set(args.split))].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    mask_columns = resolve_modality_mask_columns(df, require_all=False)
    mask_frame = normalize_mask_frame(df, list(mask_columns.values()))
    prefix_maps = _parse_prefix_maps(args.path_prefix_map)

    kaldi_native_io = _import_kaldi_native_io()
    output_dir = args.output_dir.expanduser()
    manifests_dir = output_dir / "manifests"
    channels_root = output_dir / "channels"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    channels_root.mkdir(parents=True, exist_ok=True)
    manifest_json_path = output_dir / "manifest.json"

    seen_sample_keys: set[str] = set()
    writers: dict[tuple[str, str], t.Any] = {}
    split_dirs: dict[str, str] = {}
    split_keys_by_dir: dict[str, str] = {}
    manifest_rows_by_split: dict[str, list[dict[str, t.Any]]] = {}

    with ExitStack() as stack:

        def ensure_split_writers(split_value: t.Any) -> str:
            split_key = str(split_value)
            if split_key in split_dirs:
                return split_key

            split_dir = _sanitize_key_part(split_key)
            existing_split_key = split_keys_by_dir.get(split_dir)
            if existing_split_key is not None:
                raise ValueError(
                    f"Split labels {existing_split_key!r} and {split_key!r} both map to directory {split_dir!r}."
                )
            split_dirs[split_key] = split_dir
            split_keys_by_dir[split_dir] = split_key
            split_channel_dir = channels_root / split_dir
            split_channel_dir.mkdir(parents=True, exist_ok=True)
            for modality in CANONICAL_MODALITIES:
                ark_path = split_channel_dir / f"{modality}.ark"
                scp_path = split_channel_dir / f"{modality}.scp"
                writers[(split_key, modality)] = stack.enter_context(
                    kaldi_native_io.FloatMatrixWriter(f"ark,scp:{ark_path},{scp_path}")
                )
            return split_key

        for row_number, row in df.iterrows():
            duration = float(row[columns.duration_col])
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError(f"Row {row_number} has invalid duration: {duration!r}")
            night_epoch_count = int(duration // EPOCH_SEC)
            if night_epoch_count < context_epochs:
                continue

            record_key = _record_key_from_row(row, columns)
            source_value = row[columns.source_col]
            if pd.isna(source_value) or str(source_value) == "":
                raise ValueError(f"CSV source field {columns.source_col!r} has an empty value.")

            npz_path = _resolve_npz_path(row[columns.path_col], prefix_maps)
            with load_npz(str(npz_path)) as npz:
                for start in range(0, night_epoch_count - context_epochs + 1, stride_epochs):
                    end = start + context_epochs
                    sample_key = _sample_key(
                        source_value=source_value,
                        record_key=record_key,
                        start=start,
                        end=end,
                    )
                    if sample_key in seen_sample_keys:
                        raise ValueError(f"Duplicate Sleep2Wave Kaldi sample_key generated: {sample_key}")

                    matrices: dict[str, np.ndarray] = {}
                    quality_masks: dict[str, list[float]] = {}
                    for modality in CANONICAL_MODALITIES:
                        mask_col = mask_columns.get(modality)
                        if mask_col is not None and not bool(mask_frame.loc[row_number, mask_col]):
                            continue
                        npz_key = resolve_npz_key(npz, modality)
                        if npz_key is None:
                            continue
                        matrices[modality] = _matrix_from_npz_signal(npz, npz_key, modality, start, end)
                        quality_key = _resolve_quality_key(npz, modality)
                        if quality_key is not None:
                            quality_masks[modality] = _slice_quality_mask(npz, quality_key, start, end)

                    if not matrices:
                        continue

                    seen_sample_keys.add(sample_key)
                    split_key = ensure_split_writers(row[columns.split_col])
                    for modality, matrix in matrices.items():
                        writers[(split_key, modality)].write(sample_key, matrix)

                    manifest_row = dict(row.to_dict())
                    manifest_row.update(
                        {
                            "sample_key": sample_key,
                            "record_key": record_key,
                            "sample_source": source_value,
                            "path": row[columns.path_col],
                            "split": row[columns.split_col],
                            "epoch_start": start,
                            "epoch_end": end,
                            "num_epochs": end - start,
                            "night_epoch_count": night_epoch_count,
                            "available_channels": json.dumps(list(matrices)),
                            "quality_masks": json.dumps(quality_masks),
                        }
                    )
                    manifest_rows_by_split.setdefault(split_key, []).append(manifest_row)

    if not manifest_rows_by_split:
        raise ValueError("No Sleep2Wave Kaldi samples were produced.")

    splits: dict[str, dict[str, t.Any]] = {}
    for split_key, rows in manifest_rows_by_split.items():
        split_dir = split_dirs[split_key]
        manifest_rel = Path("manifests") / f"{split_dir}.csv"
        pd.DataFrame(rows).to_csv(output_dir / manifest_rel, index=False)
        split_channel_dir = channels_root / split_dir
        for modality in CANONICAL_MODALITIES:
            scp_path = split_channel_dir / f"{modality}.scp"
            lines = scp_path.read_text().splitlines()
            lines.sort(key=lambda line: line.split(maxsplit=1)[0])
            scp_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        splits[split_key] = {
            "manifest": manifest_rel.as_posix(),
            "channels": {
                modality: {
                    "frames_per_epoch": int(MODALITY_SPECS[modality].frames_per_epoch),
                    "scp": (Path("channels") / split_dir / f"{modality}.scp").as_posix(),
                }
                for modality in CANONICAL_MODALITIES
            },
        }

    manifest_json = {
        "format_version": 2,
        "backend": "kaldi_native_io",
        "epoch_sec": EPOCH_SEC,
        "context_epochs": int(context_epochs),
        "stride_epochs": int(stride_epochs),
        "source_index": [str(path) for path in index_paths],
        "splits": splits,
    }
    manifest_json_path.write_text(json.dumps(manifest_json, indent=2) + "\n")
    return manifest_json_path


def main(argv: t.Sequence[str] | None = None) -> None:
    manifest_json_path = convert(parse_args(argv))
    print(f"Wrote {manifest_json_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")


__all__ = ["convert", "main", "parse_args"]
