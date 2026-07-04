#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import re
import sys
import time
import typing as t

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_tools.progress import write_progress
from data.psg_pretrain_dataset import _build_channel_registry
from data.utils import load_builtin_ahi_metadata, load_npz, window
from preprocess.save_dataset_presets import (
    _load_config_mapping,
    _load_model_channel_aliases,
    _load_model_channels,
    _load_preset_build_block,
    _mask_column_for_channel,
    _resolve_effective_min_channels,
    _resolve_validation_channels,
)
from preprocess.split_index_by_dataset import normalize_mask_frame

UNCOMPRESSED_BUILTIN_CHANNELS = {"stage5", "ahi"}


def parse_args(argv: t.Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CSV-indexed NPZ sleep windows to channel-separated Kaldi ark/scp files.",
    )
    parser.add_argument("--index", nargs="+", required=True, help="Input index CSV file(s).")
    parser.add_argument(
        "--split",
        nargs="+",
        default=None,
        help="Optional CSV split label(s) to convert. Defaults to every split present in the index.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config whose model.channels define channel names and input_dim values.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output Kaldi data root.")
    parser.add_argument(
        "--ark-shards",
        type=int,
        default=1,
        help="Number of ark shards to write per split/channel. Default preserves one ark per split/channel.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of record conversion worker threads.",
    )
    parser.add_argument(
        "--compress-ark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compress non-label ark matrices. Built-in stage5 and ahi stay uncompressed.",
    )
    parser.add_argument("--max-tokens", type=int, required=True, help="Maximum tokens per output sample.")
    parser.add_argument(
        "--stride-tokens",
        type=int,
        default=None,
        help="Window stride in tokens. Defaults to --max-tokens.",
    )
    parser.add_argument(
        "--include-overlap-eval-splits",
        action="store_true",
        help="Apply overlapping windows to val/test rows. By default, val/test rows use non-overlapping stride.",
    )
    parser.add_argument("--token-sec", type=int, default=30, help="Seconds represented by one token.")
    parser.add_argument(
        "--channels-from-config",
        action="store_true",
        help="Include every channel listed in YAML model.channels.",
    )
    parser.add_argument(
        "--extra-channels",
        nargs="*",
        default=[],
        help="Additional channels to include, such as built-ins stage5 and ahi.",
    )
    parser.add_argument(
        "--source-field",
        default="dataset",
        help="CSV column used as the dataset/source prefix in sample keys.",
    )
    parser.add_argument(
        "--allow-missing-channels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep samples with at least --min-channels available channels.",
    )
    parser.add_argument(
        "--min-channels",
        type=int,
        default=2,
        help="Minimum available channels required when --allow-missing-channels is enabled.",
    )
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
            "kaldi_native_io is required to write Kaldi ark/scp files. "
            "Install requirements.txt before running this converter."
        ) from exc
    return kaldi_native_io


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


def _sanitize_key_part(value: t.Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "unknown"


def _resolve_channels(args: argparse.Namespace) -> tuple[list[str], dict[str, int], int]:
    config_data = _load_config_mapping(args.config)
    model_channels, model_channel_input_dims = _load_model_channels(config_data)
    preset_required_channels, preset_min_channels = _load_preset_build_block(config_data)

    selected_channels: list[str] = []
    if preset_required_channels is not None:
        selected_channels.extend(preset_required_channels)
    elif args.channels_from_config:
        selected_channels.extend(model_channels)
    selected_channels.extend(args.extra_channels or [])
    if not selected_channels:
        raise ValueError("No channels selected. Use --channels-from-config and/or --extra-channels.")

    channel_names, channel_input_dims = _resolve_validation_channels(
        model_channels=model_channels,
        channel_input_dims=model_channel_input_dims,
        preset_required_channels=None,
        selected_channels=selected_channels,
    )
    cli_min_channels = args.min_channels if args.allow_missing_channels else min(args.min_channels, len(channel_names))
    effective_min_channels = _resolve_effective_min_channels(
        channel_names=channel_names,
        cli_min_channels=cli_min_channels,
        preset_min_channels=preset_min_channels,
    )
    return channel_names, channel_input_dims, effective_min_channels


def _load_index_df(index_paths: t.Sequence[Path], survival_key_column: str | None = None) -> pd.DataFrame:
    key_column = None if survival_key_column in (None, "") else str(survival_key_column)
    read_csv_kwargs: dict[str, t.Any] = {"low_memory": False}
    if key_column is not None:
        read_csv_kwargs["converters"] = {key_column: str}
    frames = [pd.read_csv(path, **read_csv_kwargs) for path in index_paths]
    if not frames:
        raise ValueError("At least one --index CSV is required.")
    df = pd.concat(frames, ignore_index=True)
    if key_column is not None:
        if key_column not in df.columns:
            raise ValueError(f"Input index CSV is missing required survival key column {key_column!r}.")
        missing_key = df[key_column].isna() | df[key_column].astype(str).str.strip().eq("")
        if missing_key.any():
            raise ValueError(f"Input index CSV contains missing values in required survival key column {key_column!r}.")
    return df


def _validate_required_columns(df: pd.DataFrame, source_field: str) -> None:
    required = {"path", "duration", "split", source_field}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input index CSV is missing required column(s): {missing}.")


def _row_mask_status(row: pd.Series, channel_names: t.Sequence[str]) -> dict[str, bool | None]:
    mask_columns = {channel: _mask_column_for_channel(channel) for channel in channel_names}
    available_columns = [column for column in mask_columns.values() if column in row.index]
    if not available_columns:
        return {channel: None for channel in channel_names}

    frame = pd.DataFrame([{column: row[column] for column in available_columns}])
    normalized = normalize_mask_frame(frame, available_columns).iloc[0]
    return {
        channel: bool(normalized[mask_column]) if mask_column in normalized.index else None
        for channel, mask_column in mask_columns.items()
    }


def _tokens_to_matrix(tokens, *, channel: str, sample_key: str) -> np.ndarray:
    if tokens.dim() != 2:
        raise ValueError(
            f"Channel {channel!r} for sample {sample_key!r} tokenized to rank {tokens.dim()} "
            f"with shape {tuple(tokens.shape)}; expected rank 2. "
            "Multichannel raw arrays are not supported by the Kaldi converter yet."
        )
    arr = tokens.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


def _extract_channel_matrix(
    *,
    npz,
    channel: str,
    extractor: t.Callable,
    tokenizer: t.Callable,
    start: int,
    end: int,
    sample_key: str,
) -> np.ndarray:
    payload = extractor(npz, start, end)
    tokens = tokenizer(payload)
    return _tokens_to_matrix(tokens, channel=channel, sample_key=sample_key)


def _try_extract_channel(
    *,
    npz,
    channel: str,
    extractor: t.Callable,
    tokenizer: t.Callable,
    mask_present: bool | None,
    start: int,
    end: int,
    sample_key: str,
) -> tuple[np.ndarray | None, dict[str, float]]:
    if mask_present is False:
        return None, {}

    try:
        metadata: dict[str, float] = {}
        if channel == "ahi":
            ahi_value, tst_value = load_builtin_ahi_metadata(npz)
            metadata = {"ahi": ahi_value, "tst": tst_value}
        matrix = _extract_channel_matrix(
            npz=npz,
            channel=channel,
            extractor=extractor,
            tokenizer=tokenizer,
            start=start,
            end=end,
            sample_key=sample_key,
        )
        return matrix, metadata
    except KeyError as exc:
        if mask_present is True:
            raise ValueError(
                f"Mask marks channel {channel!r} present for sample {sample_key!r}, but NPZ is missing it."
            ) from exc
        return None, {}
    except Exception as exc:
        raise ValueError(f"Failed converting channel {channel!r} for sample {sample_key!r}: {exc}") from exc


def _record_key_from_row(row: pd.Series, original_path: str) -> str:
    session_id = row.get("session_id", None)
    if session_id is not None and not pd.isna(session_id) and str(session_id):
        return _sanitize_key_part(session_id)
    path = Path(original_path)
    return _sanitize_key_part(f"{path.parent.name}_{path.stem}")


def _sample_key(*, source_value: t.Any, record_key: str, start: int, end: int) -> str:
    return f"{_sanitize_key_part(source_value)}_{record_key}_{start:06d}_{end:06d}"


def _should_keep_sample(
    available_channels: t.Sequence[str],
    requested_channels: t.Sequence[str],
    *,
    allow_missing_channels: bool,
    min_channels: int,
) -> bool:
    if allow_missing_channels:
        return len(available_channels) >= min_channels
    return len(available_channels) == len(requested_channels)


def _validate_unique_sample_keys(
    df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    channel_names: t.Sequence[str],
    stride_tokens: int,
    effective_min_channels: int,
) -> None:
    seen: dict[str, int] = {}
    for row_index, row in df.iterrows():
        source_value = row[args.source_field]
        if pd.isna(source_value) or str(source_value) == "":
            raise ValueError(f"CSV source field {args.source_field!r} has an empty value.")

        duration = int(row["duration"])
        num_record_tokens = duration // int(args.token_sec)
        if num_record_tokens <= 0:
            continue

        mask_status = _row_mask_status(row, channel_names)
        available_channels = [channel for channel in channel_names if mask_status[channel] is not False]
        if not _should_keep_sample(
            available_channels,
            channel_names,
            allow_missing_channels=args.allow_missing_channels,
            min_channels=effective_min_channels,
        ):
            continue

        record_key = _record_key_from_row(row, str(row["path"]))
        sample_stride_tokens = stride_tokens
        if (
            0 < stride_tokens < args.max_tokens
            and not args.include_overlap_eval_splits
            and str(row["split"]) in {"val", "test"}
        ):
            sample_stride_tokens = args.max_tokens
        for left, right in window(num_record_tokens, args.max_tokens, sample_stride_tokens):
            sample_key = _sample_key(
                source_value=source_value,
                record_key=record_key,
                start=int(left),
                end=int(right),
            )
            if sample_key in seen:
                raise ValueError(
                    f"Duplicate Kaldi sample_key generated before writing: {sample_key} "
                    f"(input rows {seen[sample_key]} and {row_index})."
                )
            seen[sample_key] = int(row_index)


def _convert_record(
    row: pd.Series,
    *,
    args: argparse.Namespace,
    channel_names: t.Sequence[str],
    extractors: t.Mapping[str, t.Callable],
    tokenizers: t.Mapping[str, t.Callable],
    prefix_maps: t.Sequence[tuple[str, str]],
    stride_tokens: int,
    effective_min_channels: int,
) -> list[dict[str, t.Any]]:
    source_value = row[args.source_field]
    if pd.isna(source_value) or str(source_value) == "":
        raise ValueError(f"CSV source field {args.source_field!r} has an empty value.")

    original_path = str(row["path"])
    npz_path = _resolve_npz_path(original_path, prefix_maps)
    duration = int(row["duration"])
    num_record_tokens = duration // int(args.token_sec)
    if num_record_tokens <= 0:
        return []

    record_key = _record_key_from_row(row, original_path)
    mask_status = _row_mask_status(row, channel_names)
    sample_stride_tokens = stride_tokens
    if (
        0 < stride_tokens < args.max_tokens
        and not args.include_overlap_eval_splits
        and str(row["split"]) in {"val", "test"}
    ):
        sample_stride_tokens = args.max_tokens
    samples: list[dict[str, t.Any]] = []

    with load_npz(str(npz_path)) as npz:
        for left, right in window(num_record_tokens, args.max_tokens, sample_stride_tokens):
            start = int(left)
            end = int(right)
            sample_key = _sample_key(
                source_value=source_value,
                record_key=record_key,
                start=start,
                end=end,
            )

            matrices: dict[str, np.ndarray] = {}
            scalar_metadata: dict[str, float] = {}
            for channel in channel_names:
                matrix, channel_metadata = _try_extract_channel(
                    npz=npz,
                    channel=channel,
                    extractor=extractors[channel],
                    tokenizer=tokenizers[channel],
                    mask_present=mask_status[channel],
                    start=start,
                    end=end,
                    sample_key=sample_key,
                )
                if matrix is None:
                    continue
                matrices[channel] = matrix
                scalar_metadata.update(channel_metadata)

            available_channels = [channel for channel in channel_names if channel in matrices]
            if not _should_keep_sample(
                available_channels,
                channel_names,
                allow_missing_channels=args.allow_missing_channels,
                min_channels=effective_min_channels,
            ):
                continue

            lengths = [matrix.shape[0] for matrix in matrices.values()]
            min_len = min(lengths)
            max_len = max(lengths)
            if max_len - min_len > 1:
                raise ValueError(
                    f"Sample {sample_key!r} has channel token lengths differing by more than one: "
                    f"{dict((channel, matrix.shape[0]) for channel, matrix in matrices.items())}."
                )
            if min_len < 1:
                continue
            matrices = {channel: matrix[:min_len] for channel, matrix in matrices.items()}
            actual_end = start + min_len

            manifest_row = dict(row.to_dict())
            source = manifest_row.get("source")
            if source is None or pd.isna(source) or str(source).strip() == "":
                manifest_row["source"] = source_value
            manifest_row.update(
                {
                    "sample_key": sample_key,
                    "record_key": record_key,
                    "sample_source": source_value,
                    "path": original_path,
                    "token_start": start,
                    "token_end": actual_end,
                    "num_tokens": min_len,
                    "available_channels": json.dumps(available_channels),
                }
            )
            manifest_row.update(scalar_metadata)
            samples.append(
                {
                    "sample_key": sample_key,
                    "split": row["split"],
                    "matrices": matrices,
                    "manifest_row": manifest_row,
                }
            )

    return samples


def convert(args: argparse.Namespace) -> Path:
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be >= 1.")
    if args.token_sec < 1:
        raise ValueError("--token-sec must be >= 1.")
    stride_tokens = args.stride_tokens if args.stride_tokens is not None else args.max_tokens
    if stride_tokens < 0:
        raise ValueError("--stride-tokens must be >= 0.")
    if args.min_channels < 1:
        raise ValueError("--min-channels must be >= 1.")
    if args.ark_shards < 1:
        raise ValueError("--ark-shards must be >= 1.")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1.")

    args.config = args.config.expanduser()
    if not args.config.exists():
        raise FileNotFoundError(f"Config YAML not found: {args.config}")

    index_paths = [Path(path).expanduser() for path in args.index]
    missing_indexes = [str(path) for path in index_paths if not path.exists()]
    if missing_indexes:
        raise FileNotFoundError(f"Index CSV not found: {missing_indexes}")

    channel_names, channel_input_dims, effective_min_channels = _resolve_channels(args)
    config_data = _load_config_mapping(args.config)
    model_channel_aliases = _load_model_channel_aliases(config_data)
    channel_aliases = {name: alias for name, alias in model_channel_aliases.items() if name in channel_names}
    registry = _build_channel_registry(
        channel_names=channel_names,
        channel_input_dims=channel_input_dims,
        channel_aliases=channel_aliases,
        mask_rate=0.0,
    )
    extractors = {name: registry[name][0] for name in channel_names}
    tokenizers = {name: registry[name][1] for name in channel_names}

    finetune_block = config_data.get("finetune", {})
    survival_block = finetune_block.get("survival", {}) if isinstance(finetune_block, dict) else {}
    survival_key_column = survival_block.get("key_column") if isinstance(survival_block, dict) else None
    df = _load_index_df(index_paths, survival_key_column=survival_key_column)
    _validate_required_columns(df, args.source_field)
    if args.split is not None:
        requested_splits = {str(split) for split in args.split}
        df = df[df["split"].astype(str).isin(requested_splits)].copy()
        if df.empty:
            raise ValueError(f"No rows matched requested --split values: {sorted(requested_splits)}.")
    if 0 < stride_tokens < args.max_tokens and not args.include_overlap_eval_splits:
        if df["split"].astype(str).isin({"val", "test"}).any():
            print(
                "Overlap windows enabled; keeping val/test rows with non-overlapping stride "
                "(stride_tokens=max_tokens). Pass --include-overlap-eval-splits to overlap them."
            )
    prefix_maps = _parse_prefix_maps(args.path_prefix_map)
    _validate_unique_sample_keys(
        df,
        args=args,
        channel_names=channel_names,
        stride_tokens=stride_tokens,
        effective_min_channels=effective_min_channels,
    )

    kaldi_native_io = _import_kaldi_native_io()
    compressed_channels = {
        channel for channel in channel_names if args.compress_ark and channel not in UNCOMPRESSED_BUILTIN_CHANNELS
    }
    compression_method = kaldi_native_io.CompressionMethod.kTwoByteAuto if compressed_channels else None
    output_dir = args.output_dir.expanduser()
    manifests_dir = output_dir / "manifests"
    channels_root = output_dir / "channels"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    channels_root.mkdir(parents=True, exist_ok=True)

    seen_sample_keys: set[str] = set()
    manifest_json_path = output_dir / "manifest.json"
    writers: dict[tuple[str, str, int], t.Any] = {}
    split_channel_storage: dict[tuple[str, str], str] = {}
    split_dirs: dict[str, str] = {}
    split_keys_by_dir: dict[str, str] = {}
    split_sample_counts: dict[str, int] = {}
    manifest_writers: dict[str, csv.DictWriter] = {}
    manifest_columns = list(df.columns)
    for column in (
        "source",
        "sample_key",
        "record_key",
        "sample_source",
        "token_start",
        "token_end",
        "num_tokens",
        "available_channels",
    ):
        if column not in manifest_columns:
            manifest_columns.append(column)
    if "ahi" in channel_names:
        for column in ("ahi", "tst"):
            if column not in manifest_columns:
                manifest_columns.append(column)

    with ExitStack() as stack:

        def manifest_csv_row(row: dict[str, t.Any]) -> dict[str, t.Any]:
            csv_row = {}
            for column in manifest_columns:
                value = row.get(column, "")
                try:
                    if pd.isna(value):
                        value = ""
                except (TypeError, ValueError):
                    pass
                csv_row[column] = value
            return csv_row

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
            manifest_file = stack.enter_context((manifests_dir / f"{split_dir}.csv").open("w", newline=""))
            manifest_writer = csv.DictWriter(manifest_file, fieldnames=manifest_columns, lineterminator="\n")
            manifest_writer.writeheader()
            manifest_writers[split_key] = manifest_writer
            compress_split = split_key == "train"
            for channel in channel_names:
                storage = "compressed_matrix" if compress_split and channel in compressed_channels else "float_matrix"
                split_channel_storage[(split_key, channel)] = storage
                writer_cls = (
                    kaldi_native_io.CompressedMatrixWriter
                    if storage == "compressed_matrix"
                    else kaldi_native_io.FloatMatrixWriter
                )
                if args.ark_shards == 1:
                    ark_path = split_channel_dir / f"{channel}.ark"
                    scp_path = split_channel_dir / f"{channel}.scp"
                    writers[(split_key, channel, 0)] = stack.enter_context(writer_cls(f"ark,scp:{ark_path},{scp_path}"))
                else:
                    for shard_index in range(args.ark_shards):
                        shard_no = shard_index + 1
                        ark_path = split_channel_dir / f"{channel}.{shard_no}.ark"
                        scp_path = split_channel_dir / f"{channel}.{shard_no}.scp"
                        writers[(split_key, channel, shard_index)] = stack.enter_context(
                            writer_cls(f"ark,scp:{ark_path},{scp_path}")
                        )
            return split_key

        def convert_row(row: pd.Series) -> list[dict[str, t.Any]]:
            return _convert_record(
                row,
                args=args,
                channel_names=channel_names,
                extractors=extractors,
                tokenizers=tokenizers,
                prefix_maps=prefix_maps,
                stride_tokens=stride_tokens,
                effective_min_channels=effective_min_channels,
            )

        rows = (row for _, row in df.iterrows())
        if args.num_workers == 1:
            converted_records = map(convert_row, rows)
        else:
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=args.num_workers))
            row_iter = iter(rows)
            pending = deque()
            for _ in range(args.num_workers):
                try:
                    pending.append(executor.submit(convert_row, next(row_iter)))
                except StopIteration:
                    break

            def converted_records_iter():
                while pending:
                    future = pending.popleft()
                    samples = future.result()
                    try:
                        pending.append(executor.submit(convert_row, next(row_iter)))
                    except StopIteration:
                        pass
                    yield samples

            converted_records = converted_records_iter()

        started_at = time.time()
        processed = 0
        write_progress(
            output_dir,
            status="running",
            task="convert_npz_to_kaldi",
            processed=0,
            total=len(df),
            success=0,
            failed=0,
            start_time=started_at,
        )
        try:
            for samples in tqdm(converted_records, total=len(df), desc="Converting records", unit="record"):
                processed += 1
                for sample in samples:
                    sample_key = sample["sample_key"]
                    if sample_key in seen_sample_keys:
                        raise ValueError(f"Duplicate Kaldi sample_key generated: {sample_key}")

                    seen_sample_keys.add(sample_key)
                    split_key = ensure_split_writers(sample["split"])
                    split_sample_count = split_sample_counts.get(split_key, 0)
                    shard_index = split_sample_count % args.ark_shards
                    split_sample_counts[split_key] = split_sample_count + 1
                    for channel, matrix in sample["matrices"].items():
                        writer = writers[(split_key, channel, shard_index)]
                        storage = split_channel_storage[(split_key, channel)]
                        if storage == "compressed_matrix":
                            writer.write(sample_key, matrix, method=compression_method)
                        else:
                            writer.write(sample_key, matrix)
                    manifest_writers[split_key].writerow(manifest_csv_row(sample["manifest_row"]))
                write_progress(
                    output_dir,
                    status="running",
                    task="convert_npz_to_kaldi",
                    processed=processed,
                    total=len(df),
                    success=processed,
                    failed=0,
                    start_time=started_at,
                )
        except Exception as exc:
            write_progress(
                output_dir,
                status="failed",
                task="convert_npz_to_kaldi",
                processed=processed,
                total=len(df),
                success=processed,
                failed=1,
                start_time=started_at,
                message=str(exc),
            )
            raise

    if not split_sample_counts:
        message = "No samples satisfied the requested channel availability rules."
        write_progress(
            output_dir,
            status="failed",
            task="convert_npz_to_kaldi",
            processed=len(df),
            total=len(df),
            success=0,
            failed=1,
            start_time=started_at,
            message=message,
        )
        raise ValueError(message)

    splits: dict[str, dict[str, t.Any]] = {}
    for split_key in split_sample_counts:
        split_dir = split_dirs[split_key]
        manifest_rel = Path("manifests") / f"{split_dir}.csv"
        split_channel_dir = channels_root / split_dir
        for channel in channel_names:
            scp_path = split_channel_dir / f"{channel}.scp"
            if args.ark_shards == 1:
                lines = scp_path.read_text().splitlines()
            else:
                lines = []
                for shard_index in range(args.ark_shards):
                    shard_scp_path = split_channel_dir / f"{channel}.{shard_index + 1}.scp"
                    shard_lines = shard_scp_path.read_text().splitlines() if shard_scp_path.exists() else []
                    shard_lines.sort(key=lambda line: line.split(maxsplit=1)[0])
                    shard_scp_path.write_text("\n".join(shard_lines) + ("\n" if shard_lines else ""))
                    lines.extend(shard_lines)
            lines.sort(key=lambda line: line.split(maxsplit=1)[0])
            scp_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        splits[split_key] = {
            "manifest": manifest_rel.as_posix(),
            "channels": {
                channel: {
                    "input_dim": int(channel_input_dims[channel]),
                    "scp": (Path("channels") / split_dir / f"{channel}.scp").as_posix(),
                    "ark_storage": split_channel_storage[(split_key, channel)],
                }
                for channel in channel_names
            },
        }

    manifest = {
        "backend": "kaldi_native_io",
        "token_sec": int(args.token_sec),
        "max_tokens": int(args.max_tokens),
        "stride_tokens": int(stride_tokens),
        "source_index": [str(path) for path in index_paths],
        "splits": splits,
    }
    manifest_json_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_progress(
        output_dir,
        status="completed",
        task="convert_npz_to_kaldi",
        processed=len(df),
        total=len(df),
        success=len(df),
        failed=0,
        start_time=started_at,
        message=f"Wrote {manifest_json_path}",
    )
    return manifest_json_path


def main(argv: t.Sequence[str] | None = None) -> None:
    manifest_json_path = convert(parse_args(argv))
    print(f"Wrote {manifest_json_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")
