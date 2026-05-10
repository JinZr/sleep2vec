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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.psg_pretrain_dataset import _build_channel_registry
from data.utils import load_builtin_ahi_metadata, load_npz, window
from preprocess.save_dataset_presets import (
    BUILTIN_CHANNEL_SPECS,
    _dedupe_keep_order,
    _load_config_mapping,
    _load_model_channels,
    _mask_column_for_channel,
)
from preprocess.split_index_by_dataset import normalize_mask_frame


def parse_args(argv: t.Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CSV-indexed NPZ sleep windows to channel-separated Kaldi ark/scp files.",
    )
    parser.add_argument("--index", nargs="+", required=True, help="Input index CSV file(s).")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config whose model.channels define channel names and input_dim values.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output Kaldi data root.")
    parser.add_argument("--max-tokens", type=int, required=True, help="Maximum tokens per output sample.")
    parser.add_argument(
        "--stride-tokens",
        type=int,
        default=None,
        help="Window stride in tokens. Defaults to --max-tokens.",
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


def _read_config_channels(config_path: Path) -> tuple[list[str], dict[str, int]]:
    config_data = _load_config_mapping(config_path)
    return _load_model_channels(config_data)


def _resolve_channels(args: argparse.Namespace) -> tuple[list[str], dict[str, int]]:
    model_channels, model_channel_input_dims = _read_config_channels(args.config)
    channels: list[str] = []
    if args.channels_from_config:
        channels.extend(model_channels)
    channels.extend(args.extra_channels or [])
    channels = _dedupe_keep_order(channels)
    if "ahi" in channels and "stage5" not in channels:
        channels.append("stage5")
    if not channels:
        raise ValueError("No channels selected. Use --channels-from-config and/or --extra-channels.")

    unknown = [name for name in channels if name not in model_channel_input_dims and name not in BUILTIN_CHANNEL_SPECS]
    if unknown:
        raise ValueError(
            "Requested channels must be declared in YAML model.channels or be built-in channels. "
            f"Unknown: {unknown}; model channels: {model_channels}"
        )

    channel_input_dims: dict[str, int] = {}
    for name in channels:
        if name in model_channel_input_dims:
            channel_input_dims[name] = int(model_channel_input_dims[name])
        else:
            channel_input_dims[name] = int(BUILTIN_CHANNEL_SPECS[name]["input_dim"])
    return channels, channel_input_dims


def _load_index_df(index_paths: t.Sequence[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path, low_memory=False) for path in index_paths]
    if not frames:
        raise ValueError("At least one --index CSV is required.")
    return pd.concat(frames, ignore_index=True)


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


def _tokens_to_matrix(tokens, *, channel: str, sample_key: str, expected_tokens: int) -> np.ndarray:
    if tokens.dim() != 2:
        raise ValueError(
            f"Channel {channel!r} for sample {sample_key!r} tokenized to rank {tokens.dim()} "
            f"with shape {tuple(tokens.shape)}; expected rank 2. "
            "Multichannel raw arrays are not supported by the Kaldi converter yet."
        )
    if tokens.shape[0] != expected_tokens:
        raise ValueError(
            f"Channel {channel!r} for sample {sample_key!r} produced {tokens.shape[0]} tokens, "
            f"expected {expected_tokens} from the CSV duration/window."
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
    return _tokens_to_matrix(tokens, channel=channel, sample_key=sample_key, expected_tokens=end - start)


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
    return _sanitize_key_part(Path(original_path).stem)


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


def convert(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be >= 1.")
    if args.token_sec < 1:
        raise ValueError("--token-sec must be >= 1.")
    stride_tokens = args.stride_tokens if args.stride_tokens is not None else args.max_tokens
    if stride_tokens < 0:
        raise ValueError("--stride-tokens must be >= 0.")
    if args.min_channels < 1:
        raise ValueError("--min-channels must be >= 1.")

    args.config = args.config.expanduser()
    if not args.config.exists():
        raise FileNotFoundError(f"Config YAML not found: {args.config}")

    index_paths = [Path(path).expanduser() for path in args.index]
    missing_indexes = [str(path) for path in index_paths if not path.exists()]
    if missing_indexes:
        raise FileNotFoundError(f"Index CSV not found: {missing_indexes}")

    channel_names, channel_input_dims = _resolve_channels(args)
    if args.min_channels > len(channel_names):
        raise ValueError(
            f"--min-channels={args.min_channels} exceeds selected channel count {len(channel_names)}: {channel_names}"
        )
    registry = _build_channel_registry(
        channel_names=channel_names,
        channel_input_dims=channel_input_dims,
        mask_rate=0.0,
    )
    extractors = {name: registry[name][0] for name in channel_names}
    tokenizers = {name: registry[name][1] for name in channel_names}

    df = _load_index_df(index_paths)
    _validate_required_columns(df, args.source_field)
    prefix_maps = _parse_prefix_maps(args.path_prefix_map)

    kaldi_native_io = _import_kaldi_native_io()
    output_dir = args.output_dir.expanduser()
    channels_dir = output_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, t.Any]] = []
    seen_sample_keys: set[str] = set()
    manifest_path = output_dir / "manifest.csv"
    manifest_json_path = output_dir / "manifest.json"

    with ExitStack() as stack:
        writers = {}
        for channel in channel_names:
            ark_path = channels_dir / f"{channel}.ark"
            scp_path = channels_dir / f"{channel}.scp"
            writers[channel] = stack.enter_context(
                kaldi_native_io.FloatMatrixWriter(f"ark,scp:{ark_path},{scp_path}")
            )

        for _, row in df.iterrows():
            source_value = row[args.source_field]
            if pd.isna(source_value) or str(source_value) == "":
                raise ValueError(f"CSV source field {args.source_field!r} has an empty value.")

            original_path = str(row["path"])
            npz_path = _resolve_npz_path(original_path, prefix_maps)
            duration = int(row["duration"])
            num_record_tokens = duration // int(args.token_sec)
            if num_record_tokens <= 0:
                continue

            record_key = _record_key_from_row(row, original_path)
            mask_status = _row_mask_status(row, channel_names)

            with load_npz(str(npz_path)) as npz:
                for left, right in window(num_record_tokens, args.max_tokens, stride_tokens):
                    start = int(left)
                    end = int(right)
                    sample_key = _sample_key(
                        source_value=source_value,
                        record_key=record_key,
                        start=start,
                        end=end,
                    )
                    if sample_key in seen_sample_keys:
                        raise ValueError(f"Duplicate Kaldi sample_key generated: {sample_key}")

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
                        min_channels=args.min_channels,
                    ):
                        continue

                    seen_sample_keys.add(sample_key)
                    for channel, matrix in matrices.items():
                        writers[channel].write(sample_key, matrix)

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
                            "token_end": end,
                            "num_tokens": end - start,
                            "available_channels": json.dumps(available_channels),
                        }
                    )
                    manifest_row.update(scalar_metadata)
                    manifest_rows.append(manifest_row)

    if not manifest_rows:
        raise ValueError("No samples satisfied the requested channel availability rules.")

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    manifest = {
        "format_version": 1,
        "backend": "kaldi_native_io",
        "token_sec": int(args.token_sec),
        "max_tokens": int(args.max_tokens),
        "stride_tokens": int(stride_tokens),
        "channels": {
            channel: {
                "input_dim": int(channel_input_dims[channel]),
                "scp": str((Path("channels") / f"{channel}.scp").as_posix()),
            }
            for channel in channel_names
        },
        "source_index": [str(path) for path in index_paths],
    }
    manifest_json_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path, manifest_json_path


def main(argv: t.Sequence[str] | None = None) -> None:
    manifest_path, manifest_json_path = convert(parse_args(argv))
    print(f"Wrote {manifest_path}")
    print(f"Wrote {manifest_json_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")
