#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys
import tempfile
import time
import typing as t

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_tools.progress import write_progress

DEFAULT_SPLITS = ["test", "val", "train"]
BUILTIN_CHANNEL_SPECS = {
    "stage5": {"input_dim": 1, "mask_column": "stage_mask"},
    "ahi": {"input_dim": 30, "mask_column": "ah_event_mask"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one or more PSGPretrainDataset preset pickles.",
    )
    parser.add_argument(
        "--index",
        nargs="+",
        required=True,
        help="Single index CSV file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config whose model.channels define dataset channel names and input_dim values.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name used in the output filename. If omitted, inferred from index name.",
    )
    parser.add_argument(
        "--output-template",
        default="data/{dataset}_{split}_preset_{tokens}{meta_suffix}.pickle",
        help=(
            "Output path template. Available fields: {dataset}, {split}, {tokens}, {n_tokens}, {meta}, {meta_suffix}."
        ),
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=DEFAULT_SPLITS,
        choices=["train", "val", "test", "external"],
        help="Split(s) to generate.",
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=1535,
        help="Maximum number of tokens per sample window.",
    )
    parser.add_argument(
        "--stride-tokens",
        type=int,
        default=None,
        help="Stride between windows. Default: 0 when n_tokens=1535, otherwise n_tokens.",
    )
    parser.add_argument(
        "--include-overlap-eval-splits",
        action="store_true",
        help="Keep val/test splits when overlapping windows are enabled.",
    )
    parser.add_argument(
        "--meta-data-names",
        nargs="*",
        default=[],
        help="Metadata field(s). One preset is generated per field.",
    )
    parser.add_argument(
        "--include-no-metadata",
        action="store_true",
        help="Also generate presets without metadata filtering when --meta-data-names is set.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help=(
            "Optional subset of channels declared in YAML model.channels. "
            "Built-in validation channels 'stage5' and 'ahi' are also allowed."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Dataloader batch size in preset filtering.",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to set shuffle in dataloader config.",
    )
    parser.add_argument("--mask-rate", type=float, default=0.0, help="MLM mask rate.")
    parser.add_argument(
        "--allow-missing-channels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow samples missing some channels during preset filtering.",
    )
    parser.add_argument(
        "--min-channels",
        type=int,
        default=2,
        help="Minimum available channels required when --allow-missing-channels is enabled.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing preset files.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Total parallelism budget for preset generation. Defaults to automatic worker selection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing files.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional aggregate JSON manifest path for generated presets.",
    )
    parser.add_argument(
        "--write-sidecar-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write <preset>.manifest.json next to each generated preset.",
    )
    return parser.parse_args()


def _dedupe_keep_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _infer_dataset_name(index_paths: list[Path]) -> str:
    if len(index_paths) != 1:
        return "multi"

    stem = index_paths[0].stem
    for suffix in (
        "_d_merged_with_diseases",
        "_merged_with_diseases",
        "_psg_pretrain",
        "_merged",
        "_index",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or "dataset"


def _resolve_meta_names(meta_data_names: list[str], include_no_metadata: bool) -> list[str | None]:
    if not meta_data_names:
        return [None]

    resolved: list[str | None] = []
    if include_no_metadata:
        resolved.append(None)
    resolved.extend(_dedupe_keep_order(meta_data_names))
    return resolved


def _render_output_path(
    output_template: str,
    dataset_name: str,
    split: str,
    n_tokens: int,
    meta_data_name: str | None,
) -> Path:
    meta = meta_data_name or "none"
    meta_suffix = f"_{meta_data_name}" if meta_data_name else ""
    try:
        rendered = output_template.format(
            dataset=dataset_name,
            split=split,
            tokens=n_tokens,
            n_tokens=n_tokens,
            meta=meta,
            meta_suffix=meta_suffix,
        )
    except KeyError as exc:
        name = exc.args[0]
        raise ValueError(
            f"Unknown field in --output-template: {{{name}}}. "
            "Supported fields: {dataset}, {split}, {tokens}, {n_tokens}, {meta}, {meta_suffix}.",
        ) from exc
    return Path(rendered).expanduser()


def _load_config_mapping(config_path: Path) -> dict[str, t.Any]:
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping with model.channels.")
    return data


def _load_model_channels(
    config_data: dict[str, t.Any],
) -> tuple[list[str], dict[str, int]]:
    model_block = config_data.get("model")
    if not isinstance(model_block, dict):
        raise ValueError("Config YAML must contain a top-level model.channels list.")

    channels_raw = model_block.get("channels")
    if not isinstance(channels_raw, list) or not channels_raw:
        raise ValueError("Config YAML must contain a non-empty model.channels list.")

    all_channels: list[str] = []
    all_channel_input_dims: dict[str, int] = {}
    for item in channels_raw:
        if not isinstance(item, dict):
            raise ValueError("Each model.channels entry must be a mapping.")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Each model.channels entry must define a non-empty string 'name'.")
        if "input_dim" not in item:
            raise ValueError(f"Channel '{name}' must define input_dim.")
        all_channels.append(name)
        all_channel_input_dims[name] = int(item["input_dim"])
    return all_channels, all_channel_input_dims


def _load_model_channel_aliases(config_data: dict[str, t.Any]) -> dict[str, list[str]]:
    model_block = config_data.get("model")
    if not isinstance(model_block, dict):
        raise ValueError("Config YAML must contain a top-level model.channels list.")

    channels_raw = model_block.get("channels")
    if not isinstance(channels_raw, list) or not channels_raw:
        raise ValueError("Config YAML must contain a non-empty model.channels list.")

    aliases_by_channel: dict[str, list[str]] = {}
    for item in channels_raw:
        if not isinstance(item, dict):
            raise ValueError("Each model.channels entry must be a mapping.")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Each model.channels entry must define a non-empty string 'name'.")
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(alias, str) and alias for alias in aliases):
            raise ValueError(f"Channel '{name}' aliases must be a list of non-empty strings.")
        if aliases:
            aliases_by_channel[name] = list(aliases)
    return aliases_by_channel


def _load_preset_build_block(
    config_data: dict[str, t.Any],
) -> tuple[list[str] | None, int | None]:
    raw = config_data.get("preset_build")
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        raise ValueError("preset_build must be a mapping when provided.")

    allowed = {"required_channels", "min_channels"}
    extra = sorted(set(raw.keys()) - allowed)
    if extra:
        raise ValueError(f"preset_build has unsupported fields: {extra}")

    required_channels = raw.get("required_channels")
    if required_channels is not None:
        if not isinstance(required_channels, list) or not required_channels:
            raise ValueError("preset_build.required_channels must be a non-empty list when provided.")
        if not all(isinstance(name, str) and name for name in required_channels):
            raise ValueError("preset_build.required_channels must contain non-empty strings.")
        if len(set(required_channels)) != len(required_channels):
            raise ValueError("preset_build.required_channels must not contain duplicates.")

    min_channels = raw.get("min_channels")
    if min_channels is not None:
        if not isinstance(min_channels, int) or min_channels < 1:
            raise ValueError("preset_build.min_channels must be an integer >= 1 when provided.")

    if required_channels is None or min_channels is None:
        raise ValueError("preset_build must define both preset_build.required_channels and preset_build.min_channels.")

    return required_channels, min_channels


def _load_survival_build_config(config_data: dict[str, t.Any]) -> tuple[t.Any | None, int | None]:
    finetune_block = config_data.get("finetune")
    if not isinstance(finetune_block, dict):
        return None, None

    task_block = finetune_block.get("task")
    if not isinstance(task_block, dict) or task_block.get("type") != "survival":
        if finetune_block.get("survival") is not None:
            raise ValueError("finetune.survival is only supported when finetune.task.type is survival.")
        return None, None

    output_dim = task_block.get("output_dim")
    if not isinstance(output_dim, int) or output_dim < 1:
        raise ValueError("finetune.task.output_dim must be a positive integer for survival preset generation.")

    survival_block = finetune_block.get("survival")
    if not isinstance(survival_block, dict):
        raise ValueError("finetune.survival is required for survival preset generation.")

    from sleep2vec.config import SurvivalConfig

    return SurvivalConfig(**survival_block), output_dim


def _load_multilabel_build_config(config_data: dict[str, t.Any]) -> tuple[t.Any | None, int | None]:
    finetune_block = config_data.get("finetune")
    if not isinstance(finetune_block, dict):
        return None, None

    task_block = finetune_block.get("task")
    if not isinstance(task_block, dict) or task_block.get("type") != "multilabel_classification":
        if finetune_block.get("multilabel") is not None:
            raise ValueError(
                "finetune.multilabel is only supported when finetune.task.type is multilabel_classification."
            )
        return None, None

    output_dim = task_block.get("output_dim")
    if not isinstance(output_dim, int) or output_dim < 1:
        raise ValueError("finetune.task.output_dim must be a positive integer for multilabel preset generation.")

    multilabel_block = finetune_block.get("multilabel")
    if not isinstance(multilabel_block, dict):
        raise ValueError("finetune.multilabel is required for multilabel preset generation.")

    from sleep2vec.config import MultilabelConfig

    return MultilabelConfig(**multilabel_block), output_dim


def _resolve_validation_channels(
    *,
    model_channels: list[str],
    channel_input_dims: dict[str, int],
    preset_required_channels: list[str] | None,
    selected_channels: list[str] | None,
) -> tuple[list[str], dict[str, int]]:
    if preset_required_channels is not None:
        if selected_channels is not None:
            raise ValueError("--channels cannot be used when preset_build.required_channels is set in the YAML.")
        resolved = list(preset_required_channels)
    elif selected_channels is None:
        resolved = list(model_channels)
    else:
        resolved = _dedupe_keep_order(selected_channels)
    if "ahi" in resolved and "stage5" not in resolved:
        resolved = [*resolved, "stage5"]

    unknown = [name for name in resolved if name not in channel_input_dims and name not in BUILTIN_CHANNEL_SPECS]
    if unknown:
        raise ValueError(
            "Channels must be declared in YAML model.channels or "
            "preset_build.required_channels must use built-ins only. "
            f"Unknown: {unknown}; model channels: {model_channels}"
        )

    resolved_dims: dict[str, int] = {}
    for name in resolved:
        if name in channel_input_dims:
            resolved_dims[name] = channel_input_dims[name]
        else:
            resolved_dims[name] = int(BUILTIN_CHANNEL_SPECS[name]["input_dim"])
    return resolved, resolved_dims


def _resolve_effective_min_channels(
    *,
    channel_names: t.Sequence[str],
    cli_min_channels: int,
    preset_min_channels: int | None,
) -> int:
    resolved = int(cli_min_channels if preset_min_channels is None else preset_min_channels)
    if "ahi" in channel_names:
        resolved = len(channel_names)
    if resolved < 1:
        raise ValueError("min_channels must be >= 1.")
    if resolved > len(channel_names):
        raise ValueError(
            f"min_channels={resolved} exceeds the number of validation channels "
            f"({len(channel_names)}): {list(channel_names)}"
        )
    return resolved


def _resolve_channels_and_dims(
    config_path: Path, selected_channels: list[str] | None
) -> tuple[list[str], dict[str, int]]:
    data = _load_config_mapping(config_path)
    model_channels, channel_input_dims = _load_model_channels(data)
    preset_required_channels, _ = _load_preset_build_block(data)
    return _resolve_validation_channels(
        model_channels=model_channels,
        channel_input_dims=channel_input_dims,
        preset_required_channels=preset_required_channels,
        selected_channels=selected_channels,
    )


def _mask_column_for_channel(channel_name: str) -> str:
    spec = BUILTIN_CHANNEL_SPECS.get(channel_name)
    if spec is not None:
        return str(spec["mask_column"])
    return f"{channel_name}_mask"


def _resolve_single_index_path(index_paths: list[str]) -> Path:
    if not index_paths:
        raise ValueError("index list is empty.")
    if len(index_paths) != 1:
        raise ValueError("save_dataset_presets.py accepts exactly one index CSV.")
    return Path(index_paths[0]).expanduser()


def _load_index_df(
    index_paths: list[str],
    survival_key_column: str | None = None,
    multilabel_key_column: str | None = None,
) -> pd.DataFrame:
    path = _resolve_single_index_path(index_paths)
    raw_key_column = survival_key_column if survival_key_column not in (None, "") else multilabel_key_column
    key_column = None if raw_key_column in (None, "") else str(raw_key_column)
    read_csv_kwargs: dict[str, t.Any] = {"low_memory": False}
    if key_column is not None:
        read_csv_kwargs["converters"] = {key_column: str}
    df = pd.read_csv(path, **read_csv_kwargs)
    if "source" not in df.columns:
        df["source"] = str(path)
    else:
        df["source"] = df["source"].where(df["source"].notna(), str(path))
    return df


def _filter_index_df_for_required_channels(df: pd.DataFrame, required_channels: list[str]) -> pd.DataFrame:
    from preprocess.split_index_by_dataset import normalize_mask_frame

    required = _dedupe_keep_order(required_channels)
    if not required:
        return df

    mask_columns = {channel: _mask_column_for_channel(channel) for channel in required}
    if "ahi" in required and "stage5" in required and mask_columns["stage5"] not in df.columns:
        raise ValueError(
            "Built-in AHI strict preset filtering requires index column 'stage_mask' "
            "because validation channels include both 'ahi' and 'stage5'."
        )
    available_mask_columns = [mask_columns[channel] for channel in required if mask_columns[channel] in df.columns]
    if not available_mask_columns:
        return df

    mask_frame = normalize_mask_frame(df, available_mask_columns)
    keep_mask = mask_frame.all(axis=1)
    filtered = df.loc[keep_mask].copy()
    if filtered.empty:
        raise ValueError(f"No rows satisfy required mask columns for channels: {required}")
    return filtered


def _build_preset_job(
    *,
    output_path: Path,
    index_paths: list[str],
    channel_names: list[str],
    channel_input_dims: dict[str, int],
    split: str,
    meta_data_name: str | None,
    n_tokens: int,
    stride_tokens: int,
    mask_rate: float,
    allow_missing_channels: bool,
    min_channels: int,
    batch_size: int,
    shuffle: bool,
    survival_label_config: t.Any | None = None,
    survival_output_dim: int | None = None,
    multilabel_label_config: t.Any | None = None,
    multilabel_output_dim: int | None = None,
    channel_aliases: t.Mapping[str, t.Sequence[str]] | None = None,
    filter_max_workers: int | None,
) -> tuple[Path, int]:
    from data.psg_pretrain_dataset import PSGPretrainDataset

    index = [str(_resolve_single_index_path(index_paths))]
    filtered_index_path: str | None = None
    if not allow_missing_channels:
        survival_key_column = getattr(survival_label_config, "key_column", None)
        multilabel_key_column = getattr(multilabel_label_config, "key_column", None)
        filtered_index = _filter_index_df_for_required_channels(
            _load_index_df(
                index_paths,
                survival_key_column=survival_key_column,
                multilabel_key_column=multilabel_key_column,
            ),
            channel_names,
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix=f"{output_path.stem}.required_masks.",
            delete=False,
        ) as tmp:
            filtered_index.to_csv(tmp.name, index=False)
            filtered_index_path = tmp.name
        index = [filtered_index_path]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataset = PSGPretrainDataset(
            channel_names=channel_names,
            channel_input_dims=channel_input_dims,
            save_preset_path=str(output_path),
            load_preset_path=None,
            index=index,
            meta_data_names=[meta_data_name] if meta_data_name else [],
            split=split,
            max_tokens=n_tokens,
            stride_tokens=stride_tokens,
            mask_rate=mask_rate,
            allow_missing_channels=allow_missing_channels,
            min_channels=min_channels,
            survival_label_config=survival_label_config,
            survival_output_dim=survival_output_dim,
            multilabel_label_config=multilabel_label_config,
            multilabel_output_dim=multilabel_output_dim,
            channel_aliases=channel_aliases,
            batch_size=batch_size,
            shuffle=shuffle,
            filter_max_workers=filter_max_workers,
        )
        return output_path, len(dataset)
    finally:
        if filtered_index_path is not None:
            Path(filtered_index_path).unlink(missing_ok=True)


def _summarize_preset_items(output_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    if not output_path.exists():
        return {}, {}
    with output_path.open("rb") as file_obj:
        items = pickle.load(file_obj)
    available_channels_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for item in items if isinstance(items, list) else []:
        payload = getattr(item, "payload", {})
        if isinstance(payload, dict):
            for channel in payload.get("available_channels", []) or []:
                available_channels_counts[str(channel)] = available_channels_counts.get(str(channel), 0) + 1
        metadata = getattr(item, "metadata", {})
        if isinstance(metadata, dict):
            source = metadata.get("source")
            if source not in (None, ""):
                source_counts[str(source)] = source_counts.get(str(source), 0) + 1
    return available_channels_counts, source_counts


def _preset_manifest_payload(
    *,
    output_path: Path,
    config_path: Path,
    index_paths: list[Path],
    dataset_name: str,
    split: str,
    n_tokens: int,
    stride_tokens: int,
    channels: list[str],
    allow_missing_channels: bool,
    min_channels: int,
    meta_data_name: str | None,
    sample_count: int,
) -> dict[str, t.Any]:
    available_counts, source_counts = _summarize_preset_items(output_path)
    return {
        "kind": "sleep2vec_preset",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "preprocess/save_dataset_presets.py",
        "config_path": str(config_path),
        "index_paths": [str(path) for path in index_paths],
        "dataset_name": dataset_name,
        "split": split,
        "n_tokens": n_tokens,
        "stride_tokens": stride_tokens,
        "channels": channels,
        "allow_missing_channels": allow_missing_channels,
        "min_channels": min_channels,
        "meta_data_name": meta_data_name,
        "output_path": str(output_path),
        "sample_count": sample_count,
        "available_channels_counts": available_counts,
        "source_counts": source_counts,
        "warnings": [],
    }


def _write_json(path: Path, payload: t.Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()

    args.config = args.config.expanduser()
    if not args.config.exists():
        raise FileNotFoundError(f"Config YAML not found: {args.config}")
    if args.num_workers is not None and args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    index_paths = [Path(p).expanduser() for p in args.index]
    missing = [str(p) for p in index_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Index CSV not found: {missing}")
    if len(index_paths) != 1:
        raise ValueError("save_dataset_presets.py accepts exactly one index CSV.")

    config_data = _load_config_mapping(args.config)
    model_channels, model_channel_input_dims = _load_model_channels(config_data)
    model_channel_aliases = _load_model_channel_aliases(config_data)
    preset_required_channels, preset_min_channels = _load_preset_build_block(config_data)
    survival_label_config, survival_output_dim = _load_survival_build_config(config_data)
    multilabel_label_config, multilabel_output_dim = _load_multilabel_build_config(config_data)
    channel_names, channel_input_dims = _resolve_validation_channels(
        model_channels=model_channels,
        channel_input_dims=model_channel_input_dims,
        preset_required_channels=preset_required_channels,
        selected_channels=args.channels,
    )
    channel_aliases = {name: aliases for name, aliases in model_channel_aliases.items() if name in channel_names}
    if args.allow_missing_channels:
        effective_min_channels = _resolve_effective_min_channels(
            channel_names=channel_names,
            cli_min_channels=args.min_channels,
            preset_min_channels=preset_min_channels,
        )
    else:
        effective_min_channels = int(args.min_channels if preset_min_channels is None else preset_min_channels)
    dataset_name = args.dataset_name or _infer_dataset_name(index_paths)
    splits = _dedupe_keep_order(args.split)
    meta_data_variants = _resolve_meta_names(args.meta_data_names, args.include_no_metadata)
    stride_tokens = (
        args.stride_tokens if args.stride_tokens is not None else (0 if args.n_tokens == 1535 else args.n_tokens)
    )
    if 0 < stride_tokens < args.n_tokens and not args.include_overlap_eval_splits:
        original_splits = list(splits)
        splits = [split for split in splits if split not in {"val", "test"}]
        if splits != original_splits:
            print("Overlap windows enabled; excluding val/test splits unless --include-overlap-eval-splits is set.")
        if not splits:
            raise ValueError("Overlap windows excluded val/test splits and no splits remain.")

    print(f"Dataset name: {dataset_name}")
    print(f"Config YAML: {args.config}")
    print(f"Index CSV(s): {[str(p) for p in index_paths]}")
    print(f"Channels: {channel_names}")
    print(f"Effective min_channels={effective_min_channels}")
    print(f"Splits: {splits}")
    print(f"Metadata variants: {[m if m else 'none' for m in meta_data_variants]}")
    print(f"n_tokens={args.n_tokens}, stride_tokens={stride_tokens}")
    print(f"num_workers={args.num_workers if args.num_workers is not None else 'auto'}")
    if args.allow_missing_channels:
        print("Index mask prefilter: disabled (allow_missing_channels=True)")
    else:
        print(f"Index mask prefilter: required channels {channel_names}")

    planned = 0
    created = 0
    skipped = 0
    jobs: list[dict[str, object]] = []
    manifests: list[dict[str, t.Any]] = []
    planned_output_paths: list[Path] = []

    for meta_data_name in meta_data_variants:
        for split in splits:
            output_path = _render_output_path(
                output_template=args.output_template,
                dataset_name=dataset_name,
                split=split,
                n_tokens=args.n_tokens,
                meta_data_name=meta_data_name,
            )
            planned += 1
            planned_output_paths.append(output_path)

            if output_path.exists() and not args.overwrite:
                print(f"[skip] exists: {output_path} (pass --overwrite to regenerate)")
                skipped += 1
                continue

            print(
                f"[build] split={split} meta={meta_data_name or 'none'} -> {output_path}",
            )
            if args.dry_run:
                continue

            jobs.append(
                {
                    "output_path": output_path,
                    "index_paths": [str(p) for p in index_paths],
                    "channel_names": channel_names,
                    "channel_input_dims": channel_input_dims,
                    "channel_aliases": channel_aliases,
                    "split": split,
                    "meta_data_name": meta_data_name,
                    "n_tokens": args.n_tokens,
                    "stride_tokens": stride_tokens,
                    "mask_rate": args.mask_rate,
                    "allow_missing_channels": args.allow_missing_channels,
                    "min_channels": effective_min_channels,
                    "batch_size": args.batch_size,
                    "shuffle": args.shuffle,
                    "survival_label_config": survival_label_config,
                    "survival_output_dim": survival_output_dim,
                    "multilabel_label_config": multilabel_label_config,
                    "multilabel_output_dim": multilabel_output_dim,
                    "dataset_name": dataset_name,
                    "config_path": args.config,
                }
            )

    if not args.dry_run:
        progress_dir = (
            args.manifest_output.expanduser().parent
            if args.manifest_output is not None
            else (planned_output_paths[0].parent if planned_output_paths else Path.cwd())
        )
        started_at = time.time()
        write_progress(
            progress_dir,
            status="running",
            task="save_dataset_presets",
            processed=0,
            total=len(jobs),
            success=0,
            failed=0,
            start_time=started_at,
            message=f"planned={planned} skipped={skipped}",
        )
        try:
            if len(jobs) <= 1:
                for job in jobs:
                    build_job = dict(job)
                    dataset_name_for_manifest = t.cast(str, build_job.pop("dataset_name"))
                    config_path_for_manifest = t.cast(Path, build_job.pop("config_path"))
                    output_path, sample_count = _build_preset_job(
                        **build_job,
                        filter_max_workers=args.num_workers,
                    )
                    manifest = _preset_manifest_payload(
                        output_path=output_path,
                        config_path=config_path_for_manifest,
                        index_paths=index_paths,
                        dataset_name=dataset_name_for_manifest,
                        split=t.cast(str, job["split"]),
                        n_tokens=t.cast(int, job["n_tokens"]),
                        stride_tokens=t.cast(int, job["stride_tokens"]),
                        channels=t.cast(list[str], job["channel_names"]),
                        allow_missing_channels=t.cast(bool, job["allow_missing_channels"]),
                        min_channels=t.cast(int, job["min_channels"]),
                        meta_data_name=t.cast(str | None, job["meta_data_name"]),
                        sample_count=sample_count,
                    )
                    manifests.append(manifest)
                    if args.write_sidecar_manifest:
                        _write_json(output_path.with_name(f"{output_path.name}.manifest.json"), manifest)
                    print(f"  done: {output_path} ({sample_count} samples)")
                    created += 1
                    write_progress(
                        progress_dir,
                        status="running",
                        task="save_dataset_presets",
                        processed=created,
                        total=len(jobs),
                        success=created,
                        failed=0,
                        start_time=started_at,
                        current_item=str(output_path),
                    )
            else:
                process_workers = len(jobs) if args.num_workers is None else min(args.num_workers, len(jobs))
                with ProcessPoolExecutor(max_workers=process_workers) as executor:
                    future_to_job = {
                        executor.submit(
                            _build_preset_job,
                            **{key: value for key, value in job.items() if key not in {"dataset_name", "config_path"}},
                            filter_max_workers=args.num_workers,
                        ): job
                        for job in jobs
                    }
                    for future in as_completed(future_to_job):
                        job = future_to_job[future]
                        output_path, sample_count = future.result()
                        manifest = _preset_manifest_payload(
                            output_path=output_path,
                            config_path=t.cast(Path, job["config_path"]),
                            index_paths=index_paths,
                            dataset_name=t.cast(str, job["dataset_name"]),
                            split=t.cast(str, job["split"]),
                            n_tokens=t.cast(int, job["n_tokens"]),
                            stride_tokens=t.cast(int, job["stride_tokens"]),
                            channels=t.cast(list[str], job["channel_names"]),
                            allow_missing_channels=t.cast(bool, job["allow_missing_channels"]),
                            min_channels=t.cast(int, job["min_channels"]),
                            meta_data_name=t.cast(str | None, job["meta_data_name"]),
                            sample_count=sample_count,
                        )
                        manifests.append(manifest)
                        if args.write_sidecar_manifest:
                            _write_json(output_path.with_name(f"{output_path.name}.manifest.json"), manifest)
                        print(f"  done: {output_path} ({sample_count} samples)")
                        created += 1
                        write_progress(
                            progress_dir,
                            status="running",
                            task="save_dataset_presets",
                            processed=created,
                            total=len(jobs),
                            success=created,
                            failed=0,
                            start_time=started_at,
                            current_item=str(output_path),
                        )
            if args.manifest_output is not None:
                _write_json(args.manifest_output.expanduser(), {"presets": manifests})
        except Exception as exc:
            write_progress(
                progress_dir,
                status="failed",
                task="save_dataset_presets",
                processed=created,
                total=len(jobs),
                success=created,
                failed=1,
                start_time=started_at,
                message=str(exc),
            )
            raise
        write_progress(
            progress_dir,
            status="completed",
            task="save_dataset_presets",
            processed=len(jobs),
            total=len(jobs),
            success=created,
            failed=0,
            start_time=started_at,
            message=f"Completed. Created {created}, skipped {skipped}.",
        )

    if args.dry_run:
        print(f"Dry run complete. Planned {planned} preset(s); no files were written.")
    else:
        print(f"Completed. Created {created}, skipped {skipped}.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")
