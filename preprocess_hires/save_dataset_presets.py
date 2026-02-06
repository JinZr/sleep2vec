#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CHANNELS = [
    "heartbeat",
    "breath",
    "eeg_original",
    "ecg_original",
    "eog_original",
    "emg_original",
    "spo2",
    "resp_original",
    "resp_nasal_original",
]

DEFAULT_SPLITS = ["test", "val", "train"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one or more PSGPretrainDataset preset pickles.",
    )
    parser.add_argument(
        "--index",
        nargs="+",
        required=True,
        help="One or more index CSV files.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name used in the output filename. If omitted, inferred from index name.",
    )
    parser.add_argument(
        "--output-template",
        default="data_hires/{dataset}_{split}_preset_hires_{tokens}{meta_suffix}.pickle",
        help=(
            "Output path template. Available fields: {dataset}, {split}, {tokens}, {n_tokens}, "
            "{meta}, {meta_suffix}."
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
        default=1200,
        help="Maximum number of tokens per sample window.",
    )
    parser.add_argument(
        "--stride-tokens",
        type=int,
        default=None,
        help="Stride between windows. Default: 0 when n_tokens=1535, otherwise n_tokens.",
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
        default=DEFAULT_CHANNELS,
        help="Channel names to include.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Dataloader batch size in preset filtering.")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to set shuffle in dataloader config.",
    )
    parser.add_argument("--mask-rate", type=float, default=0.0, help="MLM mask rate.")
    parser.add_argument(
        "--use-legacy-body-movement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use legacy body movement processing.",
    )
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
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing files.",
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


def main() -> None:
    args = parse_args()

    index_paths = [Path(p).expanduser() for p in args.index]
    missing = [str(p) for p in index_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Index CSV not found: {missing}")

    dataset_name = args.dataset_name or _infer_dataset_name(index_paths)
    splits = _dedupe_keep_order(args.split)
    meta_data_variants = _resolve_meta_names(args.meta_data_names, args.include_no_metadata)
    stride_tokens = (
        args.stride_tokens if args.stride_tokens is not None else (0 if args.n_tokens == 1535 else args.n_tokens)
    )
    loader_kwargs = {"batch_size": args.batch_size, "shuffle": args.shuffle}

    dataset_cls = None
    if not args.dry_run:
        from data_hires.psg_pretrain_dataset import PSGPretrainDataset

        dataset_cls = PSGPretrainDataset

    print(f"Dataset name: {dataset_name}")
    print(f"Index CSV(s): {[str(p) for p in index_paths]}")
    print(f"Splits: {splits}")
    print(f"Metadata variants: {[m if m else 'none' for m in meta_data_variants]}")
    print(f"n_tokens={args.n_tokens}, stride_tokens={stride_tokens}")

    planned = 0
    created = 0
    skipped = 0

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

            if output_path.exists() and not args.overwrite:
                print(f"[skip] exists: {output_path} (pass --overwrite to regenerate)")
                skipped += 1
                continue

            print(
                f"[build] split={split} meta={meta_data_name or 'none'} -> {output_path}",
            )
            if args.dry_run:
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            dataset = dataset_cls(
                channel_names=args.channels,
                save_preset_path=str(output_path),
                load_preset_path=None,
                index=[str(p) for p in index_paths],
                meta_data_names=[meta_data_name] if meta_data_name else [],
                split=split,
                max_tokens=args.n_tokens,
                stride_tokens=stride_tokens,
                mask_rate=args.mask_rate,
                use_legacy_body_movement=args.use_legacy_body_movement,
                allow_missing_channels=args.allow_missing_channels,
                min_channels=args.min_channels,
                **loader_kwargs,
            )
            print(f"  samples: {len(dataset)}")
            created += 1

    if args.dry_run:
        print(f"Dry run complete. Planned {planned} preset(s); no files were written.")
    else:
        print(f"Completed. Created {created}, skipped {skipped}.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")
