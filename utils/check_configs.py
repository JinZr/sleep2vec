#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import typing as t

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess.save_dataset_presets import (
    _load_config_mapping,
    _load_model_channels,
    _load_preset_build_block,
    _resolve_effective_min_channels,
    _resolve_validation_channels,
)
from sleep2vec.config import load_finetune_config, load_pretrain_config, validate_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate repository YAML configs and local config contracts.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional config paths to validate. Defaults to every YAML under configs/.",
    )
    return parser.parse_args()


def collect_config_paths(paths: list[str] | None = None) -> list[Path]:
    if not paths:
        return sorted(CONFIG_ROOT.rglob("*.yaml"))

    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        if path.is_dir():
            resolved.extend(sorted(path.rglob("*.yaml")))
        else:
            resolved.append(path)
    return sorted(dict.fromkeys(resolved))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _validate_runtime_loader_contract(path: Path, config_data: dict[str, t.Any]) -> None:
    if isinstance(config_data.get("finetune"), dict):
        bundle = load_finetune_config(path)
        validate_model_config(bundle.model)
        return

    bundle = load_pretrain_config(path)
    validate_model_config(bundle.model)


def _validate_preset_build_contract(config_data: dict[str, t.Any]) -> tuple[list[str] | None, int | None]:
    model_channels, channel_input_dims = _load_model_channels(config_data)
    preset_required_channels, preset_min_channels = _load_preset_build_block(config_data)
    if preset_required_channels is None and preset_min_channels is None:
        return None, None
    if preset_required_channels is None or preset_min_channels is None:
        raise ValueError("preset_build must define both required_channels and min_channels when provided.")

    validation_channels, _ = _resolve_validation_channels(
        model_channels=model_channels,
        channel_input_dims=channel_input_dims,
        preset_required_channels=preset_required_channels,
        selected_channels=None,
    )
    _resolve_effective_min_channels(
        channel_names=validation_channels,
        cli_min_channels=2,
        preset_min_channels=preset_min_channels,
    )
    return validation_channels, preset_min_channels


def _is_ppg_finetune_config(path: Path) -> bool:
    return path.name.startswith("ppg_") and "finetune" in path.name and path.suffix == ".yaml"


def _validate_repo_policy(path: Path, config_data: dict[str, t.Any]) -> None:
    model_channels, _ = _load_model_channels(config_data)
    finetune_block = config_data.get("finetune")
    task_block = finetune_block.get("task") if isinstance(finetune_block, dict) else None
    preset_required_channels, preset_min_channels = _load_preset_build_block(config_data)

    if not _is_ppg_finetune_config(path):
        return

    if preset_required_channels is None or preset_min_channels is None:
        raise ValueError(
            "ppg finetune configs must define both preset_build.required_channels and preset_build.min_channels."
        )

    if model_channels != ["ppg"] or not isinstance(task_block, dict):
        return

    is_seq = bool(task_block.get("is_seq", False))
    if is_seq:
        if preset_required_channels != ["ppg", "stage5"]:
            raise ValueError(
                "token-level ppg staging configs must set preset_build.required_channels to [ppg, stage5]."
            )
        if preset_min_channels != 2:
            raise ValueError("token-level ppg staging configs must set preset_build.min_channels to 2.")
        return

    if preset_required_channels != ["ppg"]:
        raise ValueError(
            "single-channel ppg non-seq finetune configs must set preset_build.required_channels to [ppg]."
        )
    if preset_min_channels != 1:
        raise ValueError("single-channel ppg non-seq finetune configs must set preset_build.min_channels to 1.")


def check_config_file(path: Path) -> None:
    config_data = _load_config_mapping(path)
    _validate_runtime_loader_contract(path, config_data)
    _validate_repo_policy(path, config_data)
    _validate_preset_build_contract(config_data)


def main() -> int:
    args = parse_args()
    config_paths = collect_config_paths(args.paths)
    failures: list[tuple[Path, str]] = []

    for path in config_paths:
        try:
            check_config_file(path)
        except Exception as exc:
            failures.append((path, str(exc)))

    if failures:
        for path, message in failures:
            print(f"[FAIL] {_display_path(path)}: {message}")
        return 1

    print(f"Checked {len(config_paths)} config files: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
