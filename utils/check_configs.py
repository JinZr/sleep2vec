#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import sys
import typing as t

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class ConfigVariant:
    config_module: str
    preset_module: str


@dataclass(frozen=True)
class ConfigTools:
    load_model_channels: t.Callable[[dict[str, t.Any]], tuple[list[str], dict[str, int]]]
    load_preset_build_block: t.Callable[[dict[str, t.Any]], tuple[list[str] | None, int | None]]
    resolve_effective_min_channels: t.Callable[..., int]
    resolve_validation_channels: t.Callable[..., tuple[list[str], dict[str, int]]]
    load_finetune_config: t.Callable[[Path], t.Any]
    load_pretrain_config: t.Callable[[Path], t.Any]
    validate_model_config: t.Callable[[t.Any], int]


BASE_VARIANT = ConfigVariant("sleep2vec.config", "preprocess.save_dataset_presets")
CONFIG_VARIANTS = {
    "sleep2expert": ConfigVariant("sleep2expert.config", "sleep2expert.preprocess.save_dataset_presets"),
    "sleep2vec2": ConfigVariant("sleep2vec2.config", "sleep2vec2.preprocess.save_dataset_presets"),
}
# `CONFIG_VARIANTS` is keyed by the path segment that names a fork, so it has no entry for
# the base variant. Declaring a variant explicitly has to be able to name it.
DECLARABLE_VARIANTS = {"sleep2vec": BASE_VARIANT, **CONFIG_VARIANTS}
SLEEP2STAT_CONFIG_DIR = "sleep2stat"
SEX_AGE_BASELINE_CONFIG_DIR = "sex_age_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate repository YAML configs and local config contracts.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional config paths to validate. Defaults to every YAML under configs/.",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(DECLARABLE_VARIANTS),
        help=(
            "Validate against this variant's loader instead of inferring one from the path or "
            "contents. Pass the variant the run will actually use, so validation cannot pass "
            "under a loader the run never invokes."
        ),
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


def _load_config_mapping(path: Path) -> dict[str, t.Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return data


def _sleep2expert_only_finetune_fields() -> frozenset[str]:
    """Finetune keys that only `sleep2expert.config` accepts.

    Derived rather than listed: the content marker below exists to route a config the path
    heuristics missed, and a marker that restates the schema by hand falls out of sync with
    it. `finetune.moe_tuning` used to be the marker and used to contain
    `moe_regularization`; lifting that field to a sibling of `tuning` left a valid dense
    `sleep2expert` config with no marker at all, and the base loader now rejects the field
    outright instead of ignoring it.
    """
    expert = import_module(CONFIG_VARIANTS["sleep2expert"].config_module).FINETUNE_BLOCK_FIELDS
    base = import_module(BASE_VARIANT.config_module).FINETUNE_BLOCK_FIELDS
    return frozenset(expert) - frozenset(base)


def _resolve_config_variant(
    path: Path, config_data: dict[str, t.Any] | None = None, variant: str | None = None
) -> ConfigVariant:
    # A declared variant is the one the run will use, so it settles the question outright.
    # Everything below is inference for configs that came with no declaration, and inference
    # that disagrees with the command about to be run is worse than no inference: it validates
    # a config against a loader nothing will call.
    if variant is not None:
        try:
            return DECLARABLE_VARIANTS[variant]
        except KeyError:
            raise ValueError(f"Unknown variant {variant!r}. Expected one of {sorted(DECLARABLE_VARIANTS)}.") from None
    try:
        rel_path = path.resolve().relative_to(CONFIG_ROOT.resolve())
    except ValueError:
        rel_path = None
    if rel_path is not None and rel_path.parts:
        return CONFIG_VARIANTS.get(rel_path.parts[0], BASE_VARIANT)

    parts = path.resolve().parts
    for idx, part in enumerate(parts[:-1]):
        if part == "configs" and parts[idx + 1] in CONFIG_VARIANTS:
            return CONFIG_VARIANTS[parts[idx + 1]]
    for part in parts:
        if part in CONFIG_VARIANTS:
            return CONFIG_VARIANTS[part]
    for name, variant in CONFIG_VARIANTS.items():
        if path.name.startswith(f"{name}_") or path.name.startswith(f"{name}-"):
            return variant

    model_block = config_data.get("model") if config_data is not None else None
    backbone_block = model_block.get("backbone") if isinstance(model_block, dict) else None
    finetune_block = config_data.get("finetune") if config_data is not None else None
    has_moe_backbone = isinstance(backbone_block, dict) and "moe" in backbone_block
    tuning_block = finetune_block.get("tuning") if isinstance(finetune_block, dict) else None
    has_moe_tuning = isinstance(tuning_block, dict) and (
        "moe" in tuning_block or str(tuning_block.get("preset", "")).startswith("moe_")
    )
    has_expert_only_field = isinstance(finetune_block, dict) and bool(
        _sleep2expert_only_finetune_fields() & set(finetune_block)
    )
    if has_moe_backbone or has_moe_tuning or has_expert_only_field:
        return CONFIG_VARIANTS["sleep2expert"]
    return BASE_VARIANT


def _is_sleep2stat_config(path: Path, config_data: dict[str, t.Any]) -> bool:
    try:
        rel_path = path.resolve().relative_to(CONFIG_ROOT.resolve())
    except ValueError:
        rel_path = None
    if rel_path is not None and rel_path.parts and rel_path.parts[0] == SLEEP2STAT_CONFIG_DIR:
        return True
    return set(config_data) >= {"run", "data", "signals", "analyzers", "reducers", "outputs"}


def _is_sex_age_baseline_config(path: Path, config_data: dict[str, t.Any]) -> bool:
    try:
        rel_path = path.resolve().relative_to(CONFIG_ROOT.resolve())
    except ValueError:
        rel_path = None
    if rel_path is not None and rel_path.parts and rel_path.parts[0] == SEX_AGE_BASELINE_CONFIG_DIR:
        return True
    model_block = config_data.get("model") if isinstance(config_data.get("model"), dict) else {}
    return model_block.get("name") == "sex_age_mlp"


def _load_config_tools(path: Path, config_data: dict[str, t.Any], declared_variant: str | None = None) -> ConfigTools:
    variant = _resolve_config_variant(path, config_data, declared_variant)
    config_module = import_module(variant.config_module)
    preset_module = import_module(variant.preset_module)
    return ConfigTools(
        load_model_channels=preset_module._load_model_channels,
        load_preset_build_block=preset_module._load_preset_build_block,
        resolve_effective_min_channels=preset_module._resolve_effective_min_channels,
        resolve_validation_channels=preset_module._resolve_validation_channels,
        load_finetune_config=config_module.load_finetune_config,
        load_pretrain_config=config_module.load_pretrain_config,
        validate_model_config=config_module.validate_model_config,
    )


def _validate_runtime_loader_contract(path: Path, config_data: dict[str, t.Any], tools: ConfigTools) -> None:
    if isinstance(config_data.get("finetune"), dict):
        bundle = tools.load_finetune_config(path)
        tools.validate_model_config(bundle.model)
        return

    bundle = tools.load_pretrain_config(path)
    tools.validate_model_config(bundle.model)


def _validate_preset_build_contract(
    config_data: dict[str, t.Any],
    tools: ConfigTools,
) -> tuple[list[str] | None, int | None]:
    model_channels, channel_input_dims = tools.load_model_channels(config_data)
    preset_required_channels, preset_min_channels = tools.load_preset_build_block(config_data)
    if preset_required_channels is None and preset_min_channels is None:
        return None, None
    if preset_required_channels is None or preset_min_channels is None:
        raise ValueError("preset_build must define both required_channels and min_channels when provided.")

    validation_channels, _ = tools.resolve_validation_channels(
        model_channels=model_channels,
        channel_input_dims=channel_input_dims,
        preset_required_channels=preset_required_channels,
        selected_channels=None,
    )
    tools.resolve_effective_min_channels(
        channel_names=validation_channels,
        cli_min_channels=2,
        preset_min_channels=preset_min_channels,
    )
    return validation_channels, preset_min_channels


def _is_ppg_finetune_config(path: Path) -> bool:
    return path.name.startswith("ppg_") and "finetune" in path.name and path.suffix == ".yaml"


def _validate_repo_policy(path: Path, config_data: dict[str, t.Any], tools: ConfigTools) -> None:
    model_channels, _ = tools.load_model_channels(config_data)
    finetune_block = config_data.get("finetune")
    task_block = finetune_block.get("task") if isinstance(finetune_block, dict) else None
    preset_required_channels, preset_min_channels = tools.load_preset_build_block(config_data)

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
        if path.name.startswith("ppg_ahi_finetune"):
            if preset_required_channels != ["ppg", "ahi", "stage5"]:
                raise ValueError(
                    "single-channel ppg ahi configs must set preset_build.required_channels to [ppg, ahi, stage5]."
                )
            if preset_min_channels != 3:
                raise ValueError("single-channel ppg ahi configs must set preset_build.min_channels to 3.")
            return
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


def check_config_file(path: Path, variant: str | None = None) -> None:
    config_data = _load_config_mapping(path)
    if _is_sleep2stat_config(path, config_data):
        from sleep2stat.config import load_config

        load_config(path)
        return
    if _is_sex_age_baseline_config(path, config_data):
        from sex_age_baseline.config import load_finetune_config, validate_model_config

        bundle = load_finetune_config(path)
        validate_model_config(bundle.model)
        return
    tools = _load_config_tools(path, config_data, variant)
    _validate_runtime_loader_contract(path, config_data, tools)
    _validate_repo_policy(path, config_data, tools)
    _validate_preset_build_contract(config_data, tools)


def main() -> int:
    args = parse_args()
    config_paths = collect_config_paths(args.paths)
    failures: list[tuple[Path, str]] = []

    for path in config_paths:
        try:
            check_config_file(path, args.variant)
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
