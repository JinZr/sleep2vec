"""Root anchor: repository paths, variant/task constants, and summary type vocabulary.

Mixed bridge. Widely imported within the package, and also imported from outside
the package too, so it stays at the package top level. Its domain coupling is
two hardcoded constants -- ``SUPPORTED_VARIANTS`` (including
``sex_age_baseline``) and ``VARIANTLESS_TASKS`` -- plus the per-variant
``*ConfigSummary`` shapes.

``CONFIG_FINETUNE_SECTION`` spells the config-schema section name so kernel
modules can read it without tripping the adapter-leak guard, which matches raw
task-name constants: the section shares its spelling with the finetune task but
is config vocabulary, not task dispatch.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Final, TypedDict, TypeGuard, overload

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_VARIANTS = ("sleep2vec", "sleep2vec2", "sleep2expert", "sex_age_baseline")
VARIANTLESS_TASKS = {"sleep2stat"}
# Domain-config schema section name. Shares its spelling with the finetune
# task name but is config vocabulary, not task dispatch (the adapter leak
# guard matches raw task-name constants, so kernel modules read the section
# through this constant).
CONFIG_FINETUNE_SECTION: Final = "finetune"
_FULL_GIT_OBJECT_ID_RE = re.compile(r"[0-9a-f]{40}")


class _ConfigProvenance(TypedDict, total=False):
    authoritative_variant: str
    _source_config_bytes: bytes
    _source_config_sha256: str


class ConfigDiagnostics(_ConfigProvenance):
    config_path: str
    data_backend: Any
    warnings: list[str]
    blocking_issues: list[str]


class SidecarDiagnostics(TypedDict):
    key_column: Any
    disease_columns_index: Any
    has_label_index: Any
    output_dim: Any
    valid: bool
    disease_count: int | None
    sidecar_key_count: int | None
    issues: list[str]


class SurvivalSummary(SidecarDiagnostics):
    event_time_index: Any
    is_event_index: Any
    covariates: Any
    covariate_embedding_dim: Any


class MultilabelSummary(SidecarDiagnostics):
    label_index: Any


class ChannelSummary(TypedDict):
    name: Any
    input_dim: Any
    tokenizer: Any
    out_dim: Any


class TaskSummary(TypedDict):
    type: Any
    output_dim: Any
    is_seq: Any
    monitor: Any
    monitor_mod: Any


class _OptionalSidecarSummaries(TypedDict, total=False):
    survival: SurvivalSummary
    multilabel: MultilabelSummary


class TaskConfigSummary(_OptionalSidecarSummaries):
    task: TaskSummary
    loss: dict[str, Any]


class FinetuneTaskSummary(TaskConfigSummary):
    tuning: dict[str, Any]
    tuning_present: bool


class AggregationSummary(TypedDict):
    name: Any
    kwargs: dict[str, Any]


class HeadSummary(TypedDict):
    name: Any
    dropout: Any
    hidden_dim: Any
    kwargs: dict[str, Any]
    channel_agg: AggregationSummary
    temporal_agg: AggregationSummary


class AveragingSummary(TypedDict):
    present: bool
    name: Any
    enabled: Any


class ClsSummary(TypedDict):
    embedding_type: Any
    downstream: Any


class NamedComponentSummary(TypedDict):
    name: Any


class FinetuneModelSummary(TypedDict):
    backbone: Any
    hidden_size: Any
    backbone_depth: Any
    channels: list[ChannelSummary]
    cls: ClsSummary
    head: NamedComponentSummary
    head_details: HeadSummary
    layer_mix_present: bool
    layer_mix: dict[str, Any]
    model_averaging: AveragingSummary


class FinetuneDataSummary(TypedDict):
    max_tokens: Any
    data_channel_names: list[Any]
    finetune_data_index: Any
    finetune_preset_path: Any
    train_dataset_names: list[Any]
    test_dataset_names: list[Any]
    kaldi_data_root: Any
    kaldi_manifest: Any


class PresetBuildSummary(TypedDict):
    required_channels: Any
    min_channels: Any


class FinetuneConfigSummary(ConfigDiagnostics):
    variant_guess: str
    is_finetune: bool
    is_pretrain: bool
    model: FinetuneModelSummary
    data: FinetuneDataSummary
    finetune: FinetuneTaskSummary
    preset_build: PresetBuildSummary
    plausible_labels: list[str]


class EmptySummary(TypedDict):
    pass


class _SexAgeModelDetails(TypedDict, total=False):
    encodings: dict[str, dict[str, Any]]
    head_details: dict[str, Any]


class SexAgeModelSummary(_SexAgeModelDetails):
    name: str
    features: list[str]


class SexAgeDataSummary(TypedDict):
    backend: str
    finetune_data_index: str | None
    finetune_preset_path: str | None
    kaldi_data_root: str | None
    kaldi_manifest: str | None
    split_column: str
    key_column: str
    deduplicate_by_key: bool
    sample_unit: str


class SexAgeConfigSummary(ConfigDiagnostics):
    variant_guess: str
    is_finetune: bool
    is_pretrain: bool
    model: SexAgeModelSummary
    data: SexAgeDataSummary | EmptySummary
    finetune: TaskConfigSummary | EmptySummary
    preset_build: EmptySummary
    plausible_labels: list[str]


class AnalyzerSummary(TypedDict):
    name: str
    type: str
    enabled: bool
    namespace: str | None
    label_name: str | None
    config: str | None
    ckpt_path: str | None
    input_channels: list[str]
    stage_source: str | None
    event_source: str | None


class ReducerSummary(TypedDict):
    name: str
    type: str
    enabled: bool
    source: str | None
    left: str | None
    right: str | None
    age_prediction: str | None
    sex_prediction: str | None
    metadata_age_column: str
    metadata_sex_column: str
    options: dict[str, Any]


class Sleep2statRunSummary(TypedDict):
    name: str
    output_dir: str


class Sleep2statDataSummary(TypedDict):
    backend: str
    index: str | None
    kaldi_data_root: str | None
    kaldi_manifest: str | None
    split: list[str]
    metadata_columns: list[str]
    token_sec: int
    max_tokens: int


class Sleep2statOutputSummary(TypedDict):
    write_global_tables: bool
    write_per_record: bool
    compression: str
    global_tables: dict[str, bool]


class SupportedAnalysisTypes(TypedDict):
    supported_analyzer_types: list[str]
    supported_reducer_types: list[str]


class Sleep2statSummary(SupportedAnalysisTypes, total=False):
    # A failed config load reports supported types without resolved configuration.
    run: Sleep2statRunSummary
    data: Sleep2statDataSummary
    analyzers: list[AnalyzerSummary]
    reducers: list[ReducerSummary]
    outputs: Sleep2statOutputSummary


class Sleep2statConfigSummary(ConfigDiagnostics):
    is_sleep2stat: bool
    sleep2stat: Sleep2statSummary
    agent_risk_issues: list[str]


ConfigSummary = FinetuneConfigSummary | SexAgeConfigSummary | Sleep2statConfigSummary
ConfigSummaryInput = ConfigSummary | dict[str, Any]


def is_full_git_object_id(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and _FULL_GIT_OBJECT_ID_RE.fullmatch(value) is not None


def recipe_name(recipe: dict[str, Any]) -> str:
    return str(recipe.get("name") or Path(str(recipe.get("_recipe_path", "recipe"))).stem)


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    if resolved is None:
        raise FileNotFoundError("Config path is required.")
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {resolved}")
    return data


def task_requires_variant(task: str | None) -> bool:
    return task not in VARIANTLESS_TASKS


def module_for_variant(variant: str, entrypoint: str) -> str:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")
    return f"{variant}.{entrypoint}"


def coerce_list(value: Any) -> list[Any]:
    if value in (None, "", "ASK_USER"):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


@overload
def repo_relative(path: Path) -> str: ...


@overload
def repo_relative(path: str | Path | None) -> str | None: ...


def repo_relative(path: str | Path | None) -> str | None:
    if path is None or path == "":
        return None
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(REPO_ROOT.resolve()))
    except (OSError, ValueError):
        return str(raw)


def resolve_repo_path(path: str | Path | None, *, relative_to: str | Path | None = None) -> Path | None:
    if path is None or path == "":
        return None
    candidate = Path(path)
    if relative_to is None:
        candidate = candidate.expanduser()
    if not candidate.is_absolute():
        candidate = Path(relative_to) / candidate if relative_to is not None else REPO_ROOT / candidate
    return candidate
