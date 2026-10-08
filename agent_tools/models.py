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

import math
from pathlib import Path
import re
from typing import Any, Final, TypeAlias, TypedDict, TypeGuard, overload

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_VARIANTS = ("sleep2vec", "sleep2vec2", "sleep2expert", "sex_age_baseline")
VARIANTLESS_TASKS = {"sleep2stat"}
# Domain-config schema section name. Shares its spelling with the finetune
# task name but is config vocabulary, not task dispatch (the adapter leak
# guard matches raw task-name constants, so kernel modules read the section
# through this constant).
CONFIG_FINETUNE_SECTION: Final = "finetune"
# libyaml parses the frozen recipes and configs about ten times faster than PyYAML's pure-Python
# scanner and composes the same node tree. The pinned pyyaml wheels bundle it; an environment
# without it fails here at import instead of silently falling back to the slow loader.
YAML_SAFE_LOADER = yaml.CSafeLoader
_FULL_GIT_OBJECT_ID_RE = re.compile(r"[0-9a-f]{40}")
#: A value of the JSON data model, as ``json.loads`` returns it and JSON-backed records hold it.
JsonValue: TypeAlias = "str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]"


def validate_json_value(value: object, location: str, *, finite: bool = True) -> None:
    """Reject non-string object keys and any value outside str, int, float, bool, None, list and dict, converting
    nothing. With ``finite=True`` also reject NaN and infinities, which JSON interchange cannot carry."""
    if value is None or isinstance(value, (str, bool)) or type(value) is int:
        return
    if isinstance(value, float):
        if finite and not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite number.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_value(item, f"{location}[{index}]", finite=finite)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} contains a non-string object key.")
            validate_json_value(item, f"{location}.{key}", finite=finite)
        return
    raise ValueError(f"{location} contains a non-JSON value of type {type(value).__name__}.")


class BoundFinalEvalConfigSnapshot(TypedDict):
    source_path: str
    bytes: bytes
    sha256: str


class _ConfigProvenance(TypedDict, total=False):
    authoritative_variant: str
    _source_config_bytes: bytes
    _source_config_sha256: str
    # Bound only by hparam plan preflight when an explicit final-eval config is frozen.
    _final_eval_config_snapshot: BoundFinalEvalConfigSnapshot


class ConfigDiagnostics(_ConfigProvenance):
    config_path: str
    data_backend: JsonValue
    warnings: list[str]
    blocking_issues: list[str]


class SidecarDiagnostics(TypedDict):
    key_column: JsonValue
    disease_columns_index: JsonValue
    has_label_index: JsonValue
    covariates: JsonValue
    covariate_embedding_dim: JsonValue
    output_dim: JsonValue
    valid: bool
    disease_count: int | None
    sidecar_key_count: int | None
    issues: list[str]


class SurvivalSummary(SidecarDiagnostics):
    event_time_index: JsonValue
    is_event_index: JsonValue
    #: The subset of ``issues`` the config loader rejects; a preset-backed run opens no survival sidecar file.
    preset_issues: list[str]


class MultilabelSummary(SidecarDiagnostics):
    label_index: JsonValue
    #: The subset of ``issues`` that still blocks a preset-backed run: disease_columns_index, the only sidecar it
    #: reads, and fields that are not non-empty strings, which the variant config loader rejects.
    preset_issues: list[str]


class ChannelSummary(TypedDict):
    name: JsonValue
    input_dim: JsonValue
    tokenizer: JsonValue
    out_dim: JsonValue


class TaskSummary(TypedDict):
    type: JsonValue
    output_dim: JsonValue
    is_seq: JsonValue
    monitor: JsonValue
    monitor_mod: JsonValue


class _OptionalSidecarSummaries(TypedDict, total=False):
    survival: SurvivalSummary
    multilabel: MultilabelSummary


class TaskConfigSummary(_OptionalSidecarSummaries):
    task: TaskSummary
    loss: dict[str, JsonValue]


class FinetuneTaskSummary(TaskConfigSummary):
    tuning: dict[str, JsonValue]
    tuning_present: bool


class AggregationSummary(TypedDict):
    name: JsonValue
    kwargs: dict[str, JsonValue]


class HeadSummary(TypedDict):
    name: JsonValue
    dropout: JsonValue
    hidden_dim: JsonValue
    kwargs: dict[str, JsonValue]
    channel_agg: AggregationSummary
    temporal_agg: AggregationSummary


class AveragingSummary(TypedDict):
    present: bool
    name: JsonValue
    enabled: JsonValue


class ClsSummary(TypedDict):
    embedding_type: JsonValue
    downstream: JsonValue


class NamedComponentSummary(TypedDict):
    name: JsonValue


class FinetuneModelSummary(TypedDict):
    backbone: JsonValue
    hidden_size: JsonValue
    backbone_depth: JsonValue
    channels: list[ChannelSummary]
    cls: ClsSummary
    head: NamedComponentSummary
    head_details: HeadSummary
    layer_mix_present: bool
    layer_mix: dict[str, JsonValue]
    model_averaging: AveragingSummary


class FinetuneDataSummary(TypedDict):
    max_tokens: JsonValue
    data_channel_names: list[JsonValue]
    finetune_data_index: JsonValue
    finetune_preset_path: JsonValue
    train_dataset_names: list[JsonValue]
    test_dataset_names: list[JsonValue]
    kaldi_data_root: JsonValue
    kaldi_manifest: JsonValue


class PresetBuildSummary(TypedDict):
    required_channels: JsonValue
    min_channels: JsonValue


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


class SexAgeHeadDetails(TypedDict):
    name: str
    hidden_dim: int
    dropout: float
    act: str
    kwargs: dict[str, int]


class _SexAgeModelDetails(TypedDict, total=False):
    head_details: SexAgeHeadDetails


class SexAgeModelSummary(_SexAgeModelDetails):
    name: str


class SexAgeDataSummary(TypedDict):
    backend: str
    finetune_data_index: str | None
    finetune_preset_path: str | None
    kaldi_data_root: str | None
    kaldi_manifest: str | None


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
    options: dict[str, JsonValue]


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
    data = yaml.load(resolved.read_text(), Loader=YAML_SAFE_LOADER)
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
