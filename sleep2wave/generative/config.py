from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import typing as t

import yaml

from sleep2wave.data.modalities import CANONICAL_MODALITIES, EPOCH_SEC, MODALITY_SPECS, validate_modality_sequence

SUPPORTED_STAGES = {"autoencoder", "diffusion", "evaluation", "inference"}
SUPPORTED_EVALUATION_METRIC_FAMILIES = {"waveform", "feature", "event", "efficiency", "downstream"}
SUPPORTED_INITIALIZATION_GROUPS = {
    "tokenizers",
    "backbone",
    "projection",
    "autoencoder_encoders",
    "diffusion_transformer",
}


@dataclass(frozen=True)
class DataConfig:
    preset_path: str | None = None
    index: str | None = None
    context_epochs: int = 15


@dataclass(frozen=True)
class ModalitiesConfig:
    epoch_sec: int
    all: list[str]
    high_frequency: list[str]
    low_frequency: list[str]
    sample_rates: dict[str, int]
    frames_per_epoch: dict[str, int]


@dataclass(frozen=True)
class AutoencoderLossConfig:
    waveform_l1_weight: float = 1.0
    waveform_l2_weight: float = 0.0
    spectral_weight: float = 0.0


@dataclass(frozen=True)
class AutoencoderConfig:
    latent_dim: int
    encoder_type: str
    decoder_type: str
    one_latent_per_epoch: bool
    modality_specific: bool
    losses: AutoencoderLossConfig


@dataclass(frozen=True)
class TransformerConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_ratio: int


@dataclass(frozen=True)
class EmbeddingsConfig:
    diffusion_step: bool
    modality: bool
    epoch_position: bool
    sleep_night_position: bool
    availability: bool
    quality: bool


@dataclass(frozen=True)
class DiffusionConfig:
    latent_dim: int
    transformer: TransformerConfig
    diffusion_steps: int
    beta_schedule: str
    prediction_type: str
    context_epochs: int
    embeddings: EmbeddingsConfig
    task_attention_mask: str
    auxiliary_restoration_token: bool
    condition_dropout: float
    autoencoder_checkpoint: str | None = None
    latent_cache_path: str | None = None


@dataclass(frozen=True)
class ReplayConfig:
    enabled: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    phase: int
    batch_size: int
    lr: float
    weight_decay: float
    max_epochs: int
    gradient_clip_val: float
    task_mix: dict[str, float] = field(default_factory=dict)
    condition_counts: list[int] = field(default_factory=list)
    replay: ReplayConfig = field(default_factory=ReplayConfig)


@dataclass(frozen=True)
class SamplerConfig:
    name: str
    steps: int
    eta: float
    num_samples: int


@dataclass(frozen=True)
class InitializationConfig:
    sleep2vec2_checkpoint: str | None = None
    strict_compatible: bool = True
    require_any_loaded: bool = False
    load_groups: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportConfig:
    output_dir: str


@dataclass(frozen=True)
class EvaluationConfig:
    generated_dir: str
    reference_npz: str | None = None
    baseline_npz: str | None = None
    events_json: str | None = None
    downstream_metrics_json: str | None = None
    metric_families: list[str] = field(default_factory=list)
    max_shift_frames: int = 0
    event_iou_threshold: float = 0.5


@dataclass(frozen=True)
class Sleep2WaveConfig:
    recipe: str
    stage: str
    data: DataConfig | None
    modalities: ModalitiesConfig
    autoencoder: AutoencoderConfig | None = None
    diffusion: DiffusionConfig | None = None
    training: TrainingConfig | None = None
    sampler: SamplerConfig | None = None
    initialization: InitializationConfig | None = None
    export: ExportConfig | None = None
    evaluation: EvaluationConfig | None = None


def _load_yaml_mapping(path: str | Path) -> dict[str, t.Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping.")
    return data


def _require_mapping(raw: t.Any, path: str) -> dict[str, t.Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a mapping.")
    return raw


def _reject_extra(raw: dict[str, t.Any], allowed: set[str], path: str) -> None:
    extra = sorted(set(raw) - allowed)
    if extra:
        raise ValueError(f"{path} has unsupported fields: {extra}")


def _require_string(raw: dict[str, t.Any], key: str, path: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}.{key} must be a non-empty string.")
    return value


def _optional_string(raw: dict[str, t.Any], key: str, path: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}.{key} must be a non-empty string when provided.")
    return value


def _require_bool(raw: dict[str, t.Any], key: str, path: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be a boolean.")
    return value


def _require_int(raw: dict[str, t.Any], key: str, path: str, *, minimum: int) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{path}.{key} must be an integer >= {minimum}.")
    return value


def _require_float(
    raw: dict[str, t.Any],
    key: str,
    path: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be a number.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{path}.{key} must be finite.")
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            raise ValueError(f"{path}.{key} must be >= {minimum}.")
        raise ValueError(f"{path}.{key} must be in [{minimum}, {maximum}].")
    return value


def _load_data(raw: t.Any) -> DataConfig:
    block = _require_mapping(raw, "data")
    _reject_extra(block, {"preset_path", "index", "context_epochs"}, "data")
    preset_path = _optional_string(block, "preset_path", "data")
    index = _optional_string(block, "index", "data")
    if (preset_path is None) == (index is None):
        raise ValueError("data must define exactly one of preset_path or index.")
    context_epochs = block.get("context_epochs", 15)
    if not isinstance(context_epochs, int) or isinstance(context_epochs, bool) or context_epochs < 1:
        raise ValueError("data.context_epochs must be an integer >= 1.")
    return DataConfig(preset_path=preset_path, index=index, context_epochs=context_epochs)


def _load_modalities(raw: t.Any) -> ModalitiesConfig:
    block = _require_mapping(raw, "modalities")
    allowed = {"epoch_sec", "all", "high_frequency", "low_frequency", "sample_rates", "frames_per_epoch"}
    _reject_extra(block, allowed, "modalities")

    epoch_sec = _require_int(block, "epoch_sec", "modalities", minimum=1)
    if epoch_sec != EPOCH_SEC:
        raise ValueError(f"modalities.epoch_sec must be {EPOCH_SEC}.")

    all_modalities = validate_modality_sequence(block.get("all"), allow_aliases=False)
    if all_modalities != list(CANONICAL_MODALITIES):
        raise ValueError(f"modalities.all must be {list(CANONICAL_MODALITIES)}.")

    high_frequency = validate_modality_sequence(block.get("high_frequency"), allow_aliases=False)
    low_frequency = validate_modality_sequence(block.get("low_frequency"), allow_aliases=False)
    expected_high = [name for name in CANONICAL_MODALITIES if MODALITY_SPECS[name].frequency_group == "high_frequency"]
    expected_low = [name for name in CANONICAL_MODALITIES if MODALITY_SPECS[name].frequency_group == "low_frequency"]
    if high_frequency != expected_high:
        raise ValueError(f"modalities.high_frequency must be {expected_high}.")
    if low_frequency != expected_low:
        raise ValueError(f"modalities.low_frequency must be {expected_low}.")

    sample_rates = _require_mapping(block.get("sample_rates"), "modalities.sample_rates")
    frames_per_epoch = _require_mapping(block.get("frames_per_epoch"), "modalities.frames_per_epoch")
    _reject_extra(sample_rates, set(CANONICAL_MODALITIES), "modalities.sample_rates")
    _reject_extra(frames_per_epoch, set(CANONICAL_MODALITIES), "modalities.frames_per_epoch")

    parsed_sample_rates: dict[str, int] = {}
    parsed_frames_per_epoch: dict[str, int] = {}
    for name in CANONICAL_MODALITIES:
        if name not in sample_rates:
            raise ValueError(f"modalities.sample_rates missing {name}.")
        if name not in frames_per_epoch:
            raise ValueError(f"modalities.frames_per_epoch missing {name}.")
        expected = MODALITY_SPECS[name]
        sample_rate = sample_rates[name]
        frames = frames_per_epoch[name]
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
            raise ValueError(f"modalities.sample_rates.{name} must be an integer.")
        if not isinstance(frames, int) or isinstance(frames, bool):
            raise ValueError(f"modalities.frames_per_epoch.{name} must be an integer.")
        if sample_rate != expected.sample_rate_hz:
            raise ValueError(f"modalities.sample_rates.{name} must be {expected.sample_rate_hz}.")
        if frames != expected.frames_per_epoch:
            raise ValueError(f"modalities.frames_per_epoch.{name} must be {expected.frames_per_epoch}.")
        parsed_sample_rates[name] = sample_rate
        parsed_frames_per_epoch[name] = frames

    return ModalitiesConfig(
        epoch_sec=epoch_sec,
        all=all_modalities,
        high_frequency=high_frequency,
        low_frequency=low_frequency,
        sample_rates=parsed_sample_rates,
        frames_per_epoch=parsed_frames_per_epoch,
    )


def _load_autoencoder_losses(raw: t.Any) -> AutoencoderLossConfig:
    block = _require_mapping(raw, "autoencoder.losses")
    allowed = {"waveform_l1_weight", "waveform_l2_weight", "spectral_weight"}
    _reject_extra(block, allowed, "autoencoder.losses")
    return AutoencoderLossConfig(
        waveform_l1_weight=_require_float(block, "waveform_l1_weight", "autoencoder.losses", minimum=0.0),
        waveform_l2_weight=_require_float(block, "waveform_l2_weight", "autoencoder.losses", minimum=0.0),
        spectral_weight=_require_float(block, "spectral_weight", "autoencoder.losses", minimum=0.0),
    )


def _load_autoencoder(raw: t.Any) -> AutoencoderConfig:
    block = _require_mapping(raw, "autoencoder")
    allowed = {"latent_dim", "encoder_type", "decoder_type", "one_latent_per_epoch", "modality_specific", "losses"}
    _reject_extra(block, allowed, "autoencoder")
    encoder_type = _require_string(block, "encoder_type", "autoencoder")
    decoder_type = _require_string(block, "decoder_type", "autoencoder")
    one_latent_per_epoch = _require_bool(block, "one_latent_per_epoch", "autoencoder")
    modality_specific = _require_bool(block, "modality_specific", "autoencoder")
    if encoder_type != "conv1d_epoch":
        raise ValueError("autoencoder.encoder_type must be 'conv1d_epoch'.")
    if decoder_type != "convtranspose1d_epoch":
        raise ValueError("autoencoder.decoder_type must be 'convtranspose1d_epoch'.")
    if not one_latent_per_epoch:
        raise ValueError("autoencoder.one_latent_per_epoch must be true.")
    if not modality_specific:
        raise ValueError("autoencoder.modality_specific must be true.")
    return AutoencoderConfig(
        latent_dim=_require_int(block, "latent_dim", "autoencoder", minimum=1),
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        one_latent_per_epoch=one_latent_per_epoch,
        modality_specific=modality_specific,
        losses=_load_autoencoder_losses(block.get("losses")),
    )


def _load_transformer(raw: t.Any) -> TransformerConfig:
    block = _require_mapping(raw, "diffusion.transformer")
    allowed = {"hidden_size", "num_layers", "num_heads", "mlp_ratio"}
    _reject_extra(block, allowed, "diffusion.transformer")
    hidden_size = _require_int(block, "hidden_size", "diffusion.transformer", minimum=1)
    num_heads = _require_int(block, "num_heads", "diffusion.transformer", minimum=1)
    if hidden_size % num_heads != 0:
        raise ValueError("diffusion.transformer.hidden_size must be divisible by num_heads.")
    return TransformerConfig(
        hidden_size=hidden_size,
        num_layers=_require_int(block, "num_layers", "diffusion.transformer", minimum=1),
        num_heads=num_heads,
        mlp_ratio=_require_int(block, "mlp_ratio", "diffusion.transformer", minimum=1),
    )


def _load_embeddings(raw: t.Any) -> EmbeddingsConfig:
    block = _require_mapping(raw, "diffusion.embeddings")
    allowed = {"diffusion_step", "modality", "epoch_position", "sleep_night_position", "availability", "quality"}
    _reject_extra(block, allowed, "diffusion.embeddings")
    return EmbeddingsConfig(
        diffusion_step=_require_bool(block, "diffusion_step", "diffusion.embeddings"),
        modality=_require_bool(block, "modality", "diffusion.embeddings"),
        epoch_position=_require_bool(block, "epoch_position", "diffusion.embeddings"),
        sleep_night_position=_require_bool(block, "sleep_night_position", "diffusion.embeddings"),
        availability=_require_bool(block, "availability", "diffusion.embeddings"),
        quality=_require_bool(block, "quality", "diffusion.embeddings"),
    )


def _load_diffusion(raw: t.Any) -> DiffusionConfig:
    block = _require_mapping(raw, "diffusion")
    allowed = {
        "latent_dim",
        "autoencoder_checkpoint",
        "latent_cache_path",
        "transformer",
        "diffusion_steps",
        "beta_schedule",
        "prediction_type",
        "context_epochs",
        "embeddings",
        "task_attention_mask",
        "auxiliary_restoration_token",
        "condition_dropout",
    }
    _reject_extra(block, allowed, "diffusion")

    autoencoder_checkpoint = _optional_string(block, "autoencoder_checkpoint", "diffusion")
    latent_cache_path = _optional_string(block, "latent_cache_path", "diffusion")
    if autoencoder_checkpoint is None:
        raise ValueError("diffusion.autoencoder_checkpoint is required until latent cache training is implemented.")

    beta_schedule = _require_string(block, "beta_schedule", "diffusion")
    if beta_schedule != "cosine":
        raise ValueError("diffusion.beta_schedule must be 'cosine'.")
    prediction_type = _require_string(block, "prediction_type", "diffusion")
    if prediction_type != "epsilon":
        raise ValueError("diffusion.prediction_type must be 'epsilon'.")
    task_attention_mask = _require_string(block, "task_attention_mask", "diffusion")
    if task_attention_mask != "directional":
        raise ValueError("diffusion.task_attention_mask must be 'directional'.")

    return DiffusionConfig(
        latent_dim=_require_int(block, "latent_dim", "diffusion", minimum=1),
        autoencoder_checkpoint=autoencoder_checkpoint,
        latent_cache_path=latent_cache_path,
        transformer=_load_transformer(block.get("transformer")),
        diffusion_steps=_require_int(block, "diffusion_steps", "diffusion", minimum=1),
        beta_schedule=beta_schedule,
        prediction_type=prediction_type,
        context_epochs=_require_int(block, "context_epochs", "diffusion", minimum=1),
        embeddings=_load_embeddings(block.get("embeddings")),
        task_attention_mask=task_attention_mask,
        auxiliary_restoration_token=_require_bool(block, "auxiliary_restoration_token", "diffusion"),
        condition_dropout=_require_float(block, "condition_dropout", "diffusion", minimum=0.0, maximum=1.0),
    )


def _load_task_mix(raw: t.Any) -> dict[str, float]:
    if raw is None:
        return {}
    block = _require_mapping(raw, "training.task_mix")
    allowed = {"restoration", "imputation", "translation", "two_condition", "partial_full"}
    _reject_extra(block, allowed, "training.task_mix")
    parsed: dict[str, float] = {}
    for name, value in block.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"training.task_mix.{name} must be a non-negative number.")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"training.task_mix.{name} must be a non-negative finite number.")
        parsed[name] = value
    if sum(parsed.values()) <= 0:
        raise ValueError("training.task_mix must include at least one positive task weight.")
    return parsed


def _load_condition_counts(raw: t.Any) -> list[int]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise ValueError("training.condition_counts must be a non-empty list when provided.")
    counts: list[int] = []
    for value in raw:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("training.condition_counts must contain positive integers.")
        counts.append(value)
    return counts


def _load_replay(raw: t.Any) -> ReplayConfig:
    if raw is None:
        return ReplayConfig()
    block = _require_mapping(raw, "training.replay")
    _reject_extra(block, {"enabled"}, "training.replay")
    return ReplayConfig(enabled=_require_bool(block, "enabled", "training.replay"))


def _load_training(raw: t.Any) -> TrainingConfig:
    block = _require_mapping(raw, "training")
    allowed = {
        "phase",
        "batch_size",
        "lr",
        "weight_decay",
        "max_epochs",
        "gradient_clip_val",
        "task_mix",
        "condition_counts",
        "replay",
    }
    _reject_extra(block, allowed, "training")
    phase = _require_int(block, "phase", "training", minimum=0)
    if phase > 5:
        raise ValueError("training.phase must be between 0 and 5.")
    return TrainingConfig(
        phase=phase,
        batch_size=_require_int(block, "batch_size", "training", minimum=1),
        lr=_require_float(block, "lr", "training", minimum=0.0),
        weight_decay=_require_float(block, "weight_decay", "training", minimum=0.0),
        max_epochs=_require_int(block, "max_epochs", "training", minimum=1),
        gradient_clip_val=_require_float(block, "gradient_clip_val", "training", minimum=0.0),
        task_mix=_load_task_mix(block.get("task_mix")),
        condition_counts=_load_condition_counts(block.get("condition_counts")),
        replay=_load_replay(block.get("replay")),
    )


def _load_sampler(raw: t.Any, diffusion_cfg: DiffusionConfig) -> SamplerConfig:
    block = _require_mapping(raw, "sampler")
    allowed = {"name", "steps", "eta", "num_samples"}
    _reject_extra(block, allowed, "sampler")
    name = _require_string(block, "name", "sampler")
    if name not in {"ddim", "ddpm"}:
        raise ValueError("sampler.name must be 'ddim' or 'ddpm'.")
    steps = _require_int(block, "steps", "sampler", minimum=1)
    if steps > diffusion_cfg.diffusion_steps:
        raise ValueError("sampler.steps must be <= diffusion.diffusion_steps.")
    if name == "ddpm" and steps != diffusion_cfg.diffusion_steps:
        raise ValueError("sampler.steps must equal diffusion.diffusion_steps for DDPM sampling.")
    return SamplerConfig(
        name=name,
        steps=steps,
        eta=_require_float(block, "eta", "sampler", minimum=0.0),
        num_samples=_require_int(block, "num_samples", "sampler", minimum=1),
    )


def _load_initialization(raw: t.Any) -> InitializationConfig | None:
    if raw is None:
        return None
    block = _require_mapping(raw, "initialization")
    allowed = {"sleep2vec2_checkpoint", "strict_compatible", "require_any_loaded", "load_groups"}
    _reject_extra(block, allowed, "initialization")
    load_groups_raw = block.get("load_groups", {})
    load_groups = _require_mapping(load_groups_raw, "initialization.load_groups")
    parsed_groups: dict[str, bool] = {}
    for key, value in load_groups.items():
        if not isinstance(key, str) or not key:
            raise ValueError("initialization.load_groups keys must be non-empty strings.")
        if key not in SUPPORTED_INITIALIZATION_GROUPS:
            raise ValueError(
                "initialization.load_groups keys must be one of "
                f"{sorted(SUPPORTED_INITIALIZATION_GROUPS)}. Got: {key}"
            )
        if not isinstance(value, bool):
            raise ValueError(f"initialization.load_groups.{key} must be a boolean.")
        parsed_groups[key] = value
    return InitializationConfig(
        sleep2vec2_checkpoint=_optional_string(block, "sleep2vec2_checkpoint", "initialization"),
        strict_compatible=block.get("strict_compatible", True),
        require_any_loaded=block.get("require_any_loaded", False),
        load_groups=parsed_groups,
    )


def _validate_initialization(initialization: InitializationConfig | None) -> None:
    if initialization is None:
        return
    if not isinstance(initialization.strict_compatible, bool):
        raise ValueError("initialization.strict_compatible must be a boolean.")
    if not isinstance(initialization.require_any_loaded, bool):
        raise ValueError("initialization.require_any_loaded must be a boolean.")
    if (any(initialization.load_groups.values()) or initialization.require_any_loaded) and (
        initialization.sleep2vec2_checkpoint is None
    ):
        raise ValueError(
            "initialization.sleep2vec2_checkpoint is required when initialization requests checkpoint loading."
        )


def _load_export(raw: t.Any) -> ExportConfig:
    block = _require_mapping(raw, "export")
    _reject_extra(block, {"output_dir"}, "export")
    return ExportConfig(output_dir=_require_string(block, "output_dir", "export"))


def _load_metric_families(raw: t.Any) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("evaluation.metric_families must be a non-empty list.")
    parsed: list[str] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, str) or not value:
            raise ValueError("evaluation.metric_families must contain non-empty strings.")
        if value not in SUPPORTED_EVALUATION_METRIC_FAMILIES:
            raise ValueError(
                "evaluation.metric_families entries must be one of "
                f"{sorted(SUPPORTED_EVALUATION_METRIC_FAMILIES)}. Got: {value}"
            )
        if value in seen:
            raise ValueError(f"Duplicate evaluation metric family: {value}")
        seen.add(value)
        parsed.append(value)
    return parsed


def _load_evaluation(raw: t.Any) -> EvaluationConfig:
    block = _require_mapping(raw, "evaluation")
    allowed = {
        "generated_dir",
        "reference_npz",
        "baseline_npz",
        "events_json",
        "downstream_metrics_json",
        "metric_families",
        "max_shift_frames",
        "event_iou_threshold",
    }
    _reject_extra(block, allowed, "evaluation")
    return EvaluationConfig(
        generated_dir=_require_string(block, "generated_dir", "evaluation"),
        reference_npz=_optional_string(block, "reference_npz", "evaluation"),
        baseline_npz=_optional_string(block, "baseline_npz", "evaluation"),
        events_json=_optional_string(block, "events_json", "evaluation"),
        downstream_metrics_json=_optional_string(block, "downstream_metrics_json", "evaluation"),
        metric_families=_load_metric_families(block.get("metric_families")),
        max_shift_frames=_require_int(block, "max_shift_frames", "evaluation", minimum=0),
        event_iou_threshold=_require_float(
            block,
            "event_iou_threshold",
            "evaluation",
            minimum=0.0,
            maximum=1.0,
        ),
    )


def load_sleep2wave_config(path: str | Path) -> Sleep2WaveConfig:
    data = _load_yaml_mapping(path)
    allowed = {
        "recipe",
        "stage",
        "data",
        "modalities",
        "autoencoder",
        "diffusion",
        "training",
        "sampler",
        "initialization",
        "export",
        "evaluation",
    }
    _reject_extra(data, allowed, "top-level sleep2wave config")

    recipe = _require_string(data, "recipe", "top-level sleep2wave config")
    if recipe != "sleep2wave":
        raise ValueError("recipe must be 'sleep2wave'.")

    stage = _require_string(data, "stage", "top-level sleep2wave config")
    if stage not in SUPPORTED_STAGES:
        raise ValueError(f"stage must be one of {sorted(SUPPORTED_STAGES)}.")

    if stage != "evaluation" and "data" not in data:
        raise ValueError("data block is required.")
    if "modalities" not in data:
        raise ValueError("modalities block is required.")
    if "export" not in data:
        raise ValueError("export block is required.")

    data_cfg = _load_data(data["data"]) if "data" in data else None
    modalities_cfg = _load_modalities(data["modalities"])
    training_cfg = _load_training(data["training"]) if "training" in data else None
    export_cfg = _load_export(data["export"])
    initialization_cfg = _load_initialization(data.get("initialization"))
    _validate_initialization(initialization_cfg)

    autoencoder_cfg: AutoencoderConfig | None = None
    diffusion_cfg: DiffusionConfig | None = None
    sampler_cfg: SamplerConfig | None = None
    evaluation_cfg: EvaluationConfig | None = None

    if stage == "autoencoder":
        if training_cfg is None:
            raise ValueError("training block is required for stage=autoencoder.")
        if training_cfg.phase != 0:
            raise ValueError("stage=autoencoder requires training.phase=0.")
        if "autoencoder" not in data:
            raise ValueError("autoencoder block is required for stage=autoencoder.")
        if "diffusion" in data:
            raise ValueError("stage=autoencoder does not support a diffusion block.")
        if "sampler" in data:
            raise ValueError("stage=autoencoder does not support a sampler block.")
        if "evaluation" in data:
            raise ValueError("stage=autoencoder does not support an evaluation block.")
        autoencoder_cfg = _load_autoencoder(data["autoencoder"])
    elif stage == "diffusion":
        if training_cfg is None:
            raise ValueError("training block is required for stage=diffusion.")
        if training_cfg.phase == 0:
            raise ValueError("stage=diffusion requires training.phase between 1 and 5.")
        if "diffusion" not in data:
            raise ValueError("diffusion block is required for stage=diffusion.")
        if "autoencoder" in data:
            raise ValueError("stage=diffusion does not support an autoencoder block.")
        if "sampler" not in data:
            raise ValueError("sampler block is required for stage=diffusion.")
        if "evaluation" in data:
            raise ValueError("stage=diffusion does not support an evaluation block.")
        diffusion_cfg = _load_diffusion(data["diffusion"])
        if data_cfg is None or data_cfg.context_epochs != diffusion_cfg.context_epochs:
            raise ValueError("data.context_epochs must match diffusion.context_epochs.")
        sampler_cfg = _load_sampler(data["sampler"], diffusion_cfg)
    elif stage == "inference":
        if "training" in data:
            raise ValueError("stage=inference does not support a training block.")
        if "autoencoder" in data:
            raise ValueError("stage=inference does not support an autoencoder block.")
        if "evaluation" in data:
            raise ValueError("stage=inference does not support an evaluation block.")
        if "diffusion" not in data:
            raise ValueError("diffusion block is required for stage=inference.")
        if "sampler" not in data:
            raise ValueError("sampler block is required for stage=inference.")
        diffusion_cfg = _load_diffusion(data["diffusion"])
        if data_cfg is None or data_cfg.context_epochs != diffusion_cfg.context_epochs:
            raise ValueError("data.context_epochs must match diffusion.context_epochs.")
        sampler_cfg = _load_sampler(data["sampler"], diffusion_cfg)
    else:
        if "data" in data:
            raise ValueError("stage=evaluation does not support a data block.")
        if "training" in data:
            raise ValueError("stage=evaluation does not support a training block.")
        if "autoencoder" in data:
            raise ValueError("stage=evaluation does not support an autoencoder block.")
        if "diffusion" in data:
            raise ValueError("stage=evaluation does not support a diffusion block.")
        if "sampler" in data:
            raise ValueError("stage=evaluation does not support a sampler block.")
        if "initialization" in data:
            raise ValueError("stage=evaluation does not support an initialization block.")
        if "evaluation" not in data:
            raise ValueError("evaluation block is required for stage=evaluation.")
        evaluation_cfg = _load_evaluation(data["evaluation"])

    return Sleep2WaveConfig(
        recipe=recipe,
        stage=stage,
        data=data_cfg,
        modalities=modalities_cfg,
        autoencoder=autoencoder_cfg,
        diffusion=diffusion_cfg,
        training=training_cfg,
        sampler=sampler_cfg,
        initialization=initialization_cfg,
        export=export_cfg,
        evaluation=evaluation_cfg,
    )


__all__ = [
    "AutoencoderConfig",
    "AutoencoderLossConfig",
    "DataConfig",
    "DiffusionConfig",
    "EvaluationConfig",
    "ExportConfig",
    "InitializationConfig",
    "ModalitiesConfig",
    "SamplerConfig",
    "Sleep2WaveConfig",
    "SUPPORTED_EVALUATION_METRIC_FAMILIES",
    "SUPPORTED_INITIALIZATION_GROUPS",
    "TrainingConfig",
    "load_sleep2wave_config",
]
