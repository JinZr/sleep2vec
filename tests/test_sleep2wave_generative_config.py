from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from sleep2wave.data.modalities import CANONICAL_MODALITIES
from sleep2wave.generative.config import load_sleep2wave_config

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTOENCODER_TINY = REPO_ROOT / "configs" / "sleep2wave" / "sleep2wave_autoencoder_tiny.yaml"
DIFFUSION_TINY = REPO_ROOT / "configs" / "sleep2wave" / "sleep2wave_diffusion_tiny_phase1.yaml"
GENERATE_TINY = REPO_ROOT / "configs" / "sleep2wave" / "sleep2wave_generate_tiny.yaml"
EVAL_TINY = REPO_ROOT / "configs" / "sleep2wave" / "sleep2wave_eval_tiny.yaml"
AUTOENCODER_MEDIUM = REPO_ROOT / "configs" / "sleep2wave" / "sleep2wave_autoencoder_medium.yaml"
GENERATE_MEDIUM = REPO_ROOT / "configs" / "sleep2wave" / "sleep2wave_generate_medium.yaml"
EVAL_MEDIUM = REPO_ROOT / "configs" / "sleep2wave" / "sleep2wave_eval_medium.yaml"
TINY_CONFIGS = sorted((REPO_ROOT / "configs" / "sleep2wave").glob("*tiny*.yaml"))
TINY_DIFFUSION_CONFIGS = sorted((REPO_ROOT / "configs" / "sleep2wave").glob("sleep2wave_diffusion_tiny_phase*.yaml"))
MEDIUM_CONFIGS = sorted((REPO_ROOT / "configs" / "sleep2wave").glob("*medium*.yaml"))
MEDIUM_DIFFUSION_CONFIGS = sorted(
    (REPO_ROOT / "configs" / "sleep2wave").glob("sleep2wave_diffusion_medium_phase*.yaml")
)


def _load_payload(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload))
    return path


def test_sleep2wave_generative_config_loads_autoencoder_tiny():
    cfg = load_sleep2wave_config(AUTOENCODER_TINY)

    assert cfg.recipe == "sleep2wave"
    assert cfg.stage == "autoencoder"
    assert cfg.modalities.all == list(CANONICAL_MODALITIES)
    assert cfg.data.context_epochs == 15
    assert cfg.autoencoder is not None
    assert cfg.autoencoder.latent_dim == 64
    assert cfg.autoencoder.encoder_type == "temporal_conv"
    assert cfg.autoencoder.decoder_type == "temporal_conv"
    assert cfg.autoencoder.latent_frames_per_epoch == {"high_frequency": 60, "low_frequency": 30}
    assert cfg.autoencoder.channel_specific is True
    assert cfg.autoencoder.losses.waveform_l2_weight == 0.1
    assert cfg.autoencoder.losses.derivative_l1_weight == 0.1
    assert cfg.autoencoder.losses.mr_stft_weight == 0.1
    assert cfg.training.validation.interval_steps == 1000
    assert cfg.training.validation.max_batches_per_modality == 5
    assert cfg.training.validation.examples.num_examples == 5
    assert cfg.training.validation.examples.modalities == list(CANONICAL_MODALITIES)
    assert cfg.diffusion is None
    assert cfg.sampler is None


def test_sleep2wave_autoencoder_validation_examples_default_to_all_modalities(tmp_path: Path):
    payload = _load_payload(AUTOENCODER_TINY)
    payload["training"]["validation"].pop("examples")
    path = _write_yaml(tmp_path / "autoencoder.yaml", payload)

    cfg = load_sleep2wave_config(path)

    assert cfg.training.validation.examples.num_examples == 1
    assert cfg.training.validation.examples.modalities == list(CANONICAL_MODALITIES)


@pytest.mark.parametrize(
    ("validation_examples", "match"),
    [
        ({"num_examples": 0, "modalities": ["eeg"]}, "num_examples must be an integer >= 1"),
        ({"num_examples": 1, "modalities": []}, "Modality sequence must be a non-empty list"),
        ({"num_examples": 1, "modalities": ["eeg", "eeg"]}, "Duplicate sleep2wave modality"),
        ({"num_examples": 1, "modalities": ["unknown"]}, "canonical modality names"),
    ],
)
def test_sleep2wave_autoencoder_validation_examples_reject_invalid_values(
    tmp_path: Path,
    validation_examples: dict,
    match: str,
):
    payload = _load_payload(AUTOENCODER_TINY)
    payload["training"]["validation"]["examples"] = validation_examples
    path = _write_yaml(tmp_path / "autoencoder.yaml", payload)

    with pytest.raises(ValueError, match=match):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_loads_diffusion_tiny():
    cfg = load_sleep2wave_config(DIFFUSION_TINY)

    assert cfg.recipe == "sleep2wave"
    assert cfg.stage == "diffusion"
    assert cfg.diffusion is not None
    assert cfg.diffusion.latent_dim == 64
    assert cfg.diffusion.latent_frames_per_epoch == {"high_frequency": 60, "low_frequency": 30}
    assert cfg.diffusion.patches_per_epoch == 6
    assert cfg.diffusion.context_epochs == 15
    assert cfg.diffusion.embeddings.channel_position is True
    assert cfg.diffusion.embeddings.patch_position is True
    assert cfg.sampler is not None
    assert cfg.sampler.name == "ddim"
    assert cfg.sampler.steps == 20
    assert cfg.training.validation.interval_steps == 1000
    assert cfg.training.validation.max_batches_per_modality == 5
    assert cfg.training.validation.examples.num_examples == 5
    assert cfg.training.validation.examples.modalities == list(CANONICAL_MODALITIES)
    assert cfg.training.restoration_condition_counts == [1, 2, 3]
    assert cfg.training.replay.enabled is True
    assert cfg.training.corruptions.restoration.default.name == "gaussian_noise"
    assert cfg.training.corruptions.restoration.default.kwargs == {"std": 0.05}
    assert cfg.training.corruptions.imputation.default.name == "contiguous_window_mask"
    assert cfg.training.corruptions.imputation.default.kwargs == {"window_frames": 120}
    assert set(cfg.training.corruptions.restoration.by_modality) == set(CANONICAL_MODALITIES)
    assert set(cfg.training.corruptions.imputation.by_modality) == set(CANONICAL_MODALITIES)
    assert [choice.name for choice in cfg.training.corruptions.restoration.by_modality["eeg"].choices] == [
        "high_frequency_contamination",
        "gaussian_noise",
    ]


def test_sleep2wave_diffusion_validation_examples_default_to_all_modalities(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["training"]["validation"].pop("examples")
    path = _write_yaml(tmp_path / "diffusion.yaml", payload)

    cfg = load_sleep2wave_config(path)

    assert cfg.training.validation.examples.num_examples == 1
    assert cfg.training.validation.examples.modalities == list(CANONICAL_MODALITIES)


def test_sleep2wave_training_validation_defaults(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["training"].pop("validation")
    path = _write_yaml(tmp_path / "diffusion.yaml", payload)

    cfg = load_sleep2wave_config(path)

    assert cfg.training.validation.interval_steps == 1000
    assert cfg.training.validation.max_batches_per_modality == 1
    assert cfg.training.validation.examples.num_examples == 1
    assert cfg.training.validation.examples.modalities == list(CANONICAL_MODALITIES)


@pytest.mark.parametrize(
    ("validation", "match"),
    [
        ({"interval_steps": 0}, "training.validation.interval_steps must be an integer >= 1"),
        ({"interval_steps": True}, "training.validation.interval_steps must be an integer >= 1"),
        ({"max_batches_per_modality": 0}, "training.validation.max_batches_per_modality must be an integer >= 1"),
        ({"unknown": 1}, "training.validation has unsupported fields"),
    ],
)
def test_sleep2wave_training_validation_rejects_invalid_values(
    tmp_path: Path,
    validation: dict,
    match: str,
):
    payload = _load_payload(DIFFUSION_TINY)
    payload["training"]["validation"] = validation
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match=match):
        load_sleep2wave_config(path)


@pytest.mark.parametrize(
    ("validation_examples", "match"),
    [
        ({"num_examples": 0, "modalities": ["eeg"]}, "num_examples must be an integer >= 1"),
        ({"num_examples": 1, "modalities": []}, "Modality sequence must be a non-empty list"),
        ({"num_examples": 1, "modalities": ["eeg", "eeg"]}, "Duplicate sleep2wave modality"),
        ({"num_examples": 1, "modalities": ["unknown"]}, "canonical modality names"),
    ],
)
def test_sleep2wave_diffusion_validation_examples_reject_invalid_values(
    tmp_path: Path,
    validation_examples: dict,
    match: str,
):
    payload = _load_payload(DIFFUSION_TINY)
    payload["training"]["validation"]["examples"] = validation_examples
    path = _write_yaml(tmp_path / "diffusion.yaml", payload)

    with pytest.raises(ValueError, match=match):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_loads_inference_tiny():
    cfg = load_sleep2wave_config(GENERATE_TINY)

    assert cfg.recipe == "sleep2wave"
    assert cfg.stage == "inference"
    assert cfg.training is None
    assert cfg.autoencoder is None
    assert cfg.diffusion is not None
    assert cfg.inference is not None
    assert cfg.inference.corruptions.restoration.default.name == "gaussian_noise"
    assert cfg.inference.corruptions.imputation.default.name == "contiguous_window_mask"
    assert cfg.sampler is not None
    assert cfg.sampler.steps == 20
    assert cfg.sampler.num_samples == 4


def test_sleep2wave_generative_config_loads_evaluation_tiny():
    cfg = load_sleep2wave_config(EVAL_TINY)

    assert cfg.recipe == "sleep2wave"
    assert cfg.stage == "evaluation"
    assert cfg.data is None
    assert cfg.training is None
    assert cfg.autoencoder is None
    assert cfg.diffusion is None
    assert cfg.sampler is None
    assert cfg.evaluation is not None
    assert cfg.evaluation.metric_families == ["waveform", "feature", "event", "efficiency", "downstream"]
    assert cfg.evaluation.max_shift_frames == 3
    assert cfg.evaluation.corruption_mask_policy == "exclude"


def test_sleep2wave_generative_config_loads_weighted_corruption_choices(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["training"]["corruptions"]["restoration"]["default"] = {
        "choices": [
            {"weight": 0.7, "name": "gaussian_noise", "kwargs": {"std": 0.1}},
            {"weight": 0.3, "name": "baseline_drift", "kwargs": {"amplitude": 0.2}},
        ]
    }
    path = _write_yaml(tmp_path / "good.yaml", payload)

    cfg = load_sleep2wave_config(path)
    spec = cfg.training.corruptions.restoration.default

    assert [(choice.weight, choice.name, choice.kwargs) for choice in spec.choices] == [
        (0.7, "gaussian_noise", {"std": 0.1}),
        (0.3, "baseline_drift", {"amplitude": 0.2}),
    ]
    assert spec.select(seed=1).name == "gaussian_noise"
    assert spec.select(seed=0).name == "baseline_drift"


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"choices": []}, "choices must be a non-empty list"),
        ({"choices": [{"name": "gaussian_noise"}]}, "weight must be a positive finite number"),
        ({"choices": [{"weight": 0.0, "name": "gaussian_noise"}]}, "weight must be a positive finite number"),
        ({"choices": [{"weight": 1.0, "name": "unknown"}]}, "name must be one of"),
        (
            {"choices": [{"weight": 1.0, "name": "gaussian_noise", "kwargs": {"nested": {"std": 0.1}}}]},
            "must be a scalar value",
        ),
        (
            {
                "name": "gaussian_noise",
                "choices": [{"weight": 1.0, "name": "baseline_drift"}],
            },
            "must define exactly one of 'name' or 'choices'",
        ),
    ],
)
def test_sleep2wave_generative_config_rejects_invalid_weighted_corruption_choices(
    tmp_path: Path,
    spec: dict,
    match: str,
):
    payload = _load_payload(DIFFUSION_TINY)
    payload["training"]["corruptions"]["restoration"]["default"] = spec
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match=match):
        load_sleep2wave_config(path)


def test_sleep2wave_tiny_configs_match_medium_non_size_specs():
    tiny_autoencoder = _load_payload(AUTOENCODER_TINY)
    medium_autoencoder = _load_payload(AUTOENCODER_MEDIUM)
    assert (
        tiny_autoencoder["data"]["preset_path"].replace("_tiny", "_medium") == medium_autoencoder["data"]["preset_path"]
    )
    assert tiny_autoencoder["data"]["context_epochs"] == medium_autoencoder["data"]["context_epochs"]
    assert tiny_autoencoder["modalities"] == medium_autoencoder["modalities"]
    tiny_autoencoder_model = dict(tiny_autoencoder["autoencoder"])
    medium_autoencoder_model = dict(medium_autoencoder["autoencoder"])
    tiny_autoencoder_model.pop("latent_dim")
    medium_autoencoder_model.pop("latent_dim")
    assert tiny_autoencoder_model == medium_autoencoder_model
    assert tiny_autoencoder["training"] == medium_autoencoder["training"]
    assert (
        tiny_autoencoder["export"]["output_dir"].replace("_tiny", "_medium")
        == medium_autoencoder["export"]["output_dir"]
    )

    tiny_by_phase = {path.stem.rsplit("_phase", 1)[1]: _load_payload(path) for path in TINY_DIFFUSION_CONFIGS}
    medium_by_phase = {path.stem.rsplit("_phase", 1)[1]: _load_payload(path) for path in MEDIUM_DIFFUSION_CONFIGS}
    assert set(tiny_by_phase) == set(medium_by_phase)
    for phase, tiny_diffusion in tiny_by_phase.items():
        medium_diffusion = medium_by_phase[phase]
        assert (
            tiny_diffusion["data"]["preset_path"].replace("_tiny", "_medium") == medium_diffusion["data"]["preset_path"]
        )
        assert tiny_diffusion["data"]["context_epochs"] == medium_diffusion["data"]["context_epochs"]
        assert tiny_diffusion["modalities"] == medium_diffusion["modalities"]
        assert tiny_diffusion["sampler"] == medium_diffusion["sampler"]

        tiny_training = dict(tiny_diffusion["training"])
        medium_training = dict(medium_diffusion["training"])
        if "phase_checkpoint" in tiny_training:
            assert tiny_training["phase_checkpoint"].replace("_tiny", "_medium") == medium_training["phase_checkpoint"]
            tiny_training.pop("phase_checkpoint")
            medium_training.pop("phase_checkpoint")
        assert tiny_training == medium_training

        tiny_diffusion_model = dict(tiny_diffusion["diffusion"])
        medium_diffusion_model = dict(medium_diffusion["diffusion"])
        assert (
            tiny_diffusion_model["autoencoder_checkpoint"].replace("_tiny", "_medium")
            == medium_diffusion_model["autoencoder_checkpoint"]
        )
        tiny_transformer = dict(tiny_diffusion_model.pop("transformer"))
        medium_transformer = dict(medium_diffusion_model.pop("transformer"))
        for key in ("latent_dim", "autoencoder_checkpoint"):
            tiny_diffusion_model.pop(key)
            medium_diffusion_model.pop(key)
        for key in ("hidden_size", "num_layers", "num_heads"):
            tiny_transformer.pop(key)
            medium_transformer.pop(key)
        assert tiny_diffusion_model == medium_diffusion_model
        assert tiny_transformer == medium_transformer
        assert (
            tiny_diffusion["export"]["output_dir"].replace("_tiny", "_medium")
            == medium_diffusion["export"]["output_dir"]
        )

    tiny_generate = _load_payload(GENERATE_TINY)
    medium_generate = _load_payload(GENERATE_MEDIUM)
    assert tiny_generate["data"]["preset_path"].replace("_tiny", "_medium") == medium_generate["data"]["preset_path"]
    assert tiny_generate["data"]["context_epochs"] == medium_generate["data"]["context_epochs"]
    assert tiny_generate["modalities"] == medium_generate["modalities"]
    assert tiny_generate["inference"] == medium_generate["inference"]
    assert tiny_generate["sampler"] == medium_generate["sampler"]
    tiny_generate_diffusion = dict(tiny_generate["diffusion"])
    medium_generate_diffusion = dict(medium_generate["diffusion"])
    assert (
        tiny_generate_diffusion["autoencoder_checkpoint"].replace("_tiny", "_medium")
        == medium_generate_diffusion["autoencoder_checkpoint"]
    )
    tiny_generate_transformer = dict(tiny_generate_diffusion.pop("transformer"))
    medium_generate_transformer = dict(medium_generate_diffusion.pop("transformer"))
    for key in ("latent_dim", "autoencoder_checkpoint"):
        tiny_generate_diffusion.pop(key)
        medium_generate_diffusion.pop(key)
    for key in ("hidden_size", "num_layers", "num_heads"):
        tiny_generate_transformer.pop(key)
        medium_generate_transformer.pop(key)
    assert tiny_generate_diffusion == medium_generate_diffusion
    assert tiny_generate_transformer == medium_generate_transformer
    assert tiny_generate["export"]["output_dir"].replace("_tiny", "_medium") == medium_generate["export"]["output_dir"]

    tiny_eval = _load_payload(EVAL_TINY)
    medium_eval = _load_payload(EVAL_MEDIUM)
    assert tiny_eval["modalities"] == medium_eval["modalities"]
    tiny_evaluation = dict(tiny_eval["evaluation"])
    medium_evaluation = dict(medium_eval["evaluation"])
    assert tiny_evaluation["generated_dir"].replace("_tiny", "_medium") == medium_evaluation["generated_dir"]
    tiny_evaluation.pop("generated_dir")
    medium_evaluation.pop("generated_dir")
    assert tiny_evaluation == medium_evaluation
    assert tiny_eval["export"]["output_dir"].replace("_tiny", "_medium") == medium_eval["export"]["output_dir"]


@pytest.mark.parametrize("path", TINY_CONFIGS)
def test_sleep2wave_generative_config_loads_tiny_bundle(path: Path):
    cfg = load_sleep2wave_config(path)

    assert cfg.recipe == "sleep2wave"
    if cfg.autoencoder is not None:
        assert cfg.autoencoder.latent_dim == 64
    if cfg.diffusion is not None:
        assert cfg.diffusion.latent_dim == 64
        assert cfg.diffusion.latent_frames_per_epoch == {"high_frequency": 60, "low_frequency": 30}
        assert cfg.diffusion.patches_per_epoch == 6
        assert cfg.diffusion.embeddings.channel_position is True
        assert cfg.diffusion.embeddings.patch_position is True
        assert cfg.diffusion.transformer.hidden_size == 64
        assert cfg.diffusion.transformer.num_layers == 2
        assert cfg.diffusion.transformer.num_heads == 4


def test_sleep2wave_generative_config_rejects_unknown_top_level_field(tmp_path: Path):
    payload = _load_payload(AUTOENCODER_TINY)
    payload["unexpected"] = True
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="top-level sleep2wave config has unsupported fields"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_unsupported_stage(tmp_path: Path):
    payload = _load_payload(AUTOENCODER_TINY)
    payload["stage"] = "unknown"
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="stage must be one of"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_unknown_modality(tmp_path: Path):
    payload = _load_payload(AUTOENCODER_TINY)
    payload["modalities"]["all"] = list(payload["modalities"]["all"]) + ["unknown"]
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="canonical modality names"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_wrong_sample_rate(tmp_path: Path):
    payload = _load_payload(AUTOENCODER_TINY)
    payload["modalities"]["sample_rates"]["eeg"] = 256
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="modalities.sample_rates.eeg must be 128"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_wrong_frames_per_epoch(tmp_path: Path):
    payload = _load_payload(AUTOENCODER_TINY)
    payload["modalities"]["frames_per_epoch"]["spo2"] = 128
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="modalities.frames_per_epoch.spo2 must be 120"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_missing_stage_required_block(tmp_path: Path):
    payload = _load_payload(AUTOENCODER_TINY)
    del payload["autoencoder"]
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="autoencoder block is required for stage=autoencoder"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_invalid_context_epochs(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["diffusion"]["context_epochs"] = 0
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="diffusion.context_epochs must be an integer >= 1"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_diffusion_data_context_mismatch(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["data"]["context_epochs"] = 2
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="data.context_epochs must match diffusion.context_epochs"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_inference_data_context_mismatch(tmp_path: Path):
    payload = _load_payload(GENERATE_TINY)
    payload["data"]["context_epochs"] = 2
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="data.context_epochs must match diffusion.context_epochs"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_cache_only_restoration_config(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    del payload["diffusion"]["autoencoder_checkpoint"]
    payload["diffusion"]["latent_cache_path"] = "latents"
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="only supports translation, two_condition, and partial_full"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_loads_cache_only_translation(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    del payload["diffusion"]["autoencoder_checkpoint"]
    payload["diffusion"]["latent_cache_path"] = "latents"
    payload["training"]["phase"] = 2
    payload["training"]["task_mix"] = {"translation": 1.0}
    path = _write_yaml(tmp_path / "good.yaml", payload)

    cfg = load_sleep2wave_config(path)

    assert cfg.diffusion.autoencoder_checkpoint is None
    assert cfg.diffusion.latent_cache_path == "latents"


def test_sleep2wave_generative_config_rejects_cache_only_inference(tmp_path: Path):
    payload = _load_payload(GENERATE_TINY)
    del payload["diffusion"]["autoencoder_checkpoint"]
    payload["diffusion"]["latent_cache_path"] = "latents"
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="stage=inference requires diffusion.autoencoder_checkpoint"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_loads_phase_checkpoint(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["training"]["phase_checkpoint"] = "previous.ckpt"
    path = _write_yaml(tmp_path / "good.yaml", payload)

    cfg = load_sleep2wave_config(path)

    assert cfg.training.phase_checkpoint == "previous.ckpt"


def test_sleep2wave_generative_config_rejects_non_finite_float(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["training"]["lr"] = float("nan")
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="training.lr must be finite"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_non_finite_task_mix(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["training"]["task_mix"] = {"restoration": float("nan")}
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="training.task_mix.restoration must be a non-negative finite number"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_all_zero_task_mix(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["training"]["task_mix"] = {"restoration": 0.0, "translation": 0.0}
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="training.task_mix must include at least one positive task weight"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_invalid_restoration_condition_counts(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["training"]["restoration_condition_counts"] = [0]
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="training.restoration_condition_counts must contain positive integers"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_both_data_sources(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["data"]["index"] = "index.csv"
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="data must define exactly one"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_non_divisible_attention_heads(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["diffusion"]["transformer"]["hidden_size"] = 10
    payload["diffusion"]["transformer"]["num_heads"] = 4
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="hidden_size must be divisible"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_diffusion_phase_zero(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["training"]["phase"] = 0
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="stage=diffusion requires training.phase between 1 and 5"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_inference_training_block(tmp_path: Path):
    payload = _load_payload(GENERATE_TINY)
    payload["training"] = {
        "phase": 1,
        "batch_size": 1,
        "lr": 0.0001,
        "weight_decay": 0.01,
        "max_epochs": 1,
        "gradient_clip_val": 1.0,
    }
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="stage=inference does not support a training block"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_inference_missing_sampler(tmp_path: Path):
    payload = _load_payload(GENERATE_TINY)
    del payload["sampler"]
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="sampler block is required for stage=inference"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_inference_diffusion_validation(tmp_path: Path):
    payload = _load_payload(GENERATE_TINY)
    payload["diffusion"]["validation"] = {"examples": {"num_examples": 1, "modalities": ["eeg"]}}
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="diffusion has unsupported fields"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_inference_missing_diffusion(tmp_path: Path):
    payload = _load_payload(GENERATE_TINY)
    del payload["diffusion"]
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="diffusion block is required for stage=inference"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_diffusion_inference_block(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["inference"] = {"corruptions": {}}
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="stage=diffusion does not support an inference block"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_accepts_two_condition_task_mix(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["training"]["task_mix"] = {"two_condition": 1.0}
    path = _write_yaml(tmp_path / "good.yaml", payload)

    cfg = load_sleep2wave_config(path)

    assert cfg.training is not None
    assert cfg.training.task_mix == {"two_condition": 1.0}


@pytest.mark.parametrize("path", MEDIUM_CONFIGS)
def test_sleep2wave_generative_config_loads_medium_bundle(path: Path):
    cfg = load_sleep2wave_config(path)

    assert cfg.recipe == "sleep2wave"
    if cfg.diffusion is not None:
        assert cfg.diffusion.latent_dim == 64
        assert cfg.diffusion.latent_frames_per_epoch == {"high_frequency": 60, "low_frequency": 30}
        assert cfg.diffusion.patches_per_epoch == 6
        assert cfg.diffusion.embeddings.channel_position is True
        assert cfg.diffusion.embeddings.patch_position is True
        assert cfg.diffusion.transformer.hidden_size == 256
        assert cfg.diffusion.transformer.num_layers == 6
        assert cfg.diffusion.transformer.num_heads == 8


@pytest.mark.parametrize("path", MEDIUM_DIFFUSION_CONFIGS)
def test_sleep2wave_medium_diffusion_configs_use_modality_specific_corruptions(path: Path):
    cfg = load_sleep2wave_config(path)

    restoration = cfg.training.corruptions.restoration.by_modality
    imputation = cfg.training.corruptions.imputation.by_modality

    assert set(restoration) == set(CANONICAL_MODALITIES)
    assert set(imputation) == set(CANONICAL_MODALITIES)
    assert restoration["eeg"].name == "high_frequency_contamination"
    assert [choice.weight for choice in restoration["eeg"].choices] == [0.7, 0.3]
    assert restoration["airflow"].name == "airflow_cannula_displacement"
    assert restoration["spo2"].name == "saturation_clipping"
    assert imputation["eeg"].kwargs == {"window_frames": 384}
    assert imputation["belt"].name == "belt_failure"
    assert imputation["spo2"].name == "spo2_plateau_dropout"


def test_sleep2wave_generative_config_rejects_invalid_data_context_epochs(tmp_path: Path):
    payload = _load_payload(AUTOENCODER_TINY)
    payload["data"]["context_epochs"] = 0
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="data.context_epochs must be an integer >= 1"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_invalid_sampler_steps(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["sampler"]["steps"] = 0
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="sampler.steps must be an integer >= 1"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_sampler_steps_above_diffusion_steps(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["sampler"]["steps"] = payload["diffusion"]["diffusion_steps"] + 1
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="sampler.steps must be <= diffusion.diffusion_steps"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_sparse_ddpm_steps(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["sampler"]["name"] = "ddpm"
    payload["sampler"]["steps"] = payload["diffusion"]["diffusion_steps"] - 1
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="sampler.steps must equal diffusion.diffusion_steps for DDPM"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_unsupported_autoencoder_type(tmp_path: Path):
    payload = _load_payload(AUTOENCODER_TINY)
    payload["autoencoder"]["encoder_type"] = "other"
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="autoencoder.encoder_type must be 'temporal_conv'"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_false_channel_specific(tmp_path: Path):
    payload = _load_payload(AUTOENCODER_TINY)
    payload["autoencoder"]["channel_specific"] = False
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="autoencoder.channel_specific must be true"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_old_autoencoder_fields(tmp_path: Path):
    payload = _load_payload(AUTOENCODER_TINY)
    payload["autoencoder"]["one_latent_per_epoch"] = True
    payload["autoencoder"]["modality_specific"] = True
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="autoencoder has unsupported fields"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_invalid_latent_frames(tmp_path: Path):
    payload = _load_payload(AUTOENCODER_TINY)
    payload["autoencoder"]["latent_frames_per_epoch"]["high_frequency"] = 0
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="autoencoder.latent_frames_per_epoch.high_frequency"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_diffusion_latent_frames_not_divisible_by_patches(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    payload["diffusion"]["latent_frames_per_epoch"]["high_frequency"] = 61
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="must be divisible by diffusion.patches_per_epoch"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_requires_patch_position_embedding_flag(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    del payload["diffusion"]["embeddings"]["patch_position"]
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="diffusion.embeddings.patch_position must be a boolean"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_requires_channel_position_embedding_flag(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    del payload["diffusion"]["embeddings"]["channel_position"]
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="diffusion.embeddings.channel_position must be a boolean"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_all_zero_autoencoder_loss_weights(tmp_path: Path):
    payload = _load_payload(AUTOENCODER_TINY)
    for key in payload["autoencoder"]["losses"]:
        payload["autoencoder"]["losses"][key] = 0.0
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="At least one sleep2wave autoencoder loss weight"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_aliases_in_configs(tmp_path: Path):
    payload = deepcopy(_load_payload(AUTOENCODER_TINY))
    payload["modalities"]["all"][0] = "eeg_original"
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="canonical modality names"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_unknown_initialization_group(tmp_path: Path):
    payload = deepcopy(_load_payload(AUTOENCODER_TINY))
    payload["initialization"] = {
        "sleep2vec2_checkpoint": "checkpoint.ckpt",
        "strict_compatible": True,
        "require_any_loaded": False,
        "load_groups": {"unknown_group": True},
    }
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="initialization.load_groups keys must be one of"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_enabled_initialization_without_checkpoint(tmp_path: Path):
    payload = deepcopy(_load_payload(AUTOENCODER_TINY))
    payload["initialization"] = {
        "strict_compatible": True,
        "require_any_loaded": False,
        "load_groups": {"autoencoder_encoders": True},
    }
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="initialization.sleep2vec2_checkpoint is required"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_evaluation_training_block(tmp_path: Path):
    payload = _load_payload(EVAL_TINY)
    payload["training"] = {
        "phase": 1,
        "batch_size": 1,
        "lr": 0.0001,
        "weight_decay": 0.01,
        "max_epochs": 1,
        "gradient_clip_val": 1.0,
    }
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="stage=evaluation does not support a training block"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_evaluation_diffusion_block(tmp_path: Path):
    payload = _load_payload(EVAL_TINY)
    payload["diffusion"] = _load_payload(DIFFUSION_TINY)["diffusion"]
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="stage=evaluation does not support a diffusion block"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_evaluation_sampler_block(tmp_path: Path):
    payload = _load_payload(EVAL_TINY)
    payload["sampler"] = {"name": "ddim", "steps": 1, "eta": 0.0, "num_samples": 1}
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="stage=evaluation does not support a sampler block"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_unknown_evaluation_metric_family(tmp_path: Path):
    payload = _load_payload(EVAL_TINY)
    payload["evaluation"]["metric_families"] = ["waveform", "unknown"]
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="evaluation.metric_families entries must be one of"):
        load_sleep2wave_config(path)


@pytest.mark.parametrize("policy", ["exclude", "include", "only_corrupted"])
def test_sleep2wave_generative_config_accepts_evaluation_corruption_mask_policy(tmp_path: Path, policy: str):
    payload = _load_payload(EVAL_TINY)
    payload["evaluation"]["corruption_mask_policy"] = policy
    path = _write_yaml(tmp_path / "good.yaml", payload)

    cfg = load_sleep2wave_config(path)

    assert cfg.evaluation.corruption_mask_policy == policy


def test_sleep2wave_generative_config_rejects_unknown_evaluation_corruption_mask_policy(tmp_path: Path):
    payload = _load_payload(EVAL_TINY)
    payload["evaluation"]["corruption_mask_policy"] = "unknown"
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="evaluation.corruption_mask_policy must be one of"):
        load_sleep2wave_config(path)
