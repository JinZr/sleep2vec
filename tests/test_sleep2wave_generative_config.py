from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from sleep2wave.data.modalities import CANONICAL_MODALITIES
from sleep2wave.generative.config import load_sleep2wave_config

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTOENCODER_TINY = REPO_ROOT / "configs" / "sleep2wave" / "sleep2wave_autoencoder_tiny.yaml"
DIFFUSION_TINY = REPO_ROOT / "configs" / "sleep2wave" / "sleep2wave_diffusion_tiny.yaml"
GENERATE_TINY = REPO_ROOT / "configs" / "sleep2wave" / "sleep2wave_generate_tiny.yaml"
EVAL_TINY = REPO_ROOT / "configs" / "sleep2wave" / "sleep2wave_eval_tiny.yaml"
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
    assert cfg.data.context_epochs == 2
    assert cfg.autoencoder is not None
    assert cfg.autoencoder.latent_dim == 64
    assert cfg.diffusion is None
    assert cfg.sampler is None


def test_sleep2wave_generative_config_loads_diffusion_tiny():
    cfg = load_sleep2wave_config(DIFFUSION_TINY)

    assert cfg.recipe == "sleep2wave"
    assert cfg.stage == "diffusion"
    assert cfg.diffusion is not None
    assert cfg.diffusion.latent_dim == 64
    assert cfg.diffusion.context_epochs == 15
    assert cfg.sampler is not None
    assert cfg.sampler.name == "ddim"
    assert cfg.sampler.steps == 10
    assert cfg.training.restoration_condition_counts == [1]
    assert cfg.training.corruptions.restoration.default.name == "gaussian_noise"
    assert cfg.training.corruptions.imputation.default.name == "contiguous_window_mask"


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
    assert cfg.sampler.num_samples == 2


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

    with pytest.raises(ValueError, match="supports only translation/partial_full"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_accepts_cache_only_translation_config(tmp_path: Path):
    payload = _load_payload(DIFFUSION_TINY)
    del payload["diffusion"]["autoencoder_checkpoint"]
    payload["diffusion"]["latent_cache_path"] = "latents"
    payload["training"]["phase"] = 2
    payload["training"]["task_mix"] = {"translation": 1.0}
    path = _write_yaml(tmp_path / "good.yaml", payload)

    cfg = load_sleep2wave_config(path)

    assert cfg.diffusion.autoencoder_checkpoint is None
    assert cfg.diffusion.latent_cache_path == "latents"


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
    payload["training"]["task_mix"]["restoration"] = float("nan")
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
        assert cfg.diffusion.latent_dim == 768
        assert cfg.diffusion.transformer.hidden_size == 768
        assert cfg.diffusion.transformer.num_layers == 12
        assert cfg.diffusion.transformer.num_heads == 16


@pytest.mark.parametrize("path", MEDIUM_DIFFUSION_CONFIGS)
def test_sleep2wave_medium_diffusion_configs_use_modality_specific_corruptions(path: Path):
    cfg = load_sleep2wave_config(path)

    restoration = cfg.training.corruptions.restoration.by_modality
    imputation = cfg.training.corruptions.imputation.by_modality

    assert set(restoration) == set(CANONICAL_MODALITIES)
    assert set(imputation) == set(CANONICAL_MODALITIES)
    assert restoration["eeg"].name == "high_frequency_contamination"
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

    with pytest.raises(ValueError, match="autoencoder.encoder_type must be 'conv1d_epoch'"):
        load_sleep2wave_config(path)


def test_sleep2wave_generative_config_rejects_false_autoencoder_flags(tmp_path: Path):
    payload = _load_payload(AUTOENCODER_TINY)
    payload["autoencoder"]["modality_specific"] = False
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ValueError, match="autoencoder.modality_specific must be true"):
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
