from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from sleep2wave.evaluate_generation import _jsonable, _load_metric_epoch_mask, _write_metrics, main
from sleep2wave.export.manifest import build_generation_manifest


def _modalities_payload() -> dict:
    return {
        "epoch_sec": 30,
        "all": ["eeg", "eog", "emg", "ecg", "airflow", "belt", "spo2", "ibi", "resp"],
        "high_frequency": ["eeg", "eog", "emg", "ecg"],
        "low_frequency": ["airflow", "belt", "spo2", "ibi", "resp"],
        "sample_rates": {
            "eeg": 128,
            "eog": 128,
            "emg": 128,
            "ecg": 128,
            "airflow": 4,
            "belt": 4,
            "spo2": 4,
            "ibi": 4,
            "resp": 4,
        },
        "frames_per_epoch": {
            "eeg": 3840,
            "eog": 3840,
            "emg": 3840,
            "ecg": 3840,
            "airflow": 120,
            "belt": 120,
            "spo2": 120,
            "ibi": 120,
            "resp": 120,
        },
    }


def _write_eval_config(tmp_path: Path, generated_dir: Path, reference_npz: Path | None) -> Path:
    payload = {
        "recipe": "sleep2wave",
        "stage": "evaluation",
        "modalities": _modalities_payload(),
        "evaluation": {
            "generated_dir": str(generated_dir),
            "reference_npz": str(reference_npz) if reference_npz is not None else None,
            "baseline_npz": None,
            "events_json": None,
            "downstream_metrics_json": None,
            "metric_families": ["waveform", "feature", "event", "efficiency", "downstream"],
            "max_shift_frames": 1,
            "event_iou_threshold": 0.5,
        },
        "export": {"output_dir": str(tmp_path / "eval")},
    }
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def _write_generated_artifacts(tmp_path: Path) -> Path:
    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    generated_eeg = np.stack(
        [
            np.ones((2, 4), dtype=np.float32),
            np.ones((2, 4), dtype=np.float32) * 3.0,
        ],
        axis=0,
    )
    generated_spo2 = np.stack(
        [
            np.array([[98.0, 94.0, 95.0, 97.0], [99.0, 96.0, 95.0, 98.0]], dtype=np.float32),
            np.array([[98.0, 94.0, 95.0, 97.0], [99.0, 96.0, 95.0, 98.0]], dtype=np.float32),
        ],
        axis=0,
    )
    np.savez(output_dir / "generated.npz", **{"generated/eeg": generated_eeg, "generated/spo2": generated_spo2})
    np.savez(
        output_dir / "uncertainty.npz",
        **{
            "mean/eeg": generated_eeg.mean(axis=0),
            "std/eeg": generated_eeg.std(axis=0),
            "sample_count/eeg": np.array([2]),
            "high_uncertainty_mask/eeg": np.array([False, False]),
            "mean/spo2": generated_spo2.mean(axis=0),
            "std/spo2": generated_spo2.std(axis=0),
            "sample_count/spo2": np.array([2]),
            "high_uncertainty_mask/spo2": np.array([False, False]),
        },
    )
    np.savez(output_dir / "masks.npz", **{"target/eeg": np.ones(2, dtype=bool)})
    (output_dir / "metadata.jsonl").write_text(json.dumps({"subject_id": "s1", "night_id": "n1"}) + "\n")
    manifest = build_generation_manifest(
        task_type="translation",
        condition_modalities=["ecg"],
        target_modalities=["eeg", "spo2"],
        diffusion_ckpt="diffusion.ckpt",
        autoencoder_ckpt="autoencoder.ckpt",
        sampler={"name": "ddim", "steps": 2, "eta": 0.0, "num_samples": 2},
        output_files=["generated.npz", "uncertainty.npz", "masks.npz", "metadata.jsonl"],
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest))
    return output_dir


def test_evaluate_cli_rejects_missing_generated_dir(tmp_path: Path):
    reference_npz = tmp_path / "reference.npz"
    np.savez(reference_npz, eeg=np.zeros((2, 4), dtype=np.float32))
    config_path = _write_eval_config(tmp_path, tmp_path / "missing", reference_npz)

    with pytest.raises(FileNotFoundError, match="Generated artifact directory not found"):
        main(["--config", str(config_path)])


def test_evaluate_cli_rejects_missing_reference_for_waveform_metrics(tmp_path: Path):
    generated_dir = _write_generated_artifacts(tmp_path)
    config_path = _write_eval_config(tmp_path, generated_dir, None)

    with pytest.raises(ValueError, match="reference_npz is required"):
        main(["--config", str(config_path)])


def test_evaluate_cli_writes_metrics_json_and_csv(tmp_path: Path):
    generated_dir = _write_generated_artifacts(tmp_path)
    reference_npz = tmp_path / "reference.npz"
    np.savez(
        reference_npz,
        eeg=np.ones((2, 4), dtype=np.float32) * 2.0,
        spo2=np.array([[98.0, 94.0, 95.0, 97.0], [99.0, 96.0, 95.0, 98.0]], dtype=np.float32),
    )
    config_path = _write_eval_config(tmp_path, generated_dir, reference_npz)
    output_dir = tmp_path / "metrics"

    main(["--config", str(config_path), "--output-dir", str(output_dir)])

    metrics = json.loads((output_dir / "metrics.json").read_text())
    csv_text = (output_dir / "metrics.csv").read_text()

    assert metrics["artifact_type"] == "sleep2wave_generation_evaluation"
    assert metrics["metrics"]["waveform"]["eeg"]["rmse"] == pytest.approx(0.0)
    assert metrics["metrics"]["feature"]["spo2"]["nadir_error"] == pytest.approx(0.0)
    assert metrics["metrics"]["efficiency"]["all"]["sampler_steps"] == 2
    assert "waveform,eeg,rmse" in csv_text


def test_evaluate_cli_applies_exported_epoch_masks(tmp_path: Path):
    generated_dir = _write_generated_artifacts(tmp_path)
    np.savez(
        generated_dir / "uncertainty.npz",
        **{
            "mean/eeg": np.array([[2.0, 2.0, 2.0, 2.0], [100.0, 100.0, 100.0, 100.0]], dtype=np.float32),
            "std/eeg": np.zeros((2, 4), dtype=np.float32),
            "sample_count/eeg": np.array([1]),
            "high_uncertainty_mask/eeg": np.array([False, False]),
        },
    )
    np.savez(generated_dir / "masks.npz", **{"target/eeg": np.array([True, False])})
    reference_npz = tmp_path / "reference.npz"
    np.savez(reference_npz, eeg=np.ones((2, 4), dtype=np.float32) * 2.0)
    config_path = _write_eval_config(tmp_path, generated_dir, reference_npz)
    output_dir = tmp_path / "metrics_masked"

    main(["--config", str(config_path), "--output-dir", str(output_dir)])

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["metrics"]["waveform"]["eeg"]["rmse"] == pytest.approx(0.0)


def test_evaluate_metric_mask_combines_target_availability_quality_and_corruption(tmp_path: Path):
    masks_path = tmp_path / "masks.npz"
    np.savez(
        masks_path,
        **{
            "target/eeg": np.array([True, True, True, False]),
            "availability/eeg": np.array([True, False, True, True]),
            "quality/eeg": np.array([1.0, 1.0, 0.0, 1.0]),
            "corruption/eeg": np.array([False, False, True, False]),
        },
    )

    with np.load(masks_path) as masks:
        mask = _load_metric_epoch_mask(masks, "eeg", epoch_count=4)

    assert mask.tolist() == [True, False, False, False]


def test_evaluate_metrics_json_sanitizes_non_finite_values(tmp_path: Path):
    output_dir = tmp_path / "metrics_nonfinite"
    payload = {
        "artifact_type": "sleep2wave_generation_evaluation",
        "metrics": {
            "waveform": {
                "eeg": {
                    "rmse": float("nan"),
                    "per_epoch": np.array([1.0, np.inf]),
                }
            }
        },
    }

    _write_metrics(output_dir, payload)

    text = (output_dir / "metrics.json").read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text)["metrics"]["waveform"]["eeg"]["rmse"] is None
    assert _jsonable(np.array([np.inf])) == [None]
