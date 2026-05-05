from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import typing as t

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sleep2wave generated PSG artifacts.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, default=None)
    parser.add_argument("--reference-npz", type=Path, default=None)
    parser.add_argument("--baseline-npz", type=Path, default=None)
    parser.add_argument("--events-json", type=Path, default=None)
    parser.add_argument("--downstream-metrics-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def _require_file(path: Path, *, description: str) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _require_artifact_dir(path: Path) -> dict[str, Path]:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Generated artifact directory not found: {path}")
    required = {
        "manifest": path / "manifest.json",
        "generated": path / "generated.npz",
        "uncertainty": path / "uncertainty.npz",
        "masks": path / "masks.npz",
        "metadata": path / "metadata.jsonl",
    }
    for name, artifact_path in required.items():
        _require_file(artifact_path, description=f"sleep2wave {name} artifact")
    return required


def _jsonable(value: t.Any) -> t.Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _npz_get(npz: np.lib.npyio.NpzFile, modality: str, *, families: t.Sequence[str]) -> np.ndarray | None:
    for family in families:
        key = f"{family}/{modality}" if family else modality
        if key in npz.files:
            return npz[key]
    return None


def _load_generated_mean(
    generated_npz: np.lib.npyio.NpzFile,
    uncertainty_npz: np.lib.npyio.NpzFile,
    modality: str,
) -> np.ndarray | None:
    uncertainty_mean = _npz_get(uncertainty_npz, modality, families=["mean"])
    if uncertainty_mean is not None:
        return uncertainty_mean
    generated = _npz_get(generated_npz, modality, families=["generated"])
    if generated is None:
        return None
    if generated.ndim < 2:
        raise ValueError(f"generated/{modality} must include a sample dimension.")
    return generated.mean(axis=0)


def _load_reference(npz: np.lib.npyio.NpzFile, modality: str) -> np.ndarray | None:
    return _npz_get(npz, modality, families=["", "reference", "clean", "target"])


def _load_baseline(npz: np.lib.npyio.NpzFile | None, modality: str) -> np.ndarray | None:
    if npz is None:
        return None
    return _npz_get(npz, modality, families=["", "baseline", "observed", "degraded"])


def _load_metric_epoch_mask(masks_npz: np.lib.npyio.NpzFile, modality: str, epoch_count: int) -> np.ndarray:
    mask = np.ones((epoch_count,), dtype=bool)
    target = _npz_get(masks_npz, modality, families=["target"])
    availability = _npz_get(masks_npz, modality, families=["availability"])
    quality = _npz_get(masks_npz, modality, families=["quality"])
    corruption = _npz_get(masks_npz, modality, families=["corruption"])

    if target is not None:
        mask &= _epoch_bool_mask(target, epoch_count, name=f"target/{modality}")
    if availability is not None:
        mask &= _epoch_bool_mask(availability, epoch_count, name=f"availability/{modality}")
    if quality is not None:
        mask &= _epoch_bool_mask(np.asarray(quality) > 0, epoch_count, name=f"quality/{modality}")
    if corruption is not None:
        mask &= ~_epoch_bool_mask(corruption, epoch_count, name=f"corruption/{modality}")
    return mask


def _epoch_bool_mask(values: np.ndarray, epoch_count: int, *, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[0] != epoch_count:
        raise ValueError(f"{name} must have first dimension {epoch_count}, got {values.shape}.")
    if values.ndim == 1:
        return values.astype(bool)
    reduce_axes = tuple(range(1, values.ndim))
    return values.astype(bool).any(axis=reduce_axes)


def _apply_metric_epoch_mask(
    reference: np.ndarray,
    generated: np.ndarray,
    baseline: np.ndarray | None,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    if reference.shape[0] != mask.shape[0] or generated.shape[0] != mask.shape[0]:
        raise ValueError("Metric mask length must match reference and generated epoch dimensions.")
    if not mask.any():
        return None
    filtered_baseline = None
    if baseline is not None:
        if baseline.shape[0] != mask.shape[0]:
            raise ValueError("Metric mask length must match baseline epoch dimension.")
        filtered_baseline = baseline[mask]
    return reference[mask], generated[mask], filtered_baseline


def _metric_rows(metrics: dict[str, t.Any]) -> list[dict[str, t.Any]]:
    rows: list[dict[str, t.Any]] = []

    def walk(prefix: list[str], value: t.Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk([*prefix, str(key)], item)
            return
        family = prefix[0] if len(prefix) > 0 else ""
        modality = prefix[1] if len(prefix) > 2 else "all"
        metric = ".".join(prefix[2:]) if len(prefix) > 2 else ".".join(prefix[1:])
        clean_value = _jsonable(value)
        rows.append(
            {
                "family": family,
                "modality": modality,
                "metric": metric,
                "value": (
                    clean_value
                    if isinstance(clean_value, (int, float, str))
                    else json.dumps(clean_value, sort_keys=True, allow_nan=False)
                ),
            }
        )

    walk([], metrics)
    return rows


def _write_metrics(output_dir: Path, payload: dict[str, t.Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    rows = _metric_rows(payload["metrics"])
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "modality", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _load_event_groups(path: Path | None) -> dict[str, dict[str, t.Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("events JSON must contain an object.")
    events = payload.get("events", payload)
    if not isinstance(events, dict):
        raise ValueError("events JSON must define an events object.")
    return events


def run_evaluation(args: argparse.Namespace) -> Path:
    from sleep2wave.data.modalities import MODALITY_SPECS
    from sleep2wave.evaluation.downstream_hooks import load_downstream_metrics
    from sleep2wave.evaluation.efficiency import summarize_generation_efficiency
    from sleep2wave.evaluation.event_metrics import compute_event_metric_groups
    from sleep2wave.evaluation.feature_metrics import compute_feature_metrics
    from sleep2wave.evaluation.waveform_metrics import compute_waveform_metrics
    from sleep2wave.generative.config import load_sleep2wave_config

    config = load_sleep2wave_config(args.config)
    if config.stage != "evaluation" or config.evaluation is None or config.export is None:
        raise ValueError("sleep2wave.evaluate_generation requires stage=evaluation config.")

    evaluation = config.evaluation
    if args.generated_dir is not None:
        evaluation = replace(evaluation, generated_dir=str(args.generated_dir))
    if args.reference_npz is not None:
        evaluation = replace(evaluation, reference_npz=str(args.reference_npz))
    if args.baseline_npz is not None:
        evaluation = replace(evaluation, baseline_npz=str(args.baseline_npz))
    if args.events_json is not None:
        evaluation = replace(evaluation, events_json=str(args.events_json))
    if args.downstream_metrics_json is not None:
        evaluation = replace(evaluation, downstream_metrics_json=str(args.downstream_metrics_json))

    metric_families = set(evaluation.metric_families)
    if {"waveform", "feature"} & metric_families and evaluation.reference_npz is None:
        raise ValueError("evaluation.reference_npz is required for waveform or feature metrics.")

    generated_dir = Path(evaluation.generated_dir)
    artifact_paths = _require_artifact_dir(generated_dir)
    reference_path = Path(evaluation.reference_npz) if evaluation.reference_npz is not None else None
    baseline_path = Path(evaluation.baseline_npz) if evaluation.baseline_npz is not None else None
    if reference_path is not None:
        _require_file(reference_path, description="Reference NPZ")
    if baseline_path is not None:
        _require_file(baseline_path, description="Baseline NPZ")

    manifest = json.loads(artifact_paths["manifest"].read_text())
    generated_npz = np.load(artifact_paths["generated"])
    uncertainty_npz = np.load(artifact_paths["uncertainty"])
    masks_npz = np.load(artifact_paths["masks"])
    reference_npz = np.load(reference_path) if reference_path is not None else None
    baseline_npz = np.load(baseline_path) if baseline_path is not None else None

    metrics: dict[str, t.Any] = {}
    generated_modalities = [
        modality
        for modality in config.modalities.all
        if _load_generated_mean(generated_npz, uncertainty_npz, modality) is not None
    ]

    if "waveform" in metric_families:
        if reference_npz is None:
            raise ValueError("evaluation.reference_npz is required for waveform metrics.")
        waveform_metrics: dict[str, dict[str, float]] = {}
        for modality in generated_modalities:
            reference = _load_reference(reference_npz, modality)
            generated = _load_generated_mean(generated_npz, uncertainty_npz, modality)
            if reference is None or generated is None:
                continue
            masked = _apply_metric_epoch_mask(
                reference,
                generated,
                _load_baseline(baseline_npz, modality),
                _load_metric_epoch_mask(masks_npz, modality, generated.shape[0]),
            )
            if masked is None:
                continue
            reference, generated, baseline = masked
            waveform_metrics[modality] = compute_waveform_metrics(
                reference,
                generated,
                baseline=baseline,
                max_shift_frames=evaluation.max_shift_frames,
            )
        metrics["waveform"] = waveform_metrics

    if "feature" in metric_families:
        if reference_npz is None:
            raise ValueError("evaluation.reference_npz is required for feature metrics.")
        feature_metrics: dict[str, dict[str, float]] = {}
        for modality in generated_modalities:
            reference = _load_reference(reference_npz, modality)
            generated = _load_generated_mean(generated_npz, uncertainty_npz, modality)
            if reference is None or generated is None:
                continue
            masked = _apply_metric_epoch_mask(
                reference,
                generated,
                None,
                _load_metric_epoch_mask(masks_npz, modality, generated.shape[0]),
            )
            if masked is None:
                continue
            reference, generated, _baseline = masked
            modality_metrics = compute_feature_metrics(
                modality,
                reference,
                generated,
                sample_rate_hz=MODALITY_SPECS[modality].sample_rate_hz,
            )
            if modality_metrics:
                feature_metrics[modality] = modality_metrics
        metrics["feature"] = feature_metrics

    if "event" in metric_families:
        event_groups = _load_event_groups(Path(evaluation.events_json) if evaluation.events_json is not None else None)
        metrics["event"] = (
            compute_event_metric_groups(
                event_groups,
                iou_threshold=evaluation.event_iou_threshold,
            )
            if event_groups
            else {}
        )

    if "efficiency" in metric_families:
        metrics["efficiency"] = summarize_generation_efficiency(manifest, generated_npz, uncertainty_npz)

    if "downstream" in metric_families:
        metrics["downstream"] = load_downstream_metrics(evaluation.downstream_metrics_json)

    output_dir = args.output_dir or Path(config.export.output_dir)
    payload = {
        "schema_version": 1,
        "artifact_type": "sleep2wave_generation_evaluation",
        "generated_dir": str(generated_dir),
        "reference_npz": str(reference_path) if reference_path is not None else None,
        "metrics": metrics,
    }
    _write_metrics(Path(output_dir), payload)
    return Path(output_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
