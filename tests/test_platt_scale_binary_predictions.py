import json

import pandas as pd

from utils.platt_scale_binary_predictions import calibrate_predictions, load_binary_predictions


def test_load_binary_predictions_accepts_scalar_prediction_columns(tmp_path):
    csv_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {"path": "a.npz", "groundtruth": 0, "prob_1": 0.2},
            {"path": "b.npz", "groundtruth": "1", "prob_1": "0.8"},
        ]
    ).to_csv(csv_path, index=False)

    _, y_true, prob = load_binary_predictions(csv_path)

    assert y_true.tolist() == [0, 1]
    assert prob.tolist() == [0.2, 0.8]


def test_platt_calibration_writes_metrics_predictions_and_manifest(tmp_path):
    calibration_csv = tmp_path / "gz_predictions.csv"
    eval_csv = tmp_path / "mgh_predictions.csv"
    output_dir = tmp_path / "platt"
    pd.DataFrame(
        [
            {"path": "gz-control-1.npz", "groundtruth": 0, "prob_1": 0.10},
            {"path": "gz-control-2.npz", "groundtruth": 0, "prob_1": 0.35},
            {"path": "gz-rbd-1.npz", "groundtruth": 1, "prob_1": 0.45},
            {"path": "gz-rbd-2.npz", "groundtruth": 1, "prob_1": 0.80},
        ]
    ).to_csv(calibration_csv, index=False)
    pd.DataFrame(
        [
            {"path": "mgh-control.npz", "groundtruth": 0, "prob_1": 0.25},
            {"path": "mgh-rbd.npz", "groundtruth": 1, "prob_1": 0.70},
        ]
    ).to_csv(eval_csv, index=False)

    manifest = calibrate_predictions(
        calibration_csv,
        [eval_csv],
        output_dir=output_dir,
        calibration_name="GZ",
        eval_names=["MGH"],
    )

    assert manifest["method"] == "platt_scaling"
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "calibrator.json").exists()
    assert (output_dir / "predictions__GZ__platt.csv").exists()
    assert (output_dir / "predictions__MGH__platt.csv").exists()
    assert (output_dir / "confusion_matrix__GZ__raw.csv").exists()
    assert (output_dir / "confusion_matrix__MGH__platt.csv").exists()

    saved_manifest = json.loads((output_dir / "calibrator.json").read_text(encoding="utf-8"))
    assert saved_manifest["calibration_name"] == "GZ"
    assert saved_manifest["eval_names"] == ["MGH"]

    metrics = pd.read_csv(output_dir / "metrics.csv")
    assert set(metrics["dataset_name"]) == {"GZ", "MGH"}
    assert set(metrics["score"]) == {"raw", "platt"}
    assert {"balanced_accuracy", "specificity", "recall", "auroc"}.issubset(metrics.columns)

    calibrated = pd.read_csv(output_dir / "predictions__MGH__platt.csv")
    assert {"raw_prob_1", "raw_logit_1", "platt_prob_1", "platt_prediction"}.issubset(calibrated.columns)


def test_platt_calibration_supports_calibration_only_output(tmp_path):
    calibration_csv = tmp_path / "gz_predictions.csv"
    output_dir = tmp_path / "platt"
    pd.DataFrame(
        [
            {"path": "gz-control-1.npz", "groundtruth": 0, "prob_1": 0.15},
            {"path": "gz-control-2.npz", "groundtruth": 0, "prob_1": 0.30},
            {"path": "gz-rbd-1.npz", "groundtruth": 1, "prob_1": 0.55},
            {"path": "gz-rbd-2.npz", "groundtruth": 1, "prob_1": 0.85},
        ]
    ).to_csv(calibration_csv, index=False)

    manifest = calibrate_predictions(
        calibration_csv,
        [],
        output_dir=output_dir,
        calibration_name="GZ",
    )

    assert manifest["eval_csvs"] == []
    assert manifest["eval_names"] == []
    assert (output_dir / "predictions__GZ__platt.csv").exists()
    assert (output_dir / "confusion_matrix__GZ__platt.csv").exists()

    metrics = pd.read_csv(output_dir / "metrics.csv")
    assert metrics["dataset_name"].tolist() == ["GZ", "GZ"]
    assert metrics["score"].tolist() == ["raw", "platt"]
