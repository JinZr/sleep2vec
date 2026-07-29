from __future__ import annotations

import importlib

import numpy as np
import pytest

METRIC_PACKAGES = ("sleep2vec", "sleep2vec2", "sleep2expert")
PROBABILITY_METRIC_KEYS = {"auprc", "brier", "ece"}


def _two_class_probabilities(positive_probability) -> np.ndarray:
    positive_probability = np.asarray(positive_probability, dtype=np.float32)
    return np.column_stack((1.0 - positive_probability, positive_probability))


@pytest.mark.parametrize("package_name", METRIC_PACKAGES)
def test_binary_probability_metrics_match_episode_weighted_goldens(package_name: str):
    metrics_mod = importlib.import_module(f"{package_name}.metrics")
    metrics = metrics_mod.compute_binary_probability_metrics(
        np.array([0, 0, 1, 0], dtype=np.int64),
        _two_class_probabilities([0.05, 0.05, 0.05, 0.95]),
        from_logits=False,
    )

    assert set(metrics) == PROBABILITY_METRIC_KEYS
    assert metrics["auprc"] == pytest.approx(0.25)
    assert metrics["brier"] == pytest.approx(0.4525)
    assert metrics["ece"] == pytest.approx(0.45)


@pytest.mark.parametrize("package_name", METRIC_PACKAGES)
def test_binary_probability_metrics_use_fixed_left_closed_bins_with_one_in_last_bin(package_name: str):
    metrics_mod = importlib.import_module(f"{package_name}.metrics")
    metrics = metrics_mod.compute_binary_probability_metrics(
        np.array([1, 0, 1, 0], dtype=np.int64),
        _two_class_probabilities([0.1, 0.15, 0.9, 1.0]),
        from_logits=False,
    )

    assert metrics["auprc"] == pytest.approx(0.5)
    assert metrics["brier"] == pytest.approx(0.460625)
    assert metrics["ece"] == pytest.approx(0.4125)


@pytest.mark.parametrize("package_name", METRIC_PACKAGES)
def test_binary_probability_metrics_match_for_two_logits_and_softmax_probabilities(package_name: str):
    metrics_mod = importlib.import_module(f"{package_name}.metrics")
    logits = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.25, 0.75],
            [0.75, 0.25],
        ],
        dtype=np.float32,
    )
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted, dtype=np.float64)
    probabilities = (exponentials / exponentials.sum(axis=1, keepdims=True)).astype(np.float32)
    targets = np.array([0, 1, 1, 0], dtype=np.int64)

    from_logits = metrics_mod.compute_binary_probability_metrics(targets, logits, from_logits=True)
    from_probabilities = metrics_mod.compute_binary_probability_metrics(targets, probabilities, from_logits=False)

    for key in PROBABILITY_METRIC_KEYS:
        assert from_logits[key] == pytest.approx(from_probabilities[key])


@pytest.mark.parametrize("package_name", METRIC_PACKAGES)
def test_prediction_rows_reject_conflicting_labels_before_duplicate_window_removal(package_name: str):
    inference_mod = importlib.import_module(f"{package_name}.sleep2vec_inference")
    records = [
        {
            "path": "episode.npz",
            "token_start": 0,
            "kind": "classification",
            "groundtruth": label,
            "probabilities": [0.8, 0.2] if label == 0 else [0.2, 0.8],
            "logits": [1.0, 0.0] if label == 0 else [0.0, 1.0],
            "prediction": label,
            "is_sequence": False,
        }
        for label in (0, 1)
    ]

    with pytest.raises(ValueError, match="Classification labels differ for duplicate window"):
        inference_mod.build_prediction_rows(records)


@pytest.mark.parametrize("package_name", METRIC_PACKAGES)
@pytest.mark.parametrize("target", [0, 1])
def test_binary_probability_metrics_keep_calibration_scores_for_single_class(package_name: str, target: int):
    metrics_mod = importlib.import_module(f"{package_name}.metrics")
    positive_probability = [0.1, 0.2] if target == 0 else [0.8, 0.9]
    metrics = metrics_mod.compute_binary_probability_metrics(
        np.full(2, target, dtype=np.int64),
        _two_class_probabilities(positive_probability),
        from_logits=False,
    )

    assert np.isnan(metrics["auprc"])
    assert metrics["brier"] == pytest.approx(0.025)
    assert metrics["ece"] == pytest.approx(0.15)


@pytest.mark.parametrize("package_name", METRIC_PACKAGES)
@pytest.mark.parametrize("task_kind", ["multiclass", "regression", "multilabel"])
def test_downstream_probability_metrics_stay_outside_nonbinary_tasks(package_name: str, task_kind: str):
    metrics_mod = importlib.import_module(f"{package_name}.metrics")

    if task_kind == "multiclass":
        metrics = metrics_mod.compute_downstream_metrics(
            np.array([0, 1, 2], dtype=np.int64),
            np.array(
                [
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8],
                ],
                dtype=np.float32,
            ),
            is_classification=True,
            output_dim=3,
        )
    elif task_kind == "multilabel":
        metrics = metrics_mod.compute_downstream_metrics(
            np.array([0, 1], dtype=np.int64),
            np.array([0.1, 0.9], dtype=np.float32),
            is_classification=True,
            is_multilabel=True,
            output_dim=2,
        )
    else:
        metrics = metrics_mod.compute_downstream_metrics(
            np.array([0.0, 1.0, 2.0], dtype=np.float32),
            np.array([0.1, 0.9, 2.1], dtype=np.float32),
            is_classification=False,
            output_dim=1,
        )

    assert PROBABILITY_METRIC_KEYS.isdisjoint(metrics)
