from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from sleep2vec.metrics import compute_ahi_pointwise_metrics, compute_downstream_metrics
from sleep2vec.sleep2vec_finetuning import Sleep2vecFinetuning
from sleep2vec.utils import _build_finetune_loader

PREDICTION_EXPORT_PACKAGES = ("sleep2vec", "sleep2vec2", "sleep2expert")


class _DummyDataset:
    last_init_kwargs = None
    last_device = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = kwargs

    def dataloader(self, device="cpu"):
        type(self).last_device = device
        return {"device": device}


class _DummyDatasetWithSamples:
    samples = []
    last_device = None

    def __init__(self, **kwargs):
        self.data = type(self).samples

    def dataloader(self, device="cpu"):
        type(self).last_device = device
        return {"device": device}


def _seq_args(
    label_name: str,
    *,
    label_source_name: str,
    output_dim: int,
    is_multilabel: bool = False,
    auxiliary_label_source_names: list[str] | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        label_name=label_name,
        label_source_name=label_source_name,
        auxiliary_label_source_names=auxiliary_label_source_names or [],
        data_channel_names=["eeg"],
        channel_input_dims={"eeg": 4},
        finetune_preset_path=None,
        finetune_data_index=Path("index.csv"),
        max_tokens=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        is_classification=True,
        output_dim=output_dim,
        is_multilabel=is_multilabel,
    )


def _metadata_args(label_name: str, *, is_classification: bool) -> argparse.Namespace:
    return argparse.Namespace(
        label_name=label_name,
        data_channel_names=["eeg"],
        channel_input_dims={"eeg": 4},
        finetune_preset_path=Path("preset.pkl"),
        finetune_data_index=None,
        max_tokens=2,
        batch_size=1,
        num_workers=0,
        device="cpu",
        is_classification=is_classification,
        output_dim=2 if is_classification else 1,
    )


def test_build_finetune_loader_uses_stage5_tokens_for_stage3(monkeypatch):
    monkeypatch.setattr("sleep2vec.utils.PSGPretrainDataset", _DummyDataset)
    args = _seq_args("stage3", label_source_name="stage5", output_dim=3)

    loader = _build_finetune_loader(
        args,
        split=["train"],
        sources=["demo"],
        shuffle=False,
        is_train_set=False,
    )

    assert loader == {"device": "cpu"}
    assert _DummyDataset.last_device == "cpu"
    assert _DummyDataset.last_init_kwargs["channel_names"] == ["eeg", "stage5"]
    assert _DummyDataset.last_init_kwargs["meta_data_names"] == []
    assert _DummyDataset.last_init_kwargs["meta_data_regression_names"] == []


@pytest.mark.parametrize(
    ("label_name", "is_classification", "metadata"),
    [
        ("age", False, {}),
        ("age", False, {"age": float("nan")}),
        ("sex", True, {}),
        ("sex", True, {"sex": float("nan")}),
    ],
)
def test_build_finetune_loader_rejects_missing_builtin_metadata_labels(
    monkeypatch,
    label_name: str,
    is_classification: bool,
    metadata: dict,
):
    _DummyDatasetWithSamples.samples = [argparse.Namespace(metadata=metadata)]
    monkeypatch.setattr("sleep2vec.utils.PSGPretrainDataset", _DummyDatasetWithSamples)
    args = _metadata_args(label_name, is_classification=is_classification)

    with pytest.raises(ValueError, match=f"invalid or missing '{label_name}' labels"):
        _build_finetune_loader(
            args,
            split=["test"],
            sources=[],
            shuffle=False,
            is_train_set=False,
        )


def test_build_finetune_loader_passes_weighted_random_sampler_for_train_metadata(monkeypatch):
    monkeypatch.setattr("sleep2vec.utils.PSGPretrainDataset", _DummyDataset)
    args = _metadata_args("src_isDep", is_classification=True)
    args.weighted_random_sampler = True

    loader = _build_finetune_loader(
        args,
        split=["train"],
        sources=["demo"],
        shuffle=True,
        is_train_set=True,
    )

    assert loader == {"device": "cpu"}
    assert _DummyDataset.last_init_kwargs["meta_data_names"] == ["src_isDep"]
    assert _DummyDataset.last_init_kwargs["weighted_random_sampler"] is True
    assert _DummyDataset.last_init_kwargs["weighted_random_sampler_target"] == "src_isDep"


def test_build_finetune_loader_keeps_weighted_random_sampler_train_only(monkeypatch):
    monkeypatch.setattr("sleep2vec.utils.PSGPretrainDataset", _DummyDataset)
    args = _metadata_args("src_isDep", is_classification=True)
    args.weighted_random_sampler = True

    _build_finetune_loader(
        args,
        split=["val"],
        sources=["demo"],
        shuffle=False,
        is_train_set=False,
    )

    assert _DummyDataset.last_init_kwargs["weighted_random_sampler"] is False
    assert _DummyDataset.last_init_kwargs["weighted_random_sampler_target"] is None


@pytest.mark.parametrize(
    "args",
    [
        _seq_args("stage5", label_source_name="stage5", output_dim=5),
        _seq_args(
            "ahi",
            label_source_name="ahi",
            output_dim=30,
            is_multilabel=True,
            auxiliary_label_source_names=["stage5"],
        ),
    ],
)
def test_build_finetune_loader_allows_sequence_tasks_without_age_or_sex(monkeypatch, args):
    _DummyDatasetWithSamples.samples = [argparse.Namespace(metadata={})]
    monkeypatch.setattr("sleep2vec.utils.PSGPretrainDataset", _DummyDatasetWithSamples)

    loader = _build_finetune_loader(
        args,
        split=["test"],
        sources=[],
        shuffle=False,
        is_train_set=False,
    )

    assert loader == {"device": "cpu"}


def test_build_finetune_loader_uses_ahi_tokens_for_ahi(monkeypatch):
    monkeypatch.setattr("sleep2vec.utils.PSGPretrainDataset", _DummyDataset)
    args = _seq_args(
        "ahi",
        label_source_name="ahi",
        output_dim=30,
        is_multilabel=True,
        auxiliary_label_source_names=["stage5"],
    )

    loader = _build_finetune_loader(
        args,
        split=["train"],
        sources=["demo"],
        shuffle=False,
        is_train_set=False,
    )

    assert loader == {"device": "cpu"}
    assert _DummyDataset.last_device == "cpu"
    assert _DummyDataset.last_init_kwargs["channel_names"] == ["eeg", "ahi", "stage5"]
    assert _DummyDataset.last_init_kwargs["meta_data_names"] == ["ahi", "tst"]
    assert _DummyDataset.last_init_kwargs["meta_data_regression_names"] == ["ahi", "tst"]


def test_get_targets_remaps_stage_labels_and_preserves_ignore_index():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(
        is_seq=True,
        label_name="stage4",
        label_source_name="stage5",
        is_multilabel=False,
        device="cpu",
    )
    batch = {
        "tokens": {
            "stage5": torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, -1.0]]),
        }
    }

    labels = module._get_targets(batch)

    assert torch.equal(labels, torch.tensor([[0.0, 1.0, 1.0, 2.0, 3.0, -1.0]]))


def test_get_targets_returns_raw_ahi_labels():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    module.args = argparse.Namespace(
        is_seq=True,
        label_name="ahi",
        label_source_name="ahi",
        is_multilabel=True,
        device="cpu",
    )
    batch = {
        "tokens": {
            "ahi": torch.tensor([[[0.0, 1.0], [1.0, -1.0]]]),
        }
    }

    labels = module._get_targets(batch)

    assert torch.equal(labels, torch.tensor([[[0.0, 1.0], [1.0, -1.0]]]))


def test_extract_ahi_event_records_keeps_sample_boundaries_and_scalar_summary():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    logits = torch.tensor([[[0.0, 0.0], [2.0, -2.0]]], dtype=torch.float32)
    batch = {
        "tokens": {
            "ahi": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
            "stage5": torch.tensor([[0.0, 2.0]], dtype=torch.float32),
        },
        "metadata": {
            "ahi": torch.tensor([16.5], dtype=torch.float32),
            "tst": torch.tensor([5.25], dtype=torch.float32),
            "path": ["sample-a.npz"],
        },
        "token_start": torch.tensor([0], dtype=torch.long),
    }

    records = module._extract_ahi_event_records(batch, logits)

    assert len(records) == 1
    assert records[0]["truth"].tolist() == [0, 1, 1, 0]
    assert records[0]["score"].shape == (4,)
    assert records[0]["true_ahi"] == 16.5
    assert records[0]["tst_hours"] == 5.25
    assert records[0]["stage5"].tolist() == [0, 2]
    assert records[0]["path"] == "sample-a.npz"
    assert records[0]["token_start"] == 0


def test_extract_ahi_event_records_keeps_stage5_tokens_with_second_level_mask():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    logits = torch.tensor([[[0.0, 0.0], [2.0, -2.0]]], dtype=torch.float32)
    batch = {
        "tokens": {
            "ahi": torch.tensor([[[0.0, 1.0], [-1.0, -1.0]]], dtype=torch.float32),
            "stage5": torch.tensor([[0.0, 2.0]], dtype=torch.float32),
        },
        "metadata": {
            "ahi": torch.tensor([8.0], dtype=torch.float32),
            "tst": torch.tensor([4.0], dtype=torch.float32),
            "path": ["sample-b.npz"],
        },
        "token_start": torch.tensor([2], dtype=torch.long),
    }

    records = module._extract_ahi_event_records(batch, logits)

    assert records[0]["stage5"].tolist() == [0, 2]
    assert records[0]["second_valid_mask"].tolist() == [True, True, False, False]


def test_extract_ahi_event_records_keeps_stage5_aligned_for_partially_masked_token():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    logits = torch.tensor([[[0.0, 0.0], [2.0, -2.0]]], dtype=torch.float32)
    batch = {
        "tokens": {
            "ahi": torch.tensor([[[0.0, -1.0], [1.0, 0.0]]], dtype=torch.float32),
            "stage5": torch.tensor([[0.0, 2.0]], dtype=torch.float32),
        },
        "metadata": {
            "ahi": torch.tensor([8.0], dtype=torch.float32),
            "tst": torch.tensor([4.0], dtype=torch.float32),
            "path": ["sample-c.npz"],
        },
        "token_start": torch.tensor([4], dtype=torch.long),
    }

    records = module._extract_ahi_event_records(batch, logits)

    assert records[0]["truth"].tolist() == [0, 1, 0]
    assert records[0]["score"].shape == (3,)
    assert records[0]["stage5"].tolist() == [0, 2]
    assert records[0]["second_valid_mask"].tolist() == [True, False, True, True]


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_build_scalar_classification_prediction_row_averages_probabilities(package_name: str):
    inference_mod = importlib.import_module(f"{package_name}.sleep2vec_inference")
    records = [
        {
            "path": "sample.npz",
            "token_start": 0,
            "kind": "classification",
            "groundtruth": 1,
            "probabilities": [0.2, 0.8],
            "logits": [0.0, 2.0],
            "prediction": 1,
            "is_sequence": False,
        },
        {
            "path": "sample.npz",
            "token_start": 5,
            "kind": "classification",
            "groundtruth": 1,
            "probabilities": [0.6, 0.4],
            "logits": [1.0, 0.0],
            "prediction": 0,
            "is_sequence": False,
        },
    ]

    rows = inference_mod.build_prediction_rows(records)

    row = rows[0]
    assert row["path"] == "sample.npz"
    assert row["groundtruth"] == 1
    assert row["prediction"] == 1
    assert row["n_predictions"] == 2
    assert row["n_windows"] == 2
    assert row["token_starts"] == [0, 5]
    assert row["prob_0"] == pytest.approx(0.4)
    assert row["prob_1"] == pytest.approx(0.6)
    assert row["logit_0"] == pytest.approx(0.5)
    assert row["logit_1"] == pytest.approx(1.0)
    assert row["logit"] == pytest.approx(0.5)


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_extract_scalar_classification_prediction_records_preserves_logits(package_name: str):
    inference_mod = importlib.import_module(f"{package_name}.sleep2vec_inference")
    args = argparse.Namespace(is_multilabel=False, is_classification=True)
    batch = {
        "metadata": {"path": ["a.npz", "b.npz"]},
        "token_start": torch.tensor([0, 5]),
    }
    logits = torch.tensor([[-1.0, 2.0], [3.0, 0.0]])
    targets = torch.tensor([1, 0])

    records = inference_mod.extract_prediction_records(args, batch, logits, targets)

    assert records[0]["logits"] == pytest.approx([-1.0, 2.0])
    assert records[0]["probabilities"] == pytest.approx(torch.softmax(logits[0], dim=-1).tolist())
    assert records[1]["logits"] == pytest.approx([3.0, 0.0])
    assert records[1]["probabilities"] == pytest.approx(torch.softmax(logits[1], dim=-1).tolist())


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_build_scalar_regression_prediction_row_averages_windows(package_name: str):
    inference_mod = importlib.import_module(f"{package_name}.sleep2vec_inference")
    records = [
        {
            "path": "sample.npz",
            "token_start": 0,
            "kind": "regression",
            "groundtruth": 60.0,
            "prediction": 61.0,
            "is_sequence": False,
        },
        {
            "path": "sample.npz",
            "token_start": 5,
            "kind": "regression",
            "groundtruth": 62.0,
            "prediction": 63.0,
            "is_sequence": False,
        },
    ]

    rows = inference_mod.build_prediction_rows(records)

    row = rows[0]
    assert row["path"] == "sample.npz"
    assert row["groundtruth"] == pytest.approx(61.0)
    assert row["prediction"] == pytest.approx(62.0)
    assert row["n_predictions"] == 2
    assert row["n_windows"] == 2
    assert row["token_starts"] == [0, 5]


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_build_sequence_classification_prediction_row_concatenates_by_token_start(package_name: str):
    inference_mod = importlib.import_module(f"{package_name}.sleep2vec_inference")
    records = [
        {
            "path": "night.npz",
            "token_start": 2,
            "kind": "classification",
            "groundtruth": [2],
            "probabilities": [[0.1, 0.2, 0.7]],
            "logits": [[0.0, 1.0, 2.0]],
            "prediction": [2],
            "is_sequence": True,
        },
        {
            "path": "night.npz",
            "token_start": 0,
            "kind": "classification",
            "groundtruth": [0, 1],
            "probabilities": [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1]],
            "logits": [[2.0, 0.0, -1.0], [0.0, 2.0, 0.0]],
            "prediction": [0, 1],
            "is_sequence": True,
        },
        {
            "path": "night.npz",
            "token_start": 0,
            "kind": "classification",
            "groundtruth": [2, 2],
            "probabilities": [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            "logits": [[-1.0, -1.0, 3.0], [-1.0, -1.0, 3.0]],
            "prediction": [2, 2],
            "is_sequence": True,
        },
    ]

    rows = inference_mod.build_prediction_rows(records)

    assert rows[0]["path"] == "night.npz"
    assert rows[0]["groundtruth"] == [0, 1, 2]
    assert rows[0]["prediction"] == [0, 1, 2]
    assert rows[0]["n_predictions"] == 3
    assert rows[0]["n_windows"] == 2
    assert rows[0]["token_starts"] == [0, 2]
    assert rows[0]["prob_0"] == pytest.approx([0.9, 0.1, 0.1])
    assert rows[0]["prob_1"] == pytest.approx([0.1, 0.8, 0.2])
    assert rows[0]["prob_2"] == pytest.approx([0.0, 0.1, 0.7])
    assert rows[0]["logit_0"] == pytest.approx([2.0, 0.0, 0.0])
    assert rows[0]["logit_1"] == pytest.approx([0.0, 2.0, 1.0])
    assert rows[0]["logit_2"] == pytest.approx([-1.0, 0.0, 2.0])


@pytest.mark.parametrize("package_name", PREDICTION_EXPORT_PACKAGES)
def test_build_ahi_prediction_row_includes_threshold_and_summary(package_name: str):
    inference_mod = importlib.import_module(f"{package_name}.sleep2vec_inference")
    record = {
        "path": "night.npz",
        "token_start": 0,
        "truth": np.array([0] * 5 + [1] * 12 + [0] * 13, dtype=np.int64),
        "score": np.array([0.1] * 5 + [0.9] * 12 + [0.1] * 13, dtype=np.float32),
        "true_ahi": 10.0,
        "tst_hours": 3.0,
        "stage5": np.array([2], dtype=np.int64),
        "second_valid_mask": np.array([True] * 30),
    }

    rows = inference_mod.build_ahi_prediction_rows([record], threshold=0.5)

    assert rows[0]["path"] == "night.npz"
    assert rows[0]["groundtruth"] == record["truth"].tolist()
    assert rows[0]["prediction"] == ([0] * 5 + [1] * 12 + [0] * 13)
    assert rows[0]["prob"] == pytest.approx(record["score"].tolist())
    assert rows[0]["ahi_threshold"] == 0.5
    assert rows[0]["true_ahi"] == 10.0
    assert rows[0]["pred_ahi"] == pytest.approx(1 / 3.0)
    assert rows[0]["tst_hours"] == 3.0


def _config_package_name(path: Path) -> str:
    rel_path = path.relative_to(Path(__file__).resolve().parents[1] / "configs")
    if rel_path.parts and rel_path.parts[0] in {"sleep2vec2", "sleep2expert"}:
        return rel_path.parts[0]
    return "sleep2vec"


def _prediction_export_supports_task(task) -> bool:
    if not task.is_seq:
        return task.type in {"classification", "regression"}
    if task.type != "classification":
        return False
    return int(task.output_dim) in {3, 4, 5, 30}


def test_prediction_export_supports_all_finetune_recipe_task_families():
    repo_root = Path(__file__).resolve().parents[1]
    recipe_paths: list[Path] = []
    for path in sorted((repo_root / "configs").rglob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        finetune = data.get("finetune")
        if isinstance(finetune, dict) and isinstance(finetune.get("task"), dict):
            recipe_paths.append(path)

    assert recipe_paths

    for path in recipe_paths:
        package_name = _config_package_name(path)
        bundle = importlib.import_module(f"{package_name}.config").load_finetune_config(path)
        assert _prediction_export_supports_task(bundle.finetune.task), str(path.relative_to(repo_root))


def test_compute_loss_ignores_ahi_padding():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    torch.nn.Module.__init__(module)
    module.args = argparse.Namespace(
        is_seq=True,
        is_classification=True,
        label_name="ahi",
        label_source_name="ahi",
        is_multilabel=True,
        device="cpu",
    )
    module._multilabel_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    logits = torch.zeros((1, 2, 3), dtype=torch.float32)
    batch = {
        "tokens": {
            "ahi": torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, -1.0]]]),
        }
    }

    loss, valid_count = module._compute_loss(logits, batch)

    assert valid_count == 5
    assert torch.isclose(loss, torch.tensor(math.log(2.0), dtype=torch.float32))


def test_compute_loss_applies_class_weights_for_binary_classification():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    torch.nn.Module.__init__(module)
    module.args = argparse.Namespace(
        is_seq=False,
        is_classification=True,
        is_multilabel=False,
        label_name="src_isDep",
        device="cpu",
    )
    class_weights = torch.tensor([1.0, 3.0], dtype=torch.float32)
    module._classification_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
    logits = torch.tensor([[3.0, 0.0], [3.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1, -1], dtype=torch.long)
    batch = {"metadata": {"src_isDep": targets}}

    loss, valid_count = module._compute_loss(logits, batch)
    expected = torch.nn.functional.cross_entropy(logits[:2], targets[:2], weight=class_weights)
    unweighted = torch.nn.functional.cross_entropy(logits[:2], targets[:2])

    assert valid_count == 2
    assert torch.isclose(loss, expected)
    assert not torch.isclose(loss, unweighted)


def test_compute_loss_applies_ahi_pos_weight_and_ignores_padding():
    module = Sleep2vecFinetuning.__new__(Sleep2vecFinetuning)
    torch.nn.Module.__init__(module)
    module.args = argparse.Namespace(
        is_seq=True,
        is_classification=True,
        label_name="ahi",
        label_source_name="ahi",
        is_multilabel=True,
        device="cpu",
    )
    pos_weight = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    module._multilabel_loss = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    logits = torch.zeros((1, 2, 3), dtype=torch.float32)
    targets = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, -1.0]]], dtype=torch.float32)
    batch = {"tokens": {"ahi": targets}}

    loss, valid_count = module._compute_loss(logits, batch)
    valid_mask = targets != -1.0
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )[valid_mask].mean()
    unweighted = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")[
        valid_mask
    ].mean()

    assert valid_count == 5
    assert torch.isclose(loss, expected)
    assert not torch.isclose(loss, unweighted)


def test_compute_downstream_metrics_reports_stage_specific_scores_for_stage3_and_stage4():
    for output_dim, stage_names in (
        (3, ["W", "NREM", "REM"]),
        (4, ["W", "N1N2", "N3", "REM"]),
    ):
        gts = np.arange(output_dim)
        preds = np.eye(output_dim, dtype=np.float32)

        metrics = compute_downstream_metrics(
            gts,
            preds,
            is_classification=True,
            output_dim=output_dim,
            stage_names=stage_names,
        )

        for stage_name in stage_names:
            assert metrics[f"f1_{stage_name}"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["specificity"] == 1.0
        assert metrics["sens"] == 1.0
        assert metrics["spec"] == 1.0


def test_compute_downstream_metrics_reports_multiclass_recall_and_specificity():
    gts = np.array([0, 0, 1, 1, 2, 2])
    preds = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.6, 0.2, 0.2],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float32,
    )

    metrics = compute_downstream_metrics(
        gts,
        preds,
        is_classification=True,
        output_dim=3,
    )

    assert metrics["recall"] == pytest.approx(2.0 / 3.0)
    assert metrics["specificity"] == pytest.approx((0.75 + 0.75 + 1.0) / 3.0)
    assert metrics["accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["cohen_kappa"] == pytest.approx(0.5)
    assert metrics["f1_weighted"] == pytest.approx(59.0 / 90.0)
    assert metrics["f1_macro"] == pytest.approx(59.0 / 90.0)


def test_compute_downstream_metrics_reports_binary_recall_and_specificity():
    metrics = compute_downstream_metrics(
        np.array([0, 0, 1, 1]),
        np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        ),
        is_classification=True,
        output_dim=2,
    )

    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5


def test_compute_downstream_metrics_preserves_macro_aliases_for_two_class_stage_names():
    metrics = compute_downstream_metrics(
        np.array([0, 0, 0, 1]),
        np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        ),
        is_classification=True,
        output_dim=2,
        stage_names=["Wake", "Sleep"],
    )

    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == pytest.approx(1.0 / 3.0)
    assert metrics["sens"] == pytest.approx(2.0 / 3.0)
    assert metrics["spec"] == pytest.approx(2.0 / 3.0)


def test_compute_downstream_metrics_reports_binary_scores_for_ahi():
    metrics = compute_downstream_metrics(
        np.array([0, 1, 1, 0]),
        np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float32),
        is_classification=True,
        is_multilabel=True,
        output_dim=30,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_ahi_pointwise_metrics_uses_namespaced_keys():
    metrics = compute_ahi_pointwise_metrics(
        np.array([0, 1, 1, 0]),
        np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float32),
    )

    assert metrics["ahi_pointwise_accuracy"] == 1.0
    assert metrics["ahi_pointwise_precision"] == 1.0
    assert metrics["ahi_pointwise_recall"] == 1.0
    assert metrics["ahi_pointwise_specificity"] == 1.0
    assert metrics["ahi_pointwise_f1"] == 1.0
    assert metrics["ahi_pointwise_roc_auc"] == 1.0
