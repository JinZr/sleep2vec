from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _sidecar_rows(columns: list[str], rows: list[tuple[str, dict[str, str]]]) -> list[str]:
    return [",".join(["eid", *columns]), *[_sidecar_row(eid, values, columns) for eid, values in rows]]


def _sidecar_row(eid: str, values: dict[str, str], columns: list[str]) -> str:
    return ",".join([eid, *[values[column] for column in columns]])


def _write_multilabel_sidecars(
    tmp_path: Path,
    *,
    duplicate_key: bool = False,
    disease_lines: list[str] | None = None,
    sidecar_columns: list[str] | None = None,
    row_keys: tuple[str, str] = ("1", "2"),
    has_label_values: tuple[dict[str, str], dict[str, str]] | None = None,
) -> SimpleNamespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    disease_columns = tmp_path / "disease_columns.txt"
    label = tmp_path / "label.csv"
    has_label = tmp_path / "has_label.csv"
    if disease_lines is None:
        disease_lines = ["d1", "d2"]
    if sidecar_columns is None:
        sidecar_columns = ["d1", "d2"]
    if has_label_values is None:
        has_label_values = ({"d1": "1", "d2": "1", "d3": "1"}, {"d1": "1", "d2": "1", "d3": "1"})

    first_key, second_key = row_keys
    disease_columns.write_text("\n".join(disease_lines) + "\n")
    label_rows = _sidecar_rows(
        sidecar_columns,
        [
            (first_key, {"d1": "1", "d2": "0", "d3": "1"}),
            (second_key, {"d1": "0", "d2": "1", "d3": "0"}),
        ],
    )
    if duplicate_key:
        label_rows.append(_sidecar_row(second_key, {"d1": "1", "d2": "1", "d3": "0"}, sidecar_columns))
    label.write_text("\n".join(label_rows) + "\n")
    has_label.write_text(
        "\n".join(_sidecar_rows(sidecar_columns, [(first_key, has_label_values[0]), (second_key, has_label_values[1])]))
        + "\n"
    )
    return SimpleNamespace(
        key_column="eid",
        disease_columns_index=str(disease_columns),
        label_index=str(label),
        has_label_index=str(has_label),
    )


def _import_finetuning_class(module_name: str):
    try:
        return __import__(module_name, fromlist=["Sleep2vecFinetuning"]).Sleep2vecFinetuning
    except (AttributeError, ImportError, RuntimeError) as exc:
        message = str(exc)
        if "torchvision" in message or "RoFormerModel" in message:
            pytest.skip(f"Skipping finetuning import because local optional dependencies are unavailable: {exc}")
        raise


class _StaticLogitModel:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def __call__(self, batch):
        return self.logits


def _capture_logged_metrics(module):
    logged = []

    def _log(name, value, **kwargs):
        if torch.is_tensor(value):
            value = float(value.detach().cpu().item())
        logged.append((name, value, kwargs))

    object.__setattr__(module, "log", _log)
    return logged


def _new_multilabel_finetuning_module(module_name: str, tmp_path: Path, *, prediction_export: bool = False):
    finetuning_cls = _import_finetuning_class(module_name)
    module = finetuning_cls.__new__(finetuning_cls)
    sidecars = _write_multilabel_sidecars(tmp_path / "sidecars")
    object.__setattr__(
        module,
        "args",
        SimpleNamespace(
            device=torch.device("cpu"),
            is_survival=False,
            is_classification=True,
            is_multilabel=True,
            is_seq=False,
            label_name="disease_detection",
            multilabel=sidecars,
            inference_prediction_csv_path="predictions.csv" if prediction_export else None,
            finetune_preset_path=None,
            inference_preset_path=None,
        ),
    )
    object.__setattr__(module, "_multilabel_loss", torch.nn.BCEWithLogitsLoss(reduction="none"))
    object.__setattr__(module, "_stage_outputs", {"train": [], "val": [], "test": []})
    object.__setattr__(module, "_eval_loss_sums", {"val": 0.0, "test": 0.0})
    object.__setattr__(module, "_eval_loss_counts", {"val": 0, "test": 0})
    object.__setattr__(module, "prediction_rows", [])
    object.__setattr__(module, "multilabel_per_disease_metric_rows", [])
    return finetuning_cls, module


@pytest.mark.parametrize(
    "multilabel_module",
    [
        "data.multilabel",
        "sleep2vec2.data.multilabel",
        "sleep2expert.data.multilabel",
    ],
)
def test_multilabel_sidecars_load_and_attach_subject_labels(multilabel_module: str, tmp_path: Path):
    module = __import__(
        multilabel_module,
        fromlist=["attach_multilabel_metadata", "load_multilabel_label_table", "stack_multilabel_metadata"],
    )
    labels = module.load_multilabel_label_table(_write_multilabel_sidecars(tmp_path), expected_output_dim=2)

    metadata = {}
    module.attach_multilabel_metadata(metadata, "1", labels)
    assert metadata["eid"] == "1"
    assert metadata["disease_label"].tolist() == [1.0, 0.0]
    assert metadata["has_label"].tolist() == [1.0, 1.0]

    stacked = module.stack_multilabel_metadata(
        [SimpleNamespace(metadata=metadata), SimpleNamespace(metadata=metadata)],
        expected_output_dim=2,
        key_column="eid",
    )
    assert stacked["disease_label"].shape == (2, 2)
    assert stacked["eid"] == ["1", "1"]


@pytest.mark.parametrize(
    "multilabel_module",
    [
        "data.multilabel",
        "sleep2vec2.data.multilabel",
        "sleep2expert.data.multilabel",
    ],
)
def test_multilabel_sidecars_validate_headers_masks_duplicates_and_output_dim(multilabel_module: str, tmp_path: Path):
    module = __import__(multilabel_module, fromlist=["attach_multilabel_metadata", "load_multilabel_label_table"])

    with pytest.raises(ValueError, match="empty line"):
        module.load_multilabel_label_table(_write_multilabel_sidecars(tmp_path / "blank", disease_lines=["d1", ""]))
    with pytest.raises(ValueError, match="duplicate disease"):
        module.load_multilabel_label_table(
            _write_multilabel_sidecars(tmp_path / "duplicate_disease", disease_lines=["d1", "d1"])
        )
    with pytest.raises(ValueError, match="columns must exactly match"):
        module.load_multilabel_label_table(_write_multilabel_sidecars(tmp_path / "missing", sidecar_columns=["d1"]))
    with pytest.raises(ValueError, match="duplicate key"):
        module.load_multilabel_label_table(_write_multilabel_sidecars(tmp_path / "duplicate_key", duplicate_key=True))
    with pytest.raises(ValueError, match="has_label.*must be 0 or 1"):
        module.load_multilabel_label_table(
            _write_multilabel_sidecars(
                tmp_path / "bad_mask",
                has_label_values=({"d1": "1", "d2": "2", "d3": "1"}, {"d1": "1", "d2": "1", "d3": "1"}),
            )
        )
    with pytest.raises(ValueError, match="output_dim"):
        module.load_multilabel_label_table(_write_multilabel_sidecars(tmp_path / "dim"), expected_output_dim=3)

    labels = module.load_multilabel_label_table(_write_multilabel_sidecars(tmp_path / "valid"))
    with pytest.raises(ValueError, match="missing from multilabel sidecars"):
        module.attach_multilabel_metadata({}, "9", labels)


@pytest.mark.parametrize(
    "multilabel_module",
    [
        "data.multilabel",
        "sleep2vec2.data.multilabel",
        "sleep2expert.data.multilabel",
    ],
)
def test_multilabel_preset_stack_requires_embedded_metadata(multilabel_module: str):
    module = __import__(multilabel_module, fromlist=["stack_multilabel_metadata"])
    sample = SimpleNamespace(metadata={"path": "sample.npz", "eid": "1"})

    with pytest.raises(ValueError, match="regenerate presets with multilabel sidecars"):
        module.stack_multilabel_metadata([sample], expected_output_dim=2, key_column="eid")


@pytest.mark.parametrize(
    "dataset_module",
    [
        "data.psg_pretrain_dataset",
        "sleep2vec2.data.psg_pretrain_dataset",
        "sleep2expert.data.psg_pretrain_dataset",
    ],
)
def test_psg_dataset_requires_multilabel_key_column(dataset_module: str, tmp_path: Path):
    dataset_cls = __import__(dataset_module, fromlist=["PSGPretrainDataset"]).PSGPretrainDataset
    np.savez(tmp_path / "sample.npz", ppg=np.asarray([0.0, 1.0], dtype=np.float32))
    index = tmp_path / "index.csv"
    index.write_text("\n".join(["path,split,duration", f"{tmp_path / 'sample.npz'},train,60"]) + "\n")

    with pytest.raises(ValueError, match="Required multilabel key column 'eid' is missing"):
        dataset_cls(
            channel_names=["ppg"],
            channel_input_dims={"ppg": 1},
            save_preset_path=None,
            load_preset_path=None,
            index=str(index),
            split=["train"],
            max_tokens=2,
            stride_tokens=0,
            mask_rate=0.0,
            allow_missing_channels=False,
            min_channels=1,
            randomly_select_channels=False,
            multilabel_label_config=_write_multilabel_sidecars(tmp_path / "sidecars"),
            multilabel_output_dim=2,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )


@pytest.mark.parametrize(
    "module_name",
    [
        "sleep2vec.sleep2vec_finetuning",
        "sleep2vec2.sleep2vec_finetuning",
        "sleep2expert.sleep2vec_finetuning",
    ],
)
def test_multilabel_masked_bce_ignores_invalid_cells(module_name: str, tmp_path: Path):
    finetuning_cls, module = _new_multilabel_finetuning_module(module_name, tmp_path)
    logits = torch.tensor([[0.0, 2.0], [4.0, -2.0]], requires_grad=True)
    batch = {
        "metadata": {
            "disease_label": torch.tensor([[1.0, float("nan")], [0.0, 1.0]]),
            "has_label": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        }
    }

    loss, valid_count = finetuning_cls._compute_multilabel_loss(module, logits, batch)

    safe_labels = torch.where(
        batch["metadata"]["has_label"] > 0.5,
        batch["metadata"]["disease_label"],
        torch.zeros_like(batch["metadata"]["disease_label"]),
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        safe_labels,
        reduction="none",
    )[batch["metadata"]["has_label"] > 0.5].mean()
    assert valid_count == 3
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "module_name",
    [
        "sleep2vec.sleep2vec_finetuning",
        "sleep2vec2.sleep2vec_finetuning",
        "sleep2expert.sleep2vec_finetuning",
    ],
)
def test_multilabel_eval_aggregates_duplicate_subject_logits_and_exports_vectors(module_name: str, tmp_path: Path):
    finetuning_cls, module = _new_multilabel_finetuning_module(module_name, tmp_path, prediction_export=True)
    logged = _capture_logged_metrics(module)
    first_batch = {
        "metadata": {
            "path": ["k1_a.npz", "k2.npz"],
            "eid": ["k1", "k2"],
            "disease_label": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "has_label": torch.ones(2, 2),
        },
        "token_start": torch.tensor([0, 0]),
    }
    second_batch = {
        "metadata": {
            "path": ["k1_b.npz"],
            "eid": ["k1"],
            "disease_label": torch.tensor([[1.0, 0.0]]),
            "has_label": torch.ones(1, 2),
        },
        "token_start": torch.tensor([10]),
    }

    finetuning_cls._shared_step(
        module, first_batch, stage="test", model=_StaticLogitModel(torch.tensor([[0.0, 2.0], [-1.0, 5.0]]))
    )
    finetuning_cls._shared_step(module, second_batch, stage="test", model=_StaticLogitModel(torch.tensor([[2.0, 4.0]])))
    finetuning_cls._finalize_epoch(module, "test")

    assert any(name == "test_macro_auroc" and value == pytest.approx(1.0) for name, value, _ in logged)
    assert [row["disease"] for row in module.multilabel_per_disease_metric_rows] == ["d1", "d2"]
    assert len(module.prediction_rows) == 2
    k1_row = next(row for row in module.prediction_rows if row["multilabel_key"] == "k1")
    assert k1_row["kind"] == "multilabel_classification"
    assert k1_row["disease_names"] == ["d1", "d2"]
    assert k1_row["logit"] == pytest.approx([1.0, 3.0])
    assert k1_row["probability"] == pytest.approx((1.0 / (1.0 + np.exp(-np.asarray([1.0, 3.0])))).tolist())
    assert k1_row["groundtruth"] == pytest.approx([1.0, 0.0])
    assert k1_row["has_label"] == [1, 1]
    assert k1_row["n_windows"] == 2
    assert k1_row["token_starts"] == [0, 10]
    assert module._stage_outputs["test"] == []


@pytest.mark.parametrize(
    "metrics_module",
    [
        "sleep2vec.metrics",
        "sleep2vec2.metrics",
        "sleep2expert.metrics",
    ],
)
def test_multilabel_per_disease_metrics_skip_single_class_diseases(metrics_module: str):
    module = __import__(metrics_module, fromlist=["compute_multilabel_metrics_by_disease"])
    labels = np.asarray([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    probs = np.asarray([[0.8, 0.9], [0.7, 0.2]], dtype=np.float32)
    has_label = np.ones_like(labels)

    rows = module.compute_multilabel_metrics_by_disease(labels, probs, has_label, ["all_positive", "valid"])

    assert [row["disease"] for row in rows] == ["valid"]
    assert rows[0]["disease_idx"] == 1
    assert rows[0]["n_positive"] == 1
    assert rows[0]["n_negative"] == 1


@pytest.mark.parametrize(
    "module_name",
    [
        "sleep2vec.sleep2vec_finetuning",
        "sleep2vec2.sleep2vec_finetuning",
        "sleep2expert.sleep2vec_finetuning",
    ],
)
def test_multilabel_per_disease_rows_preserve_source_indices(module_name: str, tmp_path: Path):
    finetuning_cls, module = _new_multilabel_finetuning_module(module_name, tmp_path)
    labels = np.asarray([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    probs = np.asarray([[0.8, 0.9], [0.7, 0.2]], dtype=np.float32)
    has_label = np.ones_like(labels)

    rows = finetuning_cls._build_multilabel_per_disease_metric_rows(
        module,
        "val",
        labels,
        probs,
        has_label,
        ["all_positive", "valid"],
    )

    assert rows[0]["stage"] == "val"
    assert rows[0]["disease"] == "valid"
    assert rows[0]["disease_idx"] == 1
