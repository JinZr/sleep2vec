from __future__ import annotations

from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.default_dataset import Sample
from data.psg_pretrain_dataset import PSGPretrainDataset
from data.survival import attach_survival_metadata, load_survival_label_table, stack_survival_metadata
from sleep2vec.config import SurvivalConfig, load_finetune_config
from sleep2vec.losses.cox import CoxPHLossVectorized


def _write_sidecars(
    tmp_path: Path,
    *,
    duplicate_key: bool = False,
    disease_lines: list[str] | None = None,
    sidecar_columns: list[str] | None = None,
) -> SurvivalConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    disease_columns = tmp_path / "disease_columns.txt"
    event_time = tmp_path / "event_time.csv"
    is_event = tmp_path / "is_event.csv"
    has_label = tmp_path / "has_label.csv"
    if disease_lines is None:
        disease_lines = ["d1", "d2"]
    if sidecar_columns is None:
        sidecar_columns = ["d1", "d2"]

    disease_columns.write_text("\n".join(disease_lines) + "\n")
    rows = _sidecar_rows(
        sidecar_columns,
        [
            ("1", {"d1": "10", "d2": "20", "d3": "30"}),
            ("2", {"d1": "40", "d2": "50", "d3": "60"}),
        ],
    )
    if duplicate_key:
        rows.append(_sidecar_row("2", {"d1": "70", "d2": "80", "d3": "90"}, sidecar_columns))
    event_time.write_text("\n".join(rows) + "\n")
    is_event.write_text(
        "\n".join(
            _sidecar_rows(
                sidecar_columns,
                [
                    ("1", {"d1": "1", "d2": "0", "d3": "1"}),
                    ("2", {"d1": "0", "d2": "1", "d3": "0"}),
                ],
            )
        )
        + "\n"
    )
    has_label.write_text(
        "\n".join(
            _sidecar_rows(
                sidecar_columns,
                [
                    ("1", {"d1": "1", "d2": "1", "d3": "1"}),
                    ("2", {"d1": "1", "d2": "1", "d3": "1"}),
                ],
            )
        )
        + "\n"
    )

    return SurvivalConfig(
        key_column="eid",
        disease_columns_index=str(disease_columns),
        event_time_index=str(event_time),
        is_event_index=str(is_event),
        has_label_index=str(has_label),
    )


def _sidecar_rows(columns: list[str], rows: list[tuple[str, dict[str, str]]]) -> list[str]:
    return [",".join(["eid", *columns]), *[_sidecar_row(eid, values, columns) for eid, values in rows]]


def _sidecar_row(eid: str, values: dict[str, str], columns: list[str]) -> str:
    return ",".join([eid, *[values[column] for column in columns]])


def _missing_sidecar_config(tmp_path: Path) -> SurvivalConfig:
    missing = tmp_path / "missing_sidecars"
    return SurvivalConfig(
        key_column="eid",
        disease_columns_index=str(missing / "disease_columns.txt"),
        event_time_index=str(missing / "event_time.csv"),
        is_event_index=str(missing / "is_event.csv"),
        has_label_index=str(missing / "has_label.csv"),
    )


def _write_survival_preset(
    tmp_path: Path,
    sample_index_cls,
    *,
    include_survival_metadata: bool = True,
) -> Path:
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, ppg=np.asarray([0.0, 1.0], dtype=np.float32))
    metadata = {
        "source": "preset",
        "path": str(npz_path),
        "split": "train",
    }
    if include_survival_metadata:
        metadata.update(
            {
                "event_time": np.asarray([10.0, 20.0], dtype=np.float32),
                "is_event": np.asarray([1.0, 0.0], dtype=np.float32),
                "has_label": np.asarray([1.0, 1.0], dtype=np.float32),
            }
        )
    preset_path = tmp_path / "preset.pkl"
    with preset_path.open("wb") as f:
        pickle.dump(
            [
                sample_index_cls(
                    id="sample",
                    path=str(npz_path),
                    start=0,
                    end=2,
                    metadata=metadata,
                )
            ],
            f,
        )
    return preset_path


def _load_dataset_from_survival_preset(
    tmp_path: Path,
    dataset_module: str,
    default_dataset_module: str,
    *,
    include_survival_metadata: bool = True,
):
    dataset_cls = __import__(dataset_module, fromlist=["PSGPretrainDataset"]).PSGPretrainDataset
    sample_index_cls = __import__(default_dataset_module, fromlist=["SampleIndex"]).SampleIndex
    preset_path = _write_survival_preset(
        tmp_path, sample_index_cls, include_survival_metadata=include_survival_metadata
    )
    return dataset_cls(
        channel_names=["ppg"],
        channel_input_dims={"ppg": 1},
        save_preset_path=None,
        load_preset_path=str(preset_path),
        index=str(tmp_path / "unused_index.csv"),
        split=["train"],
        max_tokens=2,
        stride_tokens=0,
        mask_rate=0.0,
        allow_missing_channels=False,
        min_channels=1,
        randomly_select_channels=False,
        survival_label_config=_missing_sidecar_config(tmp_path),
        survival_output_dim=2,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


def test_cox_loss_matches_manual_breslow_value():
    loss_fn = CoxPHLossVectorized()
    pred = torch.tensor([[0.0], [1.0], [2.0]])
    event_time = torch.tensor([[5.0], [3.0], [1.0]])
    is_event = torch.tensor([[1.0], [1.0], [0.0]])
    has_label = torch.ones_like(pred)

    loss = loss_fn(pred, has_label, event_time, is_event)

    expected = (torch.logsumexp(torch.tensor([0.0, 1.0]), dim=0) - 1.0) / 2.0
    assert loss == pytest.approx(float(expected))


@pytest.mark.parametrize(
    "module_name",
    [
        "sleep2vec.losses.cox",
        "sleep2vec2.losses.cox",
        "sleep2expert.losses.cox",
    ],
)
def test_cox_loss_uses_shared_risk_set_for_tied_event_times(module_name: str):
    loss_cls = __import__(module_name, fromlist=["CoxPHLossVectorized"]).CoxPHLossVectorized
    loss_fn = loss_cls()
    pred = torch.tensor([[0.0], [1.0], [2.0]])
    event_time = torch.tensor([[5.0], [5.0], [1.0]])
    is_event = torch.tensor([[1.0], [1.0], [0.0]])
    has_label = torch.ones_like(pred)

    loss = loss_fn(pred, has_label, event_time, is_event)
    permuted_loss = loss_fn(pred[[1, 0, 2]], has_label[[1, 0, 2]], event_time[[1, 0, 2]], is_event[[1, 0, 2]])

    tied_risk = torch.logsumexp(torch.tensor([0.0, 1.0]), dim=0)
    expected = ((tied_risk - 0.0) + (tied_risk - 1.0)) / 2.0
    assert loss == pytest.approx(float(expected))
    assert permuted_loss == pytest.approx(float(expected))


def test_cox_loss_ignores_missing_labels_in_events_and_risk_sets():
    loss_fn = CoxPHLossVectorized()
    pred = torch.tensor([[0.0], [1.0], [100.0]])
    event_time = torch.tensor([[5.0], [3.0], [4.0]])
    is_event = torch.ones_like(pred)
    has_label = torch.tensor([[1.0], [1.0], [0.0]])

    loss = loss_fn(pred, has_label, event_time, is_event)

    expected = (torch.logsumexp(torch.tensor([0.0, 1.0]), dim=0) - 1.0) / 2.0
    assert loss == pytest.approx(float(expected))


def test_cox_loss_uses_censored_subjects_only_in_denominator():
    loss_fn = CoxPHLossVectorized()
    pred = torch.tensor([[0.0], [1.0], [2.0]])
    event_time = torch.tensor([[5.0], [3.0], [1.0]])
    is_event = torch.tensor([[1.0], [0.0], [1.0]])
    has_label = torch.ones_like(pred)

    loss = loss_fn(pred, has_label, event_time, is_event)

    expected = (torch.logsumexp(torch.tensor([0.0, 1.0, 2.0]), dim=0) - 2.0) / 2.0
    assert loss == pytest.approx(float(expected))


def test_cox_loss_skips_disease_columns_without_events_and_returns_connected_zero():
    loss_fn = CoxPHLossVectorized()
    pred = torch.tensor([[0.0, 3.0], [1.0, 4.0]], requires_grad=True)
    event_time = torch.tensor([[5.0, 5.0], [3.0, 3.0]])
    has_label = torch.ones_like(pred)
    is_event = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    loss = loss_fn(pred, has_label, event_time, is_event)
    expected = (torch.logsumexp(torch.tensor([0.0, 1.0]), dim=0) - 1.0) / 2.0
    assert float(loss.detach()) == pytest.approx(float(expected))

    zero = loss_fn(pred, has_label, event_time, torch.zeros_like(is_event))
    zero.backward()
    assert zero.item() == pytest.approx(0.0)
    assert pred.grad is not None


def _import_finetuning_class(module_name: str):
    try:
        return __import__(module_name, fromlist=["Sleep2vecFinetuning"]).Sleep2vecFinetuning
    except (AttributeError, ImportError, RuntimeError) as exc:
        message = str(exc)
        if "torchvision" in message or "RoFormerModel" in message:
            pytest.skip(f"Skipping finetuning import because local optional vision dependencies are unavailable: {exc}")
        raise


class _StaticLogitModel:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def __call__(self, batch):
        return self.logits


def _new_survival_finetuning_module(module_name: str, loss_module_name: str, *, prediction_export: bool = False):
    finetuning_cls = _import_finetuning_class(module_name)
    loss_cls = __import__(loss_module_name, fromlist=["CoxPHLossVectorized"]).CoxPHLossVectorized
    module = finetuning_cls.__new__(finetuning_cls)
    prediction_csv_path = "predictions.csv" if prediction_export else None
    object.__setattr__(
        module,
        "args",
        SimpleNamespace(
            device=torch.device("cpu"),
            is_survival=True,
            inference_prediction_csv_path=prediction_csv_path,
        ),
    )
    object.__setattr__(module, "_survival_loss", loss_cls())
    object.__setattr__(module, "_stage_outputs", {"train": [], "val": [], "test": []})
    object.__setattr__(module, "prediction_rows", [])
    return finetuning_cls, module


def _capture_logged_metrics(module):
    logged = []

    def _log(name, value, **kwargs):
        if torch.is_tensor(value):
            value = float(value.detach().cpu().item())
        logged.append((name, value, kwargs))

    object.__setattr__(module, "log", _log)
    return logged


@pytest.mark.parametrize(
    ("module_name", "loss_module_name"),
    [
        ("sleep2vec.sleep2vec_finetuning", "sleep2vec.losses.cox"),
        ("sleep2vec2.sleep2vec_finetuning", "sleep2vec2.losses.cox"),
        ("sleep2expert.sleep2vec_finetuning", "sleep2expert.losses.cox"),
    ],
)
def test_survival_loss_reports_zero_events_for_all_censored_batches(module_name: str, loss_module_name: str):
    _, module = _new_survival_finetuning_module(module_name, loss_module_name)
    logits = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    batch = {
        "metadata": {
            "event_time": torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
            "is_event": torch.zeros(2, 2),
            "has_label": torch.ones(2, 2),
        }
    }

    loss, event_count = module._compute_survival_loss(logits, batch)

    assert event_count == 0
    assert loss.item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("module_name", "loss_module_name"),
    [
        ("sleep2vec.sleep2vec_finetuning", "sleep2vec.losses.cox"),
        ("sleep2vec2.sleep2vec_finetuning", "sleep2vec2.losses.cox"),
        ("sleep2expert.sleep2vec_finetuning", "sleep2expert.losses.cox"),
    ],
)
def test_survival_eval_loss_uses_epoch_risk_set_for_singleton_batches(module_name: str, loss_module_name: str):
    finetuning_cls, module = _new_survival_finetuning_module(module_name, loss_module_name)
    logged = _capture_logged_metrics(module)
    event_batch = {
        "metadata": {
            "path": ["event.npz"],
            "event_time": torch.tensor([[1.0]]),
            "is_event": torch.tensor([[1.0]]),
            "has_label": torch.tensor([[1.0]]),
        },
        "token_start": torch.tensor([0]),
    }
    censored_batch = {
        "metadata": {
            "path": ["censored.npz"],
            "event_time": torch.tensor([[2.0]]),
            "is_event": torch.tensor([[0.0]]),
            "has_label": torch.tensor([[1.0]]),
        },
        "token_start": torch.tensor([0]),
    }

    singleton_loss = module._survival_loss(
        torch.tensor([[0.0]]),
        event_batch["metadata"]["has_label"],
        event_batch["metadata"]["event_time"],
        event_batch["metadata"]["is_event"],
    )
    assert singleton_loss.item() == pytest.approx(0.0)

    finetuning_cls._shared_step(module, event_batch, stage="val", model=_StaticLogitModel(torch.tensor([[0.0]])))
    finetuning_cls._shared_step(module, censored_batch, stage="val", model=_StaticLogitModel(torch.tensor([[1.0]])))
    finetuning_cls._finalize_epoch(module, "val")

    expected = module._survival_loss(
        torch.tensor([[0.0], [1.0]]),
        torch.ones(2, 1),
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[1.0], [0.0]]),
    )
    assert logged[0][0] == "val_loss"
    assert logged[0][1] == pytest.approx(float(expected))
    assert module._stage_outputs["val"] == []


@pytest.mark.parametrize(
    ("module_name", "loss_module_name"),
    [
        ("sleep2vec.sleep2vec_finetuning", "sleep2vec.losses.cox"),
        ("sleep2vec2.sleep2vec_finetuning", "sleep2vec2.losses.cox"),
        ("sleep2expert.sleep2vec_finetuning", "sleep2expert.losses.cox"),
    ],
)
def test_survival_eval_epoch_without_events_does_not_log_loss(module_name: str, loss_module_name: str):
    finetuning_cls, module = _new_survival_finetuning_module(module_name, loss_module_name)
    logged = _capture_logged_metrics(module)
    batch = {
        "metadata": {
            "path": ["first.npz", "second.npz"],
            "event_time": torch.tensor([[1.0], [2.0]]),
            "is_event": torch.zeros(2, 1),
            "has_label": torch.ones(2, 1),
        },
        "token_start": torch.tensor([0, 0]),
    }

    finetuning_cls._shared_step(module, batch, stage="val", model=_StaticLogitModel(torch.tensor([[0.0], [1.0]])))
    finetuning_cls._finalize_epoch(module, "val")

    assert logged == []
    assert module._stage_outputs["val"] == []


@pytest.mark.parametrize(
    ("module_name", "loss_module_name"),
    [
        ("sleep2vec.sleep2vec_finetuning", "sleep2vec.losses.cox"),
        ("sleep2vec2.sleep2vec_finetuning", "sleep2vec2.losses.cox"),
        ("sleep2expert.sleep2vec_finetuning", "sleep2expert.losses.cox"),
    ],
)
def test_survival_test_prediction_export_preserves_raw_log_risk(module_name: str, loss_module_name: str):
    finetuning_cls, module = _new_survival_finetuning_module(module_name, loss_module_name, prediction_export=True)
    _capture_logged_metrics(module)
    first_batch = {
        "metadata": {
            "path": ["same_path.npz"],
            "event_time": torch.tensor([[10.0, 20.0]]),
            "is_event": torch.tensor([[1.0, 0.0]]),
            "has_label": torch.tensor([[1.0, 1.0]]),
        },
        "token_start": torch.tensor([0]),
    }
    second_batch = {
        "metadata": {
            "path": ["same_path.npz"],
            "event_time": torch.tensor([[10.0, 20.0]]),
            "is_event": torch.tensor([[1.0, 0.0]]),
            "has_label": torch.tensor([[1.0, 1.0]]),
        },
        "token_start": torch.tensor([10]),
    }

    result = finetuning_cls._shared_step(
        module, first_batch, stage="test", model=_StaticLogitModel(torch.tensor([[0.0, 2.0]]))
    )
    assert result is None
    finetuning_cls._shared_step(module, second_batch, stage="test", model=_StaticLogitModel(torch.tensor([[2.0, 4.0]])))
    finetuning_cls._finalize_epoch(module, "test")

    assert len(module.prediction_rows) == 1
    row = module.prediction_rows[0]
    assert row["path"] == "same_path.npz"
    assert row["kind"] == "survival"
    assert row["prediction"] == pytest.approx([1.0, 3.0])
    assert row["log_risk"] == pytest.approx([1.0, 3.0])
    assert row["event_time"] == pytest.approx([10.0, 20.0])
    assert row["is_event"] == [1, 0]
    assert row["has_label"] == [1, 1]
    assert row["groundtruth"] == {
        "event_time": [10.0, 20.0],
        "is_event": [1, 0],
        "has_label": [1, 1],
    }
    assert row["n_predictions"] == 2
    assert row["n_windows"] == 2
    assert row["token_starts"] == [0, 10]
    assert module._stage_outputs["test"] == []


@pytest.mark.parametrize(
    ("module_name", "loss_module_name"),
    [
        ("sleep2vec.sleep2vec_finetuning", "sleep2vec.losses.cox"),
        ("sleep2vec2.sleep2vec_finetuning", "sleep2vec2.losses.cox"),
        ("sleep2expert.sleep2vec_finetuning", "sleep2expert.losses.cox"),
    ],
)
def test_survival_all_censored_test_epoch_exports_predictions_without_loss(module_name: str, loss_module_name: str):
    finetuning_cls, module = _new_survival_finetuning_module(module_name, loss_module_name, prediction_export=True)
    logged = _capture_logged_metrics(module)
    batch = {
        "metadata": {
            "path": ["all_censored.npz"],
            "event_time": torch.tensor([[1.0, 2.0]]),
            "is_event": torch.zeros(1, 2),
            "has_label": torch.ones(1, 2),
        },
        "token_start": torch.tensor([0]),
    }

    finetuning_cls._shared_step(module, batch, stage="test", model=_StaticLogitModel(torch.tensor([[0.5, -0.5]])))
    finetuning_cls._finalize_epoch(module, "test")

    assert logged == []
    assert len(module.prediction_rows) == 1
    assert module.prediction_rows[0]["path"] == "all_censored.npz"
    assert module.prediction_rows[0]["prediction"] == pytest.approx([0.5, -0.5])
    assert module._stage_outputs["test"] == []


@pytest.mark.parametrize(
    ("module_name", "loss_module_name"),
    [
        ("sleep2vec.sleep2vec_finetuning", "sleep2vec.losses.cox"),
        ("sleep2vec2.sleep2vec_finetuning", "sleep2vec2.losses.cox"),
        ("sleep2expert.sleep2vec_finetuning", "sleep2expert.losses.cox"),
    ],
)
def test_survival_test_prediction_export_keeps_unlabeled_batches(module_name: str, loss_module_name: str):
    finetuning_cls, module = _new_survival_finetuning_module(module_name, loss_module_name, prediction_export=True)
    logged = _capture_logged_metrics(module)
    batch = {
        "metadata": {
            "path": ["unlabeled.npz"],
            "event_time": torch.zeros(1, 2),
            "is_event": torch.zeros(1, 2),
            "has_label": torch.zeros(1, 2),
        },
        "token_start": torch.tensor([0]),
    }

    finetuning_cls._shared_step(module, batch, stage="test", model=_StaticLogitModel(torch.tensor([[1.5, -1.5]])))
    finetuning_cls._finalize_epoch(module, "test")

    assert logged == []
    assert len(module.prediction_rows) == 1
    assert module.prediction_rows[0]["path"] == "unlabeled.npz"
    assert module.prediction_rows[0]["prediction"] == pytest.approx([1.5, -1.5])
    assert module.prediction_rows[0]["has_label"] == [0, 0]
    assert module._stage_outputs["test"] == []


def test_survival_sidecars_attach_subject_labels_to_multiple_rows(tmp_path: Path):
    labels = load_survival_label_table(_write_sidecars(tmp_path), expected_output_dim=2)
    assert labels is not None
    assert labels.label_names == ["d1", "d2"]
    first_metadata = {}
    second_metadata = {}

    attach_survival_metadata(first_metadata, 1, labels)
    attach_survival_metadata(second_metadata, "1", labels)
    samples = [
        Sample(id="a", length=1, payload={}, tokens={}, masks={}, metadata=first_metadata),
        Sample(id="b", length=1, payload={}, tokens={}, masks={}, metadata=second_metadata),
    ]

    stacked = stack_survival_metadata(samples, expected_output_dim=2)

    assert stacked["event_time"].shape == (2, 2)
    assert torch.equal(stacked["event_time"][0], stacked["event_time"][1])
    assert torch.equal(stacked["is_event"][0], stacked["is_event"][1])
    assert torch.equal(stacked["has_label"][0], stacked["has_label"][1])


def test_psg_dataset_stacks_survival_metadata_for_repeated_subject(tmp_path: Path):
    config = _write_sidecars(tmp_path / "sidecars")
    np.savez(tmp_path / "first.npz", ppg=np.asarray([0.0, 1.0], dtype=np.float32))
    np.savez(tmp_path / "second.npz", ppg=np.asarray([2.0, 3.0], dtype=np.float32))
    index = tmp_path / "index.csv"
    index.write_text(
        "\n".join(
            [
                "path,split,duration,eid",
                f"{tmp_path / 'first.npz'},train,60,1",
                f"{tmp_path / 'second.npz'},train,60,1",
            ]
        )
        + "\n"
    )

    dataset = PSGPretrainDataset(
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
        survival_label_config=config,
        survival_output_dim=2,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataset.dataloader()))

    assert batch["metadata"]["event_time"].shape == (2, 2)
    assert torch.equal(batch["metadata"]["event_time"][0], batch["metadata"]["event_time"][1])
    assert torch.equal(batch["metadata"]["is_event"][0], batch["metadata"]["is_event"][1])
    assert torch.equal(batch["metadata"]["has_label"][0], batch["metadata"]["has_label"][1])


@pytest.mark.parametrize(
    ("dataset_module", "default_dataset_module"),
    [
        ("data.psg_pretrain_dataset", "data.default_dataset"),
        ("sleep2vec2.data.psg_pretrain_dataset", "sleep2vec2.data.default_dataset"),
        ("sleep2expert.data.psg_pretrain_dataset", "sleep2expert.data.default_dataset"),
    ],
)
def test_survival_preset_loads_without_sidecar_files(
    dataset_module: str,
    default_dataset_module: str,
    tmp_path: Path,
):
    dataset = _load_dataset_from_survival_preset(tmp_path, dataset_module, default_dataset_module)

    batch = next(iter(dataset.dataloader()))

    assert batch["metadata"]["event_time"].shape == (1, 2)
    assert batch["metadata"]["is_event"].shape == (1, 2)
    assert batch["metadata"]["has_label"].shape == (1, 2)


def test_survival_preset_load_requires_embedded_survival_metadata(tmp_path: Path):
    dataset = _load_dataset_from_survival_preset(
        tmp_path,
        "data.psg_pretrain_dataset",
        "data.default_dataset",
        include_survival_metadata=False,
    )

    with pytest.raises(ValueError, match="regenerate presets with survival sidecars"):
        next(iter(dataset.dataloader()))


def test_survival_sidecars_validate_headers_duplicates_missing_keys_and_output_dim(tmp_path: Path):
    with pytest.raises(ValueError, match="empty line"):
        load_survival_label_table(_write_sidecars(tmp_path / "empty_line", disease_lines=["d1", "", "d2"]))

    with pytest.raises(ValueError, match="duplicate disease"):
        load_survival_label_table(_write_sidecars(tmp_path / "duplicate_disease", disease_lines=["d1", "d1"]))

    with pytest.raises(ValueError, match="columns must exactly match"):
        load_survival_label_table(_write_sidecars(tmp_path / "missing_column", sidecar_columns=["d1"]))

    with pytest.raises(ValueError, match="columns must exactly match"):
        load_survival_label_table(_write_sidecars(tmp_path / "extra_column", sidecar_columns=["d1", "d2", "d3"]))

    with pytest.raises(ValueError, match="columns must exactly match"):
        load_survival_label_table(_write_sidecars(tmp_path / "reordered", sidecar_columns=["d2", "d1"]))

    with pytest.raises(ValueError, match="duplicate key"):
        load_survival_label_table(_write_sidecars(tmp_path / "duplicates", duplicate_key=True))

    labels = load_survival_label_table(_write_sidecars(tmp_path / "valid"))
    assert labels is not None
    with pytest.raises(ValueError, match="missing from survival sidecars"):
        attach_survival_metadata({}, 9, labels)

    with pytest.raises(ValueError, match="output_dim"):
        load_survival_label_table(_write_sidecars(tmp_path / "dim"), expected_output_dim=3)


@pytest.mark.parametrize(
    ("module_name", "config_path"),
    [
        ("sleep2vec.config", "configs/ppg_cox_finetune_large.yaml"),
        ("sleep2vec2.config", "configs/sleep2vec2/ppg_cox_finetune_large.yaml"),
        ("sleep2expert.config", "configs/sleep2expert/heartbeat_breath_cox_finetune_large.yaml"),
    ],
)
def test_survival_config_templates_load_for_all_variants(module_name: str, config_path: str):
    module = __import__(module_name, fromlist=["load_finetune_config"])
    bundle = module.load_finetune_config(config_path)

    assert bundle.finetune.task.type == "survival"
    assert bundle.finetune.task.output_dim == 177
    assert bundle.finetune.task.monitor == "val_loss"
    assert bundle.finetune.survival.key_column == "eid"
    assert bundle.finetune.survival.disease_columns_index == "/path/to/disease_columns.txt"


def test_survival_config_rejects_invalid_task_contract(tmp_path: Path):
    payload_path = tmp_path / "invalid.yaml"
    payload = Path("configs/ppg_cox_finetune_large.yaml").read_text()
    payload_path.write_text(payload.replace("monitor_mod: min", "monitor_mod: max"))

    with pytest.raises(ValueError, match="monitor val_loss"):
        load_finetune_config(payload_path)


@pytest.mark.parametrize(
    ("module_name", "config_path"),
    [
        ("sleep2vec.config", "configs/ppg_cox_finetune_large.yaml"),
        ("sleep2vec2.config", "configs/sleep2vec2/ppg_cox_finetune_large.yaml"),
        ("sleep2expert.config", "configs/sleep2expert/heartbeat_breath_cox_finetune_large.yaml"),
    ],
)
def test_survival_config_requires_disease_columns_index(module_name: str, config_path: str, tmp_path: Path):
    module = __import__(module_name, fromlist=["load_finetune_config"])
    payload_path = tmp_path / "missing_disease_columns.yaml"
    payload = Path(config_path).read_text()
    payload_path.write_text(payload.replace("    disease_columns_index: /path/to/disease_columns.txt\n", ""))
    with pytest.raises(ValueError, match="disease_columns_index"):
        module.load_finetune_config(payload_path)
