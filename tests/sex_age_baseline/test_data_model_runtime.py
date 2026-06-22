from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch
import yaml

from sex_age_baseline.config import load_config
from sex_age_baseline.data import load_split_dataset, make_dataloader
from sex_age_baseline.model import SexAgeMLP
import sex_age_baseline.runtime as baseline_runtime
from sex_age_baseline.runtime import evaluate_model, masked_multilabel_bce


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload))
    return path


def _write_index(path: Path, rows: list[str]) -> Path:
    path.write_text("eid,split,age,sex\n" + "\n".join(rows) + "\n")
    return path


def _write_survival_sidecars(tmp_path: Path, keys: list[str], disease_count: int = 2) -> dict[str, str]:
    diseases = [f"d{i + 1}" for i in range(disease_count)]
    disease_columns = tmp_path / "disease_columns.txt"
    event_time = tmp_path / "event_time.csv"
    is_event = tmp_path / "is_event.csv"
    has_label = tmp_path / "has_label.csv"
    disease_columns.write_text("\n".join(diseases) + "\n")
    header = ",".join(["eid", *diseases])
    event_rows = [header]
    is_event_rows = [header]
    has_label_rows = [header]
    for idx, key in enumerate(keys):
        event_rows.append(",".join([key, *[str(10 + idx + j) for j in range(disease_count)]]))
        is_event_rows.append(",".join([key, *["1" for _ in diseases]]))
        has_label_rows.append(",".join([key, *["1" for _ in diseases]]))
    event_time.write_text("\n".join(event_rows) + "\n")
    is_event.write_text("\n".join(is_event_rows) + "\n")
    has_label.write_text("\n".join(has_label_rows) + "\n")
    return {
        "disease_columns_index": str(disease_columns),
        "event_time_index": str(event_time),
        "is_event_index": str(is_event),
        "has_label_index": str(has_label),
    }


def _write_multilabel_sidecars(tmp_path: Path, keys: list[str], disease_count: int = 2) -> dict[str, str]:
    diseases = [f"d{i + 1}" for i in range(disease_count)]
    disease_columns = tmp_path / "disease_columns.txt"
    label_index = tmp_path / "disease_label.csv"
    has_label = tmp_path / "has_label.csv"
    disease_columns.write_text("\n".join(diseases) + "\n")
    header = ",".join(["eid", *diseases])
    label_rows = [header]
    has_label_rows = [header]
    for idx, key in enumerate(keys):
        labels = [str((idx + j) % 2) for j in range(disease_count)]
        label_rows.append(",".join([key, *labels]))
        has_label_rows.append(",".join([key, *["1" for _ in diseases]]))
    label_index.write_text("\n".join(label_rows) + "\n")
    has_label.write_text("\n".join(has_label_rows) + "\n")
    return {
        "disease_columns_index": str(disease_columns),
        "label_index": str(label_index),
        "has_label_index": str(has_label),
    }


def _base_payload(index: Path, sidecars: dict[str, str], task_type: str) -> dict:
    finetune = {
        "task": {
            "type": task_type,
            "output_dim": 2,
            "is_seq": False,
            "monitor": "val_c_index" if task_type == "survival" else "val_macro_auroc",
            "monitor_mod": "max",
        }
    }
    if task_type == "survival":
        finetune["survival"] = {"key_column": "eid", **sidecars}
    else:
        finetune["multilabel"] = {"key_column": "eid", **sidecars}
        finetune["loss"] = {"pos_weight": None}
    return {
        "model": {
            "name": "sex_age_mlp",
            "features": ["age", "sex"],
            "age": {"transform": "divide", "scale": 100.0, "embedding_dim": 4},
            "sex": {"encoding": "binary", "embedding_dim": 4},
            "head": {"hidden_dim": 8, "dropout": 0.0, "activation": "elu"},
        },
        "data": {
            "index": str(index),
            "split_column": "split",
            "key_column": "eid",
            "deduplicate_by_key": True,
        },
        "finetune": finetune,
        "outputs": {"prediction_csv": True, "per_disease_metrics_csv": True},
    }


def _write_config(tmp_path: Path, rows: list[str], task_type: str = "survival") -> Path:
    index = _write_index(tmp_path / "index.csv", rows)
    keys = [row.split(",", 1)[0] for row in rows]
    sidecars = (
        _write_survival_sidecars(tmp_path, sorted(set(keys)))
        if task_type == "survival"
        else _write_multilabel_sidecars(tmp_path, sorted(set(keys)))
    )
    return _write_yaml(tmp_path / f"{task_type}.yaml", _base_payload(index, sidecars, task_type))


def _runtime_args(config: Path, tmp_path: Path, *, version_name: str, epochs: int = 1, test_after_fit: bool = False):
    return Namespace(
        config=config,
        label_name="unit",
        epochs=epochs,
        lr=1e-3,
        weight_decay=0.0,
        batch_size=2,
        num_workers=0,
        patience=100,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        device="cpu",
        ckpt_path=None,
        version_name=version_name,
        results_csv_path=tmp_path / "results.csv",
        seed=4523,
        test_after_fit=test_after_fit,
        ckpt_every_n_epochs=1,
    )


def test_split_filtering_and_deduplication(tmp_path: Path):
    config = _write_config(
        tmp_path,
        [
            "001,train,50,female",
            "001,train,50,0",
            "002,val,60,male",
            "003,test,55,1",
        ],
    )
    cfg = load_config(config, validate_sidecars=True)

    train = load_split_dataset(cfg, "train")
    val = load_split_dataset(cfg, "val")

    assert len(train) == 1
    assert train[0].key == "001"
    assert len(val) == 1


def test_conflicting_duplicate_metadata_fails(tmp_path: Path):
    config = _write_config(tmp_path, ["001,train,50,female", "001,val,50,female"])
    cfg = load_config(config, validate_sidecars=True)

    with pytest.raises(ValueError, match="conflicting split"):
        load_split_dataset(cfg, "train")


def test_invalid_sex_fails(tmp_path: Path):
    config = _write_config(tmp_path, ["001,train,50,unknown"])
    cfg = load_config(config, validate_sidecars=True)

    with pytest.raises(ValueError, match="sex value"):
        load_split_dataset(cfg, "train")


@pytest.mark.parametrize("task_type", ["survival", "multilabel_classification"])
def test_model_forward_shape(tmp_path: Path, task_type: str):
    config = _write_config(tmp_path, ["001,train,50,0", "002,train,60,1"], task_type=task_type)
    cfg = load_config(config, validate_sidecars=True)
    model = SexAgeMLP(cfg)

    logits = model(torch.tensor([50.0, 60.0]), torch.tensor([0, 1]))

    assert tuple(logits.shape) == (2, 2)


def test_cox_eval_reports_val_c_index(tmp_path: Path):
    pytest.importorskip("sksurv.metrics")
    config = _write_config(
        tmp_path,
        ["001,val,50,0", "002,val,60,1", "003,val,55,0"],
        task_type="survival",
    )
    cfg = load_config(config, validate_sidecars=True)
    dataset = load_split_dataset(cfg, "val")
    loader = make_dataloader(dataset, batch_size=3, num_workers=0, shuffle=False)
    model = SexAgeMLP(cfg)

    result = evaluate_model(model, loader, cfg, device=torch.device("cpu"), stage="val")

    assert "val_c_index" in result.metrics
    assert result.survival_per_disease_rows


def test_multilabel_masked_bce_ignores_invalid_cells():
    logits = torch.tensor([[0.0, 2.0], [4.0, -2.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    has_label = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    loss = masked_multilabel_bce(logits, labels, has_label)

    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")[
        has_label > 0.5
    ].mean()
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert logits.grad[0, 1].item() == pytest.approx(0.0)


def test_multilabel_eval_reports_macro_and_micro_metrics(tmp_path: Path):
    config = _write_config(
        tmp_path,
        ["001,val,50,0", "002,val,60,1", "003,val,55,0", "004,val,65,1"],
        task_type="multilabel_classification",
    )
    cfg = load_config(config, validate_sidecars=True)
    dataset = load_split_dataset(cfg, "val")
    loader = make_dataloader(dataset, batch_size=4, num_workers=0, shuffle=False)
    model = SexAgeMLP(cfg)

    result = evaluate_model(model, loader, cfg, device=torch.device("cpu"), stage="val")

    assert "val_macro_auroc" in result.metrics
    assert "val_micro_auroc" in result.metrics
    assert result.multilabel_per_disease_rows


def test_train_rejects_non_empty_run_dir_before_loading_data(tmp_path: Path, monkeypatch):
    config = _write_config(tmp_path, ["001,train,50,0"], task_type="multilabel_classification")
    cfg = load_config(config, validate_sidecars=True)
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "log-finetune" / "reused"
    run_dir.mkdir(parents=True)
    (run_dir / "marker.txt").write_text("old run\n")

    with pytest.raises(FileExistsError, match="Use a new --version-name"):
        baseline_runtime.train_and_save(_runtime_args(config, tmp_path, version_name="reused"), cfg)


def test_train_fails_when_configured_monitor_is_missing(tmp_path: Path, monkeypatch):
    config = _write_config(
        tmp_path,
        ["001,train,50,0", "002,train,60,1", "003,val,55,0", "004,val,65,1"],
        task_type="multilabel_classification",
    )
    payload = yaml.safe_load(config.read_text())
    payload["finetune"]["task"]["monitor"] = "val_missing_metric"
    _write_yaml(config, payload)
    cfg = load_config(config, validate_sidecars=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="val_missing_metric.*Available metrics"):
        baseline_runtime.train_and_save(_runtime_args(config, tmp_path, version_name="missing-monitor"), cfg)


def test_train_fails_without_finite_best_checkpoint(tmp_path: Path, monkeypatch):
    config = _write_config(
        tmp_path,
        ["001,train,50,0", "002,val,60,1"],
        task_type="multilabel_classification",
    )
    cfg = load_config(config, validate_sidecars=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="No finite best checkpoint"):
        baseline_runtime.train_and_save(_runtime_args(config, tmp_path, version_name="no-finite-best"), cfg)

    assert not (tmp_path / "log-finetune" / "no-finite-best" / "checkpoints" / "best.ckpt").exists()


def test_test_after_fit_writers_receive_test_eval_split(tmp_path: Path, monkeypatch):
    config = _write_config(
        tmp_path,
        [
            "001,train,50,0",
            "002,train,60,1",
            "003,val,55,0",
            "004,val,65,1",
            "005,test,58,0",
            "006,test,68,1",
        ],
        task_type="multilabel_classification",
    )
    cfg = load_config(config, validate_sidecars=True)
    monkeypatch.chdir(tmp_path)
    seen_splits = []

    def capture_split(*args):
        seen_splits.append(args[-1].eval_split)

    monkeypatch.setattr(baseline_runtime, "save_result_csv", capture_split)
    monkeypatch.setattr(baseline_runtime, "save_prediction_csv", capture_split)
    monkeypatch.setattr(baseline_runtime, "save_multilabel_per_disease_metrics_csv", capture_split)

    baseline_runtime.train_and_save(
        _runtime_args(config, tmp_path, version_name="test-after-fit", test_after_fit=True),
        cfg,
    )

    assert seen_splits
    assert set(seen_splits) == {"test"}
