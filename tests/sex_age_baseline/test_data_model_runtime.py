from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import pickle

import pandas as pd
import pytest
import torch
import yaml

from data.default_dataset import SampleIndex
from sex_age_baseline.config import load_config
from sex_age_baseline.data import _collate_records, load_split_dataset, make_dataloader, validate_disjoint_split_keys
from sex_age_baseline.model import SexAgeMLP
import sex_age_baseline.runtime as baseline_runtime
from sex_age_baseline.runtime import evaluate_model, masked_multilabel_bce


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload))
    return path


def _write_index(path: Path, rows: list[str]) -> Path:
    path.write_text("eid,split,age,sex\n" + "\n".join(rows) + "\n")
    return path


def _write_survival_sidecars(
    tmp_path: Path,
    keys: list[str],
    disease_count: int = 2,
    diseases: list[str] | None = None,
) -> dict[str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    diseases = diseases or [f"d{i + 1}" for i in range(disease_count)]
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
    task_block = {"key_column": "eid", **sidecars, "covariates": ["age", "sex"], "covariate_embedding_dim": 4}
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
        finetune["survival"] = task_block
    else:
        finetune["multilabel"] = task_block
        finetune["loss"] = {"pos_weight": None}
    return {
        "model": {
            "name": "sex_age_mlp",
            "head": {
                "name": "classification",
                "hidden_dim": 8,
                "dropout": 0.0,
                "act": "elu",
                "kwargs": {"num_layers": 2},
            },
        },
        "data": {
            "backend": "npz",
            "finetune_data_index": str(index),
            "finetune_preset_path": None,
            "kaldi_data_root": None,
            "kaldi_manifest": None,
            "train_dataset_names": None,
            "test_dataset_names": None,
        },
        "finetune": finetune,
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


def _parse_metadata_rows(rows: list[str]) -> list[dict[str, str]]:
    parsed = []
    for row in rows:
        eid, split, age, sex = row.split(",")
        parsed.append({"eid": eid, "split": split, "age": age, "sex": sex})
    return parsed


def _write_preset(path: Path, rows: list[str]) -> Path:
    # A signal preset embeds the survival labels of each window next to its metadata.
    labels = {"event_time": [10.0, 11.0], "is_event": [1.0, 1.0], "has_label": [1.0, 1.0]}
    samples = [
        SampleIndex(id=row["eid"], path="ignored.npz", start=0, end=1, metadata={**row, **labels})
        for row in _parse_metadata_rows(rows)
    ]
    with path.open("wb") as file_obj:
        pickle.dump(samples, file_obj)
    return path


def _write_kaldi_root(root: Path, rows: list[str]) -> tuple[Path, Path]:
    root.mkdir()
    manifest = {"splits": {}}
    by_split: dict[str, list[dict[str, str]]] = {}
    for row in _parse_metadata_rows(rows):
        by_split.setdefault(row["split"], []).append(row)
    for split, split_rows in by_split.items():
        split_csv = root / f"{split}.csv"
        split_csv.write_text(
            "eid,split,age,sex\n"
            + "\n".join(",".join([row["eid"], row["split"], row["age"], row["sex"]]) for row in split_rows)
            + "\n"
        )
        manifest["splits"][split] = {"manifest": split_csv.name}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return root, manifest_path


def _write_config_for_data(tmp_path: Path, rows: list[str], data: dict, task_type: str = "survival") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    keys = [row.split(",", 1)[0] for row in rows]
    sidecars = (
        _write_survival_sidecars(tmp_path, sorted(set(keys)))
        if task_type == "survival"
        else _write_multilabel_sidecars(tmp_path, sorted(set(keys)))
    )
    payload = _base_payload(tmp_path / "unused-index.csv", sidecars, task_type)
    payload["data"].update(data)
    return _write_yaml(tmp_path / f"{task_type}-{data['backend']}.yaml", payload)


def _backend_data_config(tmp_path: Path, rows: list[str], backend: str) -> dict:
    if backend == "npz_index":
        index = _write_index(tmp_path / "index.csv", rows)
        return {
            "backend": "npz",
            "finetune_data_index": str(index),
            "finetune_preset_path": None,
            "kaldi_data_root": None,
            "kaldi_manifest": None,
        }
    if backend == "npz_preset":
        preset = _write_preset(tmp_path / "preset.pkl", rows)
        return {
            "backend": "npz",
            "finetune_data_index": None,
            "finetune_preset_path": str(preset),
            "kaldi_data_root": None,
            "kaldi_manifest": None,
        }
    if backend == "kaldi":
        kaldi_root, kaldi_manifest = _write_kaldi_root(tmp_path / "kaldi", rows)
        return {
            "backend": "kaldi",
            "finetune_data_index": None,
            "finetune_preset_path": None,
            "kaldi_data_root": str(kaldi_root),
            "kaldi_manifest": str(kaldi_manifest),
        }
    raise ValueError(f"Unsupported test backend: {backend}")


def _subjects(dataset) -> list[tuple[str, float, int]]:
    """(key, age, sex) per record, encoded exactly as the batches the model receives."""
    if not len(dataset):
        return []
    batch = _collate_records(dataset.records)
    return list(zip(batch["key"], batch["metadata"]["age"].tolist(), batch["metadata"]["sex"].tolist()))


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
        test_after_fit=test_after_fit,
        test_all_checkpoints_after_fit=False,
        ckpt_every_n_epochs=1,
        export_predictions=False,
        wandb_mode="disabled",
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
    cfg = load_config(config)

    train = load_split_dataset(cfg, "train", sources=None)
    val = load_split_dataset(cfg, "val", sources=None)

    assert _subjects(train) == [("001", 50.0, 0)]
    assert _subjects(val) == [("002", 60.0, 1)]


@pytest.mark.parametrize("backend", ["npz_index", "npz_preset", "kaldi"])
def test_metadata_backends_produce_identical_subject_records(tmp_path: Path, backend: str):
    rows = ["001,train,50,0", "001,train,50,0", "002,val,60,1", "003,test,55,0"]
    data = _backend_data_config(tmp_path, rows, backend)
    cfg = load_config(_write_config_for_data(tmp_path, rows, data))

    dataset = load_split_dataset(cfg, "train", sources=None)

    assert _subjects(dataset) == [("001", 50.0, 0)]
    assert dataset.records[0].event_time.tolist() == [10.0, 11.0]


def test_kaldi_rows_belong_to_the_split_whose_manifest_lists_them(tmp_path: Path):
    # KaldiPSGDataset reads only the requested split's manifest CSV and keeps the rows whose split column names it.
    kaldi_root = tmp_path / "kaldi"
    kaldi_root.mkdir()
    (kaldi_root / "train.csv").write_text("eid,split,age,sex\n001,train,50,0\n002,val,60,1\n")
    (kaldi_root / "val.csv").write_text("eid,split,age,sex\n003,val,55,0\n")
    kaldi_manifest = kaldi_root / "manifest.json"
    kaldi_manifest.write_text(
        json.dumps({"splits": {"train": {"manifest": "train.csv"}, "val": {"manifest": "val.csv"}}})
    )
    data = {
        "backend": "kaldi",
        "finetune_data_index": None,
        "kaldi_data_root": str(kaldi_root),
        "kaldi_manifest": str(kaldi_manifest),
    }
    cfg = load_config(_write_config_for_data(tmp_path, ["001,train,50,0", "002,val,60,1", "003,val,55,0"], data))

    assert _subjects(load_split_dataset(cfg, "train", sources=None)) == [("001", 50.0, 0)]
    assert _subjects(load_split_dataset(cfg, "val", sources=None)) == [("003", 55.0, 0)]


@pytest.mark.parametrize("backend", ["npz_index", "npz_preset", "kaldi"])
def test_conflicting_duplicate_metadata_fails(tmp_path: Path, backend: str):
    rows = ["001,train,50,0", "001,train,51,0"]
    data = _backend_data_config(tmp_path, rows, backend)
    cfg = load_config(_write_config_for_data(tmp_path, rows, data))

    with pytest.raises(ValueError, match="conflicting age"):
        load_split_dataset(cfg, "train", sources=None)


@pytest.mark.parametrize("backend", ["npz_preset", "kaldi"])
def test_missing_metadata_columns_fail_for_non_index_backends(tmp_path: Path, backend: str):
    rows = ["001,train,50,0"]
    if backend == "npz_preset":
        sample = SampleIndex(id="001", path="ignored.npz", start=0, end=1, metadata={"eid": "001", "split": "train"})
        preset = tmp_path / "preset.pkl"
        with preset.open("wb") as file_obj:
            pickle.dump([sample], file_obj)
        data = {"backend": "npz", "finetune_data_index": None, "finetune_preset_path": str(preset)}
    else:
        kaldi_root = tmp_path / "kaldi"
        kaldi_root.mkdir()
        (kaldi_root / "train.csv").write_text("eid,split\n001,train\n")
        kaldi_manifest = kaldi_root / "manifest.json"
        kaldi_manifest.write_text(json.dumps({"splits": {"train": {"manifest": "train.csv"}}}))
        data = {
            "backend": "kaldi",
            "finetune_data_index": None,
            "kaldi_data_root": str(kaldi_root),
            "kaldi_manifest": str(kaldi_manifest),
        }
    cfg = load_config(_write_config_for_data(tmp_path, rows, data))

    with pytest.raises(ValueError, match=r"missing required columns: \['age', 'sex'\]"):
        load_split_dataset(cfg, "train", sources=None)


@pytest.mark.parametrize("backend", ["npz_index", "npz_preset", "kaldi"])
def test_rows_with_invalid_covariates_are_dropped_from_the_selected_split(tmp_path: Path, backend: str):
    # The signal loader's required-metadata filter drops these rows; the baseline keeps the same cohort.
    rows = ["001,train,50,0", "004,train,61,unknown", "002,val,60,1", "003,test,,unknown"]
    data = _backend_data_config(tmp_path, rows, backend)
    cfg = load_config(_write_config_for_data(tmp_path, rows, data))

    assert _subjects(load_split_dataset(cfg, "train", sources=None)) == [("001", 50.0, 0)]
    assert _subjects(load_split_dataset(cfg, "val", sources=None)) == [("002", 60.0, 1)]
    assert _subjects(load_split_dataset(cfg, "test", sources=None)) == []


def test_unused_split_duplicate_metadata_does_not_block_selected_split(tmp_path: Path):
    config = _write_config(tmp_path, ["001,train,50,0", "002,val,60,1", "001,test,55,1"])
    cfg = load_config(config)
    datasets = {split: load_split_dataset(cfg, split, sources=None) for split in ("train", "val")}

    assert _subjects(datasets["train"]) == [("001", 50.0, 0)]
    assert _subjects(datasets["val"]) == [("002", 60.0, 1)]
    validate_disjoint_split_keys(datasets)

    datasets["test"] = load_split_dataset(cfg, "test", sources=None)
    with pytest.raises(ValueError, match="multiple loaded splits"):
        validate_disjoint_split_keys(datasets)


@pytest.mark.parametrize("backend", ["npz_index", "npz_preset", "kaldi"])
def test_split_key_reuse_counts_only_retained_rows(tmp_path: Path, backend: str):
    # 001's val row fails the covariate filter, so only its train row is in the effective cohort.
    rows = ["001,train,50,0", "001,val,bad,unknown", "002,val,60,1", "002,test,55,1"]
    data = _backend_data_config(tmp_path, rows, backend)
    cfg = load_config(_write_config_for_data(tmp_path, rows, data))
    datasets = {split: load_split_dataset(cfg, split, sources=None) for split in ("train", "val")}

    validate_disjoint_split_keys(datasets)

    datasets["test"] = load_split_dataset(cfg, "test", sources=None)
    with pytest.raises(ValueError, match="multiple loaded splits"):
        validate_disjoint_split_keys(datasets)


def test_train_rejects_empty_requested_split_before_creating_run_dir(tmp_path: Path, monkeypatch):
    config = _write_config(tmp_path, ["001,train,50,0", "002,test,60,1"])
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="split 'val' has no rows"):
        baseline_runtime.train_and_save(_runtime_args(config, tmp_path, version_name="empty-val"), cfg)
    assert not (tmp_path / "log-finetune" / "empty-val").exists()


def test_train_rejects_key_reused_across_train_val(tmp_path: Path, monkeypatch):
    config = _write_config(tmp_path, ["001,train,50,0", "001,val,50,0"])
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="multiple loaded splits"):
        baseline_runtime.train_and_save(_runtime_args(config, tmp_path, version_name="split-leak"), cfg)
    # Data errors surface before the single-use run root exists.
    assert not (tmp_path / "log-finetune" / "split-leak").exists()


@pytest.mark.parametrize("task_type", ["survival", "multilabel_classification"])
def test_model_forward_shape(tmp_path: Path, task_type: str):
    config = _write_config(tmp_path, ["001,train,50,0", "002,train,60,1"], task_type=task_type)
    cfg = load_config(config)
    model = SexAgeMLP(cfg)

    logits = model({"age": torch.tensor([50.0, 60.0]), "sex": torch.tensor([0, 1])})

    assert tuple(logits.shape) == (2, 2)


def test_cox_eval_reports_val_c_index(tmp_path: Path):
    pytest.importorskip("sksurv.metrics")
    config = _write_config(
        tmp_path,
        ["001,val,50,0", "002,val,60,1", "003,val,55,0"],
        task_type="survival",
    )
    cfg = load_config(config)
    dataset = load_split_dataset(cfg, "val", sources=None)
    loader = make_dataloader(dataset, batch_size=3, num_workers=0, shuffle=False)
    model = SexAgeMLP(cfg)

    result = evaluate_model(model, loader, cfg, device=torch.device("cpu"), stage="val", export_predictions=True)

    assert "val_c_index" in result.metrics
    assert result.survival_per_disease_rows
    assert [row["path"] for row in result.prediction_rows] == ["001", "002", "003"]
    assert all(row["path"] == row["survival_key"] for row in result.prediction_rows)
    assert all(row["n_windows"] == 1 and row["token_starts"] == [0] for row in result.prediction_rows)


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
    cfg = load_config(config)
    dataset = load_split_dataset(cfg, "val", sources=None)
    loader = make_dataloader(dataset, batch_size=4, num_workers=0, shuffle=False)
    model = SexAgeMLP(cfg)

    result = evaluate_model(model, loader, cfg, device=torch.device("cpu"), stage="val", export_predictions=True)

    assert "val_macro_auroc" in result.metrics
    assert "val_micro_auroc" in result.metrics
    assert result.multilabel_per_disease_rows
    assert [row["paths"] for row in result.prediction_rows] == [["001"], ["002"], ["003"], ["004"]]
    assert all(row["path"] == row["multilabel_key"] for row in result.prediction_rows)


def test_train_rejects_non_empty_run_dir_before_loading_data(tmp_path: Path, monkeypatch):
    config = _write_config(tmp_path, ["001,train,50,0"], task_type="multilabel_classification")
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "log-finetune" / "reused"
    run_dir.mkdir(parents=True)
    (run_dir / "marker.txt").write_text("old run\n")

    with pytest.raises(FileExistsError, match="Use a new --version-name"):
        baseline_runtime.train_and_save(_runtime_args(config, tmp_path, version_name="reused"), cfg)


def test_train_rejects_run_directory_symlink_before_loading_data(tmp_path: Path, monkeypatch):
    config = _write_config(tmp_path, ["001,train,50,0"], task_type="multilabel_classification")
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "empty-target"
    target.mkdir()
    run_dir = tmp_path / "log-finetune" / "linked"
    run_dir.parent.mkdir()
    run_dir.symlink_to(target, target_is_directory=True)

    with pytest.raises(FileExistsError, match="run directory must not be a symlink"):
        baseline_runtime.train_and_save(_runtime_args(config, tmp_path, version_name="linked"), cfg)

    assert not any(target.iterdir())


def test_train_fails_when_configured_monitor_is_missing(tmp_path: Path, monkeypatch):
    config = _write_config(
        tmp_path,
        ["001,train,50,0", "002,train,60,1", "003,val,55,0", "004,val,65,1"],
        task_type="multilabel_classification",
    )
    payload = yaml.safe_load(config.read_text())
    payload["finetune"]["task"]["monitor"] = "val_missing_metric"
    _write_yaml(config, payload)
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="val_missing_metric.*Available metrics"):
        baseline_runtime.train_and_save(_runtime_args(config, tmp_path, version_name="missing-monitor"), cfg)


def test_train_fails_without_finite_best_checkpoint(tmp_path: Path, monkeypatch):
    config = _write_config(
        tmp_path,
        ["001,train,50,0", "002,val,60,1"],
        task_type="multilabel_classification",
    )
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="No finite best checkpoint"):
        args = _runtime_args(config, tmp_path, version_name="no-finite-best")
        args.batch_size = 1
        baseline_runtime.train_and_save(args, cfg)

    assert not (tmp_path / "log-finetune" / "no-finite-best" / "checkpoints" / "best.ckpt").exists()


@pytest.mark.parametrize("test_after_fit", [True, False])
def test_zero_epoch_train_requires_checkpoint(tmp_path: Path, monkeypatch, test_after_fit: bool):
    config = _write_config(
        tmp_path,
        ["001,train,50,0", "002,val,60,1"],
        task_type="multilabel_classification",
    )
    cfg = load_config(config)
    version_name = f"zero-epoch-no-ckpt-{test_after_fit}"
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="--epochs 0 requires --ckpt-path"):
        baseline_runtime.train_and_save(
            _runtime_args(
                config,
                tmp_path,
                version_name=version_name,
                epochs=0,
                test_after_fit=test_after_fit,
            ),
            cfg,
        )

    assert not (tmp_path / "log-finetune" / version_name).exists()


@pytest.mark.parametrize("test_after_fit", [True, False])
def test_negative_epochs_fail_before_run_directory(tmp_path: Path, monkeypatch, test_after_fit: bool):
    config = _write_config(
        tmp_path,
        ["001,train,50,0", "002,val,60,1"],
        task_type="multilabel_classification",
    )
    cfg = load_config(config)
    version_name = f"negative-epochs-{test_after_fit}"
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="--epochs must be non-negative"):
        baseline_runtime.train_and_save(
            _runtime_args(
                config,
                tmp_path,
                version_name=version_name,
                epochs=-1,
                test_after_fit=test_after_fit,
            ),
            cfg,
        )

    assert not (tmp_path / "log-finetune" / version_name).exists()


def test_all_checkpoint_mode_requires_test_after_fit_positive_epochs_and_every_epoch_checkpoints(tmp_path: Path):
    config = tmp_path / "config.yaml"
    args = _runtime_args(config, tmp_path, version_name="invalid-all-checkpoints", test_after_fit=False)
    args.test_all_checkpoints_after_fit = True
    with pytest.raises(ValueError, match="requires --test-after-fit"):
        baseline_runtime.train_and_save(args, object())

    args.test_after_fit = True
    args.epochs = 0
    with pytest.raises(ValueError, match="requires --epochs greater than 0"):
        baseline_runtime.train_and_save(args, object())

    args.epochs = 1
    args.ckpt_every_n_epochs = 2
    with pytest.raises(ValueError, match="requires --ckpt-every-n-epochs 1"):
        baseline_runtime.train_and_save(args, object())

    assert not (tmp_path / "log-finetune" / "invalid-all-checkpoints").exists()


@pytest.mark.parametrize("ckpt_every_n_epochs", [0, -1])
def test_nonpositive_checkpoint_interval_fails_before_run_directory(
    tmp_path: Path,
    monkeypatch,
    ckpt_every_n_epochs: int,
):
    config = _write_config(
        tmp_path,
        ["001,train,50,0", "002,val,60,1"],
        task_type="multilabel_classification",
    )
    cfg = load_config(config)
    version_name = f"bad-ckpt-interval-{ckpt_every_n_epochs}"
    args = _runtime_args(config, tmp_path, version_name=version_name)
    args.ckpt_every_n_epochs = ckpt_every_n_epochs
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="--ckpt-every-n-epochs must be positive"):
        baseline_runtime.train_and_save(args, cfg)

    assert not (tmp_path / "log-finetune" / version_name).exists()


def test_zero_epoch_checkpoint_eval_skips_train_val_splits(tmp_path: Path, monkeypatch):
    config = _write_config(
        tmp_path,
        [
            "001,test,50,0",
            "002,test,60,1",
            "003,test,55,0",
            "004,test,65,1",
        ],
        task_type="multilabel_classification",
    )
    cfg = load_config(config)
    ckpt = tmp_path / "model.ckpt"
    baseline_runtime.save_checkpoint(ckpt, SexAgeMLP(cfg), cfg, epoch=0, global_step=0, metrics={})
    monkeypatch.chdir(tmp_path)
    args = _runtime_args(config, tmp_path, version_name="zero-epoch-test-only", epochs=0, test_after_fit=True)
    args.ckpt_path = str(ckpt)

    baseline_runtime.train_and_save(args, cfg)

    manifest = json.loads((tmp_path / "log-finetune" / "zero-epoch-test-only" / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["best_model_path"] == str(ckpt)


def test_checkpoint_rejects_incompatible_label_order(tmp_path: Path):
    rows = ["001,test,50,0", "002,test,60,1"]
    index = _write_index(tmp_path / "index.csv", rows)
    saved_config = _write_yaml(
        tmp_path / "saved.yaml",
        _base_payload(
            index,
            _write_survival_sidecars(tmp_path / "saved-sidecars", ["001", "002"], diseases=["d1", "d2"]),
            "survival",
        ),
    )
    current_config = _write_yaml(
        tmp_path / "current.yaml",
        _base_payload(
            index,
            _write_survival_sidecars(tmp_path / "current-sidecars", ["001", "002"], diseases=["d2", "d1"]),
            "survival",
        ),
    )
    saved_cfg = load_config(saved_config)
    current_cfg = load_config(current_config)
    ckpt = tmp_path / "model.ckpt"
    baseline_runtime.save_checkpoint(ckpt, SexAgeMLP(saved_cfg), saved_cfg, epoch=0, global_step=0, metrics={})

    with pytest.raises(ValueError, match="label contract"):
        baseline_runtime.load_checkpoint(SexAgeMLP(current_cfg), ckpt, device=torch.device("cpu"), cfg=current_cfg)


def test_checkpoint_rejects_incompatible_model_contract(tmp_path: Path):
    rows = ["001,test,50,0", "002,test,60,1"]
    index = _write_index(tmp_path / "index.csv", rows)
    sidecars = _write_survival_sidecars(tmp_path / "sidecars", ["001", "002"])
    saved_config = _write_yaml(tmp_path / "saved.yaml", _base_payload(index, sidecars, "survival"))
    current_payload = _base_payload(index, sidecars, "survival")
    # Same parameter shapes, different frozen covariate scaling.
    current_payload["finetune"]["survival"]["covariate_normalization"] = {"age": {"mean": 50.0, "std": 10.0}}
    current_config = _write_yaml(tmp_path / "current.yaml", current_payload)
    saved_cfg = load_config(saved_config)
    current_cfg = load_config(current_config)
    ckpt = tmp_path / "model.ckpt"
    baseline_runtime.save_checkpoint(ckpt, SexAgeMLP(saved_cfg), saved_cfg, epoch=0, global_step=0, metrics={})

    with pytest.raises(ValueError, match="model contract"):
        baseline_runtime.load_checkpoint(SexAgeMLP(current_cfg), ckpt, device=torch.device("cpu"), cfg=current_cfg)


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
    cfg = load_config(config)
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


def test_all_checkpoint_test_after_fit_records_every_epoch_and_preserves_best_metrics(tmp_path: Path, monkeypatch):
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
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lexical").mkdir()
    frozen_checkpoint_dir = tmp_path / "lexical" / ".." / "log-finetune" / "all-checkpoints" / "checkpoints"
    monkeypatch.setenv("_SLEEP2VEC_FROZEN_CHECKPOINT_DIR", str(frozen_checkpoint_dir))
    args = _runtime_args(config, tmp_path, version_name="all-checkpoints", epochs=2, test_after_fit=True)
    args.test_all_checkpoints_after_fit = True
    args.export_predictions = True
    events = []
    original_save_prediction = baseline_runtime.save_prediction_csv
    original_save_survival = baseline_runtime.save_survival_per_disease_metrics_csv
    original_save_multilabel = baseline_runtime.save_multilabel_per_disease_metrics_csv
    original_save_matrix = baseline_runtime.save_result_rows_csv
    original_save_manifest = baseline_runtime.save_training_run_manifest
    original_evaluate = baseline_runtime._test

    def evaluate_with_survival_rows(*call_args, **call_kwargs):
        result = original_evaluate(*call_args, **call_kwargs)
        result.survival_per_disease_rows = [{"disease": "unit"}]
        return result

    def save_prediction(*call_args, **call_kwargs):
        events.append("prediction")
        return original_save_prediction(*call_args, **call_kwargs)

    def save_multilabel(*call_args, **call_kwargs):
        events.append("multilabel")
        return original_save_multilabel(*call_args, **call_kwargs)

    def save_survival(*call_args, **call_kwargs):
        events.append("survival")
        return original_save_survival(*call_args, **call_kwargs)

    def save_matrix(*call_args, **call_kwargs):
        events.append("matrix")
        return original_save_matrix(*call_args, **call_kwargs)

    def save_manifest(*call_args, **call_kwargs):
        events.append("manifest")
        return original_save_manifest(*call_args, **call_kwargs)

    monkeypatch.setattr(baseline_runtime, "_test", evaluate_with_survival_rows)
    monkeypatch.setattr(baseline_runtime, "save_prediction_csv", save_prediction)
    monkeypatch.setattr(baseline_runtime, "save_survival_per_disease_metrics_csv", save_survival)
    monkeypatch.setattr(baseline_runtime, "save_multilabel_per_disease_metrics_csv", save_multilabel)
    monkeypatch.setattr(baseline_runtime, "save_result_rows_csv", save_matrix)
    monkeypatch.setattr(baseline_runtime, "save_training_run_manifest", save_manifest)

    baseline_runtime.train_and_save(args, cfg)

    run_dir = tmp_path / "log-finetune" / "all-checkpoints"
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    checkpoint_results = manifest["checkpoint_test_results"]
    assert manifest["test_all_checkpoints_after_fit"] is True
    assert {row["epoch"] for row in checkpoint_results} == {0, 1}
    assert {row["checkpoint_path"] for row in checkpoint_results} == {
        str(frozen_checkpoint_dir / "epoch=00.ckpt"),
        str(frozen_checkpoint_dir / "epoch=01.ckpt"),
    }
    assert {Path(row["checkpoint_path"]).name for row in checkpoint_results} == {"epoch=00.ckpt", "epoch=01.ckpt"}
    saved_checkpoint = torch.load(run_dir / "checkpoints" / "best.ckpt", weights_only=False)
    best_epoch = saved_checkpoint["epoch"]
    assert len(saved_checkpoint["optimizer_states"]) == 1
    assert len(saved_checkpoint["lr_schedulers"]) == 1
    assert saved_checkpoint["optimizer_states"][0]["param_groups"][0]["betas"] == (0.9, 0.95)
    assert saved_checkpoint["lr_schedulers"][0]["last_epoch"] == saved_checkpoint["global_step"]
    assert checkpoint_results[-1]["epoch"] == best_epoch
    best_result = next(row for row in checkpoint_results if row["epoch"] == best_epoch)
    assert manifest["metrics"] == best_result["metrics"]
    result_rows = pd.read_csv(args.results_csv_path)
    assert len(result_rows) == 2
    assert set(result_rows["ckpt_path"]) == {row["checkpoint_path"] for row in checkpoint_results}
    # Like sleep2vec, every evaluated checkpoint contributes prediction and per-disease rows.
    prediction_rows = pd.read_csv(run_dir / "predictions.csv")
    assert manifest["prediction_csv_path"] == str(Path("log-finetune") / "all-checkpoints" / "predictions.csv")
    assert len(prediction_rows) == 4
    assert set(prediction_rows["ckpt_path"]) == {row["checkpoint_path"] for row in checkpoint_results}
    per_disease_rows = pd.read_csv(run_dir / "multilabel_per_disease_metrics.csv")
    assert set(per_disease_rows["ckpt_path"]) == {row["checkpoint_path"] for row in checkpoint_results}
    assert events == [
        "prediction",
        "prediction",
        "survival",
        "multilabel",
        "survival",
        "multilabel",
        "matrix",
        "manifest",
    ]


def test_finetune_writes_predictions_only_with_export_flag(tmp_path: Path, monkeypatch):
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
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)

    baseline_runtime.train_and_save(_runtime_args(config, tmp_path, version_name="no-export", test_after_fit=True), cfg)
    exported_args = _runtime_args(config, tmp_path, version_name="export", test_after_fit=True)
    exported_args.export_predictions = True
    baseline_runtime.train_and_save(exported_args, cfg)

    default_dir = tmp_path / "log-finetune" / "no-export"
    assert not (default_dir / "predictions.csv").exists()
    assert json.loads((default_dir / "run_manifest.json").read_text())["prediction_csv_path"] == ""
    assert (default_dir / "multilabel_per_disease_metrics.csv").is_file()
    assert len(pd.read_csv(tmp_path / "log-finetune" / "export" / "predictions.csv")) == 2


@pytest.mark.parametrize(("test_after_fit", "status"), [(True, "completed"), (False, "skipped_test")])
def test_finetune_finishes_wandb_before_terminal_manifest(
    tmp_path: Path, monkeypatch, test_after_fit: bool, status: str
):
    config = _write_config(
        tmp_path,
        ["001,train,50,0", "002,train,60,1", "003,val,55,0", "004,val,65,1", "005,test,58,0", "006,test,68,1"],
        task_type="multilabel_classification",
    )
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)
    events = []
    monkeypatch.setattr(baseline_runtime, "_finish_wandb_run", lambda preexisting, stage: events.append("finish"))
    monkeypatch.setattr(
        baseline_runtime,
        "save_training_run_manifest",
        lambda *args, **kwargs: events.append(f"manifest:{kwargs['status']}"),
    )

    baseline_runtime.train_and_save(
        _runtime_args(config, tmp_path, version_name="ordered", test_after_fit=test_after_fit), cfg
    )

    # W&B finalization can fail, so the terminal manifest is written only after it succeeds.
    assert events[:2] == ["finish", f"manifest:{status}"]


def test_all_checkpoint_test_rejects_missing_validation_best_periodic_checkpoint(tmp_path: Path, monkeypatch):
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
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)
    args = _runtime_args(config, tmp_path, version_name="missing-best-periodic", epochs=2, test_after_fit=True)
    args.test_all_checkpoints_after_fit = True
    original_save_checkpoint = baseline_runtime.ModelCheckpoint._save_checkpoint

    def omit_first_periodic(self, trainer, path):
        if Path(path).name.startswith("epoch="):
            return None
        return original_save_checkpoint(self, trainer, path)

    monkeypatch.setattr(baseline_runtime.ModelCheckpoint, "_save_checkpoint", omit_first_periodic)

    with pytest.raises(ValueError, match="No regular epoch=.*checkpoints"):
        baseline_runtime.train_and_save(args, cfg)


def test_all_checkpoint_test_failure_preserves_existing_results_csv(tmp_path: Path, monkeypatch):
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
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)
    args = _runtime_args(config, tmp_path, version_name="failed-all-checkpoints", epochs=2, test_after_fit=True)
    args.test_all_checkpoints_after_fit = True
    args.results_csv_path.write_text("experiment_version,test_loss\nold,1.0\n")
    results_before = args.results_csv_path.read_bytes()
    original_evaluate = baseline_runtime._test
    test_calls = 0

    def fail_second_test(*call_args, **call_kwargs):
        nonlocal test_calls
        test_calls += 1
        if test_calls == 2:
            raise RuntimeError("second checkpoint test failed")
        return original_evaluate(*call_args, **call_kwargs)

    monkeypatch.setattr(baseline_runtime, "_test", fail_second_test)

    with pytest.raises(RuntimeError, match="second checkpoint test failed"):
        baseline_runtime.train_and_save(args, cfg)

    assert test_calls == 2
    assert args.results_csv_path.read_bytes() == results_before


@pytest.mark.parametrize(
    ("artifact_writer", "emit_survival_rows"),
    (
        ("save_prediction_csv", False),
        ("save_survival_per_disease_metrics_csv", True),
        ("save_multilabel_per_disease_metrics_csv", False),
    ),
)
def test_all_checkpoint_artifact_failure_preserves_existing_results_csv(
    artifact_writer: str,
    emit_survival_rows: bool,
    tmp_path: Path,
    monkeypatch,
):
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
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)
    args = _runtime_args(config, tmp_path, version_name=f"failed-{artifact_writer}", epochs=2, test_after_fit=True)
    args.test_all_checkpoints_after_fit = True
    args.export_predictions = True
    args.results_csv_path.write_text("experiment_version,test_loss\nold,1.0\n")
    results_before = args.results_csv_path.read_bytes()

    def fail_artifact(*_args, **_kwargs):
        raise RuntimeError("required artifact failed")

    if emit_survival_rows:
        original_evaluate = baseline_runtime._test

        def evaluate_with_survival_rows(*call_args, **call_kwargs):
            result = original_evaluate(*call_args, **call_kwargs)
            result.survival_per_disease_rows = [{"disease": "unit"}]
            return result

        monkeypatch.setattr(baseline_runtime, "_test", evaluate_with_survival_rows)
    monkeypatch.setattr(baseline_runtime, artifact_writer, fail_artifact)

    with pytest.raises(RuntimeError, match="required artifact failed"):
        baseline_runtime.train_and_save(args, cfg)

    assert args.results_csv_path.read_bytes() == results_before
    assert not (tmp_path / "log-finetune" / args.version_name / "run_manifest.json").exists()


def test_infer_run_inference_callable_validates_and_delegates(tmp_path: Path, monkeypatch):
    import sex_age_baseline.infer as infer_mod

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("placeholder")
    config = tmp_path / "config.yaml"
    config.write_text("model: {}\n")
    cfg = object()
    calls = []

    def fake_load_config(path):
        calls.append(("load_config", path))
        return cfg

    def fake_run_inference_and_save(args, loaded_cfg):
        calls.append(("run", args.ckpt_path, args.device, loaded_cfg))

    monkeypatch.setattr(infer_mod, "load_config", fake_load_config)
    monkeypatch.setattr(infer_mod, "run_inference_and_save", fake_run_inference_and_save)
    args = Namespace(
        config=config,
        ckpt_path=str(ckpt),
        label_name="unit",
        inference_preset_path=None,
        eval_split="val",
        batch_size=2,
        num_workers=0,
        devices=[0],
        accelerator="cpu",
        device="cuda",
        precision="bf16-mixed",
        lr=1e-6,
        weight_decay=1e-5,
        avg_ckpts=1,
        avg_ckpt_dir=None,
        seed=4523,
        wandb_mode=None,
    )

    infer_mod.run_inference(args)

    assert calls == [
        ("load_config", config),
        ("run", str(ckpt), "cpu", cfg),
    ]


def test_inference_runtime_passes_custom_results_root(tmp_path: Path, monkeypatch):
    results_root = tmp_path / "attempt-001"
    captured = {}
    args = Namespace(
        ckpt_path=str(tmp_path / "model.ckpt"),
        label_name="age",
        eval_split="test",
        batch_size=2,
        num_workers=0,
        device="cpu",
        seed=4523,
        inference_preset_path=None,
        override_dataset_names=None,
        results_root=results_root,
        avg_ckpts=1,
        wandb=False,
    )
    cfg = Namespace(
        finetune=Namespace(task=Namespace(type="regression")),
        data=Namespace(train_dataset_names=None, test_dataset_names=None),
    )
    result = baseline_runtime.EvaluationResult(
        metrics={"test_mae": 1.0},
        prediction_rows=[],
        survival_per_disease_rows=[],
        multilabel_per_disease_rows=[],
    )

    class _DummyTrainer:
        is_global_zero = True

        def test(self, module, *, dataloaders, verbose):
            captured["stage"] = module.evaluation_stage
            module.evaluation_result = result

    def _prepare_paths(runtime_args, *, namespace, root, checkpoint_paths):
        captured["namespace"] = namespace
        captured["root"] = root
        captured["checkpoint_paths"] = checkpoint_paths
        for name in (
            "inference_metrics_csv_path",
            "inference_overview_csv_path",
            "inference_prediction_csv_path",
            "inference_survival_per_disease_metrics_csv_path",
            "inference_multilabel_per_disease_metrics_csv_path",
        ):
            setattr(runtime_args, name, root / f"{name}.csv")

    monkeypatch.setattr(baseline_runtime, "configure_result_args", lambda *args: None)
    monkeypatch.setattr(baseline_runtime, "_seed_everything", lambda *args: None)
    monkeypatch.setattr(baseline_runtime, "BaselineModule", lambda cfg, args: Namespace(model="model"))
    monkeypatch.setattr(baseline_runtime, "_trainer", lambda args: _DummyTrainer())
    monkeypatch.setattr(baseline_runtime, "_load_inference_checkpoint", lambda *args: None)
    monkeypatch.setattr(baseline_runtime, "prepare_inference_result_paths", _prepare_paths)
    monkeypatch.setattr(baseline_runtime, "_required_dataset", lambda *args, **kwargs: "dataset")
    monkeypatch.setattr(baseline_runtime, "make_dataloader", lambda *args, **kwargs: "loader")
    for writer in (
        "save_result_csv",
        "save_prediction_csv",
        "save_survival_per_disease_metrics_csv",
        "save_multilabel_per_disease_metrics_csv",
        "save_inference_manifest",
    ):
        monkeypatch.setattr(baseline_runtime, writer, lambda *args, **kwargs: None)

    baseline_runtime.run_inference_and_save(args, cfg)

    assert captured == {
        "namespace": "sex_age_baseline",
        "root": results_root,
        "checkpoint_paths": None,
        "stage": "test",
    }


@pytest.mark.parametrize("checkpoint_cadence", [1, 2])
def test_validation_and_checkpoint_cadence_preserve_actual_last_state(tmp_path: Path, monkeypatch, checkpoint_cadence):
    config = _write_config(
        tmp_path,
        ["001,train,50,0", "002,train,60,1", "003,val,55,0", "004,val,65,1"],
        task_type="multilabel_classification",
    )
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)
    args = _runtime_args(config, tmp_path, version_name="cadence", epochs=3)
    args.check_val_every_n_epoch = 2
    args.ckpt_every_n_epochs = checkpoint_cadence
    baseline_runtime.train_and_save(args, cfg)
    root = tmp_path / "log-finetune" / "cadence" / "checkpoints"
    expected = ["epoch=00.ckpt", "epoch=01.ckpt", "epoch=02.ckpt"] if checkpoint_cadence == 1 else ["epoch=01.ckpt"]
    assert sorted(path.name for path in root.glob("epoch=*.ckpt")) == expected
    best = torch.load(root / "best.ckpt", weights_only=False)
    last = torch.load(root / "last.ckpt", weights_only=False)
    assert best["epoch"] == 1
    assert last["epoch"] == 2
    assert "val_loss" in best["metrics"]
    assert "train_loss" in last["metrics"]
    assert not any(name.startswith("val_") for name in last["metrics"])
    for path in root.glob("epoch=*.ckpt"):
        state = torch.load(path, weights_only=False)
        validation = {name: value for name, value in state["metrics"].items() if name.startswith("val_")}
        if state["epoch"] == 1:
            assert validation == {name: value for name, value in best["metrics"].items() if name.startswith("val_")}
        else:
            assert validation == {}


@pytest.mark.parametrize("training", [True, False])
def test_trainer_sets_same_float32_matmul_precision_for_fit_and_inference(monkeypatch, training):
    previous = torch.get_float32_matmul_precision()
    observed = []
    monkeypatch.setattr(
        baseline_runtime.pl, "Trainer", lambda **kwargs: observed.append(torch.get_float32_matmul_precision())
    )
    try:
        torch.set_float32_matmul_precision("highest")
        baseline_runtime._trainer(
            Namespace(device="cpu", devices=[0], epochs=1, precision="32-true"), training=training
        )
        assert observed == ["high"]
    finally:
        torch.set_float32_matmul_precision(previous)


def test_trainer_forwards_runtime_settings_and_resolves_auto_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(baseline_runtime.pl, "Trainer", lambda **kwargs: kwargs)
    args = Namespace(
        device="cuda",
        accelerator="auto",
        devices=[0, 1],
        precision="32-true",
        epochs=4,
        accumulate_grad_batches=3,
        gradient_clip_val=0.5,
        check_val_every_n_epoch=2,
    )
    settings = baseline_runtime._trainer(args, training=True)
    assert settings["accelerator"] == "cpu"
    assert settings["devices"] == 2
    assert settings["strategy"] == "ddp"
    assert settings["precision"] == "32-true"
    assert settings["max_epochs"] == 4
    assert settings["accumulate_grad_batches"] == 3
    assert settings["gradient_clip_val"] == 0.5
    assert settings["check_val_every_n_epoch"] == 2


def test_trainer_maps_cuda_device_index_to_gpu_devices_and_forwards_logger(monkeypatch):
    monkeypatch.setattr(baseline_runtime.pl, "Trainer", lambda **kwargs: kwargs)
    logger = object()
    settings = baseline_runtime._trainer(
        Namespace(device="cuda:1", devices=[1], precision="bf16", epochs=1), training=True, logger=logger
    )
    assert settings["accelerator"] == "gpu"
    assert settings["devices"] == [1]
    assert settings["strategy"] == "auto"
    assert settings["logger"] is logger


@pytest.mark.parametrize(
    ("project", "mode", "expected_project"),
    [("frozen-project", "offline", "frozen-project"), (None, "disabled", "sex-age-baseline")],
)
def test_finetune_creates_wandb_logger_like_sleep2vec(tmp_path: Path, monkeypatch, project, mode, expected_project):
    config = _write_config(tmp_path, ["001,train,50,0", "002,val,60,1"], task_type="multilabel_classification")
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(baseline_runtime, "WandbLogger", lambda **kwargs: calls.append(kwargs) or "logger")

    def stop_before_fit(args, **kwargs):
        calls.append(kwargs["logger"])
        raise RuntimeError("stop before fit")

    monkeypatch.setattr(baseline_runtime, "_trainer", stop_before_fit)
    args = _runtime_args(config, tmp_path, version_name=None)
    args.version_prefix = "psg-finetune"
    args.version_tag = "tag"
    args.wandb_project = project
    args.wandb_group = "frozen-group"
    args.wandb_mode = mode

    with pytest.raises(RuntimeError, match="stop before fit"):
        baseline_runtime.train_and_save(args, cfg)

    assert calls == [
        {
            "project": expected_project,
            "name": "psg-finetune-sex-age-baseline-multilabel-unit-tag",
            "group": "frozen-group",
            "mode": mode,
            "save_dir": "./wandb_logs",
            "log_model": False,
        },
        "logger",
    ]


def test_accumulation_uses_actual_optimizer_step_budget(tmp_path: Path, monkeypatch):
    config = _write_config(
        tmp_path,
        ["001,train,50,0", "002,train,60,1", "003,val,55,0", "004,val,65,1"],
        task_type="multilabel_classification",
    )
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)
    args = _runtime_args(config, tmp_path, version_name="accumulation", epochs=2)
    args.batch_size = 1
    args.accumulate_grad_batches = 2
    args.warmup_steps = 0
    args.lr_decay_shape = "linear"
    args.lr_decay_floor = 0.2
    baseline_runtime.train_and_save(args, cfg)
    checkpoint = torch.load(
        tmp_path / "log-finetune" / "accumulation" / "checkpoints" / "last.ckpt", weights_only=False
    )
    assert checkpoint["global_step"] == 2
    assert checkpoint["lr_schedulers"][0]["last_epoch"] == 2
    assert checkpoint["optimizer_states"][0]["param_groups"][0]["lr"] == pytest.approx(args.lr * 0.2)


def _scheduler_training_args(config: Path, tmp_path: Path, version_name: str, **scheduler) -> Namespace:
    args = _runtime_args(config, tmp_path, version_name=version_name, epochs=2)
    args.batch_size = 1
    args.lr_decay_floor = 0.2
    for name, value in scheduler.items():
        setattr(args, name, value)
    return args


def test_wsd_scheduler_holds_lr_until_final_decay_steps(tmp_path: Path, monkeypatch):
    config = _write_config(
        tmp_path,
        ["001,train,50,0", "002,train,60,1", "003,val,55,0", "004,val,65,1"],
        task_type="multilabel_classification",
    )
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)
    args = _scheduler_training_args(config, tmp_path, "wsd", lr_scheduler="wsd", lr_decay_ratio=0.25)

    baseline_runtime.train_and_save(args, cfg)

    root = tmp_path / "log-finetune" / "wsd" / "checkpoints"
    stable = torch.load(root / "epoch=00.ckpt", weights_only=False)
    final = torch.load(root / "last.ckpt", weights_only=False)
    # Four optimizer steps with one final decay step: the plain decay schedule would already be at 0.6 here.
    assert stable["optimizer_states"][0]["param_groups"][0]["lr"] == pytest.approx(args.lr)
    assert final["optimizer_states"][0]["param_groups"][0]["lr"] == pytest.approx(args.lr * 0.2)


def test_plateau_scheduler_steps_once_per_validation_with_task_monitor(tmp_path: Path, monkeypatch):
    config = _write_config(
        tmp_path,
        ["001,train,50,0", "002,train,60,1", "003,val,55,0", "004,val,65,1"],
        task_type="multilabel_classification",
    )
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)
    args = _scheduler_training_args(
        config, tmp_path, "plateau", lr_scheduler="plateau", lr_plateau_factor=0.5, lr_plateau_patience=0
    )

    baseline_runtime.train_and_save(args, cfg)

    final = torch.load(tmp_path / "log-finetune" / "plateau" / "checkpoints" / "last.ckpt", weights_only=False)
    scheduler = final["lr_schedulers"][0]
    assert scheduler["mode"] == "max"
    assert scheduler["factor"] == 0.5
    assert scheduler["patience"] == 0
    assert scheduler["min_lrs"] == [pytest.approx(args.lr * 0.2)] * len(scheduler["min_lrs"])
    assert scheduler["last_epoch"] == 2


@pytest.mark.parametrize(
    ("scheduler", "message"),
    [
        ({"lr_scheduler": "wsd"}, "WSD requires lr_decay_ratio"),
        ({"lr_scheduler": "decay", "lr_decay_ratio": 0.5}, "lr_decay_ratio"),
        ({"lr_scheduler": "decay", "lr_plateau_patience": 3}, "lr_plateau"),
    ],
)
def test_scheduler_arguments_are_validated_before_run_directory(tmp_path: Path, monkeypatch, scheduler, message):
    config = _write_config(tmp_path, ["001,train,50,0", "002,val,60,1"], task_type="multilabel_classification")
    cfg = load_config(config)
    monkeypatch.chdir(tmp_path)
    args = _scheduler_training_args(config, tmp_path, "bad-scheduler", **scheduler)

    with pytest.raises(ValueError, match=message):
        baseline_runtime.train_and_save(args, cfg)

    assert not (tmp_path / "log-finetune" / "bad-scheduler").exists()


@pytest.mark.parametrize("failing_call", [None, "publish", "finish"])
def test_inference_withdraws_manifest_when_wandb_publication_fails(tmp_path: Path, monkeypatch, failing_call):
    events = []
    created_run = object()
    args = Namespace(
        ckpt_path=str(tmp_path / "model.ckpt"),
        label_name="age",
        eval_split="test",
        batch_size=2,
        num_workers=0,
        device="cpu",
        seed=4523,
        inference_preset_path=None,
        override_dataset_names=None,
        results_root=tmp_path / "results",
        avg_ckpts=1,
        wandb=True,
    )
    cfg = Namespace(
        finetune=Namespace(task=Namespace(type="regression")),
        data=Namespace(train_dataset_names=None, test_dataset_names=None),
    )
    result = baseline_runtime.EvaluationResult(
        metrics={"test_mae": 1.0}, prediction_rows=[], survival_per_disease_rows=[], multilabel_per_disease_rows=[]
    )

    class _DummyTrainer:
        is_global_zero = True

        def test(self, module, *, dataloaders, verbose):
            module.evaluation_result = result

    def _prepare_paths(runtime_args, *, namespace, root, checkpoint_paths):
        root.mkdir(parents=True)
        runtime_args.manifest_path = root / "run_manifest.json"
        for name in (
            "inference_metrics_csv_path",
            "inference_overview_csv_path",
            "inference_prediction_csv_path",
            "inference_survival_per_disease_metrics_csv_path",
            "inference_multilabel_per_disease_metrics_csv_path",
        ):
            setattr(runtime_args, name, root / f"{name}.csv")

    def _init_wandb(runtime_args):
        baseline_runtime.wandb.run = created_run
        return created_run

    def _record(name, clears_run=False):
        def _call(*call_args, **call_kwargs):
            events.append(name)
            if name == failing_call:
                raise RuntimeError(f"{name} failure")
            if clears_run:
                baseline_runtime.wandb.run = None

        return _call

    monkeypatch.setattr(baseline_runtime, "configure_result_args", lambda *args: None)
    monkeypatch.setattr(baseline_runtime, "_seed_everything", lambda *args: None)
    monkeypatch.setattr(baseline_runtime, "BaselineModule", lambda cfg, args: Namespace(model="model"))
    monkeypatch.setattr(baseline_runtime, "_trainer", lambda args: _DummyTrainer())
    monkeypatch.setattr(baseline_runtime, "_load_inference_checkpoint", lambda *args: None)
    monkeypatch.setattr(baseline_runtime, "prepare_inference_result_paths", _prepare_paths)
    monkeypatch.setattr(baseline_runtime, "_required_dataset", lambda *args, **kwargs: "dataset")
    monkeypatch.setattr(baseline_runtime, "make_dataloader", lambda *args, **kwargs: "loader")
    for writer in (
        "save_result_csv",
        "save_prediction_csv",
        "save_survival_per_disease_metrics_csv",
        "save_multilabel_per_disease_metrics_csv",
    ):
        monkeypatch.setattr(baseline_runtime, writer, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        baseline_runtime,
        "save_inference_manifest",
        lambda runtime_args, *a, **k: runtime_args.manifest_path.write_text("{}"),
    )
    monkeypatch.setattr(baseline_runtime.wandb, "run", None, raising=False)
    monkeypatch.setattr(baseline_runtime, "_init_wandb", _init_wandb)
    monkeypatch.setattr(baseline_runtime, "_log_inference_outputs_to_wandb", _record("publish"))
    monkeypatch.setattr(baseline_runtime.wandb, "finish", _record("finish", clears_run=True))

    if failing_call is None:
        baseline_runtime.run_inference_and_save(args, cfg)
    else:
        with pytest.raises(RuntimeError, match=f"{failing_call} failure"):
            baseline_runtime.run_inference_and_save(args, cfg)

    assert events[:2] == ["publish", "finish"]
    # run_manifest.json is uploaded with the artifact, so it is written first and withdrawn on failure.
    assert args.manifest_path.exists() is (failing_call is None)


def _save_epoch_checkpoints(root: Path, cfg, count: int = 3) -> list[SexAgeMLP]:
    models = []
    for epoch in range(count):
        torch.manual_seed(epoch)
        model = SexAgeMLP(cfg)
        baseline_runtime.save_checkpoint(
            root / f"epoch={epoch:02d}.ckpt", model, cfg, epoch=epoch, global_step=epoch, metrics={}
        )
        models.append(model)
    # Lightning aliases are not epoch files, so sleep2vec selection never averages them.
    baseline_runtime.save_checkpoint(root / "best.ckpt", models[0], cfg, epoch=0, global_step=0, metrics={})
    baseline_runtime.save_checkpoint(root / "last.ckpt", models[0], cfg, epoch=0, global_step=0, metrics={})
    return models


def _average_state(models: list[SexAgeMLP]) -> dict[str, torch.Tensor]:
    states = [model.state_dict() for model in models]
    return {name: sum(state[name] for state in states) / len(states) for name in states[0]}


def _inference_ckpt_args(ckpt_path: str, avg_ckpts: int, avg_ckpt_dir: Path | None) -> Namespace:
    return Namespace(ckpt_path=ckpt_path, avg_ckpts=avg_ckpts, avg_ckpt_dir=avg_ckpt_dir)


def test_inference_averages_epoch_checkpoints_ending_at_ckpt_path(tmp_path: Path):
    cfg = load_config(_write_config(tmp_path, ["001,test,50,0", "002,test,60,1"]))
    root = tmp_path / "checkpoints"
    models = _save_epoch_checkpoints(root, cfg)
    model = SexAgeMLP(cfg)

    selected = baseline_runtime._load_inference_checkpoint(
        model, _inference_ckpt_args(str(root / "epoch=01.ckpt"), 2, None), cfg
    )

    assert selected == [root / "epoch=00.ckpt", root / "epoch=01.ckpt"]
    expected = _average_state(models[:2])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])


@pytest.mark.parametrize("alias", ["best", "last"])
def test_inference_alias_selects_latest_epoch_checkpoints_in_avg_dir(tmp_path: Path, alias: str):
    cfg = load_config(_write_config(tmp_path, ["001,test,50,0", "002,test,60,1"]))
    root = tmp_path / "checkpoints"
    models = _save_epoch_checkpoints(root, cfg)
    model = SexAgeMLP(cfg)

    selected = baseline_runtime._load_inference_checkpoint(model, _inference_ckpt_args(alias, 2, root), cfg)

    assert selected == [root / "epoch=01.ckpt", root / "epoch=02.ckpt"]
    expected = _average_state(models[1:])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])


@pytest.mark.parametrize(
    ("avg_ckpts", "with_dir", "message"),
    [(1, True, "only with --avg-ckpts > 1"), (2, False, "Use --avg-ckpt-dir")],
)
def test_inference_alias_requires_averaging_directory(tmp_path: Path, avg_ckpts: int, with_dir: bool, message: str):
    cfg = load_config(_write_config(tmp_path, ["001,test,50,0", "002,test,60,1"]))
    root = tmp_path / "checkpoints"
    _save_epoch_checkpoints(root, cfg)

    with pytest.raises(ValueError, match=message):
        baseline_runtime._load_inference_checkpoint(
            SexAgeMLP(cfg), _inference_ckpt_args("best", avg_ckpts, root if with_dir else None), cfg
        )


@pytest.mark.parametrize("avg_ckpts", [0, -1])
def test_inference_rejects_non_positive_avg_ckpts(tmp_path: Path, avg_ckpts: int):
    cfg = load_config(_write_config(tmp_path, ["001,test,50,0", "002,test,60,1"]))
    root = tmp_path / "checkpoints"
    _save_epoch_checkpoints(root, cfg)

    with pytest.raises(ValueError, match="--avg-ckpts must be a positive integer"):
        baseline_runtime._load_inference_checkpoint(
            SexAgeMLP(cfg), _inference_ckpt_args(str(root / "epoch=01.ckpt"), avg_ckpts, None), cfg
        )


def test_inference_averaging_rejects_incompatible_model_contract(tmp_path: Path):
    rows = ["001,test,50,0", "002,test,60,1"]
    index = _write_index(tmp_path / "index.csv", rows)
    sidecars = _write_survival_sidecars(tmp_path / "sidecars", ["001", "002"])
    saved_cfg = load_config(_write_yaml(tmp_path / "saved.yaml", _base_payload(index, sidecars, "survival")))
    current_payload = _base_payload(index, sidecars, "survival")
    # Same parameter shapes, different frozen covariate scaling.
    current_payload["finetune"]["survival"]["covariate_normalization"] = {"age": {"mean": 50.0, "std": 10.0}}
    current_cfg = load_config(_write_yaml(tmp_path / "current.yaml", current_payload))
    root = tmp_path / "checkpoints"
    _save_epoch_checkpoints(root, saved_cfg)

    with pytest.raises(ValueError, match="model contract"):
        baseline_runtime._load_inference_checkpoint(
            SexAgeMLP(current_cfg), _inference_ckpt_args("last", 2, root), current_cfg
        )
