from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import pickle

import pytest
import torch
import yaml

from data.default_dataset import SampleIndex
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
            "backend": "npz",
            "finetune_data_index": str(index),
            "finetune_preset_path": None,
            "kaldi_data_root": None,
            "kaldi_manifest": None,
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


def _parse_metadata_rows(rows: list[str]) -> list[dict[str, str]]:
    parsed = []
    for row in rows:
        eid, split, age, sex = row.split(",")
        parsed.append({"eid": eid, "split": split, "age": age, "sex": sex})
    return parsed


def _write_preset(path: Path, rows: list[str]) -> Path:
    samples = [
        SampleIndex(id=row["eid"], path="ignored.npz", start=0, end=1, metadata=row)
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


def test_metadata_backends_produce_identical_subject_records(tmp_path: Path):
    rows = ["001,train,50,0", "001,train,50,0", "002,val,60,1", "003,test,55,0"]
    index = _write_index(tmp_path / "index.csv", rows)
    preset = _write_preset(tmp_path / "preset.pkl", rows)
    kaldi_root, kaldi_manifest = _write_kaldi_root(tmp_path / "kaldi", rows)
    configs = [
        _write_config_for_data(
            tmp_path / "npz-index",
            rows,
            {
                "backend": "npz",
                "finetune_data_index": str(index),
                "finetune_preset_path": None,
                "kaldi_data_root": None,
                "kaldi_manifest": None,
            },
        ),
        _write_config_for_data(
            tmp_path / "npz-preset",
            rows,
            {
                "backend": "npz",
                "finetune_data_index": None,
                "finetune_preset_path": str(preset),
                "kaldi_data_root": None,
                "kaldi_manifest": None,
            },
        ),
        _write_config_for_data(
            tmp_path / "kaldi-config",
            rows,
            {
                "backend": "kaldi",
                "finetune_data_index": None,
                "finetune_preset_path": None,
                "kaldi_data_root": str(kaldi_root),
                "kaldi_manifest": str(kaldi_manifest),
            },
        ),
    ]

    records = []
    for config in configs:
        cfg = load_config(config, validate_sidecars=True)
        records.append([(record.key, record.age, record.sex) for record in load_split_dataset(cfg, "train")])

    assert records == [[("001", 50.0, 0)], [("001", 50.0, 0)], [("001", 50.0, 0)]]


def test_kaldi_manifest_split_key_fills_missing_split_column(tmp_path: Path):
    rows = ["001,train,50,0", "002,val,60,1"]
    kaldi_root = tmp_path / "kaldi"
    kaldi_root.mkdir()
    (kaldi_root / "train.csv").write_text("eid,age,sex\n001,50,0\n")
    (kaldi_root / "val.csv").write_text("eid,age,sex\n002,60,1\n")
    kaldi_manifest = kaldi_root / "manifest.json"
    kaldi_manifest.write_text(
        json.dumps({"splits": {"train": {"manifest": "train.csv"}, "val": {"manifest": "val.csv"}}})
    )
    config = _write_config_for_data(
        tmp_path,
        rows,
        {
            "backend": "kaldi",
            "finetune_data_index": None,
            "finetune_preset_path": None,
            "kaldi_data_root": str(kaldi_root),
            "kaldi_manifest": str(kaldi_manifest),
        },
    )
    cfg = load_config(config, validate_sidecars=True)

    train = load_split_dataset(cfg, "train")
    val = load_split_dataset(cfg, "val")

    assert [(record.key, record.age, record.sex) for record in train] == [("001", 50.0, 0)]
    assert [(record.key, record.age, record.sex) for record in val] == [("002", 60.0, 1)]


def test_conflicting_duplicate_metadata_fails(tmp_path: Path):
    config = _write_config(tmp_path, ["001,train,50,female", "001,train,51,female"])
    cfg = load_config(config, validate_sidecars=True)

    with pytest.raises(ValueError, match="conflicting age"):
        load_split_dataset(cfg, "train")


@pytest.mark.parametrize("backend", ["npz_preset", "kaldi"])
def test_conflicting_duplicate_metadata_fails_for_non_index_backends(tmp_path: Path, backend: str):
    rows = ["001,train,50,0", "001,train,51,0"]
    if backend == "npz_preset":
        preset = _write_preset(tmp_path / "preset.pkl", rows)
        data = {
            "backend": "npz",
            "finetune_data_index": None,
            "finetune_preset_path": str(preset),
            "kaldi_data_root": None,
            "kaldi_manifest": None,
        }
    else:
        kaldi_root, kaldi_manifest = _write_kaldi_root(tmp_path / "kaldi", rows)
        data = {
            "backend": "kaldi",
            "finetune_data_index": None,
            "finetune_preset_path": None,
            "kaldi_data_root": str(kaldi_root),
            "kaldi_manifest": str(kaldi_manifest),
        }
    config = _write_config_for_data(tmp_path, rows, data)
    cfg = load_config(config, validate_sidecars=True)

    with pytest.raises(ValueError, match="conflicting age"):
        load_split_dataset(cfg, "train")


@pytest.mark.parametrize("backend", ["npz_preset", "kaldi"])
def test_missing_metadata_columns_fail_for_non_index_backends(tmp_path: Path, backend: str):
    rows = ["001,train,50,0"]
    if backend == "npz_preset":
        sample = SampleIndex(id="001", path="ignored.npz", start=0, end=1, metadata={"eid": "001", "split": "train"})
        preset = tmp_path / "preset.pkl"
        with preset.open("wb") as file_obj:
            pickle.dump([sample], file_obj)
        data = {
            "backend": "npz",
            "finetune_data_index": None,
            "finetune_preset_path": str(preset),
            "kaldi_data_root": None,
            "kaldi_manifest": None,
        }
    else:
        kaldi_root = tmp_path / "kaldi"
        kaldi_root.mkdir()
        (kaldi_root / "train.csv").write_text("eid,split\n001,train\n")
        kaldi_manifest = kaldi_root / "manifest.json"
        kaldi_manifest.write_text(json.dumps({"splits": {"train": {"manifest": "train.csv"}}}))
        data = {
            "backend": "kaldi",
            "finetune_data_index": None,
            "finetune_preset_path": None,
            "kaldi_data_root": str(kaldi_root),
            "kaldi_manifest": str(kaldi_manifest),
        }
    config = _write_config_for_data(tmp_path, rows, data)
    cfg = load_config(config, validate_sidecars=True)

    with pytest.raises(ValueError, match="missing|required"):
        load_split_dataset(cfg, "train")


def test_invalid_sex_fails(tmp_path: Path):
    config = _write_config(tmp_path, ["001,train,50,unknown"])
    cfg = load_config(config, validate_sidecars=True)

    with pytest.raises(ValueError, match="sex value"):
        load_split_dataset(cfg, "train")


@pytest.mark.parametrize("backend", ["npz_index", "npz_preset", "kaldi"])
def test_unused_split_metadata_values_do_not_block_selected_split(tmp_path: Path, backend: str):
    rows = ["001,train,50,0", "002,val,60,1", "003,test,,unknown"]
    data = _backend_data_config(tmp_path, rows, backend)
    config = _write_config_for_data(tmp_path, rows, data)
    cfg = load_config(config, validate_sidecars=True)

    assert [(record.key, record.age, record.sex) for record in load_split_dataset(cfg, "train")] == [("001", 50.0, 0)]
    assert [(record.key, record.age, record.sex) for record in load_split_dataset(cfg, "val")] == [("002", 60.0, 1)]

    with pytest.raises(ValueError, match="age"):
        load_split_dataset(cfg, "test")


def test_unused_split_duplicate_metadata_does_not_block_selected_split(tmp_path: Path):
    config = _write_config(tmp_path, ["001,train,50,0", "002,val,60,1", "001,test,55,1"])
    cfg = load_config(config, validate_sidecars=True)

    assert [
        (record.key, record.age, record.sex)
        for record in load_split_dataset(cfg, "train", loaded_splits=["train", "val"])
    ] == [("001", 50.0, 0)]
    assert [
        (record.key, record.age, record.sex)
        for record in load_split_dataset(cfg, "val", loaded_splits=["train", "val"])
    ] == [("002", 60.0, 1)]

    with pytest.raises(ValueError, match="multiple loaded splits"):
        load_split_dataset(cfg, "train", loaded_splits=["train", "val", "test"])


@pytest.mark.parametrize("backend", ["npz_index", "npz_preset", "kaldi"])
def test_loaded_split_key_reuse_fails_before_metadata_parsing(tmp_path: Path, backend: str):
    rows = ["001,train,50,0", "001,val,bad,unknown", "002,test,,unknown"]
    data = _backend_data_config(tmp_path, rows, backend)
    config = _write_config_for_data(tmp_path, rows, data)
    cfg = load_config(config, validate_sidecars=True)

    with pytest.raises(ValueError, match="multiple loaded splits"):
        load_split_dataset(cfg, "train", loaded_splits=["train", "val"])


def test_train_rejects_key_reused_across_train_val(tmp_path: Path, monkeypatch):
    config = _write_config(tmp_path, ["001,train,50,0", "001,val,50,0"])
    cfg = load_config(config, validate_sidecars=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="multiple loaded splits"):
        baseline_runtime.train_and_save(_runtime_args(config, tmp_path, version_name="split-leak"), cfg)


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


@pytest.mark.parametrize("test_after_fit", [True, False])
def test_zero_epoch_train_requires_checkpoint(tmp_path: Path, monkeypatch, test_after_fit: bool):
    config = _write_config(
        tmp_path,
        ["001,train,50,0", "002,val,60,1"],
        task_type="multilabel_classification",
    )
    cfg = load_config(config, validate_sidecars=True)
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
    cfg = load_config(config, validate_sidecars=True)
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
    saved_cfg = load_config(saved_config, validate_sidecars=True)
    current_cfg = load_config(current_config, validate_sidecars=True)
    ckpt = tmp_path / "model.ckpt"
    baseline_runtime.save_checkpoint(ckpt, SexAgeMLP(saved_cfg), saved_cfg, epoch=0, global_step=0, metrics={})

    with pytest.raises(ValueError, match="label contract"):
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


def test_infer_run_inference_callable_validates_and_delegates(tmp_path: Path, monkeypatch):
    import sex_age_baseline.infer as infer_mod

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("placeholder")
    config = tmp_path / "config.yaml"
    config.write_text("model: {}\n")
    cfg = object()
    calls = []

    def fake_load_config(path, *, validate_sidecars):
        calls.append(("load_config", path, validate_sidecars))
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
        ("load_config", config, True),
        ("run", str(ckpt), "cpu", cfg),
    ]
