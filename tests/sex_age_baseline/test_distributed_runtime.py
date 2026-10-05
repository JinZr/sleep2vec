from argparse import Namespace
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
import yaml

from sex_age_baseline.config import (
    BaselineConfig,
    DataConfig,
    FinetuneConfig,
    FinetuneLossConfig,
    HeadConfig,
    ModelConfig,
    MultilabelConfig,
    SurvivalConfig,
    TaskConfig,
)
from sex_age_baseline.data import BaselineRecord, SexAgeDataset, make_dataloader
from sex_age_baseline.runtime import FINETUNE_SEED, BaselineModule, _batch_loss, _evaluate_records, _evaluation_record
from sleep2vec.losses.cox import CoxPHLossVectorized


def _config(root):
    return BaselineConfig(
        ModelConfig("sex_age_mlp", HeadConfig("classification", 4, 0.0, "elu", {"num_layers": 3})),
        DataConfig("npz", "unused.csv", None, None, None, None, None),
        FinetuneConfig(
            TaskConfig("survival", 1, False, "val_c_index", "max"),
            SurvivalConfig("eid", str(root / "columns.txt"), "unused", "unused", "unused", ["age"], 2, {}),
            loss=FinetuneLossConfig(),
        ),
    )


def _record(key):
    return BaselineRecord(
        key=str(key),
        metadata={"age": 40 + key},
        event_time=np.array([key + 1.0]),
        is_event=np.array([1.0]),
        has_label=np.array([1.0]),
    )


def _dataset(count=5):
    # The loader keeps one record per task key, so every record is a distinct subject.
    return SexAgeDataset([_record(key) for key in range(count)], task_type="survival", label_names=["disease"])


class _Trace(pl.Callback):
    def __init__(self):
        self.batches = []
        self.lrs = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batches.append(list(batch["key"]))
        self.lrs.append(trainer.optimizers[0].param_groups[0]["lr"])


@pytest.mark.parametrize("batch_size,expected_sizes", [(2, [2, 2, 1]), (8, [5])])
def test_single_device_training_keeps_incomplete_batch(tmp_path, batch_size, expected_sizes):
    module = BaselineModule(
        _config(tmp_path),
        Namespace(
            lr=0.001,
            weight_decay=0.01,
            warmup_steps=0,
            lr_decay_shape="linear",
            lr_decay_floor=0.1,
            batch_size=batch_size,
            num_workers=0,
        ),
    )
    module.train_set = _dataset()
    trace = _Trace()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        callbacks=[trace],
        default_root_dir=str(tmp_path),
        enable_progress_bar=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    trainer.fit(module)
    assert trainer.global_step == len(expected_sizes) * 2
    assert [len(batch) for batch in trace.batches] == expected_sizes * 2
    expected = {record.key for record in module.train_set.records}
    for epoch in range(2):
        batches = trace.batches[epoch * len(expected_sizes) : (epoch + 1) * len(expected_sizes)]
        identities = [identity for batch in batches for identity in batch]
        assert len(identities) == len(expected)
        assert set(identities) == expected


def _worker(root):
    root = Path(root)
    torch.set_num_threads(1)
    pl.seed_everything(42)
    cfg = _config(root)
    args = Namespace(
        lr=0.001,
        weight_decay=0.01,
        warmup_steps=1,
        lr_decay_shape="linear",
        lr_decay_floor=0.1,
        batch_size=2,
        num_workers=0,
        inference_prediction_csv_path=str(root / "predictions.csv"),
    )
    module = BaselineModule(cfg, args)
    dataset = _dataset()
    module.train_set = _dataset(7)
    trace = _Trace()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp",
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        callbacks=[trace],
        default_root_dir=str(root),
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    # Five subjects over two ranks: the distributed sampler pads one repeated row that evaluation must drop.
    loader = make_dataloader(dataset, batch_size=2, num_workers=0, shuffle=False)
    trainer.fit(module, val_dataloaders=loader)
    trainer.test(module, dataloaders=loader, verbose=False)
    result = module.evaluation_result
    payload = {
        "steps": trainer.global_step,
        "batches": trace.batches,
        "lrs": trace.lrs,
        "predictions": result.prediction_rows,
        "metrics": result.metrics,
    }
    (root / f"rank-{trainer.global_rank}.json").write_text(json.dumps(payload))


def test_two_cpu_rank_training_and_padding_aggregation(tmp_path):
    (tmp_path / "columns.txt").write_text("disease\n")
    entry = tmp_path / "smoke.py"
    entry.write_text(
        f"import sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from test_distributed_runtime import _worker\n"
        f"if __name__ == '__main__':\n    _worker({str(tmp_path)!r})\n"
    )
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, str(entry)], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    ranks = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert ranks[0]["metrics"] == ranks[1]["metrics"]
    for rank in ranks:
        assert rank["steps"] == 2  # floor(floor(7 / 2 ranks) / batch2) * 2 epochs
        assert [len(batch) for batch in rank["batches"]] == [2, 2]
        assert rank["lrs"] == pytest.approx([0, 0.001])
        rows = rank["predictions"]
        assert [row["survival_key"] for row in rows] == [str(key) for key in range(5)]
        assert all(row["path"] == row["survival_key"] and row["n_windows"] == 1 for row in rows)
    for epoch in range(2):
        actual = [key for rank in ranks for batch in rank["batches"][epoch : epoch + 1] for key in batch]
        assert len(actual) == len(set(actual)) == 4
        identities = [str(key) for key in range(7)]
        generator = torch.Generator().manual_seed(FINETUNE_SEED + epoch)
        expected = torch.randperm(7, generator=generator).tolist()[:4]
        assert set(actual) == {identities[index] for index in expected}


def test_cox_loss_uses_rank_local_batch(tmp_path):
    cfg = _config(tmp_path)
    logits = torch.tensor([[0.1], [0.7], [-0.2], [0.3]])
    batch = {
        "has_label": torch.ones(4, 1),
        "event_time": torch.arange(1, 5).reshape(-1, 1),
        "is_event": torch.ones(4, 1),
    }
    local = {name: value[:2] for name, value in batch.items()}
    observed = _batch_loss(logits[:2], local, cfg)
    expected = CoxPHLossVectorized()(logits[:2], local["has_label"], local["event_time"], local["is_event"])
    assert observed == expected
    assert not torch.isclose(observed, _batch_loss(logits, batch, cfg))


def _masked_gradient_worker(rank, root):
    torch.distributed.init_process_group("gloo", init_method=f"file://{root}/process-group", rank=rank, world_size=2)
    try:
        cfg = replace(
            _config(Path(root)),
            finetune=FinetuneConfig(
                TaskConfig("multilabel_classification", 2, False, "val_macro_auroc", "max"),
                multilabel=MultilabelConfig("eid", "unused", "unused", "unused", ["age"], 2, {}),
                loss=FinetuneLossConfig(pos_weight=[1.5, 2.0]),
            ),
        )
        features = torch.tensor([[1.0, 0.5], [0.2, 1.0], [1.5, -0.5], [-0.2, 0.3]])
        labels = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        for mask in (
            [[1, 0], [0, 0], [1, 1], [0, 0]],
            [[0, 0], [0, 0], [1, 1], [1, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
        ):
            torch.manual_seed(42)
            reference = torch.nn.Linear(2, 2)
            module = BaselineModule(cfg, Namespace())
            model = torch.nn.Linear(2, 2)
            model.load_state_dict(reference.state_dict())
            module.model = torch.nn.parallel.DistributedDataParallel(model)
            module._trainer = Namespace(world_size=2)
            module.log = lambda *args, **kwargs: None
            # The module forwards batch["metadata"] to its model; a plain Linear stands in for the covariate MLP.
            batch = {"metadata": features, "disease_label": labels, "has_label": torch.tensor(mask)}
            expected = _batch_loss(reference(features), batch, cfg)
            expected.backward()
            local = {name: value[rank * 2 : (rank + 1) * 2] for name, value in batch.items()}
            actual = module.training_step(local, 0)
            actual.backward()
            for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter.grad, reference_parameter.grad)
            mean_loss = actual.detach().clone()
            torch.distributed.all_reduce(mean_loss)
            torch.testing.assert_close(mean_loss / 2, expected.detach())
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_masked_multilabel_gradients_match_global_reference(tmp_path):
    entry = tmp_path / "masked_gradient.py"
    entry.write_text(
        f"import sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import torch\nfrom test_distributed_runtime import _masked_gradient_worker\n"
        "if __name__ == '__main__':\n"
        f"    torch.multiprocessing.spawn(_masked_gradient_worker, args=({str(tmp_path)!r},), nprocs=2)\n"
    )
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, str(entry)], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("task", ["survival", "multilabel_classification"])
def test_evaluation_drops_distributed_padding_and_keeps_dataset_order(tmp_path, task):
    (tmp_path / "columns.txt").write_text("disease\n")
    cfg = _config(tmp_path)
    dataset = _dataset(4)
    if task != "survival":
        cfg = replace(
            cfg,
            finetune=FinetuneConfig(
                TaskConfig("multilabel_classification", 1, False, "val_macro_auroc", "max"),
                multilabel=MultilabelConfig("eid", str(tmp_path / "columns.txt"), "unused", "unused", ["age"], 2, {}),
                loss=FinetuneLossConfig(),
            ),
        )
        dataset = SexAgeDataset(
            [
                replace(record, event_time=None, is_event=None, disease_label=np.array([key % 2.0]))
                for key, record in enumerate(dataset.records)
            ],
            task_type=task,
            label_names=["disease"],
        )
    batch = next(iter(make_dataloader(dataset, batch_size=4, num_workers=0, shuffle=False)))
    logits = torch.tensor([[0.0], [4.0], [1.0], [2.0]])
    record = _evaluation_record(batch, logits)
    # Ranks return interleaved slices, and a padded sampler repeats rows; both collapse onto the dataset order.
    first = {name: value[2:] for name, value in record.items()}
    second = {name: value[:3] for name, value in record.items()}

    result = _evaluate_records([first, second], cfg, "test", True)

    key = "survival_key" if task == "survival" else "multilabel_key"
    assert [row[key] for row in result.prediction_rows] == ["0", "1", "2", "3"]
    assert all(row["path"] == row[key] and row["n_windows"] == 1 for row in result.prediction_rows)
    value = "log_risk" if task == "survival" else "logit"
    assert [row[value] for row in result.prediction_rows] == [[0.0], [4.0], [1.0], [2.0]]


@pytest.mark.parametrize("task", ["survival", "multilabel_classification"])
def test_two_rank_training_and_independent_inference_cli(tmp_path, task):
    from test_data_model_runtime import _write_config

    rows = []
    for offset, split in enumerate(["train", "val", "test"]):
        rows.extend(f"{offset * 10 + i},{split},{40 + i},{i % 2}" for i in range(4))
    config = _write_config(tmp_path, rows, task)
    payload = yaml.safe_load(config.read_text())
    payload["model"]["head"]["kwargs"]["num_layers"] = 3
    config.write_text(yaml.safe_dump(payload))
    # A signal-style window index: three subjects per split contribute a second window that must collapse.
    index = pd.read_csv(tmp_path / "index.csv")
    index["path"] = index.eid.map(lambda eid: f"absent-signal-{eid}.npz")
    index["token_start"] = 0
    repeated = index.groupby("split", sort=False).head(3).copy()
    repeated["token_start"] = 10
    pd.concat([index, repeated], ignore_index=True).to_csv(tmp_path / "index.csv", index=False)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    common = [
        "--config",
        str(config),
        "--label-name",
        "smoke",
        "--device",
        "cpu",
        "--devices",
        "0",
        "1",
        "--precision",
        "32-true",
        "--batch-size",
        "2",
        "--num-workers",
        "0",
        "--wandb-mode",
        "disabled",
    ]
    training = subprocess.run(
        [
            sys.executable,
            "-m",
            "sex_age_baseline.finetune",
            *common,
            "--epochs",
            "2",
            "--lr",
            "0.001",
            "--warmup-steps",
            "1",
            "--lr-decay-shape",
            "linear",
            "--version-name",
            "ddp-cli",
            "--results-csv-path",
            str(tmp_path / "results.csv"),
            "--export-predictions",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (tmp_path / "training.log").write_text(training.stdout + training.stderr)
    assert training.returncode == 0, training.stdout + training.stderr
    assert "Starting with 2 processes" in training.stderr
    run_dir = tmp_path / "log-finetune" / "ddp-cli"
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert len(pd.read_csv(tmp_path / "results.csv")) == 1
    assert len(pd.read_csv(run_dir / "predictions.csv")) == 4
    checkpoint = run_dir / "checkpoints" / "best.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["optimizer_states"] and state["lr_schedulers"]
    assert state["model_contract"]["covariates"] == ["age", "sex"]
    last = torch.load(run_dir / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
    assert last["global_step"] == 2  # 7 windows -> 4 subjects -> 2 per rank -> one full batch each.
    inference_root = tmp_path / "inference"
    inference = subprocess.run(
        [
            sys.executable,
            "-m",
            "sex_age_baseline.infer",
            *common,
            "--ckpt-path",
            str(checkpoint),
            "--results-root",
            str(inference_root),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (tmp_path / "inference.log").write_text(inference.stdout + inference.stderr)
    assert inference.returncode == 0, inference.stdout + inference.stderr
    assert "Starting with 2 processes" in inference.stderr
    manifests = list(inference_root.rglob("run_manifest.json"))
    assert len(manifests) == 1
    inferred = json.loads(manifests[0].read_text())
    assert inferred["prediction_row_count"] == 4
    metric = "test_c_index" if task == "survival" else "test_macro_auroc"
    assert inferred["metrics"][metric] == pytest.approx(manifest["metrics"][metric])
