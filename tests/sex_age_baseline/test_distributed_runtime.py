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
    AgeConfig,
    BaselineConfig,
    DataConfig,
    FinetuneConfig,
    FinetuneLossConfig,
    HeadConfig,
    ModelConfig,
    MultilabelConfig,
    OutputsConfig,
    SurvivalConfig,
    TaskConfig,
)
from sex_age_baseline.data import BaselineRecord, SexAgeDataset, make_dataloader
from sex_age_baseline.runtime import BaselineModule, _batch_loss, _evaluate_records, _evaluation_record
from sleep2vec.losses.cox import CoxPHLossVectorized


def _config(root):
    return BaselineConfig(
        ModelConfig(
            "sex_age_mlp",
            ["age"],
            AgeConfig("divide", 100, 2, "default"),
            None,
            None,
            HeadConfig("classification", 4, 0.0, "elu", {"num_layers": 3}),
        ),
        DataConfig("npz", "unused.csv", None, None, None, "split", "eid", False),
        FinetuneConfig(
            TaskConfig("survival", 1, False, "val_c_index", "max"),
            SurvivalConfig("eid", str(root / "columns.txt"), "unused", "unused", "unused"),
            loss=FinetuneLossConfig(),
        ),
        OutputsConfig(True, True),
    )


def _dataset():
    return SexAgeDataset(
        [
            BaselineRecord(
                key=str(key),
                features={"age": float(40 + key)},
                path=f"record-{key}",
                token_start=start,
                event_time=np.array([key + 1.0]),
                is_event=np.array([1.0]),
                has_label=np.array([1.0]),
            )
            for key, start in [(0, 0), (0, 10), (1, 0), (2, 0), (3, 0)]
        ],
        task_type="survival",
        label_names=["disease"],
    )


class _Trace(pl.Callback):
    def __init__(self):
        self.batches = []
        self.lrs = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batches.append(list(zip(batch["key"], batch["token_start"].tolist())))
        self.lrs.append(trainer.optimizers[0].param_groups[0]["lr"])


def _worker(root):
    root = Path(root)
    torch.set_num_threads(1)
    pl.seed_everything(42)
    cfg = _config(root)
    args = Namespace(lr=0.001, weight_decay=0.01, warmup_steps=1, lr_decay_shape="linear", lr_decay_floor=0.1)
    module = BaselineModule(cfg, args)
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
    loader = make_dataloader(_dataset(), batch_size=2, num_workers=0, shuffle=False)
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
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
        assert rank["steps"] == 4  # ceil(5 / 2 ranks / batch2) * 2 epochs
        assert [len(batch) for batch in rank["batches"]] == [2, 1, 2, 1]
        assert rank["lrs"] == pytest.approx([0, 0.001, 0.0007, 0.0004])
        rows = rank["predictions"]
        assert len(rows) == 4
        assert {row["survival_key"]: row["n_windows"] for row in rows} == {"0": 2, "1": 1, "2": 1, "3": 1}
    for epoch in range(2):
        actual = [
            tuple(pair) for rank in ranks for batch in rank["batches"][epoch * 2 : epoch * 2 + 2] for pair in batch
        ]
        assert len(actual) == 6  # one distributed sampler padding slot
        assert set(actual) == {("0", 0), ("0", 10), ("1", 0), ("2", 0), ("3", 0)}


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


def test_subject_logrisk_mean_after_window_dedup(tmp_path):
    (tmp_path / "columns.txt").write_text("disease\n")
    cfg = _config(tmp_path)
    batch = next(iter(make_dataloader(_dataset(), batch_size=5, num_workers=0, shuffle=False)))
    record = _evaluation_record(batch, torch.tensor([[0.0], [4.0], [1.0], [2.0], [3.0]]))
    result = _evaluate_records([record, record], cfg, "test", True)
    by_key = {row["survival_key"]: row for row in result.prediction_rows}
    assert by_key["0"]["log_risk"] == [2.0]
    assert by_key["0"]["n_windows"] == 2
    assert len(by_key) == 4


def test_multilabel_averages_logits_before_sigmoid(tmp_path):
    (tmp_path / "columns.txt").write_text("disease\n")
    cfg = _config(tmp_path)
    cfg = replace(
        cfg,
        finetune=FinetuneConfig(
            TaskConfig("multilabel_classification", 1, False, "val_macro_auroc", "max"),
            multilabel=MultilabelConfig("eid", str(tmp_path / "columns.txt"), "unused", "unused"),
            loss=FinetuneLossConfig(),
        ),
    )
    record = {
        "key": ["1", "1", "2"],
        "path": ["a", "a", "b"],
        "token_start": [0, 10, 0],
        "logits": torch.tensor([[0.0], [4.0], [-1.0]]),
        "disease_label": torch.tensor([[1.0], [1.0], [0.0]]),
        "has_label": torch.ones(3, 1),
    }
    result = _evaluate_records([record, record], cfg, "test", True)
    rows = {row["multilabel_key"]: row for row in result.prediction_rows}
    assert rows["1"]["logit"] == [2.0]
    assert rows["1"]["probability"] == pytest.approx([torch.sigmoid(torch.tensor(2.0)).item()])
    assert rows["1"]["n_windows"] == 2
    assert len(rows) == 2


@pytest.mark.parametrize("task", ["survival", "multilabel_classification"])
def test_two_rank_training_and_independent_inference_cli(tmp_path, task):
    from test_data_model_runtime import _write_config

    rows = []
    for offset, split in enumerate(["train", "val", "test"]):
        rows.extend(f"{offset * 10 + i},{split},{40 + i},{i % 2}" for i in range(4))
    config = _write_config(tmp_path, rows, task)
    payload = yaml.safe_load(config.read_text())
    payload["model"]["head"]["kwargs"]["num_layers"] = 3
    payload["data"]["deduplicate_by_key"] = False
    config.write_text(yaml.safe_dump(payload))
    index = pd.read_csv(tmp_path / "index.csv")
    index["path"] = index.eid.map(lambda eid: f"absent-signal-{eid}.npz")
    index["token_start"] = 0
    repeated = index.groupby("split", sort=False).head(1).copy()
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
    assert state["model_contract"]["features"] == ["age", "sex"]
    last = torch.load(run_dir / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
    assert last["global_step"] == 2  # 3 sampler slots per rank, training drops the batch-size-one tail.
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
