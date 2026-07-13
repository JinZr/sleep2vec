from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest


@pytest.fixture(scope="module")
def training_runtime_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", "import torch; import pytorch_lightning; import wandb"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("Training runtime dependencies are unavailable in this test environment.")


@pytest.mark.parametrize("namespace", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_training_run_manifest_writer_serializes_checkpoint_and_score(tmp_path: Path, monkeypatch, namespace: str):
    torch_module = types.ModuleType("torch")
    distributed_module = types.ModuleType("torch.distributed")
    distributed_module.is_available = lambda: False
    distributed_module.is_initialized = lambda: False
    distributed_module.get_rank = lambda: 0
    distributed_module.get_world_size = lambda: 1
    torch_module.distributed = distributed_module
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed_module)
    sys.modules.pop(f"{namespace}.results", None)
    results = importlib.import_module(f"{namespace}.results")

    manifest_path = tmp_path / "run_manifest.json"
    args = argparse.Namespace(
        version="unit",
        config=Path("config.yaml"),
        label_name="ahi",
        test_after_fit=False,
    )

    results.save_training_run_manifest(
        args,
        manifest_path=manifest_path,
        status="skipped_test",
        monitor="val_ahi_pearson",
        monitor_mode="max",
        best_model_path=tmp_path / "best.ckpt",
        best_model_score=0.5,
        last_checkpoint_path=tmp_path / "last.ckpt",
        survival_per_disease_metrics_csv_path=tmp_path / "survival_per_disease_metrics.csv",
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["kind"] == "sleep2vec_finetune_run"
    assert manifest["best_model_path"].endswith("best.ckpt")
    assert manifest["best_model_score"] == 0.5
    assert manifest["test_after_fit"] is False
    assert manifest["survival_per_disease_metrics_csv_path"].endswith("survival_per_disease_metrics.csv")


@pytest.mark.parametrize("namespace", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_failed_manifest_write_does_not_mask_primary_training_error(
    monkeypatch, training_runtime_dependencies, namespace: str
):
    importlib.import_module("torch")
    importlib.import_module("pytorch_lightning")
    importlib.import_module("wandb")
    sys.modules.pop(f"{namespace}.finetune", None)
    finetune = importlib.import_module(f"{namespace}.finetune")
    args = argparse.Namespace(
        version="unit",
        monitor="val_loss",
        monitor_mod="min",
        patience=1,
        print_diagnostics=False,
    )
    config_bundle = types.SimpleNamespace(model={}, finetune={}, averaging={})
    logger_kwargs = {}

    class DummyLogger:
        experiment = types.SimpleNamespace(log=lambda *args, **kwargs: None)

        def log_hyperparams(self, *args, **kwargs):
            return None

    class DummyModel:
        moe_finetune_status = {}

        def moe_finetune_hparams(self):
            return {}

        def moe_finetune_param_group_rows(self):
            return []

    monkeypatch.setattr(finetune, "persist_run_config_and_args", lambda *args, **kwargs: None)
    monkeypatch.setattr(finetune, "prepare_dataloader", lambda args: ([], [], []))
    monkeypatch.setattr(finetune, "Sleep2vecFinetuning", lambda *args, **kwargs: DummyModel())

    def build_logger(**kwargs):
        logger_kwargs.update(kwargs)
        return DummyLogger()

    monkeypatch.setattr(finetune, "WandbLogger", build_logger)
    if hasattr(finetune, "is_rank_zero_process"):
        monkeypatch.setattr(finetune, "is_rank_zero_process", lambda: False)

    def raise_primary_error(*args, **kwargs):
        raise RuntimeError("primary training failure")

    def raise_manifest_error(*args, **kwargs):
        raise OSError("manifest write failure")

    monkeypatch.setattr(finetune, "EarlyStopping", raise_primary_error)
    monkeypatch.setattr(finetune, "save_training_run_manifest", raise_manifest_error)

    with pytest.raises(RuntimeError, match="primary training failure"):
        finetune.supervised(args, config_bundle)

    assert logger_kwargs["name"] == args.version
