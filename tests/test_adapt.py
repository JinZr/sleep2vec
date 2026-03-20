from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from typing import Callable

import pytest


def _load_resolve_adapt_run_artifacts() -> Callable:
    module_name = "_sleep2vec_adapt_test"
    stubbed_modules = {
        "pytorch_lightning": types.ModuleType("pytorch_lightning"),
        "pytorch_lightning.callbacks": types.ModuleType("pytorch_lightning.callbacks"),
        "pytorch_lightning.callbacks.early_stopping": types.ModuleType("pytorch_lightning.callbacks.early_stopping"),
        "pytorch_lightning.loggers": types.ModuleType("pytorch_lightning.loggers"),
        "pytorch_lightning.strategies": types.ModuleType("pytorch_lightning.strategies"),
        "wandb": types.ModuleType("wandb"),
        "data.samplers": types.ModuleType("data.samplers"),
        "sleep2vec.callbacks.pair_acc_logger": types.ModuleType("sleep2vec.callbacks.pair_acc_logger"),
        "sleep2vec.common": types.ModuleType("sleep2vec.common"),
        "sleep2vec.config": types.ModuleType("sleep2vec.config"),
        "sleep2vec.sleep2vec_adaptation": types.ModuleType("sleep2vec.sleep2vec_adaptation"),
        "sleep2vec.utils": types.ModuleType("sleep2vec.utils"),
    }

    stubbed_modules["pytorch_lightning"].Trainer = object
    stubbed_modules["pytorch_lightning.callbacks"].LearningRateMonitor = object
    stubbed_modules["pytorch_lightning.callbacks"].ModelCheckpoint = object
    stubbed_modules["pytorch_lightning.callbacks.early_stopping"].EarlyStopping = object
    stubbed_modules["pytorch_lightning.loggers"].WandbLogger = object
    stubbed_modules["pytorch_lightning.strategies"].DDPStrategy = object
    stubbed_modules["pytorch_lightning.strategies"].DeepSpeedStrategy = object
    stubbed_modules["data.samplers"].handles_distributed_sharding = lambda _sampler: False
    stubbed_modules["sleep2vec.callbacks.pair_acc_logger"].PairAccLoggerCallback = object
    stubbed_modules["sleep2vec.common"].apply_model_config_args = lambda *args, **kwargs: None
    stubbed_modules["sleep2vec.common"].persist_run_config_and_args = lambda *args, **kwargs: None
    stubbed_modules["sleep2vec.config"].load_pretrain_config = lambda *_args, **_kwargs: None
    stubbed_modules["sleep2vec.sleep2vec_adaptation"].AdaptPairScheduleCallback = object
    stubbed_modules["sleep2vec.sleep2vec_adaptation"].Sleep2vecAdaptation = object
    stubbed_modules["sleep2vec.sleep2vec_adaptation"].initial_pair_probs_for_phase = lambda *_args, **_kwargs: None
    stubbed_modules["sleep2vec.utils"].get_pretrain_dataloader = lambda *_args, **_kwargs: (None, None)

    originals = {name: sys.modules.get(name) for name in stubbed_modules}
    originals[module_name] = sys.modules.get(module_name)

    try:
        for name, module in stubbed_modules.items():
            sys.modules[name] = module

        spec = importlib.util.spec_from_file_location(
            module_name,
            Path(__file__).resolve().parents[1] / "sleep2vec/adapt.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load sleep2vec.adapt for testing.")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module._resolve_adapt_run_artifacts
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@pytest.fixture(scope="module")
def resolve_adapt_run_artifacts() -> Callable:
    return _load_resolve_adapt_run_artifacts()


def test_resolve_adapt_run_artifacts_builds_new_run_path_without_ckpt(resolve_adapt_run_artifacts: Callable):
    save_path, run_name, wandb_id = resolve_adapt_run_artifacts(
        ckpt_path=None,
        version_name="wearable-v1",
        backbone_arch="roformer",
        phase="stage1",
        exp_info="ppg actigraphy",
    )

    assert save_path == Path("log-adapt/wearable-v1-roformer-adapt-stage1-ppg_actigraphy/checkpoints")
    assert run_name == "wearable-v1-roformer-adapt-stage1-ppg_actigraphy"
    assert wandb_id is None


def test_resolve_adapt_run_artifacts_rejects_missing_ckpt(
    tmp_path: Path,
    resolve_adapt_run_artifacts: Callable,
):
    missing = tmp_path / "missing.ckpt"

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        resolve_adapt_run_artifacts(
            ckpt_path=missing,
            version_name="wearable-v1",
            backbone_arch="roformer",
            phase="stage2",
        )


def test_resolve_adapt_run_artifacts_rejects_directory_ckpt(
    tmp_path: Path,
    resolve_adapt_run_artifacts: Callable,
):
    ckpt_dir = tmp_path / "run_a" / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="Checkpoint path must be a file"):
        resolve_adapt_run_artifacts(
            ckpt_path=ckpt_dir,
            version_name="wearable-v1",
            backbone_arch="roformer",
            phase="stage2",
        )


def test_resolve_adapt_run_artifacts_uses_existing_ckpt_parent(
    tmp_path: Path,
    resolve_adapt_run_artifacts: Callable,
):
    ckpt_path = tmp_path / "run_a" / "checkpoints" / "epoch=1.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("checkpoint")

    save_path, run_name, wandb_id = resolve_adapt_run_artifacts(
        ckpt_path=ckpt_path,
        version_name="ignored",
        backbone_arch="ignored",
        phase="stage2",
    )

    assert save_path == ckpt_path.parent
    assert run_name == "run_a"
    assert wandb_id == "run_a"
