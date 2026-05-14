from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import types
from types import ModuleType

import pytest
import yaml

from sleep2vec.common import persist_run_config_and_args


def _load_adapt_module() -> ModuleType:
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
    stubbed_modules["sleep2vec.common"].apply_data_backend_args = lambda *args, **kwargs: None
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
        return module
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@pytest.fixture(scope="module")
def adapt_module() -> ModuleType:
    return _load_adapt_module()


def test_optional_path_accepts_null_for_kaldi_backend(adapt_module: ModuleType):
    assert adapt_module._optional_path("null") is None
    assert adapt_module._optional_path("none") is None
    assert adapt_module._optional_path("preset.pkl") == Path("preset.pkl")


def test_sleep2vec_adapt_applies_data_backend_args_before_loader(adapt_module: ModuleType, monkeypatch):
    sentinel = RuntimeError("backend helper called")
    data_cfg = types.SimpleNamespace(
        backend="kaldi",
        kaldi_data_root="kaldi/root",
        kaldi_manifest="kaldi/root/manifest.json",
        mask_rate=0.15,
        max_tokens=120,
    )
    config_bundle = types.SimpleNamespace(
        data=data_cfg,
        model=object(),
        loss=object(),
        averaging=None,
        adapt=types.SimpleNamespace(new_channels=["ppg"]),
    )

    def _apply_data_backend_args(args, received_data_cfg, *, preset_attr=None):
        assert received_data_cfg is data_cfg
        assert preset_attr == "pretrain_preset_path"
        raise sentinel

    monkeypatch.setattr(adapt_module, "load_pretrain_config", lambda _path: config_bundle)
    monkeypatch.setattr(adapt_module, "apply_data_backend_args", _apply_data_backend_args)
    monkeypatch.setattr(
        adapt_module,
        "get_pretrain_dataloader",
        lambda _args: (_ for _ in ()).throw(AssertionError("loader should not be reached")),
    )

    args = argparse.Namespace(
        config=Path("adapt.yaml"),
        phase="stage1",
        channel_names=["ppg"],
        pretrain_preset_path=Path("preset.pkl"),
    )

    with pytest.raises(RuntimeError, match="backend helper called") as exc_info:
        adapt_module.sleep2vec_adapt(args)
    assert exc_info.value is sentinel


def _write_cli_args(path: Path, *, phase: str) -> None:
    path.write_text(yaml.safe_dump({"phase": phase}))


def test_resolve_adapt_run_artifacts_builds_new_run_path_without_ckpt(adapt_module: ModuleType):
    artifacts = adapt_module._resolve_adapt_run_artifacts(
        ckpt_path=None,
        pretrained_backbone_path=None,
        version_name="wearable-v1",
        backbone_arch="roformer",
        phase="stage1",
        exp_info="ppg actigraphy",
    )

    assert artifacts.save_path == Path("log-adapt/wearable-v1-roformer-adapt-stage1-ppg_actigraphy/checkpoints")
    assert artifacts.run_name == "wearable-v1-roformer-adapt-stage1-ppg_actigraphy"
    assert artifacts.wandb_id is None
    assert artifacts.trainer_ckpt_path is None
    assert artifacts.write_root_files is True


def test_resolve_adapt_run_artifacts_rejects_missing_ckpt(
    tmp_path: Path,
    adapt_module: ModuleType,
):
    missing = tmp_path / "missing.ckpt"

    with pytest.raises(FileNotFoundError, match="ckpt_path not found"):
        adapt_module._resolve_adapt_run_artifacts(
            ckpt_path=missing,
            pretrained_backbone_path=None,
            version_name="wearable-v1",
            backbone_arch="roformer",
            phase="stage2",
        )


def test_resolve_adapt_run_artifacts_rejects_directory_ckpt(
    tmp_path: Path,
    adapt_module: ModuleType,
):
    ckpt_dir = tmp_path / "run_a" / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="ckpt_path must be a file"):
        adapt_module._resolve_adapt_run_artifacts(
            ckpt_path=ckpt_dir,
            pretrained_backbone_path=None,
            version_name="wearable-v1",
            backbone_arch="roformer",
            phase="stage2",
        )


def test_resolve_adapt_run_artifacts_uses_existing_ckpt_parent(
    tmp_path: Path,
    adapt_module: ModuleType,
):
    ckpt_path = tmp_path / "run_a" / "checkpoints.stage2" / "epoch=1.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("checkpoint")
    _write_cli_args(ckpt_path.parent.parent / "cli_args.stage2.yaml", phase="stage2")

    artifacts = adapt_module._resolve_adapt_run_artifacts(
        ckpt_path=ckpt_path,
        pretrained_backbone_path=None,
        version_name="ignored",
        backbone_arch="ignored",
        phase="stage2",
    )

    assert artifacts.save_path == ckpt_path.parent
    assert artifacts.run_name == "run_a"
    assert artifacts.wandb_id == "run_a"
    assert artifacts.trainer_ckpt_path == ckpt_path
    assert artifacts.write_root_files is False


def test_resolve_adapt_run_artifacts_rejects_cross_phase_ckpt_resume(
    tmp_path: Path,
    adapt_module: ModuleType,
):
    ckpt_path = tmp_path / "run_a" / "checkpoints" / "epoch=1.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("checkpoint")
    _write_cli_args(ckpt_path.parent.parent / "cli_args.yaml", phase="stage1")

    with pytest.raises(ValueError, match="resumes an exact adapt stage2 run only"):
        adapt_module._resolve_adapt_run_artifacts(
            ckpt_path=ckpt_path,
            pretrained_backbone_path=None,
            version_name="ignored",
            backbone_arch="ignored",
            phase="stage2",
        )


def test_resolve_adapt_run_artifacts_reuses_stage1_run_for_stage2_transition(
    tmp_path: Path,
    adapt_module: ModuleType,
):
    ckpt_path = tmp_path / "run_a" / "checkpoints" / "epoch=1.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("checkpoint")
    _write_cli_args(ckpt_path.parent.parent / "cli_args.yaml", phase="stage1")

    artifacts = adapt_module._resolve_adapt_run_artifacts(
        ckpt_path=None,
        pretrained_backbone_path=ckpt_path,
        version_name="ignored",
        backbone_arch="ignored",
        phase="stage2",
    )

    assert artifacts.save_path == ckpt_path.parent.parent / "checkpoints.stage2"
    assert artifacts.run_name == "run_a"
    assert artifacts.wandb_id == "run_a"
    assert artifacts.trainer_ckpt_path is None
    assert artifacts.write_root_files is False


def test_resolve_adapt_run_artifacts_prefers_stage1_snapshot_when_present(
    tmp_path: Path,
    adapt_module: ModuleType,
):
    ckpt_path = tmp_path / "run_a" / "checkpoints" / "epoch=1.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("checkpoint")
    _write_cli_args(ckpt_path.parent.parent / "cli_args.yaml", phase="stage2")
    _write_cli_args(ckpt_path.parent.parent / "cli_args.stage1.yaml", phase="stage1")

    artifacts = adapt_module._resolve_adapt_run_artifacts(
        ckpt_path=None,
        pretrained_backbone_path=ckpt_path,
        version_name="ignored",
        backbone_arch="ignored",
        phase="stage2",
    )

    assert artifacts.save_path == ckpt_path.parent.parent / "checkpoints.stage2"
    assert artifacts.run_name == "run_a"
    assert artifacts.trainer_ckpt_path is None


def test_resolve_adapt_run_artifacts_rejects_non_empty_stage2_dir_for_fresh_transition(
    tmp_path: Path,
    adapt_module: ModuleType,
):
    ckpt_path = tmp_path / "run_a" / "checkpoints" / "epoch=1.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("checkpoint")
    _write_cli_args(ckpt_path.parent.parent / "cli_args.yaml", phase="stage1")
    stage2_ckpt = ckpt_path.parent.parent / "checkpoints.stage2" / "epoch=9.ckpt"
    stage2_ckpt.parent.mkdir(parents=True)
    stage2_ckpt.write_text("stage2 checkpoint")

    with pytest.raises(ValueError, match="refuses to reuse a non-empty checkpoints.stage2 directory"):
        adapt_module._resolve_adapt_run_artifacts(
            ckpt_path=None,
            pretrained_backbone_path=ckpt_path,
            version_name="ignored",
            backbone_arch="ignored",
            phase="stage2",
        )


def test_resolve_adapt_run_artifacts_rejects_stage1_checkpoint_for_stage2_resume_after_transition(
    tmp_path: Path,
    adapt_module: ModuleType,
):
    ckpt_path = tmp_path / "run_a" / "checkpoints" / "epoch=1.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("checkpoint")
    _write_cli_args(ckpt_path.parent.parent / "cli_args.stage1.yaml", phase="stage1")
    _write_cli_args(ckpt_path.parent.parent / "cli_args.stage2.yaml", phase="stage2")

    with pytest.raises(ValueError, match="resumes an exact adapt stage2 run only"):
        adapt_module._resolve_adapt_run_artifacts(
            ckpt_path=ckpt_path,
            pretrained_backbone_path=None,
            version_name="ignored",
            backbone_arch="ignored",
            phase="stage2",
        )


def test_resolve_adapt_run_artifacts_rejects_non_stage1_transition_checkpoint(
    tmp_path: Path,
    adapt_module: ModuleType,
):
    ckpt_path = tmp_path / "run_a" / "checkpoints" / "epoch=1.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("checkpoint")
    _write_cli_args(ckpt_path.parent.parent / "cli_args.yaml", phase="stage2")

    with pytest.raises(ValueError, match="prior adapt stage1 checkpoint"):
        adapt_module._resolve_adapt_run_artifacts(
            ckpt_path=None,
            pretrained_backbone_path=ckpt_path,
            version_name="ignored",
            backbone_arch="ignored",
            phase="stage2",
        )


def test_resolve_adapt_run_artifacts_rejects_stage2_checkpoint_for_stage2_transition(
    tmp_path: Path,
    adapt_module: ModuleType,
):
    ckpt_path = tmp_path / "run_a" / "checkpoints.stage2" / "epoch=1.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("checkpoint")
    _write_cli_args(ckpt_path.parent.parent / "cli_args.stage1.yaml", phase="stage1")
    _write_cli_args(ckpt_path.parent.parent / "cli_args.stage2.yaml", phase="stage2")

    with pytest.raises(ValueError, match="prior adapt stage1 checkpoint"):
        adapt_module._resolve_adapt_run_artifacts(
            ckpt_path=None,
            pretrained_backbone_path=ckpt_path,
            version_name="ignored",
            backbone_arch="ignored",
            phase="stage2",
        )


def _args_with_config(config_path: Path, *, phase: str) -> argparse.Namespace:
    return argparse.Namespace(
        config=config_path,
        phase=phase,
        version_name="wearable-v1",
    )


def test_persist_run_config_and_args_writes_root_and_stage1_snapshots(tmp_path: Path):
    config_path = tmp_path / "stage1_config.yaml"
    config_path.write_text("stage: 1\n")
    exp_dir = tmp_path / "log-adapt" / "run_a"

    persist_run_config_and_args(
        _args_with_config(config_path, phase="stage1"),
        exp_dir,
        phase_name="stage1",
        write_root_files=True,
    )

    assert (exp_dir / "config.yaml").read_text() == "stage: 1\n"
    assert (exp_dir / "config.stage1.yaml").read_text() == "stage: 1\n"
    assert yaml.safe_load((exp_dir / "cli_args.yaml").read_text())["phase"] == "stage1"
    assert yaml.safe_load((exp_dir / "cli_args.stage1.yaml").read_text())["phase"] == "stage1"


def test_persist_run_config_and_args_preserves_existing_root_files_for_stage2(tmp_path: Path):
    stage1_config = tmp_path / "stage1_config.yaml"
    stage1_config.write_text("stage: 1\n")
    stage2_config = tmp_path / "stage2_config.yaml"
    stage2_config.write_text("stage: 2\n")
    exp_dir = tmp_path / "log-adapt" / "run_a"

    persist_run_config_and_args(
        _args_with_config(stage1_config, phase="stage1"),
        exp_dir,
        phase_name="stage1",
        write_root_files=True,
    )
    root_config_before = (exp_dir / "config.yaml").read_text()
    root_cli_before = (exp_dir / "cli_args.yaml").read_text()

    persist_run_config_and_args(
        _args_with_config(stage2_config, phase="stage2"),
        exp_dir,
        phase_name="stage2",
        write_root_files=False,
    )

    assert (exp_dir / "config.yaml").read_text() == root_config_before
    assert (exp_dir / "cli_args.yaml").read_text() == root_cli_before
    assert (exp_dir / "config.stage2.yaml").read_text() == "stage: 2\n"
    assert yaml.safe_load((exp_dir / "cli_args.stage2.yaml").read_text())["phase"] == "stage2"


def test_load_saved_cli_args_accepts_persisted_pair_probs(tmp_path: Path, adapt_module: ModuleType):
    config_path = tmp_path / "stage1_config.yaml"
    config_path.write_text("stage: 1\n")
    exp_dir = tmp_path / "log-adapt" / "run_a"
    args = argparse.Namespace(
        config=config_path,
        phase="stage1",
        version_name="wearable-v1",
        train_pair_probs={("breath", "ppg"): 0.4, ("ecg", "ppg"): 0.6},
    )

    persist_run_config_and_args(
        args,
        exp_dir,
        phase_name="stage1",
        write_root_files=True,
    )

    _, loaded = adapt_module._load_saved_cli_args(exp_dir, phase="stage1")
    assert loaded["phase"] == "stage1"
    assert loaded["train_pair_probs"]["['breath', 'ppg']"] == 0.4
    assert loaded["train_pair_probs"]["['ecg', 'ppg']"] == 0.6
