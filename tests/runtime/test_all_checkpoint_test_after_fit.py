import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

FINETUNE_MODULES = ("sleep2vec.finetune", "sleep2vec2.finetune", "sleep2expert.finetune")
RESULT_PACKAGES = ("sleep2vec", "sleep2vec2", "sleep2expert")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_finetune_module(module_name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    namespace = module_name.split(".", 1)[0]
    stubbed_modules = {
        "pytorch_lightning": ModuleType("pytorch_lightning"),
        "pytorch_lightning.callbacks": ModuleType("pytorch_lightning.callbacks"),
        "pytorch_lightning.callbacks.early_stopping": ModuleType("pytorch_lightning.callbacks.early_stopping"),
        "pytorch_lightning.loggers": ModuleType("pytorch_lightning.loggers"),
        "pytorch_lightning.strategies": ModuleType("pytorch_lightning.strategies"),
        "pytorch_lightning.strategies.ddp": ModuleType("pytorch_lightning.strategies.ddp"),
        "wandb": ModuleType("wandb"),
        f"{namespace}.callbacks": ModuleType(f"{namespace}.callbacks"),
        f"{namespace}.callbacks.grad_scale_logger": ModuleType(f"{namespace}.callbacks.grad_scale_logger"),
        f"{namespace}.common": ModuleType(f"{namespace}.common"),
        f"{namespace}.distributed": ModuleType(f"{namespace}.distributed"),
        f"{namespace}.results": ModuleType(f"{namespace}.results"),
        f"{namespace}.sleep2vec_finetuning": ModuleType(f"{namespace}.sleep2vec_finetuning"),
        f"{namespace}.utils": ModuleType(f"{namespace}.utils"),
    }

    stubbed_modules["pytorch_lightning"].Trainer = object
    stubbed_modules["pytorch_lightning.callbacks"].LearningRateMonitor = object
    stubbed_modules["pytorch_lightning.callbacks"].ModelCheckpoint = object
    stubbed_modules["pytorch_lightning.callbacks.early_stopping"].EarlyStopping = object
    stubbed_modules["pytorch_lightning.loggers"].WandbLogger = object
    stubbed_modules["pytorch_lightning.strategies.ddp"].DDPStrategy = lambda **_kwargs: object()
    stubbed_modules["wandb"].run = None
    stubbed_modules["wandb"].finish = lambda: None
    stubbed_modules["wandb"].Table = lambda **_kwargs: object()
    stubbed_modules[f"{namespace}.callbacks"].build_distributed_ahi_progress_bar = lambda: object()
    stubbed_modules[f"{namespace}.callbacks.grad_scale_logger"].GradScaleLoggerCallback = object
    stubbed_modules[f"{namespace}.common"].apply_finetune_config = lambda *_args, **_kwargs: (None, None)
    stubbed_modules[f"{namespace}.common"].persist_run_config_and_args = lambda *_args, **_kwargs: None
    stubbed_modules[f"{namespace}.distributed"].is_rank_zero_process = lambda: True
    for name in (
        "save_multilabel_per_disease_metrics_csv",
        "save_result_csv",
        "save_result_rows_csv",
        "save_survival_per_disease_metrics_csv",
        "save_training_run_manifest",
    ):
        setattr(stubbed_modules[f"{namespace}.results"], name, lambda *_args, **_kwargs: None)
    stubbed_modules[f"{namespace}.sleep2vec_finetuning"].Sleep2vecFinetuning = object
    stubbed_modules[f"{namespace}.utils"].get_finetune_dataloaders = lambda *_args, **_kwargs: (None, None, None)

    for name, module in stubbed_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    loaded_name = f"_all_checkpoint_{namespace}_finetune"
    spec = importlib.util.spec_from_file_location(loaded_name, REPO_ROOT / namespace / "finetune.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_name} for testing.")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, loaded_name, module)
    spec.loader.exec_module(module)
    return module


def _run_supervised(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    checkpoint_names: tuple[str, ...],
    best_epoch: int = 1,
    test_all_checkpoints_after_fit: bool = True,
):
    finetune_mod = _load_finetune_module(module_name, monkeypatch)
    checkpoints = []
    test_calls = []
    result_rows = []
    manifest_calls = []

    class DummyModel:
        moe_finetune_status = {}

        def moe_finetune_hparams(self):
            return {}

        def moe_finetune_param_group_rows(self):
            return []

    class DummyLogger:
        experiment = SimpleNamespace(log=lambda *args, **kwargs: None)

        def log_hyperparams(self, *args, **kwargs):
            return None

    class DummyCheckpoint:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.dirpath = kwargs["dirpath"]
            self.last_model_path = str(Path(self.dirpath) / "last.ckpt")
            self.best_model_score = 0.75
            self.best_model_path = (
                str(Path(self.dirpath) / f"best-epoch={best_epoch:02d}.ckpt") if "monitor" in kwargs else ""
            )
            checkpoints.append(self)

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            self.is_global_zero = True

        def fit(self, *args, **kwargs):
            checkpoint_dir = Path(checkpoints[0].dirpath)
            checkpoint_dir.mkdir(parents=True)
            for name in checkpoint_names:
                (checkpoint_dir / name).write_text(name)
            if (checkpoint_dir / "epoch=02.ckpt").exists():
                (checkpoint_dir / "epoch=03.ckpt").symlink_to(checkpoint_dir / "epoch=02.ckpt")
            (checkpoint_dir / f"best-epoch={best_epoch:02d}.ckpt").write_text("best")
            (checkpoint_dir / "last.ckpt").write_text("last")

        def test(self, *args, **kwargs):
            checkpoint_path = Path(kwargs["ckpt_path"])
            test_calls.append(str(checkpoint_path))
            match = re.fullmatch(r"epoch=(\d+)(?:-step=\d+)?\.ckpt", checkpoint_path.name)
            score = float(match.group(1)) if match is not None else 99.0
            return [{"test_score": score}]

    args = argparse.Namespace(
        version="unit-test",
        monitor="val_score",
        monitor_mod="max",
        patience=1,
        ckpt_every_n_epochs=1,
        devices=[0],
        epochs=3,
        gradient_clip_val=0.0,
        precision=32,
        check_val_every_n_epoch=1,
        print_diagnostics=False,
        ckpt_path="",
        results_csv_path=tmp_path / "results.csv",
        label_name="custom",
        test_after_fit=True,
        test_all_checkpoints_after_fit=test_all_checkpoints_after_fit,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        finetune_mod,
        "persist_run_config_and_args",
        lambda _args, exp_root: Path(exp_root).mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(finetune_mod, "prepare_dataloader", lambda args: ("train", "val", "test"))
    monkeypatch.setattr(finetune_mod, "Sleep2vecFinetuning", lambda *args, **kwargs: DummyModel())
    monkeypatch.setattr(finetune_mod, "WandbLogger", lambda *args, **kwargs: DummyLogger())
    monkeypatch.setattr(finetune_mod, "EarlyStopping", lambda *args, **kwargs: object())
    monkeypatch.setattr(finetune_mod, "LearningRateMonitor", lambda *args, **kwargs: object())
    monkeypatch.setattr(finetune_mod, "ModelCheckpoint", DummyCheckpoint)
    monkeypatch.setattr(finetune_mod.pl, "Trainer", DummyTrainer)
    monkeypatch.setattr(finetune_mod.shutil, "copy2", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        finetune_mod,
        "save_result_csv",
        lambda metrics, _path, current_args: result_rows.append((current_args.ckpt_path, metrics)),
    )
    monkeypatch.setattr(
        finetune_mod,
        "save_training_run_manifest",
        lambda current_args, **kwargs: manifest_calls.append((current_args.ckpt_path, kwargs)),
    )
    monkeypatch.setattr(finetune_mod.wandb, "run", None, raising=False)
    if hasattr(finetune_mod, "is_rank_zero_process"):
        monkeypatch.setattr(finetune_mod, "is_rank_zero_process", lambda: True)

    finetune_mod.supervised(args, SimpleNamespace(model=object(), averaging=None, finetune=None))
    return checkpoints, test_calls, result_rows, manifest_calls, args


@pytest.mark.parametrize("module_name", FINETUNE_MODULES)
@pytest.mark.parametrize("existing_entry", ("marker.txt", "checkpoints/epoch=00.ckpt"))
def test_finetune_rejects_nonempty_run_directory_before_persisting(
    module_name: str,
    existing_entry: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    finetune_mod = _load_finetune_module(module_name, monkeypatch)
    run_dir = tmp_path / "log-finetune" / "unit-test"
    entry = run_dir / existing_entry
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("stale\n")
    persist_calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(finetune_mod, "is_rank_zero_process", lambda: True)
    monkeypatch.setattr(finetune_mod, "persist_run_config_and_args", lambda *args: persist_calls.append(args))
    args = argparse.Namespace(
        version="unit-test",
        epochs=1,
        test_after_fit=True,
        test_all_checkpoints_after_fit=False,
    )

    with pytest.raises(FileExistsError, match="Use a new --version-name"):
        finetune_mod.supervised(args, SimpleNamespace(model=object(), averaging=None, finetune=None))

    assert persist_calls == []


@pytest.mark.parametrize("module_name", FINETUNE_MODULES)
def test_finetune_nonzero_rank_does_not_reject_rank_zero_runtime_files(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    finetune_mod = _load_finetune_module(module_name, monkeypatch)
    run_dir = tmp_path / "log-finetune" / "unit-test"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("current run\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(finetune_mod, "is_rank_zero_process", lambda: False)
    monkeypatch.setattr(
        finetune_mod,
        "persist_run_config_and_args",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("continued past output preflight")),
    )
    args = argparse.Namespace(
        version="unit-test",
        epochs=1,
        test_after_fit=True,
        test_all_checkpoints_after_fit=False,
    )

    with pytest.raises(RuntimeError, match="continued past output preflight"):
        finetune_mod.supervised(args, SimpleNamespace(model=object(), averaging=None, finetune=None))


@pytest.mark.parametrize("module_name", FINETUNE_MODULES)
def test_direct_finetune_keeps_single_best_checkpoint_behavior(
    module_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoints, test_calls, result_rows, manifest_calls, args = _run_supervised(
        module_name,
        tmp_path,
        monkeypatch,
        checkpoint_names=(),
        test_all_checkpoints_after_fit=False,
    )

    expected_best = str(Path("log-finetune/unit-test/checkpoints/best-epoch=01.ckpt"))
    assert "save_on_train_epoch_end" not in checkpoints[0].kwargs
    assert test_calls == [expected_best]
    assert result_rows == [("", {"test_score": 99.0})]
    assert manifest_calls[-1][1]["metrics"] == {"test_score": 99.0}
    assert manifest_calls[-1][1]["checkpoint_test_results"] == []
    assert args.ckpt_path == ""


@pytest.mark.parametrize("module_name", FINETUNE_MODULES)
def test_all_checkpoint_mode_tests_every_regular_checkpoint_and_keeps_best_last(
    module_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoints, test_calls, result_rows, manifest_calls, args = _run_supervised(
        module_name,
        tmp_path,
        monkeypatch,
        checkpoint_names=("epoch=00.ckpt", "epoch=01.ckpt", "epoch=02.ckpt"),
    )

    checkpoint_dir = (tmp_path / "log-finetune" / "unit-test" / "checkpoints").resolve()
    expected_paths = [
        str(checkpoint_dir / "epoch=00.ckpt"),
        str(checkpoint_dir / "epoch=02.ckpt"),
        str(checkpoint_dir / "epoch=01.ckpt"),
    ]
    assert checkpoints[0].kwargs["save_on_train_epoch_end"] is False
    assert test_calls == expected_paths
    assert [path for path, _metrics in result_rows] == expected_paths
    assert args.ckpt_path == ""

    restored_path, manifest = manifest_calls[-1]
    assert restored_path == ""
    assert manifest["metrics"] == {"test_score": 1.0}
    assert manifest["checkpoint_test_results"] == [
        {"checkpoint_path": expected_paths[0], "epoch": 0, "metrics": {"test_score": 0.0}},
        {"checkpoint_path": expected_paths[1], "epoch": 2, "metrics": {"test_score": 2.0}},
        {"checkpoint_path": expected_paths[2], "epoch": 1, "metrics": {"test_score": 1.0}},
    ]


def test_all_checkpoint_mode_evaluates_best_alias_when_best_epoch_was_not_periodically_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _checkpoints, test_calls, _result_rows, manifest_calls, _args = _run_supervised(
        "sleep2vec.finetune",
        tmp_path,
        monkeypatch,
        checkpoint_names=("epoch=00.ckpt", "epoch=02.ckpt"),
        best_epoch=1,
    )

    checkpoint_dir = (tmp_path / "log-finetune" / "unit-test" / "checkpoints").resolve()
    assert test_calls == [
        str(checkpoint_dir / "epoch=00.ckpt"),
        str(checkpoint_dir / "epoch=02.ckpt"),
        str(Path("log-finetune/unit-test/checkpoints/best-epoch=01.ckpt")),
    ]
    assert manifest_calls[-1][1]["metrics"] == {"test_score": 99.0}
    assert len(manifest_calls[-1][1]["checkpoint_test_results"]) == 2


@pytest.mark.parametrize(
    ("checkpoint_names", "error"),
    [
        ((), "No regular epoch=\\*\\.ckpt"),
        (("epoch=bad.ckpt",), "Malformed periodic checkpoint name"),
        (("epoch=1.ckpt", "epoch=01.ckpt"), "Duplicate periodic checkpoint epoch"),
    ],
)
def test_all_checkpoint_mode_fails_on_incomplete_checkpoint_evidence(
    checkpoint_names: tuple[str, ...], error: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    with pytest.raises(ValueError, match=error):
        _run_supervised(
            "sleep2vec.finetune",
            tmp_path,
            monkeypatch,
            checkpoint_names=checkpoint_names,
        )


@pytest.mark.parametrize("module_name", FINETUNE_MODULES)
def test_all_checkpoint_mode_requires_test_after_fit_and_positive_epochs(
    module_name: str, monkeypatch: pytest.MonkeyPatch
):
    finetune_mod = _load_finetune_module(module_name, monkeypatch)
    bundle = SimpleNamespace(model=object(), averaging=None, finetune=None)

    with pytest.raises(ValueError, match="requires --test-after-fit"):
        finetune_mod.supervised(
            argparse.Namespace(test_after_fit=False, test_all_checkpoints_after_fit=True, epochs=1),
            bundle,
        )
    with pytest.raises(ValueError, match="requires --epochs greater than 0"):
        finetune_mod.supervised(
            argparse.Namespace(test_after_fit=True, test_all_checkpoints_after_fit=True, epochs=0),
            bundle,
        )


@pytest.mark.parametrize("package_name", RESULT_PACKAGES)
def test_training_manifest_serializes_checkpoint_test_results(package_name: str, tmp_path: Path, monkeypatch):
    results_mod = importlib.import_module(f"{package_name}.results")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    checkpoint = (tmp_path / "epoch=00.ckpt").resolve()
    args = argparse.Namespace(
        version="unit-test",
        config=tmp_path / "config.yaml",
        label_name="custom",
        test_after_fit=True,
        test_all_checkpoints_after_fit=True,
    )
    manifest_path = tmp_path / f"{package_name}.json"

    results_mod.save_training_run_manifest(
        args,
        manifest_path=manifest_path,
        status="completed",
        metrics={"test_score": 0.5},
        checkpoint_test_results=[{"checkpoint_path": str(checkpoint), "epoch": 0, "metrics": {"test_score": 0.5}}],
    )

    payload = json.loads(manifest_path.read_text())
    assert payload["test_all_checkpoints_after_fit"] is True
    assert payload["checkpoint_test_results"] == [
        {"checkpoint_path": str(checkpoint), "epoch": 0, "metrics": {"test_score": 0.5}}
    ]
