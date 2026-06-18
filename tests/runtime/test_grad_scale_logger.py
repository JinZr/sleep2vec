from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CALLBACK_PATHS = (
    Path("sleep2vec/callbacks/grad_scale_logger.py"),
    Path("sleep2vec2/callbacks/grad_scale_logger.py"),
    Path("sleep2expert/callbacks/grad_scale_logger.py"),
)


class _DummyModule:
    def __init__(self) -> None:
        self.logs = []

    def log(self, name, value, **kwargs) -> None:
        self.logs.append((name, value, kwargs))


def _load_callback(monkeypatch, relative_path: Path):
    fake_pl = ModuleType("pytorch_lightning")
    fake_pl.Callback = object
    monkeypatch.setitem(sys.modules, "pytorch_lightning", fake_pl)

    spec = importlib.util.spec_from_file_location(
        f"grad_scale_logger_{relative_path.parts[0]}",
        REPO_ROOT / relative_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.GradScaleLoggerCallback


@pytest.mark.parametrize("relative_path", CALLBACK_PATHS)
def test_grad_scale_logger_logs_scaler_value(monkeypatch, relative_path: Path) -> None:
    callback_cls = _load_callback(monkeypatch, relative_path)
    callback = callback_cls()
    module = _DummyModule()
    trainer = SimpleNamespace(
        is_global_zero=True,
        precision_plugin=SimpleNamespace(scaler=SimpleNamespace(get_scale=lambda: 32768.0)),
    )

    callback.on_before_optimizer_step(trainer, module, object())

    assert module.logs == [
        (
            "train/grad_scale",
            32768.0,
            {
                "prog_bar": False,
                "logger": True,
                "sync_dist": False,
                "on_step": True,
                "on_epoch": False,
                "rank_zero_only": True,
            },
        )
    ]


@pytest.mark.parametrize("relative_path", CALLBACK_PATHS)
def test_grad_scale_logger_skips_without_scaler(monkeypatch, relative_path: Path) -> None:
    callback_cls = _load_callback(monkeypatch, relative_path)
    callback = callback_cls()
    module = _DummyModule()
    trainer = SimpleNamespace(is_global_zero=True, precision_plugin=SimpleNamespace(scaler=None))

    callback.on_before_optimizer_step(trainer, module, object())

    assert module.logs == []


@pytest.mark.parametrize("relative_path", CALLBACK_PATHS)
def test_grad_scale_logger_skips_non_global_zero(monkeypatch, relative_path: Path) -> None:
    callback_cls = _load_callback(monkeypatch, relative_path)
    callback = callback_cls()
    module = _DummyModule()
    trainer = SimpleNamespace(
        is_global_zero=False,
        precision_plugin=SimpleNamespace(scaler=SimpleNamespace(get_scale=lambda: 32768.0)),
    )

    callback.on_before_optimizer_step(trainer, module, object())

    assert module.logs == []
