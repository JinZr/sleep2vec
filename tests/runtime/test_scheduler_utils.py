from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")


SCHEDULER_MODULES = [
    "sleep2vec.schedulers",
    "sleep2vec2.schedulers",
    "sleep2expert.schedulers",
]


def _lr_lambda(module_name: str, *, total_steps: int, warmup_steps: int | None):
    module = importlib.import_module(module_name)
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = module.build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    return scheduler.lr_lambdas[0]


@pytest.mark.parametrize("module_name", SCHEDULER_MODULES)
def test_explicit_warmup_matches_existing_schedule(module_name: str):
    lr_lambda = _lr_lambda(module_name, total_steps=100, warmup_steps=10)

    assert lr_lambda(0) == pytest.approx(0.0)
    assert lr_lambda(5) == pytest.approx(0.5)
    assert lr_lambda(10) == pytest.approx(1.0)
    assert lr_lambda(100) == pytest.approx(0.1)


@pytest.mark.parametrize("module_name", SCHEDULER_MODULES)
def test_default_warmup_is_three_percent(module_name: str):
    lr_lambda = _lr_lambda(module_name, total_steps=1000, warmup_steps=None)

    assert lr_lambda(15) == pytest.approx(0.5)
    assert lr_lambda(30) == pytest.approx(1.0)


@pytest.mark.parametrize("module_name", SCHEDULER_MODULES)
def test_warmup_is_clamped_to_total_steps(module_name: str):
    lr_lambda = _lr_lambda(module_name, total_steps=100, warmup_steps=200)

    assert lr_lambda(50) == pytest.approx(0.5)
    assert lr_lambda(100) == pytest.approx(1.0)
