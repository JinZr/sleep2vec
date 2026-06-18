from types import SimpleNamespace

import torch

import sleep2vec.callbacks.pair_acc_logger as pair_acc_logger_module
from sleep2vec.callbacks.pair_acc_logger import PairAccLoggerCallback


class _DummyModule:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.logged: dict[str, float] = {}

    def all_gather(self, tensor):
        return tensor

    def log(self, name, value, **kwargs) -> None:
        self.logged[str(name)] = float(value)


def _build_loader(pairs):
    dataset = SimpleNamespace(reset_pair_selector=lambda: None)
    batch_sampler = SimpleNamespace(pairs=list(pairs))
    return SimpleNamespace(dataset=dataset, batch_sampler=batch_sampler)


def test_pair_acc_logger_tracks_pairs_from_single_validation_loader(monkeypatch) -> None:
    monkeypatch.setattr(pair_acc_logger_module.wandb, "run", None, raising=False)

    callback = PairAccLoggerCallback(["a", "b", "c"])
    trainer = SimpleNamespace(
        val_dataloaders=_build_loader([("a", "b"), ("a", "c"), ("b", "c")]),
        is_global_zero=True,
    )
    module = _DummyModule()

    callback.on_validation_epoch_start(trainer, module)
    callback.on_validation_batch_end(
        trainer,
        module,
        outputs={"acc": torch.tensor(0.5), "batch_size": 4},
        batch={"pair": ("a", "b")},
        batch_idx=0,
    )
    callback.on_validation_batch_end(
        trainer,
        module,
        outputs={"acc": torch.tensor(0.25), "batch_size": 2},
        batch={"pair": ("a", "c")},
        batch_idx=1,
    )
    callback.on_validation_epoch_end(trainer, module)

    assert module.logged["val_pair_acc/a__b"] == 0.5
    assert module.logged["val_pair_acc/a__c"] == 0.25
    assert module.logged["val_pair_acc/b__c"] == 0.0
