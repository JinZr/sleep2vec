from argparse import Namespace
import logging

from sleep2vec2 import finetune, utils


def test_no_test_after_fit_does_not_build_test_loader(monkeypatch):
    calls = []

    def build_loader(_args, **kwargs):
        calls.append(kwargs["split"])
        return kwargs["split"][0]

    monkeypatch.setattr(utils, "_build_finetune_loader", build_loader)
    args = Namespace(
        train_dataset_names=[],
        test_dataset_names=[],
        n_few_shot=0,
        test_after_fit=False,
    )

    train_loader, val_loader, test_loader = utils.get_finetune_dataloaders(args)

    assert (train_loader, val_loader, test_loader) == ("train", "val", None)
    assert calls == [["train"], ["val"]]


def test_prepare_dataloader_reports_disabled_test_as_empty(monkeypatch, caplog):
    monkeypatch.setattr(finetune, "get_finetune_dataloaders", lambda _args: ([1], [2], None))

    with caplog.at_level(logging.INFO):
        loaders = finetune.prepare_dataloader(Namespace())

    assert loaders == ([1], [2], None)
    assert "Prepared dataloaders: train=1 val=1 test=0" in caplog.text
