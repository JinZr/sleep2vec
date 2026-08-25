from argparse import Namespace
import importlib

import pytest


@pytest.mark.parametrize("package_name", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_no_test_after_fit_does_not_build_test_loader(package_name, monkeypatch):
    utils = importlib.import_module(f"{package_name}.utils")
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
