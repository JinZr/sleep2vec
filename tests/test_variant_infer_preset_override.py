import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest


VARIANT_PACKAGES = ("sleep2vec2", "sleep2expert")


@pytest.mark.parametrize("package_name", VARIANT_PACKAGES)
def test_variant_infer_parse_args_accepts_inference_preset_path(monkeypatch: pytest.MonkeyPatch, package_name: str):
    infer_mod = importlib.import_module(f"{package_name}.infer")
    monkeypatch.setattr(
        "sys.argv",
        [
            f"{package_name}.infer",
            "--config",
            "config.yaml",
            "--ckpt-path",
            "best.ckpt",
            "--label-name",
            "ahi",
            "--inference-preset-path",
            "preset.pkl",
        ],
    )

    args = infer_mod.parse_args()

    assert args.inference_preset_path == Path("preset.pkl")


@pytest.mark.parametrize("package_name", VARIANT_PACKAGES)
def test_variant_run_inference_applies_inference_preset_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, package_name: str
):
    infer_mod = importlib.import_module(f"{package_name}.infer")
    captured: dict[str, object] = {}
    config_preset = tmp_path / "config.pkl"
    override_preset = tmp_path / "override.pkl"

    @dataclass
    class _DummyBundle:
        finetune: object = None
        averaging: object = None

    class _DummyModule:
        def __init__(self, args, model_cfg, finetune_config=None, averaging_config=None):
            captured["module_preset_path"] = args.finetune_preset_path

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def test(self, model=None, ckpt_path=None, dataloaders=None):
            return [{"ahi_pearson": 0.5}]

    def _apply_config(args):
        args.finetune_preset_path = config_preset
        return _DummyBundle(), object()

    def _build_loader(args):
        captured["loader_preset_path"] = args.finetune_preset_path
        return "loader"

    monkeypatch.setattr(infer_mod, "apply_finetune_config", _apply_config)
    monkeypatch.setattr(infer_mod, "_build_inference_loader", _build_loader)
    monkeypatch.setattr(infer_mod, "Sleep2vecFinetuning", _DummyModule)
    monkeypatch.setattr(infer_mod.pl, "Trainer", _DummyTrainer)
    monkeypatch.setattr(infer_mod, "_init_wandb", lambda args: None)

    args = argparse.Namespace(
        label_name="ahi",
        avg_ckpts=1,
        ckpt_path="/tmp/model.ckpt",
        avg_ckpt_dir=None,
        config=Path("dummy.yaml"),
        precision=32,
        accelerator="cpu",
        devices=[0],
        batch_size=4,
        eval_split="test",
        seed=4523,
        wandb=False,
        results_csv_path=None,
        inference_preset_path=override_preset,
    )

    infer_mod.run_inference(args)

    assert captured["loader_preset_path"] == override_preset
    assert captured["module_preset_path"] == override_preset


@pytest.mark.parametrize("package_name", VARIANT_PACKAGES)
def test_variant_result_csv_records_effective_preset_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, package_name: str
):
    results_mod = importlib.import_module(f"{package_name}.results")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "results.csv"
    args = argparse.Namespace(
        config=Path("configs/ppg_ahi_finetune_large.yaml"),
        label_name="ahi",
        eval_split="test",
        ckpt_path="best",
        lr=1e-4,
        batch_size=8,
        n_few_shot=None,
        channel_names=["ppg"],
        wandb_name=None,
        wandb_id=None,
        finetune_preset_path=Path("config/preset.pkl"),
        inference_preset_path=Path("override/preset.pkl"),
    )

    results_mod.save_result_csv({"test_loss": 0.1}, str(csv_path), args)

    df = pd.read_csv(csv_path)

    assert df.loc[0, "preset_path"] == "override/preset.pkl"
