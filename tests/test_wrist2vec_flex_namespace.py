import argparse
from pathlib import Path
import re

import wrist2vec_flex
import wrist2vec_flex.adapt as adapt_module
import wrist2vec_flex.data as wrist_data
from wrist2vec_flex.downstream_model import Wrist2vecDownstreamModel
import wrist2vec_flex.finetune as finetune_module
import wrist2vec_flex.infer as infer_module
import wrist2vec_flex.preprocess.mask_missing_stats as mask_missing_stats_module
import wrist2vec_flex.preprocess.merge_dataset_presets as merge_dataset_presets_module
import wrist2vec_flex.preprocess.save_dataset_presets as save_dataset_presets_module
import wrist2vec_flex.preprocess.split_index_by_dataset as split_index_by_dataset_module
import wrist2vec_flex.pretrain as pretrain_module
from wrist2vec_flex.pretrain_model import Wrist2vecPretrainModel
import wrist2vec_flex.registry as wrist_registry
from wrist2vec_flex.wrist2vec_adaptation import Wrist2vecAdaptation
from wrist2vec_flex.wrist2vec_finetuning import Wrist2vecFinetuning
from wrist2vec_flex.wrist2vec_modelling import Wrist2vecPretraining


def test_wrist2vec_package_and_entrypoints_import():
    assert wrist2vec_flex.__all__ == []
    assert callable(pretrain_module.wrist2vec_pretrain)
    assert callable(adapt_module.wrist2vec_adapt)
    assert callable(finetune_module.supervised)
    assert callable(infer_module.run_inference)
    assert hasattr(wrist_data, "PSGPretrainDataset")
    assert callable(save_dataset_presets_module.main)
    assert callable(merge_dataset_presets_module.main)
    assert callable(split_index_by_dataset_module.main)
    assert callable(mask_missing_stats_module.main)


def test_wrist2vec_public_classes_resolve():
    assert Wrist2vecPretrainModel.__name__ == "Wrist2vecPretrainModel"
    assert Wrist2vecDownstreamModel.__name__ == "Wrist2vecDownstreamModel"
    assert Wrist2vecPretraining.__name__ == "Wrist2vecPretraining"
    assert Wrist2vecAdaptation.__name__ == "Wrist2vecAdaptation"
    assert Wrist2vecFinetuning.__name__ == "Wrist2vecFinetuning"


def test_wrist2vec_registry_exposes_resnet1d_tokenizer():
    assert "resnet1d" in wrist_registry.available_tokenizers()


def test_wrist2vec_infer_parse_args_accepts_inference_preset_path(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "wrist2vec_flex.infer",
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

    args = infer_module.parse_args()

    assert args.inference_preset_path == Path("preset.pkl")


def test_wrist2vec_run_inference_applies_inference_preset_override(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    config_preset = tmp_path / "config.pkl"
    override_preset = tmp_path / "override.pkl"

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
        return argparse.Namespace(finetune=None, averaging=None), object()

    def _build_loader(args):
        captured["loader_preset_path"] = args.finetune_preset_path
        return "loader"

    monkeypatch.setattr(infer_module, "apply_finetune_config", _apply_config)
    monkeypatch.setattr(infer_module, "_build_inference_loader", _build_loader)
    monkeypatch.setattr(infer_module, "Wrist2vecFinetuning", _DummyModule)
    monkeypatch.setattr(infer_module.pl, "Trainer", _DummyTrainer)
    monkeypatch.setattr(infer_module, "_init_wandb", lambda args: None)

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

    infer_module.run_inference(args)

    assert captured["loader_preset_path"] == override_preset
    assert captured["module_preset_path"] == override_preset


def test_wrist2vec_flex_runtime_uses_local_namespace():
    stale_import = re.compile(r"(^|\s)(from|import) wrist2vec(\.|\s|$)", re.MULTILINE)
    offenders: list[str] = []

    for path in sorted((Path(__file__).resolve().parents[1] / "wrist2vec_flex").rglob("*.py")):
        if stale_import.search(path.read_text()):
            offenders.append(str(path.relative_to(Path(__file__).resolve().parents[1])))

    assert offenders == []
