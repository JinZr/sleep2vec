from __future__ import annotations

from copy import deepcopy
import importlib
from pathlib import Path
import pickle
import re

import pytest
import yaml

from sleep2expert.config import load_finetune_config, load_pretrain_config, validate_model_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sleep2expert_copied_runtime_uses_local_namespace():
    stale_import = re.compile(r"(^|\s)(from|import) (sleep2vec2|sleep2vec|data|preprocess)(\.|\s|$)", re.MULTILINE)
    offenders: list[str] = []

    for path in sorted((REPO_ROOT / "sleep2expert").rglob("*.py")):
        match = stale_import.search(path.read_text())
        if match:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_sleep2expert_data_and_preprocess_modules_import_independently():
    pytest.importorskip("torch")
    pytest.importorskip("pandas")

    modules = [
        "sleep2expert.data.default_dataset",
        "sleep2expert.data.kaldi_io",
        "sleep2expert.data.kaldi_psg_dataset",
        "sleep2expert.data.psg_pretrain_dataset",
        "sleep2expert.preprocess.convert_npz_to_kaldi",
        "sleep2expert.preprocess.save_dataset_presets",
        "sleep2expert.preprocess.split_index_by_dataset",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name


def test_sleep2expert_loads_base_sampleindex_preset(tmp_path: Path):
    pytest.importorskip("torch")

    base_dataset = importlib.import_module("data.default_dataset")
    variant_dataset = importlib.import_module("sleep2expert.data.default_dataset")
    preset_path = tmp_path / "legacy_preset.pkl"
    legacy_item = base_dataset.SampleIndex(
        id="legacy",
        path="sample.npz",
        start=0,
        end=10,
        payload={"available_channels": ["ppg", "ahi"]},
        metadata={"split": "train"},
    )
    with preset_path.open("wb") as f:
        pickle.dump([legacy_item], f)

    dataset = variant_dataset.DefaultDataset(
        save_preset_path=None,
        load_preset_path=str(preset_path),
        data=None,
        split=["train"],
        extractors={},
        tokenizers={},
        mask_generators={},
        dataloader_config={},
    )

    assert isinstance(dataset.data[0], variant_dataset.SampleIndex)
    assert dataset.data[0].payload == legacy_item.payload
    assert dataset.data[0].metadata == legacy_item.metadata


def test_sleep2expert_rejects_legacy_roformer_checkpoint_keys(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from sleep2expert.backbones.roformer import RoFormerConfig, RoFormerEncoderModel
    from sleep2expert.checkpoints import load_pretrain_init_weights

    model = RoFormerEncoderModel(
        RoFormerConfig(
            vocab_size=37,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=64,
        )
    )
    ckpt_path = tmp_path / "legacy.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.encoder.layer.0.attention.self.query.weight": torch.randn(32, 32),
                "model.encoder.layer.0.attention.output.LayerNorm.weight": torch.ones(32),
            }
        },
        ckpt_path,
    )

    with pytest.raises(ValueError, match="does not support loading legacy sleep2vec/HF RoFormer checkpoints"):
        load_pretrain_init_weights(model, ckpt_path, device="cpu", strict=False)


def test_sleep2expert_configs_parse_with_sleep2expert_loaders():
    config_root = REPO_ROOT / "configs" / "sleep2expert"
    config_paths = sorted(config_root.rglob("*.yaml"))

    assert (config_root / "sleep2expert_dense_pretrain.yaml") in config_paths

    for path in config_paths:
        data = yaml.safe_load(path.read_text())
        if isinstance(data.get("finetune"), dict):
            bundle = load_finetune_config(path)
        else:
            bundle = load_pretrain_config(path)
        validate_model_config(bundle.model)


def test_sleep2expert_finetune_configs_disable_lora():
    config_root = REPO_ROOT / "configs" / "sleep2expert"
    expected = {
        "freeze_backbone_and_insert_lora": False,
        "insert_lora": False,
        "separate_adapters": False,
    }
    offenders: dict[str, dict[str, object]] = {}

    for path in sorted(config_root.rglob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        finetune = data.get("finetune")
        if not isinstance(finetune, dict):
            continue
        lora = finetune.get("lora")
        actual = {key: lora.get(key) if isinstance(lora, dict) else None for key in expected}
        if actual != expected:
            offenders[str(path.relative_to(REPO_ROOT))] = actual

    assert offenders == {}


@pytest.mark.parametrize(
    "target_modules",
    [["query", "key", "value"], ["query", "dense_in", "dense_out"]],
)
def test_sleep2expert_finetune_config_accepts_lora_flags(tmp_path: Path, target_modules: list[str]):
    source = REPO_ROOT / "configs" / "sleep2expert" / "heartbeat_breath_ahi_finetune_large.yaml"
    data = yaml.safe_load(source.read_text())
    payload = deepcopy(data)
    payload["finetune"]["lora"].update(
        {
            "freeze_backbone_and_insert_lora": True,
            "insert_lora": True,
            "separate_adapters": True,
            "r": 4,
            "alpha": 12,
            "dropout": 0.15,
            "target_modules": target_modules,
            "use_dora": True,
        }
    )
    path = tmp_path / "lora_enabled.yaml"
    path.write_text(yaml.safe_dump(payload))

    bundle = load_finetune_config(path)

    assert bundle.finetune.lora.freeze_backbone_and_insert_lora is True
    assert bundle.finetune.lora.insert_lora is True
    assert bundle.finetune.lora.separate_adapters is True
    assert bundle.finetune.lora.r == 4
    assert bundle.finetune.lora.alpha == 12
    assert bundle.finetune.lora.dropout == 0.15
    assert bundle.finetune.lora.target_modules == target_modules
    assert bundle.finetune.lora.use_dora is True


@pytest.mark.parametrize("target_modules", [["router"], ["moe_ffn.router"], ["query", "router"]])
def test_sleep2expert_finetune_config_rejects_router_lora_targets(tmp_path: Path, target_modules: list[str]):
    source = REPO_ROOT / "configs" / "sleep2expert" / "heartbeat_breath_ahi_finetune_large.yaml"
    data = yaml.safe_load(source.read_text())
    payload = deepcopy(data)
    payload["finetune"]["lora"]["target_modules"] = target_modules
    path = tmp_path / "router_lora.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="does not support router target modules"):
        load_finetune_config(path)
