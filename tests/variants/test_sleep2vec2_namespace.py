from __future__ import annotations

from copy import deepcopy
import difflib
import importlib
from pathlib import Path
import pickle
import re

import pytest
import yaml

from sleep2vec2.config import load_finetune_config, load_pretrain_config, validate_model_config

REPO_ROOT = Path(__file__).resolve().parents[2]
# Variants carry only the preprocessing modules their runtime imports. Each copy must equal root once its
# package-local imports are rewritten to root form. The only allowed drift is the root-only preset manifest
# feature (--manifest-output, --write-sidecar-manifest, <preset>.manifest.json; doc/agent_contracts/task_recipe.md)
# plus root's in-function normalize_mask_frame import, pinned hunk by hunk in the fixture diff.
VARIANT_PREPROCESS_ROOT_ONLY_HUNKS = {
    "convert_npz_to_kaldi.py": None,
    "save_dataset_presets.py": REPO_ROOT / "tests" / "fixtures" / "save_dataset_presets_root_only.diff",
    "split_index_by_dataset.py": None,
}
PARENTHESIZED_IMPORT = re.compile(r"^([ \t]*from \S+ import )\(([^)]*)\)", re.MULTILINE)


def test_sleep2vec2_copied_runtime_uses_local_namespace():
    stale_import = re.compile(r"(^|\s)(from|import) (sleep2vec|data|preprocess)(\.|\s|$)", re.MULTILINE)
    # data/whole_night_index.py is the documented single owner of whole-night index validation and embedding
    # sample keys (doc/code_index/REUSE_GUIDE.md); the variant extractors share it.
    shared_import_path = "sleep2vec2/extract_embeddings.py"
    shared_import = "from data.whole_night_index import "
    offenders: list[str] = []

    for path in sorted((REPO_ROOT / "sleep2vec2").rglob("*.py")):
        relative_path = str(path.relative_to(REPO_ROOT))
        source = path.read_text()
        if relative_path == shared_import_path:
            source = source.replace(shared_import, "", 1)
        if stale_import.search(source):
            offenders.append(relative_path)

    assert offenders == []


def _join_parenthesized_imports(source: str) -> str:
    # Black wraps the longer variant import lines; joining them keeps line wrapping from reading as drift.
    return PARENTHESIZED_IMPORT.sub(
        lambda match: match.group(1) + ", ".join(name.strip() for name in match.group(2).split(",") if name.strip()),
        source,
    )


def _variant_source_as_root(source: str, package: str) -> str:
    source = re.sub(rf"\b{package}\.(data|preprocess)\.", r"\1.", source)
    source = source.replace(f"{package}.", "sleep2vec.").replace("parents[2]", "parents[1]")
    return _join_parenthesized_imports(source)


def _hunks(source: str, target: str) -> str:
    # Zero-context hunks without line numbers, so an edit made to every copy leaves the pinned hunks unchanged.
    diff = list(difflib.unified_diff(source.splitlines(), target.splitlines(), lineterm="", n=0))[2:]
    return "".join(("@@" if line.startswith("@@") else line) + "\n" for line in diff)


@pytest.mark.parametrize("filename", sorted(VARIANT_PREPROCESS_ROOT_ONLY_HUNKS))
def test_variant_preprocess_copies_match_root(filename: str):
    sleep2vec2_source = (REPO_ROOT / "sleep2vec2" / "preprocess" / filename).read_text()
    sleep2expert_source = (REPO_ROOT / "sleep2expert" / "preprocess" / filename).read_text()
    root_source = _join_parenthesized_imports((REPO_ROOT / "preprocess" / filename).read_text())
    root_only_hunks = VARIANT_PREPROCESS_ROOT_ONLY_HUNKS[filename]

    assert _hunks(sleep2expert_source.replace("sleep2expert", "sleep2vec2"), sleep2vec2_source) == ""
    assert _hunks(_variant_source_as_root(sleep2vec2_source, "sleep2vec2"), root_source) == (
        "" if root_only_hunks is None else root_only_hunks.read_text()
    )


def test_sleep2vec2_data_and_preprocess_modules_import_independently():
    pytest.importorskip("torch")
    pytest.importorskip("pandas")

    modules = [
        "sleep2vec2.data.default_dataset",
        "sleep2vec2.data.kaldi_io",
        "sleep2vec2.data.kaldi_psg_dataset",
        "sleep2vec2.data.psg_pretrain_dataset",
        "sleep2vec2.preprocess.convert_npz_to_kaldi",
        "sleep2vec2.preprocess.save_dataset_presets",
        "sleep2vec2.preprocess.split_index_by_dataset",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name


def test_sleep2vec2_loads_base_sampleindex_preset(tmp_path: Path):
    pytest.importorskip("torch")

    base_dataset = importlib.import_module("data.default_dataset")
    variant_dataset = importlib.import_module("sleep2vec2.data.default_dataset")
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


def test_sleep2vec2_rejects_legacy_roformer_checkpoint_keys(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from sleep2vec2.backbones.roformer import RoFormerConfig, RoFormerEncoderModel
    from sleep2vec2.checkpoints import load_pretrain_init_weights

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


def test_sleep2vec2_configs_parse_with_sleep2vec2_loaders():
    config_root = REPO_ROOT / "configs" / "sleep2vec2"
    config_paths = sorted(config_root.rglob("*.yaml"))

    assert (config_root / "sleep2vec_dense_pretrain.yaml") in config_paths

    for path in config_paths:
        data = yaml.safe_load(path.read_text())
        if isinstance(data.get("finetune"), dict):
            bundle = load_finetune_config(path)
        else:
            bundle = load_pretrain_config(path)
        validate_model_config(bundle.model)


def test_sleep2vec2_finetune_configs_do_not_train_lora():
    config_root = REPO_ROOT / "configs" / "sleep2vec2"
    offenders: dict[str, object] = {}

    for path in sorted(config_root.rglob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if not isinstance(data.get("finetune"), dict):
            continue
        tuning = load_finetune_config(path).finetune.tuning
        if tuning.trains("lora"):
            offenders[str(path.relative_to(REPO_ROOT))] = tuning.preset

    assert offenders == {}


@pytest.mark.parametrize(
    "target_modules",
    [["query", "key", "value"], ["query", "dense"]],
)
def test_sleep2vec2_finetune_config_accepts_lora_flags(tmp_path: Path, target_modules: list[str]):
    source = REPO_ROOT / "configs" / "sleep2vec2" / "ppg_ahi_finetune.yaml"
    data = yaml.safe_load(source.read_text())
    payload = deepcopy(data)
    payload["finetune"]["tuning"] = {
        "preset": "lora",
        "lora": {
            "separate_adapters": True,
            "r": 4,
            "alpha": 12,
            "dropout": 0.15,
            "target_modules": target_modules,
            "use_dora": True,
        },
    }
    path = tmp_path / "lora_enabled.yaml"
    path.write_text(yaml.safe_dump(payload))

    bundle = load_finetune_config(path)

    tuning = bundle.finetune.tuning
    assert tuning.trains("lora") is True
    assert tuning.trains("encoder") is False
    assert tuning.lora.separate_adapters is True
    assert tuning.lora.r == 4
    assert tuning.lora.alpha == 12
    assert tuning.lora.dropout == 0.15
    assert tuning.lora.target_modules == target_modules
    assert tuning.lora.use_dora is True
