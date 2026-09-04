"""The three variants are enforced forks: they duplicate the finetune tuning schema
rather than importing a shared module (see tests/variants/test_sleep2expert_namespace.py).
Duplication is only safe if the copies stay in agreement, so this file pins the parts
that must be identical across variants and the parts that are allowed to differ.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import re

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

VARIANTS = ("sleep2vec", "sleep2vec2", "sleep2expert")
NON_MOE_VARIANTS = ("sleep2vec", "sleep2vec2")

# Groups every variant must know about, and the presets every variant must offer.
SHARED_GROUPS = ("head", "encoder", "tokenizers", "projection", "lora")
SHARED_PRESETS = ("full", "head_only", "lora", "custom")
# Only the MoE fork adds these; the other two must not know them.
MOE_GROUPS = ("experts", "routers")
MOE_PRESETS = ("moe_conservative", "moe_conservative_routers", "moe_top_experts")


def _config(variant: str):
    return importlib.import_module(f"{variant}.config")


def _presets(variant: str) -> dict[str, dict[str, tuple[bool, float]] | None]:
    return _config(variant)._FINETUNE_TUNING_PRESETS


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_exposes_the_shared_groups(variant: str):
    groups = _config(variant).FINETUNE_TUNING_GROUPS

    assert set(SHARED_GROUPS) <= set(groups)
    assert len(groups) == len(set(groups))
    # Relative ordering is load-bearing: it fixes optimizer parameter-group order in the
    # logs, so a shared group must not overtake another shared group in one fork only.
    assert tuple(group for group in groups if group in SHARED_GROUPS) == SHARED_GROUPS


@pytest.mark.parametrize("variant", NON_MOE_VARIANTS)
def test_non_moe_variants_have_exactly_the_shared_groups(variant: str):
    assert _config(variant).FINETUNE_TUNING_GROUPS == SHARED_GROUPS


def test_sleep2expert_adds_only_the_moe_groups():
    groups = _config("sleep2expert").FINETUNE_TUNING_GROUPS

    assert set(groups) - set(SHARED_GROUPS) == set(MOE_GROUPS)
    assert _config("sleep2expert").FINETUNE_TUNING_MOE_GROUPS == frozenset(MOE_GROUPS)


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_offers_the_shared_presets(variant: str):
    presets = _presets(variant)

    assert set(SHARED_PRESETS) <= set(presets)
    assert presets["custom"] is None  # `custom` carries no table by construction


@pytest.mark.parametrize("variant", NON_MOE_VARIANTS)
def test_non_moe_variants_reject_the_moe_presets(variant: str):
    assert set(_presets(variant)) == set(SHARED_PRESETS)


def test_sleep2expert_adds_only_the_moe_presets():
    assert set(_presets("sleep2expert")) - set(SHARED_PRESETS) == set(MOE_PRESETS)


@pytest.mark.parametrize("preset", ["full", "head_only", "lora"])
def test_shared_presets_agree_on_the_shared_groups(preset: str):
    """A preset name must mean the same thing in every variant, otherwise a config
    moved between variants would silently change what trains."""
    tables = {variant: _presets(variant)[preset] for variant in VARIANTS}
    restricted = {variant: {group: table[group] for group in SHARED_GROUPS} for variant, table in tables.items()}

    assert len(set(map(str, restricted.values()))) == 1, restricted


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_preset_covers_every_group_of_its_variant(variant: str):
    groups = set(_config(variant).FINETUNE_TUNING_GROUPS)

    for preset, table in _presets(variant).items():
        if table is None:
            continue
        assert set(table) == groups, f"{variant}:{preset}"


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_preset_trains_the_head(variant: str):
    """Nothing downstream can recover from a run where no head parameter trains."""
    for preset, table in _presets(variant).items():
        if table is None:
            continue
        assert table["head"][0] is True, f"{variant}:{preset}"


@pytest.mark.parametrize("variant", VARIANTS)
def test_no_preset_trains_the_encoder_and_lora_together(variant: str):
    """Adapter insertion freezes the backbone first, so the pair is unrepresentable."""
    for preset, table in _presets(variant).items():
        if table is None:
            continue
        assert not (table["encoder"][0] and table["lora"][0]), f"{variant}:{preset}"


@pytest.mark.parametrize("variant", VARIANTS)
def test_no_preset_pairs_a_frozen_group_with_a_learning_rate_scale(variant: str):
    """lr_scale == 0 used to double as a freeze switch; a frozen group now carries the
    neutral scale so the two axes cannot be confused again."""
    for preset, table in _presets(variant).items():
        if table is None:
            continue
        for group, (train, lr_scale) in table.items():
            assert lr_scale > 0.0, f"{variant}:{preset}:{group}"
            if not train:
                assert lr_scale == 1.0, f"{variant}:{preset}:{group}"


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_rejects_the_same_legacy_keys(variant: str):
    assert set(_config(variant)._LEGACY_FINETUNE_TRAINABILITY_KEYS) >= {"freeze_tokenizer", "lora"}


REPRESENTATIVE_CONFIGS = {
    "sleep2vec": "configs/ppg_ahi_finetune_large.yaml",
    "sleep2vec2": "configs/sleep2vec2/ppg_ahi_finetune_large.yaml",
    "sleep2expert": "configs/sleep2expert/moe/heartbeat_breath_ahi_finetune.yaml",
}


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_requires_a_tuning_block(variant: str, tmp_path: Path):
    """`finetune.tuning` has no default: a config must state its trainability policy."""
    payload = yaml.safe_load((REPO_ROOT / REPRESENTATIVE_CONFIGS[variant]).read_text())
    payload["finetune"].pop("tuning")
    path = tmp_path / "no_tuning.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="finetune.tuning is required"):
        _config(variant).load_finetune_config(path)


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_points_a_legacy_config_at_the_conversion_table(variant: str, tmp_path: Path):
    payload = yaml.safe_load((REPO_ROOT / REPRESENTATIVE_CONFIGS[variant]).read_text())
    payload["finetune"].pop("tuning")
    payload["finetune"]["freeze_tokenizer"] = True
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match=re.escape("doc/finetune_tuning_schema_refactor.md")):
        _config(variant).load_finetune_config(path)


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_rejects_an_unknown_finetune_block(variant: str, tmp_path: Path):
    """A misspelled block name must fail rather than fall back to a default.

    `moe_regularization` used to sit inside `moe_tuning`, which rejected unknown keys.
    Lifting it to a sibling of `tuning` would otherwise let `moe_regulrization` silently
    disable the auxiliary loss a run was configured around.
    """
    payload = yaml.safe_load((REPO_ROOT / REPRESENTATIVE_CONFIGS[variant]).read_text())
    payload["finetune"]["moe_regulrization"] = {"enabled": True}
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="finetune has unsupported fields"):
        _config(variant).load_finetune_config(path)


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_tells_an_absent_groups_key_from_a_malformed_one(variant: str, tmp_path: Path):
    """Only absence means "no overrides"; anything else is a mapping that failed to parse."""
    payload = yaml.safe_load((REPO_ROOT / REPRESENTATIVE_CONFIGS[variant]).read_text())
    payload["finetune"]["tuning"] = {"preset": "head_only", "groups": []}
    path = tmp_path / "empty_groups.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="finetune.tuning.groups must be a mapping"):
        _config(variant).load_finetune_config(path)

    payload["finetune"]["tuning"] = {"preset": "head_only", "groups": None}
    path = tmp_path / "null_groups.yaml"
    path.write_text(yaml.safe_dump(payload))

    assert _config(variant).load_finetune_config(path).finetune.tuning.preset == "head_only"
