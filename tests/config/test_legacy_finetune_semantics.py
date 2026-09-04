"""Pin the legacy semantics the `finetune.tuning` schema replaced.

The equivalence gate in `test_finetune_tuning_equivalence.py` replays a manifest that was
generated from `legacy_finetune_semantics.py`, so a misreading of the legacy schema would be
baked into both sides of that comparison and pass. These cases state the legacy behaviour
directly, from the pre-refactor `sleep2expert/config.py`, and they cover the defaults no
shipped config happened to exercise -- exactly where a silent misreading would survive.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.config.legacy_finetune_semantics import LEGACY_INSERT_LORA_DEFAULTS, legacy_trainability_table  # noqa: E402


def test_omitted_lr_scales_follow_the_mode_they_were_defaulted_from() -> None:
    """`_default_finetune_moe_lr_scales(mode)` was mode-dependent, and 0.0 froze a group.

    Under `head_only` the backbone and experts defaulted to 0.0. Reading one flat default
    table would migrate this config to a policy that trains a backbone the legacy run held
    frozen -- a silent behaviour change wearing the name of a faithful migration.
    """
    table = legacy_trainability_table({"moe_tuning": {"mode": "head_only"}}, variant="sleep2expert")

    assert table["head"] == [True, 1.0]
    assert table["backbone"] == [False, 0.0]
    assert table["experts"] == [False, 0.0]

    trainable_routers = legacy_trainability_table(
        {"moe_tuning": {"mode": "conservative_full_router_trainable"}}, variant="sleep2expert"
    )
    assert trainable_routers["routers"] == [True, 0.01]


def test_explicit_lr_scales_still_win_over_the_mode_default() -> None:
    """The mode only supplies values for the groups a config left out.

    `head_only` still freezes the backbone on its own, so the explicit scale shows up as a
    scale rather than as a thaw -- which is how the two mechanisms differed.
    """
    table = legacy_trainability_table(
        {"moe_tuning": {"mode": "head_only", "lr_scales": {"backbone": 0.1}}}, variant="sleep2expert"
    )

    assert table["backbone"] == [False, 0.1]

    thawed = legacy_trainability_table(
        {"moe_tuning": {"mode": "custom", "lr_scales": {"backbone": 0.1}}}, variant="sleep2expert"
    )
    assert thawed["backbone"] == [True, 0.1]


def test_top_moe_layer_expert_only_defaults_to_the_deepest_moe_layer() -> None:
    """Validation filled `train_moe_layer_indices` when the config left it out.

    Such a config names no layer in its own text, so reading the key alone would migrate it
    to a policy that trains no experts at all.
    """
    block = {"moe_tuning": {"mode": "top_moe_layer_expert_only", "lr_scales": {"backbone": 0.0}}}

    assert legacy_trainability_table(block, moe_layer_indices=[6, 10], variant="sleep2expert")["experts"] == [True, 0.1]
    # With no MoE layers there is nothing to default to, and nothing trains.
    assert legacy_trainability_table(block, variant="sleep2expert")["experts"] == [False, 0.1]


# The legacy mode each preset is the documented hard-cut conversion of, and the group rename
# that came with the schema (`backbone` became `encoder`).
PRESET_CONVERSIONS = {
    "head_only": "head_only",
    "moe_conservative_routers": "conservative_full_router_trainable",
    "moe_top_experts": "top_moe_layer_expert_only",
}
LEGACY_GROUP_NAMES = {"encoder": "backbone"}


def test_each_preset_carries_the_lr_scales_of_the_mode_it_converts() -> None:
    """The conversion table in the README is only safe if the presets mean what the modes meant.

    `moe_conservative_routers` shipped with `routers: 0.1` while
    `conservative_full_router_trainable` defaulted it to `0.01`, so anyone following the
    manual conversion ran routers at ten times the legacy rate. The one checked-in config on
    that preset carries an explicit `routers` override, which is exactly why the equivalence
    gate could not see the gap -- it replays configs, and no config exercised the default.
    """
    from sleep2expert.config import _FINETUNE_TUNING_PRESETS

    for preset_name, mode in PRESET_CONVERSIONS.items():
        preset = _FINETUNE_TUNING_PRESETS[preset_name]
        legacy = legacy_trainability_table(
            {"moe_tuning": {"mode": mode}}, moe_layer_indices=[6, 10], variant="sleep2expert"
        )
        for group, (trains, lr_scale) in preset.items():
            if group == "lora":
                continue
            legacy_trains, legacy_scale = legacy[LEGACY_GROUP_NAMES.get(group, group)]
            assert trains == legacy_trains, f"{preset_name}.{group} trainability"
            # A legacy 0.0 scale *was* the freeze switch; the schema says `train: false`
            # instead and leaves the scale neutral, so only trained groups compare scales.
            if trains:
                assert lr_scale == legacy_scale, f"{preset_name}.{group} lr_scale"


def test_the_base_variant_inserted_lora_when_the_config_omitted_the_key() -> None:
    """`LoraConfig.insert_lora` defaulted to True on `sleep2vec` and False on the two forks.

    So the same legacy file -- `freeze_backbone_and_insert_lora: true` with no `insert_lora` --
    trained a LoRA group on the base variant and nothing but the head on the forks. One default
    for all three transcribes head-only trainability for a run that adapted the backbone, and a
    hand conversion following it drops the adapters. No checked-in config omitted the key, which
    is why nothing replaying configs could catch this.
    """
    block = {"lora": {"freeze_backbone_and_insert_lora": True}}

    assert legacy_trainability_table(block, variant="sleep2vec")["lora"] == [True, 1.0]
    for fork in ("sleep2vec2", "sleep2expert"):
        assert legacy_trainability_table(block, variant=fork)["lora"] == [False, 1.0], fork

    # An explicit key still wins over the variant's default, in both directions.
    explicit_off = {"lora": {"freeze_backbone_and_insert_lora": True, "insert_lora": False}}
    assert legacy_trainability_table(explicit_off, variant="sleep2vec")["lora"] == [False, 1.0]
    explicit_on = {"lora": {"freeze_backbone_and_insert_lora": True, "insert_lora": True}}
    assert legacy_trainability_table(explicit_on, variant="sleep2expert")["lora"] == [True, 1.0]


def test_the_transcription_covers_every_variant_that_had_the_legacy_schema() -> None:
    """The oracle speaks for the three variants that rejected the legacy keys, and only those.

    A fourth variant growing a `finetune.tuning` schema would need its own legacy default
    recorded here rather than borrowing one silently.
    """
    import importlib

    with_a_legacy_schema = {
        variant
        for variant in ("sleep2vec", "sleep2vec2", "sleep2expert", "sex_age_baseline")
        if hasattr(importlib.import_module(f"{variant}.config"), "_LEGACY_FINETUNE_TRAINABILITY_KEYS")
    }
    assert set(LEGACY_INSERT_LORA_DEFAULTS) == with_a_legacy_schema

    import pytest

    with pytest.raises(ValueError, match="No legacy transcription"):
        legacy_trainability_table({}, variant="sex_age_baseline")
