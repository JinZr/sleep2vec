"""Pin the legacy semantics `utils/migrate_finetune_tuning.py` migrates away from.

The equivalence gate in `test_finetune_tuning_equivalence.py` replays a manifest that the
migration tool itself produced, so a misreading of the legacy schema would be baked into
both sides of that comparison and pass. These cases state the legacy behaviour directly,
from the pre-refactor `sleep2expert/config.py`, and they cover the defaults no shipped
config happened to exercise -- exactly where a silent misreading would survive.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.migrate_finetune_tuning import legacy_trainability_table  # noqa: E402


def test_omitted_lr_scales_follow_the_mode_they_were_defaulted_from() -> None:
    """`_default_finetune_moe_lr_scales(mode)` was mode-dependent, and 0.0 froze a group.

    Under `head_only` the backbone and experts defaulted to 0.0. Reading one flat default
    table would migrate this config to a policy that trains a backbone the legacy run held
    frozen -- a silent behaviour change wearing the name of a faithful migration.
    """
    table = legacy_trainability_table({"moe_tuning": {"mode": "head_only"}})

    assert table["head"] == [True, 1.0]
    assert table["backbone"] == [False, 0.0]
    assert table["experts"] == [False, 0.0]

    trainable_routers = legacy_trainability_table({"moe_tuning": {"mode": "conservative_full_router_trainable"}})
    assert trainable_routers["routers"] == [True, 0.01]


def test_explicit_lr_scales_still_win_over_the_mode_default() -> None:
    """The mode only supplies values for the groups a config left out.

    `head_only` still freezes the backbone on its own, so the explicit scale shows up as a
    scale rather than as a thaw -- which is how the two mechanisms differed.
    """
    table = legacy_trainability_table({"moe_tuning": {"mode": "head_only", "lr_scales": {"backbone": 0.1}}})

    assert table["backbone"] == [False, 0.1]

    thawed = legacy_trainability_table({"moe_tuning": {"mode": "custom", "lr_scales": {"backbone": 0.1}}})
    assert thawed["backbone"] == [True, 0.1]


def test_top_moe_layer_expert_only_defaults_to_the_deepest_moe_layer() -> None:
    """Validation filled `train_moe_layer_indices` when the config left it out.

    Such a config names no layer in its own text, so reading the key alone would migrate it
    to a policy that trains no experts at all.
    """
    block = {"moe_tuning": {"mode": "top_moe_layer_expert_only", "lr_scales": {"backbone": 0.0}}}

    assert legacy_trainability_table(block, moe_layer_indices=[6, 10])["experts"] == [True, 0.1]
    # With no MoE layers there is nothing to default to, and nothing trains.
    assert legacy_trainability_table(block)["experts"] == [False, 0.1]
