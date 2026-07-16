"""Single source of truth for decision-field <-> recipe/config path mappings.

Layer 0 leaf: imports only stdlib and ``models``; kernel modules (decisions,
plans) consume these tables so the read side (``_recipe_field_value``) and the
write side (``canonical_fields``) share one mapping instead of mirroring each
other. Per-task write-target overrides still live on the adapters as
``decision_recipe_targets``; this module holds the cross-task base tables and
the merge helper.

Note the deliberate read/write asymmetry: the read path is fully task-agnostic
(every task resolves the same fallback chain), while write targets are merged
per task. ``merged_write_targets`` therefore folds adapter overrides in, but
the read tables never do.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import CONFIG_FINETUNE_SECTION


@dataclass(frozen=True)
class RecipeField:
    #: Ordered (section, key) lookups; the first section whose mapping contains
    #: the key wins (mirrors ``.get(key, <fallback>)`` chains, so a present but
    #: falsy value like ``overwrite=False`` still counts as a hit).
    read_path: tuple[tuple[str, str], ...]
    #: Single materialization target, or None for read-only fields (the base
    #: table has no write target; an adapter may inject one).
    write_target: tuple[str, str] | None = None
    #: Read side maps a present ``None`` to _MISSING (pretrained_backbone_path).
    none_is_missing: bool = False
    #: Write side only materializes when the value is in this set (None = always).
    write_value_whitelist: frozenset[str] | None = None


# Cross-task base table. Mirrors decisions._recipe_field_value (read) and
# plans._materialize_decisions' canonical_fields (write). Fields whose write
# target is None are written only when an adapter declares a target via
# decision_recipe_targets (required_channels, hparam_*).
BASE_RECIPE_FIELDS: dict[str, RecipeField] = {
    "task": RecipeField(read_path=()),  # read from recipe root; special-cased
    "label_name": RecipeField((("inputs", "label_name"),), ("inputs", "label_name")),
    "data_backend": RecipeField((("inputs", "data_backend"),), ("inputs", "data_backend")),
    "train_val_test_policy": RecipeField(
        (("evaluation_policy", "selection_split"),),
        ("evaluation_policy", "selection_split"),
        write_value_whitelist=frozenset({"train", "val", "test"}),
    ),
    "external_test_locked": RecipeField(
        (("evaluation_policy", "external_test_locked"),),
        ("evaluation_policy", "external_test_locked"),
    ),
    "selection_metric": RecipeField(
        (("evaluation_policy", "selection_metric"),),
        ("evaluation_policy", "selection_metric"),
    ),
    "selection_mode": RecipeField(
        (("evaluation_policy", "selection_mode"),),
        ("evaluation_policy", "selection_mode"),
    ),
    "pretrained_backbone_path": RecipeField(
        (("inputs", "pretrained_backbone_path"),),
        ("inputs", "pretrained_backbone_path"),
        none_is_missing=True,
    ),
    "config": RecipeField((("inputs", "config"),), ("inputs", "config")),
    "ckpt_path": RecipeField((("inputs", "ckpt_path"),), ("inputs", "ckpt_path")),
    "eval_split": RecipeField((("inputs", "eval_split"),), ("inputs", "eval_split")),
    "final_eval_config_path": RecipeField(
        (("inputs", "final_eval_config_path"),),
        ("inputs", "final_eval_config_path"),
    ),
    "overwrite_policy": RecipeField(
        (("artifacts", "overwrite"), ("preset", "overwrite")),
        ("artifacts", "overwrite"),
    ),
    "required_channels": RecipeField(
        (("preset", "required_channels"), ("preset", "channels")),
    ),
    "min_channels": RecipeField((("preset", "min_channels"),), ("preset", "min_channels")),
    "hparam_search_space": RecipeField((("search", "parameters"),)),
    "hparam_budget": RecipeField((("search", "max_runs"),)),
    "final_eval_unlock": RecipeField(
        (("evaluation_policy", "final_test_unlocked"),),
        ("evaluation_policy", "final_test_unlocked"),
    ),
    "test_after_fit": RecipeField(
        (("evaluation_policy", "test_after_fit"),),
        ("evaluation_policy", "test_after_fit"),
    ),
}


@dataclass(frozen=True)
class ConfigField:
    #: Nested lookup into config_summary (each hop assumes a mapping).
    summary_path: tuple[str, ...]
    #: Human-readable config path for contract mismatch messages.
    display_path: str


# Config-summary path knowledge, shared by decisions._config_field_value (read)
# and plans' config_contracts (decision-vs-config consistency check).
CONFIG_FIELDS: dict[str, ConfigField] = {
    "data_backend": ConfigField(("data_backend",), "data.backend"),
    "selection_metric": ConfigField((CONFIG_FINETUNE_SECTION, "task", "monitor"), "finetune.task.monitor"),
    "selection_mode": ConfigField((CONFIG_FINETUNE_SECTION, "task", "monitor_mod"), "finetune.task.monitor_mod"),
    "required_channels": ConfigField(("preset_build", "required_channels"), "preset_build.required_channels"),
    "min_channels": ConfigField(("preset_build", "min_channels"), "preset_build.min_channels"),
}


def merged_write_targets(adapter_targets: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
    """Base write targets folded together with an adapter's per-task overrides.

    Equivalent to plans._materialize_decisions' ``canonical_fields`` base dict
    plus ``canonical_fields.update(adapter.decision_recipe_targets)``.
    """
    targets = {
        name: spec.write_target for name, spec in BASE_RECIPE_FIELDS.items() if spec.write_target is not None
    }
    targets.update(adapter_targets)
    return targets
