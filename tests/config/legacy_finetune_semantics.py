"""The legacy finetune trainability semantics, transcribed once and frozen.

Two tests need to know what the pre-`finetune.tuning` schema *did* at runtime:
`test_legacy_finetune_semantics.py` states it directly, and
`test_finetune_tuning_equivalence.py` maps the frozen legacy tables in
`finetune_tuning_migration.json` onto the new group names before replaying them
through the current parser.

This is a transcription of code that no longer exists -- `_set_param_trainability_from_policy`
in the pre-refactor `sleep2expert`, and the `freeze_backbone_and_insert_lora` ->
`freeze_tokenizer` sequence the two non-MoE variants ran in `Sleep2vecFinetuning.__init__`.
Nothing in the repository reads the legacy schema any more, so this file is evidence for
the migration rather than a code path: it should change only if the transcription is found
to be wrong about history.

`doc/finetune_tuning_schema_refactor.md` documents the same mapping in prose, for a reader
converting a config by hand.
"""

from __future__ import annotations

import typing as t

LEGACY_GROUPS = ("head", "backbone", "experts", "routers", "tokenizers", "projection", "lora")
# Mirrors `_default_finetune_moe_lr_scales(mode)`, which filled in every group a config
# omitted from moe_tuning.lr_scales. The defaults were mode-dependent, and the difference
# is not cosmetic: under `head_only` the backbone defaulted to 0.0, and a 0.0 scale was
# itself a freeze switch. Reading one flat table for every mode would migrate such a
# config to a policy that trains the backbone the legacy run kept frozen.
LEGACY_DEFAULT_SCALES = {
    "head": 1.0,
    "backbone": 0.1,
    "experts": 0.1,
    "routers": 0.0,
    "tokenizers": 0.0,
    "projection": 0.0,
    "lora": 1.0,
}
LEGACY_MODE_SCALES = {
    "head_only": {**LEGACY_DEFAULT_SCALES, "backbone": 0.0, "experts": 0.0},
    "conservative_full_router_trainable": {**LEGACY_DEFAULT_SCALES, "routers": 0.01},
    "top_moe_layer_expert_only": {**LEGACY_DEFAULT_SCALES, "backbone": 0.0},
}
# The legacy `backbone` group is the new `encoder` group: same parameters, and the
# semantic classifier already excluded tokenizers/experts/routers/projection from it.
LEGACY_TO_NEW_GROUP = {"backbone": "encoder"}
# `LoraConfig.insert_lora` did not default the same way across the forks -- `sleep2vec` had
# `True`, the two forks `False` -- so an omitted key does not describe one behaviour. A legacy
# base config with `freeze_backbone_and_insert_lora: true` and no `insert_lora` inserted
# adapters; the same file under either fork did not. Reading one default for all three
# transcribes head-only trainability for a run that trained a LoRA group, which is the wrong
# parameter set for anyone converting such a config by hand.
LEGACY_INSERT_LORA_DEFAULTS = {"sleep2vec": True, "sleep2vec2": False, "sleep2expert": False}


def legacy_trainability_table(
    finetune_block: dict[str, t.Any],
    moe_layer_indices: t.Sequence[int] = (),
    *,
    variant: str,
) -> dict[str, list[t.Any]]:
    """Return {group: [train, lr_scale]} under the *legacy* runtime semantics.

    Transcribed from `_set_param_trainability_from_policy` (sleep2expert) and from the
    `freeze_backbone_and_insert_lora` -> `freeze_tokenizer` sequence that the two
    non-MoE variants run in `Sleep2vecFinetuning.__init__`. `moe_layer_indices` comes
    from `model.backbone.moe`, because one legacy mode defaulted to it.

    `variant` is required rather than defaulted: the one key whose legacy default differed
    per fork is `insert_lora`, and a default here would silently pick a fork and describe
    the wrong parameter set -- the failure this argument exists to prevent.
    """
    try:
        insert_lora_default = LEGACY_INSERT_LORA_DEFAULTS[variant]
    except KeyError:
        raise ValueError(
            f"No legacy transcription for {variant!r}. Expected one of {sorted(LEGACY_INSERT_LORA_DEFAULTS)}."
        ) from None
    lora_block = finetune_block.get("lora") or {}
    freeze_backbone = bool(lora_block.get("freeze_backbone_and_insert_lora", False))
    insert_lora = bool(lora_block.get("insert_lora", insert_lora_default))
    # Adapters exist only when the backbone was frozen *and* insertion was requested;
    # `insert_lora: true` alone was inert, which is why 30 configs set it to no effect.
    adapters_inserted = freeze_backbone and insert_lora
    freeze_tokenizer = bool(finetune_block.get("freeze_tokenizer", True))
    moe_tuning = finetune_block.get("moe_tuning")

    if moe_tuning is None:
        # Non-MoE path. Order matters: the backbone freeze ran first and swept the
        # tokenizers with it, then freeze_tokenizer unfroze them back out.
        if freeze_backbone:
            train = {group: False for group in LEGACY_GROUPS}
            train["head"] = True
            train["lora"] = insert_lora
        else:
            train = {group: True for group in LEGACY_GROUPS}
            train["lora"] = False  # never inserted unless the backbone was frozen
        train["tokenizers"] = not freeze_tokenizer
        # The non-MoE variants build a single-LR optimizer: there is no lr_scales key and
        # no per-group scaling, so every trained group runs at the base learning rate.
        return {group: [train[group], 1.0] for group in LEGACY_GROUPS}

    mode = moe_tuning.get("mode", "conservative_full_router_frozen")
    scales = dict(LEGACY_MODE_SCALES.get(mode, LEGACY_DEFAULT_SCALES))
    scales.update({key: float(value) for key, value in (moe_tuning.get("lr_scales") or {}).items()})
    selected_layers = set(legacy_selected_moe_layers(moe_tuning, moe_layer_indices))

    table: dict[str, list[t.Any]] = {}
    for group in LEGACY_GROUPS:
        if group == "lora":
            trainable = scales[group] > 0.0
        elif mode == "head_only":
            trainable = group == "head"
        elif mode == "conservative_full_router_frozen":
            trainable = group in {"head", "backbone", "experts"}
        elif mode == "conservative_full_router_trainable":
            trainable = group in {"head", "backbone", "experts", "routers"}
        elif mode == "top_moe_layer_expert_only":
            trainable = group == "head" or (group == "experts" and bool(selected_layers))
        elif mode == "custom":
            trainable = scales[group] > 0.0
            if group == "routers" and moe_tuning.get("freeze_router"):
                trainable = False
            if group == "experts" and moe_tuning.get("freeze_experts"):
                trainable = False
        else:
            raise ValueError(f"{mode!r} is not a legacy moe_tuning.mode")
        if group == "tokenizers" and freeze_tokenizer:
            trainable = False
        if scales[group] == 0.0:
            trainable = False
        table[group] = [trainable, scales[group]]

    if not adapters_inserted:
        table["lora"][0] = False
    return table


def legacy_selected_moe_layers(
    moe_tuning: dict[str, t.Any],
    moe_layer_indices: t.Sequence[int] = (),
) -> list[int]:
    """Which MoE layers the legacy run trained experts in.

    `top_moe_layer_expert_only` did not require `train_moe_layer_indices`: validation
    filled it with the deepest MoE layer. A config relying on that default names no
    layers in its own text, so reading the key alone would migrate it to a policy that
    trains no experts at all.
    """
    explicit = moe_tuning.get("train_moe_layer_indices")
    if explicit:
        return list(explicit)
    if moe_tuning.get("mode") == "top_moe_layer_expert_only" and moe_layer_indices:
        return [max(moe_layer_indices)]
    return []


def to_new_groups(legacy_table: dict[str, list[t.Any]], legal_groups: t.Sequence[str]) -> dict[str, list[t.Any]]:
    renamed = {LEGACY_TO_NEW_GROUP.get(group, group): value for group, value in legacy_table.items()}
    return {group: renamed[group] for group in legal_groups if group in renamed}
