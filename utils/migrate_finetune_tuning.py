#!/usr/bin/env python3
"""Rewrite `finetune` YAML blocks from the legacy trainability keys to `finetune.tuning`.

The legacy schema spread one decision -- which parameters receive gradient -- across
four keys with different shapes and an order-dependent interaction:

    finetune.freeze_tokenizer
    finetune.lora.freeze_backbone_and_insert_lora
    finetune.lora.insert_lora
    finetune.moe_tuning.mode / .freeze_router / .freeze_experts
    finetune.moe_tuning.lr_scales.<group>   # doubles as a freeze switch at 0.0

This tool does not translate keys by name. It evaluates the *legacy runtime semantics*
(`legacy_trainability_table`, a transcription of the apply sites in
`*/sleep2vec_finetuning.py`), compares the resulting per-group table against the new
presets, and emits the preset that reproduces it -- with an explicit `groups` override
for any group the preset does not match, or `preset: custom` when nothing is close.

The in-tree configs were migrated once, and the legacy table each one was computed
from is frozen in ``tests/config/finetune_tuning_migration.json``, which
``tests/config/test_finetune_tuning_equivalence.py`` replays against the new parser --
so the claim "this migration preserved behaviour" is checked rather than asserted. That
manifest is test data now; nothing regenerates it.

What is left is for configs the repo does not own. The conversion is printed, never
written: it is a semantic rewrite (the preset is derived from legacy runtime behaviour,
not from key names), so it wants a human read before it lands.

Usage:
    python -m utils.migrate_finetune_tuning --config path/to/config.yaml > migrated.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import typing as t

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------------------
# Legacy semantics
# --------------------------------------------------------------------------------------

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


def legacy_trainability_table(
    finetune_block: dict[str, t.Any],
    moe_layer_indices: t.Sequence[int] = (),
) -> dict[str, list[t.Any]]:
    """Return {group: [train, lr_scale]} under the *legacy* runtime semantics.

    Transcribed from `_set_param_trainability_from_policy` (sleep2expert) and from the
    `freeze_backbone_and_insert_lora` -> `freeze_tokenizer` sequence that the two
    non-MoE variants run in `Sleep2vecFinetuning.__init__`. `moe_layer_indices` comes
    from `model.backbone.moe`, because one legacy mode defaulted to it.
    """
    lora_block = finetune_block.get("lora") or {}
    freeze_backbone = bool(lora_block.get("freeze_backbone_and_insert_lora", False))
    insert_lora = bool(lora_block.get("insert_lora", False))
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


# --------------------------------------------------------------------------------------
# Variant resolution
# --------------------------------------------------------------------------------------

NON_MOE_GROUPS = ("head", "encoder", "tokenizers", "projection", "lora")


def _preset_tables() -> dict[str, dict[str, tuple[bool, float]] | None]:
    from sleep2expert.config import _FINETUNE_TUNING_PRESETS

    return _FINETUNE_TUNING_PRESETS


def legal_groups_for(config_data: dict[str, t.Any], config_module: str) -> tuple[str, ...]:
    from sleep2expert.config import FINETUNE_TUNING_GROUPS, FINETUNE_TUNING_MOE_GROUPS

    if config_module != "sleep2expert.config":
        return NON_MOE_GROUPS
    backbone = (config_data.get("model") or {}).get("backbone") or {}
    moe = backbone.get("moe") or {}
    if not moe.get("enabled", False):
        return tuple(group for group in FINETUNE_TUNING_GROUPS if group not in FINETUNE_TUNING_MOE_GROUPS)
    return FINETUNE_TUNING_GROUPS


def config_module_for(path: Path, config_data: dict[str, t.Any]) -> str:
    """The config module that owns this file, e.g. ``sleep2vec2.config``.

    The variants are enforced forks with no cross-imports, so the manifest records the
    module by name rather than a base/expert flag: replaying a `configs/sleep2vec2/**`
    entry through `sleep2vec.config` would check a parser that never loads that file.

    `check_configs` resolves the variant from the migrated shape and knows nothing about
    `finetune.moe_tuning`, which only ever existed in sleep2expert. This tool is the last
    reader of that block, so the legacy marker is checked here rather than teaching a dead
    schema to a module that has no other use for it. It matters for a config passed by
    path, where the directory layout says nothing about the variant.
    """
    from utils.check_configs import CONFIG_VARIANTS, _resolve_config_variant

    finetune_block = config_data.get("finetune")
    if isinstance(finetune_block, dict) and "moe_tuning" in finetune_block:
        return CONFIG_VARIANTS["sleep2expert"].config_module
    return _resolve_config_variant(path, config_data).config_module


# --------------------------------------------------------------------------------------
# Preset selection
# --------------------------------------------------------------------------------------


def select_preset(
    target: dict[str, list[t.Any]],
    legal_groups: t.Sequence[str],
) -> tuple[str, dict[str, list[t.Any]]]:
    """Pick the preset needing the fewest overrides; return (preset, overrides)."""
    best: tuple[int, int, str, dict[str, list[t.Any]]] | None = None
    for preset, table in _preset_tables().items():
        if table is None:  # `custom` carries no table
            continue
        overrides: dict[str, list[t.Any]] = {}
        frozen_mismatches = 0
        for group in legal_groups:
            want_train, want_scale = target[group]
            has_train, has_scale = table[group]
            if bool(want_train) != bool(has_train):
                frozen_mismatches += 1
            elif not want_train or want_scale == has_scale:
                continue
            overrides[group] = [want_train, want_scale]
        # A preset that already freezes the right groups reads truer than one that
        # happens to need the same number of edits but disagrees about what trains.
        candidate = (frozen_mismatches, len(overrides), preset, overrides)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    assert best is not None
    _, _, preset, overrides = best
    # Nothing matched closely enough to still read as that preset.
    if len(overrides) > 2:
        return "custom", {group: target[group] for group in legal_groups}
    return preset, overrides


def render_tuning_block(
    preset: str,
    overrides: dict[str, list[t.Any]],
    legal_groups: t.Sequence[str],
    lora_shape: dict[str, t.Any] | None,
    moe_layer_indices: list[int] | None,
    indent: str = "  ",
) -> list[str]:
    lines = [f"{indent}tuning:", f"{indent}  preset: {preset}"]
    if overrides:
        lines.append(f"{indent}  groups:")
        for group in legal_groups:
            if group not in overrides:
                continue
            train, scale = overrides[group]
            if train:
                lines.append(f"{indent}    {group}: {{train: true, lr_scale: {scale}}}")
            else:
                lines.append(f"{indent}    {group}: {{train: false}}")
    if lora_shape:
        lines.append(f"{indent}  lora:")
        for key in ("r", "alpha", "dropout", "target_modules", "use_dora", "separate_adapters"):
            if key not in lora_shape:
                continue
            lines.append(f"{indent}    {key}: {_scalar(lora_shape[key])}")
    if moe_layer_indices is not None:
        lines.append(f"{indent}  moe:")
        lines.append(f"{indent}    layer_indices: {_scalar(moe_layer_indices)}")
    return lines


def _scalar(value: t.Any) -> str:
    return yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip().removesuffix("...").strip()


# --------------------------------------------------------------------------------------
# Surgical YAML editing
# --------------------------------------------------------------------------------------
#
# The configs carry explanatory comments ("# 1-based transformer block indices"), and
# round-tripping through yaml.safe_dump would delete every one of them plus reflow the
# whole file. So the rewrite is textual: locate the `finetune:` block, splice out the
# legacy keys, and splice in the new ones at the same position.


def _block_span(lines: list[str], start: int, indent: int) -> int:
    """Return the exclusive end of the block whose header is at `start`."""
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if not line.strip() or line.lstrip().startswith("#"):
            end += 1
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        end += 1
    # Trailing blank/comment lines belong to whatever comes next, not to this block.
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return end


def _child_spans(lines: list[str], start: int, end: int, indent: int) -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    cursor = start
    while cursor < end:
        line = lines[cursor]
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or (len(line) - len(line.lstrip())) != indent:
            cursor += 1
            continue
        key = stripped.split(":", 1)[0].strip()
        span_end = min(_block_span(lines, cursor, indent), end)
        spans[key] = (cursor, span_end)
        cursor = span_end
    return spans


def _dedent(lines: list[str], amount: int) -> list[str]:
    return [line[amount:] if line.startswith(" " * amount) else line for line in lines]


def migrate_text(text: str, path: Path) -> tuple[str | None, dict[str, t.Any] | None]:
    """Return (new_text, manifest_entry), or (None, None) when there is nothing to do."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return None, None
    finetune_block = data.get("finetune")
    if not isinstance(finetune_block, dict):
        return None, None
    if "tuning" in finetune_block:
        return None, None  # already migrated; the rewrite is idempotent
    from utils.check_configs import _is_sex_age_baseline_config

    if _is_sex_age_baseline_config(path, data):
        return None, None  # a different runtime entirely; it has no trainability groups
    # A config with none of the legacy keys still needs a tuning block, because
    # finetune.tuning is required now. Its legacy defaults trained everything.

    config_module = config_module_for(path, data)
    legal = legal_groups_for(data, config_module)
    model_moe = ((data.get("model") or {}).get("backbone") or {}).get("moe") or {}
    model_moe_layers = list(model_moe.get("layer_indices") or [])
    legacy_table = legacy_trainability_table(finetune_block, model_moe_layers)
    target = to_new_groups(legacy_table, legal)
    preset, overrides = select_preset(target, legal)

    lora_block = finetune_block.get("lora") or {}
    lora_shape = None
    if target.get("lora", [False])[0]:
        lora_shape = {
            key: lora_block[key]
            for key in ("r", "alpha", "dropout", "target_modules", "use_dora", "separate_adapters")
            if key in lora_block
        }
    moe_tuning = finetune_block.get("moe_tuning") or {}
    moe_layer_indices = None
    if preset == "moe_top_experts":
        moe_layer_indices = legacy_selected_moe_layers(moe_tuning, model_moe_layers)

    lines = text.splitlines()
    finetune_start = next(index for index, line in enumerate(lines) if line.startswith("finetune:"))
    finetune_end = _block_span(lines, finetune_start, 0)
    spans = _child_spans(lines, finetune_start + 1, finetune_end, 2)

    # moe_regularization is an auxiliary-loss setting, not a trainability setting. It
    # was only ever nested under moe_tuning; lift it to a sibling of `tuning`.
    regularization_lines: list[str] = []
    if "moe_tuning" in spans:
        mt_start, mt_end = spans["moe_tuning"]
        inner = _child_spans(lines, mt_start + 1, mt_end, 4)
        if "moe_regularization" in inner:
            reg_start, reg_end = inner["moe_regularization"]
            regularization_lines = _dedent(lines[reg_start:reg_end], 2)
        elif moe_tuning.get("moe_regularization") is not None:
            # The lift is textual so it keeps the block's comments, and a flow-style
            # `moe_tuning: {...}` puts moe_regularization on a line this splice deletes
            # wholesale. Refuse rather than silently disable an auxiliary loss.
            raise ValueError(
                "finetune.moe_tuning carries moe_regularization on an inline mapping; "
                "rewrite it as a block mapping before migrating."
            )

    legacy_spans = [spans[key] for key in ("freeze_tokenizer", "lora", "moe_tuning") if key in spans]
    doomed = sorted(legacy_spans)
    # With no legacy keys to replace in place, the tuning block goes first inside `finetune:`.
    insert_at = doomed[0][0] if doomed else finetune_start + 1
    new_lines = render_tuning_block(preset, overrides, legal, lora_shape, moe_layer_indices)
    new_lines.extend(regularization_lines)

    keep: list[str] = []
    for index, line in enumerate(lines):
        if index == insert_at:
            keep.extend(new_lines)
        if any(start <= index < end for start, end in doomed):
            continue
        keep.append(line)

    new_text = "\n".join(keep) + "\n"
    _assert_converted(new_text, path)
    entry = {
        "path": _display_path(path),
        "config_module": config_module,
        "legacy": legacy_table,
        "groups": legal,
        "expected": target,
        "preset": preset,
    }
    return new_text, entry


def _assert_converted(new_text: str, path: Path) -> None:
    """Check the splice landed instead of trusting that it did.

    The rewrite is line-based so it keeps the file's comments, which means `_child_spans`
    has to recognise a block mapping. Given a flow-style `finetune: {...}` it finds no
    children and the insertion point lands after a mapping that is already closed: the
    block is dropped when that line was last, and produces unparsable YAML when it was not.
    Both exited zero and printed something that read as converted, which is worse than
    either failing -- so the tool asserts the postcondition it just claimed.
    """
    try:
        data = yaml.safe_load(new_text)
    except yaml.YAMLError as error:
        raise ValueError(
            "the converted YAML does not parse. A flow-style `finetune: {...}` block is the "
            "usual cause; rewrite it as a block mapping before migrating."
        ) from error
    finetune_block = data.get("finetune") if isinstance(data, dict) else None
    if not isinstance(finetune_block, dict) or "tuning" not in finetune_block:
        raise ValueError(
            "the converted config has no finetune.tuning block. A flow-style `finetune: {...}` "
            "block is the usual cause; rewrite it as a block mapping before migrating."
        )
    survivors = sorted(key for key in ("freeze_tokenizer", "lora", "moe_tuning") if key in finetune_block)
    if survivors:
        raise ValueError(
            f"the converted config still carries finetune.{{{','.join(survivors)}}}, which every "
            "variant loader rejects."
        )


def _display_path(path: Path) -> str:
    """Repo-relative when the file is in the tree, verbatim otherwise."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def convert_one(path: Path) -> int:
    """Print the converted YAML to stdout and what it concluded to stderr."""
    try:
        text = path.read_text()
    except OSError as error:
        print(f"{path}: {error}", file=sys.stderr)
        return 2
    try:
        new_text, entry = migrate_text(text, path)
    except Exception as error:  # noqa: BLE001 - surface the offending file
        print(f"{path}: {error}", file=sys.stderr)
        return 1
    if new_text is None or entry is None:
        print(f"{path}: nothing to migrate (no finetune block, or already on finetune.tuning).", file=sys.stderr)
        return 1
    # The preset is derived from legacy runtime semantics, so the diff alone does not show
    # why it was chosen. Report the table it was matched against.
    print(f"{path}: {entry['config_module']} -> preset '{entry['preset']}'", file=sys.stderr)
    for group, (trains, scale) in sorted(entry["expected"].items()):
        print(f"  {group}: train={trains} lr_scale={scale}", file=sys.stderr)
    sys.stdout.write(new_text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="The config to convert.")
    return convert_one(parser.parse_args().config)


if __name__ == "__main__":
    raise SystemExit(main())
