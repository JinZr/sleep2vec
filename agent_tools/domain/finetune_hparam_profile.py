from __future__ import annotations

from itertools import combinations, product
import json
import math
from typing import Any

from ..decision_models import DecisionIssue, DecisionStatus
from ..plan_rendering import DEFAULT_FINETUNE_LR, DEFAULT_FINETUNE_WEIGHT_DECAY

PROFILE_ID = "finetune_balanced"
DEFAULT_MAX_RUNS = 12
MAX_RUNS = 32
_SUPPORTED_VARIANTS = {"sleep2vec", "sleep2vec2"}
_SUPPORTED_LABELS = {"ahi", "arousal", "stage4"}


def compile_finetune_balanced_profile(
    recipe: dict[str, Any],
    config_summary: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[DecisionIssue]]:
    search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}
    if search.get("profile") != PROFILE_ID:
        return None, [
            _issue(
                DecisionStatus.NEEDS_USER_INPUT,
                "Unsupported automatic hparam search profile.",
                {"profile": search.get("profile"), "supported_profiles": [PROFILE_ID]},
            )
        ]
    authored_spaces = sorted(field for field in ("parameters", "configurations") if field in search)
    if authored_spaces:
        return None, [
            _issue(
                DecisionStatus.FAIL,
                "search.profile is mutually exclusive with authored search.parameters and search.configurations.",
                {"conflicting_fields": authored_spaces},
            )
        ]
    if search.get("method") not in (None, "grid"):
        return None, [
            _issue(
                DecisionStatus.FAIL,
                "finetune_balanced only supports search.method=grid.",
                {"method": search.get("method")},
            )
        ]
    adaptive = recipe.get("adaptive") if isinstance(recipe.get("adaptive"), dict) else {}
    if adaptive.get("enabled") is True:
        return None, [
            _issue(
                DecisionStatus.NEEDS_USER_INPUT,
                "finetune_balanced does not support adaptive.enabled=true; staged adaptive search is deferred.",
                {"adaptive_enabled": True},
            )
        ]
    variant = recipe.get("variant")
    label = (recipe.get("inputs") or {}).get("label_name") if isinstance(recipe.get("inputs"), dict) else None
    if variant not in _SUPPORTED_VARIANTS or label not in _SUPPORTED_LABELS:
        return None, [
            _issue(
                DecisionStatus.NEEDS_USER_INPUT,
                "No unique finetune_balanced profile is registered for this variant and label.",
                {
                    "variant": variant,
                    "label_name": label,
                    "supported_variants": sorted(_SUPPORTED_VARIANTS),
                    "supported_labels": sorted(_SUPPORTED_LABELS),
                },
            )
        ]
    if not isinstance(config_summary, dict):
        return None, [
            _issue(
                DecisionStatus.NEEDS_USER_INPUT,
                "finetune_balanced requires one readable resolved finetune config.",
                {},
            )
        ]

    try:
        axes = _profile_axes(recipe, config_summary)
    except ValueError as exc:
        return None, [_issue(DecisionStatus.FAIL, str(exc), {})]
    minimum_runs = max(4, *(len(axis["levels"]) for axis in axes))
    requested_runs = search.get("max_runs", DEFAULT_MAX_RUNS)
    if type(requested_runs) is not int or not minimum_runs <= requested_runs <= MAX_RUNS:
        return None, [
            _issue(
                DecisionStatus.NEEDS_USER_INPUT,
                "finetune_balanced requires a run budget that covers every profile level and does not exceed 32.",
                {
                    "max_runs": requested_runs,
                    "minimum_runs": minimum_runs,
                    "default_max_runs": DEFAULT_MAX_RUNS,
                    "maximum_runs": MAX_RUNS,
                    "profile_level_counts": {axis["id"]: len(axis["levels"]) for axis in axes},
                },
            )
        ]
    configurations = _balanced_configurations(axes, requested_runs)
    return {
        "profile": PROFILE_ID,
        "method": "grid",
        "max_runs": requested_runs,
        "configurations": configurations,
    }, []


def finetune_balanced_profile_audit(search: dict[str, Any]) -> dict[str, Any]:
    configurations = search.get("configurations") if isinstance(search.get("configurations"), list) else []
    keys = sorted({str(key) for point in configurations if isinstance(point, dict) for key in point})
    family_keys = {
        "optimization.lr": [key for key in keys if key == "runtime.lr"],
        "optimization.weight_decay": [key for key in keys if key == "runtime.weight_decay"],
        "model.layer_mix": [key for key in keys if key == "yaml:/finetune/layer_mix"],
        "regularization.dropout": [
            key
            for key in keys
            if key
            in {
                "yaml:/model/head/dropout",
                "yaml:/model/head/kwargs/attn_dropout",
                "yaml:/model/head/kwargs/temporal_dropout",
            }
        ],
        "adaptation.strategy": [key for key in keys if key == "yaml:/finetune/lora"],
        "loss.pos_weight": [key for key in keys if key == "yaml:/finetune/loss/pos_weight"],
    }
    families = []
    for family_id, selected_keys in family_keys.items():
        if not selected_keys:
            continue
        levels = {
            json.dumps(
                {key: point[key] for key in selected_keys},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for point in configurations
        }
        if len(levels) < 2:
            continue
        families.append({"id": family_id, "keys": selected_keys, "covered_levels": len(levels)})
    return {
        "id": PROFILE_ID,
        "budget": search.get("max_runs"),
        "candidate_count": len(configurations),
        "searched_families": families,
    }


def _profile_axes(recipe: dict[str, Any], config_summary: dict[str, Any]) -> list[dict[str, Any]]:
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    model = config_summary.get("model") if isinstance(config_summary.get("model"), dict) else {}
    finetune = config_summary.get("finetune") if isinstance(config_summary.get("finetune"), dict) else {}

    lr = _finite_number(runtime.get("lr", DEFAULT_FINETUNE_LR), "runtime.lr", positive=True)
    weight_decay = _finite_number(
        runtime.get("weight_decay", DEFAULT_FINETUNE_WEIGHT_DECAY),
        "runtime.weight_decay",
        non_negative=True,
    )
    weight_decay_levels = (
        [0.0, 1.0e-5, 1.0e-4]
        if weight_decay == 0
        else _baseline_first(weight_decay, [0.0, weight_decay, weight_decay * 10.0])
    )
    axes = [
        _axis("optimization.lr", "runtime.lr", _baseline_first(lr, [lr / 3.0, lr, lr * 3.0])),
        _axis(
            "optimization.weight_decay",
            "runtime.weight_decay",
            weight_decay_levels,
        ),
    ]

    depth = model.get("backbone_depth")
    if type(depth) is not int or depth < 1:
        raise ValueError("finetune_balanced requires model.backbone.num_hidden_layers to be a positive integer.")
    layer_mix = model.get("layer_mix")
    if not model.get("layer_mix_present") or not isinstance(layer_mix, dict):
        raise ValueError("finetune_balanced requires a complete finetune.layer_mix mapping.")
    for field in ("enabled", "shared_across_modalities", "layer_indices"):
        if field not in layer_mix:
            raise ValueError(f"finetune_balanced requires finetune.layer_mix.{field} in the source config.")
    if type(layer_mix["enabled"]) is not bool or type(layer_mix["shared_across_modalities"]) is not bool:
        raise ValueError("finetune_balanced requires boolean LayerMix enabled/shared_across_modalities values.")
    if not layer_mix["enabled"] and (layer_mix["layer_indices"] is not None or layer_mix["shared_across_modalities"]):
        raise ValueError(
            "finetune_balanced requires disabled source LayerMix to use layer_indices=null and "
            "shared_across_modalities=false."
        )
    channel_count = len(model.get("channels") or [])
    if channel_count <= 1 and layer_mix["shared_across_modalities"]:
        raise ValueError("finetune_balanced requires single-channel source LayerMix to disable modality sharing.")
    layer_levels = [
        _canonical(layer_mix),
        _canonical(
            {
                **layer_mix,
                "enabled": False,
                "shared_across_modalities": False,
                "layer_indices": None,
            }
        ),
    ]
    shared_levels = (False, True) if channel_count > 1 else (False,)
    for indices in (
        list(range(max(1, depth - 1), depth + 1)),
        list(range(max(1, depth - 3), depth + 1)),
        _even_layers(depth),
    ):
        for shared in shared_levels:
            layer_levels.append(
                _canonical(
                    {
                        **layer_mix,
                        "enabled": True,
                        "shared_across_modalities": shared,
                        "layer_indices": indices,
                    }
                )
            )
    axes.append(_axis("model.layer_mix", "yaml:/finetune/layer_mix", _stable_unique(layer_levels)))

    head = model.get("head_details") if isinstance(model.get("head_details"), dict) else {}
    head_dropout = _finite_dropout(head.get("dropout"), "model.head.dropout")
    head_kwargs = head.get("kwargs") if isinstance(head.get("kwargs"), dict) else {}
    dropout_keys = ["yaml:/model/head/dropout"]
    source_dropout = {"yaml:/model/head/dropout": head_dropout}
    for field in ("attn_dropout", "temporal_dropout"):
        if field in head_kwargs:
            key = f"yaml:/model/head/kwargs/{field}"
            dropout_keys.append(key)
            source_dropout[key] = _finite_dropout(head_kwargs[field], f"model.head.kwargs.{field}")
    synchronized = [source_dropout]
    for value in _baseline_first(head_dropout, [0.0, head_dropout, min(0.5, head_dropout + 0.1)]):
        synchronized.append({key: value for key in dropout_keys})
    axes.append({"id": "regularization.dropout", "levels": _stable_unique(synchronized)})

    lora = finetune.get("lora")
    if not finetune.get("lora_present") or not isinstance(lora, dict):
        raise ValueError("finetune_balanced requires an explicit finetune.lora control mapping.")
    freeze = lora.get("freeze_backbone_and_insert_lora")
    insert = lora.get("insert_lora")
    if type(freeze) is not bool or type(insert) is not bool:
        raise ValueError("finetune_balanced requires explicit boolean LoRA freeze and insert values.")
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    has_trained_backbone = inputs.get("pretrained_backbone_path") not in (None, "")
    if freeze and not has_trained_backbone:
        raise ValueError("finetune_balanced cannot freeze a source backbone without a pretrained backbone.")
    lora_levels = [lora]
    if has_trained_backbone:
        head_only = {**lora, "freeze_backbone_and_insert_lora": True, "insert_lora": False}
        with_lora = {**lora, "freeze_backbone_and_insert_lora": True, "insert_lora": True}
        if freeze:
            lora_levels.append({**lora, "freeze_backbone_and_insert_lora": False, "insert_lora": False})
        lora_levels.extend((head_only, with_lora))
    lora_levels = _stable_unique(lora_levels)
    axes.append(_axis("adaptation.strategy", "yaml:/finetune/lora", lora_levels))

    loss = finetune.get("loss") if isinstance(finetune.get("loss"), dict) else {}
    pos_weight = loss.get("pos_weight")
    if _is_finite_number(pos_weight) and float(pos_weight) > 0:
        value = float(pos_weight)
        axes.append(
            _axis(
                "loss.pos_weight",
                "yaml:/finetune/loss/pos_weight",
                _baseline_first(value, [value * 0.5, value, value * 2.0]),
            )
        )
    return axes


def _balanced_configurations(axes: list[dict[str, Any]], max_runs: int) -> list[dict[str, Any]]:
    pool = list(product(*(range(len(axis["levels"])) for axis in axes)))
    selected = [pool[0]]
    covered_levels: set[tuple[int, int]] = set()
    covered_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    _record_coverage(selected[0], covered_levels, covered_pairs)
    remaining = pool[1:]
    while remaining and len(selected) < max_runs:
        best_position = 0
        best_score = (-1, -1)
        for position, candidate in enumerate(remaining):
            levels, pairs = _candidate_coverage(candidate)
            score = (len(levels - covered_levels), len(pairs - covered_pairs))
            if score > best_score:
                best_position = position
                best_score = score
        chosen = remaining.pop(best_position)
        selected.append(chosen)
        _record_coverage(chosen, covered_levels, covered_pairs)

    configurations = []
    for indexes in selected:
        point: dict[str, Any] = {}
        for axis, level_index in zip(axes, indexes):
            point.update(axis["levels"][level_index])
        configurations.append(_canonical(point))
    return configurations


def _candidate_coverage(
    indexes: tuple[int, ...],
) -> tuple[set[tuple[int, int]], set[tuple[tuple[int, int], tuple[int, int]]]]:
    levels = {(axis_index, level_index) for axis_index, level_index in enumerate(indexes)}
    return levels, {tuple(pair) for pair in combinations(sorted(levels), 2)}


def _record_coverage(
    indexes: tuple[int, ...],
    levels: set[tuple[int, int]],
    pairs: set[tuple[tuple[int, int], tuple[int, int]]],
) -> None:
    candidate_levels, candidate_pairs = _candidate_coverage(indexes)
    levels.update(candidate_levels)
    pairs.update(candidate_pairs)


def _axis(axis_id: str, key: str, values: list[Any]) -> dict[str, Any]:
    return {"id": axis_id, "levels": [{key: _canonical(value)} for value in values]}


def _baseline_first(baseline: Any, candidates: list[Any]) -> list[Any]:
    return _stable_unique([baseline, *candidates])


def _stable_unique(values: list[Any]) -> list[Any]:
    unique = []
    seen = set()
    for value in values:
        canonical = json.dumps(
            _canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if canonical not in seen:
            seen.add(canonical)
            unique.append(_canonical(value))
    return unique


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _even_layers(depth: int) -> list[int]:
    if depth == 1:
        return [1]
    return _stable_unique([math.floor(1 + index * (depth - 1) / 3 + 0.5) for index in range(4)])


def _finite_number(value: Any, field: str, *, positive: bool = False, non_negative: bool = False) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"finetune_balanced requires finite numeric {field}.")
    number = float(value)
    if positive and number <= 0:
        raise ValueError(f"finetune_balanced requires {field} > 0.")
    if non_negative and number < 0:
        raise ValueError(f"finetune_balanced requires {field} >= 0.")
    return number


def _finite_dropout(value: Any, field: str) -> float:
    number = _finite_number(value, field, non_negative=True)
    if number > 0.5:
        raise ValueError(f"finetune_balanced requires {field} <= 0.5.")
    return number


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _issue(status: DecisionStatus, message: str, evidence: dict[str, Any]) -> DecisionIssue:
    return DecisionIssue(
        status,
        "hparam_search_profile",
        message,
        (
            "Which supported automatic search profile and run budget should this tuning task use?"
            if status == DecisionStatus.NEEDS_USER_INPUT
            else None
        ),
        {**evidence, "preflight_before_workspace": True},
    )
