"""The deterministic ``finetune_balanced`` candidate compiler and its audit view.

Layer 0 domain leaf, invoked through the hparam adapter's effective-recipe
binding hook before consultation, so generic decision code never duplicates
sleep2vec config fields.

Requires a complete explicit LayerMix block and an explicit ``finetune.tuning``
preset, preserves the exact source config as candidate zero, and expands only
its registered technical axes. Omitted ``finetune.tuning.lora``
hyper-parameters retain the canonical variant loader defaults frozen by config
and runtime identity.
"""

from __future__ import annotations

from itertools import combinations, product
import json
import math
from typing import Any, TypeGuard

from ..decision_models import DecisionIssue, DecisionStatus
from ..models import ConfigSummaryInput
from ..plan_rendering import DEFAULT_FINETUNE_LR, DEFAULT_FINETUNE_WEIGHT_DECAY

PROFILE_ID = "finetune_balanced"
DEFAULT_MAX_RUNS = 12
MAX_RUNS = 32
_SUPPORTED_VARIANTS = {"sleep2vec", "sleep2vec2"}
_SUPPORTED_LABELS = {"ahi", "arousal", "stage4", "age", "sex"}


def compile_finetune_balanced_profile(
    recipe: dict[str, Any],
    config_summary: ConfigSummaryInput | None,
) -> tuple[dict[str, Any] | None, list[DecisionIssue]]:
    search_value = recipe.get("search")
    search = search_value if isinstance(search_value, dict) else {}
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
    adaptive_value = recipe.get("adaptive")
    adaptive = adaptive_value if isinstance(adaptive_value, dict) else {}
    if adaptive.get("enabled") is True:
        return None, [
            _issue(
                DecisionStatus.NEEDS_USER_INPUT,
                "finetune_balanced does not support adaptive.enabled=true; staged adaptive search is deferred.",
                {"adaptive_enabled": True},
            )
        ]
    variant = recipe.get("variant")
    inputs = recipe.get("inputs")
    label = (inputs or {}).get("label_name") if isinstance(inputs, dict) else None
    if not isinstance(config_summary, dict):
        return None, [
            _issue(
                DecisionStatus.NEEDS_USER_INPUT,
                "finetune_balanced requires one readable resolved finetune config.",
                {},
            )
        ]
    finetune: Any = config_summary.get("finetune") or {}
    task = finetune.get("task") or {}
    task_type = task.get("type")
    if variant not in _SUPPORTED_VARIANTS or (
        label not in _SUPPORTED_LABELS and task_type not in {"survival", "multilabel_classification"}
    ):
        return None, [
            _issue(
                DecisionStatus.NEEDS_USER_INPUT,
                "No unique finetune_balanced profile is registered for this variant and resolved task.",
                {
                    "variant": variant,
                    "label_name": label,
                    "task_type": task_type,
                    "supported_variants": sorted(_SUPPORTED_VARIANTS),
                    "supported_labels": sorted(_SUPPORTED_LABELS),
                    "supported_custom_task_types": ["multilabel_classification", "survival"],
                },
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
    configurations_value = search.get("configurations")
    configurations = configurations_value if isinstance(configurations_value, list) else []
    keys = sorted({str(key) for point in configurations if isinstance(point, dict) for key in point})
    family_keys = {
        "optimization.lr": [key for key in keys if key == "runtime.lr"],
        "optimization.weight_decay": [key for key in keys if key == "runtime.weight_decay"],
        "optimization.schedule": [
            key
            for key in keys
            if key.startswith("runtime.lr_")
            or key in {"runtime.epochs", "runtime.warmup_steps", "runtime.check_val_every_n_epoch", "runtime.patience"}
        ],
        "optimization.gradient_clip": [key for key in keys if key == "runtime.gradient_clip_val"],
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
        "adaptation.strategy": [key for key in keys if key == "yaml:/finetune/tuning"],
        "loss.pos_weight": [key for key in keys if key == "yaml:/finetune/loss/pos_weight"],
    }
    families = []
    fixed_schedule_parameters = []
    applicable_schedulers = {
        "runtime.warmup_steps": {"decay", "wsd"},
        "runtime.lr_decay_shape": {"decay", "wsd"},
        "runtime.lr_decay_ratio": {"wsd"},
        "runtime.lr_plateau_factor": {"plateau"},
        "runtime.lr_plateau_patience": {"plateau"},
    }
    for family_id, selected_keys in family_keys.items():
        if not selected_keys:
            continue
        varying_keys = []
        for key in selected_keys:
            values = _stable_unique(
                [
                    point[key]
                    for point in configurations
                    if key not in applicable_schedulers
                    or point.get("runtime.lr_scheduler") in applicable_schedulers[key]
                ]
            )
            if len(values) > 1:
                varying_keys.append(key)
            elif family_id == "optimization.schedule" and values:
                fixed_schedule_parameters.append(
                    {
                        "key": key,
                        "value": values[0],
                        "reason": "Fixed within applicable scheduler candidates to keep the joint search bounded.",
                    }
                )
        selected_keys = varying_keys
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
        "fixed_schedule_parameters": fixed_schedule_parameters,
        "schedule_policy": {
            "design": "Joint schedule candidates; individual effects are not isolated by the run budget.",
            "warmup": (
                "Decay/WSD null uses 3% of optimizer steps; Plateau has no warmup. "
                "Explicit step ratios need runtime counts. Shortened WSD candidates disable warmup."
            ),
            "plateau": "Validation-driven reductions use the frozen monitor and direction.",
            "validation": "Shortened candidates cap the validation interval at their epoch count.",
            "early_stopping": (
                "Patience counts validation checks, not epochs. The short level caps source patience at half its "
                "planned checks (rounded up, at least one); the long level allows its full validation horizon. "
                "The generated Plateau level allows one interval after a reduction on flat metrics; "
                "short horizons may still end before any reduction."
            ),
            "gradient_clip": (
                "Positive clipping sources add zero and half the source value; zero sources add 0.5 and 1.0."
            ),
            "fixed": (
                "Batch size and gradient accumulation remain source settings; "
                "accumulation does not enlarge Cox risk sets."
            ),
        },
    }


def _profile_axes(recipe: dict[str, Any], config_summary: ConfigSummaryInput) -> list[dict[str, Any]]:
    runtime_value = recipe.get("runtime")
    runtime = runtime_value if isinstance(runtime_value, dict) else {}
    model_value = config_summary.get("model")
    model = model_value if isinstance(model_value, dict) else {}
    finetune_value = config_summary.get("finetune")
    finetune = finetune_value if isinstance(finetune_value, dict) else {}

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

    clip = runtime.get("gradient_clip_val")
    clip = _finite_number(1.0 if clip is None else clip, "runtime.gradient_clip_val", non_negative=True)
    axes.append(
        _axis(
            "optimization.gradient_clip",
            "runtime.gradient_clip_val",
            [0.0, 0.5, 1.0] if clip == 0 else [clip, 0.0, clip / 2],
        )
    )
    patience = runtime.get("patience")
    patience = 100 if patience is None else patience
    if type(patience) is not int or patience < 0:
        raise ValueError("finetune_balanced requires runtime.patience to be a nonnegative integer.")

    epochs = runtime.get("epochs")
    epochs = 30 if epochs is None else epochs
    if type(epochs) is not int or epochs < 1:
        raise ValueError("finetune_balanced requires runtime.epochs to be a positive integer.")
    scheduler = runtime.get("lr_scheduler")
    scheduler = "decay" if scheduler is None else scheduler
    floor = runtime.get("lr_decay_floor")
    floor = 0.1 if floor is None else floor
    shape = runtime.get("lr_decay_shape")
    shape = "cosine" if shape is None else shape
    validation_interval = runtime.get("check_val_every_n_epoch")
    validation_interval = 1 if validation_interval is None else validation_interval
    if type(validation_interval) is not int or validation_interval < 1:
        raise ValueError("finetune_balanced requires runtime.check_val_every_n_epoch to be a positive integer.")
    shortened_epochs = max(1, (epochs + 1) // 2)
    shortened_interval = min(validation_interval, shortened_epochs)
    shortened_checks = shortened_epochs // shortened_interval
    baseline_schedule = {
        "runtime.epochs": epochs,
        "runtime.patience": patience,
        "runtime.check_val_every_n_epoch": validation_interval,
        "runtime.lr_scheduler": scheduler,
        "runtime.warmup_steps": runtime.get("warmup_steps"),
        "runtime.lr_decay_floor": floor,
        "runtime.lr_decay_shape": shape,
        "runtime.lr_decay_ratio": runtime.get("lr_decay_ratio"),
        "runtime.lr_plateau_factor": runtime.get("lr_plateau_factor"),
        "runtime.lr_plateau_patience": runtime.get("lr_plateau_patience"),
    }
    if scheduler == "plateau":
        if baseline_schedule["runtime.lr_plateau_factor"] is None:
            baseline_schedule["runtime.lr_plateau_factor"] = 0.1
        if baseline_schedule["runtime.lr_plateau_patience"] is None:
            baseline_schedule["runtime.lr_plateau_patience"] = 10
    decay_schedule = {
        **baseline_schedule,
        "runtime.lr_scheduler": "decay",
        "runtime.lr_decay_ratio": None,
        "runtime.lr_plateau_factor": None,
        "runtime.lr_plateau_patience": None,
    }
    schedule_levels = [
        baseline_schedule,
        {
            **baseline_schedule,
            "runtime.epochs": shortened_epochs,
            "runtime.warmup_steps": 0 if scheduler == "wsd" else baseline_schedule["runtime.warmup_steps"],
            "runtime.check_val_every_n_epoch": shortened_interval,
            "runtime.patience": min(patience, max(1, (shortened_checks + 1) // 2)),
        },
        {
            **baseline_schedule,
            "runtime.epochs": epochs * 2,
            "runtime.patience": max(patience, (epochs * 2) // validation_interval),
        },
        {**decay_schedule, "runtime.warmup_steps": 0},
        {**decay_schedule, "runtime.warmup_steps": None},
        {**decay_schedule, "runtime.lr_decay_floor": 0.01},
        {**decay_schedule, "runtime.lr_decay_floor": 1.0},
        {**decay_schedule, "runtime.lr_decay_shape": "linear" if shape == "cosine" else "cosine"},
        *[
            {
                **decay_schedule,
                "runtime.lr_scheduler": "wsd",
                "runtime.warmup_steps": None,
                "runtime.lr_decay_ratio": ratio,
            }
            for ratio in (0.2, 0.5)
        ],
        {
            **decay_schedule,
            "runtime.lr_scheduler": "plateau",
            "runtime.warmup_steps": None,
            "runtime.lr_decay_shape": "cosine",
            "runtime.lr_plateau_factor": 0.1,
            "runtime.lr_plateau_patience": 2,
            "runtime.patience": max(patience, 4),
        },
    ]
    axes.append({"id": "optimization.schedule", "levels": _stable_unique(schedule_levels)})

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

    head_value = model.get("head_details")
    head = head_value if isinstance(head_value, dict) else {}
    head_dropout = _finite_dropout(head.get("dropout"), "model.head.dropout")
    head_kwargs_value = head.get("kwargs")
    head_kwargs = head_kwargs_value if isinstance(head_kwargs_value, dict) else {}
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

    tuning = finetune.get("tuning")
    if not finetune.get("tuning_present") or not isinstance(tuning, dict):
        raise ValueError("finetune_balanced requires an explicit finetune.tuning mapping.")
    preset = tuning.get("preset")
    if type(preset) is not str or not preset:
        raise ValueError("finetune_balanced requires an explicit finetune.tuning.preset.")
    inputs_value = recipe.get("inputs")
    inputs = inputs_value if isinstance(inputs_value, dict) else {}
    has_trained_backbone = inputs.get("pretrained_backbone_path") not in (None, "")
    # What matters is which parameters receive gradient, not what the preset is called: a
    # `head_only` config that unfreezes the encoder through `groups` does train a backbone,
    # and a `full` config that freezes it through `groups` does not. Read the effective value.
    if not has_trained_backbone and not _trains_encoder(tuning):
        raise ValueError(
            f"finetune_balanced cannot freeze the encoder under preset '{preset}' without a pretrained backbone."
        )
    # The three adaptation strategies are now presets, so the axis sweeps preset names
    # instead of the boolean pair that used to encode them. It has to replace the whole
    # `tuning` block rather than just the preset: leaving the source config's `groups`
    # overrides in place would let an arm labelled `head_only` keep training whatever the
    # source unfroze. The adapter shape rides along -- it is hyperparameters, not a switch,
    # so a preset that leaves the lora group untrained may still carry it.
    lora_shape = {"lora": tuning["lora"]} if isinstance(tuning.get("lora"), dict) else {}
    tuning_levels = [_canonical(tuning)]
    if has_trained_backbone:
        tuning_levels.extend(_canonical({"preset": name, **lora_shape}) for name in ("full", "head_only", "lora"))
    axes.append(_axis("adaptation.strategy", "yaml:/finetune/tuning", _stable_unique(tuning_levels)))

    loss_value = finetune.get("loss")
    loss = loss_value if isinstance(loss_value, dict) else {}
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
    return levels, {(left, right) for left, right in combinations(sorted(levels), 2)}


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


# Duplicated from the variants' preset tables, which agent_tools cannot import across the
# enforced-fork boundary. `full` is the only preset of the supported variants whose encoder
# trains; `custom` carries no table at all and must state every group explicitly.
_ENCODER_TRAINING_PRESETS = {"full"}


def _trains_encoder(tuning: dict[str, Any]) -> bool:
    groups_value = tuning.get("groups")
    groups = groups_value if isinstance(groups_value, dict) else {}
    override_value = groups.get("encoder")
    override = override_value if isinstance(override_value, dict) else {}
    if "train" in override:
        return bool(override["train"])
    return tuning.get("preset") in _ENCODER_TRAINING_PRESETS


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


def _is_finite_number(value: Any) -> TypeGuard[int | float]:
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
