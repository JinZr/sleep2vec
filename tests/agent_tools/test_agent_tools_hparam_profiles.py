from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from agent_tool_test_helpers import run_execution_preflight_fixture
import pytest
import yaml

from agent_tools import managed_scheduler, plans
from agent_tools.configs import config_summary
from agent_tools.decision_models import DecisionStatus
from agent_tools.domain.finetune_hparam_profile import (
    _balanced_configurations,
    compile_finetune_balanced_profile,
    finetune_balanced_profile_audit,
)
from agent_tools.models import REPO_ROOT
from agent_tools.plan_hparam import apply_search_overrides, validate_finetune_config_bytes


@pytest.fixture(autouse=True)
def _stub_execution_target(monkeypatch):
    monkeypatch.setattr(managed_scheduler, "run_execution_command", run_execution_preflight_fixture)


def _recipe(*, label: str = "ahi", variant: str = "sleep2vec", max_runs: int | None = None) -> dict:
    search = {"profile": "finetune_balanced"}
    if max_runs is not None:
        search["max_runs"] = max_runs
    return {
        "variant": variant,
        "inputs": {"label_name": label, "pretrained_backbone_path": "/pretrained.ckpt"},
        "runtime": {"lr": 1.0e-6, "weight_decay": 1.0e-5},
        "search": search,
    }


def _summary(*, depth: int = 12, channels: int = 1, temporal: str | None = None, pos_weight=None) -> dict:
    kwargs = {temporal: 0.1} if temporal else {}
    return {
        "model": {
            "backbone_depth": depth,
            "channels": [{"name": f"channel-{index}"} for index in range(channels)],
            "head_details": {"dropout": 0.1, "kwargs": kwargs},
            "layer_mix_present": True,
            "layer_mix": {
                "enabled": False,
                "shared_across_modalities": False,
                "layer_indices": None,
            },
        },
        "finetune": {
            "tuning_present": True,
            "tuning": {
                "preset": "full",
                "groups": {"tokenizers": {"train": False}},
                "lora": {"r": 8, "alpha": 16},
            },
            "loss": {"pos_weight": pos_weight, "class_weights": [1.0, 1.0, 1.0, 1.0]},
        },
    }


def _compile(recipe: dict | None = None, summary: dict | None = None) -> dict:
    compiled, issues = compile_finetune_balanced_profile(recipe or _recipe(), summary or _summary())
    assert issues == []
    assert compiled is not None
    return compiled


def _unique_values(configurations: list[dict], key: str) -> list:
    values = []
    seen = set()
    for point in configurations:
        value = point[key]
        marker = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if marker not in seen:
            seen.add(marker)
            values.append(value)
    return values


def test_default_profile_materializes_twelve_deterministic_unique_joint_points():
    first = _compile()
    reordered = _summary()
    reordered["model"]["layer_mix"] = dict(reversed(list(reordered["model"]["layer_mix"].items())))
    reordered["finetune"]["tuning"] = dict(reversed(list(reordered["finetune"]["tuning"].items())))
    second = _compile(summary=reordered)

    assert first == second
    assert first["method"] == "grid"
    assert first["max_runs"] == 12
    assert len(first["configurations"]) == 12
    assert len({json.dumps(point, sort_keys=True) for point in first["configurations"]}) == 12
    assert all(set(point) == set(first["configurations"][0]) for point in first["configurations"])

    baseline = first["configurations"][0]
    assert baseline["runtime.lr"] == 1.0e-6
    assert baseline["runtime.weight_decay"] == 1.0e-5
    assert baseline["yaml:/model/head/dropout"] == 0.1
    assert baseline["yaml:/finetune/layer_mix"] == {
        "enabled": False,
        "layer_indices": None,
        "shared_across_modalities": False,
    }
    assert baseline["yaml:/finetune/tuning"] == {
        "groups": {"tokenizers": {"train": False}},
        "lora": {"alpha": 16, "r": 8},
        "preset": "full",
    }

    audit = finetune_balanced_profile_audit(first)
    assert audit["candidate_count"] == 12
    assert {family["id"]: family["covered_levels"] for family in audit["searched_families"]} == {
        "optimization.lr": 3,
        "optimization.weight_decay": 3,
        "model.layer_mix": 4,
        "regularization.dropout": 3,
        # Four, not three: this source carries a `groups` override, so it is a policy the
        # three clean preset arms do not reproduce.
        "adaptation.strategy": 4,
    }


def test_balanced_selection_prioritizes_levels_then_pairs_with_stable_ties():
    axes = [
        {"id": "a", "levels": [{"a": 0}, {"a": 1}]},
        {"id": "b", "levels": [{"b": 0}, {"b": 1}]},
        {"id": "c", "levels": [{"c": 0}, {"c": 1}]},
    ]
    configurations = _balanced_configurations(axes, 4)

    assert configurations == [
        {"a": 0, "b": 0, "c": 0},
        {"a": 1, "b": 1, "c": 1},
        {"a": 0, "b": 0, "c": 1},
        {"a": 0, "b": 1, "c": 0},
    ]


def test_zero_weight_decay_uses_profile_owned_positive_anchors():
    recipe = _recipe()
    recipe["runtime"]["weight_decay"] = 0.0

    compiled = _compile(recipe=recipe)
    configurations = compiled["configurations"]

    assert _unique_values(configurations, "runtime.weight_decay") == [0.0, 1.0e-5, 1.0e-4]
    audit = finetune_balanced_profile_audit(compiled)
    weight_decay = next(family for family in audit["searched_families"] if family["id"] == "optimization.weight_decay")
    assert weight_decay["covered_levels"] == 3


def test_omitted_runtime_axes_use_canonical_finetune_defaults():
    recipe = _recipe()
    recipe["runtime"] = {}

    configurations = _compile(recipe=recipe)["configurations"]

    assert configurations[0]["runtime.lr"] == 1.0e-6
    assert configurations[0]["runtime.weight_decay"] == 1.0e-5


@pytest.mark.parametrize(
    ("depth", "expected_indices"),
    [
        (4, {(3, 4), (1, 2, 3, 4)}),
        (12, {(11, 12), (9, 10, 11, 12), (1, 5, 8, 12)}),
    ],
)
def test_layer_mix_is_atomic_depth_derived_and_deduplicated(depth: int, expected_indices: set[tuple[int, ...]]):
    configurations = _compile(summary=_summary(depth=depth))["configurations"]
    values = _unique_values(configurations, "yaml:/finetune/layer_mix")

    assert any(value["enabled"] is False and value["layer_indices"] is None for value in values)
    assert {tuple(value["layer_indices"]) for value in values if value["enabled"] is True} == expected_indices
    assert {value["shared_across_modalities"] for value in values} == {False}
    assert all("yaml:/finetune/layer_mix/enabled" not in point for point in configurations)


def test_layer_mix_profile_keeps_enabled_source_first_and_retains_normalized_off_arm():
    summary = _summary(depth=12)
    summary["model"]["layer_mix"].update({"enabled": True, "layer_indices": [6, 12]})

    values = _unique_values(_compile(summary=summary)["configurations"], "yaml:/finetune/layer_mix")

    assert values[0] == summary["model"]["layer_mix"]
    assert any(
        value["enabled"] is False and value["layer_indices"] is None and value["shared_across_modalities"] is False
        for value in values
    )


def test_multichannel_layer_mix_searches_both_shared_modes_atomically():
    compiled = _compile(summary=_summary(depth=12, channels=2))
    values = _unique_values(compiled["configurations"], "yaml:/finetune/layer_mix")

    enabled = [value for value in values if value["enabled"] is True]
    disabled = [value for value in values if value["enabled"] is False and value["layer_indices"] is None]
    assert {value["shared_across_modalities"] for value in enabled} == {False, True}
    assert disabled == [{"enabled": False, "layer_indices": None, "shared_across_modalities": False}]
    audit = finetune_balanced_profile_audit(compiled)
    layer_family = next(family for family in audit["searched_families"] if family["id"] == "model.layer_mix")
    assert layer_family["covered_levels"] == 7


@pytest.mark.parametrize(
    "layer_mix",
    [
        {"enabled": False, "shared_across_modalities": False, "layer_indices": [1, 2]},
        {"enabled": False, "shared_across_modalities": True, "layer_indices": None},
        {"enabled": True, "shared_across_modalities": True, "layer_indices": [6, 12]},
    ],
)
def test_profile_rejects_inert_source_layer_mix_states(layer_mix):
    summary = _summary()
    summary["model"]["layer_mix"] = layer_mix

    compiled, issues = compile_finetune_balanced_profile(_recipe(), summary)

    assert compiled is None
    assert [issue.status for issue in issues] == [DecisionStatus.FAIL]


@pytest.mark.parametrize("temporal", ["temporal_dropout", "attn_dropout"])
def test_arousal_synchronizes_dropout_and_searches_scalar_pos_weight(temporal: str):
    configurations = _compile(
        recipe=_recipe(label="arousal"),
        summary=_summary(depth=4, channels=2, temporal=temporal, pos_weight=1.0),
    )["configurations"]
    temporal_key = f"yaml:/model/head/kwargs/{temporal}"

    assert all(point["yaml:/model/head/dropout"] == point[temporal_key] for point in configurations)
    assert set(_unique_values(configurations, "yaml:/finetune/loss/pos_weight")) == {0.5, 1.0, 2.0}


def test_stage4_keeps_class_weights_frozen():
    configurations = _compile(recipe=_recipe(label="stage4"), summary=_summary(pos_weight=None))["configurations"]

    assert all("yaml:/finetune/loss/pos_weight" not in point for point in configurations)
    assert all("yaml:/finetune/loss/class_weights" not in point for point in configurations)


def test_regularization_profile_synchronizes_a_mismatched_source_dropout():
    summary = _summary(temporal="temporal_dropout")
    summary["model"]["head_details"]["kwargs"]["temporal_dropout"] = 0.2

    configurations = _compile(summary=summary)["configurations"]
    values = {
        (
            point["yaml:/model/head/dropout"],
            point["yaml:/model/head/kwargs/temporal_dropout"],
        )
        for point in configurations
    }

    assert values == {(0.0, 0.0), (0.1, 0.1), (0.1, 0.2), (0.2, 0.2)}
    assert configurations[0]["yaml:/model/head/dropout"] == 0.1
    assert configurations[0]["yaml:/model/head/kwargs/temporal_dropout"] == 0.2


def test_adaptation_keeps_source_first_and_sweeps_the_preset_arms():
    configurations = _compile()["configurations"]
    values = _unique_values(configurations, "yaml:/finetune/tuning")

    # The source block stays first, verbatim. Each swept arm then replaces the whole
    # block: it keeps the LoRA hyperparameters, which are shape rather than a switch, and
    # drops the source's `groups` overrides -- an arm named head_only that inherited them
    # would not be head-only.
    assert values[0] == {
        "groups": {"tokenizers": {"train": False}},
        "lora": {"alpha": 16, "r": 8},
        "preset": "full",
    }
    assert values[1:] == [
        {"lora": {"alpha": 16, "r": 8}, "preset": "full"},
        {"lora": {"alpha": 16, "r": 8}, "preset": "head_only"},
        {"lora": {"alpha": 16, "r": 8}, "preset": "lora"},
    ]
    assert all("yaml:/finetune/tuning/preset" not in point for point in configurations)


@pytest.mark.parametrize("source_preset", ["head_only", "lora"])
def test_adaptation_adds_full_arm_when_source_backbone_is_frozen(source_preset: str):
    summary = _summary()
    summary["finetune"]["tuning"]["preset"] = source_preset

    values = _unique_values(_compile(summary=summary)["configurations"], "yaml:/finetune/tuning")

    assert values[0]["preset"] == source_preset
    assert {value["preset"] for value in values} == {"full", "head_only", "lora"}


def test_adaptation_does_not_freeze_a_randomly_initialized_backbone():
    recipe = _recipe()
    recipe["inputs"]["pretrained_backbone_path"] = None

    compiled = _compile(recipe=recipe)
    configurations = compiled["configurations"]
    values = _unique_values(configurations, "yaml:/finetune/tuning")

    assert [value["preset"] for value in values] == ["full"]
    assert "adaptation.strategy" not in {
        family["id"] for family in finetune_balanced_profile_audit(compiled)["searched_families"]
    }


def test_final_evaluation_checkpoint_does_not_enable_frozen_backbone_arms():
    recipe = _recipe()
    recipe["inputs"].update({"pretrained_backbone_path": None, "ckpt_path": "/resume.ckpt"})

    values = _unique_values(_compile(recipe=recipe)["configurations"], "yaml:/finetune/tuning")

    assert [value["preset"] for value in values] == ["full"]


def test_random_backbone_preflight_reads_the_groups_override_not_the_preset_name():
    """A `groups` override decides trainability, so the preflight has to read it.

    Both halves matter: `head_only` that unfreezes the encoder does train a backbone from
    scratch, and `full` that freezes it does not, whatever the preset is called.
    """
    recipe = _recipe()
    recipe["inputs"]["pretrained_backbone_path"] = None

    trains_anyway = _summary()
    trains_anyway["finetune"]["tuning"] = {"preset": "head_only", "groups": {"encoder": {"train": True}}}
    compiled, issues = compile_finetune_balanced_profile(recipe, trains_anyway)
    assert issues == []
    assert compiled is not None

    frozen_anyway = _summary()
    frozen_anyway["finetune"]["tuning"] = {"preset": "full", "groups": {"encoder": {"train": False}}}
    compiled, issues = compile_finetune_balanced_profile(recipe, frozen_anyway)
    assert compiled is None
    assert [issue.status for issue in issues] == [DecisionStatus.FAIL]


def test_profile_rejects_freezing_random_source_backbone():
    recipe = _recipe()
    recipe["inputs"]["pretrained_backbone_path"] = None
    summary = _summary()
    summary["finetune"]["tuning"]["preset"] = "head_only"

    compiled, issues = compile_finetune_balanced_profile(recipe, summary)

    assert compiled is None
    assert [issue.status for issue in issues] == [DecisionStatus.FAIL]


@pytest.mark.parametrize(
    ("mutate", "status"),
    [
        (lambda recipe: recipe["search"].update({"parameters": {"runtime.lr": [1e-6]}}), DecisionStatus.FAIL),
        (lambda recipe: recipe.update({"variant": "sleep2expert"}), DecisionStatus.NEEDS_USER_INPUT),
        (lambda recipe: recipe["inputs"].update({"label_name": "stage5"}), DecisionStatus.NEEDS_USER_INPUT),
        (lambda recipe: recipe["inputs"].update({"label_name": "custom_label"}), DecisionStatus.NEEDS_USER_INPUT),
        (lambda recipe: recipe["search"].update({"max_runs": 3}), DecisionStatus.NEEDS_USER_INPUT),
        (lambda recipe: recipe["search"].update({"max_runs": 33}), DecisionStatus.NEEDS_USER_INPUT),
        (lambda recipe: recipe.update({"adaptive": {"enabled": True}}), DecisionStatus.NEEDS_USER_INPUT),
        (lambda recipe: recipe["runtime"].update({"lr": 0}), DecisionStatus.FAIL),
    ],
)
def test_profile_rejects_ambiguous_or_invalid_inputs(mutate, status: DecisionStatus):
    recipe = _recipe()
    mutate(recipe)

    compiled, issues = compile_finetune_balanced_profile(recipe, _summary())

    assert compiled is None
    assert [issue.status for issue in issues] == [status]
    assert issues[0].evidence["preflight_before_workspace"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda summary: summary["finetune"].update({"tuning_present": False, "tuning": {}}),
        lambda summary: summary["finetune"]["tuning"].pop("preset"),
    ],
)
def test_profile_requires_explicit_tuning_mapping(mutate):
    summary = _summary()
    mutate(summary)

    compiled, issues = compile_finetune_balanced_profile(_recipe(), summary)

    assert compiled is None
    assert [issue.status for issue in issues] == [DecisionStatus.FAIL]
    assert issues[0].evidence["preflight_before_workspace"] is True


def test_legacy_explicit_search_is_not_a_profile_compilation_request():
    recipe = _recipe()
    recipe["search"] = {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1e-6]}}
    before = copy.deepcopy(recipe)

    compiled, _issues = compile_finetune_balanced_profile(recipe, _summary())

    assert compiled is None
    assert recipe == before


def _profile_recipe(tmp_path: Path, *, max_runs: int | None = None) -> tuple[Path, Path]:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    index = inputs_dir / "index.csv"
    index.write_text("path,split,duration,ppg_mask,ah_event_mask,stage_mask\nx.npz,train,60,1,1,1\n")
    config_payload = yaml.safe_load((REPO_ROOT / "configs/ppg_ahi_finetune_large.yaml").read_text())
    config_payload["data"]["finetune_data_index"] = str(index)
    config = inputs_dir / "config.yaml"
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False))

    workspace = tmp_path / "workspace"
    experiment = {
        "id": "auto-profile",
        "title": "Automatic profile",
        "objective": "Exercise bounded automatic tuning.",
        "root": str(workspace),
        "baseline": {"type": "none", "rationale": "unit fixture"},
    }
    base = yaml.safe_load((REPO_ROOT / "recipes/templates/finetune_ppg_ahi.yaml").read_text())
    base["experiment"] = experiment
    base["inputs"]["config"] = str(config)
    base_path = inputs_dir / "base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))

    recipe = yaml.safe_load((REPO_ROOT / "recipes/templates/hparam_tune_ppg_ahi.yaml").read_text())
    recipe["experiment"] = experiment
    recipe["step"] = {"id": "auto-tune", "phase": "train", "purpose": "Select a bounded candidate."}
    recipe["base_recipe"] = str(base_path)
    if max_runs is not None:
        recipe["search"]["max_runs"] = max_runs
    recipe["evaluation_policy"].update(
        {
            "selection_metric": "val_ahi_pearson",
            "selection_split": "val",
            "external_test_locked": True,
            "test_after_fit": False,
            "final_test_unlocked": False,
        }
    )
    recipe["decisions"].update(
        {
            "external_test_locked": {"value": True, "source": "explicit_recipe"},
            "train_val_test_policy": {"value": "val", "source": "explicit_recipe"},
            "test_after_fit": {"value": False, "source": "explicit_recipe"},
        }
    )
    recipe_path = inputs_dir / "profile.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    return recipe_path, workspace


def test_profile_binding_and_plan_freeze_exact_expansion(tmp_path: Path):
    recipe_path, workspace = _profile_recipe(tmp_path)
    authored_recipe_bytes = recipe_path.read_bytes()

    effective, _summary, doctor = plans.evaluate_recipe(recipe_path)

    assert doctor.exit_code == 0
    assert effective["search"]["profile"] == "finetune_balanced"
    assert effective["search"]["method"] == "grid"
    assert len(effective["search"]["configurations"]) == 12

    plan_dir = workspace / "plans" / "auto"
    source_config_path = Path(effective["inputs"]["config"])
    source_config_bytes = source_config_path.read_bytes()
    report = plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)

    assert report.exit_code == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    assert plan["recipe"]["search"] == resolved["search"]
    assert plan["recipe"]["search"]["configurations"] == effective["search"]["configurations"]
    assert "search_profile" not in plan
    assert len(plan["runs"]) == len(effective["search"]["configurations"])
    base_config = yaml.safe_load(source_config_bytes)
    assert effective["inputs"]["pretrained_backbone_path"] is None
    assert _unique_values(effective["search"]["configurations"], "yaml:/finetune/tuning") == [
        base_config["finetune"]["tuning"]
    ]
    for run, point in zip(plan["runs"], effective["search"]["configurations"]):
        assert {key: run[key] for key in point} == point
        expected_config = copy.deepcopy(base_config)
        apply_search_overrides(expected_config, point)
        run_config_bytes = Path(run["config"]).read_bytes()
        assert yaml.safe_load(run_config_bytes) == expected_config
        assert hashlib.sha256(run_config_bytes).hexdigest() == run["config_sha256"]
    assert (plan_dir / "config.source.yaml").read_bytes() == source_config_bytes
    assert recipe_path.read_bytes() == authored_recipe_bytes
    assert source_config_path.read_bytes() == source_config_bytes
    markdown = (plan_dir / "plan.md").read_text()
    assert "finetune_balanced" in markdown
    assert "best observed candidate within frozen search domain" in markdown
    assert "adaptation.strategy" not in markdown


@pytest.mark.parametrize("max_runs", [3, 33])
def test_invalid_profile_budget_fails_before_workspace_mutation(tmp_path: Path, max_runs: int):
    recipe_path, workspace = _profile_recipe(tmp_path, max_runs=max_runs)

    report = plans.build_plan(recipe_path=recipe_path, output_dir=workspace / "plans" / "auto")

    assert report.exit_code == 2
    assert not workspace.exists()
    issues = [issue for issue in report.issues if issue.field == "hparam_search_profile"]
    assert len(issues) == 1
    assert issues[0].evidence["preflight_before_workspace"] is True


def test_tiny_fixture_keeps_legacy_explicit_search():
    recipe, _summary, report = plans.evaluate_recipe(REPO_ROOT / "recipes/examples/tiny_fixture_hparam.yaml")

    assert report.exit_code == 0
    assert "profile" not in recipe["search"]
    assert recipe["search"]["parameters"] == {"runtime.lr": [1.0e-6]}


def test_checked_in_templates_select_only_supported_profile_variants():
    root = yaml.safe_load((REPO_ROOT / "recipes/templates/hparam_tune_ppg_ahi.yaml").read_text())
    sleep2vec2 = yaml.safe_load((REPO_ROOT / "recipes/templates/hparam_tune_sleep2vec2_ppg_ahi.yaml").read_text())
    sleep2expert = yaml.safe_load((REPO_ROOT / "recipes/templates/hparam_tune_sleep2expert_ahi.yaml").read_text())

    assert root["search"] == {"profile": "finetune_balanced"}
    assert sleep2vec2["search"] == {"profile": "finetune_balanced"}
    for template in (root, sleep2vec2):
        assert template["evaluation_policy"] == {
            "selection_metric": "val_ahi_pearson",
            "selection_mode": "max",
            "selection_split": "val",
            "final_eval_split": "test",
            "external_test_locked": True,
            "test_after_fit": False,
            "require_manual_unlock_for_final_test": True,
            "final_test_unlocked": False,
        }
        assert template["decisions"]["external_test_locked"]["value"] is True
        assert template["decisions"]["test_after_fit"]["value"] is False
        assert template["decisions"]["train_val_test_policy"]["value"] == "val"
    assert "profile" not in sleep2expert["search"]
    assert "parameters" in sleep2expert["search"]


def test_task_recipe_schema_profile_skeleton_keeps_test_locked():
    text = (REPO_ROOT / "recipes/schemas/task_recipe.schema.md").read_text()
    block = text.split("```yaml\n", 1)[1].split("\n```", 1)[0]
    skeleton = yaml.safe_load(block)

    policy = skeleton["evaluation_policy"]
    assert policy["selection_metric"] == "val_ahi_pearson"
    assert policy["selection_split"] == "val"
    assert policy["external_test_locked"] is True
    assert policy["test_after_fit"] is False
    assert policy["final_test_unlocked"] is False


@pytest.mark.parametrize(
    ("config_path", "variant", "label"),
    [
        ("configs/ppg_ahi_finetune_large.yaml", "sleep2vec", "ahi"),
        ("configs/ppg_age_finetune_large.yaml", "sleep2vec", "age"),
        ("configs/ppg_sex_finetune_large.yaml", "sleep2vec", "sex"),
        ("configs/examples/arousal/FINETUNE_EXAMPLE.yaml", "sleep2vec", "arousal"),
        ("configs/examples/stage4/FINETUNE_EXAMPLE.yaml", "sleep2vec", "stage4"),
        ("configs/sleep2vec2/ppg_ahi_finetune_large.yaml", "sleep2vec2", "ahi"),
        ("configs/sleep2vec2/ppg_age_finetune_large.yaml", "sleep2vec2", "age"),
        ("configs/sleep2vec2/ppg_sex_finetune_large.yaml", "sleep2vec2", "sex"),
    ],
)
def test_generated_points_pass_variant_config_validation(config_path: str, variant: str, label: str):
    source = REPO_ROOT / config_path
    payload = yaml.safe_load(source.read_text())
    summary = config_summary(source, variant=variant, validate_survival_local_paths=False)
    compiled = _compile(recipe=_recipe(label=label, variant=variant), summary=summary)

    assert compiled["max_runs"] == 12
    assert len(compiled["configurations"]) == 12
    first = compiled["configurations"][0]
    assert first["yaml:/finetune/layer_mix"] == payload["finetune"]["layer_mix"]
    assert first["yaml:/finetune/tuning"] == payload["finetune"]["tuning"]
    assert first["yaml:/model/head/dropout"] == payload["model"]["head"]["dropout"]
    for field in ("attn_dropout", "temporal_dropout"):
        if field in payload["model"]["head"].get("kwargs", {}):
            assert first[f"yaml:/model/head/kwargs/{field}"] == payload["model"]["head"]["kwargs"][field]

    for point in compiled["configurations"]:
        candidate = copy.deepcopy(payload)
        apply_search_overrides(candidate, point)
        validate_finetune_config_bytes(
            _recipe(label=label, variant=variant),
            yaml.safe_dump(candidate, sort_keys=False).encode(),
        )


@pytest.mark.parametrize(
    ("config_path", "variant", "label"),
    [
        ("configs/ppg_sex_finetune_large.yaml", "sleep2vec", "age"),
        ("configs/sleep2vec2/ppg_age_finetune_large.yaml", "sleep2vec2", "sex"),
    ],
)
def test_profile_candidate_validation_rejects_label_task_mismatch(config_path: str, variant: str, label: str):
    source = REPO_ROOT / config_path

    with pytest.raises(ValueError, match=f"when --label-name is '{label}'"):
        validate_finetune_config_bytes(_recipe(label=label, variant=variant), source.read_bytes())


@pytest.mark.parametrize(
    ("config_path", "variant", "label", "section", "field", "value", "message"),
    [
        (
            "configs/ppg_age_finetune_large.yaml",
            "sleep2vec",
            "age",
            "loss",
            "class_weights",
            [1.0],
            "class_weights is only supported for single-label classification",
        ),
        (
            "configs/sleep2vec2/ppg_age_finetune_large.yaml",
            "sleep2vec2",
            "age",
            "loss",
            "pos_weight",
            1.0,
            "pos_weight is only supported for multilabel classification",
        ),
        (
            "configs/ppg_age_finetune_large.yaml",
            "sleep2vec",
            "age",
            "sampler",
            "weighted_random",
            True,
            "weighted_random is only supported for binary non-sequence classification",
        ),
        (
            "configs/sleep2vec2/ppg_sex_finetune_large.yaml",
            "sleep2vec2",
            "sex",
            "loss",
            "pos_weight",
            1.0,
            "pos_weight is only supported for multilabel classification",
        ),
    ],
)
def test_profile_candidate_validation_rejects_task_incompatible_imbalance(
    config_path: str,
    variant: str,
    label: str,
    section: str,
    field: str,
    value,
    message: str,
):
    payload = yaml.safe_load((REPO_ROOT / config_path).read_text())
    payload["finetune"][section][field] = value

    with pytest.raises(ValueError, match=message):
        validate_finetune_config_bytes(
            _recipe(label=label, variant=variant),
            yaml.safe_dump(payload, sort_keys=False).encode(),
        )


@pytest.mark.parametrize("case", ["conflicting_search", "unsupported_variant", "missing_tuning"])
def test_profile_contract_failure_precedes_workspace_mutation(tmp_path: Path, case: str):
    recipe_path, workspace = _profile_recipe(tmp_path)
    recipe = yaml.safe_load(recipe_path.read_text())
    if case == "conflicting_search":
        recipe["search"]["parameters"] = {"runtime.lr": [1.0e-6]}
    elif case == "unsupported_variant":
        recipe["variant"] = "sleep2expert"
        base_path = Path(recipe["base_recipe"])
        base = yaml.safe_load(base_path.read_text())
        base["variant"] = "sleep2expert"
        base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    else:
        base = yaml.safe_load(Path(recipe["base_recipe"]).read_text())
        config_path = Path(base["inputs"]["config"])
        config = yaml.safe_load(config_path.read_text())
        config["finetune"].pop("tuning")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))

    report = plans.build_plan(recipe_path=recipe_path, output_dir=workspace / "plans" / "auto")

    assert report.exit_code != 0
    assert not workspace.exists()
    assert any(issue.field == "hparam_search_profile" for issue in report.issues)


def test_generated_config_validation_failure_precedes_workspace_mutation(tmp_path: Path, monkeypatch):
    recipe_path, workspace = _profile_recipe(tmp_path)
    before = {
        str(path.relative_to(tmp_path)): path.read_bytes() for path in sorted(tmp_path.rglob("*")) if path.is_file()
    }

    def reject_generated_config(_recipe, _config_bytes):
        raise ValueError("generated profile config is invalid")

    monkeypatch.setattr("agent_tools.plan_hparam.validate_finetune_config_bytes", reject_generated_config)

    report = plans.build_plan(recipe_path=recipe_path, output_dir=workspace / "plans" / "auto")

    after = {
        str(path.relative_to(tmp_path)): path.read_bytes() for path in sorted(tmp_path.rglob("*")) if path.is_file()
    }
    lock_path = workspace.parent / f".{workspace.name}.plan-registration.lock"
    assert report.exit_code == 1
    assert lock_path.is_file() and not lock_path.is_symlink()
    assert after.pop(str(lock_path.relative_to(tmp_path))) == b""
    assert before == after
    assert not workspace.exists()
    issue = next(issue for issue in report.issues if issue.field == "hparam_search_space")
    assert issue.evidence["preflight_before_workspace"] is True


def test_profile_label_task_mismatch_fails_before_workspace_mutation(tmp_path: Path):
    recipe_path, workspace = _profile_recipe(tmp_path)
    recipe = yaml.safe_load(recipe_path.read_text())
    base_path = Path(recipe["base_recipe"])
    base = yaml.safe_load(base_path.read_text())
    base["inputs"]["label_name"] = "age"
    base["decisions"]["label_name"]["value"] = "age"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    recipe["decisions"]["label_name"]["value"] = "age"
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))

    report = plans.build_plan(recipe_path=recipe_path, output_dir=workspace / "plans" / "auto")

    assert report.exit_code == 1
    assert not workspace.exists()
    issue = next(issue for issue in report.issues if issue.field == "hparam_search_space")
    assert "when --label-name is 'age'" in issue.message
    assert issue.evidence["preflight_before_workspace"] is True


@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2vec2"])
def test_real_profile_candidate_validation_precedes_target_probe(tmp_path: Path, monkeypatch, variant: str):
    from agent_tools.domain import finetune_hparam_profile

    recipe_path, workspace = _profile_recipe(tmp_path)
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe["variant"] = variant
    base_path = Path(recipe["base_recipe"])
    base = yaml.safe_load(base_path.read_text())
    base["variant"] = variant
    base_path.write_text(yaml.safe_dump(base))
    recipe_path.write_text(yaml.safe_dump(recipe))
    compile_profile = finetune_hparam_profile.compile_finetune_balanced_profile

    def invalid_compilation(*args, **kwargs):
        compiled, issues = compile_profile(*args, **kwargs)
        assert issues == []
        assert compiled is not None
        # Corrupt one produced point without adding a new profile axis.
        compiled["configurations"][-1]["yaml:/finetune/layer_mix"] = {
            "enabled": True,
            "layer_indices": [999],
            "shared_across_modalities": False,
        }
        return compiled, issues

    monkeypatch.setattr(finetune_hparam_profile, "compile_finetune_balanced_profile", invalid_compilation)
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda *_args, **_kwargs: pytest.fail("Invalid candidate reached target preflight"),
    )

    report = plans.build_plan(recipe_path=recipe_path, output_dir=workspace / "plans" / "auto")

    assert report.exit_code == 1
    issue = next(issue for issue in report.issues if issue.field == "hparam_search_space")
    assert "layer_indices" in issue.message
    assert issue.evidence["preflight_before_workspace"] is True
    assert not workspace.exists()
