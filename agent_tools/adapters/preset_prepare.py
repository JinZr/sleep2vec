from __future__ import annotations

from typing import Any

from ..decision_models import DecisionIssue, DecisionStatus, ResolvedDecision, needs_issue
from ..decision_paths import multilabel_sidecar_issue, survival_sidecar_issue
from ..models import coerce_list
from ..plan_rendering import PRESET_FIELDS, preset_cli_args, render_command
from .base import TaskAdapter


class PresetPrepareAdapter(TaskAdapter):
    task = "preset_prepare"

    recipe_extra_fields = frozenset({"execution", "inputs", "preset"})
    contract_sections = {
        "inputs": frozenset({"config", "dataset_name", "index"}),
        "preset": PRESET_FIELDS,
    }
    extra_decision_fields = frozenset({"config"})
    decision_recipe_targets = {
        "overwrite_policy": ("preset", "overwrite"),
        "required_channels": ("preset", "channels"),
    }
    unsupported_variants = frozenset({"sex_age_baseline"})
    requires_survival_sidecars = True
    requires_multilabel_sidecars = True

    def required_input_paths(self, recipe: dict[str, Any]) -> list[tuple[str, Any]]:
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        return [("index", path) for path in inputs.get("index") or []]

    def task_issues(
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        decisions: dict[str, ResolvedDecision],
        high_impact: dict[str, dict[str, Any]],
    ) -> list[DecisionIssue]:
        issues: list[DecisionIssue] = []
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        preset = recipe.get("preset") if isinstance(recipe.get("preset"), dict) else {}

        for input_field, value in {
            "index": inputs.get("index"),
            "dataset_name": inputs.get("dataset_name"),
            "split": preset.get("split"),
            "n_tokens": preset.get("n_tokens"),
            "allow_missing_channels": preset.get("allow_missing_channels"),
        }.items():
            if value in (None, "", []):
                issues.append(
                    needs_issue(
                        input_field,
                        f"{input_field} is required for preset preparation.",
                        high_impact,
                        {"recipe": value},
                    )
                )
        if preset.get("allow_missing_channels") is True and preset.get("min_channels") is None:
            issues.append(
                needs_issue("min_channels", "min_channels is required when missing channels are allowed.", high_impact)
            )
        if recipe.get("variant") in {"sleep2vec2", "sleep2expert"}:
            if preset.get("manifest_output") not in (None, ""):
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        "manifest_output",
                        f"{recipe['variant']} preset preparation does not support manifest_output.",
                        None,
                        {"variant": recipe["variant"]},
                    )
                )
            if "write_sidecar_manifest" in preset:
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        "write_sidecar_manifest",
                        f"{recipe['variant']} preset preparation does not support write_sidecar_manifest.",
                        None,
                        {"variant": recipe["variant"]},
                    )
                )
        survival_issue = survival_sidecar_issue(
            self.task, recipe, config_summary, required=self.requires_survival_sidecars
        )
        if survival_issue is not None:
            issues.append(survival_issue)
        multilabel_issue = multilabel_sidecar_issue(
            self.task,
            recipe,
            config_summary,
            required=self.requires_multilabel_sidecars,
        )
        if multilabel_issue is not None:
            issues.append(multilabel_issue)
        return issues

    def commands(self, recipe: dict[str, Any], config_summary: dict[str, Any] | None) -> list[str]:
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        preset = recipe.get("preset") if isinstance(recipe.get("preset"), dict) else {}
        preset_script = {
            "sleep2vec": "preprocess/save_dataset_presets.py",
            "sleep2vec2": "sleep2vec2/preprocess/save_dataset_presets.py",
            "sleep2expert": "sleep2expert/preprocess/save_dataset_presets.py",
        }[str(recipe.get("variant"))]
        return [
            render_command(
                [
                    "python",
                    preset_script,
                    "--config",
                    inputs.get("config"),
                    "--index",
                    *coerce_list(inputs.get("index")),
                    "--dataset-name",
                    inputs.get("dataset_name"),
                    "--n-tokens",
                    preset.get("n_tokens"),
                    "--split",
                    *coerce_list(preset.get("split")),
                    *preset_cli_args(preset),
                ]
            )
        ]

    def index_summary_inputs_override(
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None
    ) -> tuple[list[Any], Any, list[Any]] | None:
        if recipe.get("task") != self.task:
            return None
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        preset = recipe.get("preset") if isinstance(recipe.get("preset"), dict) else {}
        return coerce_list(inputs.get("index")), inputs.get("config"), coerce_list(preset.get("split"))


PRESET_PREPARE_ADAPTER = PresetPrepareAdapter()
