from __future__ import annotations

import sys
from typing import Any

from ..decision_models import DecisionIssue, DecisionStatus, ResolvedDecision, needs_issue
from ..decision_paths import execution_contract_issues, multilabel_sidecar_issue, survival_sidecar_issue
from ..models import REPO_ROOT, coerce_list
from ..plan_rendering import PRESET_FIELDS, preset_cli_args, render_command
from ..repo import repo_summary
from .base import TaskAdapter


class PresetPrepareAdapter(TaskAdapter):
    task = "preset_prepare"
    supports_runtime_identity = True
    direct_launch_subcommand = "preset-launch"

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

    def frozen_command_prefix(self, recipe: dict[str, Any]) -> tuple[str, ...]:
        preset_script = {
            "sleep2vec": "preprocess/save_dataset_presets.py",
            "sleep2vec2": "sleep2vec2/preprocess/save_dataset_presets.py",
            "sleep2expert": "sleep2expert/preprocess/save_dataset_presets.py",
        }[str(recipe.get("variant"))]
        execution = recipe.get("execution") or {}
        # Historical frozen plans without runtime identity retain their original command.
        return (str(execution.get("python") or "python"), preset_script)

    def bind_effective_recipe(
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        *,
        source_recipe: dict[str, Any] | None = None,
    ) -> list[DecisionIssue]:
        issues: list[DecisionIssue] = []
        execution = recipe.get("execution") or {}
        # Bind only new effective recipes, never registered-plan reconstruction.
        if not {"python", "runtime_commit"}.intersection(execution):
            manager_runtime = (
                execution.get("target") in (None, "", "local")
                and execution.get("workdir") in (None, "", str(REPO_ROOT))
                and execution.get("path_context") != "remote"
            )
            if not manager_runtime:
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        "execution",
                        "Preset runtime identity cannot be inferred for this execution context. Provide "
                        "execution.python, execution.runtime_commit, and execution.workdir together for the "
                        "intended local runtime. This plan is not an SSH launcher; plan on the execution host.",
                        None,
                        {"execution": execution, "preflight_before_workspace": True},
                    )
                )
            else:
                repository = repo_summary()["git"]
                if not repository["available"] or not repository["commit"]:
                    issues.append(
                        DecisionIssue(
                            DecisionStatus.FAIL,
                            "execution.runtime_commit",
                            "Cannot freeze the preset runtime commit because the manager repository is unavailable.",
                            None,
                            {"preflight_before_workspace": True},
                        )
                    )
                else:
                    recipe["execution"] = {
                        **execution,
                        "python": sys.executable,
                        "runtime_commit": repository["commit"],
                        "workdir": str(REPO_ROOT),
                    }
                    # Binding follows source validation; generated identity must pass the same contract.
                    issues.extend(
                        execution_contract_issues(
                            recipe,
                            source_layer="effective",
                            supports_runtime_identity=self.supports_runtime_identity,
                            supports_direct=True,
                        )
                    )
        # Only new effective recipes acquire managed launch semantics; registered readers never bind defaults.
        recipe.setdefault("execution", {}).setdefault("scheduler", {"type": "direct"})
        preset_build = (config_summary or {}).get("preset_build") or {}
        if not preset_build:
            return issues

        preset = recipe.get("preset")
        if not isinstance(preset, dict):
            preset = {}
        decisions = recipe.get("decisions")
        if not isinstance(decisions, dict):
            decisions = {}
        source_preset = (source_recipe or {}).get("preset")
        if not isinstance(source_preset, dict):
            source_preset = {}
        source_decisions = (source_recipe or {}).get("decisions")
        if not isinstance(source_decisions, dict):
            source_decisions = {}
        for decision_field, preset_field, config_field in (
            ("required_channels", "channels", "required_channels"),
            ("min_channels", "min_channels", "min_channels"),
        ):
            config_value = preset_build.get(config_field)
            if config_value is None:
                continue
            recipe_value = preset.get(preset_field)
            raw_decision = decisions.get(decision_field)
            decision_value = raw_decision.get("value") if isinstance(raw_decision, dict) else raw_decision
            source_recipe_value = source_preset.get(preset_field)
            raw_source_decision = source_decisions.get(decision_field)
            source_decision_value = (
                raw_source_decision.get("value") if isinstance(raw_source_decision, dict) else raw_source_decision
            )
            if any(
                value not in (None, "", [], "ASK_USER") and value != config_value
                for value in (source_recipe_value, source_decision_value, recipe_value, decision_value)
            ):
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        decision_field,
                        f"{decision_field} differs from config preset_build.{config_field}.",
                        None,
                        {
                            "source_recipe": source_recipe_value,
                            "source_decision": source_decision_value,
                            "recipe": recipe_value,
                            "decision": decision_value,
                            "config": config_value,
                            "preflight_before_workspace": True,
                        },
                    )
                )
                continue
            preset.pop(preset_field, None)
        recipe["preset"] = preset
        return issues

    def required_input_paths(self, recipe: dict[str, Any]) -> list[tuple[str, Any]]:
        inputs = recipe.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
        return [("index", path) for path in inputs.get("index") or []]

    def task_issues(
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        decisions: dict[str, ResolvedDecision],
        high_impact: dict[str, dict[str, Any]],
    ) -> list[DecisionIssue]:
        issues: list[DecisionIssue] = []
        inputs = recipe.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
        preset = recipe.get("preset")
        if not isinstance(preset, dict):
            preset = {}

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
        config_min_channels = ((config_summary or {}).get("preset_build") or {}).get("min_channels")
        if (
            preset.get("allow_missing_channels") is True
            and preset.get("min_channels") is None
            and config_min_channels is None
        ):
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
        inputs = recipe.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
        preset = recipe.get("preset")
        if not isinstance(preset, dict):
            preset = {}
        return [
            render_command(
                [
                    *self.frozen_command_prefix(recipe),
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
        inputs = recipe.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
        preset = recipe.get("preset")
        if not isinstance(preset, dict):
            preset = {}
        return coerce_list(inputs.get("index")), inputs.get("config"), coerce_list(preset.get("split"))


PRESET_PREPARE_ADAPTER = PresetPrepareAdapter()
