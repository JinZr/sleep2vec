"""The ``preset_prepare`` task adapter.

Dataset-preset preparation: preset field rendering, the survival and multilabel
sidecar checks, and the execution contract for the preparation step.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from ..decision_models import DecisionIssue, DecisionStatus, ResolvedDecision, needs_issue
from ..decision_paths import execution_contract_issues, multilabel_sidecar_issue, survival_sidecar_issue
from ..models import REPO_ROOT, ConfigSummaryInput, JsonValue, coerce_list
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
        recipe: dict[str, JsonValue],
        config_summary: ConfigSummaryInput | None,
        *,
        source_recipe: dict[str, Any] | None = None,
    ) -> list[DecisionIssue]:
        issues: list[DecisionIssue] = []
        execution_value = recipe.get("execution")
        execution = execution_value if isinstance(execution_value, dict) else {}
        # Bind only new effective recipes, never registered-plan reconstruction.
        if not {"python", "runtime_commit"}.intersection(execution):
            if execution.get("workdir") not in (None, "", str(REPO_ROOT)):
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        "execution",
                        "Preset runtime identity cannot be inferred for this execution context. Provide "
                        "execution.python, execution.runtime_commit, and execution.workdir together for the "
                        "intended local runtime.",
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
        effective_execution = recipe.setdefault("execution", {})
        if isinstance(effective_execution, dict):
            effective_execution.setdefault("scheduler", {"type": "direct"})
        preset_build: Any = (config_summary or {}).get("preset_build") or {}
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
        recipe: dict[str, JsonValue],
        config_summary: ConfigSummaryInput | None,
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
            "output_template": preset.get("output_template"),
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
        preset_build: Any = (config_summary or {}).get("preset_build") or {}
        config_min_channels = preset_build.get("min_channels")
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

    def commands(self, recipe: dict[str, Any], config_summary: ConfigSummaryInput | None) -> list[str]:
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

    def external_output_paths(self, recipe: dict[str, JsonValue]) -> list[Path]:
        # Mirrors save_dataset_presets.py output planning: one pickle per metadata variant and split, resolved
        # against the run workdir, plus the sleep2vec manifests; a dry run writes nothing.
        preset, inputs, execution = recipe["preset"], recipe["inputs"], recipe["execution"]
        assert isinstance(preset, dict) and isinstance(inputs, dict) and isinstance(execution, dict)
        if preset.get("dry_run") is True:
            return []
        n_tokens = preset["n_tokens"]
        stride_tokens = preset.get("stride_tokens")
        if stride_tokens is None:
            stride_tokens = 0 if n_tokens == 1535 else n_tokens
        assert isinstance(n_tokens, int) and isinstance(stride_tokens, int)
        splits = list(dict.fromkeys(coerce_list(preset["split"])))
        if 0 < stride_tokens < n_tokens and not preset.get("include_overlap_eval_splits"):
            splits = [split for split in splits if split not in {"val", "test"}]
        meta_names = list(dict.fromkeys(coerce_list(preset.get("meta_data_names"))))
        meta_variants: list[str | None] = [None] if not meta_names or preset.get("include_no_metadata") else []
        meta_variants.extend(meta_names)
        workdir = Path(str(execution.get("workdir") or REPO_ROOT))
        sidecars = recipe.get("variant") == "sleep2vec" and preset.get("write_sidecar_manifest") is not False
        paths: list[Path] = []
        for meta in meta_variants:
            for split in splits:
                rendered = str(preset["output_template"]).format(
                    dataset=inputs["dataset_name"],
                    split=split,
                    tokens=n_tokens,
                    n_tokens=n_tokens,
                    meta=meta or "none",
                    meta_suffix=f"_{meta}" if meta else "",
                )
                path = workdir / Path(rendered).expanduser()
                paths.append(path)
                if sidecars:
                    paths.append(path.with_name(f"{path.name}.manifest.json"))
        if preset.get("manifest_output") not in (None, ""):
            paths.append(workdir / Path(str(preset["manifest_output"])).expanduser())
        return paths

    def index_summary_inputs_override(
        self, recipe: dict[str, Any], config_summary: ConfigSummaryInput | None
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
