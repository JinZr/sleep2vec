from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from data.whole_night_index import validate_whole_night_index

from ..decision_models import DecisionIssue, DecisionStatus, ResolvedDecision, needs_issue
from ..models import REPO_ROOT, SUPPORTED_VARIANTS, coerce_list, resolve_repo_path
from ..plan_rendering import render_command, variant_module
from .base import TaskAdapter


def _mapping(recipe: dict[str, Any], section: str) -> dict[str, Any]:
    value = recipe.get(section)
    return value if isinstance(value, dict) else {}


def _fail(field: str, message: str, value: Any = None) -> DecisionIssue:
    return DecisionIssue(
        DecisionStatus.FAIL,
        field,
        message,
        None,
        {"value": value, "preflight_before_workspace": True},
    )


class EmbeddingExtractionAdapter(TaskAdapter):
    task = "embedding_extraction"

    recipe_extra_fields = frozenset({"artifacts", "evaluation_policy", "extraction", "inputs", "runtime"})
    artifact_fields = frozenset({"embedding_dir", "overwrite"})
    contract_sections = {
        "inputs": frozenset({"ckpt_path", "config", "data_index", "eval_split"}),
        "extraction": frozenset(
            {"channels", "embedding_kind", "layer_index", "max_source_tokens", "output_format", "sequence_mode"}
        ),
        "evaluation_policy": frozenset({"external_test_locked", "final_test_unlocked"}),
    }
    decision_recipe_targets = {
        "embedding_channels": ("extraction", "channels"),
        "embedding_kind": ("extraction", "embedding_kind"),
        "layer_index": ("extraction", "layer_index"),
        "max_source_tokens": ("extraction", "max_source_tokens"),
        "output_format": ("extraction", "output_format"),
        "sequence_mode": ("extraction", "sequence_mode"),
    }
    extra_decision_fields = frozenset({"external_test_locked", "final_eval_unlock"})
    unsupported_variants = frozenset({"sex_age_baseline"})
    accepts_pretrain_config = True
    preflight_on_unresolved = True

    def runtime_fields(self, variant: Any) -> frozenset[str]:
        return frozenset({"device", "num_workers"})

    def required_input_paths(self, recipe: dict[str, Any]) -> list[tuple[str, Any]]:
        inputs = _mapping(recipe, "inputs")
        paths: list[tuple[str, Any]] = []
        checkpoint = inputs.get("ckpt_path")
        if checkpoint not in (None, "", "ASK_USER"):
            paths.append(("ckpt_path", checkpoint))
        for index, path in enumerate(coerce_list(inputs.get("data_index"))):
            paths.append((f"data_index[{index}]", path))
        return paths

    def frozen_input_paths(self, recipe: dict[str, Any]) -> list[tuple[str, Path]]:
        paths: list[tuple[str, Path]] = []
        for field, path in self.required_input_paths(recipe):
            resolved = resolve_repo_path(path)
            if resolved is not None:
                paths.append((f"inputs.{field}", resolved))
        return paths

    def task_issues(
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        decisions: dict[str, ResolvedDecision],
        high_impact: dict[str, dict[str, Any]],
    ) -> list[DecisionIssue]:
        issues: list[DecisionIssue] = []
        inputs = _mapping(recipe, "inputs")
        extraction = _mapping(recipe, "extraction")
        runtime = _mapping(recipe, "runtime")
        artifacts = _mapping(recipe, "artifacts")
        evaluation = _mapping(recipe, "evaluation_policy")
        model = (config_summary or {}).get("model") or {}
        model_channels = {
            item.get("name") for item in model.get("channels", []) if isinstance(item, dict) and item.get("name")
        }

        data_index = inputs.get("data_index")
        if data_index in (None, "", "ASK_USER") or not coerce_list(data_index):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "data_index",
                    "Whole-night extraction requires a non-empty inputs.data_index.",
                    "Which NPZ index CSV should whole-night extraction use?",
                    {"value": data_index},
                )
            )

        channels = extraction.get("channels")
        if channels not in (None, "", "ASK_USER"):
            channel_list = coerce_list(channels)
            if not channel_list or any(not isinstance(item, str) or not item for item in channel_list):
                issues.append(
                    _fail("extraction.channels", "extraction.channels must contain non-empty strings.", channels)
                )
            elif len(channel_list) != len(set(channel_list)):
                issues.append(
                    _fail("extraction.channels", "extraction.channels must be non-empty and unique.", channels)
                )
            elif model_channels and (unknown := sorted(set(channel_list) - model_channels)):
                issues.append(
                    _fail(
                        "extraction.channels",
                        f"Extraction channels are absent from model.channels: {unknown}.",
                        channels,
                    )
                )

        fixed_values = {
            "embedding_kind": "both",
            "layer_index": -1,
            "output_format": "npz",
            "sequence_mode": "whole-night",
        }
        for field, expected in fixed_values.items():
            value = extraction.get(field)
            if value not in (None, "", "ASK_USER") and value != expected:
                issues.append(_fail(f"extraction.{field}", f"extraction.{field} must be {expected!r}.", value))

        max_source_tokens = extraction.get("max_source_tokens")
        if max_source_tokens not in (None, "", "ASK_USER") and (
            type(max_source_tokens) is not int or not 1 <= max_source_tokens <= 4095
        ):
            issues.append(
                _fail(
                    "extraction.max_source_tokens",
                    "extraction.max_source_tokens must be an integer in [1, 4095].",
                    max_source_tokens,
                )
            )

        device = runtime.get("device", "cuda")
        if device == "ASK_USER":
            issues.append(needs_issue("runtime.device", "runtime.device is unresolved.", high_impact))
        elif not isinstance(device, str) or not device:
            issues.append(_fail("runtime.device", "runtime.device must be a non-empty string.", device))
        num_workers = runtime.get("num_workers", 8)
        if num_workers == "ASK_USER":
            issues.append(needs_issue("runtime.num_workers", "runtime.num_workers is unresolved.", high_impact))
        elif type(num_workers) is not int or not 0 <= num_workers <= 8:
            issues.append(
                _fail("runtime.num_workers", "runtime.num_workers must be an integer in [0, 8].", num_workers)
            )

        embedding_dir = artifacts.get("embedding_dir")
        if embedding_dir in (None, "", "ASK_USER"):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "artifacts.embedding_dir",
                    "artifacts.embedding_dir must be an explicit absolute path.",
                    "Which fresh absolute directory should receive the embeddings?",
                    {"value": embedding_dir},
                )
            )
        else:
            output = Path(str(embedding_dir))
            if str(embedding_dir).startswith("~") or not output.is_absolute():
                issues.append(
                    _fail(
                        "artifacts.embedding_dir",
                        "artifacts.embedding_dir must be an absolute path without ~ shorthand.",
                        embedding_dir,
                    )
                )
            elif output.exists() and (not output.is_dir() or any(output.iterdir())):
                issues.append(
                    _fail(
                        "artifacts.embedding_dir",
                        f"Embedding output directory must be absent or empty: {embedding_dir}",
                        embedding_dir,
                    )
                )

        overwrite = artifacts.get("overwrite")
        if overwrite not in (None, "", "ASK_USER") and overwrite is not False:
            issues.append(
                _fail("artifacts.overwrite", "Embedding extraction requires artifacts.overwrite: false.", overwrite)
            )

        if inputs.get("eval_split") == "test":
            if evaluation.get("external_test_locked") is not False:
                issues.append(
                    DecisionIssue(
                        DecisionStatus.NEEDS_USER_INPUT,
                        "external_test_locked",
                        "Test extraction requires external_test_locked=false.",
                        "Should the external test set be unlocked for this extraction?",
                        {"value": evaluation.get("external_test_locked")},
                    )
                )
            if evaluation.get("final_test_unlocked") is not True:
                issues.append(
                    needs_issue("final_eval_unlock", "Test extraction requires explicit final unlock.", high_impact)
                )

        if config_summary is not None:
            if config_summary.get("data_backend") != "npz":
                issues.append(_fail("config", "Whole-night extraction requires config data.backend=npz."))
            backbone = model.get("backbone")
            if backbone not in (None, "roformer"):
                issues.append(_fail("config", "Whole-night extraction requires a RoFormer config.", backbone))
            cls_embedding_type = (model.get("cls") or {}).get("embedding_type")
            if cls_embedding_type != "bert":
                issues.append(
                    _fail(
                        "config",
                        "Whole-night dual embedding extraction requires model.cls.embedding_type=bert.",
                        cls_embedding_type,
                    )
                )
            if config_summary.get("is_finetune") is True:
                data = config_summary.get("data") or {}
                if data.get("finetune_preset_path"):
                    issues.append(
                        _fail(
                            "config",
                            "Whole-night extraction requires recipe-owned data_index; "
                            "config data.finetune_preset_path must be null.",
                            data.get("finetune_preset_path"),
                        )
                    )
                data_channels = set(data.get("data_channel_names") or [])
                if model_channels and data_channels and data_channels != model_channels:
                    issues.append(
                        _fail(
                            "config",
                            "Finetune config data.data_channel_names must match model.channels for "
                            "whole-night extraction.",
                            sorted(data_channels),
                        )
                    )
            issues.extend(self._strict_config_issues(recipe, config_summary))
        return issues

    def _strict_config_issues(self, recipe: dict[str, Any], config_summary: dict[str, Any]) -> list[DecisionIssue]:
        variant = recipe.get("variant")
        config_path = resolve_repo_path(_mapping(recipe, "inputs").get("config"))
        if variant not in SUPPORTED_VARIANTS or config_path is None or not config_path.is_file():
            return []
        if config_summary.get("is_finetune") is True:
            loader_name = "load_finetune_config"
        elif config_summary.get("is_pretrain") is True:
            loader_name = "load_pretrain_config"
        else:
            return [_fail("config", "Extraction config must be a pretrain or finetune model YAML.")]
        try:
            loader = getattr(importlib.import_module(f"{variant}.config"), loader_name)
            loader(config_path)
        except Exception as exc:
            return [_fail("config", f"Extraction config failed strict runtime loading: {exc}", str(config_path))]
        return []

    def configured_input_issues(
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None
    ) -> list[DecisionIssue]:
        inputs = _mapping(recipe, "inputs")
        extraction = _mapping(recipe, "extraction")
        raw_indices = inputs.get("data_index")
        eval_split = inputs.get("eval_split")
        max_source_tokens = extraction.get("max_source_tokens")
        if (
            raw_indices in (None, "", "ASK_USER")
            or eval_split in (None, "", "ASK_USER")
            or type(max_source_tokens) is not int
            or not 1 <= max_source_tokens <= 4095
        ):
            return []
        index_paths = [resolve_repo_path(path) for path in coerce_list(raw_indices)]
        if not index_paths or any(path is None or not path.is_file() for path in index_paths):
            return []
        try:
            data = (config_summary or {}).get("data") or {}
            source_field = "test_dataset_names" if eval_split == "test" else "train_dataset_names"
            validate_whole_night_index(
                [path for path in index_paths if path is not None],
                eval_split=str(eval_split),
                max_source_tokens=max_source_tokens,
                path_base=REPO_ROOT,
                sources=coerce_list(data.get(source_field)) if (config_summary or {}).get("is_finetune") else (),
            )
        except (OSError, ValueError) as exc:
            return [_fail("inputs.data_index", str(exc), raw_indices)]
        return []

    def preflight_issues(
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        *,
        unlock_final_test: bool,
        output_dir: Path | None = None,
    ) -> list[DecisionIssue]:
        embedding_dir = _mapping(recipe, "artifacts").get("embedding_dir")
        if embedding_dir in (None, "", "ASK_USER") or output_dir is None:
            return []
        embedding_path = Path(str(embedding_dir))
        if not embedding_path.is_absolute():
            return []
        embedding_path = embedding_path.resolve()
        plan_path = output_dir.resolve()
        try:
            plan_path.relative_to(embedding_path)
            overlaps = True
        except ValueError:
            try:
                embedding_path.relative_to(plan_path)
                overlaps = True
            except ValueError:
                overlaps = False
        if not overlaps:
            return []
        return [
            _fail(
                "artifacts.embedding_dir",
                "Embedding output and agent plan directories must not contain one another.",
                embedding_dir,
            )
        ]

    def commands(self, recipe: dict[str, Any], config_summary: dict[str, Any] | None) -> list[str]:
        inputs = _mapping(recipe, "inputs")
        extraction = _mapping(recipe, "extraction")
        runtime = _mapping(recipe, "runtime")
        artifacts = _mapping(recipe, "artifacts")
        return [
            render_command(
                [
                    "python",
                    "-m",
                    variant_module(recipe, "extract_embeddings"),
                    "--config",
                    inputs.get("config"),
                    "--ckpt-path",
                    inputs.get("ckpt_path"),
                    "--output-dir",
                    artifacts.get("embedding_dir"),
                    "--output-format",
                    "npz",
                    "--embedding-kind",
                    "both",
                    "--layer-index",
                    -1,
                    "--batch-size",
                    1,
                    "--num-workers",
                    runtime.get("num_workers", 8),
                    "--device",
                    runtime.get("device", "cuda"),
                    "--channels",
                    *coerce_list(extraction.get("channels")),
                    "--sequence-mode",
                    "whole-night",
                    "--max-source-tokens",
                    extraction.get("max_source_tokens"),
                    "--data-backend",
                    "npz",
                    "--data-index",
                    *coerce_list(inputs.get("data_index")),
                    "--eval-split",
                    inputs.get("eval_split"),
                ]
            )
        ]

    def expected_artifacts(self, recipe: dict[str, Any], config_summary: dict[str, Any] | None) -> list[dict[str, str]]:
        embedding_dir = _mapping(recipe, "artifacts").get("embedding_dir")
        if embedding_dir in (None, "", "ASK_USER"):
            return []
        return [{"name": "embedding_manifest", "path": str(Path(str(embedding_dir)) / "manifest.json")}]

    def index_summary_inputs_override(
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None
    ) -> tuple[list[Any], Any, list[Any]] | None:
        if recipe.get("task") != self.task:
            return None
        return [], _mapping(recipe, "inputs").get("config"), []


EMBEDDING_EXTRACTION_ADAPTER = EmbeddingExtractionAdapter()
