from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import repo_relative, resolve_repo_path
from .base import TaskAdapter


def _looks_like_placeholder_path(value: str | Path | None) -> bool:
    if value in (None, ""):
        return True
    text = str(value).strip()
    lowered = text.lower()
    return (
        lowered in {"ask_user", "none", "null", "todo", "tbd", "placeholder"}
        or text.startswith("/path/to")
        or text.startswith("<")
        or "ASK_USER" in text
    )


def sleep2stat_config_summary(config_path: str | Path) -> dict[str, Any]:
    from sleep2stat.config import SUPPORTED_ANALYZER_TYPES, SUPPORTED_REDUCER_TYPES, load_config

    resolved = resolve_repo_path(config_path)
    if resolved is None:
        raise FileNotFoundError("Config path is required.")
    supported = {
        "supported_analyzer_types": sorted(SUPPORTED_ANALYZER_TYPES),
        "supported_reducer_types": sorted(SUPPORTED_REDUCER_TYPES),
    }
    try:
        cfg = load_config(resolved)
    except Exception as exc:
        return {
            "config_path": repo_relative(resolved),
            "is_sleep2stat": True,
            "data_backend": None,
            "sleep2stat": supported,
            "warnings": [],
            "blocking_issues": [str(exc)],
            "agent_risk_issues": [],
        }

    analyzers = []
    reducers = []
    agent_risk_issues = []
    for item in cfg.analyzers:
        analyzer = {
            "name": item.name,
            "type": item.type,
            "enabled": item.enabled,
            "namespace": item.namespace,
            "label_name": item.label_name,
            "config": str(item.config) if item.config else None,
            "ckpt_path": str(item.ckpt_path) if item.ckpt_path else None,
            "input_channels": list(item.input_channels),
            "stage_source": item.stage_source,
            "event_source": item.event_source,
        }
        analyzers.append(analyzer)
        if item.enabled and item.type == "sleep2vec_downstream":
            if _looks_like_placeholder_path(item.config):
                agent_risk_issues.append(
                    f"Analyzer {item.name} downstream config is missing or placeholder: {item.config}"
                )
            if _looks_like_placeholder_path(item.ckpt_path):
                agent_risk_issues.append(f"Analyzer {item.name} ckpt_path is missing or placeholder: {item.ckpt_path}")
    for item in cfg.reducers:
        reducers.append(
            {
                "name": item.name,
                "type": item.type,
                "enabled": item.enabled,
                "source": item.source,
                "left": item.left,
                "right": item.right,
                "age_prediction": item.age_prediction,
                "sex_prediction": item.sex_prediction,
                "metadata_age_column": item.metadata_age_column,
                "metadata_sex_column": item.metadata_sex_column,
                "options": dict(item.options),
            }
        )

    return {
        "config_path": repo_relative(resolved),
        "is_sleep2stat": True,
        "data_backend": cfg.data.backend,
        "sleep2stat": {
            "run": {
                "name": cfg.run.name,
                "output_dir": str(cfg.run.output_dir),
            },
            "data": {
                "backend": cfg.data.backend,
                "index": str(cfg.data.index) if cfg.data.index else None,
                "kaldi_data_root": str(cfg.data.kaldi_data_root) if cfg.data.kaldi_data_root else None,
                "kaldi_manifest": str(cfg.data.kaldi_manifest) if cfg.data.kaldi_manifest else None,
                "split": list(cfg.data.split),
                "metadata_columns": list(cfg.data.metadata_columns),
                "token_sec": cfg.data.token_sec,
                "max_tokens": cfg.data.max_tokens,
            },
            "analyzers": analyzers,
            "reducers": reducers,
            **supported,
            "outputs": {
                "write_global_tables": cfg.outputs.write_global_tables,
                "write_per_record": cfg.outputs.write_per_record,
                "compression": cfg.outputs.compression,
                "global_tables": dict(cfg.outputs.global_tables),
            },
        },
        "warnings": [],
        "blocking_issues": [],
        "agent_risk_issues": agent_risk_issues,
    }


class Sleep2statAdapter(TaskAdapter):
    task = "sleep2stat"
    requires_variant = False

    def matches_config_data(self, data: dict[str, Any]) -> bool:
        return {"run", "data", "signals", "analyzers", "reducers", "outputs"}.issubset(set(data))

    def config_summary(self, config_path: str | Path) -> dict[str, Any]:
        return sleep2stat_config_summary(config_path)


SLEEP2STAT_ADAPTER = Sleep2statAdapter()
