from __future__ import annotations

from typing import Any, NamedTuple

from .models import coerce_list


class GpuRuleIssue(NamedTuple):
    code: str
    field: str
    message: str
    warning: bool
    evidence: dict[str, Any]


def gpu_group_plan(
    execution: dict[str, Any],
    runtime: dict[str, Any],
    *,
    max_concurrent: int | None = None,
) -> tuple[list[list[Any]], list[GpuRuleIssue]]:
    devices = coerce_list(runtime.get("devices"))
    pool = coerce_list(execution.get("gpu_pool")) or devices
    if not pool:
        if "gpus_per_run" in execution:
            try:
                per_run_evidence: Any = int(execution["gpus_per_run"])
            except (TypeError, ValueError):
                per_run_evidence = execution["gpus_per_run"]
            return [], [
                GpuRuleIssue(
                    "empty_pool",
                    "execution.gpus_per_run",
                    "execution.gpus_per_run requires a non-empty execution.gpu_pool or runtime.devices.",
                    False,
                    {"gpus_per_run": per_run_evidence, "preflight_before_workspace": True},
                )
            ]
        return [], []
    if len({str(item) for item in pool}) != len(pool):
        pool_field = "execution.gpu_pool" if coerce_list(execution.get("gpu_pool")) else "runtime.devices"
        return [], [
            GpuRuleIssue(
                "duplicate_pool",
                pool_field,
                "The effective GPU pool must not contain duplicate GPU identifiers.",
                False,
                {"gpu_pool": pool},
            )
        ]
    per_run = int(execution["gpus_per_run"]) if "gpus_per_run" in execution else len(devices) or 1
    if per_run <= 0:
        return [], [
            GpuRuleIssue(
                "per_run_not_positive",
                "execution.gpus_per_run",
                "execution.gpus_per_run must be a positive integer.",
                False,
                {"gpus_per_run": per_run},
            )
        ]
    if per_run > len(pool):
        return [], [
            GpuRuleIssue(
                "per_run_exceeds_pool",
                "execution.gpus_per_run",
                "execution.gpus_per_run cannot exceed the effective GPU pool size.",
                False,
                {"gpus_per_run": per_run, "gpu_pool": pool},
            )
        ]
    if len(pool) % per_run != 0:
        return [], [
            GpuRuleIssue(
                "not_divisible",
                "execution.gpus_per_run",
                "The effective GPU pool must divide evenly into disjoint per-run GPU groups.",
                False,
                {"gpus_per_run": per_run, "gpu_pool": pool},
            )
        ]
    groups = [pool[index : index + per_run] for index in range(0, len(pool), per_run)]
    issues: list[GpuRuleIssue] = []
    if max_concurrent is not None and max_concurrent > len(groups):
        group_count = len(groups)
        issues.append(
            GpuRuleIssue(
                "oversubscribed",
                "execution.max_concurrent",
                (
                    f"execution.max_concurrent={max_concurrent} exceeds the {group_count} available GPU "
                    "group(s); GPU oversubscription is explicitly enabled."
                ),
                True,
                {"max_concurrent": max_concurrent, "gpu_group_count": group_count},
            )
        )
    return groups, issues
