"""Declared external-matrix and cohort-selection pipeline validation.

Layer 0 leaf behind the ``experiment_pipeline`` orchestrator. Validates the
parsed spec, including resolved source-plan containment in the experiment root,
before workspace publication or run registration. Does not read run state,
publish plans, or call the scheduler.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

from .models import is_full_git_object_id

PIPELINE_KIND = "external_matrix"
COHORT_SELECTION_KIND = "cohort_selection"

_COMMON_TOP_LEVEL_FIELDS = {
    "pipeline",
    "runtime",
    "execution",
    "evaluation_policy",
    "checkpoint_policy",
    "checkpoint_sources",
    "jobs",
}
_TOP_LEVEL_FIELDS = _COMMON_TOP_LEVEL_FIELDS | {"schema_version"}
_COHORT_TOP_LEVEL_FIELDS = _COMMON_TOP_LEVEL_FIELDS | {"candidates", "selector"}
_PIPELINE_FIELDS = {"id", "kind", "experiment_id", "step", "finalize"}
_STEP_FIELDS = {"id", "phase", "purpose"}
_RUNTIME_FIELDS = {
    "workdir",
    "python",
    "runtime_commit",
    "accelerator",
    "device",
    "precision",
    "batch_size",
    "seed",
}
_EXECUTION_FIELDS = {"gpu_pool", "gpus_per_run", "max_concurrent", "max_attempts", "scheduler"}
_EVALUATION_FIELDS = {"external_test_locked", "final_test_unlocked"}
_CHECKPOINT_POLICY_FIELDS = {
    "avg_ckpts",
    "require_no_model_averaging",
    "forbidden_state_dict_prefixes",
    "require_ahi_eval_threshold",
}
_CHECKPOINT_SOURCE_FIELDS = {
    "plan",
    "selection_metric",
    "selection_mode",
    "task",
    "variant",
    "label_name",
}
_JOB_FIELDS = {
    "id",
    "checkpoint_source",
    "cohort",
    "modality",
    "inference_preset_path",
    "num_workers",
    "task",
    "variant",
    "label_name",
}
_COHORT_JOB_FIELDS = (_JOB_FIELDS - {"checkpoint_source"}) | {"role", "provenance"}
_CANDIDATE_FIELDS = {"kind", "count"}
_SELECTOR_FIELDS = {"strategy", "gates", "tie_breaker", "on_no_feasible"}
_GATE_FIELDS = {"job", "metric", "mode", "threshold", "strict"}


def validate_spec(spec: dict[str, Any], root: Path, *, unlock_final_test: bool | None) -> None:
    raw_pipeline = spec.get("pipeline")
    kind = raw_pipeline.get("kind") if isinstance(raw_pipeline, dict) else None
    legacy_version = spec.get("schema_version")
    if kind == PIPELINE_KIND and (type(legacy_version) is not int or legacy_version != 1):
        raise ValueError("Legacy external_matrix specs require schema_version: 1.")
    _reject_unknown_fields(
        spec,
        _COHORT_TOP_LEVEL_FIELDS if kind == COHORT_SELECTION_KIND else _TOP_LEVEL_FIELDS,
        "spec",
    )
    pipeline = _mapping(spec, "pipeline")
    _reject_unknown_fields(pipeline, _PIPELINE_FIELDS, "pipeline")
    pipeline_id = _required_slug(pipeline, "id", "pipeline")
    if kind not in {PIPELINE_KIND, COHORT_SELECTION_KIND}:
        raise ValueError(f"pipeline.kind must be {PIPELINE_KIND!r} or {COHORT_SELECTION_KIND!r}.")
    _required_slug(pipeline, "experiment_id", "pipeline")
    step = _mapping(pipeline, "step")
    _reject_unknown_fields(step, _STEP_FIELDS, "pipeline.step")
    _required_slug(step, "id", "pipeline.step")
    if step.get("phase") != "evaluate":
        raise ValueError("pipeline.step.phase must be 'evaluate'.")
    if not str(step.get("purpose") or "").strip():
        raise ValueError("pipeline.step.purpose is required.")
    if pipeline.get("finalize") is not True:
        raise ValueError("pipeline.finalize must be true.")

    _validate_runtime_execution(spec)

    evaluation = _mapping(spec, "evaluation_policy")
    _reject_unknown_fields(evaluation, _EVALUATION_FIELDS, "evaluation_policy")
    if evaluation.get("external_test_locked") is not False or evaluation.get("final_test_unlocked") is not True:
        raise ValueError("External pipeline spec must explicitly unlock final test evaluation.")
    if unlock_final_test is False:
        raise ValueError("External pipeline execution also requires --unlock-final-test.")

    checkpoint_policy = _mapping(spec, "checkpoint_policy")
    _reject_unknown_fields(checkpoint_policy, _CHECKPOINT_POLICY_FIELDS, "checkpoint_policy")
    if type(checkpoint_policy.get("avg_ckpts")) is not int or checkpoint_policy["avg_ckpts"] != 1:
        raise ValueError("checkpoint_policy.avg_ckpts must be 1.")
    if checkpoint_policy.get("require_no_model_averaging") is not True:
        raise ValueError("checkpoint_policy.require_no_model_averaging must be true.")
    prefixes = checkpoint_policy.get("forbidden_state_dict_prefixes")
    if (
        not isinstance(prefixes, list)
        or not prefixes
        or any(not isinstance(item, str) or not item for item in prefixes)
    ):
        raise ValueError("checkpoint_policy.forbidden_state_dict_prefixes must be a non-empty string list.")
    if not {"ema_model.", "running_mean_model."}.issubset(prefixes):
        raise ValueError("checkpoint_policy.forbidden_state_dict_prefixes must include EMA and running-mean keys.")
    if checkpoint_policy.get("require_ahi_eval_threshold") is not True:
        raise ValueError("checkpoint_policy.require_ahi_eval_threshold must be true.")

    _validate_pipeline_sources_and_jobs(spec, root=root, kind=kind)
    if kind == COHORT_SELECTION_KIND:
        _validate_cohort_selection_contract(spec)
    if not pipeline_id:
        raise AssertionError("validated pipeline id is empty")


def _validate_runtime_execution(spec: dict[str, Any]) -> None:
    runtime = _mapping(spec, "runtime")
    _reject_unknown_fields(runtime, _RUNTIME_FIELDS, "runtime")
    for field in ("workdir", "runtime_commit"):
        value = runtime.get(field)
        if not isinstance(value, str) or not value.strip() or value == "ASK_USER":
            raise ValueError(f"runtime.{field} must be an explicit non-empty string.")
    python_command = runtime.get("python")
    if (
        not isinstance(python_command, str)
        or not python_command.strip()
        or python_command == "ASK_USER"
        or python_command.startswith("~")
        or re.search(r"\s", python_command) is not None
    ):
        raise ValueError(
            "runtime.python must be a single executable name or path without whitespace, arguments, or ~ shorthand."
        )
    for field in ("accelerator", "device", "precision"):
        if runtime.get(field) in (None, ""):
            raise ValueError(f"runtime.{field} is required.")
    if not Path(runtime["workdir"]).is_absolute():
        raise ValueError("runtime.workdir must be absolute.")
    if not is_full_git_object_id(runtime["runtime_commit"]):
        raise ValueError("runtime.runtime_commit must be a full lowercase 40-character Git commit ID.")
    if runtime.get("accelerator") != "gpu" or runtime.get("device") != "cuda":
        raise ValueError("Managed evaluation pipelines require GPU/CUDA runtime.")
    if str(runtime.get("precision")) not in {"32", "32-true"}:
        raise ValueError("Managed evaluation pipelines require FP32 precision.")
    if type(runtime.get("batch_size")) is not int or runtime["batch_size"] != 128:
        raise ValueError("Managed evaluation pipelines require runtime.batch_size=128.")
    if isinstance(runtime.get("seed"), bool) or not isinstance(runtime.get("seed"), int):
        raise ValueError("runtime.seed must be an integer.")

    execution = _mapping(spec, "execution")
    _reject_unknown_fields(execution, _EXECUTION_FIELDS, "execution")
    if "scheduler" in execution:
        scheduler = _mapping(execution, "scheduler")
        _reject_unknown_fields(scheduler, {"type"}, "execution.scheduler")
        if scheduler.get("type") != "direct":
            raise ValueError("Managed evaluation pipeline supports only execution.scheduler.type=direct.")
    gpu_pool = execution.get("gpu_pool")
    if (
        not isinstance(gpu_pool, list)
        or not gpu_pool
        or any(isinstance(item, bool) or not isinstance(item, int) for item in gpu_pool)
    ):
        raise ValueError("execution.gpu_pool must be a non-empty list of GPU integers.")
    if len(gpu_pool) != len(set(gpu_pool)):
        raise ValueError("execution.gpu_pool contains duplicate GPUs.")
    if type(execution.get("gpus_per_run")) is not int or execution["gpus_per_run"] != 1:
        raise ValueError("Managed evaluation pipelines require execution.gpus_per_run=1.")
    max_concurrent = execution.get("max_concurrent")
    if (
        isinstance(max_concurrent, bool)
        or not isinstance(max_concurrent, int)
        or not 1 <= max_concurrent <= len(gpu_pool)
    ):
        raise ValueError("execution.max_concurrent must be between 1 and the GPU pool size.")
    if type(execution.get("max_attempts")) is not int or execution["max_attempts"] != 2:
        raise ValueError("Managed evaluation pipelines require execution.max_attempts=2.")


def _validate_pipeline_sources_and_jobs(spec: dict[str, Any], *, root: Path, kind: str) -> None:
    sources = _mapping(spec, "checkpoint_sources")
    if not sources:
        raise ValueError("checkpoint_sources must not be empty.")
    for source_id, source in sources.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", str(source_id)):
            raise ValueError(f"Invalid checkpoint source id: {source_id}")
        if not isinstance(source, dict):
            raise ValueError(f"checkpoint_sources.{source_id} must be a mapping.")
        _reject_unknown_fields(source, _CHECKPOINT_SOURCE_FIELDS, f"checkpoint_sources.{source_id}")
        plan = Path(str(source.get("plan") or ""))
        if not plan.is_absolute():
            raise ValueError(f"checkpoint_sources.{source_id}.plan must be absolute.")
        try:
            plan.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"checkpoint_sources.{source_id}.plan must be inside the experiment root.") from exc
        if not str(source.get("selection_metric") or ""):
            raise ValueError(f"checkpoint_sources.{source_id}.selection_metric is required.")
        if source.get("selection_mode") not in {"min", "max"}:
            raise ValueError(f"checkpoint_sources.{source_id}.selection_mode must be min or max.")

    if kind == COHORT_SELECTION_KIND and len(sources) != 1:
        raise ValueError("cohort_selection requires exactly one checkpoint source.")

    jobs = spec.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a non-empty list.")
    seen = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"jobs[{index}] must be a mapping.")
        _reject_unknown_fields(
            job,
            _COHORT_JOB_FIELDS if kind == COHORT_SELECTION_KIND else _JOB_FIELDS,
            f"jobs[{index}]",
        )
        job_id = _required_slug(job, "id", f"jobs[{index}]")
        if job_id in seen:
            raise ValueError(f"Duplicate external job id: {job_id}")
        seen.add(job_id)
        if kind == PIPELINE_KIND:
            source_id = str(job.get("checkpoint_source") or "")
            if source_id not in sources:
                raise ValueError(f"jobs[{index}].checkpoint_source is unknown: {source_id}")
        else:
            if job.get("role") not in {"selection", "report_only"}:
                raise ValueError(f"jobs[{index}].role must be selection or report_only.")
            if job.get("provenance") not in {"internal", "external"}:
                raise ValueError(f"jobs[{index}].provenance must be internal or external.")
            if job["role"] == "report_only" and job["provenance"] != "external":
                raise ValueError(f"jobs[{index}].provenance must be external for report_only jobs.")
        for field in ("cohort", "modality"):
            if not str(job.get(field) or "").strip():
                raise ValueError(f"jobs[{index}].{field} is required.")
        preset = Path(str(job.get("inference_preset_path") or ""))
        if not preset.is_absolute():
            raise ValueError(f"jobs[{index}].inference_preset_path must be absolute.")
        workers = job.get("num_workers")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
            raise ValueError(f"jobs[{index}].num_workers must be a non-negative integer.")
        expected_workers = {"psg": 8, "bcg": 16}.get(str(job["modality"]).lower())
        if expected_workers is not None and workers != expected_workers:
            raise ValueError(f"jobs[{index}].num_workers must be {expected_workers} for {job['modality']} inference.")


def _validate_cohort_selection_contract(spec: dict[str, Any]) -> None:
    candidates = _mapping(spec, "candidates")
    _reject_unknown_fields(candidates, _CANDIDATE_FIELDS, "candidates")
    if candidates.get("kind") == "top_k":
        count = candidates.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("candidates.count must be a positive integer for candidates.kind=top_k.")
    elif candidates.get("kind") == "all":
        if "count" in candidates:
            raise ValueError("candidates.count is not allowed for candidates.kind=all.")
    else:
        raise ValueError("candidates.kind must be top_k or all.")

    roles = {job["role"] for job in spec["jobs"]}
    if "selection" not in roles:
        raise ValueError("cohort_selection requires at least one selection job.")
    for first_index, first in enumerate(spec["jobs"]):
        for second in spec["jobs"][first_index + 1 :]:
            if first["role"] == second["role"]:
                continue
            if first["cohort"] == second["cohort"]:
                raise ValueError("The same cohort cannot be both selection and report_only.")
            if first["inference_preset_path"] == second["inference_preset_path"]:
                raise ValueError("The same preset cannot be both selection and report_only.")

    selector = _mapping(spec, "selector")
    _reject_unknown_fields(selector, _SELECTOR_FIELDS, "selector")
    expected = {
        "strategy": "target_gate",
        "tie_breaker": "internal_rank",
        "on_no_feasible": "no_winner",
    }
    for field, value in expected.items():
        if selector.get(field) != value:
            raise ValueError(f"selector.{field} must be {value!r}.")
    gates = selector.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("selector.gates must be a non-empty list.")
    jobs = {job["id"]: job for job in spec["jobs"]}
    seen_gates = set()
    referenced_jobs = set()
    for index, gate in enumerate(gates):
        if not isinstance(gate, dict):
            raise ValueError(f"selector.gates[{index}] must be a mapping.")
        _reject_unknown_fields(gate, _GATE_FIELDS, f"selector.gates[{index}]")
        job_id = str(gate.get("job") or "")
        metric = str(gate.get("metric") or "")
        if job_id not in jobs or jobs[job_id]["role"] != "selection":
            raise ValueError(f"selector.gates[{index}].job must identify a selection job.")
        if not metric:
            raise ValueError(f"selector.gates[{index}].metric is required.")
        if gate.get("mode") not in {"min", "max"}:
            raise ValueError(f"selector.gates[{index}].mode must be min or max.")
        if "strict" in gate and not isinstance(gate["strict"], bool):
            raise ValueError(f"selector.gates[{index}].strict must be a boolean.")
        threshold = gate.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError(f"selector.gates[{index}].threshold must be finite.")
        identity = (job_id, metric)
        if identity in seen_gates:
            raise ValueError(f"Duplicate selector gate: {job_id} / {metric}")
        seen_gates.add(identity)
        referenced_jobs.add(job_id)
    selection_jobs = {job["id"] for job in spec["jobs"] if job["role"] == "selection"}
    if referenced_jobs != selection_jobs:
        raise ValueError("Every selection job must contribute at least one selector gate.")


def _mapping(payload: dict[str, Any], field: str) -> dict[str, Any]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping.")
    return value


def _reject_unknown_fields(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} field(s): {', '.join(unknown)}")


def _required_slug(payload: dict[str, Any], field: str, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
        raise ValueError(f"{label}.{field} must use lowercase letters, digits, hyphens, and underscores.")
    return value
