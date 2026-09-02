from __future__ import annotations

import csv
import io
import math
import os
from pathlib import Path
from typing import Any

from . import experiment_io as exp_io, experiment_pipeline_cohort_selection as cohort_selection
from .experiment_workspace import file_sha256
from .manifests import read_json, read_rows

UNCERTAIN_STATUSES = {"missing_pid", "unknown_remote"}
RETRYABLE_STATUSES = {"failed", "launch_failed"}


def logical_job_states(spec: dict[str, Any], attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    logical = []
    for job in spec["jobs"]:
        rows = sorted(
            [row for row in attempts if row["job_id"] == job["id"]],
            key=lambda row: int(row["attempt"]),
        )
        successful = next(
            (row for row in rows if str(row.get("verified") or "").lower() == "true"),
            None,
        )
        if successful is not None:
            status = "completed"
        elif any(row.get("status") in UNCERTAIN_STATUSES | {"stopped", "superseded"} for row in rows):
            status = "blocked"
        elif rows and rows[-1].get("retry_blocker") not in (None, ""):
            status = "blocked"
        elif rows and rows[-1].get("validation_error") not in (None, ""):
            status = "failed"
        elif rows and rows[-1].get("retry_preparation_error") not in (None, ""):
            status = "failed"
        elif (
            rows
            and int(rows[-1]["attempt"]) >= spec["execution"]["max_attempts"]
            and rows[-1].get("status") in RETRYABLE_STATUSES
        ):
            status = "failed"
        else:
            status = "running"
        projection = {
            "job_id": job["id"],
            "status": status,
            "attempt_count": len(rows),
            "successful_run_id": successful.get("run_id", "") if successful else "",
            "result_manifest": successful.get("result_manifest", "") if successful else "",
            "cohort": job["cohort"],
            "modality": job["modality"],
            "checkpoint_source": job["checkpoint_source"],
            "retry_preparation_error": rows[-1].get("retry_preparation_error", "") if rows else "",
        }
        for field in ("candidate_id", "job_template_id", "role", "provenance"):
            if field in job:
                projection[field] = job[field]
        logical.append(projection)
    return logical


def read_result_manifest(attempt: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    result_root = Path(str(attempt["result_root"]))
    if result_root.is_symlink() or not result_root.is_dir():
        raise ValueError(f"Inference result root is missing or aliased: {result_root}")
    for path in result_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Inference result tree contains a symlink: {path}")
    manifests = [path for path in result_root.rglob("run_manifest.json") if path.is_file() and not path.is_symlink()]
    if len(manifests) != 1:
        raise ValueError(f"Inference result root must contain exactly one run_manifest.json: {result_root}")
    manifest_path = manifests[0]
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"Inference result manifest is malformed: {manifest_path}")
    manifest_paths = manifest.get("paths")
    if not isinstance(manifest_paths, dict):
        raise ValueError("Inference result manifest paths must be a mapping.")
    resolved_result_root = result_root.resolve()
    for field, raw_path in manifest_paths.items():
        if raw_path in (None, ""):
            continue
        path = Path(str(raw_path))
        try:
            path.resolve().relative_to(resolved_result_root)
        except ValueError as exc:
            raise ValueError(f"Inference result manifest path escapes result_root: {field}") from exc
    if Path(str(manifest_paths.get("manifest_path") or "")).resolve() != manifest_path.resolve():
        raise ValueError("Inference result manifest path does not identify itself.")
    required_result_paths = {}
    for field in ("metrics_csv_path", "prediction_csv_path"):
        raw_path = manifest_paths.get(field)
        if raw_path in (None, ""):
            raise ValueError(f"Inference result manifest paths.{field} is required.")
        required_result_paths[field] = Path(str(raw_path))
    exp_io.validate_managed_output_paths(result_root, [manifest_path, *required_result_paths.values()])
    for field, path in required_result_paths.items():
        if not path.is_file():
            raise ValueError(f"Inference result manifest path is missing or not a regular file: {field}")
    return manifest_path, manifest


def validate_result_manifest(spec: dict[str, Any], attempt: dict[str, Any], run: dict[str, Any]) -> Path:
    manifest_path, manifest = read_result_manifest(attempt)
    expected_paths = {
        "config_path": Path(str(run["config"])),
        "checkpoint.input": Path(str(attempt["checkpoint"])),
        "checkpoint.resolved_path": Path(str(attempt["checkpoint"])),
        "runtime.inference_preset_path": Path(str(attempt["preset"])),
    }
    actual_values = {
        "config_path": manifest.get("config_path"),
        "checkpoint.input": (manifest.get("checkpoint") or {}).get("input"),
        "checkpoint.resolved_path": (manifest.get("checkpoint") or {}).get("resolved_path"),
        "runtime.inference_preset_path": (manifest.get("runtime") or {}).get("inference_preset_path"),
    }
    for field, expected in expected_paths.items():
        actual = actual_values[field]
        if actual in (None, "") or Path(str(actual)).resolve() != expected.resolve():
            raise ValueError(f"Inference result manifest differs from frozen {field}.")
    if file_sha256(run["config"]) != attempt["config_sha256"]:
        raise ValueError("Inference attempt config bytes differ from the selected source config.")
    if manifest.get("label_name") != attempt["label_name"] or manifest.get("eval_split") != "test":
        raise ValueError("Inference result manifest label or split differs from the frozen job.")
    checkpoint = manifest.get("checkpoint") or {}
    runtime = manifest.get("runtime") or {}
    if type(checkpoint.get("avg_ckpts")) is not int or checkpoint["avg_ckpts"] != 1:
        raise ValueError("Inference result manifest does not prove avg_ckpts=1.")
    expected_runtime = spec["runtime"]
    for field in ("batch_size", "accelerator"):
        if runtime.get(field) != expected_runtime[field]:
            raise ValueError(f"Inference result manifest runtime.{field} differs from the frozen job.")
    if str(runtime.get("precision")) != str(expected_runtime["precision"]):
        raise ValueError("Inference result manifest runtime.precision differs from the frozen job.")
    if runtime.get("devices") != [0]:
        raise ValueError("Inference child process must use logical device 0.")
    count = manifest.get("prediction_row_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("Inference result manifest prediction_row_count must be positive.")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Inference result manifest metrics must be a mapping.")
    expected_namespace = attempt["variant"]
    if manifest.get("namespace") != expected_namespace:
        raise ValueError("Inference result namespace differs from the source variant.")
    return manifest_path


def build_result_rows(
    spec: dict[str, Any],
    selections: dict[str, dict[str, Any]],
    successful: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = []
    metric_rows = []
    for job in spec["jobs"]:
        attempt = successful[job["id"]]
        manifest = read_json(attempt["result_manifest"])
        selection = cohort_selection.candidate_for_job(job, selections)
        metrics = manifest.get("metrics") or {}
        summary = {
            "job_id": job["id"],
            "cohort": job["cohort"],
            "modality": job["modality"],
            "label_name": selection["label_name"],
            "variant": selection["variant"],
            "attempt": attempt["attempt"],
            "step_id": attempt["step_id"],
            "run_id": attempt["run_id"],
            "checkpoint": selection["checkpoint"],
            "checkpoint_sha256": selection["checkpoint_sha256"],
            "selection_metric": selection["selection_metric"],
            "selection_score": selection["score"],
            "preset": job["inference_preset_path"],
            "runtime_commit": attempt["runtime_commit"],
            "result_root": attempt["result_root"],
            "result_manifest": attempt["result_manifest"],
            "prediction_row_count": manifest["prediction_row_count"],
        }
        if "candidate_id" in job:
            summary.update(
                {
                    "candidate_id": job["candidate_id"],
                    "source_rank": selection["source_rank"],
                    "job_template_id": job["job_template_id"],
                    "role": job["role"],
                    "provenance": job["provenance"],
                }
            )
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            rendered = render_scalar(value)
            summary[f"metric.{name}"] = rendered
            metric = {
                **{key: summary[key] for key in ("job_id", "cohort", "modality", "label_name")},
                "metric": name,
                "value": rendered,
            }
            for field in ("candidate_id", "source_rank", "job_template_id", "role", "provenance"):
                if field in summary:
                    metric[field] = summary[field]
            metric_rows.append(metric)
        summary_rows.append(summary)
    return summary_rows, metric_rows


def write_result_summary(
    pipeline_dir: Path,
    spec: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> Path:
    write_rows_atomic(pipeline_dir / "results.csv", summary_rows)
    write_rows_atomic(pipeline_dir / "metrics.csv", metric_rows)
    markdown = summary_markdown(spec, summary_rows, metric_rows)
    atomic_write_text(pipeline_dir / "summary.md", markdown)
    report = pipeline_dir / "final.md"
    atomic_write_text(report, markdown)
    return report


def aggregate_results(
    root: Path,
    pipeline_dir: Path,
    spec: dict[str, Any],
    selections: dict[str, dict[str, Any]],
    logical_jobs: list[dict[str, Any]],
) -> Path:
    if len(logical_jobs) != len(spec["jobs"]) or any(job["status"] != "completed" for job in logical_jobs):
        raise ValueError("Cannot aggregate an incomplete external matrix.")
    attempts = read_rows(pipeline_dir / "jobs.tsv", require_managed_identity=True)
    successful_rows = [row for row in attempts if str(row.get("verified") or "").lower() == "true"]
    successful = {row["job_id"]: row for row in successful_rows}
    if len(successful_rows) != len(spec["jobs"]) or len(successful) != len(spec["jobs"]):
        raise ValueError("External matrix does not have one verified success per logical job.")
    summary_rows, metric_rows = build_result_rows(spec, selections, successful)
    return write_result_summary(pipeline_dir, spec, summary_rows, metric_rows)


def selection_evidence(
    phase_dir: Path,
    spec: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    attempts = read_rows(phase_dir / "jobs.tsv", require_managed_identity=True)
    successful_rows = [row for row in attempts if str(row.get("verified") or "").lower() == "true"]
    successful = {str(row["job_id"]): row for row in successful_rows}
    if len(successful_rows) != len(spec["jobs"]) or len(successful) != len(spec["jobs"]):
        raise ValueError("Selection matrix does not have one verified success per logical job.")
    evidence = []
    for job in spec["jobs"]:
        attempt = successful[job["id"]]
        manifest_path = Path(str(attempt["result_manifest"]))
        manifest = read_json(manifest_path)
        metrics = manifest.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Selection result metrics are malformed: {job['id']}")
        candidate = cohort_selection.candidate_for_job(job, candidates)
        evidence.append(
            {
                "job_id": job["id"],
                "job_template_id": job["job_template_id"],
                "candidate_id": candidate["candidate_id"],
                "cohort": job["cohort"],
                "metrics": metrics,
                "result_manifest": str(manifest_path),
                "result_manifest_sha256": file_sha256(manifest_path),
            }
        )
    return evidence


def write_cohort_result_summary(
    pipeline_dir: Path,
    spec: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    winner: dict[str, Any],
    selection_spec: dict[str, Any],
    report_spec: dict[str, Any],
) -> Path:
    summary_rows = []
    metric_rows = []
    for phase_name, phase_spec in (("selection", selection_spec), ("report_only", report_spec)):
        attempts = read_rows(pipeline_dir / "phases" / phase_name / "jobs.tsv", require_managed_identity=True)
        successful_rows = [row for row in attempts if str(row.get("verified") or "").lower() == "true"]
        successful = {str(row["job_id"]): row for row in successful_rows}
        if len(successful_rows) != len(phase_spec["jobs"]) or len(successful) != len(phase_spec["jobs"]):
            raise ValueError(f"Cohort-selection {phase_name} phase is incomplete.")
        phase_summary, phase_metrics = build_result_rows(phase_spec, candidates, successful)
        summary_rows.extend(phase_summary)
        metric_rows.extend(phase_metrics)
    for row in summary_rows:
        row["is_winner"] = row["candidate_id"] == winner["candidate_id"]
    for row in metric_rows:
        row["is_winner"] = row["candidate_id"] == winner["candidate_id"]
    write_rows_atomic(pipeline_dir / "results.csv", summary_rows)
    write_rows_atomic(pipeline_dir / "metrics.csv", metric_rows)
    markdown = cohort_summary_markdown(pipeline_dir, spec, summary_rows, metric_rows, winner)
    atomic_write_text(pipeline_dir / "summary.md", markdown)
    report = pipeline_dir / "final.md"
    atomic_write_text(report, markdown)
    return report


def cohort_summary_markdown(
    pipeline_dir: Path,
    spec: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    winner: dict[str, Any],
) -> str:
    winner_path = pipeline_dir / "cohort_selection_winner.json"
    lines = [
        f"# Cohort Selection Pipeline: {spec['pipeline']['id']}",
        "",
        f"Status: completed ({len(summary_rows)} jobs)",
        "",
        f"Winner: `{winner['candidate_id']}` (internal rank {winner['source_rank']})",
        "",
    ]
    lines.extend([f"Frozen winner: `{winner_path}` (`{file_sha256(winner_path)}`)", ""])
    for role, title in (("selection", "Selection evidence"), ("report_only", "Report-only results")):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Job | Candidate | Cohort | Metric | Value |",
                "|---|---|---|---|---:|",
            ]
        )
        for row in metric_rows:
            if row["role"] == role:
                lines.append(
                    f"| {row['job_template_id']} | {row['candidate_id']} | {row['cohort']} | "
                    f"{row['metric']} | {row['value']} |"
                )
        lines.append("")
    return "\n".join(lines)


def render_scalar(value: int | float) -> int | float | str:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def summary_markdown(
    spec: dict[str, Any], summary_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]
) -> str:
    lines = [
        f"# External Evaluation Pipeline: {spec['pipeline']['id']}",
        "",
        f"Status: completed ({len(summary_rows)}/{len(spec['jobs'])} jobs)",
        "",
        "| Job | Cohort | Modality | Label | Attempt | Checkpoint | Result |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['job_id']} | {row['cohort']} | {row['modality']} | {row['label_name']} | "
            f"{row['attempt']} | `{row['checkpoint']}` | `{row['result_manifest']}` |"
        )
    lines.extend(["", "## Scalar metrics", "", "| Job | Metric | Value |", "|---|---|---:|"])
    for row in metric_rows:
        lines.append(f"| {row['job_id']} | {row['metric']} | {row['value']} |")
    return "\n".join(lines) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    created = False
    try:
        with temp.open("x") as file_obj:
            created = True
            file_obj.write(text)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp, path)
    except BaseException:
        if created:
            temp.unlink(missing_ok=True)
        raise


def write_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, render_rows(path, rows))


def render_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["run_id"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t" if path.suffix == ".tsv" else ",")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
