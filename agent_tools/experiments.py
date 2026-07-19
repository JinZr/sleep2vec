from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from . import experiment_io as exp_io, experiment_tracking as tracking
from .experiment_workspace import (
    TERMINAL_STATUSES,
    canonical_local_experiment_root,
    commit_step_manifest,
    experiment_metadata_issues,
    experiment_readme_text,
    managed_run_key,
    merge_run_manifest,
    read_managed_yaml_mapping,
    read_run_manifest,
    validate_existing_experiment_manifest,
    validate_frozen_run_update,
    validate_managed_run_rows,
    write_initial_experiment_manifest,
)
from .manifests import utc_now


def init_experiment(run_dir: str | Path, spec_path: str | Path, *, remote: str | None = None) -> Path:
    root = _target_root(run_dir, remote)
    raw = read_managed_yaml_mapping(Path(spec_path).read_text(), source=f"Experiment spec {spec_path}")
    experiment = raw.get("experiment") if isinstance(raw, dict) and isinstance(raw.get("experiment"), dict) else raw
    if not isinstance(experiment, dict):
        raise ValueError("Experiment spec must be a YAML mapping.")
    experiment = dict(experiment)
    experiment["root"] = str(root)
    issues = experiment_metadata_issues(
        {
            "experiment": experiment,
            "step": {"id": "init", "phase": "prepare", "purpose": "initialize experiment"},
        }
    )
    if issues:
        raise ValueError("; ".join(issue["message"] for issue in issues))
    existing_text = exp_io.read_text_at(root / "experiment.yaml", remote=remote)
    if not remote and root.exists() and any(root.iterdir()) and not existing_text:
        raise ValueError(f"Experiment root is non-empty: {root}")
    if remote and not existing_text and exp_io.remote_dir_nonempty(root, remote):
        raise ValueError(f"Experiment root is non-empty: {root}")
    manifest = root / "experiment_manifest.tsv"
    rows = []
    if existing_text:
        validate_existing_experiment_manifest(existing_text, experiment, root)
        _managed_rows(root, remote=remote)
        rows = exp_io.read_rows_at(manifest, remote=remote, strict=True)
    exp_io.validate_managed_output_paths(
        root,
        [
            root / "experiment.yaml",
            root / "run_manifest.tsv",
            root / "events.jsonl",
            root / "README.md",
            manifest,
        ],
        remote=remote,
    )
    exp_io.mkdir_experiment_dirs(root, remote=remote)
    if not existing_text:
        write_initial_experiment_manifest(root, experiment, remote=remote)
        exp_io.append_event_at(root, "experiment_initialized", {"experiment_id": experiment["id"]}, remote=remote)
    exp_io.write_text_at(root / "README.md", experiment_readme_text(experiment), remote=remote)
    if rows:
        row = rows[0]
        row.update(
            {
                "experiment_id": experiment["id"],
                "experiment_root": str(root),
                "title": experiment["title"],
                "objective": experiment["objective"],
                "remote_host": remote or row.get("remote_host", ""),
                "updated_at": utc_now(),
            }
        )
    else:
        row = {
            "experiment_id": experiment["id"],
            "experiment_root": str(root),
            "title": experiment["title"],
            "objective": experiment["objective"],
            "remote_host": remote or "",
            "task": "",
            "selection_metric": "",
            "selection_mode": "",
            "wandb_entity": "",
            "wandb_project": "",
            "wandb_group": "",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    exp_io.write_rows_at(manifest, [row], remote=remote)
    return manifest


def register_experiment_step(run_dir: str | Path, spec_path: str | Path, *, remote: str | None = None) -> Path:
    root = _target_root(run_dir, remote)
    _managed_rows(root, remote=remote)
    experiment_text = exp_io.read_text_at(root / "experiment.yaml", remote=remote)
    experiment_manifest = (
        read_managed_yaml_mapping(experiment_text, source=f"Managed experiment manifest {root / 'experiment.yaml'}")
        if experiment_text
        else {}
    )
    experiment = experiment_manifest.get("experiment") if isinstance(experiment_manifest, dict) else None
    if not isinstance(experiment, dict):
        raise ValueError("experiment.yaml is missing. Initialize the experiment first.")
    raw = read_managed_yaml_mapping(Path(spec_path).read_text(), source=f"Step spec {spec_path}")
    step = raw.get("step") if isinstance(raw, dict) and isinstance(raw.get("step"), dict) else raw
    if not isinstance(step, dict):
        raise ValueError("Step spec must be a YAML mapping.")
    for field in ("id", "phase", "purpose", "inputs", "outputs"):
        if step.get(field) in (None, "", "ASK_USER"):
            raise ValueError(f"step.{field} is required.")
    issues = experiment_metadata_issues({"experiment": experiment, "step": step}, allow_step_io=True)
    if issues:
        raise ValueError("; ".join(issue["message"] for issue in issues))
    path = root / "steps" / str(step["id"]) / "step.yaml"
    exp_io.validate_managed_output_paths(root, [path, root / "events.jsonl"], remote=remote)
    _merged, created = commit_step_manifest(
        root,
        {"step": step, "experiment_id": experiment["id"], "recipe_path": "", "plans": []},
        remote=remote,
    )
    if created:
        exp_io.append_event_at(root, "step_registered", {"step_id": step["id"], "phase": step["phase"]}, remote=remote)
    return path


def finalize_experiment(run_dir: str | Path, report_path: str | Path, *, remote: str | None = None) -> Path:
    if remote and not Path(report_path).is_absolute():
        raise ValueError("Remote final report path must be absolute.")
    root = _target_root(run_dir, remote)
    rows = _managed_rows(root, remote=remote)
    if not rows:
        raise ValueError("Experiment has no managed runs to finalize.")
    unresolved = [row["run_id"] for row in rows if row.get("status") not in TERMINAL_STATUSES]
    if unresolved:
        raise ValueError(f"Experiment still has unresolved runs: {unresolved}")
    report_text = exp_io.read_text_at(report_path, remote=remote)
    if not report_text.strip():
        raise ValueError("Final report is missing or empty.")
    manifest_text = exp_io.read_text_at(root / "experiment.yaml", remote=remote)
    manifest = read_managed_yaml_mapping(
        manifest_text, source=f"Managed experiment manifest {root / 'experiment.yaml'}"
    )
    if not isinstance(manifest.get("experiment"), dict):
        raise ValueError("experiment.yaml is missing.")
    target = root / "reports" / "final.md"
    exp_io.validate_managed_output_paths(
        root,
        [target, root / "experiment.yaml", root / "events.jsonl"],
        remote=remote,
    )
    target_exists = exp_io.path_exists_at(target, remote=remote)
    target_text = exp_io.read_text_at(target, remote=remote) if target_exists else ""
    target_sha256 = hashlib.sha256(target_text.encode()).hexdigest() if target_exists else None
    if not exp_io.conditional_atomic_replace_text_at(target, report_text, target_sha256, remote=remote):
        raise RuntimeError(f"Final report changed during publication: {target}")
    manifest["experiment"]["status"] = "completed"
    manifest["experiment"]["completed_at"] = utc_now()
    # The experiment manifest is the terminal commit, so publish it only after the report is durable.
    if not exp_io.conditional_atomic_replace_text_at(
        root / "experiment.yaml",
        yaml.safe_dump(manifest, sort_keys=False),
        hashlib.sha256(manifest_text.encode()).hexdigest(),
        remote=remote,
    ):
        raise RuntimeError("Experiment manifest changed during finalization.")
    exp_io.append_event_at(root, "experiment_finalized", {"report": str(target)}, remote=remote)
    return target


def sync_wandb_runs(
    run_dir: str | Path,
    *,
    entity: str,
    project: str,
    group: str | None = None,
    remote: str | None = None,
) -> Path:
    root = _target_root(run_dir, remote)
    managed_rows = _managed_rows(root, remote=remote)
    existing_metrics = exp_io.read_rows_at(root / "metrics_manifest.tsv", remote=remote, require_managed_identity=True)
    _validate_evidence_rows(managed_rows, existing_metrics, "metrics_manifest.tsv")
    blocked = root / "reports" / "wandb_blocked.md"
    exp_io.validate_managed_output_paths(
        root,
        [
            blocked,
            root / "wandb" / "summaries.jsonl",
            root / "wandb" / "runs.tsv",
            root / "metrics_manifest.tsv",
            root / "experiment_manifest.tsv",
            root / "reports" / "wandb.md",
            root / "run_manifest.tsv",
            root / "run_matrix.csv",
            root / "reports" / "run_matrix.md",
            root / "events.jsonl",
        ],
        remote=remote,
    )
    try:
        runs = tracking.wandb_runs(entity, project, group)
    except Exception as exc:
        exp_io.write_text_at(blocked, f"# W&B Sync Blocked\n\n{type(exc).__name__}: {exc}\n", remote=remote)
        raise RuntimeError(f"W&B sync blocked; wrote {blocked}") from exc

    payloads = [tracking.wandb_run_payload(run, entity=entity, project=project) for run in runs]
    run_rows = [payload["run_row"] for payload in payloads]
    metric_rows = [row for payload in payloads for row in payload["metric_rows"]]
    summary_lines = [payload["summary_line"] for payload in payloads]
    observations = tracking.wandb_run_observations(managed_rows, run_rows)
    managed_metrics = tracking.managed_metric_rows(managed_rows, metric_rows)
    merged_metrics = tracking.merge_rows(existing_metrics, managed_metrics)
    exp_io.validate_managed_output_paths(
        root,
        [root / "wandb" / "history" / payload["history_filename"] for payload in payloads],
        remote=remote,
    )

    exp_io.mkdir_experiment_dirs(root, remote=remote)
    for payload in payloads:
        tracking.write_history_csv(
            root / "wandb" / "history" / payload["history_filename"],
            payload["history_rows"],
            remote=remote,
        )

    exp_io.write_text_at(
        root / "wandb" / "summaries.jsonl",
        "\n".join(summary_lines) + ("\n" if summary_lines else ""),
        remote=remote,
    )
    exp_io.write_rows_at(root / "wandb" / "runs.tsv", run_rows, remote=remote)
    metrics_path = root / "metrics_manifest.tsv"
    if merged_metrics:
        exp_io.write_rows_at(metrics_path, merged_metrics, remote=remote)
    else:
        exp_io.write_text_at(metrics_path, "step_id\trun_id\n", remote=remote)
    tracking.update_experiment_wandb(root, entity=entity, project=project, group=group or "", remote=remote)
    tracking.write_wandb_report(root, run_rows, remote=remote)
    merge_run_manifest(root, observations, remote=remote)
    return root / "wandb" / "runs.tsv"


def index_checkpoints(run_dir: str | Path, *, remote: str | None = None) -> Path:
    root = _target_root(run_dir, remote)
    managed_rows = _managed_rows(root, remote=remote)
    metrics_path = root / "metrics_manifest.tsv"
    checkpoint_path = root / "checkpoint_manifest.tsv"
    # Workspace location proves evidence ownership only when the table path is not an alias.
    exp_io.validate_managed_output_paths(root, [metrics_path, checkpoint_path], remote=remote)
    metrics = exp_io.read_rows_at(metrics_path, remote=remote, require_managed_identity=True)
    _validate_evidence_rows(managed_rows, metrics, "metrics_manifest.tsv")
    rows = tracking.checkpoint_rows(root, remote=remote)
    for row in rows:
        row.update(tracking.best_metric_for_checkpoint(row, metrics))
    validate_managed_run_rows(rows, source="checkpoint_manifest.tsv", cardinality="many_per_run")
    if rows:
        exp_io.write_rows_at(checkpoint_path, rows, remote=remote)
    else:
        exp_io.write_text_at(checkpoint_path, "step_id\trun_id\n", remote=remote)
    return root / "checkpoint_manifest.tsv"


def monitor_experiment(run_dir: str | Path, *, remote: str | None = None) -> dict[str, Any]:
    root = _target_root(run_dir, remote)
    previous_rows = _managed_rows(root, remote=remote)
    report_path = root / "reports" / "monitor.md"
    exp_io.validate_managed_output_paths(
        root,
        [
            report_path,
            root / "run_manifest.tsv",
            root / "run_matrix.csv",
            root / "reports" / "run_matrix.md",
            root / "events.jsonl",
        ],
        remote=remote,
    )
    run_rows = tracking.experiment_run_rows(root, remote=remote)
    observations = [tracking.monitor_run_row(root, row, previous_rows, remote=remote) for row in run_rows]
    committed = merge_run_manifest(root, observations, remote=remote)
    report = tracking.monitor_report(committed)
    exp_io.write_text_at(report_path, report, remote=remote)
    return {"run_dir": str(root), "runs": committed, "report": str(report_path)}


def rank_experiment_candidates(run_dir: str | Path, *, metric: str, mode: str, remote: str | None = None) -> Path:
    root = _target_root(run_dir, remote)
    managed_rows = _managed_rows(root, remote=remote)
    out = root / "reports" / "experiment_ranking.csv"
    exp_io.validate_managed_output_paths(
        root,
        [
            out,
            root / "reports" / "experiment_ranking.md",
            root / "metrics_manifest.tsv",
            root / "checkpoint_manifest.tsv",
        ],
        remote=remote,
    )
    metric_rows = exp_io.read_rows_at(root / "metrics_manifest.tsv", remote=remote, require_managed_identity=True)
    checkpoint_rows = exp_io.read_rows_at(
        root / "checkpoint_manifest.tsv", remote=remote, require_managed_identity=True
    )
    _validate_evidence_rows(managed_rows, metric_rows, "metrics_manifest.tsv")
    _validate_evidence_rows(
        managed_rows,
        checkpoint_rows,
        "checkpoint_manifest.tsv",
        checkpoint_evidence=True,
        remote=remote,
    )
    run_rows = tracking.experiment_run_rows(root, remote=remote)
    rows = tracking.candidate_rows(run_rows, metric_rows, metric)
    ranked = tracking.rank_candidates(rows, checkpoint_rows, mode=mode)
    validate_managed_run_rows(ranked, source="experiment_ranking.csv", cardinality="one_per_run")
    if ranked:
        exp_io.write_rows_at(out, ranked, remote=remote)
    else:
        exp_io.write_text_at(out, "step_id,run_id\n", remote=remote)
    tracking.write_rank_report(root, metric, mode, ranked, remote=remote)
    return out


def _target_root(run_dir: str | Path, remote: str | None) -> Path:
    root = Path(run_dir)
    if remote:
        if not root.is_absolute():
            raise ValueError("Remote experiment root must be an absolute path.")
        return root
    return canonical_local_experiment_root(root, Path.cwd())


def _managed_rows(root: Path, *, remote: str | None) -> list[dict[str, str]]:
    manifest_path = root / "experiment.yaml"
    exp_io.validate_managed_output_paths(root, [manifest_path], remote=remote)
    experiment_text = exp_io.read_text_at(manifest_path, remote=remote)
    if not experiment_text:
        raise ValueError("experiment.yaml is missing. Initialize the experiment first.")
    manifest = read_managed_yaml_mapping(experiment_text, source=f"Managed experiment manifest {manifest_path}")
    experiment = manifest.get("experiment") if isinstance(manifest, dict) else None
    issues = experiment_metadata_issues(
        {
            "experiment": experiment,
            "step": {"id": "preflight", "phase": "prepare", "purpose": "validate experiment workspace"},
        }
    )
    if issues:
        raise ValueError("; ".join(issue["message"] for issue in issues))
    if str(experiment["root"]) != str(root):
        raise ValueError(f"experiment.root differs from the target workspace: {root}")

    for legacy_path in (root / "trial_status.tsv", root / "adaptive" / "trial_registry.tsv"):
        if exp_io.path_exists_at(legacy_path, remote=remote):
            raise ValueError(f"Historical experiment artifacts are read-only: {legacy_path}")

    experiment_manifest = root / "experiment_manifest.tsv"
    if exp_io.path_exists_at(experiment_manifest, remote=remote):
        manifest_rows = exp_io.read_rows_at(experiment_manifest, remote=remote, strict=True)
        if len(manifest_rows) != 1:
            raise ValueError("experiment_manifest.tsv must contain exactly one row.")
        manifest_row = manifest_rows[0]
        if manifest_row.get("experiment_id") != experiment["id"]:
            raise ValueError("experiment_manifest.tsv belongs to a different experiment.")
        if manifest_row.get("experiment_root") != str(root):
            raise ValueError("experiment_manifest.tsv root differs from the target workspace.")

    rows = read_run_manifest(root, remote=remote)
    for row in rows:
        if row["experiment_id"] != experiment["id"]:
            raise ValueError("run_manifest.tsv contains a run owned by a different experiment.")
    return rows


def _validate_evidence_rows(
    managed_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    source: str,
    *,
    checkpoint_evidence: bool = False,
    remote: str | None = None,
) -> None:
    validate_managed_run_rows(evidence_rows, source=source, cardinality="many_per_run")
    managed_by_key = {managed_run_key(row): row for row in managed_rows}
    for row in evidence_rows:
        managed = managed_by_key.get(managed_run_key(row))
        if managed is None:
            raise ValueError(f"{source} contains a run outside the canonical manifest.")
        validate_frozen_run_update(managed, row)
    if checkpoint_evidence:
        tracking.validate_checkpoint_evidence_rows(managed_rows, evidence_rows, remote=remote)
