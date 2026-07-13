from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from . import experiment_io as exp_io, run_artifacts as artifacts
from .experiment_workspace import (
    append_event,
    experiment_root,
    managed_run_key,
    managed_run_parameters,
    merge_run_manifest,
    read_run_manifest,
    resolve_run_row,
    validate_frozen_run_update,
    validate_managed_run_rows,
)
from .manifests import read_json, read_rows, write_rows


def select_hparam_candidates(
    run_dir: str | Path,
    metric: str | None = None,
    mode: str | None = None,
) -> Path:
    root = Path(run_dir)
    plan = artifacts.read_hparam_plan(root)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    frozen_metric = evaluation.get("selection_metric")
    frozen_mode = evaluation.get("selection_mode")
    if metric not in (None, frozen_metric) or mode not in (None, frozen_mode):
        raise ValueError("hparam-select must use the selection metric and mode frozen in the recipe.")
    metric = str(frozen_metric or "")
    mode = str(frozen_mode or "")
    if not metric or mode not in {"min", "max"}:
        raise ValueError("Recipe must define evaluation_policy.selection_metric and selection_mode.")
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    canonical_rows = read_run_manifest(workspace)
    rows = []
    for run in plan["runs"]:
        canonical = resolve_run_row(canonical_rows, run)
        if canonical is None:
            raise ValueError(f"Managed run is missing from run_manifest.tsv: {run['step_id']} / {run['run_id']}")
        manifest_path = artifacts.find_run_manifest(run)
        manifest = read_json(manifest_path) if manifest_path else {}
        score = artifacts.metric_value(manifest, metric)
        ckpt = artifacts.fixed_checkpoint_path(manifest, Path(str(run["checkpoint_dir"])))
        rows.append(
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "run_name": run["run_name"],
                "parameter_summary": run.get("parameter_summary", ""),
                "version": run["version"],
                "metric": metric,
                "score": score,
                "config": run.get("config"),
                "checkpoint_path": ckpt,
                "run_manifest": str(manifest_path or ""),
                "status": canonical.get("status", ""),
                **managed_run_parameters(run),
            }
        )
    reverse = mode == "max"
    ranked = sorted(
        rows,
        key=lambda row: artifacts.sortable_score(row.get("score"), reverse),
        reverse=reverse,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    current_ranked = ranked
    out = workspace / "reports" / "ranking.csv"
    existing_ranked = read_rows(out, require_managed_identity=True)
    validate_managed_run_rows(existing_ranked, source=str(out), cardinality="one_per_run")
    validate_managed_run_rows(current_ranked, source="current ranking", cardinality="one_per_run")
    for row in existing_ranked:
        canonical = resolve_run_row(canonical_rows, row)
        if canonical is None:
            raise ValueError(
                f"Existing ranking row is outside the canonical manifest: "
                f"{row.get('step_id', '')} / {row.get('run_id', '')}"
            )
        validate_frozen_run_update(canonical, row)
    step_id = str((recipe.get("step") or {}).get("id") or "")
    current_keys = {managed_run_key(row) for row in current_ranked}
    preserved = [row for row in existing_ranked if managed_run_key(row) not in current_keys]
    step_ranked = [row for row in [*preserved, *current_ranked] if row.get("step_id") == step_id]
    step_ranked = sorted(
        step_ranked,
        key=lambda row: artifacts.sortable_score(row.get("score"), reverse),
        reverse=reverse,
    )
    for rank, row in enumerate(step_ranked, start=1):
        row["rank"] = rank
    all_ranked = [row for row in preserved if row.get("step_id") != step_id] + step_ranked
    exp_io.validate_managed_output_paths(
        workspace,
        [
            out,
            workspace / "run_manifest.tsv",
            workspace / "run_matrix.csv",
            workspace / "reports" / "run_matrix.md",
            workspace / "events.jsonl",
        ],
    )
    write_rows(out, all_ranked)
    merge_run_manifest(
        workspace,
        [
            {
                "step_id": row.get("step_id"),
                "run_id": row.get("run_id"),
                "run_name": row.get("run_name"),
                "metric": metric,
                "score": row.get("score"),
                "rank": row.get("rank"),
                "checkpoint_path": row.get("checkpoint_path"),
            }
            for row in step_ranked
        ],
    )
    append_event(
        workspace,
        "candidate_selected",
        {
            "step_id": (recipe.get("step") or {}).get("id"),
            "metric": metric,
            "mode": mode,
            "ranking": str(out),
            "selected_run_id": step_ranked[0].get("run_id") if step_ranked else None,
        },
    )
    return out


def scan_hparam_checkpoints(run_dir: str | Path, metric: str, mode: str, *, top_k: int | None = None) -> Path:
    root = Path(run_dir)
    plan = artifacts.read_hparam_plan(root)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    canonical_rows = read_run_manifest(workspace)
    out = root / "checkpoint_ranking.csv"
    existing_ranked = read_rows(out, require_managed_identity=True)
    validate_managed_run_rows(existing_ranked, source=str(out), cardinality="many_per_run")
    for row in existing_ranked:
        canonical = resolve_run_row(canonical_rows, row)
        if canonical is None:
            raise ValueError(
                f"Existing checkpoint ranking row is outside the canonical manifest: "
                f"{row.get('step_id', '')} / {row.get('run_id', '')}"
            )
        validate_frozen_run_update(canonical, row)
    rows = []
    for run in plan["runs"]:
        manifest_path = artifacts.find_run_manifest(run)
        manifest = read_json(manifest_path) if manifest_path else {}
        rows.extend(_checkpoint_scan_rows(run, metric, manifest_path, manifest))
    reverse = mode == "max"
    ranked = sorted(
        rows,
        key=lambda row: artifacts.sortable_score(row.get("score"), reverse),
        reverse=reverse,
    )
    if top_k is not None:
        ranked = ranked[:top_k]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    validate_managed_run_rows(ranked, source="checkpoint ranking", cardinality="many_per_run")
    if ranked:
        write_rows(out, ranked)
    else:
        out.write_text("step_id,run_id\n")
    return out


def _checkpoint_scan_rows(
    run: dict[str, Any],
    metric: str,
    manifest_path: Path | None,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    if manifest_path:
        runtime_dir = Path(str(run["runtime_dir"]))
        checkpoint_dir = Path(str(run["checkpoint_dir"]))
        for epoch, score in _history_metric_rows(runtime_dir, metric):
            checkpoint = artifacts.checkpoint_for_epoch_in_dir(checkpoint_dir, epoch)
            if checkpoint:
                rows.append(
                    {
                        "step_id": run["step_id"],
                        "run_id": run["run_id"],
                        "version": run["version"],
                        "config": run.get("config"),
                        "metric": metric,
                        "score": score,
                        "epoch": epoch,
                        "checkpoint_path": str(checkpoint),
                        "run_manifest": str(manifest_path),
                        "source": "history",
                        **managed_run_parameters(run),
                    }
                )
    if rows:
        return rows
    score = artifacts.metric_value(manifest, metric)
    checkpoint = artifacts.fixed_checkpoint_path(manifest, Path(str(run["checkpoint_dir"])))
    if score not in ("", None) and checkpoint:
        rows.append(
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "version": run["version"],
                "config": run.get("config"),
                "metric": metric,
                "score": score,
                "epoch": manifest.get("epoch") or artifacts.epoch_from_checkpoint_name(Path(checkpoint).name),
                "checkpoint_path": checkpoint,
                "run_manifest": str(manifest_path or ""),
                "source": "manifest",
                **managed_run_parameters(run),
            }
        )
    return rows


def _history_metric_rows(run_dir: Path, metric: str) -> list[tuple[int, float]]:
    by_epoch: dict[int, float] = {}
    for record in _history_records(run_dir):
        if metric not in record:
            continue
        epoch = _history_epoch(record)
        score = artifacts.float_or_none(record.get(metric))
        if epoch is not None and score is not None:
            by_epoch[epoch] = score
    return sorted(by_epoch.items())


def _history_records(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("wandb/**/wandb-history*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            if line.strip():
                records.append(json.loads(line))
    for path in sorted(run_dir.glob("wandb/**/wandb-history*.csv")):
        with path.open(newline="") as file_obj:
            records.extend(csv.DictReader(file_obj))
    history = (
        read_json(run_dir / "run_manifest.json").get("history") if (run_dir / "run_manifest.json").exists() else None
    )
    if isinstance(history, list):
        records.extend(row for row in history if isinstance(row, dict))
    return records


def _history_epoch(record: dict[str, Any]) -> int | None:
    for key in ("epoch", "trainer/epoch", "current_epoch", "global_epoch"):
        value = artifacts.float_or_none(record.get(key))
        if value is not None:
            return int(value)
    return None
