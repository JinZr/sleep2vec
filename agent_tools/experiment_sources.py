"""Validation of recorded checkpoint evidence against the canonical run manifest.

Layer 0 leaf. Checks that checkpoint rows belong to a managed run, stay inside
its frozen checkpoint directory, and still exist on the host that owns them;
per-run checkpoint inventory and epoch resolution live in ``run_artifacts``.
``experiment_tracking`` re-exports these checks for finalization and hparam
selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import experiment_io as exp_io
from .experiment_workspace import managed_run_key, validate_checkpoint_ownership, validate_frozen_run_update


def validate_checkpoint_evidence_rows(
    runs: list[dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    remote: str | None = None,
) -> None:
    runs_by_key = {managed_run_key(run): run for run in runs}
    grouped_paths: dict[tuple[str | None, str], list[Path]] = {}
    for row in rows:
        checkpoint_path = row.get("checkpoint_path")
        if checkpoint_path in (None, ""):
            continue
        run = runs_by_key.get(managed_run_key(row))
        if run is None:
            raise ValueError(
                f"Checkpoint evidence is outside the canonical manifest: "
                f"{row.get('step_id', '')} / {row.get('run_id', '')}"
            )
        validate_frozen_run_update(run, row)
        validate_checkpoint_ownership(run, row)
        evidence_host = checkpoint_evidence_host(run, remote)
        checkpoint_dir = str(run["checkpoint_dir"])
        grouped_paths.setdefault((evidence_host, checkpoint_dir), []).append(Path(str(checkpoint_path)))
    for (evidence_host, checkpoint_dir), paths in grouped_paths.items():
        exp_io.validate_managed_output_paths(checkpoint_dir, paths, remote=evidence_host)
        for path in paths:
            if not exp_io.path_exists_at(path, remote=evidence_host):
                raise ValueError(f"Checkpoint evidence is missing: {path}")


def checkpoint_evidence_host(run: dict[str, Any], remote: str | None) -> str | None:
    if run.get("target") != "ssh":
        return remote
    host = str(run.get("host") or "")
    if not host:
        raise ValueError(f"Managed SSH run is missing its host: {run.get('step_id', '')} / {run.get('run_id', '')}")
    return host
