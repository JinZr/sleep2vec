"""Validation of recorded checkpoint evidence against the canonical run manifest.

Checks that checkpoint rows belong to a managed run, stay inside its frozen
checkpoint directory, and still exist on disk; per-run checkpoint inventory and
epoch resolution live in ``run_artifacts``. ``experiment_tracking`` re-exports
these checks for finalization and hparam selection.
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
) -> None:
    runs_by_key = {managed_run_key(run): run for run in runs}
    grouped_paths: dict[str, list[Path]] = {}
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
        grouped_paths.setdefault(str(run["checkpoint_dir"]), []).append(Path(str(checkpoint_path)))
    for checkpoint_dir, paths in grouped_paths.items():
        exp_io.validate_managed_output_paths(checkpoint_dir, paths)
        for path in paths:
            if not exp_io.path_exists_at(path):
                raise ValueError(f"Checkpoint evidence is missing: {path}")
