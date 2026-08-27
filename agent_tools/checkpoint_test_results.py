from __future__ import annotations

from pathlib import Path
from typing import Any

from . import run_artifacts as artifacts


def expected_epoch_checkpoints(
    checkpoint_dir: str,
    checkpoint_names: list[str],
    *,
    step_id: str,
    run_id: str,
) -> dict[str, int]:
    expected = {}
    seen_epochs = set()
    for name in sorted(name for name in checkpoint_names if name.startswith("epoch=")):
        checkpoint_path = str(Path(checkpoint_dir) / name)
        epoch = artifacts.epoch_number_from_checkpoint_name(name)
        if epoch is None:
            raise ValueError(f"Saved epoch checkpoint has an invalid epoch for {step_id} / {run_id}: {checkpoint_path}")
        if epoch in seen_epochs:
            raise ValueError(f"Saved epoch checkpoints contain a duplicate epoch for {step_id} / {run_id}: {epoch}")
        seen_epochs.add(epoch)
        expected[checkpoint_path] = epoch
    if not expected:
        raise ValueError(f"Completed hparam run has no saved epoch checkpoints: {step_id} / {run_id}")
    return expected


def validate_checkpoint_test_results(
    results: list[Any],
    metric: str,
    expected_epochs: dict[str, int],
    *,
    step_id: str,
    run_id: str,
) -> list[dict[str, Any]]:
    rows = []
    seen_paths = set()
    seen_epochs = set()
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"checkpoint_test_results[{index}] must be a mapping: {step_id} / {run_id}")
        checkpoint_path = str(result.get("checkpoint_path") or "")
        if checkpoint_path not in expected_epochs:
            raise ValueError(
                f"checkpoint_test_results contains an unmanaged epoch checkpoint: "
                f"{step_id} / {run_id} / {checkpoint_path}"
            )
        if checkpoint_path in seen_paths:
            raise ValueError(
                f"checkpoint_test_results contains a duplicate checkpoint: " f"{step_id} / {run_id} / {checkpoint_path}"
            )
        seen_paths.add(checkpoint_path)
        epoch = artifacts.epoch_number(result.get("epoch"))
        if epoch != expected_epochs[checkpoint_path]:
            raise ValueError(
                f"checkpoint_test_results epoch differs from checkpoint_path: "
                f"{step_id} / {run_id} / {checkpoint_path}"
            )
        if epoch in seen_epochs:
            raise ValueError(f"checkpoint_test_results contains a duplicate epoch: {step_id} / {run_id} / {epoch}")
        seen_epochs.add(epoch)
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        raw_score = metrics.get(metric)
        score = None if isinstance(raw_score, bool) else artifacts.float_or_none(raw_score)
        if score is None:
            raise ValueError(
                f"checkpoint_test_results is missing a finite {metric}: " f"{step_id} / {run_id} / {checkpoint_path}"
            )
        rows.append({"checkpoint_path": checkpoint_path, "epoch": epoch, "score": score})
    missing_paths = sorted(set(expected_epochs) - seen_paths)
    if missing_paths:
        raise ValueError(f"checkpoint_test_results is incomplete for {step_id} / {run_id}: " + ", ".join(missing_paths))
    return rows
