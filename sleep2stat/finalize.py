from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from sleep2stat.io.writers import _utc_now, _write_json

FAILURE_COLUMNS = ["record_id", "analyzer", "error_type", "message"]


def cohort_finalize(output_run_dir: Path, input_run_dirs: list[Path]) -> dict[str, Any]:
    output_run_dir = Path(output_run_dir)
    if output_run_dir.exists() and any(output_run_dir.iterdir()):
        raise FileExistsError(f"sleep2stat finalize output_run_dir already exists: {output_run_dir}")
    for run_dir in input_run_dirs:
        _validate_input_run_dir(Path(run_dir))
    output_run_dir.mkdir(parents=True, exist_ok=True)
    (output_run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_run_dir / "status").mkdir(parents=True, exist_ok=True)

    manifests = []
    night_stats = []
    failures = []
    global_failed_ids: set[str] = set()
    for run_dir in input_run_dirs:
        manifest = _read_csv(run_dir / "record_manifest.csv")
        night = _read_csv(run_dir / "tables" / "night_stats.csv")
        failure = _read_csv(run_dir / "status" / "failures.csv", columns=FAILURE_COLUMNS)
        manifests.append(manifest)
        night_stats.append(night)
        failures.append(failure)
        if _has_global_failure(failure):
            # Scope "__all__" to this input run so one failed shard does not fail other shards.
            global_failed_ids.update(_record_ids(manifest) - _record_ids(night))

    manifest_frame = _dedupe_by_record_id(_concat(manifests))
    night_frame = _dedupe_by_record_id(_concat(night_stats))
    failure_frame = _concat(failures, columns=FAILURE_COLUMNS).drop_duplicates().reset_index(drop=True)
    if not night_frame.empty and "record_id" in failure_frame.columns:
        completed_ids = set(night_frame["record_id"].astype(str))
        failure_frame = failure_frame[~failure_frame["record_id"].astype(str).isin(completed_ids)].reset_index(
            drop=True
        )
    manifest_ids = _record_ids(manifest_frame)
    completed_ids = _record_ids(night_frame)
    failed_ids = {
        record_id
        for record_id in _record_ids(failure_frame)
        if record_id not in {"", "__all__"} and record_id not in completed_ids
    }
    failed_ids.update(record_id for record_id in global_failed_ids if record_id not in completed_ids)
    pending_ids = sorted(manifest_ids - completed_ids - failed_ids)

    manifest_frame.to_csv(output_run_dir / "record_manifest.csv", index=False)
    night_frame.to_csv(output_run_dir / "tables" / "night_stats.csv", index=False)
    failure_frame.to_csv(output_run_dir / "status" / "failures.csv", index=False)

    status = "incomplete" if pending_ids else "completed_with_failures" if not failure_frame.empty else "completed"
    manifest = {
        "kind": "sleep2stat_cohort_finalize",
        "status": status,
        "source_run_dirs": [str(Path(path)) for path in input_run_dirs],
        "total_records": int(len(manifest_frame)),
        "night_stats_rows": int(len(night_frame)),
        "failure_rows": int(len(failure_frame)),
        "pending_records": int(len(pending_ids)),
        "pending_record_ids": pending_ids[:20],
        "updated_at_utc": _utc_now(),
    }
    _write_json(manifest, output_run_dir / "run_manifest.json")
    _write_json(
        {
            "status": status,
            "total_records": int(len(manifest_frame)),
            "completed_records": int(len(night_frame)),
            "failed_records": int(len(failed_ids)),
            "failure_rows": int(len(failure_frame)),
            "pending_records": int(len(pending_ids)),
            "pending_record_ids": pending_ids[:20],
            "updated_at_utc": _utc_now(),
        },
        output_run_dir / "status" / "progress.json",
    )
    return manifest


def _validate_input_run_dir(run_dir: Path) -> None:
    if not run_dir.exists():
        raise FileNotFoundError(f"sleep2stat finalize input run_dir not found: {run_dir}")
    if not (run_dir / "record_manifest.csv").exists():
        raise FileNotFoundError(f"sleep2stat finalize input run_dir missing record_manifest.csv: {run_dir}")


def _read_csv(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def _concat(frames: list[pd.DataFrame], *, columns: list[str] | None = None) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    return pd.concat(non_empty, ignore_index=True, sort=False)


def _dedupe_by_record_id(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "record_id" not in frame.columns:
        return frame.reset_index(drop=True)
    return frame.drop_duplicates(subset=["record_id"], keep="last").reset_index(drop=True)


def _record_ids(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "record_id" not in frame.columns:
        return set()
    return set(frame["record_id"].astype(str))


def _has_global_failure(frame: pd.DataFrame) -> bool:
    return not frame.empty and "record_id" in frame.columns and frame["record_id"].astype(str).eq("__all__").any()
