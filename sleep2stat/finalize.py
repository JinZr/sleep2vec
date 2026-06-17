from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from sleep2stat.io.writers import RUN_ANALYSIS_TERMINAL_STATUSES, _require_terminal_run_manifest, _utc_now, _write_json


def cohort_finalize(output_run_dir: Path, input_run_dirs: list[Path]) -> dict[str, Any]:
    output_run_dir = Path(output_run_dir)
    if output_run_dir.exists() and any(output_run_dir.iterdir()):
        raise FileExistsError(f"sleep2stat finalize output_run_dir already exists: {output_run_dir}")
    for run_dir in input_run_dirs:
        _validate_input_run_dir(Path(run_dir))

    manifests = []
    night_stats = []
    for run_dir in input_run_dirs:
        manifest = _read_csv(run_dir / "record_manifest.csv")
        night = _read_csv(run_dir / "tables" / "night_stats.csv")
        if "record_id" not in manifest.columns:
            raise ValueError(f"sleep2stat finalize input record_manifest.csv missing record_id column: {run_dir}")
        manifest_ids = set(manifest["record_id"].astype(str))
        night_ids = set(night["record_id"].astype(str)) if "record_id" in night.columns else set()
        missing_ids = sorted(manifest_ids - night_ids)
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(
                f"sleep2stat finalize input missing night_stats rows for {len(missing_ids)} record(s): {preview}"
            )
        manifests.append(manifest)
        night_stats.append(night)

    manifest_frame = _dedupe_by_record_id(_concat(manifests))
    night_frame = _dedupe_by_record_id(_concat(night_stats))

    # Input preflight must finish before touching the single-use output directory.
    output_run_dir.mkdir(parents=True, exist_ok=True)
    (output_run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_run_dir / "status").mkdir(parents=True, exist_ok=True)

    manifest_frame.to_csv(output_run_dir / "record_manifest.csv", index=False)
    night_frame.to_csv(output_run_dir / "tables" / "night_stats.csv", index=False)

    status = "completed"
    manifest = {
        "kind": "sleep2stat_cohort_finalize",
        "status": status,
        "source_run_dirs": [str(Path(path)) for path in input_run_dirs],
        "total_records": int(len(manifest_frame)),
        "night_stats_rows": int(len(night_frame)),
        "updated_at_utc": _utc_now(),
    }
    _write_json(manifest, output_run_dir / "run_manifest.json")
    _write_json(
        {
            "status": status,
            "total_records": int(len(manifest_frame)),
            "completed_records": int(len(night_frame)),
            "updated_at_utc": _utc_now(),
        },
        output_run_dir / "status" / "progress.json",
    )
    return manifest


def _validate_input_run_dir(run_dir: Path) -> None:
    if not run_dir.exists():
        raise FileNotFoundError(f"sleep2stat finalize input run_dir not found: {run_dir}")
    _require_terminal_run_manifest(run_dir, RUN_ANALYSIS_TERMINAL_STATUSES, command="cohort-finalize")
    if not (run_dir / "record_manifest.csv").exists():
        raise FileNotFoundError(f"sleep2stat finalize input run_dir missing record_manifest.csv: {run_dir}")
    if not (run_dir / "tables" / "night_stats.csv").exists():
        raise FileNotFoundError(f"sleep2stat finalize input run_dir missing night_stats.csv: {run_dir}")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True, sort=False)


def _dedupe_by_record_id(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "record_id" not in frame.columns:
        return frame.reset_index(drop=True)
    return frame.drop_duplicates(subset=["record_id"], keep="last").reset_index(drop=True)
