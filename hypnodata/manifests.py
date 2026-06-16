from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from hypnodata.config import HypnodataConfig

RECORD_COLUMNS = [
    "record_id",
    "center",
    "source",
    "subject_id",
    "session_id",
    "split",
    "path",
    "duration",
    "backend",
    "qc_status",
]
SIGNAL_COLUMNS = [
    "record_id",
    "center",
    "canonical_channel",
    "kind",
    "available",
    "required",
    "raw_file",
    "raw_label",
    "selection_reason",
    "raw_sfreq",
    "target_sfreq",
    "raw_unit",
    "target_unit",
    "scale_applied",
    "polarity_applied",
    "raw_n_samples",
    "output_n_samples",
    "preprocess_steps",
    "qc_status",
    "output_key",
    "mask_column",
]
QC_COLUMNS = ["record_id", "scope", "canonical_channel", "code", "severity", "message"]
FAILURE_COLUMNS = ["record_id", "center", "error_type", "message"]


def mask_column_for_channel(channel_name: str) -> str:
    if channel_name == "stage5":
        return "stage_mask"
    if channel_name == "ahi":
        return "ah_event_mask"
    return f"{channel_name}_mask"


def write_manifests(
    output_dir: Path,
    config: HypnodataConfig,
    *,
    record_rows: list[dict[str, Any]],
    signal_rows: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    manifest_dir = output_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    mask_columns = [mask_column_for_channel(channel) for channel in config.signals]
    base_record_columns = RECORD_COLUMNS + mask_columns
    extra_record_columns = sorted({key for row in record_rows for key in row} - set(base_record_columns))
    record_columns = base_record_columns + extra_record_columns
    _write_csv(manifest_dir / "record_manifest.csv", record_rows, record_columns)
    _write_csv(manifest_dir / "signal_manifest.csv", signal_rows, SIGNAL_COLUMNS)
    _write_csv(manifest_dir / "qc_summary.csv", qc_rows, QC_COLUMNS)
    _write_csv(manifest_dir / "failures.csv", failure_rows, FAILURE_COLUMNS)
    backend_manifest = {
        "backend": config.backend.type,
        "enabled_backends": [config.backend.type],
        "dry_run": bool(dry_run),
        "record_manifest": "manifest/record_manifest.csv",
        "signal_manifest": "manifest/signal_manifest.csv",
        "qc_summary": "manifest/qc_summary.csv",
        "failures": "manifest/failures.csv",
        "npz_records_dir": "backends/npz/records",
        "record_count": len(record_rows),
        "failure_count": len(failure_rows),
        "channels": {
            name: {
                "kind": spec.kind,
                "target_sfreq": spec.target_sfreq,
                "target_unit": spec.target_unit,
                "mask_column": mask_column_for_channel(name),
                "output_key": name,
            }
            for name, spec in config.signals.items()
        },
    }
    (manifest_dir / "backend_manifest.json").write_text(json.dumps(backend_manifest, indent=2, sort_keys=True) + "\n")


def write_discovery_preview(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    manifest_dir = output_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    _write_csv(manifest_dir / "discovery_preview.csv", rows, columns)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False)
