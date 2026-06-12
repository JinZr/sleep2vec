from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from sleep2stat.config import Sleep2statConfig
from sleep2stat.core.artifacts import AnalyzerResult, FailureRecord
from sleep2stat.io.records import SleepRecord, records_to_frame

COMPLETION_MARKER = "_SUCCESS.json"
TABLE_NAMES = ("epoch_alignment", "second_alignment", "event_alignment", "night_stats")
SUMMARY_TABLE_NAMES = ("model_summary", "analyzer_summary")


class AnalysisBundleWriter:
    def __init__(self, config: Sleep2statConfig):
        self.config = config
        self.run_dir = config.run.output_dir
        self.status_dir = self.run_dir / "status"
        self.tables_dir = self.run_dir / "tables"
        self.per_record_dir = self.run_dir / "per_record"

    def prepare(self, *, args: argparse.Namespace) -> None:
        if self.run_dir.exists() and self.config.run.overwrite:
            shutil.rmtree(self.run_dir)
        if self.run_dir.exists() and any(self.run_dir.iterdir()) and not self.config.run.skip_existing:
            raise FileExistsError(f"sleep2stat output_dir already exists: {self.run_dir}")
        if self.run_dir.exists() and self.config.run.skip_existing and not self.config.outputs.write_per_record:
            raise ValueError("run.skip_existing requires outputs.write_per_record=true in sleep2stat v0.1.")
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        if self.config.outputs.write_per_record:
            self.per_record_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.config.path, self.run_dir / "config.yaml")
        with (self.run_dir / "cli_args.yaml").open("w") as f:
            yaml.safe_dump(_to_yamlable(args), f, sort_keys=True)

    def write_record_manifest(self, records: list[SleepRecord]) -> None:
        records_to_frame(records).to_csv(self.run_dir / "record_manifest.csv", index=False)

    def write_progress(self, *, total_records: int, completed_records: int, status: str) -> None:
        payload = {
            "status": status,
            "total_records": int(total_records),
            "completed_records": int(completed_records),
            "updated_at_utc": _utc_now(),
        }
        _write_json(payload, self.status_dir / "progress.json")

    def write_failures(self, failures: Iterable[FailureRecord]) -> None:
        rows = [asdict(failure) for failure in failures]
        pd.DataFrame(rows, columns=["record_id", "analyzer", "error_type", "message"]).to_csv(
            self.status_dir / "failures.csv", index=False
        )

    def write_run_manifest(
        self,
        *,
        status: str,
        records: list[SleepRecord],
        failures: list[FailureRecord],
        dry_run: bool,
    ) -> None:
        manifest = {
            "kind": "sleep2stat_run",
            "status": status,
            "dry_run": bool(dry_run),
            "run_name": self.config.run.name,
            "created_at_utc": _utc_now(),
            "config_path": str(self.config.path),
            "record_count": len(records),
            "failure_count": len(failures),
            "paths": {
                "run_dir": str(self.run_dir),
                "record_manifest": str(self.run_dir / "record_manifest.csv"),
                "failures": str(self.status_dir / "failures.csv"),
                "tables_dir": str(self.tables_dir),
                "per_record_dir": str(self.per_record_dir) if self.config.outputs.write_per_record else None,
            },
        }
        _write_json(manifest, self.run_dir / "run_manifest.json")

    def write_results(
        self,
        records: list[SleepRecord],
        results: list[AnalyzerResult],
        failures: Iterable[FailureRecord] | None = None,
    ) -> None:
        tables = collect_tables(records, results)
        summary_tables = collect_summary_tables(self.config, results, list(failures or []))
        if self.config.outputs.write_global_tables:
            for name, frame in {**tables, **summary_tables}.items():
                path = self._table_path(name)
                frame = self._merge_existing_global_table(name, path, frame)
                frame.to_csv(path, index=False, compression=self._compression_for(path))
        if self.config.outputs.write_per_record:
            result_record_ids = {result.record_id for result in results}
            for record in records:
                if record.record_id not in result_record_ids:
                    continue
                record_dir = self.per_record_dir / record.record_id
                record_dir.mkdir(parents=True, exist_ok=True)
                for name, frame in tables.items():
                    if "record_id" in frame.columns:
                        frame = frame[frame["record_id"] == record.record_id]
                    path = record_dir / self._per_record_filename(name)
                    frame.to_csv(path, index=False, compression=self._compression_for(path))
                self._write_per_record_sidecars(record, tables, results, record_dir)

    def write_completion_markers(self, record_ids: Iterable[str]) -> None:
        if not self.config.outputs.write_per_record:
            return
        for record_id in record_ids:
            record_dir = self.per_record_dir / str(record_id)
            if not record_dir.exists():
                continue
            _write_json({"status": "completed", "updated_at_utc": _utc_now()}, record_dir / COMPLETION_MARKER)

    def _table_path(self, name: str) -> Path:
        uncompressed = {"night_stats", *SUMMARY_TABLE_NAMES}
        suffix = ".csv.gz" if self.config.outputs.compression == "gzip" and name not in uncompressed else ".csv"
        return self.tables_dir / f"{name}{suffix}"

    def _per_record_filename(self, name: str) -> str:
        suffix = ".csv.gz" if self.config.outputs.compression == "gzip" and name != "night_stats" else ".csv"
        return f"{name}{suffix}"

    def _write_per_record_sidecars(
        self,
        record: SleepRecord,
        tables: dict[str, pd.DataFrame],
        results: list[AnalyzerResult],
        record_dir: Path,
    ) -> None:
        events = tables["event_alignment"]
        if not events.empty and "record_id" in events.columns:
            events = events[events["record_id"] == record.record_id]
        events_path = record_dir / ("events.csv.gz" if self.config.outputs.compression == "gzip" else "events.csv")
        events.to_csv(events_path, index=False, compression=self._compression_for(events_path))

        night = tables["night_stats"]
        if not night.empty and "record_id" in night.columns:
            night = night[night["record_id"] == record.record_id]
        night_payload = {}
        if not night.empty:
            night_payload = _json_safe(night.iloc[0].to_dict())
        _write_json(night_payload, record_dir / "night_stats.json")

        arrays = {}
        for result in results:
            if result.record_id != record.record_id:
                continue
            for key, value in result.arrays.items():
                arrays[f"{result.name}__{key}"] = np.asarray(value)
        if arrays:
            np.savez_compressed(record_dir / "arrays.npz", **arrays)

    @staticmethod
    def _compression_for(path: Path) -> str | None:
        return "gzip" if path.suffix == ".gz" else None

    def filter_records_for_run(self, records: list[SleepRecord]) -> list[SleepRecord]:
        if not self.config.run.skip_existing or not self.config.outputs.write_per_record:
            return records
        return [record for record in records if not self._record_has_outputs(record)]

    def _record_has_outputs(self, record: SleepRecord) -> bool:
        return (self.per_record_dir / record.record_id / COMPLETION_MARKER).exists()

    def _merge_existing_global_table(self, name: str, path: Path, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.config.run.skip_existing or not path.exists() or path.stat().st_size == 0:
            return frame
        try:
            existing = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return frame
        if existing.empty:
            return frame
        merged = pd.concat([existing, frame], ignore_index=True, sort=False)
        subset = _dedupe_columns_for_table(name, merged)
        if subset:
            merged = merged.drop_duplicates(subset=subset, keep="last")
        return merged


def collect_tables(records: list[SleepRecord], results: list[AnalyzerResult]) -> dict[str, pd.DataFrame]:
    record_base = {
        record.record_id: {
            "record_id": record.record_id,
            "path": str(record.path),
            "split": record.split,
            "source": record.source,
            "duration_sec": record.duration_sec,
        }
        for record in records
    }
    epoch_frames = [result.epoch for result in results if result.epoch is not None and not result.epoch.empty]
    second_frames = [result.second for result in results if result.second is not None and not result.second.empty]
    event_frames = [result.events for result in results if result.events is not None and not result.events.empty]
    night_rows = []
    for result in results:
        if result.night is None:
            continue
        row = dict(record_base.get(result.record_id, {"record_id": result.record_id}))
        row.update(result.night)
        if result.warnings:
            row.setdefault("warnings_json", json.dumps(result.warnings))
        night_rows.append(row)
    return {
        "epoch_alignment": _merge_alignment_frames(epoch_frames),
        "second_alignment": _merge_alignment_frames(second_frames),
        "event_alignment": pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame(),
        "night_stats": _merge_night_rows(night_rows),
    }


def collect_summary_tables(
    config: Sleep2statConfig,
    results: list[AnalyzerResult],
    failures: list[FailureRecord],
) -> dict[str, pd.DataFrame]:
    failure_counts: dict[str, int] = {}
    for failure in failures:
        failure_counts[failure.analyzer] = failure_counts.get(failure.analyzer, 0) + 1

    analyzer_rows = []
    for kind, configs in (("analyzer", config.analyzers), ("reducer", config.reducers)):
        for item in configs:
            item_results = [result for result in results if result.name == item.name]
            analyzer_rows.append(
                {
                    "kind": kind,
                    "name": item.name,
                    "type": item.type,
                    "enabled": bool(item.enabled),
                    "record_count": len({result.record_id for result in item_results}),
                    "result_count": len(item_results),
                    "failure_count": failure_counts.get(item.name, 0),
                    "epoch_rows": _result_row_count(item_results, "epoch"),
                    "second_rows": _result_row_count(item_results, "second"),
                    "event_rows": _result_row_count(item_results, "events"),
                    "night_rows": sum(1 for result in item_results if result.night is not None),
                }
            )

    model_rows = []
    for analyzer in config.analyzers:
        if analyzer.type != "sleep2vec_downstream":
            continue
        model_rows.append(
            {
                "name": analyzer.name,
                "type": analyzer.type,
                "namespace": analyzer.namespace,
                "label_name": analyzer.label_name,
                "config": None if analyzer.config is None else str(analyzer.config),
                "ckpt_path": None if analyzer.ckpt_path is None else str(analyzer.ckpt_path),
                "input_channels": ",".join(analyzer.input_channels),
                "batch_size": analyzer.batch_size,
                "threshold": analyzer.threshold,
            }
        )
    return {
        "model_summary": pd.DataFrame(
            model_rows,
            columns=[
                "name",
                "type",
                "namespace",
                "label_name",
                "config",
                "ckpt_path",
                "input_channels",
                "batch_size",
                "threshold",
            ],
        ),
        "analyzer_summary": pd.DataFrame(
            analyzer_rows,
            columns=[
                "kind",
                "name",
                "type",
                "enabled",
                "record_count",
                "result_count",
                "failure_count",
                "epoch_rows",
                "second_rows",
                "event_rows",
                "night_rows",
            ],
        ),
    }


def _merge_alignment_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    key_columns = ["record_id", "path", "token_idx", "start_sec", "end_sec"]
    result = frames[0].copy()
    for frame in frames[1:]:
        common = [column for column in key_columns if column in result.columns and column in frame.columns]
        if common:
            result = result.merge(frame, on=common, how="outer")
        else:
            result = pd.concat([result, frame], ignore_index=True, sort=False)
    return result.sort_values([column for column in key_columns if column in result.columns]).reset_index(drop=True)


def _merge_night_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "record_id" not in frame.columns:
        return frame
    merged_rows = []
    for record_id, group in frame.groupby("record_id", sort=False):
        merged: dict[str, Any] = {}
        for _, row in group.iterrows():
            for key, value in row.items():
                if _is_missing(value):
                    continue
                merged[key] = value
        merged["record_id"] = record_id
        merged_rows.append(merged)
    return pd.DataFrame(merged_rows)


def _dedupe_columns_for_table(name: str, frame: pd.DataFrame) -> list[str]:
    candidates = {
        "epoch_alignment": ["record_id", "token_idx"],
        "second_alignment": ["record_id", "second_idx"],
        "event_alignment": ["record_id", "event_id"],
        "night_stats": ["record_id"],
        "model_summary": ["name"],
        "analyzer_summary": ["kind", "name"],
    }.get(name, ["record_id"])
    return [column for column in candidates if column in frame.columns]


def _result_row_count(results: list[AnalyzerResult], attr: str) -> int:
    count = 0
    for result in results:
        value = getattr(result, attr)
        if value is not None:
            count += len(value)
    return count


def _to_yamlable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_yamlable(asdict(obj))
    if isinstance(obj, argparse.Namespace):
        return _to_yamlable(vars(obj))
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return _to_yamlable(vars(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(key): _to_yamlable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yamlable(value) for value in obj]
    return obj


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and pd.isna(obj):
        return None
    if _is_missing(obj):
        return None
    return obj


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
