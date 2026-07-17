from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Iterable
import uuid

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from sleep2stat.config import Sleep2statConfig
from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.io.records import SleepRecord, records_to_frame

RESULT_MANIFEST = "result_manifest.csv"
TABLE_NAMES = ("epoch_alignment", "second_alignment", "event_alignment", "night_stats")
ALIGNMENT_TABLE_NAMES = ("epoch_alignment", "second_alignment", "event_alignment")
SUMMARY_TABLE_NAMES = ("model_summary", "analyzer_summary")
RUN_TERMINAL_STATUSES = frozenset({"completed", "dry_run"})
RUN_ANALYSIS_TERMINAL_STATUSES = frozenset({"completed"})


class AnalysisBundleWriter:
    def __init__(self, config: Sleep2statConfig):
        self.config = config
        self.run_dir = config.run.output_dir
        self.status_dir = self.run_dir / "status"
        self.tables_dir = self.run_dir / "tables"
        self.per_record_dir = self.run_dir / "per_record"
        self.shards_dir = self.tables_dir / "_shards"

    def prepare(self, *, args: argparse.Namespace) -> None:
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(f"sleep2stat output_dir already exists: {self.run_dir}")
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        if self.config.outputs.write_per_record:
            self.per_record_dir.mkdir(parents=True, exist_ok=True)
        _write_json({"pid": os.getpid(), "updated_at_utc": _utc_now()}, self.status_dir / "pid.json")
        config_copy = self.run_dir / "config.yaml"
        if self.config.path.resolve() != config_copy.resolve():
            copyfile(self.config.path, config_copy)
        with (self.run_dir / "cli_args.yaml").open("w") as f:
            yaml.safe_dump(_to_yamlable(args), f, sort_keys=True)

    def write_record_manifest(self, records: list[SleepRecord]) -> None:
        path = self.run_dir / "record_manifest.csv"
        frame = records_to_frame(records, metadata_columns=self.config.data.metadata_columns)
        frame.to_csv(path, index=False)

    def write_progress(
        self,
        *,
        total_records: int,
        completed_records: int,
        status: str,
        num_workers: int | None = None,
        execution_split: str | None = None,
    ) -> None:
        payload = {
            "status": status,
            "total_records": int(total_records),
            "completed_records": int(completed_records),
            "config_fingerprint": _config_fingerprint_from_path(self.config.path),
            "updated_at_utc": _utc_now(),
        }
        if num_workers is not None:
            payload["num_workers"] = int(num_workers)
        if execution_split is not None:
            payload["execution_split"] = execution_split
        _write_json(payload, self.status_dir / "progress.json")

    def write_run_manifest(
        self,
        *,
        status: str,
        records: list[SleepRecord],
        dry_run: bool,
        num_workers: int | None = None,
        execution_split: str | None = None,
    ) -> None:
        manifest = {
            "kind": "sleep2stat_run",
            "status": status,
            "dry_run": bool(dry_run),
            "run_name": self.config.run.name,
            "created_at_utc": _utc_now(),
            "config_path": str(self.config.path),
            "config_fingerprint": _config_fingerprint_from_path(self.config.path),
            "record_count": len(records),
            "num_workers": None if num_workers is None else int(num_workers),
            "execution_split": execution_split,
            "paths": {
                "run_dir": str(self.run_dir),
                "record_manifest": str(self.run_dir / "record_manifest.csv"),
                "tables_dir": str(self.tables_dir),
                "per_record_dir": str(self.per_record_dir) if self.config.outputs.write_per_record else None,
            },
        }
        _write_json(manifest, self.run_dir / "run_manifest.json")

    def write_results(
        self,
        records: list[SleepRecord],
        results: list[AnalyzerResult],
    ) -> None:
        completed_record_ids = {record.record_id for record in records}
        self.write_chunk(records, results, completed_record_ids=completed_record_ids)
        self.rebuild_global_tables(records)

    def write_chunk(
        self,
        records: list[SleepRecord],
        results: list[AnalyzerResult],
        *,
        completed_record_ids: Iterable[str],
    ) -> None:
        tables = collect_tables(records, results)
        if self.config.outputs.write_global_tables:
            completed = {str(record_id) for record_id in completed_record_ids}
            for name in ALIGNMENT_TABLE_NAMES:
                if not self._global_table_enabled(name):
                    continue
                frame = tables[name]
                if not frame.empty and "record_id" in frame.columns:
                    frame = frame[frame["record_id"].astype(str).isin(completed)]
                self._write_global_table_shard(name, frame)
        if self.config.outputs.write_per_record:
            output_record_ids = {result.record_id for result in results}
            for record in records:
                if record.record_id not in output_record_ids:
                    continue
                record_dir = self.per_record_dir / record.record_id
                record_dir.mkdir(parents=True, exist_ok=True)
                for name, frame in tables.items():
                    if "record_id" in frame.columns:
                        frame = frame[frame["record_id"] == record.record_id]
                    path = record_dir / self._per_record_filename(name)
                    frame.to_csv(path, index=False, compression=self._compression_for(path))
                self._write_per_record_sidecars(record, tables, results, record_dir)

    def rebuild_global_tables(
        self,
        records: list[SleepRecord],
        *,
        num_workers: int = 0,
    ) -> None:
        if not self.config.outputs.write_global_tables:
            return
        self._rebuild_alignment_tables_from_shards()
        if self._global_table_enabled("night_stats"):
            self._collect_night_stats(num_workers=num_workers).to_csv(self._table_path("night_stats"), index=False)
        summary_tables = self._collect_cumulative_summary_tables(num_workers=num_workers)
        for name, frame in summary_tables.items():
            frame.to_csv(self._table_path(name), index=False)

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

        self._result_manifest(record, results).to_csv(record_dir / RESULT_MANIFEST, index=False)

    @staticmethod
    def _compression_for(path: Path) -> str | None:
        return "gzip" if path.suffix == ".gz" else None

    def _write_global_table_shard(self, name: str, frame: pd.DataFrame) -> None:
        if not self._global_table_enabled(name):
            return
        if frame.empty:
            return
        path = self._next_shard_path(name)
        frame.to_csv(path, index=False, compression=self._compression_for(path))

    def _next_shard_path(self, name: str) -> Path:
        shard_dir = self.shards_dir / name
        shard_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".csv.gz" if self.config.outputs.compression == "gzip" else ".csv"
        idx = len(list(shard_dir.glob(f"part-*{suffix}")))
        path = shard_dir / f"part-{idx:06d}{suffix}"
        while path.exists():
            idx += 1
            path = shard_dir / f"part-{idx:06d}{suffix}"
        return path

    def _alignment_shard_paths(self, name: str) -> list[Path]:
        return sorted((self.shards_dir / name).glob("part-*.csv*"))

    def _rebuild_alignment_tables_from_shards(self) -> None:
        for name in ALIGNMENT_TABLE_NAMES:
            if not self._global_table_enabled(name):
                continue
            shards = self._alignment_shard_paths(name)
            if not shards:
                continue
            columns = _union_csv_columns(shards)
            if not columns:
                continue
            path = self._table_path(name)
            if path.exists():
                path.unlink()
            wrote = False
            for shard in tqdm(shards, desc=f"sleep2stat rebuild {name}", unit="shard"):
                try:
                    chunk_iter = pd.read_csv(shard, chunksize=100_000)
                    for frame in chunk_iter:
                        for column in columns:
                            if column not in frame.columns:
                                frame[column] = pd.NA
                        frame = frame.reindex(columns=columns)
                        frame.to_csv(
                            path,
                            mode="a" if wrote else "w",
                            header=not wrote,
                            index=False,
                            compression=self._compression_for(path),
                        )
                        wrote = True
                except pd.errors.EmptyDataError:
                    continue

    def _global_table_enabled(self, name: str) -> bool:
        return bool(self.config.outputs.global_tables.get(name, True))

    def _collect_night_stats(
        self,
        *,
        num_workers: int = 0,
    ) -> pd.DataFrame:
        rows = []
        if not self.per_record_dir.exists():
            return _merge_night_rows(rows)
        paths = []
        for manifest_path in sorted(self.per_record_dir.glob(f"*/{RESULT_MANIFEST}")):
            manifest = _read_result_manifest(manifest_path)
            if not _result_manifest_is_complete(manifest):
                continue
            night_path = manifest_path.parent / "night_stats.json"
            if night_path.exists():
                paths.append(night_path)
        workers = int(num_workers or 0)
        if workers <= 1:
            for path in tqdm(paths, desc="sleep2stat collect night_stats", unit="record"):
                payload = _read_night_stats(path)
                if payload:
                    rows.append(payload)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                payloads = executor.map(_read_night_stats, paths)
                for payload in tqdm(
                    payloads,
                    total=len(paths),
                    desc=f"sleep2stat collect night_stats split{workers}",
                    unit="record",
                ):
                    if payload:
                        rows.append(payload)
        return _merge_night_rows(rows)

    def _collect_cumulative_summary_tables(
        self,
        *,
        num_workers: int = 0,
    ) -> dict[str, pd.DataFrame]:
        paths = []
        if self.per_record_dir.exists():
            paths = sorted(self.per_record_dir.glob(f"*/{RESULT_MANIFEST}"))
        workers = int(num_workers or 0)
        if workers <= 1:
            manifests = [
                frame
                for frame in (
                    _read_result_manifest(path)
                    for path in tqdm(paths, desc="sleep2stat collect result_manifest", unit="record")
                )
                if _result_manifest_is_complete(frame)
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                frames = executor.map(_read_result_manifest, paths)
                manifests = [
                    frame
                    for frame in tqdm(
                        frames,
                        total=len(paths),
                        desc=f"sleep2stat collect result_manifest split{workers}",
                        unit="record",
                    )
                    if _result_manifest_is_complete(frame)
                ]
        manifest = pd.concat(manifests, ignore_index=True, sort=False) if manifests else pd.DataFrame()

        rows = []
        for kind, configs in (("analyzer", self.config.analyzers), ("reducer", self.config.reducers)):
            for item in configs:
                item_rows = manifest[
                    (manifest.get("kind", pd.Series(dtype=object)) == kind)
                    & (manifest.get("name", pd.Series(dtype=object)) == item.name)
                ]
                rows.append(
                    {
                        "kind": kind,
                        "name": item.name,
                        "type": item.type,
                        "enabled": bool(item.enabled),
                        "record_count": (
                            len(set(item_rows.loc[item_rows.get("result_count", 0) > 0, "record_id"]))
                            if not item_rows.empty
                            else 0
                        ),
                        "result_count": int(item_rows.get("result_count", pd.Series(dtype=int)).sum()),
                        "epoch_rows": int(item_rows.get("epoch_rows", pd.Series(dtype=int)).sum()),
                        "second_rows": int(item_rows.get("second_rows", pd.Series(dtype=int)).sum()),
                        "event_rows": int(item_rows.get("event_rows", pd.Series(dtype=int)).sum()),
                        "night_rows": int(item_rows.get("night_rows", pd.Series(dtype=int)).sum()),
                    }
                )
        return {
            "model_summary": _model_summary_frame(self.config),
            "analyzer_summary": pd.DataFrame(
                rows,
                columns=[
                    "kind",
                    "name",
                    "type",
                    "enabled",
                    "record_count",
                    "result_count",
                    "epoch_rows",
                    "second_rows",
                    "event_rows",
                    "night_rows",
                ],
            ),
        }

    def _result_manifest(
        self,
        record: SleepRecord,
        results: list[AnalyzerResult],
    ) -> pd.DataFrame:
        rows = []
        record_results = [result for result in results if result.record_id == record.record_id]
        for kind, configs in (("analyzer", self.config.analyzers), ("reducer", self.config.reducers)):
            for item in configs:
                item_results = [result for result in record_results if result.name == item.name]
                rows.append(
                    {
                        "record_id": record.record_id,
                        "kind": kind,
                        "name": item.name,
                        "type": item.type,
                        "enabled": bool(item.enabled),
                        "result_count": len(item_results),
                        "epoch_rows": _result_row_count(item_results, "epoch"),
                        "second_rows": _result_row_count(item_results, "second"),
                        "event_rows": _result_row_count(item_results, "events"),
                        "night_rows": sum(1 for result in item_results if result.night is not None),
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "record_id",
                "kind",
                "name",
                "type",
                "enabled",
                "result_count",
                "epoch_rows",
                "second_rows",
                "event_rows",
                "night_rows",
            ],
        )


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
) -> dict[str, pd.DataFrame]:
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
                    "epoch_rows": _result_row_count(item_results, "epoch"),
                    "second_rows": _result_row_count(item_results, "second"),
                    "event_rows": _result_row_count(item_results, "events"),
                    "night_rows": sum(1 for result in item_results if result.night is not None),
                }
            )

    return {
        "model_summary": _model_summary_frame(config),
        "analyzer_summary": pd.DataFrame(
            analyzer_rows,
            columns=[
                "kind",
                "name",
                "type",
                "enabled",
                "record_count",
                "result_count",
                "epoch_rows",
                "second_rows",
                "event_rows",
                "night_rows",
            ],
        ),
    }


def _model_summary_frame(config: Sleep2statConfig) -> pd.DataFrame:
    rows = []
    for analyzer in config.analyzers:
        if analyzer.type != "sleep2vec_downstream":
            continue
        rows.append(
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
    return pd.DataFrame(
        rows,
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
    )


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


def _union_csv_columns(paths: list[Path]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for path in paths:
        try:
            frame = pd.read_csv(path, nrows=0)
        except pd.errors.EmptyDataError:
            continue
        for column in frame.columns:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def _result_row_count(results: list[AnalyzerResult], attr: str) -> int:
    count = 0
    for result in results:
        value = getattr(result, attr)
        if value is not None:
            count += len(value)
    return count


def _read_night_stats(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload or None


def _read_result_manifest(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _result_manifest_is_complete(frame: pd.DataFrame | None) -> bool:
    return frame is not None and not frame.empty


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


def _config_fingerprint_from_path(path: Path) -> str:
    if path.exists():
        try:
            payload = yaml.safe_load(path.read_text())
            normalized = yaml.safe_dump(payload, sort_keys=True)
        except Exception:
            normalized = path.read_text()
    else:
        normalized = str(path)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


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
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _require_terminal_run_manifest(
    run_dir: Path,
    allowed_statuses: frozenset[str],
    *,
    command: str,
) -> None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"sleep2stat {command} requires run_manifest.json: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"sleep2stat {command} cannot read run_manifest.json: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"sleep2stat {command} run_manifest.json must contain a JSON object: {manifest_path}")
    status = manifest.get("status")
    if status not in allowed_statuses:
        expected = ", ".join(sorted(allowed_statuses))
        raise ValueError(
            f"sleep2stat {command} requires a completed run directory; "
            f"got status={status!r}, expected one of: {expected}"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
