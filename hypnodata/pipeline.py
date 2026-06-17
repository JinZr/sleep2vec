from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
import time
from typing import Any

import numpy as np

from hypnodata.adapters import call_fix_header, call_read_annotations, call_resolve_metadata, load_adapter
from hypnodata.annotations import ANNOTATION_MATERIALIZATIONS, AnnotationSignal
from hypnodata.backends import npz_record_path, write_npz_record
from hypnodata.channels import ChannelResolutionError, ChannelSelection, resolve_channels
from hypnodata.config import HypnodataConfig, declared_target_sfreq
from hypnodata.discovery import discover_records
from hypnodata.edf import EdfInventory, read_edf_inventory, read_edf_signal
from hypnodata.manifests import mask_column_for_channel, write_discovery_preview, write_manifests
from hypnodata.preprocess import ProcessedSignal, preprocess_signal, truncate_to_common
from hypnodata.qc import QCIssue, issue_row
from hypnodata.records import RecordTask
from hypnodata.status import write_hypnodata_progress


@dataclass
class ProcessResult:
    record_id: str
    record_row: dict[str, Any] | None = None
    signal_rows: list[dict[str, Any]] = field(default_factory=list)
    qc_rows: list[dict[str, Any]] = field(default_factory=list)
    failure_row: dict[str, Any] | None = None


def run_pipeline(
    config: HypnodataConfig,
    *,
    output_dir: Path,
    num_workers: int = 1,
    dry_run: bool = False,
    crash: bool = False,
) -> Path:
    if num_workers < 1:
        raise ValueError("--num-workers must be >= 1.")
    output_dir = output_dir.expanduser()
    if not dry_run and output_dir.exists() and output_dir.is_dir() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty for hypnodata run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = load_adapter(config)
    records = discover_records(config, adapter=adapter)
    if dry_run:
        write_discovery_preview(output_dir, [_preview_row(record) for record in records])

    if not dry_run:
        for record in records:
            npz_path = npz_record_path(output_dir, record.record_id)
            if npz_path.exists():
                raise FileExistsError(f"Output NPZ already exists for record {record.record_id!r}: {npz_path}")

    started_at = time.time()
    write_hypnodata_progress(
        output_dir,
        status="running",
        total_records=len(records),
        processed_records=0,
        succeeded_records=0,
        failed_records=0,
        skipped_records=0,
        started_at=started_at,
    )
    result_items: list[tuple[int, ProcessResult]] = []
    processed = 0
    failed = 0
    counted_failure = False

    def handle_result(index: int, result: ProcessResult) -> None:
        nonlocal counted_failure, failed, processed
        processed += 1
        if result.failure_row is not None:
            failed += 1
            if crash:
                counted_failure = True
                raise RuntimeError(result.failure_row["message"])
        result_items.append((index, result))
        write_hypnodata_progress(
            output_dir,
            status="running",
            total_records=len(records),
            processed_records=processed,
            succeeded_records=processed - failed,
            failed_records=failed,
            skipped_records=0,
            started_at=started_at,
            current_record_id=result.record_id,
        )

    try:
        if num_workers == 1:
            for index, record in enumerate(records):
                handle_result(
                    index,
                    _process_record(
                        config,
                        record,
                        adapter=adapter,
                        output_dir=output_dir,
                        dry_run=dry_run,
                    ),
                )
        else:
            executor = ThreadPoolExecutor(max_workers=num_workers)
            futures = {
                executor.submit(
                    _process_record,
                    config,
                    record,
                    adapter=adapter,
                    output_dir=output_dir,
                    dry_run=dry_run,
                ): index
                for index, record in enumerate(records)
            }
            try:
                for future in as_completed(futures):
                    handle_result(futures[future], future.result())
            except Exception:
                for future in futures:
                    future.cancel()
                raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
    except Exception as exc:
        failed_records = failed if counted_failure else failed + 1
        write_hypnodata_progress(
            output_dir,
            status="failed",
            total_records=len(records),
            processed_records=processed,
            succeeded_records=processed - failed,
            failed_records=failed_records,
            skipped_records=0,
            started_at=started_at,
            message=str(exc),
        )
        raise

    results = [result for _, result in sorted(result_items, key=lambda item: item[0])]

    record_rows = [result.record_row for result in results if result.record_row is not None]
    signal_rows = [row for result in results for row in result.signal_rows]
    qc_rows = [row for result in results for row in result.qc_rows]
    failure_rows = [result.failure_row for result in results if result.failure_row is not None]
    write_manifests(
        output_dir,
        config,
        record_rows=record_rows,
        signal_rows=signal_rows,
        qc_rows=qc_rows,
        failure_rows=failure_rows,
        dry_run=dry_run,
    )
    write_hypnodata_progress(
        output_dir,
        status="completed",
        total_records=len(records),
        processed_records=processed,
        succeeded_records=len([result for result in results if result.record_row is not None]),
        failed_records=len([result for result in results if result.failure_row is not None]),
        skipped_records=0,
        started_at=started_at,
        message=f"Wrote {output_dir / 'manifest' / 'record_manifest.csv'}",
    )
    return output_dir


def _process_record(
    config: HypnodataConfig,
    record: RecordTask,
    *,
    adapter,
    output_dir: Path,
    dry_run: bool,
) -> ProcessResult:
    npz_path = npz_record_path(output_dir, record.record_id)
    if npz_path.exists() and not dry_run:
        raise FileExistsError(f"Output NPZ already exists for record {record.record_id!r}: {npz_path}")
    try:
        metadata = {**record.metadata, **call_resolve_metadata(adapter, record, config)}
        record = replace(record, metadata=metadata)
        inventories = call_fix_header(adapter, record, _read_inventories(record), config)
        selections, warnings = resolve_channels(config.signals, inventories)
        processed: dict[str, ProcessedSignal] = {}
        qc_rows: list[dict[str, Any]] = []
        for warning in warnings:
            qc_rows.append(
                issue_row(
                    QCIssue(
                        record_id=record.record_id,
                        scope="signal",
                        canonical_channel="",
                        code="channel_ambiguity",
                        severity="warning",
                        message=warning,
                    )
                )
            )
        for canonical, selection in selections.items():
            spec = config.signals[canonical]
            if not selection.available:
                continue
            assert (
                selection.raw_file is not None and selection.raw_label is not None and selection.raw_index is not None
            )
            raw = read_edf_signal(
                Path(selection.raw_file),
                selection.raw_label,
                raw_unit=selection.raw_unit,
                raw_index=selection.raw_index,
            )
            signal = preprocess_signal(raw, selection, spec)
            processed[canonical] = signal
        if processed:
            processed, duration, truncated_channels = truncate_to_common(processed)
            for channel in truncated_channels:
                qc_rows.append(
                    issue_row(
                        QCIssue(
                            record_id=record.record_id,
                            scope="signal",
                            canonical_channel=channel,
                            code="length_mismatch",
                            severity="warning",
                            message="Signal was truncated to the common record duration.",
                        )
                    )
                )
        else:
            try:
                duration = float(record.metadata.get("duration"))
            except (TypeError, ValueError):
                raise ValueError(
                    "Record has no raw signals; record.metadata['duration'] is required for annotation-only output."
                ) from None
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError(
                    "Record has no raw signals; record.metadata['duration'] must be a positive finite number "
                    "for annotation-only output."
                )
        if config.backend.min_duration is not None and duration < config.backend.min_duration:
            raise ValueError(f"Record duration {duration:g}s is shorter than backend.min_duration.")
        annotations = call_read_annotations(adapter, record, config, duration)
        annotation_by_name = _annotation_by_name(annotations.signals)
        _validate_annotations(config, processed, annotation_by_name, duration)
        for canonical, selection in selections.items():
            if selection.required and not selection.available and canonical not in annotation_by_name:
                raise ChannelResolutionError(f"Missing required channel {canonical!r}.")
        arrays = {name: signal.data for name, signal in processed.items()}
        _add_annotation_outputs(arrays, annotation_by_name)
        if not arrays:
            raise ValueError("No available signals to write.")
        output_path = npz_path if dry_run else write_npz_record(output_dir, record.record_id, arrays)
        signal_rows: list[dict[str, Any]] = []
        for canonical, selection in selections.items():
            spec = config.signals[canonical]
            signal = processed.get(canonical)
            annotation = annotation_by_name.get(canonical)
            if annotation is not None:
                signal_rows.append(_annotation_signal_row(record, annotation, spec))
                continue
            if selection.required and signal is None:
                raise ChannelResolutionError(f"Missing required channel {canonical!r}.")
            if selection.available:
                signal_rows.append(
                    _signal_row(record, selection, spec, signal, qc_status="ok" if signal else "missing")
                )
            else:
                signal_rows.append(_signal_row(record, selection, spec, None, qc_status="missing_optional"))
        return ProcessResult(
            record_id=record.record_id,
            record_row=_record_row(
                config,
                record,
                selections,
                output_path,
                duration,
                qc_status="ok",
                annotation_names=set(annotation_by_name),
            ),
            signal_rows=signal_rows,
            qc_rows=qc_rows,
        )
    except Exception as exc:
        if isinstance(exc, ChannelResolutionError):
            code = "channel_resolution"
        elif isinstance(exc, FileNotFoundError):
            code = "file_not_found"
        else:
            code = type(exc).__name__
        return ProcessResult(
            record_id=record.record_id,
            qc_rows=[
                issue_row(
                    QCIssue(
                        record_id=record.record_id,
                        scope="record",
                        canonical_channel="",
                        code=code,
                        severity="error",
                        message=str(exc),
                    )
                )
            ],
            failure_row={
                "record_id": record.record_id,
                "center": record.center,
                "error_type": code,
                "message": str(exc),
            },
        )


def _read_inventories(record: RecordTask) -> dict[str, EdfInventory]:
    inventories = {}
    for key, path in record.files.items():
        if path.suffix.lower() not in {".edf", ".bdf", ".rec"}:
            continue
        inventories[key] = read_edf_inventory(path)
    return inventories


def _record_row(
    config: HypnodataConfig,
    record: RecordTask,
    selections: dict[str, ChannelSelection],
    output_path: Path,
    duration: float,
    *,
    qc_status: str,
    annotation_names: set[str] | None = None,
) -> dict[str, Any]:
    row = {
        "record_id": record.record_id,
        "center": record.center,
        "source": _metadata_value(record, "source", record.center),
        "subject_id": _metadata_value(record, "subject_id", ""),
        "session_id": _metadata_value(record, "session_id", record.record_id),
        "split": _metadata_value(record, "split", ""),
        "path": str(output_path),
        "duration": float(duration),
        "backend": config.backend.type,
        "qc_status": qc_status,
    }
    annotation_names = annotation_names or set()
    for canonical, selection in selections.items():
        row[mask_column_for_channel(canonical)] = int(selection.available or canonical in annotation_names)
    for key, value in sorted(record.metadata.items()):
        if key not in row and _is_manifest_scalar(value):
            row[key] = value
    return row


def _signal_row(
    record: RecordTask,
    selection: ChannelSelection,
    spec,
    signal: ProcessedSignal | None,
    *,
    qc_status: str,
) -> dict[str, Any]:
    scale_applied = spec.scale if selection.available else ""
    polarity_applied = spec.polarity if selection.available else ""
    return {
        "record_id": record.record_id,
        "center": record.center,
        "canonical_channel": selection.canonical_channel,
        "kind": selection.kind,
        "available": int(selection.available),
        "required": int(selection.required),
        "raw_file": selection.raw_file or "",
        "raw_label": selection.raw_label or "",
        "selection_reason": selection.selection_reason,
        "raw_sfreq": selection.raw_sfreq if selection.raw_sfreq is not None else "",
        "target_sfreq": selection.target_sfreq if selection.target_sfreq is not None else "",
        "raw_unit": selection.raw_unit or "",
        "target_unit": selection.target_unit or "",
        "scale_applied": scale_applied,
        "polarity_applied": polarity_applied,
        "raw_n_samples": selection.raw_n_samples if selection.raw_n_samples is not None else "",
        "output_n_samples": "" if signal is None else int(signal.data.shape[0]),
        "preprocess_steps": "" if signal is None else ",".join(signal.steps),
        "qc_status": qc_status,
        "output_key": selection.canonical_channel if signal is not None else "",
        "mask_column": mask_column_for_channel(selection.canonical_channel),
    }


def _annotation_signal_row(record: RecordTask, annotation: AnnotationSignal, spec) -> dict[str, Any]:
    sfreq = "" if annotation.sfreq is None else annotation.sfreq
    configured_sfreq = declared_target_sfreq(spec)
    target_sfreq = configured_sfreq if configured_sfreq is not None else sfreq
    output_key = annotation.output_key or annotation.canonical_channel
    return {
        "record_id": record.record_id,
        "center": record.center,
        "canonical_channel": annotation.canonical_channel,
        "kind": spec.kind,
        "available": 1,
        "required": int(spec.required),
        "raw_file": annotation.raw_file,
        "raw_label": annotation.raw_label,
        "selection_reason": "annotation",
        "raw_sfreq": sfreq,
        "target_sfreq": target_sfreq,
        "raw_unit": annotation.unit or "",
        "target_unit": spec.target_unit or "",
        "scale_applied": "",
        "polarity_applied": "",
        "raw_n_samples": int(annotation.data.shape[0]),
        "output_n_samples": int(annotation.data.shape[0]),
        "preprocess_steps": ",".join(annotation.steps),
        "qc_status": "ok",
        "output_key": output_key,
        "mask_column": mask_column_for_channel(annotation.canonical_channel),
    }


def _add_annotation_outputs(arrays: dict[str, np.ndarray], annotations: dict[str, AnnotationSignal]) -> None:
    for annotation in annotations.values():
        output_key = annotation.output_key or annotation.canonical_channel
        _add_output_array(arrays, output_key, annotation.data)
        for key, value in annotation.extra_outputs.items():
            _add_output_array(arrays, key, np.asarray(value))


def _add_output_array(arrays: dict[str, np.ndarray], key: str, value: np.ndarray) -> None:
    if key in arrays:
        raise ValueError(f"Duplicate output key {key!r}.")
    arrays[key] = value


def _annotation_by_name(signals: list[AnnotationSignal]) -> dict[str, AnnotationSignal]:
    annotations: dict[str, AnnotationSignal] = {}
    for signal in signals:
        if signal.canonical_channel in annotations:
            raise ValueError(f"Duplicate annotation channel {signal.canonical_channel!r}.")
        annotations[signal.canonical_channel] = signal
    return annotations


def _validate_annotations(
    config: HypnodataConfig,
    processed: dict[str, ProcessedSignal],
    annotations: dict[str, AnnotationSignal],
    duration: float,
) -> None:
    for canonical, annotation in annotations.items():
        if canonical not in config.signals:
            raise ValueError(f"Annotation channel {canonical!r} must be declared under signals.")
        if canonical in processed:
            raise ValueError(f"Annotation channel {canonical!r} duplicates a raw signal output.")
        if annotation.materialization not in ANNOTATION_MATERIALIZATIONS:
            raise ValueError(f"Annotation channel {canonical!r} has unknown materialization.")
        spec = config.signals[canonical]
        if spec.kind != annotation.materialization:
            raise ValueError(
                f"Annotation channel {canonical!r} materialization {annotation.materialization!r} "
                f"does not match configured kind {spec.kind!r}."
            )
        configured_sfreq = declared_target_sfreq(spec)
        if configured_sfreq is not None:
            if annotation.sfreq is None:
                raise ValueError(f"Annotation channel {canonical!r} does not have a sampling frequency.")
            if not np.isclose(float(configured_sfreq), float(annotation.sfreq)):
                raise ValueError(
                    f"Annotation channel {canonical!r} sfreq {annotation.sfreq:g} "
                    f"does not match configured output frequency {configured_sfreq:g}."
                )
        _validate_annotation_shape(canonical, annotation)
        _validate_annotation_duration(canonical, annotation, duration)


def _validate_annotation_shape(canonical: str, annotation: AnnotationSignal) -> None:
    if annotation.materialization in {"stage", "event_dense", "ahi"}:
        if annotation.data.ndim != 1:
            raise ValueError(f"Annotation channel {canonical!r} must be one-dimensional.")
    elif annotation.materialization == "event_table":
        if annotation.data.ndim != 2 or annotation.data.shape[1] != 3:
            raise ValueError(f"Annotation channel {canonical!r} must have shape (N, 3).")
    elif annotation.materialization == "event_anchor":
        if annotation.data.ndim != 2 or annotation.data.shape[1] < 3 or annotation.data.shape[1] % 3 != 0:
            raise ValueError(f"Annotation channel {canonical!r} must have 3 columns per anchor.")
    if annotation.materialization == "ahi":
        _validate_ahi_outputs(canonical, annotation)
    else:
        # Non-AHI annotations must keep NPZ keys aligned with canonical manifest and mask contracts.
        if annotation.output_key not in {None, canonical}:
            raise ValueError(f"Annotation channel {canonical!r} must write to its canonical output key.")
        if annotation.extra_outputs:
            raise ValueError(f"Annotation channel {canonical!r} has unexpected extra outputs.")


def _validate_ahi_outputs(canonical: str, annotation: AnnotationSignal) -> None:
    if annotation.output_key != "ah_event":
        raise ValueError(f"Annotation channel {canonical!r} must write output_key='ah_event'.")
    if set(annotation.extra_outputs) != {"ahi", "tst"}:
        raise ValueError(f"Annotation channel {canonical!r} must provide scalar 'ahi' and 'tst' outputs.")
    ahi = np.asarray(annotation.extra_outputs["ahi"])
    tst = np.asarray(annotation.extra_outputs["tst"])
    if ahi.ndim != 0 or tst.ndim != 0:
        raise ValueError(f"Annotation channel {canonical!r} must provide scalar 'ahi' and 'tst' outputs.")
    ahi_value = float(ahi)
    tst_value = float(tst)
    if not np.isfinite(ahi_value) or ahi_value < 0:
        raise ValueError(f"Annotation channel {canonical!r} has invalid scalar 'ahi'.")
    if not np.isfinite(tst_value) or tst_value <= 0:
        raise ValueError(f"Annotation channel {canonical!r} has invalid scalar 'tst'.")


def _validate_annotation_duration(canonical: str, annotation: AnnotationSignal, duration: float) -> None:
    tolerance = 1e-6
    if annotation.materialization in {"stage", "event_dense", "ahi"}:
        if annotation.sfreq is None:
            return
        # Stage and dense timelines use floor-sized arrays, matching their materializers.
        # Short arrays would make record_manifest advertise labels beyond the stored timeline.
        expected_samples = int(np.floor(float(duration) * float(annotation.sfreq) + tolerance))
        actual_samples = int(annotation.data.shape[0])
        if actual_samples != expected_samples:
            raise ValueError(
                f"Annotation channel {canonical!r} length {actual_samples} does not match record duration "
                f"{duration:g}s; expected {expected_samples} samples."
            )
    elif annotation.materialization == "event_anchor":
        if annotation.sfreq is None:
            return
        # Anchor labels use ceil-sized windows, including the final partial window.
        expected_windows = int(np.ceil(max(float(duration) * float(annotation.sfreq) - tolerance, 0.0)))
        actual_windows = int(annotation.data.shape[0])
        if actual_windows != expected_windows:
            raise ValueError(
                f"Annotation channel {canonical!r} length {actual_windows} does not match record duration "
                f"{duration:g}s; expected {expected_windows} windows."
            )
    elif annotation.materialization == "event_table" and annotation.data.size:
        # Event tables carry second-based extents directly.
        starts = annotation.data[:, 1]
        durations = annotation.data[:, 2]
        stops = starts + durations
        invalid_extents = (
            not np.isfinite(starts).all()
            or not np.isfinite(durations).all()
            or (starts < 0).any()
            or (durations <= 0).any()
        )
        if invalid_extents:
            raise ValueError(f"Annotation channel {canonical!r} contains invalid event extents.")
        if (stops > float(duration) + tolerance).any():
            raise ValueError(f"Annotation channel {canonical!r} exceeds record duration {duration:g}s.")


def _metadata_value(record: RecordTask, key: str, default: Any) -> Any:
    value = record.metadata.get(key, default)
    return default if value is None else value


def _is_manifest_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _preview_row(record: RecordTask) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "center": record.center,
        "files": {key: str(path) for key, path in record.files.items()},
        **record.metadata,
    }
