from __future__ import annotations

import argparse

from sleep2stat import analyzers, reducers  # noqa: F401
from sleep2stat.config import Sleep2statConfig
from sleep2stat.core.artifacts import FailureRecord
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import load_records
from sleep2stat.io.writers import AnalysisBundleWriter
from sleep2stat.registry import create_analyzer, create_reducer


def run_pipeline(config: Sleep2statConfig, args: argparse.Namespace):
    split_override = list(args.split) if getattr(args, "split", None) else None
    records = load_records(config.data, split_override=split_override, limit=args.limit_records)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=args)
    pending_records = writer.filter_records_for_run(records)
    skipped_records = len(records) - len(pending_records)
    writer.write_record_manifest(records)
    writer.write_progress(
        total_records=len(records), completed_records=skipped_records, status="dry_run" if args.dry_run else "running"
    )

    failures: list[FailureRecord] = []
    if args.dry_run:
        results = []
        writer.write_failures(failures)
        writer.write_results(records, results, failures)
        writer.write_progress(total_records=len(records), completed_records=skipped_records, status="dry_run")
        writer.write_run_manifest(status="dry_run", records=records, failures=failures, dry_run=True)
        return writer.run_dir

    if not pending_records:
        writer.write_failures(failures)
        writer.rebuild_global_tables(records, failures)
        writer.write_progress(total_records=len(records), completed_records=skipped_records, status="completed")
        writer.write_run_manifest(status="completed", records=records, failures=failures, dry_run=False)
        return writer.run_dir

    context = Sleep2statContext(
        config=config, device=args.device, num_workers=args.num_workers, batch_size=args.batch_size
    )
    prepared_analyzers = []
    for analyzer_cfg in config.analyzers:
        if not analyzer_cfg.enabled:
            continue
        analyzer = create_analyzer(analyzer_cfg)
        try:
            analyzer.prepare(context)
        except Exception as exc:
            failures.append(
                FailureRecord(
                    record_id="__all__",
                    analyzer=analyzer_cfg.name,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            analyzer.close()
        else:
            prepared_analyzers.append(analyzer)

    completed_record_ids: set[str] = set()
    if not any(failure.record_id == "__all__" for failure in failures):
        for chunk in _record_chunks(pending_records, _chunk_size(config, args)):
            chunk_results = []
            chunk_failures: list[FailureRecord] = []
            for analyzer in prepared_analyzers:
                try:
                    analyzer_results, analyzer_failures = analyzer.run(
                        chunk, context, prior_results=list(chunk_results)
                    )
                    chunk_results.extend(analyzer_results)
                    chunk_failures.extend(analyzer_failures)
                except Exception as exc:
                    chunk_failures.extend(
                        FailureRecord(
                            record_id=record.record_id,
                            analyzer=analyzer.config.name,
                            error_type=type(exc).__name__,
                            message=str(exc),
                        )
                        for record in chunk
                    )
            for reducer_cfg in config.reducers:
                if not reducer_cfg.enabled:
                    continue
                try:
                    reducer = create_reducer(reducer_cfg)
                    chunk_results.extend(reducer.reduce(chunk, chunk_results, context))
                except Exception:
                    base_results = list(chunk_results)
                    for record in chunk:
                        record_results = [result for result in base_results if result.record_id == record.record_id]
                        try:
                            reducer = create_reducer(reducer_cfg)
                            chunk_results.extend(reducer.reduce([record], record_results, context))
                        except Exception as record_exc:
                            chunk_failures.append(
                                FailureRecord(
                                    record_id=record.record_id,
                                    analyzer=reducer_cfg.name,
                                    error_type=type(record_exc).__name__,
                                    message=str(record_exc),
                                )
                            )
            failed_record_ids = {failure.record_id for failure in chunk_failures if failure.record_id != "__all__"}
            result_record_ids = {result.record_id for result in chunk_results}
            chunk_completed = {
                record.record_id
                for record in chunk
                if record.record_id in result_record_ids and record.record_id not in failed_record_ids
            }
            writer.write_chunk(chunk, chunk_results, chunk_failures, completed_record_ids=chunk_completed)
            writer.write_completion_markers(chunk_completed)
            completed_record_ids.update(chunk_completed)
            failures.extend(chunk_failures)
            writer.write_progress(
                total_records=len(records),
                completed_records=skipped_records + len(completed_record_ids),
                status="running",
            )

    for analyzer in prepared_analyzers:
        analyzer.close()

    writer.write_failures(failures)
    writer.rebuild_global_tables(records, failures)
    completed = skipped_records + len(completed_record_ids)
    status = "completed_with_failures" if failures else "completed"
    writer.write_progress(total_records=len(records), completed_records=completed, status=status)
    writer.write_run_manifest(status=status, records=records, failures=failures, dry_run=False)
    return writer.run_dir


def _chunk_size(config: Sleep2statConfig, args: argparse.Namespace) -> int:
    configured = [analyzer.batch_size for analyzer in config.analyzers if analyzer.enabled and analyzer.batch_size]
    if getattr(args, "batch_size", None):
        configured.append(int(args.batch_size))
    return min(max(configured or [1]), 32)


def _record_chunks(records, chunk_size: int):
    for start in range(0, len(records), chunk_size):
        yield records[start : start + chunk_size]
