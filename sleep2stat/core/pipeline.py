from __future__ import annotations

import argparse

from sleep2stat import analyzers, reducers  # noqa: F401
from sleep2stat.config import Sleep2statConfig
from sleep2stat.core.artifacts import AnalyzerResult, FailureRecord
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
    results: list[AnalyzerResult] = []
    if args.dry_run:
        writer.write_failures(failures)
        writer.write_results(records, results, failures)
        writer.write_progress(total_records=len(records), completed_records=skipped_records, status="dry_run")
        writer.write_run_manifest(status="dry_run", records=records, failures=failures, dry_run=True)
        return writer.run_dir

    context = Sleep2statContext(
        config=config, device=args.device, num_workers=args.num_workers, batch_size=args.batch_size
    )
    for analyzer_cfg in config.analyzers:
        if not analyzer_cfg.enabled:
            continue
        analyzer = create_analyzer(analyzer_cfg)
        try:
            analyzer.prepare(context)
            analyzer_results, analyzer_failures = analyzer.run(pending_records, context)
            results.extend(analyzer_results)
            failures.extend(analyzer_failures)
        except Exception as exc:
            failures.append(
                FailureRecord(
                    record_id="__all__",
                    analyzer=analyzer_cfg.name,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
        finally:
            analyzer.close()

    for reducer_cfg in config.reducers:
        if not reducer_cfg.enabled:
            continue
        try:
            reducer = create_reducer(reducer_cfg)
            results.extend(reducer.reduce(pending_records, results, context))
        except Exception as exc:
            failures.append(
                FailureRecord(
                    record_id="__all__",
                    analyzer=reducer_cfg.name,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )

    writer.write_results(pending_records, results, failures)
    has_global_failure = any(failure.record_id == "__all__" for failure in failures)
    failed_record_ids = {failure.record_id for failure in failures if failure.record_id != "__all__"}
    result_record_ids = {result.record_id for result in results}
    completed_record_ids = set()
    if not has_global_failure:
        completed_record_ids = {
            record.record_id
            for record in pending_records
            if record.record_id in result_record_ids and record.record_id not in failed_record_ids
        }
    writer.write_completion_markers(completed_record_ids)
    completed = skipped_records + len(completed_record_ids)
    writer.write_failures(failures)
    status = "completed_with_failures" if failures else "completed"
    writer.write_progress(total_records=len(records), completed_records=completed, status=status)
    writer.write_run_manifest(status=status, records=records, failures=failures, dry_run=False)
    return writer.run_dir
