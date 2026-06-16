from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from sleep2stat import analyzers, reducers  # noqa: F401
from sleep2stat.config import Sleep2statConfig
from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import load_records
from sleep2stat.io.writers import AnalysisBundleWriter
from sleep2stat.registry import create_analyzer, create_reducer


def run_pipeline(config: Sleep2statConfig, args: argparse.Namespace):
    split_override = list(args.split) if getattr(args, "split", None) else None
    records = load_records(config.data, split_override=split_override, limit=args.limit_records)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=args)
    writer.write_record_manifest(records)
    execution_split = _execution_split(config, args)
    writer.write_progress(
        total_records=len(records),
        completed_records=0,
        status="dry_run" if args.dry_run else "running",
        num_workers=args.num_workers,
        execution_split=execution_split,
    )

    if args.dry_run:
        results = []
        writer.write_results(records, results)
        writer.write_progress(
            total_records=len(records),
            completed_records=0,
            status="dry_run",
            num_workers=args.num_workers,
            execution_split=execution_split,
        )
        writer.write_run_manifest(
            status="dry_run",
            records=records,
            dry_run=True,
            num_workers=args.num_workers,
            execution_split=execution_split,
        )
        return writer.run_dir

    context = Sleep2statContext(
        config=config, device=args.device, num_workers=args.num_workers, batch_size=args.batch_size
    )
    if _use_record_split(config, args):
        completed_record_ids = _run_record_split(
            records,
            context,
            writer,
            total_records=len(records),
            chunk_size=_chunk_size(config, args),
        )
        writer.rebuild_global_tables(records, num_workers=args.num_workers)
        completed = len(completed_record_ids)
        writer.write_progress(
            total_records=len(records),
            completed_records=completed,
            status="completed",
            num_workers=args.num_workers,
            execution_split=execution_split,
        )
        writer.write_run_manifest(
            status="completed",
            records=records,
            dry_run=False,
            num_workers=args.num_workers,
            execution_split=execution_split,
        )
        return writer.run_dir

    prepared_analyzers = []
    completed_record_ids: set[str] = set()
    try:
        for analyzer_cfg in config.analyzers:
            if not analyzer_cfg.enabled:
                continue
            analyzer = create_analyzer(analyzer_cfg)
            try:
                analyzer.prepare(context)
            except Exception:
                analyzer.close()
                raise
            prepared_analyzers.append(analyzer)

        for chunk in _record_chunks(records, _chunk_size(config, args)):
            chunk_results = []
            for analyzer in prepared_analyzers:
                chunk_results.extend(analyzer.run(chunk, context, prior_results=list(chunk_results)))
            for reducer_cfg in config.reducers:
                if not reducer_cfg.enabled:
                    continue
                reducer = create_reducer(reducer_cfg)
                chunk_results.extend(reducer.reduce(chunk, chunk_results, context))
            chunk_completed = {record.record_id for record in chunk}
            writer.write_chunk(chunk, chunk_results, completed_record_ids=chunk_completed)
            completed_record_ids.update(chunk_completed)
            writer.write_progress(
                total_records=len(records),
                completed_records=len(completed_record_ids),
                status="running",
                num_workers=args.num_workers,
                execution_split=execution_split,
            )
    finally:
        for analyzer in prepared_analyzers:
            analyzer.close()

    writer.rebuild_global_tables(records, num_workers=args.num_workers)
    completed = len(completed_record_ids)
    writer.write_progress(
        total_records=len(records),
        completed_records=completed,
        status="completed",
        num_workers=args.num_workers,
        execution_split=execution_split,
    )
    writer.write_run_manifest(
        status="completed",
        records=records,
        dry_run=False,
        num_workers=args.num_workers,
        execution_split=execution_split,
    )
    return writer.run_dir


def _use_record_split(config: Sleep2statConfig, args: argparse.Namespace) -> bool:
    if int(getattr(args, "num_workers", 0) or 0) <= 1:
        return False
    return not any(analyzer.enabled and analyzer.type == "sleep2vec_downstream" for analyzer in config.analyzers)


def _execution_split(config: Sleep2statConfig, args: argparse.Namespace) -> str:
    if _use_record_split(config, args):
        return f"split{int(args.num_workers)}"
    return "sequential"


def _run_record_split(
    records,
    context: Sleep2statContext,
    writer: AnalysisBundleWriter,
    *,
    total_records: int,
    chunk_size: int,
) -> set[str]:
    completed_record_ids: set[str] = set()
    buffered_records = []
    buffered_results = []
    buffered_completed: set[str] = set()

    with ThreadPoolExecutor(max_workers=context.num_workers) as executor:
        futures = []
        for split in _record_splits(records, context.num_workers):
            for record in split:
                futures.append(executor.submit(_run_one_record, record, context))
        with tqdm(total=len(futures), desc=f"sleep2stat split{context.num_workers}", unit="record") as progress:
            for future in as_completed(futures):
                try:
                    record, results = future.result()
                except Exception:
                    # A split-worker failure invalidates the run; stop queued work before re-raising.
                    for pending in futures:
                        if pending is not future:
                            pending.cancel()
                    raise
                record_completed = {record.record_id}
                buffered_records.append(record)
                buffered_results.extend(results)
                buffered_completed.update(record_completed)
                completed_record_ids.update(record_completed)
                progress.update(1)
                if len(buffered_records) >= chunk_size:
                    _write_pending(
                        writer,
                        buffered_records,
                        buffered_results,
                        buffered_completed,
                    )
                    writer.write_progress(
                        total_records=total_records,
                        completed_records=len(completed_record_ids),
                        status="running",
                        num_workers=context.num_workers,
                        execution_split=f"split{context.num_workers}",
                    )
                    buffered_records = []
                    buffered_results = []
                    buffered_completed = set()
    _write_pending(writer, buffered_records, buffered_results, buffered_completed)
    return completed_record_ids


def _write_pending(
    writer: AnalysisBundleWriter,
    records,
    results: list[AnalyzerResult],
    completed_record_ids: set[str],
) -> None:
    if not records:
        return
    writer.write_chunk(records, results, completed_record_ids=completed_record_ids)


def _run_one_record(record, context: Sleep2statContext):
    results: list[AnalyzerResult] = []
    for analyzer_cfg in context.config.analyzers:
        if not analyzer_cfg.enabled:
            continue
        analyzer = create_analyzer(analyzer_cfg)
        try:
            analyzer.prepare(context)
            results.extend(analyzer.run([record], context, prior_results=list(results)))
        finally:
            analyzer.close()

    for reducer_cfg in context.config.reducers:
        if not reducer_cfg.enabled:
            continue
        reducer = create_reducer(reducer_cfg)
        results.extend(reducer.reduce([record], results, context))
    return record, results


def _chunk_size(config: Sleep2statConfig, args: argparse.Namespace) -> int:
    configured = [analyzer.batch_size for analyzer in config.analyzers if analyzer.enabled and analyzer.batch_size]
    if getattr(args, "batch_size", None):
        configured.append(int(args.batch_size))
    return min(max(configured or [1]), 32)


def _record_chunks(records, chunk_size: int):
    for start in range(0, len(records), chunk_size):
        yield records[start : start + chunk_size]


def _record_splits(records, num_workers: int):
    return [records[start::num_workers] for start in range(num_workers)]
