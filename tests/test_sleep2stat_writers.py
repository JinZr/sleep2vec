from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleep2stat.cli import _csv_row_count
from sleep2stat.config import AnalyzerConfig, DataConfig, OutputsConfig, RunConfig, SignalsConfig, Sleep2statConfig
from sleep2stat.core.artifacts import AnalyzerResult, FailureRecord
from sleep2stat.io.records import SleepRecord
from sleep2stat.io.writers import AnalysisBundleWriter


def _config(tmp_path: Path) -> Sleep2statConfig:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run: {}\n")
    return Sleep2statConfig(
        path=config_path,
        run=RunConfig(name="unit", output_dir=tmp_path / "run"),
        data=DataConfig(
            backend="npz",
            index=tmp_path / "index.csv",
            split=["test"],
            path_column="path",
            duration_column="duration",
            split_column="split",
            token_sec=30,
            max_tokens=2,
        ),
        signals=SignalsConfig(channels={}),
        analyzers=[
            AnalyzerConfig(
                name="stage5_model",
                type="sleep2vec_downstream",
                namespace="sleep2vec2",
                label_name="stage5",
                config=Path("config.yaml"),
                ckpt_path=Path("model.ckpt"),
                input_channels=["ppg"],
            )
        ],
        reducers=[],
        outputs=OutputsConfig(
            write_global_tables=True,
            write_per_record=True,
            compression="gzip",
            global_tables={
                "epoch_alignment": True,
                "second_alignment": True,
                "event_alignment": True,
                "night_stats": True,
            },
        ),
    )


def _record(record_id: str = "rec1", *, source: str = "unit", metadata: dict | None = None) -> SleepRecord:
    return SleepRecord(
        record_id=record_id,
        path=Path(f"{record_id}.npz"),
        split="test",
        source=source,
        duration_sec=60,
        token_sec=30,
        max_tokens=2,
        metadata=metadata or {},
    )


def test_writer_creates_global_and_per_record_tables(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    records = [_record()]
    epoch = pd.DataFrame(
        {
            "record_id": ["rec1", "rec1"],
            "path": ["rec1.npz", "rec1.npz"],
            "token_idx": [0, 1],
            "start_sec": [0.0, 30.0],
            "end_sec": [30.0, 60.0],
            "stage5_model_pred": [0, 2],
        }
    )
    results = [
        AnalyzerResult(
            "stage5_model",
            "rec1",
            epoch=epoch,
            events=pd.DataFrame(
                {
                    "record_id": ["rec1"],
                    "event_id": ["rec1__event0"],
                    "onset_sec": [5.0],
                    "offset_sec": [15.0],
                }
            ),
            night={"stage5_model_TST_min": 0.5},
            arrays={"probabilities": np.ones((2, 5), dtype=np.float32)},
        ),
    ]

    writer.write_record_manifest(records)
    writer.write_results(records, results)
    writer.write_failures([FailureRecord("rec1", "stage5_model", "ValueError", "bad")])
    writer.write_run_manifest(status="completed_with_failures", records=records, failures=[], dry_run=False)

    assert (config.run.output_dir / "record_manifest.csv").exists()
    assert (config.run.output_dir / "tables" / "epoch_alignment.csv.gz").exists()
    assert (config.run.output_dir / "tables" / "night_stats.csv").exists()
    assert (config.run.output_dir / "tables" / "model_summary.csv").exists()
    assert (config.run.output_dir / "tables" / "analyzer_summary.csv").exists()
    assert (config.run.output_dir / "per_record" / "rec1" / "epoch_alignment.csv.gz").exists()
    assert (config.run.output_dir / "per_record" / "rec1" / "events.csv.gz").exists()
    assert (config.run.output_dir / "per_record" / "rec1" / "night_stats.json").exists()
    assert (config.run.output_dir / "per_record" / "rec1" / "arrays.npz").exists()
    assert (config.run.output_dir / "per_record" / "rec1" / "result_manifest.csv").exists()
    assert len(pd.read_csv(config.run.output_dir / "tables" / "epoch_alignment.csv.gz")) == 2
    assert pd.read_csv(config.run.output_dir / "tables" / "analyzer_summary.csv")["result_count"].tolist() == [1]
    arrays = np.load(config.run.output_dir / "per_record" / "rec1" / "arrays.npz")
    assert "stage5_model__probabilities" in arrays


def test_csv_row_count_treats_empty_tables_as_zero(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("")

    assert _csv_row_count(path) == 0


def test_writer_skip_existing_filters_records_and_merges_global_tables(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    record = _record()
    existing_epoch = pd.DataFrame(
        {
            "record_id": ["rec1"],
            "path": ["rec1.npz"],
            "token_idx": [0],
            "start_sec": [0.0],
            "end_sec": [30.0],
            "stage5_model_pred": [1],
        }
    )
    writer.write_results([record], [AnalyzerResult("stage5_model", "rec1", epoch=existing_epoch)])
    writer.write_completion_markers([record.record_id])

    assert writer.filter_records_for_run([record]) == []

    new_epoch = pd.DataFrame(
        {
            "record_id": ["rec2"],
            "path": ["rec2.npz"],
            "token_idx": [0],
            "start_sec": [0.0],
            "end_sec": [30.0],
            "stage5_model_pred": [2],
        }
    )
    record2 = SleepRecord("rec2", Path("rec2.npz"), "test", "unit", 30, 30, 1, {})
    writer.write_results([record2], [AnalyzerResult("stage5_model", "rec2", epoch=new_epoch)])

    frame = pd.read_csv(config.run.output_dir / "tables" / "epoch_alignment.csv.gz")
    assert sorted(frame["record_id"].tolist()) == ["rec1", "rec2"]


def test_writer_rebuilds_global_alignment_with_sparse_chunk_columns(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    rec1 = _record()
    rec2 = SleepRecord("rec2", Path("rec2.npz"), "test", "unit", 30, 30, 1, {})
    epoch1 = pd.DataFrame(
        {
            "record_id": ["rec1"],
            "path": ["rec1.npz"],
            "token_idx": [0],
            "start_sec": [0.0],
            "end_sec": [30.0],
            "stage5_model_pred": [1],
        }
    )
    epoch2 = pd.DataFrame(
        {
            "record_id": ["rec2"],
            "path": ["rec2.npz"],
            "token_idx": [0],
            "start_sec": [0.0],
            "end_sec": [30.0],
            "stage5_model_pred": [2],
            "stage5_model_prob_REM": [0.7],
        }
    )

    writer.write_results([rec1], [AnalyzerResult("stage5_model", "rec1", epoch=epoch1)])
    writer.write_results([rec2], [AnalyzerResult("stage5_model", "rec2", epoch=epoch2)])

    frame = pd.read_csv(config.run.output_dir / "tables" / "epoch_alignment.csv.gz")
    assert "stage5_model_prob_REM" in frame.columns
    assert pd.isna(frame.loc[frame["record_id"] == "rec1", "stage5_model_prob_REM"].item())
    assert frame.loc[frame["record_id"] == "rec2", "stage5_model_prob_REM"].item() == 0.7


def test_writer_skip_existing_does_not_treat_empty_per_record_outputs_as_complete(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    record = _record()

    writer.write_results([record], [])

    assert writer.filter_records_for_run([record]) == [record]
    assert not (config.run.output_dir / "per_record" / "rec1" / "_SUCCESS.json").exists()


def test_writer_rebuilds_cumulative_summary_across_resume(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    rec1 = _record()
    rec2 = SleepRecord("rec2", Path("rec2.npz"), "test", "unit", 30, 30, 1, {})

    writer.write_results([rec1], [AnalyzerResult("stage5_model", "rec1", night={"stage5_model_TST_min": 1.0})])
    writer.write_results([rec2], [AnalyzerResult("stage5_model", "rec2", night={"stage5_model_TST_min": 1.0})])

    summary = pd.read_csv(config.run.output_dir / "tables" / "analyzer_summary.csv")
    night_stats = pd.read_csv(config.run.output_dir / "tables" / "night_stats.csv")
    assert summary.loc[summary["name"] == "stage5_model", "record_count"].item() == 2
    assert summary.loc[summary["name"] == "stage5_model", "result_count"].item() == 2
    assert sorted(night_stats["record_id"].tolist()) == ["rec1", "rec2"]

    writer.rebuild_global_tables([rec2], [])
    night_stats = pd.read_csv(config.run.output_dir / "tables" / "night_stats.csv")
    assert sorted(night_stats["record_id"].tolist()) == ["rec1", "rec2"]


def test_writer_resume_from_archived_config_skips_same_file_copy(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    archived_config = Sleep2statConfig(
        path=config.run.output_dir / "config.yaml",
        run=config.run,
        data=config.data,
        signals=config.signals,
        analyzers=config.analyzers,
        reducers=config.reducers,
        outputs=config.outputs,
    )

    writer = AnalysisBundleWriter(archived_config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())

    assert (config.run.output_dir / "config.yaml").exists()


def test_writer_drops_resolved_failures_after_successful_resume(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    record = _record()

    writer.write_failures([FailureRecord("rec1", "stage5_model", "ValueError", "old failure")])
    writer.write_results([record], [AnalyzerResult("stage5_model", "rec1", night={"stage5_model_TST_min": 1.0})])
    writer.write_failures([])
    writer.rebuild_global_tables([record], [])

    failures = pd.read_csv(config.run.output_dir / "status" / "failures.csv")
    summary = pd.read_csv(config.run.output_dir / "tables" / "analyzer_summary.csv")
    assert failures.empty
    assert summary.loc[summary["name"] == "stage5_model", "failure_count"].item() == 0


def test_writer_preserves_record_manifest_rows_across_subset_resume(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())

    writer.write_record_manifest([_record("rec1", source="site_a"), _record("rec2", source="site_b")])
    writer.write_record_manifest([_record("rec2", source="site_c")])

    manifest = pd.read_csv(config.run.output_dir / "record_manifest.csv")
    by_record = manifest.set_index("record_id")
    assert set(by_record.index) == {"rec1", "rec2"}
    assert by_record.loc["rec1", "source"] == "site_a"
    assert by_record.loc["rec2", "source"] == "site_c"


def test_writer_drops_resolved_global_failure_after_successful_resume(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    record = _record()

    writer.write_failures([FailureRecord("__all__", "stage5_model", "FileNotFoundError", "missing")])
    writer.write_results([record], [AnalyzerResult("stage5_model", "rec1", night={"stage5_model_TST_min": 1.0})])
    writer.write_failures([])
    writer.rebuild_global_tables([record], [])

    failures = pd.read_csv(config.run.output_dir / "status" / "failures.csv")
    summary = pd.read_csv(config.run.output_dir / "tables" / "analyzer_summary.csv")
    assert failures.empty
    assert summary.loc[summary["name"] == "stage5_model", "failure_count"].item() == 0


def test_writer_keeps_unresolved_global_failure(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())

    writer.write_failures([FailureRecord("__all__", "stage5_model", "FileNotFoundError", "missing")])
    writer.rebuild_global_tables([], [])

    failures = pd.read_csv(config.run.output_dir / "status" / "failures.csv")
    summary = pd.read_csv(config.run.output_dir / "tables" / "analyzer_summary.csv")
    assert failures["record_id"].tolist() == ["__all__"]
    assert summary.loc[summary["name"] == "stage5_model", "failure_count"].item() == 1


def test_writer_keeps_failed_record_out_of_global_alignment(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    record = _record()
    epoch = pd.DataFrame(
        {
            "record_id": ["rec1"],
            "path": ["rec1.npz"],
            "token_idx": [0],
            "start_sec": [0.0],
            "end_sec": [30.0],
            "stage5_model_pred": [1],
        }
    )

    writer.write_chunk(
        [record],
        [AnalyzerResult("stage5_model", "rec1", epoch=epoch)],
        [FailureRecord("rec1", "stage5_model", "ValueError", "bad")],
        completed_record_ids=set(),
    )
    writer.rebuild_global_tables([record], [FailureRecord("rec1", "stage5_model", "ValueError", "bad")])

    assert (config.run.output_dir / "per_record" / "rec1" / "epoch_alignment.csv.gz").exists()
    assert not (config.run.output_dir / "per_record" / "rec1" / "_SUCCESS.json").exists()
    assert not (config.run.output_dir / "tables" / "epoch_alignment.csv.gz").exists()
    summary = pd.read_csv(config.run.output_dir / "tables" / "analyzer_summary.csv")
    assert summary.loc[summary["name"] == "stage5_model", "result_count"].item() == 0
    assert summary.loc[summary["name"] == "stage5_model", "failure_count"].item() == 1


def test_writer_default_global_tables_skip_epoch_and_second(tmp_path: Path):
    config = _config(tmp_path)
    config = Sleep2statConfig(
        path=config.path,
        run=config.run,
        data=config.data,
        signals=config.signals,
        analyzers=config.analyzers,
        reducers=config.reducers,
        outputs=OutputsConfig(write_global_tables=True, write_per_record=True, compression="gzip"),
    )
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    record = _record()
    epoch = pd.DataFrame(
        {
            "record_id": ["rec1"],
            "path": ["rec1.npz"],
            "token_idx": [0],
            "start_sec": [0.0],
            "end_sec": [30.0],
            "stage5_model_pred": [1],
        }
    )

    writer.write_results([record], [AnalyzerResult("stage5_model", "rec1", epoch=epoch)])

    assert (config.run.output_dir / "per_record" / "rec1" / "epoch_alignment.csv.gz").exists()
    assert not (config.run.output_dir / "tables" / "epoch_alignment.csv.gz").exists()
    assert not (config.run.output_dir / "tables" / "second_alignment.csv.gz").exists()
    assert (config.run.output_dir / "tables" / "night_stats.csv").exists()


def test_writer_resume_rejects_config_fingerprint_mismatch(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    config.path.write_text("run:\n  name: changed\n")

    writer = AnalysisBundleWriter(config)

    with pytest.raises(ValueError, match="different config fingerprint"):
        writer.prepare(args=type("Args", (), {"dry_run": False})())
