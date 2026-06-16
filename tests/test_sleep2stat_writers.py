import json
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
    assert (config.run.output_dir / "run_manifest.json").exists()
    assert not list(config.run.output_dir.glob(".run_manifest.json.tmp.*"))
    assert (config.run.output_dir / "tables" / "epoch_alignment.csv.gz").exists()
    assert (config.run.output_dir / "tables" / "night_stats.csv").exists()
    assert (config.run.output_dir / "tables" / "model_summary.csv").exists()
    assert (config.run.output_dir / "tables" / "analyzer_summary.csv").exists()
    assert (config.run.output_dir / "per_record" / "rec1" / "epoch_alignment.csv.gz").exists()
    assert (config.run.output_dir / "per_record" / "rec1" / "events.csv.gz").exists()
    assert (config.run.output_dir / "per_record" / "rec1" / "night_stats.json").exists()
    assert (config.run.output_dir / "per_record" / "rec1" / "arrays.npz").exists()
    assert (config.run.output_dir / "per_record" / "rec1" / "result_manifest.csv").exists()
    assert not (config.run.output_dir / "per_record" / "rec1" / "_SUCCESS.json").exists()
    assert len(pd.read_csv(config.run.output_dir / "tables" / "epoch_alignment.csv.gz")) == 2
    assert pd.read_csv(config.run.output_dir / "tables" / "analyzer_summary.csv")["result_count"].tolist() == [1]
    arrays = np.load(config.run.output_dir / "per_record" / "rec1" / "arrays.npz")
    assert "stage5_model__probabilities" in arrays


def test_writer_progress_and_pid_json_are_complete(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())

    writer.write_progress(total_records=2, completed_records=1, status="running", num_workers=2)

    progress = json.loads((config.run.output_dir / "status" / "progress.json").read_text())
    pid = json.loads((config.run.output_dir / "status" / "pid.json").read_text())
    assert progress["status"] == "running"
    assert progress["completed_records"] == 1
    assert isinstance(pid["pid"], int)
    assert not list((config.run.output_dir / "status").glob(".progress.json.tmp.*"))


def test_csv_row_count_treats_empty_tables_as_zero(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("")

    assert _csv_row_count(path) == 0


def test_writer_prepare_rejects_non_empty_run_directory(tmp_path: Path):
    config = _config(tmp_path)
    config.run.output_dir.mkdir(parents=True)
    marker = config.run.output_dir / "existing.txt"
    marker.write_text("keep\n")

    writer = AnalysisBundleWriter(config)
    with pytest.raises(FileExistsError, match="output_dir already exists"):
        writer.prepare(args=type("Args", (), {"dry_run": False})())

    assert marker.read_text() == "keep\n"


def test_writer_prepare_allows_existing_empty_run_directory(tmp_path: Path):
    config = _config(tmp_path)
    config.run.output_dir.mkdir(parents=True)

    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())

    assert (config.run.output_dir / "status" / "pid.json").exists()


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


def test_writer_empty_results_do_not_create_success_marker(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    record = _record()

    writer.write_results([record], [])

    assert not (config.run.output_dir / "per_record" / "rec1" / "_SUCCESS.json").exists()


def test_writer_rebuilds_cumulative_summary_from_sidecars(tmp_path: Path):
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


def test_writer_rebuild_ignores_partial_sidecar_without_result_manifest(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    record_dir = config.run.output_dir / "per_record" / "rec1"
    record_dir.mkdir(parents=True)
    (record_dir / "night_stats.json").write_text(json.dumps({"record_id": "rec1", "stage5_model_TST_min": 1.0}))

    writer.rebuild_global_tables([_record()], [])

    assert _csv_row_count(config.run.output_dir / "tables" / "night_stats.csv") == 0
    summary = pd.read_csv(config.run.output_dir / "tables" / "analyzer_summary.csv")
    assert summary.loc[summary["name"] == "stage5_model", "record_count"].item() == 0
    assert summary.loc[summary["name"] == "stage5_model", "result_count"].item() == 0


def test_writer_rebuild_includes_sidecar_with_result_manifest(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    record_dir = config.run.output_dir / "per_record" / "rec1"
    record_dir.mkdir(parents=True)
    (record_dir / "night_stats.json").write_text(json.dumps({"record_id": "rec1", "stage5_model_TST_min": 1.0}))
    pd.DataFrame(
        [
            {
                "record_id": "rec1",
                "kind": "analyzer",
                "name": "stage5_model",
                "type": "sleep2vec_downstream",
                "enabled": True,
                "result_count": 1,
                "failure_count": 0,
                "epoch_rows": 0,
                "second_rows": 0,
                "event_rows": 0,
                "night_rows": 1,
            }
        ]
    ).to_csv(record_dir / "result_manifest.csv", index=False)

    writer.rebuild_global_tables([_record()], [])

    night_stats = pd.read_csv(config.run.output_dir / "tables" / "night_stats.csv")
    summary = pd.read_csv(config.run.output_dir / "tables" / "analyzer_summary.csv")
    assert night_stats["record_id"].tolist() == ["rec1"]
    assert summary.loc[summary["name"] == "stage5_model", "record_count"].item() == 1
    assert summary.loc[summary["name"] == "stage5_model", "result_count"].item() == 1


def test_writer_rebuild_ignores_failed_manifest_without_failures_csv(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())
    record = _record()

    writer.write_chunk(
        [record],
        [AnalyzerResult("stage5_model", "rec1", night={"stage5_model_TST_min": 1.0})],
        [FailureRecord("rec1", "stage5_model", "ValueError", "bad")],
        completed_record_ids=set(),
    )
    writer.rebuild_global_tables([record], [])

    assert _csv_row_count(config.run.output_dir / "tables" / "night_stats.csv") == 0
    summary = pd.read_csv(config.run.output_dir / "tables" / "analyzer_summary.csv")
    assert summary.loc[summary["name"] == "stage5_model", "record_count"].item() == 0
    assert summary.loc[summary["name"] == "stage5_model", "result_count"].item() == 0


def test_writer_write_failures_replaces_previous_file(tmp_path: Path):
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


def test_writer_record_manifest_writes_current_records_only(tmp_path: Path):
    config = _config(tmp_path)
    writer = AnalysisBundleWriter(config)
    writer.prepare(args=type("Args", (), {"dry_run": False})())

    writer.write_record_manifest([_record("rec1", source="site_a"), _record("rec2", source="site_b")])
    writer.write_record_manifest([_record("rec2", source="site_c")])

    manifest = pd.read_csv(config.run.output_dir / "record_manifest.csv")
    by_record = manifest.set_index("record_id")
    assert set(by_record.index) == {"rec2"}
    assert by_record.loc["rec2", "source"] == "site_c"


def test_writer_rebuild_uses_current_global_failures(tmp_path: Path):
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
