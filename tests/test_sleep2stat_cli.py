import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from sleep2stat.cli import main
from sleep2stat.config import load_config
from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.pipeline import run_pipeline
from sleep2stat.plot import _plot_cohort_stage_composition, _respiratory_metric_specs


def _write_dry_run_config(tmp_path: Path) -> Path:
    index_path = tmp_path / "index.csv"
    index_path.write_text("path,duration,split,source,patient_id,session_id\n/tmp/sample.npz,60,test,unit,p001,s001\n")
    config_path = tmp_path / "config.yaml"
    payload = {
        "run": {
            "name": "cli",
            "output_dir": str(tmp_path / "run"),
            "overwrite": True,
            "skip_existing": True,
        },
        "data": {
            "backend": "npz",
            "index": str(index_path),
            "split": ["test"],
            "path_column": "path",
            "duration_column": "duration",
            "split_column": "split",
            "source_column": "source",
            "record_id_columns": ["source", "patient_id", "session_id"],
            "token_sec": 30,
            "max_tokens": 2,
        },
        "signals": {
            "channels": {
                "ppg": {
                    "source": "ppg",
                    "sfreq": 100,
                    "kind": "ppg",
                    "input_dim": 3000,
                }
            }
        },
        "analyzers": [
            {
                "name": "stage5_model",
                "type": "sleep2vec_downstream",
                "namespace": "sleep2vec2",
                "label_name": "stage5",
                "config": "configs/sleep2vec2/ppg_stage5_finetune_large.yaml",
                "ckpt_path": "/tmp/missing.ckpt",
                "input_channels": ["ppg"],
            }
        ],
        "reducers": [{"name": "stage5_stats", "type": "hypnogram_stats", "source": "stage5_model"}],
        "outputs": {
            "write_global_tables": True,
            "write_per_record": True,
            "include_probabilities": True,
            "include_raw_logits": False,
            "compression": "gzip",
            "global_tables": {
                "epoch_alignment": True,
                "second_alignment": False,
                "event_alignment": True,
                "night_stats": True,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(payload))
    return config_path


def test_cli_run_dry_run_and_summarize(tmp_path: Path, capsys):
    config_path = _write_dry_run_config(tmp_path)

    assert main(["run", "--config", str(config_path), "--split", "test", "--limit-records", "1", "--dry-run"]) == 0
    assert (tmp_path / "run" / "run_manifest.json").exists()

    assert main(["summarize", "--run-dir", str(tmp_path / "run")]) == 0
    captured = capsys.readouterr()
    assert "records: 1" in captured.out


def test_cli_summarize_uses_supplied_run_dir_over_config_output_dir(tmp_path: Path):
    config_path = _write_dry_run_config(tmp_path)
    run_dir = tmp_path / "run"
    wrong_dir = tmp_path / "wrong_run_dir"

    assert main(["run", "--config", str(config_path), "--split", "test", "--limit-records", "1", "--dry-run"]) == 0
    copied_config = yaml.safe_load((run_dir / "config.yaml").read_text())
    copied_config["run"]["output_dir"] = str(wrong_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(copied_config))
    (run_dir / "tables" / "night_stats.csv").unlink()

    assert main(["summarize", "--run-dir", str(run_dir)]) == 0

    assert (run_dir / "tables" / "night_stats.csv").exists()
    assert not wrong_dir.exists()


def test_cli_validate_config_check_records_flags_unconvertible_yasa_sex(tmp_path: Path, capsys):
    index_path = tmp_path / "index.csv"
    index_path.write_text("path,duration,split,source,patient_id,age,sex\nmissing.npz,60,test,unit,p001,60,unknown\n")
    config_path = tmp_path / "yasa.yaml"
    payload = {
        "run": {"name": "yasa", "output_dir": str(tmp_path / "run"), "overwrite": True},
        "data": {
            "backend": "npz",
            "index": str(index_path),
            "split": ["test"],
            "path_column": "path",
            "duration_column": "duration",
            "split_column": "split",
            "record_id_columns": ["patient_id"],
            "metadata_columns": ["age", "sex"],
            "token_sec": 30,
            "max_tokens": 2,
        },
        "signals": {
            "channels": {"eeg": {"source": "eeg", "sfreq": 100, "kind": "eeg", "input_dim": 3000}},
        },
        "analyzers": [{"name": "yasa_stage", "type": "yasa_stage", "input_channels": ["eeg"]}],
        "reducers": [],
        "outputs": {"write_global_tables": True, "write_per_record": True, "compression": "gzip"},
    }
    config_path.write_text(yaml.safe_dump(payload))

    assert main(["validate-config", "--config", str(config_path), "--check-records"]) == 1

    captured = capsys.readouterr()
    assert "YASA metadata sex: present 1/1, convertible_to_male 0/1" in captured.out
    assert "0 are convertible to male" in captured.out


def test_cli_resume_status_and_repair_mark_dead_running(tmp_path: Path, capsys):
    run_dir = tmp_path / "run"
    (run_dir / "status").mkdir(parents=True)
    (run_dir / "per_record" / "rec1").mkdir(parents=True)
    (run_dir / "record_manifest.csv").write_text(
        "record_id,path,split,source,duration_sec,token_sec,max_tokens\n"
        "rec1,rec1.npz,test,unit,60,30,2\n"
        "rec2,rec2.npz,test,unit,60,30,2\n"
    )
    (run_dir / "per_record" / "rec1" / "_SUCCESS.json").write_text("{}\n")
    (run_dir / "status" / "progress.json").write_text('{"status": "running"}\n')
    (run_dir / "status" / "pid.json").write_text('{"pid": 999999999}\n')

    assert main(["resume-status", "--run-dir", str(run_dir), "--json"]) == 0
    status = json.loads(capsys.readouterr().out)

    assert status["status"] == "stale_running"
    assert status["pending_records"] == 1
    assert status["pending_record_ids"] == ["rec2"]

    assert main(["resume-status", "--run-root", str(tmp_path), "--glob", "run", "--json"]) == 0
    root_status = json.loads(capsys.readouterr().out)
    assert root_status["dead_runs"] == [str(run_dir)]

    assert main(["repair", "--run-dir", str(run_dir), "--json"]) == 0
    repaired = json.loads(capsys.readouterr().out)

    assert repaired["repair_status"] == "interrupted"
    assert repaired["status"] == "interrupted"
    progress = json.loads((run_dir / "status" / "progress.json").read_text())
    assert progress["status"] == "interrupted"
    assert progress["pending_records"] == 1


def test_cli_resume_status_and_repair_treat_global_failure_as_failed_run(tmp_path: Path, capsys):
    run_dir = tmp_path / "run"
    (run_dir / "status").mkdir(parents=True)
    (run_dir / "record_manifest.csv").write_text(
        "record_id,path,split,source,duration_sec,token_sec,max_tokens\n"
        "rec1,rec1.npz,test,unit,60,30,2\n"
        "rec2,rec2.npz,test,unit,60,30,2\n"
    )
    pd.DataFrame(
        [{"record_id": "__all__", "analyzer": "yasa_stage", "error_type": "ImportError", "message": "missing"}]
    ).to_csv(run_dir / "status" / "failures.csv", index=False)
    (run_dir / "status" / "progress.json").write_text('{"status": "completed_with_failures"}\n')
    (run_dir / "status" / "pid.json").write_text('{"pid": 999999999}\n')

    assert main(["resume-status", "--run-dir", str(run_dir), "--json"]) == 0
    status = json.loads(capsys.readouterr().out)

    assert status["status"] == "completed_with_failures"
    assert status["failed_records"] == 2
    assert status["pending_records"] == 0

    assert main(["repair", "--run-dir", str(run_dir), "--json"]) == 0
    repaired = json.loads(capsys.readouterr().out)

    assert repaired["repair_status"] == "completed_with_failures"
    assert repaired["status"] == "completed_with_failures"


def test_cli_cohort_finalize_merges_runs_and_drops_resolved_failures(tmp_path: Path):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    for run in (run1, run2):
        (run / "tables").mkdir(parents=True)
        (run / "status").mkdir()
    pd.DataFrame({"record_id": ["rec1", "rec2"], "source": ["A", "A"]}).to_csv(
        run1 / "record_manifest.csv", index=False
    )
    pd.DataFrame({"record_id": ["rec1", "rec2"], "metric": [1.0, 2.0]}).to_csv(
        run1 / "tables" / "night_stats.csv", index=False
    )
    pd.DataFrame(
        [{"record_id": "rec2", "analyzer": "yasa_stage", "error_type": "ValueError", "message": "old"}]
    ).to_csv(run1 / "status" / "failures.csv", index=False)
    pd.DataFrame({"record_id": ["rec2"], "source": ["B"]}).to_csv(run2 / "record_manifest.csv", index=False)
    pd.DataFrame({"record_id": ["rec2"], "metric": [3.0]}).to_csv(run2 / "tables" / "night_stats.csv", index=False)
    pd.DataFrame(columns=["record_id", "analyzer", "error_type", "message"]).to_csv(
        run2 / "status" / "failures.csv", index=False
    )

    out = tmp_path / "final"

    assert (
        main(
            [
                "cohort-finalize",
                "--output-run-dir",
                str(out),
                "--input-run-dir",
                str(run1),
                "--input-run-dir",
                str(run2),
            ]
        )
        == 0
    )

    night = pd.read_csv(out / "tables" / "night_stats.csv").set_index("record_id")
    failures = pd.read_csv(out / "status" / "failures.csv")
    progress = json.loads((out / "status" / "progress.json").read_text())
    assert night.loc["rec2", "metric"] == 3.0
    assert failures.empty
    assert progress["status"] == "completed"


def test_cli_cohort_finalize_reports_pending_records(tmp_path: Path):
    run1 = tmp_path / "run1"
    (run1 / "tables").mkdir(parents=True)
    (run1 / "status").mkdir()
    pd.DataFrame({"record_id": ["rec1", "rec2"], "source": ["A", "A"]}).to_csv(
        run1 / "record_manifest.csv", index=False
    )
    pd.DataFrame({"record_id": ["rec1"], "metric": [1.0]}).to_csv(run1 / "tables" / "night_stats.csv", index=False)
    pd.DataFrame(columns=["record_id", "analyzer", "error_type", "message"]).to_csv(
        run1 / "status" / "failures.csv", index=False
    )
    out = tmp_path / "final"

    assert main(["cohort-finalize", "--output-run-dir", str(out), "--input-run-dir", str(run1)]) == 0

    progress = json.loads((out / "status" / "progress.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert progress["status"] == "incomplete"
    assert progress["pending_records"] == 1
    assert progress["pending_record_ids"] == ["rec2"]
    assert manifest["status"] == "incomplete"


def test_cli_cohort_finalize_counts_global_failure_as_failed_records(tmp_path: Path):
    run1 = tmp_path / "run1"
    (run1 / "tables").mkdir(parents=True)
    (run1 / "status").mkdir()
    pd.DataFrame({"record_id": ["rec1", "rec2"], "source": ["A", "A"]}).to_csv(
        run1 / "record_manifest.csv", index=False
    )
    pd.DataFrame(columns=["record_id", "metric"]).to_csv(run1 / "tables" / "night_stats.csv", index=False)
    pd.DataFrame(
        [{"record_id": "__all__", "analyzer": "yasa_stage", "error_type": "ImportError", "message": "missing"}]
    ).to_csv(run1 / "status" / "failures.csv", index=False)
    out = tmp_path / "final"

    assert main(["cohort-finalize", "--output-run-dir", str(out), "--input-run-dir", str(run1)]) == 0

    progress = json.loads((out / "status" / "progress.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert progress["status"] == "completed_with_failures"
    assert progress["failed_records"] == 2
    assert progress["pending_records"] == 0
    assert manifest["status"] == "completed_with_failures"


def test_cli_cohort_finalize_drops_resolved_global_failure(tmp_path: Path):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    for run in (run1, run2):
        (run / "tables").mkdir(parents=True)
        (run / "status").mkdir()
        pd.DataFrame({"record_id": ["rec1", "rec2"], "source": ["A", "A"]}).to_csv(
            run / "record_manifest.csv", index=False
        )
    pd.DataFrame(columns=["record_id", "metric"]).to_csv(run1 / "tables" / "night_stats.csv", index=False)
    pd.DataFrame(
        [{"record_id": "__all__", "analyzer": "yasa_stage", "error_type": "ImportError", "message": "missing"}]
    ).to_csv(run1 / "status" / "failures.csv", index=False)
    pd.DataFrame({"record_id": ["rec1", "rec2"], "metric": [1.0, 2.0]}).to_csv(
        run2 / "tables" / "night_stats.csv", index=False
    )
    pd.DataFrame(columns=["record_id", "analyzer", "error_type", "message"]).to_csv(
        run2 / "status" / "failures.csv", index=False
    )
    out = tmp_path / "final"

    assert (
        main(
            [
                "cohort-finalize",
                "--output-run-dir",
                str(out),
                "--input-run-dir",
                str(run1),
                "--input-run-dir",
                str(run2),
            ]
        )
        == 0
    )

    failures = pd.read_csv(out / "status" / "failures.csv")
    progress = json.loads((out / "status" / "progress.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert failures.empty
    assert progress["status"] == "completed"
    assert progress["failed_records"] == 0
    assert progress["failure_rows"] == 0
    assert manifest["status"] == "completed"


def test_cli_cohort_finalize_rejects_missing_input_manifest(tmp_path: Path):
    run1 = tmp_path / "run1"
    run1.mkdir()

    with pytest.raises(FileNotFoundError, match="record_manifest"):
        main(["cohort-finalize", "--output-run-dir", str(tmp_path / "final"), "--input-run-dir", str(run1)])


def test_pipeline_skip_existing_preserves_per_record_outputs(tmp_path: Path, monkeypatch):
    np.savez(tmp_path / "rec1.npz", stage5=np.array([0, 1], dtype=np.int64))
    np.savez(tmp_path / "rec2.npz", stage5=np.array([2, 4], dtype=np.int64))
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "path,duration,split,source,patient_id,session_id\n"
        f"{tmp_path / 'rec1.npz'},60,test,unit,p001,s001\n"
        f"{tmp_path / 'rec2.npz'},60,test,unit,p002,s001\n"
    )
    config_path = tmp_path / "config.yaml"
    payload = {
        "run": {
            "name": "skip",
            "output_dir": str(tmp_path / "run"),
            "overwrite": False,
            "skip_existing": True,
        },
        "data": {
            "backend": "npz",
            "index": str(index_path),
            "split": ["test"],
            "path_column": "path",
            "duration_column": "duration",
            "split_column": "split",
            "record_id_columns": ["source", "patient_id", "session_id"],
            "token_sec": 30,
            "max_tokens": 2,
        },
        "signals": {
            "channels": {
                "ppg": {"source": "ppg", "sfreq": 100, "kind": "ppg", "input_dim": 3000},
            }
        },
        "analyzers": [{"name": "reference_stage5", "type": "npz_stage_reference", "stage_key": "stage5"}],
        "reducers": [{"name": "stage5_stats", "type": "hypnogram_stats", "source": "reference_stage5"}],
        "outputs": {
            "write_global_tables": True,
            "write_per_record": True,
            "include_probabilities": True,
            "include_raw_logits": False,
            "compression": "gzip",
            "global_tables": {
                "epoch_alignment": True,
                "second_alignment": False,
                "event_alignment": True,
                "night_stats": True,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(payload))
    config = load_config(config_path)
    args = SimpleNamespace(split=["test"], limit_records=1, dry_run=False, device="cpu", num_workers=0, batch_size=None)

    run_pipeline(config, args)
    rec1_path = tmp_path / "run" / "per_record" / "unit__p001__s001" / "epoch_alignment.csv.gz"
    assert len(pd.read_csv(rec1_path)) == 2

    args.limit_records = 2
    args.num_workers = 2
    run_pipeline(config, args)

    assert len(pd.read_csv(rec1_path)) == 2
    global_epoch = pd.read_csv(tmp_path / "run" / "tables" / "epoch_alignment.csv.gz")
    assert sorted(global_epoch["record_id"].unique().tolist()) == ["unit__p001__s001", "unit__p002__s001"]
    manifest = yaml.safe_load((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["execution_split"] == "split2"

    monkeypatch.setattr(
        "sleep2stat.core.pipeline.create_analyzer",
        lambda config: (_ for _ in ()).throw(AssertionError("no-op resume should not prepare analyzers")),
    )
    run_pipeline(config, args)


def test_pipeline_record_split_matches_sequential_outputs_and_summarize(tmp_path: Path):
    records = []
    for idx, stages in enumerate(([0, 1], [2, 4], [0, 0], [3, 4])):
        path = tmp_path / f"rec{idx}.npz"
        np.savez(path, stage5=np.array(stages, dtype=np.int64))
        records.append(f"{path},60,test,unit,p{idx:03d},s001\n")
    index_path = tmp_path / "index.csv"
    index_path.write_text("path,duration,split,source,patient_id,session_id\n" + "".join(records))
    payload = {
        "run": {"name": "reference", "output_dir": str(tmp_path / "run_seq"), "overwrite": True},
        "data": {
            "backend": "npz",
            "index": str(index_path),
            "split": ["test"],
            "path_column": "path",
            "duration_column": "duration",
            "split_column": "split",
            "record_id_columns": ["source", "patient_id", "session_id"],
            "token_sec": 30,
            "max_tokens": 2,
        },
        "signals": {
            "channels": {"ppg": {"source": "ppg", "sfreq": 100, "kind": "ppg", "input_dim": 3000}},
        },
        "analyzers": [{"name": "reference_stage5", "type": "npz_stage_reference", "stage_key": "stage5"}],
        "reducers": [{"name": "stage5_stats", "type": "hypnogram_stats", "source": "reference_stage5"}],
        "outputs": {
            "write_global_tables": True,
            "write_per_record": True,
            "include_probabilities": True,
            "include_raw_logits": False,
            "compression": "gzip",
            "global_tables": {
                "epoch_alignment": True,
                "second_alignment": False,
                "event_alignment": True,
                "night_stats": True,
            },
        },
    }
    seq_config_path = tmp_path / "seq.yaml"
    seq_config_path.write_text(yaml.safe_dump(payload))
    seq_config = load_config(seq_config_path)
    seq_args = SimpleNamespace(
        split=["test"], limit_records=None, dry_run=False, device="cpu", num_workers=1, batch_size=2
    )

    run_pipeline(seq_config, seq_args)

    payload["run"]["output_dir"] = str(tmp_path / "run_split")
    split_config_path = tmp_path / "split.yaml"
    split_config_path.write_text(yaml.safe_dump(payload))
    split_config = load_config(split_config_path)
    split_args = SimpleNamespace(
        split=["test"], limit_records=None, dry_run=False, device="cpu", num_workers=2, batch_size=2
    )

    run_pipeline(split_config, split_args)

    seq_night = pd.read_csv(tmp_path / "run_seq" / "tables" / "night_stats.csv").sort_values("record_id")
    split_night = pd.read_csv(tmp_path / "run_split" / "tables" / "night_stats.csv").sort_values("record_id")
    pd.testing.assert_frame_equal(seq_night.reset_index(drop=True), split_night.reset_index(drop=True))
    manifest = yaml.safe_load((tmp_path / "run_split" / "run_manifest.json").read_text())
    assert manifest["execution_split"] == "split2"

    (tmp_path / "run_split" / "tables" / "night_stats.csv").unlink()
    assert main(["summarize", "--run-dir", str(tmp_path / "run_split"), "--num-workers", "2"]) == 0
    assert (tmp_path / "run_split" / "tables" / "night_stats.csv").exists()


def test_pipeline_model_analyzer_keeps_canonical_chunk_path_with_num_workers(tmp_path: Path, monkeypatch):
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "path,duration,split,patient_id\n"
        f"{tmp_path / 'rec1.npz'},60,test,rec1\n"
        f"{tmp_path / 'rec2.npz'},60,test,rec2\n"
    )
    config_path = tmp_path / "config.yaml"
    payload = {
        "run": {"name": "model-fallback", "output_dir": str(tmp_path / "run"), "overwrite": True},
        "data": {
            "backend": "npz",
            "index": str(index_path),
            "split": ["test"],
            "path_column": "path",
            "duration_column": "duration",
            "split_column": "split",
            "record_id_columns": ["patient_id"],
            "token_sec": 30,
            "max_tokens": 2,
        },
        "signals": {
            "channels": {"ppg": {"source": "ppg", "sfreq": 100, "kind": "ppg", "input_dim": 3000}},
        },
        "analyzers": [
            {
                "name": "stage_model",
                "type": "sleep2vec_downstream",
                "namespace": "sleep2vec2",
                "label_name": "stage5",
                "config": "configs/sleep2vec2/ppg_stage5_finetune_large.yaml",
                "ckpt_path": "/tmp/stage.ckpt",
                "input_channels": ["ppg"],
            }
        ],
        "reducers": [],
        "outputs": {
            "write_global_tables": True,
            "write_per_record": True,
            "include_probabilities": True,
            "include_raw_logits": False,
            "compression": "gzip",
            "global_tables": {
                "epoch_alignment": False,
                "second_alignment": False,
                "event_alignment": False,
                "night_stats": True,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(payload))
    prepare_workers = []
    run_sizes = []

    class FakeAnalyzer:
        def __init__(self, config):
            self.config = config

        def prepare(self, context):
            prepare_workers.append(context.num_workers)

        def run(self, records, context, prior_results=None):
            run_sizes.append(len(records))
            return [
                AnalyzerResult(self.config.name, record.record_id, night={f"{self.config.name}_pred": 1})
                for record in records
            ], []

        def close(self):
            return None

    monkeypatch.setattr("sleep2stat.core.pipeline.create_analyzer", lambda config: FakeAnalyzer(config))
    config = load_config(config_path)
    args = SimpleNamespace(split=["test"], limit_records=None, dry_run=False, device="cpu", num_workers=2, batch_size=2)

    run_pipeline(config, args)

    assert prepare_workers == [2]
    assert run_sizes == [2]
    manifest = yaml.safe_load((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["execution_split"] == "sequential"


def test_pipeline_limits_reducer_failures_to_affected_records(tmp_path: Path, monkeypatch):
    index_path = tmp_path / "index.csv"
    index_path.write_text(
        "path,duration,split,patient_id\n"
        f"{tmp_path / 'good.npz'},60,test,good\n"
        f"{tmp_path / 'bad.npz'},60,test,bad\n"
    )
    config_path = tmp_path / "config.yaml"
    payload = {
        "run": {"name": "reducer-failure", "output_dir": str(tmp_path / "run"), "overwrite": True},
        "data": {
            "backend": "npz",
            "index": str(index_path),
            "split": ["test"],
            "path_column": "path",
            "duration_column": "duration",
            "split_column": "split",
            "record_id_columns": ["patient_id"],
            "token_sec": 30,
            "max_tokens": 2,
        },
        "signals": {
            "channels": {
                "ppg": {"source": "ppg", "sfreq": 100, "kind": "ppg", "input_dim": 3000},
            }
        },
        "analyzers": [{"name": "source_model", "type": "npz_stage_reference", "stage_key": "stage5"}],
        "reducers": [{"name": "stage5_stats", "type": "hypnogram_stats", "source": "source_model"}],
        "outputs": {
            "write_global_tables": True,
            "write_per_record": True,
            "include_probabilities": True,
            "include_raw_logits": False,
            "compression": "gzip",
            "global_tables": {
                "epoch_alignment": False,
                "second_alignment": False,
                "event_alignment": False,
                "night_stats": True,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(payload))

    class FakeAnalyzer:
        def __init__(self, config):
            self.config = config

        def prepare(self, context):
            return None

        def close(self):
            return None

        def run(self, records, context, prior_results=None):
            return [
                AnalyzerResult(self.config.name, record.record_id, night={f"{self.config.name}_ok": 1})
                for record in records
            ], []

    class FakeReducer:
        def __init__(self, config):
            self.config = config

        def reduce(self, records, results, context):
            if len(records) > 1:
                raise RuntimeError("chunk reducer failed")
            record = records[0]
            if record.record_id == "bad":
                raise ValueError("bad reducer source")
            return [AnalyzerResult(self.config.name, record.record_id, night={f"{self.config.name}_ok": 1})]

    monkeypatch.setattr("sleep2stat.core.pipeline.create_analyzer", lambda config: FakeAnalyzer(config))
    monkeypatch.setattr("sleep2stat.core.pipeline.create_reducer", lambda config: FakeReducer(config))
    config = load_config(config_path)
    args = SimpleNamespace(split=["test"], limit_records=None, dry_run=False, device="cpu", num_workers=0, batch_size=2)

    run_pipeline(config, args)

    assert (tmp_path / "run" / "per_record" / "good" / "_SUCCESS.json").exists()
    assert not (tmp_path / "run" / "per_record" / "bad" / "_SUCCESS.json").exists()
    failures = pd.read_csv(tmp_path / "run" / "status" / "failures.csv")
    assert failures[["record_id", "analyzer", "error_type"]].to_dict("records") == [
        {"record_id": "bad", "analyzer": "stage5_stats", "error_type": "ValueError"}
    ]


def test_cli_plot_record_creates_pngs_from_per_record_outputs(tmp_path: Path):
    pytest.importorskip("matplotlib")
    record_dir = tmp_path / "run" / "per_record" / "rec1"
    record_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "record_id": ["rec1", "rec1"],
            "start_sec": [0.0, 30.0],
            "stage5_model_pred": [0, 2],
        }
    ).to_csv(record_dir / "epoch_alignment.csv.gz", index=False, compression="gzip")
    pd.DataFrame(
        {
            "record_id": ["rec1", "rec1"],
            "start_sec": [0.0, 1.0],
            "ahi_model_prob": [0.1, 0.8],
        }
    ).to_csv(record_dir / "second_alignment.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"record_id": ["rec1"], "onset_sec": [1.0], "offset_sec": [12.0]}).to_csv(
        record_dir / "events.csv.gz", index=False, compression="gzip"
    )

    assert main(["plot-record", "--run-dir", str(tmp_path / "run"), "--record-id", "rec1"]) == 0

    assert (record_dir / "plots" / "hypnogram_overlay.png").exists()
    assert (record_dir / "plots" / "ahi_spo2_trace.png").exists()


def test_cli_plot_cohort_creates_core_harmonization_and_stage_panels(tmp_path: Path):
    pytest.importorskip("matplotlib")
    run_dir = tmp_path / "run"
    (run_dir / "tables").mkdir(parents=True)
    records = [f"rec{idx}" for idx in range(6)]
    centers = ["A", "A", "A", "B", "B", "B"]
    pd.DataFrame(
        {
            "record_id": records,
            "source": ["unused"] * 6,
            "center": centers,
            "age": [45, 52, 61, 48, 57, 66],
            "sex": ["F", "M", "F", "M", "F", "M"],
            "bmi": [24.1, 28.0, 31.2, 25.4, 29.1, 33.0],
        }
    ).to_csv(run_dir / "record_manifest.csv", index=False)
    pd.DataFrame(
        {
            "record_id": records,
            "stage5_model_TST_min": [390, 410, 430, 360, 380, 400],
            "stage5_model_TIB_min": [480, 480, 480, 480, 480, 480],
            "stage5_model_SE_pct": [81, 85, 89, 75, 79, 83],
            "stage5_model_WASO_SPT_min": [40, 34, 28, 65, 58, 49],
            "stage5_model_SOL_min": [18, 16, 12, 30, 26, 22],
            "stage5_model_REM_latency_min": [96, 88, 82, 130, 118, 110],
            "stage5_model_sleep_to_wake_transition_index": [4.2, 3.9, 3.4, 6.5, 5.8, 5.0],
            "stage5_model_stage_shift_rate_per_sleep_hour": [12.0, 11.1, 10.4, 15.2, 14.6, 13.2],
            "stage5_model_N1_ratio_TST": [0.08, 0.07, 0.06, 0.12, 0.11, 0.10],
            "stage5_model_N2_ratio_TST": [0.48, 0.50, 0.51, 0.46, 0.47, 0.48],
            "stage5_model_N3_ratio_TST": [0.20, 0.22, 0.23, 0.14, 0.15, 0.16],
            "stage5_model_REM_ratio_TST": [0.24, 0.21, 0.20, 0.28, 0.27, 0.26],
            "ahi_model_pred_ahi": [5.0, 9.0, 12.0, 22.0, 28.0, 34.0],
            "ahi_model_pred_REM_AHI_onset_stage": [8.0, 12.0, 16.0, 35.0, 42.0, 50.0],
            "ahi_model_pred_NREM_AHI_onset_stage": [4.0, 7.0, 10.0, 18.0, 24.0, 29.0],
            "ODI3_per_recording_hour": [6.0, 10.0, 11.0, 24.0, 31.0, 38.0],
            "ODI4_per_recording_hour": [3.0, 5.0, 7.0, 18.0, 22.0, 27.0],
            "spo2_t90_pct_recording": [1, 2, 3, 10, 13, 16],
            "spo2_nadir": [90, 88, 87, 82, 80, 78],
            "desaturation_area_burden_pctmin_per_recording_hour": [0.2, 0.4, 0.6, 2.2, 3.0, 3.8],
            "yasa_bandpower_N3_delta_mean": [0.38, 0.42, 0.44, 0.31, 0.33, 0.35],
            "yasa_bandpower_REM_alpha_mean": [0.09, 0.08, 0.08, 0.12, 0.13, 0.14],
            "yasa_bandpower_sigma_rel_mean": [0.17, 0.18, 0.19, 0.13, 0.14, 0.15],
            "yasa_spindles_spindle_density_per_min_N2N3": [2.1, 2.4, 2.6, 1.4, 1.6, 1.8],
            "yasa_slowwaves_slowwave_density_per_min_NREM": [5.0, 5.4, 5.8, 3.2, 3.5, 3.8],
            "yasa_hrv_stage_REM_RMSSD": [38, 41, 44, 31, 33, 35],
        }
    ).to_csv(run_dir / "tables" / "night_stats.csv", index=False)

    assert (
        main(
            [
                "plot-cohort",
                "--run-dir",
                str(run_dir),
                "--group-column",
                "center",
                "--stage-source",
                "stage5_model",
                "--adjust-covariates",
                "age",
                "sex",
                "bmi",
                "missing_covariate",
            ]
        )
        == 0
    )

    plot_dir = run_dir / "plots" / "cohort"
    assert (plot_dir / "cohort_stage_composition.png").exists()
    assert (plot_dir / "cohort_sleep_metrics.png").exists()
    assert (plot_dir / "cohort_stage_ratio_distribution.png").exists()
    assert (plot_dir / "cohort_harmonization_diagnostics.png").exists()
    assert (plot_dir / "cohort_respiratory_risk.png").exists()
    assert (plot_dir / "cohort_microstructure_autonomic.png").exists()


def test_plot_cohort_stage_composition_adds_wake_when_tib_is_available(tmp_path: Path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    captured = []
    original_bar = Axes.bar

    def capture_bar(self, *args, **kwargs):
        captured.append((kwargs.get("label"), np.asarray(args[1], dtype=float)))
        return original_bar(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "bar", capture_bar)
    frame = pd.DataFrame(
        {
            "record_id": ["rec1", "rec2"],
            "source": ["A", "A"],
            "stage5_model_TST_min": [360, 360],
            "stage5_model_TIB_min": [480, 480],
            "stage5_model_N1_ratio_TST": [0.10, 0.10],
            "stage5_model_N2_ratio_TST": [0.50, 0.50],
            "stage5_model_N3_ratio_TST": [0.20, 0.20],
            "stage5_model_REM_ratio_TST": [0.20, 0.20],
        }
    )

    _plot_cohort_stage_composition(frame, "stage5_model", "source", tmp_path / "composition.png")

    assert [item[0] for item in captured[:5]] == ["Wake", "N1", "N2", "N3", "REM"]
    assert captured[0][1][0] == pytest.approx(0.25)
    assert captured[1][1][0] == pytest.approx(0.075)


def test_plot_cohort_resp_metric_prefers_clinical_ahi_and_new_denominators():
    frame = pd.DataFrame(
        {
            "ahi_model_pred_event_rate_per_recording_hour": [40.0],
            "ahi_model_pred_REM_AHI_onset_stage": [30.0],
            "ahi_model_pred_NREM_AHI_onset_stage": [20.0],
            "ahi_model_pred_ahi": [12.0],
            "ODI3_per_recording_hour": [10.0],
        }
    )

    specs = _respiratory_metric_specs(frame)

    assert specs[:4] == [
        ("Pred AHI", "ahi_model_pred_ahi", 1.0),
        ("Pred REM AHI", "ahi_model_pred_REM_AHI_onset_stage", 1.0),
        ("Pred NREM AHI", "ahi_model_pred_NREM_AHI_onset_stage", 1.0),
        ("ODI3", "ODI3_per_recording_hour", 1.0),
    ]


def test_plot_cohort_resp_metric_rejects_legacy_resp_columns():
    frame = pd.DataFrame(
        {
            "ahi_model_pred_ahi_rem_denominator": [30.0],
            "ahi_model_pred_ahi_nrem_denominator": [20.0],
            "ODI3_recording": [10.0],
            "ODI4_recording": [5.0],
            "pred_event_hypoxic_burden_pctmin_per_hour": [2.0],
        }
    )

    assert _respiratory_metric_specs(frame) == []


def test_cli_plot_cohort_allows_respiratory_only_bundle(tmp_path: Path):
    pytest.importorskip("matplotlib")
    run_dir = tmp_path / "run"
    (run_dir / "tables").mkdir(parents=True)
    pd.DataFrame(
        {
            "record_id": ["rec1", "rec2"],
            "source": ["A", "B"],
            "ODI3_per_recording_hour": [10.0, 18.0],
            "ODI4_per_recording_hour": [5.0, 9.0],
            "spo2_t90_pct_recording": [2.5, 8.0],
            "spo2_nadir": [88.0, 82.0],
            "resp_event_hypoxic_burden_pctmin_per_recording_hour": [0.8, 2.4],
        }
    ).to_csv(run_dir / "tables" / "night_stats.csv", index=False)

    assert main(["plot-cohort", "--run-dir", str(run_dir)]) == 0

    plot_dir = run_dir / "plots" / "cohort"
    assert (plot_dir / "cohort_respiratory_risk.png").exists()
    assert not (plot_dir / "cohort_stage_composition.png").exists()
    assert not (plot_dir / "cohort_sleep_metrics.png").exists()
    assert not (plot_dir / "cohort_stage_ratio_distribution.png").exists()


def test_cli_plot_cohort_skips_harmonization_for_single_center(tmp_path: Path):
    pytest.importorskip("matplotlib")
    run_dir = tmp_path / "run"
    (run_dir / "tables").mkdir(parents=True)
    pd.DataFrame({"record_id": ["rec1", "rec2"], "source": ["A", "A"]}).to_csv(
        run_dir / "record_manifest.csv", index=False
    )
    pd.DataFrame(
        {
            "record_id": ["rec1", "rec2"],
            "stage5_model_N1_ratio_TST": [0.1, 0.12],
            "stage5_model_N2_ratio_TST": [0.5, 0.48],
            "stage5_model_N3_ratio_TST": [0.2, 0.22],
            "stage5_model_REM_ratio_TST": [0.2, 0.18],
            "stage5_model_TST_min": [400, 410],
            "stage5_model_SE_pct": [86, 88],
        }
    ).to_csv(run_dir / "tables" / "night_stats.csv", index=False)

    assert main(["plot-cohort", "--run-dir", str(run_dir), "--stage-source", "stage5_model"]) == 0

    assert not (run_dir / "plots" / "cohort" / "cohort_harmonization_diagnostics.png").exists()


def test_cli_plot_cohort_requires_night_stats(tmp_path: Path):
    (tmp_path / "run" / "tables").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="night_stats.csv"):
        main(["plot-cohort", "--run-dir", str(tmp_path / "run")])


def test_cli_plot_cohort_rejects_missing_explicit_stage_ratio_columns(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "tables").mkdir(parents=True)
    pd.DataFrame({"record_id": ["rec1"], "stage5_model_TST_min": [400]}).to_csv(
        run_dir / "tables" / "night_stats.csv", index=False
    )

    with pytest.raises(ValueError, match="complete N1/N2/N3/REM ratio columns"):
        main(["plot-cohort", "--run-dir", str(run_dir), "--stage-source", "stage5_model"])


def test_cli_plot_cohort_treats_stage_source_auto_as_plain_missing_source(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "tables").mkdir(parents=True)
    pd.DataFrame(
        {
            "record_id": ["rec1"],
            "stage5_model_N1_ratio_TST": [0.1],
            "stage5_model_N2_ratio_TST": [0.5],
            "stage5_model_N3_ratio_TST": [0.2],
            "stage5_model_REM_ratio_TST": [0.2],
        }
    ).to_csv(run_dir / "tables" / "night_stats.csv", index=False)

    with pytest.raises(ValueError, match="stage source 'auto' does not have complete"):
        main(["plot-cohort", "--run-dir", str(run_dir), "--stage-source", "auto"])


def test_cli_plot_cohort_rejects_legacy_stage_pct_columns(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "tables").mkdir(parents=True)
    pd.DataFrame(
        {
            "record_id": ["rec1"],
            "stage5_model_pct_N1": [0.1],
            "stage5_model_pct_N2": [0.5],
            "stage5_model_pct_N3": [0.2],
            "stage5_model_pct_REM": [0.2],
        }
    ).to_csv(run_dir / "tables" / "night_stats.csv", index=False)

    with pytest.raises(ValueError, match="complete N1/N2/N3/REM ratio columns"):
        main(["plot-cohort", "--run-dir", str(run_dir), "--stage-source", "stage5_model"])
