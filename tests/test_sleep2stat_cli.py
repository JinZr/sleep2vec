from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from sleep2stat.cli import main
from sleep2stat.config import load_config
from sleep2stat.core.pipeline import run_pipeline


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


def test_pipeline_skip_existing_preserves_per_record_outputs(tmp_path: Path):
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
            "duration_column": "duration",
            "record_id_columns": ["source", "patient_id", "session_id"],
            "token_sec": 30,
            "max_tokens": 2,
        },
        "signals": {
            "channels": {
                "ppg": {"source": "ppg", "sfreq": 100, "kind": "ppg", "input_dim": 3000},
            }
        },
        "analyzers": [
            {"name": "reference_stage5", "type": "npz_stage_reference", "label_name": "stage5", "stage_key": "stage5"}
        ],
        "reducers": [{"name": "stage5_stats", "type": "hypnogram_stats", "source": "reference_stage5"}],
        "outputs": {
            "write_global_tables": True,
            "write_per_record": True,
            "include_probabilities": True,
            "include_raw_logits": False,
            "compression": "gzip",
        },
    }
    config_path.write_text(yaml.safe_dump(payload))
    config = load_config(config_path)
    args = SimpleNamespace(split=["test"], limit_records=1, dry_run=False, device="cpu", num_workers=0, batch_size=None)

    run_pipeline(config, args)
    rec1_path = tmp_path / "run" / "per_record" / "unit__p001__s001" / "epoch_alignment.csv.gz"
    assert len(pd.read_csv(rec1_path)) == 2

    args.limit_records = 2
    run_pipeline(config, args)

    assert len(pd.read_csv(rec1_path)) == 2
    global_epoch = pd.read_csv(tmp_path / "run" / "tables" / "epoch_alignment.csv.gz")
    assert sorted(global_epoch["record_id"].unique().tolist()) == ["unit__p001__s001", "unit__p002__s001"]
