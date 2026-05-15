import argparse
from pathlib import Path

import pandas as pd

from sleep2vec.distributed import get_rank_world_size, is_rank_zero_process, is_torch_distributed_ready
from sleep2vec.results import save_result_csv
from wrist2vec.results import save_result_csv as save_wrist_result_csv


def _finetune_args(*, version: str) -> argparse.Namespace:
    return argparse.Namespace(
        version=version,
        config=Path("configs/ppg_ahi_finetune_large.yaml"),
        label_name="ahi",
        ckpt_path="best",
        lr=1e-4,
        batch_size=8,
        n_few_shot=None,
        channel_names=["ppg"],
    )


def _infer_args() -> argparse.Namespace:
    return argparse.Namespace(
        config=Path("configs/ppg_ahi_finetune_large.yaml"),
        label_name="ahi",
        eval_split="test",
        ckpt_path="log-finetune/ppg_ahi_large/checkpoints/best.ckpt",
        lr=1e-4,
        batch_size=8,
        n_few_shot=None,
        channel_names=["ppg"],
        wandb_name=None,
        wandb_id=None,
    )


def test_save_result_csv_appends_rows_with_experiment_version(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "results.csv"

    save_result_csv({"test_loss": 1.0}, str(csv_path), _finetune_args(version="exp-a"))
    save_result_csv({"test_loss": 0.5}, str(csv_path), _finetune_args(version="exp-b"))

    df = pd.read_csv(csv_path)

    assert df["experiment_version"].tolist() == ["exp-a", "exp-b"]
    assert df["result_source"].tolist() == ["finetune", "finetune"]
    assert df["test_loss"].tolist() == [1.0, 0.5]


def test_save_result_csv_recovers_from_empty_existing_file(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "results.csv"
    csv_path.touch()

    save_result_csv({"test_loss": 0.25}, str(csv_path), _finetune_args(version="exp-a"))

    df = pd.read_csv(csv_path)

    assert len(df) == 1
    assert df.loc[0, "experiment_version"] == "exp-a"
    assert df.loc[0, "test_loss"] == 0.25


def test_save_result_csv_preserves_old_rows_when_schema_expands(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "results.csv"
    pd.DataFrame([{"test_loss": 1.0, "label_name": "ahi"}]).to_csv(csv_path, index=False)

    save_result_csv({"test_loss": 0.5, "test_ahi_pearson": 0.8}, str(csv_path), _finetune_args(version="exp-b"))

    df = pd.read_csv(csv_path)

    assert len(df) == 2
    assert df.loc[0, "test_loss"] == 1.0
    assert pd.isna(df.loc[0, "experiment_version"])
    assert df.loc[1, "experiment_version"] == "exp-b"
    assert df.loc[1, "test_ahi_pearson"] == 0.8


def test_save_result_csv_skips_nonzero_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "results.csv"

    save_result_csv({"test_loss": 1.0}, str(csv_path), _finetune_args(version="exp-a"))

    assert not csv_path.exists()


def test_save_result_csv_builds_infer_experiment_version_when_version_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "results.csv"

    save_result_csv({"test_loss": 0.1}, str(csv_path), _infer_args())

    df = pd.read_csv(csv_path)

    assert df.loc[0, "result_source"] == "infer"
    assert df.loc[0, "experiment_version"] == "ppg_ahi_finetune_large-ahi-test-best"


def test_save_result_csv_records_effective_preset_path(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "results.csv"
    args = _infer_args()
    args.finetune_preset_path = Path("config/preset.pkl")
    args.inference_preset_path = Path("index_parallel/presets/ahi/cuhk_test_preset_1535.pickle")

    save_result_csv({"test_loss": 0.1}, str(csv_path), args)

    df = pd.read_csv(csv_path)

    assert df.loc[0, "preset_path"] == "index_parallel/presets/ahi/cuhk_test_preset_1535.pickle"


def test_wrist2vec_save_result_csv_records_effective_preset_path(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "wrist_results.csv"
    args = _infer_args()
    args.config = Path("configs/write2vec/wrist2vec_ppg_ahi_finetune.yaml")
    args.finetune_preset_path = Path("config/wrist_preset.pkl")
    args.inference_preset_path = Path("index_parallel/presets/ahi/wrist_test_preset_1535.pickle")

    save_wrist_result_csv({"test_loss": 0.1}, str(csv_path), args)

    df = pd.read_csv(csv_path)

    assert df.loc[0, "preset_path"] == "index_parallel/presets/ahi/wrist_test_preset_1535.pickle"


def test_is_rank_zero_process_defaults_true_without_rank_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert is_rank_zero_process() is True


def test_is_rank_zero_process_accepts_zero_rank(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert is_rank_zero_process() is True


def test_is_rank_zero_process_rejects_nonzero_rank(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert is_rank_zero_process() is False


def test_is_rank_zero_process_falls_back_true_on_invalid_rank(monkeypatch):
    monkeypatch.setenv("RANK", "not-a-number")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert is_rank_zero_process() is True


def test_is_torch_distributed_ready_false_when_dist_is_unavailable(monkeypatch):
    monkeypatch.setattr("sleep2vec.distributed.dist.is_available", lambda: False)

    assert is_torch_distributed_ready() is False


def test_is_torch_distributed_ready_true_when_initialized(monkeypatch):
    monkeypatch.setattr("sleep2vec.distributed.dist.is_available", lambda: True)
    monkeypatch.setattr("sleep2vec.distributed.dist.is_initialized", lambda: True)

    assert is_torch_distributed_ready() is True


def test_get_rank_world_size_defaults_to_single_process(monkeypatch):
    monkeypatch.setattr("sleep2vec.distributed.dist.is_available", lambda: False)

    assert get_rank_world_size() == (0, 1)


def test_get_rank_world_size_reads_torch_distributed_state(monkeypatch):
    monkeypatch.setattr("sleep2vec.distributed.dist.is_available", lambda: True)
    monkeypatch.setattr("sleep2vec.distributed.dist.is_initialized", lambda: True)
    monkeypatch.setattr("sleep2vec.distributed.dist.get_rank", lambda: 2)
    monkeypatch.setattr("sleep2vec.distributed.dist.get_world_size", lambda: 8)

    assert get_rank_world_size() == (2, 8)
