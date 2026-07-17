import argparse
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from sleep2vec.distributed import get_rank_world_size, is_rank_zero_process, is_torch_distributed_ready
from sleep2vec.results import (
    make_prediction_run_id,
    prepare_inference_result_paths,
    save_prediction_csv,
    save_result_csv,
)
from wrist2vec.results import save_result_csv as save_wrist_result_csv

RESULT_PACKAGES = ("sleep2vec", "sleep2vec2", "sleep2expert")


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


def _write_lightning_ckpt(path: Path, *, epoch: int, step: int) -> None:
    torch.save({"state_dict": {}, "epoch": epoch, "global_step": step}, path)


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


@pytest.mark.parametrize("package_name", RESULT_PACKAGES)
def test_finetune_result_csv_omits_inference_metadata(tmp_path, monkeypatch, package_name: str):
    results_mod = importlib.import_module(f"{package_name}.results")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "results.csv"

    results_mod.save_result_csv({"test_loss": 1.0}, str(csv_path), _finetune_args(version="exp-a"))

    df = pd.read_csv(csv_path)
    assert "prediction_run_id" not in df.columns
    assert "run_dir" not in df.columns
    assert "prediction_csv_path" not in df.columns


def test_prepare_inference_result_paths_builds_run_directory(tmp_path):
    ckpt_path = tmp_path / "model.ckpt"
    _write_lightning_ckpt(ckpt_path, epoch=7, step=1234)
    args = _infer_args()
    args.ckpt_path = str(ckpt_path)

    prepare_inference_result_paths(args, root=tmp_path / "results" / "inference", timestamp="20260524T000000Z")

    assert args.prediction_run_id.startswith("20260524T000000Z__sleep2vec__")
    assert args.prediction_run_id in str(args.run_dir)
    assert args.run_dir == tmp_path / "results" / "inference" / "sleep2vec" / "ahi" / args.prediction_run_id
    assert args.inference_metrics_csv_path.name == "metrics__ahi__test__epoch07_step1234.csv"
    assert args.inference_prediction_csv_path.name == "predictions__ahi__test__epoch07_step1234.csv"
    assert (
        args.inference_survival_per_disease_metrics_csv_path.name
        == "survival_per_disease_metrics__ahi__test__epoch07_step1234.csv"
    )
    assert (
        args.inference_multilabel_per_disease_metrics_csv_path.name
        == "multilabel_per_disease_metrics__ahi__test__epoch07_step1234.csv"
    )
    assert args.inference_overview_csv_path == tmp_path / "results" / "inference" / "overview.csv"
    assert args.ckpt_epoch == 7
    assert args.ckpt_step == 1234
    assert args.ckpt_tag == "epoch07_step1234"


def test_prepare_inference_result_paths_falls_back_to_checkpoint_filename(tmp_path):
    ckpt_path = tmp_path / "epoch=12.ckpt"
    args = _infer_args()
    args.ckpt_path = str(ckpt_path)

    prepare_inference_result_paths(args, root=tmp_path / "results" / "inference", timestamp="20260524T000000Z")

    assert args.ckpt_epoch == 12
    assert args.ckpt_step is None
    assert args.ckpt_tag == "epoch12"


@pytest.mark.parametrize("package_name", RESULT_PACKAGES)
def test_prepare_inference_result_paths_describes_averaged_checkpoints_without_loading_payload(
    tmp_path, monkeypatch, package_name: str
):
    results_mod = importlib.import_module(f"{package_name}.results")
    paths = []
    for epoch in (3, 4, 5):
        path = tmp_path / f"epoch={epoch}-step={epoch * 100}.ckpt"
        path.touch()
        paths.append(path)
    monkeypatch.setattr(
        results_mod.torch, "load", lambda *args, **kwargs: pytest.fail("metadata path loaded checkpoint")
    )
    args = _infer_args()
    args.avg_ckpts = 3
    args.ckpt_path = str(paths[-1])

    results_mod.prepare_inference_result_paths(
        args,
        namespace=package_name,
        root=tmp_path / "results" / "inference",
        checkpoint_paths=paths,
        timestamp="20260524T000000Z",
    )

    assert args.ckpt_tag == "avg3_epoch03-05"
    assert args.ckpt_epoch == 5
    assert args.ckpt_step == 500
    assert args.inference_checkpoint_paths == [str(path) for path in paths]


def test_save_prediction_csv_appends_rows_and_serializes_lists(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "predictions.csv"
    args = _infer_args()
    args.prediction_run_id = make_prediction_run_id(args, timestamp="20260524T000000Z")

    save_prediction_csv(
        [
            {
                "path": "a.npz",
                "groundtruth": [1, 2],
                "prediction": [1, 1],
                "n_predictions": 2,
                "n_windows": 1,
                "token_starts": [0],
                "prob_0": [0.1, 0.2],
                "prob_1": [0.9, 0.8],
            }
        ],
        str(csv_path),
        args,
    )
    save_prediction_csv(
        [
            {
                "path": "b.npz",
                "groundtruth": 0,
                "prediction": 1,
                "n_predictions": 1,
                "n_windows": 1,
                "token_starts": [0],
                "prob_0": 0.4,
                "prob_1": 0.6,
            }
        ],
        str(csv_path),
        args,
    )

    df = pd.read_csv(csv_path)

    assert len(df) == 2
    assert df["prediction_run_id"].tolist() == [args.prediction_run_id, args.prediction_run_id]
    assert json.loads(df.loc[0, "groundtruth"]) == [1, 2]
    assert json.loads(df.loc[0, "token_starts"]) == [0]
    assert df.loc[1, "path"] == "b.npz"


@pytest.mark.parametrize("package_name", RESULT_PACKAGES)
def test_save_survival_per_disease_metrics_csv_appends_rows_with_metadata(tmp_path, monkeypatch, package_name: str):
    results_mod = importlib.import_module(f"{package_name}.results")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    args = _infer_args()
    results_mod.prepare_inference_result_paths(
        args,
        namespace=package_name,
        root=tmp_path / "results" / "inference",
        timestamp="20260524T000000Z",
    )

    results_mod.save_survival_per_disease_metrics_csv(
        [{"stage": "test", "disease_idx": 0, "disease": "d1", "n_labeled": 3, "n_events": 2, "c_index": 0.75}],
        str(args.inference_survival_per_disease_metrics_csv_path),
        args,
    )
    results_mod.save_survival_per_disease_metrics_csv(
        [
            {
                "stage": "test",
                "disease_idx": 1,
                "disease": "d2",
                "n_labeled": 1,
                "n_events": 0,
                "c_index": float("nan"),
                "extra_stat": 4.0,
            }
        ],
        str(args.inference_survival_per_disease_metrics_csv_path),
        args,
    )

    df = pd.read_csv(args.inference_survival_per_disease_metrics_csv_path)
    assert len(df) == 2
    assert df["prediction_run_id"].tolist() == [args.prediction_run_id, args.prediction_run_id]
    assert df["disease"].tolist() == ["d1", "d2"]
    assert df["n_events"].tolist() == [2, 0]
    assert "extra_stat" in df.columns


@pytest.mark.parametrize("package_name", RESULT_PACKAGES)
def test_save_multilabel_per_disease_metrics_csv_appends_rows_with_metadata(tmp_path, monkeypatch, package_name: str):
    results_mod = importlib.import_module(f"{package_name}.results")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    args = _infer_args()
    results_mod.prepare_inference_result_paths(
        args,
        namespace=package_name,
        root=tmp_path / "results" / "inference",
        timestamp="20260524T000000Z",
    )

    results_mod.save_multilabel_per_disease_metrics_csv(
        [
            {
                "stage": "test",
                "disease_idx": 0,
                "disease": "d1",
                "n_positive": 2,
                "n_negative": 3,
                "prevalence": 0.4,
                "auroc": 0.75,
                "auprc": 0.7,
            }
        ],
        str(args.inference_multilabel_per_disease_metrics_csv_path),
        args,
    )
    results_mod.save_multilabel_per_disease_metrics_csv(
        [
            {
                "stage": "test",
                "disease_idx": 1,
                "disease": "d2",
                "n_positive": 1,
                "n_negative": 4,
                "prevalence": 0.2,
                "auroc": 0.5,
                "auprc": 0.3,
                "extra_stat": 4.0,
            }
        ],
        str(args.inference_multilabel_per_disease_metrics_csv_path),
        args,
    )

    df = pd.read_csv(args.inference_multilabel_per_disease_metrics_csv_path)
    assert len(df) == 2
    assert df["prediction_run_id"].tolist() == [args.prediction_run_id, args.prediction_run_id]
    assert df["disease"].tolist() == ["d1", "d2"]
    assert df["n_positive"].tolist() == [2, 1]
    assert "extra_stat" in df.columns


def test_prediction_run_id_matches_results_and_predictions(tmp_path, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    args = _infer_args()
    prepare_inference_result_paths(args, root=tmp_path / "results" / "inference", timestamp="20260524T000000Z")

    save_result_csv({"test_loss": 0.1}, str(args.inference_metrics_csv_path), args)
    save_result_csv({"test_loss": 0.1}, str(args.inference_overview_csv_path), args)
    save_prediction_csv(
        [
            {
                "path": "a.npz",
                "groundtruth": 1,
                "prediction": 1,
                "n_predictions": 1,
                "n_windows": 1,
                "token_starts": [0],
            }
        ],
        str(args.inference_prediction_csv_path),
        args,
    )

    result_df = pd.read_csv(args.inference_metrics_csv_path)
    overview_df = pd.read_csv(args.inference_overview_csv_path)
    prediction_df = pd.read_csv(args.inference_prediction_csv_path)

    assert result_df.loc[0, "prediction_run_id"] == args.prediction_run_id
    assert overview_df.loc[0, "prediction_run_id"] == args.prediction_run_id
    assert prediction_df.loc[0, "prediction_run_id"] == args.prediction_run_id
    assert result_df.loc[0, "run_dir"] == str(args.run_dir)


def test_sleep2expert_inference_outputs_record_route_filter_metadata(tmp_path, monkeypatch):
    results_mod = importlib.import_module("sleep2expert.results")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    args = _infer_args()
    args.route_filter_active = True
    args.route_filter_groups = ["shared", "cardiac"]
    args.route_filter_expert_ids = [0, 3, 4]
    results_mod.prepare_inference_result_paths(
        args,
        namespace="sleep2expert",
        root=tmp_path / "results" / "inference",
        timestamp="20260524T000000Z",
    )

    results_mod.save_result_csv({"test_loss": 0.1}, str(args.inference_metrics_csv_path), args)
    results_mod.save_result_csv({"test_loss": 0.1}, str(args.inference_overview_csv_path), args)
    results_mod.save_prediction_csv(
        [
            {
                "path": "a.npz",
                "groundtruth": 1,
                "prediction": 1,
                "n_predictions": 1,
                "n_windows": 1,
                "token_starts": [0],
            }
        ],
        str(args.inference_prediction_csv_path),
        args,
    )
    results_mod.save_multilabel_per_disease_metrics_csv(
        [
            {
                "stage": "test",
                "disease_idx": 0,
                "disease": "d1",
                "n_positive": 2,
                "n_negative": 3,
                "prevalence": 0.4,
                "auroc": 0.75,
                "auprc": 0.7,
            }
        ],
        str(args.inference_multilabel_per_disease_metrics_csv_path),
        args,
    )
    results_mod.save_inference_manifest(args, {"test_loss": 0.1}, prediction_row_count=1)

    result_df = pd.read_csv(args.inference_metrics_csv_path)
    overview_df = pd.read_csv(args.inference_overview_csv_path)
    prediction_df = pd.read_csv(args.inference_prediction_csv_path)
    multilabel_df = pd.read_csv(args.inference_multilabel_per_disease_metrics_csv_path)
    manifest = json.loads(Path(args.manifest_path).read_text())

    for frame in (result_df, overview_df, multilabel_df):
        assert bool(frame.loc[0, "route_filter_active"]) is True
        assert frame.loc[0, "route_filter_groups"] == "shared,cardiac"
        assert frame.loc[0, "route_filter_expert_ids"] == "0,3,4"
    assert "route_filter_active" not in prediction_df.columns
    assert "route_filter_groups" not in prediction_df.columns
    assert "route_filter_expert_ids" not in prediction_df.columns
    assert manifest["route_filter"] == {
        "active": True,
        "groups": ["shared", "cardiac"],
        "expert_ids": [0, 3, 4],
    }


def test_sleep2expert_inference_result_csv_records_inactive_route_filter(tmp_path, monkeypatch):
    results_mod = importlib.import_module("sleep2expert.results")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    args = _infer_args()
    results_mod.prepare_inference_result_paths(
        args,
        namespace="sleep2expert",
        root=tmp_path / "results" / "inference",
        timestamp="20260524T000000Z",
    )

    results_mod.save_result_csv({"test_loss": 0.1}, str(args.inference_metrics_csv_path), args)

    df = pd.read_csv(args.inference_metrics_csv_path)

    assert str(df.loc[0, "route_filter_active"]).lower() == "false"
    assert pd.isna(df.loc[0, "route_filter_groups"]) or df.loc[0, "route_filter_groups"] == ""
    assert pd.isna(df.loc[0, "route_filter_expert_ids"]) or df.loc[0, "route_filter_expert_ids"] == ""


def test_sleep2expert_prediction_run_id_hash_includes_route_filter(monkeypatch):
    results_mod = importlib.import_module("sleep2expert.results")

    class _FixedUuid:
        hex = "abcdef1234567890"

    monkeypatch.setattr(results_mod.uuid, "uuid4", lambda: _FixedUuid())
    full_args = _infer_args()
    filtered_args = _infer_args()
    filtered_args.route_filter_active = True
    filtered_args.route_filter_groups = ["shared"]
    filtered_args.route_filter_expert_ids = [0, 2]

    full_id = results_mod.make_prediction_run_id(
        full_args,
        timestamp="20260524T000000Z",
        namespace="sleep2expert",
    )
    filtered_id = results_mod.make_prediction_run_id(
        filtered_args,
        timestamp="20260524T000000Z",
        namespace="sleep2expert",
    )

    assert full_id.rsplit("__", 1)[0] == filtered_id.rsplit("__", 1)[0]
    assert full_id.rsplit("__", 1)[1] != filtered_id.rsplit("__", 1)[1]


def test_save_prediction_csv_skips_nonzero_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "predictions.csv"

    save_prediction_csv(
        [
            {
                "path": "a.npz",
                "groundtruth": 1,
                "prediction": 1,
                "n_predictions": 1,
                "n_windows": 1,
                "token_starts": [0],
            }
        ],
        str(csv_path),
        _infer_args(),
    )

    assert not csv_path.exists()


@pytest.mark.parametrize("package_name", RESULT_PACKAGES)
def test_automatic_inference_paths_across_namespaces(tmp_path, package_name: str):
    results_mod = importlib.import_module(f"{package_name}.results")
    args = _infer_args()
    args.label_name = "stage5"
    args.eval_split = "val"
    args.ckpt_path = str(tmp_path / "epoch=9-step=42.ckpt")

    results_mod.prepare_inference_result_paths(
        args,
        namespace=package_name,
        root=tmp_path / "results" / "inference",
        timestamp="20260524T000000Z",
    )

    assert args.run_dir == (tmp_path / "results" / "inference" / package_name / "stage5" / args.prediction_run_id)
    assert args.inference_metrics_csv_path.name == "metrics__stage5__val__epoch09_step42.csv"
    assert args.inference_prediction_csv_path.name == "predictions__stage5__val__epoch09_step42.csv"
    assert (
        args.inference_survival_per_disease_metrics_csv_path.name
        == "survival_per_disease_metrics__stage5__val__epoch09_step42.csv"
    )


@pytest.mark.parametrize("package_name", RESULT_PACKAGES)
def test_prediction_csv_append_overview_and_manifest_across_namespaces(tmp_path, monkeypatch, package_name: str):
    results_mod = importlib.import_module(f"{package_name}.results")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    args = _infer_args()
    results_mod.prepare_inference_result_paths(
        args,
        namespace=package_name,
        root=tmp_path / "results" / "inference",
        timestamp="20260524T000000Z",
    )

    results_mod.save_result_csv({"test_loss": 0.1}, str(args.inference_metrics_csv_path), args)
    results_mod.save_result_csv({"test_loss": 0.1}, str(args.inference_overview_csv_path), args)
    results_mod.save_prediction_csv(
        [
            {
                "path": "a.npz",
                "groundtruth": [1, 2],
                "prediction": [1, 1],
                "n_predictions": 2,
                "n_windows": 1,
                "token_starts": [0],
            }
        ],
        str(args.inference_prediction_csv_path),
        args,
    )
    results_mod.save_prediction_csv(
        [
            {
                "path": "b.npz",
                "groundtruth": 0,
                "prediction": 1,
                "n_predictions": 1,
                "n_windows": 1,
                "token_starts": [5],
            }
        ],
        str(args.inference_prediction_csv_path),
        args,
    )
    results_mod.save_survival_per_disease_metrics_csv(
        [{"stage": "test", "disease_idx": 0, "disease": "d1", "n_labeled": 3, "n_events": 2, "c_index": 0.75}],
        str(args.inference_survival_per_disease_metrics_csv_path),
        args,
    )
    results_mod.save_multilabel_per_disease_metrics_csv(
        [
            {
                "stage": "test",
                "disease_idx": 0,
                "disease": "d1",
                "n_positive": 2,
                "n_negative": 1,
                "prevalence": 2 / 3,
                "auroc": 0.75,
                "auprc": 0.7,
            }
        ],
        str(args.inference_multilabel_per_disease_metrics_csv_path),
        args,
    )
    results_mod.save_inference_manifest(args, {"test_loss": 0.1}, prediction_row_count=2)

    result_df = pd.read_csv(args.inference_metrics_csv_path)
    overview_df = pd.read_csv(args.inference_overview_csv_path)
    prediction_df = pd.read_csv(args.inference_prediction_csv_path)
    survival_df = pd.read_csv(args.inference_survival_per_disease_metrics_csv_path)
    multilabel_df = pd.read_csv(args.inference_multilabel_per_disease_metrics_csv_path)
    manifest = json.loads(Path(args.manifest_path).read_text())

    assert len(prediction_df) == 2
    assert result_df.loc[0, "prediction_run_id"] == args.prediction_run_id
    assert overview_df.loc[0, "prediction_run_id"] == args.prediction_run_id
    assert prediction_df["prediction_run_id"].tolist() == [args.prediction_run_id, args.prediction_run_id]
    assert json.loads(prediction_df.loc[0, "groundtruth"]) == [1, 2]
    assert survival_df.loc[0, "disease"] == "d1"
    assert multilabel_df.loc[0, "disease"] == "d1"
    assert manifest["prediction_run_id"] == args.prediction_run_id
    assert manifest["prediction_row_count"] == 2
    assert manifest["paths"]["prediction_csv_path"] == str(args.inference_prediction_csv_path)
    assert manifest["paths"]["survival_per_disease_metrics_csv_path"] == str(
        args.inference_survival_per_disease_metrics_csv_path
    )
    assert manifest["paths"]["multilabel_per_disease_metrics_csv_path"] == str(
        args.inference_multilabel_per_disease_metrics_csv_path
    )


@pytest.mark.parametrize("package_name", RESULT_PACKAGES)
def test_overview_csv_appends_runs_across_namespaces(tmp_path, monkeypatch, package_name: str):
    results_mod = importlib.import_module(f"{package_name}.results")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    overview_path = tmp_path / "results" / "inference" / "overview.csv"

    for timestamp in ("20260524T000000Z", "20260524T000001Z"):
        args = _infer_args()
        results_mod.prepare_inference_result_paths(
            args,
            namespace=package_name,
            root=tmp_path / "results" / "inference",
            timestamp=timestamp,
        )
        results_mod.save_result_csv({"test_loss": 0.1}, str(args.inference_overview_csv_path), args)

    df = pd.read_csv(overview_path)

    assert len(df) == 2
    assert df["prediction_run_id"].is_unique
    assert df["namespace"].tolist() == [package_name, package_name]


@pytest.mark.parametrize("package_name", RESULT_PACKAGES)
def test_empty_prediction_csv_is_created_across_namespaces(tmp_path, monkeypatch, package_name: str):
    results_mod = importlib.import_module(f"{package_name}.results")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "predictions.csv"

    results_mod.save_prediction_csv([], str(csv_path), _infer_args())

    df = pd.read_csv(csv_path)
    assert len(df) == 0
    assert {"prediction_run_id", "path", "groundtruth", "prediction"}.issubset(df.columns)


@pytest.mark.parametrize("package_name", RESULT_PACKAGES)
def test_prediction_csv_rank_zero_only_across_namespaces(tmp_path, monkeypatch, package_name: str):
    results_mod = importlib.import_module(f"{package_name}.results")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "predictions.csv"

    results_mod.save_prediction_csv(
        [
            {
                "path": "a.npz",
                "groundtruth": 1,
                "prediction": 1,
                "n_predictions": 1,
                "n_windows": 1,
                "token_starts": [0],
            }
        ],
        str(csv_path),
        _infer_args(),
    )

    assert not csv_path.exists()


@pytest.mark.parametrize("package_name", RESULT_PACKAGES)
def test_survival_per_disease_metrics_csv_rank_zero_only(tmp_path, monkeypatch, package_name: str):
    results_mod = importlib.import_module(f"{package_name}.results")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "survival_per_disease_metrics.csv"

    results_mod.save_survival_per_disease_metrics_csv(
        [{"stage": "test", "disease_idx": 0, "disease": "d1", "n_labeled": 3, "n_events": 2, "c_index": 0.75}],
        str(csv_path),
        _infer_args(),
    )

    assert not csv_path.exists()


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
