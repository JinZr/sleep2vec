import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path

import pandas as pd
import pytest

VARIANT_PACKAGES = ("sleep2vec2", "sleep2expert")
RUNTIME_PACKAGES = ("sleep2vec", "sleep2vec2", "sleep2expert")


@pytest.mark.parametrize("package_name", VARIANT_PACKAGES)
def test_variant_infer_parse_args_accepts_inference_preset_path(monkeypatch: pytest.MonkeyPatch, package_name: str):
    infer_mod = importlib.import_module(f"{package_name}.infer")
    monkeypatch.setattr(
        "sys.argv",
        [
            f"{package_name}.infer",
            "--config",
            "config.yaml",
            "--ckpt-path",
            "best.ckpt",
            "--label-name",
            "ahi",
            "--inference-preset-path",
            "preset.pkl",
        ],
    )

    args = infer_mod.parse_args()

    assert args.inference_preset_path == Path("preset.pkl")


@pytest.mark.parametrize("package_name", RUNTIME_PACKAGES)
def test_infer_parse_args_does_not_require_csv_paths(monkeypatch: pytest.MonkeyPatch, package_name: str):
    infer_mod = importlib.import_module(f"{package_name}.infer")
    monkeypatch.setattr(
        "sys.argv",
        [
            f"{package_name}.infer",
            "--config",
            "config.yaml",
            "--ckpt-path",
            "best.ckpt",
            "--label-name",
            "ahi",
        ],
    )

    args = infer_mod.parse_args()

    assert not hasattr(args, "results_csv_path")
    assert not hasattr(args, "predictions_csv_path")


@pytest.mark.parametrize("package_name", VARIANT_PACKAGES)
def test_variant_run_inference_applies_inference_preset_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, package_name: str
):
    infer_mod = importlib.import_module(f"{package_name}.infer")
    results_mod = importlib.import_module(f"{package_name}.results")
    captured: dict[str, object] = {}
    config_preset = tmp_path / "config.pkl"
    override_preset = tmp_path / "override.pkl"

    @dataclass
    class _DummyBundle:
        finetune: object = None
        averaging: object = None

    class _DummyModule:
        def __init__(self, args, model_cfg, finetune_config=None, averaging_config=None):
            captured["module_preset_path"] = args.finetune_preset_path

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def test(self, model=None, ckpt_path=None, dataloaders=None):
            return [{"ahi_pearson": 0.5}]

    def _apply_config(args):
        args.finetune_preset_path = config_preset
        return _DummyBundle(), object()

    def _build_loader(args):
        captured["loader_preset_path"] = args.finetune_preset_path
        return "loader"

    monkeypatch.setattr(infer_mod, "apply_finetune_config", _apply_config)
    monkeypatch.setattr(infer_mod, "_build_inference_loader", _build_loader)
    monkeypatch.setattr(infer_mod, "Sleep2vecFinetuning", _DummyModule)
    monkeypatch.setattr(infer_mod.pl, "Trainer", _DummyTrainer)
    monkeypatch.setattr(infer_mod, "_init_wandb", lambda args: None)
    monkeypatch.setattr(
        infer_mod,
        "prepare_inference_result_paths",
        lambda args, namespace, checkpoint_paths=None, timestamp=None: results_mod.prepare_inference_result_paths(
            args,
            namespace=namespace,
            root=tmp_path / "results" / "inference",
            checkpoint_paths=checkpoint_paths,
            timestamp=timestamp or "20260524T000000Z",
        ),
    )
    monkeypatch.setattr(infer_mod, "save_result_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(infer_mod, "save_prediction_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(infer_mod, "save_inference_manifest", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        label_name="ahi",
        avg_ckpts=1,
        ckpt_path="/tmp/model.ckpt",
        avg_ckpt_dir=None,
        config=Path("dummy.yaml"),
        precision=32,
        accelerator="cpu",
        devices=[0],
        batch_size=4,
        eval_split="test",
        seed=4523,
        wandb=False,
        inference_preset_path=override_preset,
    )

    infer_mod.run_inference(args)

    assert captured["loader_preset_path"] == override_preset
    assert captured["module_preset_path"] == override_preset


@pytest.mark.parametrize("package_name", RUNTIME_PACKAGES)
def test_run_inference_writes_automatic_prediction_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, package_name: str
):
    infer_mod = importlib.import_module(f"{package_name}.infer")
    results_mod = importlib.import_module(f"{package_name}.results")
    captured: dict[str, object] = {}

    @dataclass
    class _DummyBundle:
        finetune: object = None
        averaging: object = None

    class _DummyModule:
        def __init__(self, args, model_cfg, finetune_config=None, averaging_config=None):
            self.prediction_rows = [{"path": "sample.npz", "groundtruth": 1, "prediction": 1}]

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def test(self, model=None, ckpt_path=None, dataloaders=None):
            return [{"test_loss": 0.5}]

    def _apply_config(args):
        args.finetune_preset_path = Path("config.pkl")
        return _DummyBundle(), object()

    def _save_result(metrics, csv_path, args):
        captured.setdefault("result_csv_paths", []).append(csv_path)
        captured.setdefault("result_prediction_run_ids", []).append(args.prediction_run_id)

    def _save_prediction(rows, csv_path, args):
        captured["prediction_rows"] = rows
        captured["prediction_csv_path"] = csv_path
        captured["prediction_run_id"] = args.prediction_run_id

    def _save_manifest(args, metrics, prediction_row_count=0):
        captured["manifest_path"] = args.manifest_path
        captured["manifest_prediction_run_id"] = args.prediction_run_id
        captured["manifest_prediction_row_count"] = prediction_row_count

    monkeypatch.setattr(infer_mod, "apply_finetune_config", _apply_config)
    monkeypatch.setattr(infer_mod, "_build_inference_loader", lambda args: "loader")
    monkeypatch.setattr(infer_mod, "Sleep2vecFinetuning", _DummyModule)
    monkeypatch.setattr(infer_mod.pl, "Trainer", _DummyTrainer)
    monkeypatch.setattr(infer_mod, "_init_wandb", lambda args: None)
    monkeypatch.setattr(
        infer_mod,
        "prepare_inference_result_paths",
        lambda args, namespace, checkpoint_paths=None, timestamp=None: results_mod.prepare_inference_result_paths(
            args,
            namespace=namespace,
            root=tmp_path / "results" / "inference",
            checkpoint_paths=checkpoint_paths,
            timestamp=timestamp or "20260524T000000Z",
        ),
    )
    monkeypatch.setattr(infer_mod, "save_result_csv", _save_result)
    monkeypatch.setattr(infer_mod, "save_prediction_csv", _save_prediction)
    monkeypatch.setattr(infer_mod, "save_inference_manifest", _save_manifest)

    args = argparse.Namespace(
        label_name="sex",
        avg_ckpts=1,
        ckpt_path="/tmp/model.ckpt",
        avg_ckpt_dir=None,
        config=Path("dummy.yaml"),
        precision=32,
        accelerator="cpu",
        devices=[0],
        batch_size=4,
        eval_split="test",
        seed=4523,
        wandb=False,
        inference_preset_path=None,
    )

    infer_mod.run_inference(args)

    assert captured["result_csv_paths"] == [
        str(args.inference_metrics_csv_path),
        str(args.inference_overview_csv_path),
    ]
    assert args.prediction_run_id in str(args.run_dir)
    assert args.inference_metrics_csv_path.name == "metrics__sex__test__model.csv"
    assert args.inference_prediction_csv_path.name == "predictions__sex__test__model.csv"
    assert captured["prediction_csv_path"] == str(args.inference_prediction_csv_path)
    assert captured["prediction_rows"] == [{"path": "sample.npz", "groundtruth": 1, "prediction": 1}]
    assert captured["prediction_run_id"] == captured["result_prediction_run_ids"][0]
    assert captured["manifest_prediction_run_id"] == args.prediction_run_id
    assert captured["manifest_prediction_row_count"] == 1


@pytest.mark.parametrize("package_name", RUNTIME_PACKAGES)
def test_run_inference_logs_metrics_and_files_to_wandb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, package_name: str
):
    infer_mod = importlib.import_module(f"{package_name}.infer")
    results_mod = importlib.import_module(f"{package_name}.results")
    captured: dict[str, object] = {"artifact_files": [], "events": []}

    @dataclass
    class _DummyBundle:
        finetune: object = None
        averaging: object = None

    class _DummyModule:
        def __init__(self, args, model_cfg, finetune_config=None, averaging_config=None):
            self.prediction_rows = [{"path": "sample.npz", "groundtruth": 1, "prediction": 1}]

    class _DummyTrainer:
        def __init__(self, *args, **kwargs):
            pass

        def test(self, model=None, ckpt_path=None, dataloaders=None):
            return [{"test_loss": 0.5, "test_recall": 0.75}]

    class _DummyArtifact:
        def __init__(self, name, type):
            captured["artifact_name"] = name
            captured["artifact_type"] = type

        def add_file(self, path, name=None):
            assert Path(path).exists()
            captured["artifact_files"].append((Path(path).name, name))

    def _apply_config(args):
        args.finetune_preset_path = Path("config.pkl")
        return _DummyBundle(), object()

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(infer_mod, "apply_finetune_config", _apply_config)
    monkeypatch.setattr(infer_mod, "_build_inference_loader", lambda args: "loader")
    monkeypatch.setattr(infer_mod, "Sleep2vecFinetuning", _DummyModule)
    monkeypatch.setattr(infer_mod.pl, "Trainer", _DummyTrainer)
    monkeypatch.setattr(infer_mod, "_init_wandb", lambda args: object())
    monkeypatch.setattr(
        infer_mod,
        "prepare_inference_result_paths",
        lambda args, namespace, checkpoint_paths=None, timestamp=None: results_mod.prepare_inference_result_paths(
            args,
            namespace=namespace,
            root=tmp_path / "results" / "inference",
            checkpoint_paths=checkpoint_paths,
            timestamp=timestamp or "20260524T000000Z",
        ),
    )
    monkeypatch.setattr(
        infer_mod.wandb,
        "log",
        lambda payload: (captured.__setitem__("wandb_log", payload), captured["events"].append("log")),
    )
    monkeypatch.setattr(infer_mod.wandb, "Artifact", _DummyArtifact)
    monkeypatch.setattr(
        infer_mod.wandb,
        "log_artifact",
        lambda artifact: (captured.__setitem__("logged_artifact", artifact), captured["events"].append("artifact")),
    )
    monkeypatch.setattr(infer_mod.wandb, "finish", lambda: captured["events"].append("finish"))

    args = argparse.Namespace(
        label_name="sex",
        avg_ckpts=1,
        ckpt_path="/tmp/model.ckpt",
        avg_ckpt_dir=None,
        config=Path("dummy.yaml"),
        precision=32,
        accelerator="cpu",
        devices=[0],
        batch_size=4,
        eval_split="test",
        seed=4523,
        wandb=True,
        inference_preset_path=None,
        lr=1e-4,
        n_few_shot=None,
        channel_names=["ppg"],
    )

    infer_mod.run_inference(args)

    assert captured["wandb_log"] == {
        "test_loss": 0.5,
        "test_recall": 0.75,
        "prediction_row_count": 1,
    }
    assert captured["artifact_name"] == f"inference-{args.prediction_run_id}"
    assert captured["artifact_type"] == "inference"
    assert captured["artifact_files"] == [
        (args.inference_metrics_csv_path.name, "metrics.csv"),
        (args.inference_prediction_csv_path.name, "predictions.csv"),
        ("run_manifest.json", "run_manifest.json"),
        ("overview.csv", "overview.csv"),
    ]
    assert captured["events"] == ["log", "artifact", "finish"]


@pytest.mark.parametrize("package_name", VARIANT_PACKAGES)
def test_variant_result_csv_records_effective_preset_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, package_name: str
):
    results_mod = importlib.import_module(f"{package_name}.results")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    csv_path = tmp_path / "results.csv"
    args = argparse.Namespace(
        config=Path("configs/ppg_ahi_finetune_large.yaml"),
        label_name="ahi",
        eval_split="test",
        ckpt_path="best",
        lr=1e-4,
        batch_size=8,
        n_few_shot=None,
        channel_names=["ppg"],
        wandb_name=None,
        wandb_id=None,
        finetune_preset_path=Path("config/preset.pkl"),
        inference_preset_path=Path("override/preset.pkl"),
    )

    results_mod.save_result_csv({"test_loss": 0.1}, str(csv_path), args)

    df = pd.read_csv(csv_path)

    assert df.loc[0, "preset_path"] == "override/preset.pkl"
