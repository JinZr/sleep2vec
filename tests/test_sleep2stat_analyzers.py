import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from sleep2stat.analyzers.model_downstream import (
    Sleep2vecDownstreamAnalyzer,
    _build_kaldi_datasets,
    decode_ahi_logits,
    decode_classification_logits,
    decode_regression_logits,
)
from sleep2stat.analyzers.yasa import YasaBandpowerAnalyzer, YasaStageAnalyzer
from sleep2stat.config import AnalyzerConfig, ChannelSpec, DataConfig, SignalsConfig
from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import SleepRecord, load_records


def _record() -> SleepRecord:
    return SleepRecord(
        record_id="rec1",
        path="rec1.npz",
        split="test",
        source="unit",
        duration_sec=600,
        token_sec=30,
        max_tokens=20,
        metadata={"age": 60, "sex": 1},
    )


def _batch():
    return {
        "metadata": {"path": ["rec1.npz"]},
        "token_start": torch.tensor([0]),
        "length": torch.tensor([3]),
    }


def test_decode_sequence_classification_logits_to_epoch_alignment():
    record = _record()
    logits = torch.zeros(1, 3, 5)
    logits[0, :, 2] = 4.0

    results = decode_classification_logits(
        "stage5_model",
        logits,
        _batch(),
        {"rec1.npz": record},
        include_probabilities=True,
        include_logits=False,
    )

    frame = results[0].epoch
    assert results[0].record_id == "rec1"
    assert frame["stage5_model_pred"].tolist() == [2, 2, 2]
    assert frame["start_sec"].tolist() == [0.0, 30.0, 60.0]
    assert "stage5_model_prob_REM" in frame.columns


def test_decode_scalar_regression_logits_to_night_stats():
    record = _record()
    logits = torch.tensor([[62.5]])

    results = decode_regression_logits("age_model", logits, _batch(), {"rec1.npz": record})

    assert results[0].night["age_model_pred"] == 62.5
    assert results[0].night["age_model_abs_error_vs_metadata"] == 2.5


def test_decode_ahi_logits_to_second_events_and_model_ahi():
    record = _record()
    prob = np.array([0.1] * 5 + [0.9] * 12 + [0.1] * 73, dtype=np.float32)
    logits = torch.from_numpy(np.log(prob / (1.0 - prob))).reshape(1, 3, 30)

    results = decode_ahi_logits("ahi_model", logits, _batch(), {"rec1.npz": record}, threshold=0.5)

    assert len(results) == 1
    assert results[0].second["ahi_model_pred"].sum() == 12
    assert len(results[0].events) == 1
    assert results[0].night["ahi_model_pred_event_count"] == 1
    assert results[0].night["ahi_model_pred_ahi"] == 40.0


def test_decode_ahi_uses_strict_threshold_boundary():
    record = _record()
    logits = torch.zeros(1, 1, 30)
    results = decode_ahi_logits("ahi_model", logits, _batch(), {"rec1.npz": record}, threshold=0.5)

    assert results[0].second["ahi_model_pred"].sum() == 0
    assert len(results[0].events) == 0


def test_decode_scalar_classification_respects_probability_and_logit_flags():
    record = _record()
    logits = torch.tensor([[1.0, 3.0]])

    no_prob = decode_classification_logits(
        "sex_model",
        logits,
        _batch(),
        {"rec1.npz": record},
        include_probabilities=False,
        include_logits=False,
    )
    with_logits = decode_classification_logits(
        "sex_model",
        logits,
        _batch(),
        {"rec1.npz": record},
        include_probabilities=False,
        include_logits=True,
    )

    assert "sex_model_prob_male" not in no_prob[0].night
    assert "sex_model_logit_0" in with_logits[0].night


def test_downstream_analyzer_prepare_and_run_with_mock_checkpoint(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, ppg=np.arange(4, dtype=np.float32))
    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": {}}, ckpt_path)
    calls = {"on_load_checkpoint": 0, "load_state_dict": 0, "eval_model": 0}
    captured_tokens = {}

    class FakeEvalModel:
        def __call__(self, batch):
            captured_tokens["ppg"] = batch["tokens"]["ppg"].detach().cpu()
            return torch.tensor([[0.0, 3.0]])

    class FakeModule:
        def __init__(self, *args, **kwargs):
            self.model = FakeEvalModel()

        def on_load_checkpoint(self, checkpoint):
            calls["on_load_checkpoint"] += 1

        def load_state_dict(self, state_dict, strict=True):
            calls["load_state_dict"] += int(strict)

        def eval(self):
            return self

        def to(self, device):
            return self

        def _get_eval_model(self):
            calls["eval_model"] += 1
            return self.model

    def fake_apply_config(args):
        args.is_classification = True
        args.is_seq = False
        args.output_dim = 2
        return SimpleNamespace(finetune=None, averaging=None), SimpleNamespace()

    def fake_import(name):
        if name.endswith(".common"):
            return SimpleNamespace(apply_finetune_config=fake_apply_config)
        if name.endswith(".sleep2vec_finetuning"):
            return SimpleNamespace(Sleep2vecFinetuning=FakeModule)
        if name.endswith(".utils"):
            return SimpleNamespace(move_to_device=lambda data, device: data)
        raise AssertionError(name)

    monkeypatch.setattr("sleep2stat.analyzers.model_downstream.importlib.import_module", fake_import)
    analyzer = Sleep2vecDownstreamAnalyzer(
        AnalyzerConfig(
            name="sex_model",
            type="sleep2vec_downstream",
            namespace="sleep2vec",
            label_name="sex",
            config=tmp_path / "config.yaml",
            ckpt_path=ckpt_path,
            input_channels=["ppg"],
            batch_size=1,
        )
    )
    context = Sleep2statContext(
        config=SimpleNamespace(
            signals=SignalsConfig(
                channels={"ppg": ChannelSpec(source="ppg", sfreq=1, kind="ppg", input_dim=2, scale=2.0)}
            ),
            outputs=SimpleNamespace(include_probabilities=True, include_raw_logits=False),
        ),
        device="cpu",
        num_workers=0,
    )

    analyzer.prepare(context)
    results, failures = analyzer.run(
        [
            SleepRecord(
                record_id="rec1",
                path=npz_path,
                split="test",
                source="unit",
                duration_sec=60,
                token_sec=30,
                max_tokens=2,
                metadata={"sex": 1},
            )
        ],
        context,
    )

    assert calls == {"on_load_checkpoint": 1, "load_state_dict": 1, "eval_model": 1}
    assert failures == []
    torch.testing.assert_close(captured_tokens["ppg"], torch.tensor([[[0.0, 2.0], [4.0, 6.0]]]))
    assert results[0].night["sex_model_pred"] == 1


def test_downstream_analyzer_preserves_duplicate_npz_path_records(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "shared.npz"
    np.savez(npz_path, ppg=np.arange(4, dtype=np.float32))
    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": {}}, ckpt_path)
    captured_ids = {}

    class FakeEvalModel:
        def __call__(self, batch):
            captured_ids["id"] = list(batch["id"])
            logits = torch.zeros(len(batch["id"]), 2)
            logits[:, 1] = 3.0
            return logits

    class FakeModule:
        def __init__(self, *args, **kwargs):
            self.model = FakeEvalModel()

        def on_load_checkpoint(self, checkpoint):
            return None

        def load_state_dict(self, state_dict, strict=True):
            return None

        def eval(self):
            return self

        def to(self, device):
            return self

        def _get_eval_model(self):
            return self.model

    def fake_apply_config(args):
        args.is_classification = True
        args.is_seq = False
        args.output_dim = 2
        return SimpleNamespace(finetune=None, averaging=None), SimpleNamespace()

    def fake_import(name):
        if name.endswith(".common"):
            return SimpleNamespace(apply_finetune_config=fake_apply_config)
        if name.endswith(".sleep2vec_finetuning"):
            return SimpleNamespace(Sleep2vecFinetuning=FakeModule)
        if name.endswith(".utils"):
            return SimpleNamespace(move_to_device=lambda data, device: data)
        raise AssertionError(name)

    monkeypatch.setattr("sleep2stat.analyzers.model_downstream.importlib.import_module", fake_import)
    analyzer = Sleep2vecDownstreamAnalyzer(
        AnalyzerConfig(
            name="sex_model",
            type="sleep2vec_downstream",
            namespace="sleep2vec",
            label_name="sex",
            config=tmp_path / "config.yaml",
            ckpt_path=ckpt_path,
            input_channels=["ppg"],
            batch_size=2,
        )
    )
    context = Sleep2statContext(
        config=SimpleNamespace(
            signals=SignalsConfig(channels={"ppg": ChannelSpec(source="ppg", sfreq=1, kind="ppg", input_dim=2)}),
            outputs=SimpleNamespace(include_probabilities=True, include_raw_logits=False),
        ),
        device="cpu",
        num_workers=0,
    )

    analyzer.prepare(context)
    results, failures = analyzer.run(
        [
            SleepRecord(
                record_id="rec_a",
                path=npz_path,
                split="test",
                source="unit",
                duration_sec=60,
                token_sec=30,
                max_tokens=2,
                metadata={"sex": 1},
            ),
            SleepRecord(
                record_id="rec_b",
                path=npz_path,
                split="test",
                source="unit",
                duration_sec=60,
                token_sec=30,
                max_tokens=2,
                metadata={"sex": 1},
            ),
        ],
        context,
    )

    assert failures == []
    assert captured_ids["id"] == ["rec_a", "rec_b"]
    assert [result.record_id for result in results] == ["rec_a", "rec_b"]


def test_downstream_analyzer_retries_failed_batch_by_record(monkeypatch, tmp_path: Path):
    for name in ("good", "bad"):
        np.savez(tmp_path / f"{name}.npz", ppg=np.arange(4, dtype=np.float32))
    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": {}}, ckpt_path)

    class FakeEvalModel:
        def __call__(self, batch):
            ids = list(batch["id"])
            if len(ids) > 1:
                raise RuntimeError("batch failed")
            if ids[0] == "bad":
                raise ValueError("bad record")
            return torch.tensor([[0.0, 3.0]])

    class FakeModule:
        def __init__(self, *args, **kwargs):
            self.model = FakeEvalModel()

        def on_load_checkpoint(self, checkpoint):
            return None

        def load_state_dict(self, state_dict, strict=True):
            return None

        def eval(self):
            return self

        def to(self, device):
            return self

        def _get_eval_model(self):
            return self.model

    def fake_apply_config(args):
        args.is_classification = True
        args.is_seq = False
        args.output_dim = 2
        return SimpleNamespace(finetune=None, averaging=None), SimpleNamespace()

    def fake_import(name):
        if name.endswith(".common"):
            return SimpleNamespace(apply_finetune_config=fake_apply_config)
        if name.endswith(".sleep2vec_finetuning"):
            return SimpleNamespace(Sleep2vecFinetuning=FakeModule)
        if name.endswith(".utils"):
            return SimpleNamespace(move_to_device=lambda data, device: data)
        raise AssertionError(name)

    monkeypatch.setattr("sleep2stat.analyzers.model_downstream.importlib.import_module", fake_import)
    analyzer = Sleep2vecDownstreamAnalyzer(
        AnalyzerConfig(
            name="sex_model",
            type="sleep2vec_downstream",
            namespace="sleep2vec",
            label_name="sex",
            config=tmp_path / "config.yaml",
            ckpt_path=ckpt_path,
            input_channels=["ppg"],
            batch_size=2,
        )
    )
    context = Sleep2statContext(
        config=SimpleNamespace(
            signals=SignalsConfig(channels={"ppg": ChannelSpec(source="ppg", sfreq=1, kind="ppg", input_dim=2)}),
            outputs=SimpleNamespace(include_probabilities=True, include_raw_logits=False),
        ),
        device="cpu",
        num_workers=0,
    )

    analyzer.prepare(context)
    results, failures = analyzer.run(
        [
            SleepRecord("good", tmp_path / "good.npz", "test", "unit", 60, 30, 2, {}),
            SleepRecord("bad", tmp_path / "bad.npz", "test", "unit", 60, 30, 2, {}),
        ],
        context,
    )

    assert [result.record_id for result in results] == ["good"]
    assert [(failure.record_id, failure.error_type) for failure in failures] == [("bad", "ValueError")]


def test_downstream_analyzer_respects_epoch_probability_flag(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, ppg=np.arange(4, dtype=np.float32))
    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": {}}, ckpt_path)

    class FakeEvalModel:
        def __call__(self, batch):
            logits = torch.zeros(1, 2, 5)
            logits[0, :, 2] = 4.0
            return logits

    class FakeModule:
        def __init__(self, *args, **kwargs):
            self.model = FakeEvalModel()

        def on_load_checkpoint(self, checkpoint):
            return None

        def load_state_dict(self, state_dict, strict=True):
            return None

        def eval(self):
            return self

        def to(self, device):
            return self

        def _get_eval_model(self):
            return self.model

    def fake_apply_config(args):
        args.is_classification = True
        args.is_seq = True
        args.output_dim = 5
        return SimpleNamespace(finetune=None, averaging=None), SimpleNamespace()

    def fake_import(name):
        if name.endswith(".common"):
            return SimpleNamespace(apply_finetune_config=fake_apply_config)
        if name.endswith(".sleep2vec_finetuning"):
            return SimpleNamespace(Sleep2vecFinetuning=FakeModule)
        if name.endswith(".utils"):
            return SimpleNamespace(move_to_device=lambda data, device: data)
        raise AssertionError(name)

    monkeypatch.setattr("sleep2stat.analyzers.model_downstream.importlib.import_module", fake_import)
    analyzer = Sleep2vecDownstreamAnalyzer(
        AnalyzerConfig(
            name="stage5_model",
            type="sleep2vec_downstream",
            namespace="sleep2vec",
            label_name="stage5",
            config=tmp_path / "config.yaml",
            ckpt_path=ckpt_path,
            input_channels=["ppg"],
            batch_size=1,
            outputs={"epoch_proba": False},
        )
    )
    context = Sleep2statContext(
        config=SimpleNamespace(
            signals=SignalsConfig(channels={"ppg": ChannelSpec(source="ppg", sfreq=1, kind="ppg", input_dim=2)}),
            outputs=SimpleNamespace(include_probabilities=True, include_raw_logits=False),
        ),
        device="cpu",
        num_workers=0,
    )

    analyzer.prepare(context)
    results, failures = analyzer.run(
        [
            SleepRecord(
                record_id="rec1",
                path=npz_path,
                split="test",
                source="unit",
                duration_sec=60,
                token_sec=30,
                max_tokens=2,
                metadata={},
            )
        ],
        context,
    )

    frame = results[0].epoch
    assert failures == []
    assert "stage5_model_pred" in frame.columns
    assert "stage5_model_confidence" in frame.columns
    assert not any(column.startswith("stage5_model_prob_") for column in frame.columns)


def test_downstream_analyzer_reports_prefiltered_records(monkeypatch, tmp_path: Path):
    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": {}}, ckpt_path)

    class FakeModule:
        def __init__(self, *args, **kwargs):
            self.model = lambda batch: torch.tensor([[0.0, 1.0]])

        def on_load_checkpoint(self, checkpoint):
            return None

        def load_state_dict(self, state_dict, strict=True):
            return None

        def eval(self):
            return self

        def to(self, device):
            return self

        def _get_eval_model(self):
            return self.model

    def fake_apply_config(args):
        args.is_classification = True
        args.is_seq = False
        args.output_dim = 2
        return SimpleNamespace(finetune=None, averaging=None), SimpleNamespace()

    def fake_import(name):
        if name.endswith(".common"):
            return SimpleNamespace(apply_finetune_config=fake_apply_config)
        if name.endswith(".sleep2vec_finetuning"):
            return SimpleNamespace(Sleep2vecFinetuning=FakeModule)
        if name.endswith(".utils"):
            return SimpleNamespace(move_to_device=lambda data, device: data)
        raise AssertionError(name)

    monkeypatch.setattr("sleep2stat.analyzers.model_downstream.importlib.import_module", fake_import)
    analyzer = Sleep2vecDownstreamAnalyzer(
        AnalyzerConfig(
            name="sex_model",
            type="sleep2vec_downstream",
            namespace="sleep2vec",
            label_name="sex",
            config=tmp_path / "config.yaml",
            ckpt_path=ckpt_path,
            input_channels=["ppg"],
            batch_size=1,
        )
    )
    context = Sleep2statContext(
        config=SimpleNamespace(
            signals=SignalsConfig(channels={"ppg": ChannelSpec(source="ppg", sfreq=1, kind="ppg", input_dim=2)}),
            outputs=SimpleNamespace(include_probabilities=True, include_raw_logits=False),
        ),
        device="cpu",
        num_workers=0,
    )

    analyzer.prepare(context)
    results, failures = analyzer.run(
        [
            SleepRecord(
                record_id="rec1",
                path=tmp_path / "missing.npz",
                split="test",
                source="unit",
                duration_sec=60,
                token_sec=30,
                max_tokens=2,
                metadata={},
            )
        ],
        context,
    )

    assert results == []
    assert failures[0].record_id == "rec1"
    assert failures[0].error_type == "RecordUnavailable"


def test_yasa_stage_analyzer_with_mock_sleep_staging(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, eeg=np.ones(6000, dtype=np.float32))

    class FakeStaging:
        def __init__(self, raw, eeg_name=None, eog_name=None, emg_name=None, metadata=None):
            assert eeg_name == "EEG"
            assert metadata == {"age": 60.0, "male": True}

        def predict(self):
            return np.array(["W", "N2"])

        def predict_proba(self):
            return pd.DataFrame({"W": [0.9, 0.1], "N2": [0.1, 0.9]})

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: _fake_mne_module() if name == "mne" else SimpleNamespace(SleepStaging=FakeStaging),
    )
    analyzer = YasaStageAnalyzer(AnalyzerConfig(name="yasa_stage", type="yasa_stage", input_channels=["eeg"]))
    context = _yasa_context(tmp_path)

    analyzer.prepare(context)
    results, failures = analyzer.run([_yasa_record(npz_path)], context)

    assert failures == []
    frame = results[0].epoch
    assert frame["yasa_stage_pred"].tolist() == [0, 2]
    assert frame["yasa_stage_label"].tolist() == ["W", "N2"]
    assert frame["yasa_stage_prob_W"].tolist() == [0.9, 0.1]
    assert "yasa_stage_probabilities" in results[0].arrays


def test_yasa_stage_probability_arrays_follow_output_flag(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, eeg=np.ones(6000, dtype=np.float32))

    class FakeStaging:
        def __init__(self, *args, **kwargs):
            return None

        def predict(self):
            return np.array(["W", "N2"])

        def predict_proba(self):
            return pd.DataFrame({"W": [0.9, 0.1], "N2": [0.1, 0.9]})

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: _fake_mne_module() if name == "mne" else SimpleNamespace(SleepStaging=FakeStaging),
    )
    analyzer = YasaStageAnalyzer(AnalyzerConfig(name="yasa_stage", type="yasa_stage", input_channels=["eeg"]))
    context = _yasa_context(tmp_path, include_probabilities=False)

    analyzer.prepare(context)
    results, failures = analyzer.run([_yasa_record(npz_path)], context)

    assert failures == []
    assert "yasa_stage_prob_W" not in results[0].epoch.columns
    assert results[0].arrays == {}


def test_yasa_stage_drops_non_binary_numeric_sex_metadata(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, eeg=np.ones(6000, dtype=np.float32))
    captured = {}

    class FakeStaging:
        def __init__(self, raw, eeg_name=None, eog_name=None, emg_name=None, metadata=None):
            captured["metadata"] = metadata

        def predict(self):
            return np.array(["W", "N2"])

        def predict_proba(self):
            return None

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: _fake_mne_module() if name == "mne" else SimpleNamespace(SleepStaging=FakeStaging),
    )
    analyzer = YasaStageAnalyzer(AnalyzerConfig(name="yasa_stage", type="yasa_stage", input_channels=["eeg"]))
    context = _yasa_context(tmp_path)
    record = _yasa_record(npz_path)
    record.metadata["sex"] = -1

    analyzer.prepare(context)
    _, failures = analyzer.run([record], context)

    assert failures == []
    assert captured["metadata"] == {"age": 60.0}


def test_yasa_bandpower_analyzer_with_mock_bandpower(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, eeg=np.ones(6000, dtype=np.float32))

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: (
            _fake_mne_module()
            if name == "mne"
            else SimpleNamespace(bandpower=lambda raw: pd.DataFrame({"Chan": ["EEG"], "Delta": [0.4], "Alpha": [0.2]}))
        ),
    )
    analyzer = YasaBandpowerAnalyzer(
        AnalyzerConfig(
            name="yasa_bandpower",
            type="yasa_bandpower",
            input_channels=["eeg"],
            outputs={"by_stage": False, "bands": ["delta", "alpha"]},
        )
    )
    context = _yasa_context(tmp_path)

    analyzer.prepare(context)
    results, failures = analyzer.run([_yasa_record(npz_path)], context)

    assert failures == []
    assert results[0].epoch["yasa_bandpower_delta_rel"].tolist() == [0.4, 0.4]
    assert results[0].night["yasa_bandpower_delta_rel_mean"] == 0.4
    assert results[0].night["yasa_bandpower_alpha_rel_mean"] == 0.2


def test_yasa_bandpower_by_stage_uses_prior_stage_source(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, eeg=np.ones(6000, dtype=np.float32))

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: (
            _fake_mne_module()
            if name == "mne"
            else SimpleNamespace(bandpower=lambda raw: pd.DataFrame({"Chan": ["EEG"], "Sigma": [0.3]}))
        ),
    )
    analyzer = YasaBandpowerAnalyzer(
        AnalyzerConfig(
            name="yasa_bandpower",
            type="yasa_bandpower",
            input_channels=["eeg"],
            outputs={"stage_source": "yasa_stage", "bands": ["sigma"]},
        )
    )
    context = _yasa_context(tmp_path)
    stage_epoch = pd.DataFrame({"record_id": ["rec1", "rec1"], "token_idx": [0, 1], "yasa_stage_pred": [2, 3]})

    analyzer.prepare(context)
    results, failures = analyzer.run(
        [_yasa_record(npz_path)],
        context,
        prior_results=[AnalyzerResult("yasa_stage", "rec1", epoch=stage_epoch)],
    )

    assert failures == []
    assert results[0].night["yasa_bandpower_N2_sigma_mean"] == 0.3
    assert results[0].night["yasa_bandpower_N3_sigma_mean"] == 0.3


def test_yasa_bandpower_by_stage_requires_stage_source(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, eeg=np.ones(6000, dtype=np.float32))
    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: _fake_mne_module() if name == "mne" else SimpleNamespace(bandpower=lambda raw: pd.DataFrame()),
    )
    analyzer = YasaBandpowerAnalyzer(
        AnalyzerConfig(name="yasa_bandpower", type="yasa_bandpower", input_channels=["eeg"])
    )
    context = _yasa_context(tmp_path)

    analyzer.prepare(context)
    results, failures = analyzer.run([_yasa_record(npz_path)], context)

    assert results == []
    assert failures[0].record_id == "rec1"
    assert "stage_source" in failures[0].message


def test_load_records_reads_kaldi_manifest(tmp_path: Path):
    root = _write_tiny_kaldi_manifest(tmp_path)

    records = load_records(
        DataConfig(
            backend="kaldi",
            index=None,
            split=["test"],
            kaldi_data_root=root,
            kaldi_manifest=root / "manifest.json",
            token_sec=30,
            max_tokens=2,
        )
    )

    assert [record.record_id for record in records] == ["sample_a", "sample_b"]
    assert records[0].duration_sec == 60
    assert records[0].metadata["sample_key"] == "sample_a"


def test_load_records_rejects_duplicate_record_ids_by_default(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text(
        "path,duration,split,source,patient_id\n" "/tmp/a.npz,60,test,unit,p001\n" "/tmp/b.npz,60,test,unit,p001\n"
    )

    with pytest.raises(ValueError, match="duplicate record_id"):
        load_records(
            DataConfig(
                backend="npz",
                index=index,
                split=["test"],
                record_id_columns=["source", "patient_id"],
            )
        )


def test_kaldi_dataset_routing_filters_to_pending_sample_keys(tmp_path: Path):
    root = _write_tiny_kaldi_manifest(tmp_path)
    context = Sleep2statContext(
        config=SimpleNamespace(
            data=DataConfig(
                backend="kaldi",
                index=None,
                split=["test"],
                kaldi_data_root=root,
                kaldi_manifest=root / "manifest.json",
                max_tokens=2,
            ),
            signals=SignalsConfig(channels={}),
        ),
        device="cpu",
        num_workers=0,
    )

    datasets = _build_kaldi_datasets(
        records=[
            SleepRecord(
                record_id="sample_b",
                path=Path("/original/sample_b.npz"),
                split="test",
                source="unit",
                duration_sec=60,
                token_sec=30,
                max_tokens=2,
                metadata={"sample_key": "sample_b"},
            )
        ],
        channel_specs={"ppg": ChannelSpec(source="ppg", sfreq=100, kind="ppg", input_dim=2)},
        batch_size=1,
        num_workers=0,
        context=context,
    )

    assert [str(sample.id) for sample in datasets[0].data] == ["sample_b"]


def _fake_mne_module():
    class FakeRawArray:
        def __init__(self, data, info, verbose=False):
            self.data = data
            self.info = info

    return SimpleNamespace(
        create_info=lambda ch_names, sfreq, ch_types: {"ch_names": ch_names, "sfreq": sfreq, "ch_types": ch_types},
        io=SimpleNamespace(RawArray=FakeRawArray),
    )


def _yasa_context(tmp_path: Path, *, include_probabilities: bool = True) -> Sleep2statContext:
    return Sleep2statContext(
        config=SimpleNamespace(
            data=DataConfig(backend="npz", index=tmp_path / "index.csv", split=["test"]),
            signals=SignalsConfig(
                channels={
                    "eeg": ChannelSpec(
                        source="eeg",
                        sfreq=100,
                        kind="eeg",
                        input_dim=3000,
                        scale=1.0,
                        mne_name="EEG",
                    )
                }
            ),
            outputs=SimpleNamespace(include_probabilities=include_probabilities),
        ),
        device="cpu",
        num_workers=0,
    )


def _yasa_record(path: Path) -> SleepRecord:
    return SleepRecord(
        record_id="rec1",
        path=path,
        split="test",
        source="unit",
        duration_sec=60,
        token_sec=30,
        max_tokens=2,
        metadata={"age": 60, "sex": 1},
    )


def _kaldi_row(sample_key: str) -> dict:
    return {
        "sample_key": sample_key,
        "path": f"/original/{sample_key}.npz",
        "split": "test",
        "token_start": 0,
        "token_end": 2,
        "num_tokens": 2,
        "available_channels": json.dumps(["ppg"]),
    }


def _write_tiny_kaldi_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "kaldi"
    (root / "manifests").mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "splits": {
                    "test": {
                        "manifest": "manifests/test.csv",
                        "channels": {"ppg": {"input_dim": 2, "scp": "channels/test/ppg.scp"}},
                    }
                },
            }
        )
    )
    pd.DataFrame([_kaldi_row("sample_a"), _kaldi_row("sample_b")]).to_csv(root / "manifests" / "test.csv", index=False)
    return root
