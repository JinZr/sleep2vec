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
from sleep2stat.analyzers.spo2 import EventRelatedHypoxicBurdenAnalyzer, Spo2DesaturationAnalyzer, Spo2SummaryAnalyzer
from sleep2stat.analyzers.yasa import (
    YasaBandpowerAnalyzer,
    YasaHrvStageAnalyzer,
    YasaRemAnalyzer,
    YasaSpindlesAnalyzer,
    YasaStageAnalyzer,
    _sex_to_male,
)
from sleep2stat.config import AnalyzerConfig, ChannelSpec, DataConfig, SignalsConfig
from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.core.stage_sources import StageSourceResolver
from sleep2stat.io.records import SleepRecord, load_records, records_to_frame


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


def test_yasa_sex_to_male_accepts_numeric_strings():
    assert _sex_to_male("1.0") is True
    assert _sex_to_male("0.0") is False
    assert _sex_to_male("unknown") is None


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
    assert results[0].night["ahi_model_pred_event_rate_per_model_hour"] == 40.0
    assert results[0].night["ahi_model_pred_event_rate_per_recording_hour"] == pytest.approx(6.0)
    assert results[0].night["ahi_model_model_denominator_hours"] == pytest.approx(0.025)
    assert results[0].night["ahi_model_recording_denominator_hours"] == pytest.approx(600 / 3600)
    assert results[0].night["ahi_model_covered_duration_sec"] == 90
    assert results[0].night["ahi_model_coverage_ratio_recording"] == pytest.approx(0.15)
    assert results[0].night["ahi_model_truncated_by_max_tokens"] is False
    assert "ahi_model_pred_ahi" not in results[0].night


def test_decode_ahi_uses_strict_threshold_boundary():
    record = _record()
    logits = torch.zeros(1, 1, 30)
    results = decode_ahi_logits("ahi_model", logits, _batch(), {"rec1.npz": record}, threshold=0.5)

    assert results[0].second["ahi_model_pred"].sum() == 0
    assert len(results[0].events) == 0


def test_decode_ahi_postprocess_controls_outputs_and_stage_denominators():
    record = _record()
    prob = np.array([0.1] * 30 + [0.9] * 12 + [0.1] * 48, dtype=np.float32)
    logits = torch.from_numpy(np.log(prob / (1.0 - prob))).reshape(1, 3, 30)
    stage_epoch = pd.DataFrame({"record_id": ["rec1"] * 3, "token_idx": [0, 1, 2], "stage5_model_pred": [0, 2, 4]})
    resolver = StageSourceResolver([record], [AnalyzerResult("stage5_model", "rec1", epoch=stage_epoch)])

    results = decode_ahi_logits(
        "ahi_model",
        logits,
        _batch(),
        {"rec1.npz": record},
        threshold=0.5,
        threshold_source="postprocess",
        min_event_duration_sec=10,
        merge_tolerance_sec=3,
        denominator_stage_source="stage5_model",
        output_second_alignment=False,
        output_event_alignment=True,
        stage_resolver=resolver,
    )

    assert results[0].second is None
    assert len(results[0].events) == 1
    assert results[0].night["ahi_model_threshold_source"] == "postprocess"
    assert results[0].night["ahi_model_stage_assignment"] == "onset"
    assert results[0].events["ahi_model_stage_at_onset"].tolist() == [2]
    assert results[0].night["ahi_model_pred_AHI_sleep_denominator"] == pytest.approx(60.0)
    assert results[0].night["ahi_model_pred_REM_AHI_onset_stage"] == pytest.approx(0.0)
    assert results[0].night["ahi_model_pred_NREM_AHI_onset_stage"] == pytest.approx(120.0)
    assert results[0].night["ahi_model_pred_ahi"] == pytest.approx(60.0)


def test_decode_ahi_fails_when_configured_stage_denominator_is_missing():
    record = _record()
    logits = torch.zeros(1, 3, 30)
    resolver = StageSourceResolver([record], [])

    with pytest.raises(ValueError, match="denominator stage source 'stage5_model' not found"):
        decode_ahi_logits(
            "ahi_model",
            logits,
            _batch(),
            {"rec1.npz": record},
            threshold=0.5,
            denominator_stage_source="stage5_model",
            stage_resolver=resolver,
        )


def test_decode_ahi_reports_max_token_truncation():
    record = SleepRecord(
        record_id="rec1",
        path="rec1.npz",
        split="test",
        source="unit",
        duration_sec=600,
        token_sec=30,
        max_tokens=3,
        metadata={},
    )
    logits = torch.zeros(1, 3, 30)

    results = decode_ahi_logits("ahi_model", logits, _batch(), {"rec1.npz": record}, threshold=0.5)

    assert results[0].night["ahi_model_covered_duration_sec"] == 90
    assert results[0].night["ahi_model_coverage_ratio_recording"] == pytest.approx(0.15)
    assert results[0].night["ahi_model_truncated_by_max_tokens"] is True


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
            signals=SignalsConfig(
                channels={"ppg": ChannelSpec(source="ppg", sfreq=1, kind="ppg", input_dim=2, scale=2.0)}
            ),
            outputs=SimpleNamespace(include_probabilities=True, include_raw_logits=False),
        ),
        device="cpu",
        num_workers=0,
    )

    analyzer.prepare(context)
    results = analyzer.run(
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
            signals=SignalsConfig(channels={"ppg": ChannelSpec(source="ppg", sfreq=1, kind="ppg", input_dim=2)}),
            outputs=SimpleNamespace(include_probabilities=True, include_raw_logits=False),
        ),
        device="cpu",
        num_workers=0,
    )

    analyzer.prepare(context)
    results = analyzer.run(
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
    assert captured_ids["id"] == ["rec_a", "rec_b"]
    assert [result.record_id for result in results] == ["rec_a", "rec_b"]


def test_downstream_analyzer_batch_failure_raises(monkeypatch, tmp_path: Path):
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
            signals=SignalsConfig(channels={"ppg": ChannelSpec(source="ppg", sfreq=1, kind="ppg", input_dim=2)}),
            outputs=SimpleNamespace(include_probabilities=True, include_raw_logits=False),
        ),
        device="cpu",
        num_workers=0,
    )

    analyzer.prepare(context)
    with pytest.raises(RuntimeError, match="batch failed"):
        analyzer.run(
            [
                SleepRecord("good", tmp_path / "good.npz", "test", "unit", 60, 30, 2, {}),
                SleepRecord("bad", tmp_path / "bad.npz", "test", "unit", 60, 30, 2, {}),
            ],
            context,
        )


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
            signals=SignalsConfig(channels={"ppg": ChannelSpec(source="ppg", sfreq=1, kind="ppg", input_dim=2)}),
            outputs=SimpleNamespace(include_probabilities=True, include_raw_logits=False),
        ),
        device="cpu",
        num_workers=0,
    )

    analyzer.prepare(context)
    results = analyzer.run(
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
            signals=SignalsConfig(channels={"ppg": ChannelSpec(source="ppg", sfreq=1, kind="ppg", input_dim=2)}),
            outputs=SimpleNamespace(include_probabilities=True, include_raw_logits=False),
        ),
        device="cpu",
        num_workers=0,
    )

    analyzer.prepare(context)
    with pytest.raises(ValueError, match="Records were dropped before model inference"):
        analyzer.run(
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


def test_yasa_stage_analyzer_with_mock_sleep_staging(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, eeg=np.ones(6000, dtype=np.float32))

    class FakeStaging:
        def __init__(self, raw, eeg_name=None, eog_name=None, emg_name=None, metadata=None):
            assert eeg_name == "EEG"
            assert metadata == {"age": 60.0, "male": True}

        def predict(self):
            return _FakeHypnogram(["W", "N2"], {"W": [0.9, 0.1], "N2": [0.1, 0.9]})

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: _fake_mne_module() if name == "mne" else SimpleNamespace(SleepStaging=FakeStaging),
    )
    analyzer = YasaStageAnalyzer(AnalyzerConfig(name="yasa_stage", type="yasa_stage", input_channels=["eeg"]))
    context = _yasa_context(tmp_path)

    analyzer.prepare(context)
    results = analyzer.run([_yasa_record(npz_path)], context)
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
            return _FakeHypnogram(["W", "N2"], {"W": [0.9, 0.1], "N2": [0.1, 0.9]})

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: _fake_mne_module() if name == "mne" else SimpleNamespace(SleepStaging=FakeStaging),
    )
    analyzer = YasaStageAnalyzer(AnalyzerConfig(name="yasa_stage", type="yasa_stage", input_channels=["eeg"]))
    context = _yasa_context(tmp_path, include_probabilities=False)

    analyzer.prepare(context)
    results = analyzer.run([_yasa_record(npz_path)], context)
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
            return _FakeHypnogram(["W", "N2"])

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: _fake_mne_module() if name == "mne" else SimpleNamespace(SleepStaging=FakeStaging),
    )
    analyzer = YasaStageAnalyzer(AnalyzerConfig(name="yasa_stage", type="yasa_stage", input_channels=["eeg"]))
    context = _yasa_context(tmp_path)
    record = _yasa_record(npz_path)
    record.metadata["sex"] = -1

    analyzer.prepare(context)
    analyzer.run([record], context)
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
            outputs={
                "by_epoch": True,
                "by_stage": False,
                "by_night": True,
                "relative": True,
                "bands": ["delta", "alpha"],
            },
        )
    )
    context = _yasa_context(tmp_path)

    analyzer.prepare(context)
    results = analyzer.run([_yasa_record(npz_path)], context)
    assert results[0].epoch["yasa_bandpower_delta_rel"].tolist() == [0.4, 0.4]
    assert results[0].night["yasa_bandpower_delta_rel_mean"] == 0.4
    assert results[0].night["yasa_bandpower_alpha_rel_mean"] == 0.2


def test_yasa_bandpower_array_api_receives_uv(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, eeg=np.ones(6000, dtype=np.float32))
    captured = {}

    def fake_bandpower(data, sf=None, ch_names=None, bands=None, relative=True):
        captured["mean"] = float(np.mean(data))
        captured["relative"] = relative
        return pd.DataFrame({"Chan": ["EEG"], "Delta": [captured["mean"]]})

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: _fake_mne_module() if name == "mne" else SimpleNamespace(bandpower=fake_bandpower),
    )
    analyzer = YasaBandpowerAnalyzer(
        AnalyzerConfig(
            name="yasa_bandpower",
            type="yasa_bandpower",
            input_channels=["eeg"],
            outputs={
                "by_epoch": True,
                "by_stage": False,
                "by_night": True,
                "relative": False,
                "bands": ["delta"],
            },
        )
    )
    context = _yasa_context(tmp_path)
    context.config.signals.channels["eeg"] = ChannelSpec(
        source="eeg",
        sfreq=100,
        kind="eeg",
        input_dim=3000,
        scale=0.000001,
        mne_name="EEG",
    )

    analyzer.prepare(context)
    results = analyzer.run([_yasa_record(npz_path)], context)
    assert captured["mean"] == pytest.approx(1.0)
    assert captured["relative"] is False
    assert results[0].epoch["yasa_bandpower_delta_abs"].tolist() == [1.0, 1.0]


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
            stage_source="yasa_stage",
            outputs={"by_epoch": True, "by_stage": True, "by_night": True, "relative": True, "bands": ["sigma"]},
        )
    )
    context = _yasa_context(tmp_path)
    stage_epoch = pd.DataFrame({"record_id": ["rec1", "rec1"], "token_idx": [0, 1], "yasa_stage_pred": [2, 3]})

    analyzer.prepare(context)
    results = analyzer.run(
        [_yasa_record(npz_path)],
        context,
        prior_results=[AnalyzerResult("yasa_stage", "rec1", epoch=stage_epoch)],
    )
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
        AnalyzerConfig(
            name="yasa_bandpower",
            type="yasa_bandpower",
            input_channels=["eeg"],
            outputs={"by_epoch": True, "by_stage": True, "by_night": True, "relative": True},
        )
    )
    context = _yasa_context(tmp_path)

    analyzer.prepare(context)
    with pytest.raises(ValueError, match="stage_source"):
        analyzer.run([_yasa_record(npz_path)], context)


def test_yasa_spindles_analyzer_with_mock_detector(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, eeg=np.ones(6000, dtype=np.float32))

    def fake_spindles(raw):
        return pd.DataFrame({"Start": [10.0], "Duration": [1.5], "Confidence": [0.8], "Frequency": [13.0]})

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: _fake_mne_module() if name == "mne" else SimpleNamespace(spindles_detect=fake_spindles),
    )
    analyzer = YasaSpindlesAnalyzer(AnalyzerConfig(name="yasa_spindles", type="yasa_spindles", input_channels=["eeg"]))
    context = _yasa_context(tmp_path)

    analyzer.prepare(context)
    results = analyzer.run([_yasa_record(npz_path)], context)
    assert results[0].events["event_type"].tolist() == ["yasa_spindle"]
    assert results[0].events["yasa_spindles_confidence"].tolist() == [0.8]
    assert results[0].night["yasa_spindles_event_count"] == 1


def test_yasa_spindles_stage_filter_uses_sample_level_hypno(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, eeg=np.ones(6000, dtype=np.float32))
    captured = {}

    def fake_spindles(raw, hypno=None):
        captured["raw_samples"] = raw.data.shape[1]
        captured["hypno"] = np.asarray(hypno)
        return pd.DataFrame()

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: _fake_mne_module() if name == "mne" else SimpleNamespace(spindles_detect=fake_spindles),
    )
    analyzer = YasaSpindlesAnalyzer(
        AnalyzerConfig(
            name="yasa_spindles",
            type="yasa_spindles",
            input_channels=["eeg"],
            stage_source="stage5_model",
            stages=["N2"],
        )
    )
    context = _yasa_context(tmp_path)
    stage_epoch = pd.DataFrame({"record_id": ["rec1", "rec1"], "token_idx": [0, 1], "stage5_model_pred": [2, 4]})

    analyzer.prepare(context)
    results = analyzer.run(
        [_yasa_record(npz_path)],
        context,
        prior_results=[AnalyzerResult("stage5_model", "rec1", epoch=stage_epoch)],
    )
    assert len(results) == 1
    assert captured["hypno"].shape == (captured["raw_samples"],)
    assert captured["raw_samples"] == 6000
    assert np.all(captured["hypno"][:3000] == 2)
    assert np.all(captured["hypno"][3000:] == 0)


def test_yasa_spindles_outputs_stage_denominator_density(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, eeg=np.ones(6000, dtype=np.float32))

    def fake_spindles(raw, hypno=None):
        return pd.DataFrame(
            {
                "Start": [10.0, 40.0],
                "Duration": [1.0, 1.0],
                "Stage": [2, 3],
            }
        )

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: _fake_mne_module() if name == "mne" else SimpleNamespace(spindles_detect=fake_spindles),
    )
    analyzer = YasaSpindlesAnalyzer(
        AnalyzerConfig(
            name="yasa_spindles",
            type="yasa_spindles",
            input_channels=["eeg"],
            stage_source="stage5_model",
            stages=["N2", "N3"],
        )
    )
    context = _yasa_context(tmp_path)
    stage_epoch = pd.DataFrame({"record_id": ["rec1", "rec1"], "token_idx": [0, 1], "stage5_model_pred": [2, 3]})

    analyzer.prepare(context)
    results = analyzer.run(
        [_yasa_record(npz_path)],
        context,
        prior_results=[AnalyzerResult("stage5_model", "rec1", epoch=stage_epoch)],
    )
    assert results[0].night["yasa_spindles_spindle_density_per_min_N2"] == pytest.approx(2.0)
    assert results[0].night["yasa_spindles_spindle_density_per_min_N2N3"] == pytest.approx(2.0)


def test_yasa_spindles_stage_density_assigns_missing_event_stage_from_onset(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, eeg=np.ones(6000, dtype=np.float32))

    def fake_spindles(raw, hypno=None):
        return pd.DataFrame({"Start": [10.0, 40.0], "Duration": [1.0, 1.0]})

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: _fake_mne_module() if name == "mne" else SimpleNamespace(spindles_detect=fake_spindles),
    )
    analyzer = YasaSpindlesAnalyzer(
        AnalyzerConfig(
            name="yasa_spindles",
            type="yasa_spindles",
            input_channels=["eeg"],
            stage_source="stage5_model",
            stages=["N2", "N3"],
        )
    )
    context = _yasa_context(tmp_path)
    stage_epoch = pd.DataFrame({"record_id": ["rec1", "rec1"], "token_idx": [0, 1], "stage5_model_pred": [2, 3]})

    analyzer.prepare(context)
    results = analyzer.run(
        [_yasa_record(npz_path)],
        context,
        prior_results=[AnalyzerResult("stage5_model", "rec1", epoch=stage_epoch)],
    )
    assert results[0].night["yasa_spindles_spindle_density_per_min_N2"] == pytest.approx(2.0)
    assert results[0].night["yasa_spindles_spindle_density_per_min_N2N3"] == pytest.approx(2.0)


def test_yasa_rem_calls_two_eog_array_api_with_sample_level_hypno(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(
        npz_path,
        eog_loc=np.ones(6000, dtype=np.float32),
        eog_roc=np.ones(6000, dtype=np.float32) * 2,
    )
    captured = {}

    def fake_rem_detect(loc, roc, sf, *, hypno=None):
        captured["loc"] = np.asarray(loc)
        captured["roc"] = np.asarray(roc)
        captured["sf"] = sf
        captured["hypno"] = np.asarray(hypno)
        return pd.DataFrame({"Start": [10.0], "Duration": [1.0]})

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: _fake_mne_module() if name == "mne" else SimpleNamespace(rem_detect=fake_rem_detect),
    )
    analyzer = YasaRemAnalyzer(
        AnalyzerConfig(
            name="yasa_rem",
            type="yasa_rem",
            input_channels=["eog_loc", "eog_roc"],
            stage_source="stage5_model",
            stages=["REM"],
        )
    )
    context = _yasa_context(tmp_path)
    context.config.signals.channels["eog_loc"] = ChannelSpec(
        source="eog_loc",
        sfreq=100,
        kind="eog",
        input_dim=3000,
        scale=0.000001,
        mne_name="LOC",
    )
    context.config.signals.channels["eog_roc"] = ChannelSpec(
        source="eog_roc",
        sfreq=100,
        kind="eog",
        input_dim=3000,
        scale=0.000001,
        mne_name="ROC",
    )
    stage_epoch = pd.DataFrame({"record_id": ["rec1", "rec1"], "token_idx": [0, 1], "stage5_model_pred": [4, 2]})

    analyzer.prepare(context)
    results = analyzer.run(
        [_yasa_record(npz_path)],
        context,
        prior_results=[AnalyzerResult("stage5_model", "rec1", epoch=stage_epoch)],
    )
    assert captured["loc"].shape == (6000,)
    assert captured["roc"].shape == (6000,)
    assert captured["loc"].mean() == pytest.approx(1.0)
    assert captured["roc"].mean() == pytest.approx(2.0)
    assert captured["sf"] == 100
    assert captured["hypno"].shape == (6000,)
    assert np.all(captured["hypno"][:3000] == 4)
    assert np.all(captured["hypno"][3000:] == 0)
    assert results[0].events["event_type"].tolist() == ["yasa_rem"]


def test_yasa_hrv_stage_uses_single_channel_sample_level_hypno(monkeypatch, tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, ecg=np.ones(6000, dtype=np.float32))
    captured = {}

    def fake_hrv_stage(data, sf, *, hypno=None):
        captured["data"] = np.asarray(data)
        captured["sf"] = sf
        captured["hypno"] = np.asarray(hypno)
        epochs = pd.DataFrame(
            {
                "start": [0.0, 120.0, 240.0],
                "duration": [120.0, 120.0, 120.0],
                "hr_mean": [60.0, 70.0, 80.0],
                "hr_std": [2.0, 4.0, 5.0],
                "hrv_rmssd": [40.0, 60.0, 90.0],
            },
            index=pd.MultiIndex.from_tuples([(2, 0), (2, 1), (4, 0)], names=["values", "epoch"]),
        )
        return epochs, pd.DataFrame()

    monkeypatch.setattr(
        "sleep2stat.analyzers.yasa.importlib.import_module",
        lambda name: _fake_mne_module() if name == "mne" else SimpleNamespace(hrv_stage=fake_hrv_stage),
    )
    analyzer = YasaHrvStageAnalyzer(
        AnalyzerConfig(
            name="yasa_hrv_stage",
            type="yasa_hrv_stage",
            input_channels=["ecg"],
            stage_source="stage5_model",
        )
    )
    context = _yasa_context(tmp_path)
    context.config.signals.channels["ecg"] = ChannelSpec(source="ecg", sfreq=100, kind="ecg", input_dim=3000)
    stage_epoch = pd.DataFrame({"record_id": ["rec1", "rec1"], "token_idx": [0, 1], "stage5_model_pred": [2, 4]})

    analyzer.prepare(context)
    results = analyzer.run(
        [_yasa_record(npz_path)],
        context,
        prior_results=[AnalyzerResult("stage5_model", "rec1", epoch=stage_epoch)],
    )
    assert captured["data"].shape == (6000,)
    assert captured["sf"] == 100
    assert captured["hypno"].shape == (6000,)
    assert np.all(captured["hypno"][:3000] == 2)
    assert np.all(captured["hypno"][3000:] == 4)
    night = results[0].night
    assert night["yasa_hrv_stage_N2_hr_mean"] == pytest.approx(65.0)
    assert night["yasa_hrv_stage_N2_hr_std"] == pytest.approx(3.0)
    assert night["yasa_hrv_stage_N2_hrv_rmssd"] == pytest.approx(50.0)
    assert night["yasa_hrv_stage_REM_hr_mean"] == pytest.approx(80.0)
    assert not any(key.endswith(("_start", "_duration", "_epoch")) for key in night)


def test_spo2_summary_handles_artifact_and_threshold_time(tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, spo2=np.array([95, 89, 87, 101], dtype=np.float32))
    analyzer = Spo2SummaryAnalyzer(
        AnalyzerConfig(
            name="spo2_summary",
            type="spo2_summary",
            input_channels=["spo2"],
            artifact={"valid_min": 50, "valid_max": 100},
        )
    )

    results = analyzer.run([_spo2_record(npz_path)], _spo2_context(tmp_path))
    night = results[0].night
    assert night["spo2_nadir"] == 87.0
    assert night["spo2_t90_min"] == pytest.approx(2 / 60)
    assert night["spo2_t90_ratio_recording"] == pytest.approx(0.5)
    assert night["spo2_t90_pct_recording"] == pytest.approx(50.0)
    assert night["spo2_t88_min"] == pytest.approx(1 / 60)
    assert night["spo2_artifact_pct"] == pytest.approx(0.25)


def test_spo2_summary_excludes_large_jump_artifact(tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, spo2=np.array([95, 70, 95], dtype=np.float32))
    analyzer = Spo2SummaryAnalyzer(
        AnalyzerConfig(
            name="spo2_summary",
            type="spo2_summary",
            input_channels=["spo2"],
            artifact={"valid_min": 50, "valid_max": 100, "max_abs_change_per_sec": 8},
        )
    )

    results = analyzer.run([_spo2_record(npz_path)], _spo2_context(tmp_path))
    assert results[0].night["spo2_nadir"] == 95.0
    assert results[0].night["spo2_t90_min"] == 0.0


def test_spo2_desaturation_detects_events_and_odi(tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    signal = np.array([96] * 5 + [92] * 12 + [96] * 43, dtype=np.float32)
    np.savez(npz_path, spo2=signal)
    analyzer = Spo2DesaturationAnalyzer(
        AnalyzerConfig(
            name="spo2_desaturation",
            type="spo2_desaturation",
            input_channels=["spo2"],
            drop_thresholds=[3, 4],
            min_duration_sec=10,
        )
    )

    results = analyzer.run([_spo2_record(npz_path)], _spo2_context(tmp_path))
    assert sorted(results[0].events["drop_threshold_pct"].tolist()) == [3.0, 4.0]
    assert results[0].events["spo2_desat_area_pctsec"].tolist() == [48.0, 48.0]
    assert results[0].events["spo2_desat_area_pctmin"].tolist() == [0.8, 0.8]
    assert results[0].events["spo2_time_to_nadir_sec"].tolist() == [0.0, 0.0]
    assert results[0].events["spo2_recovery_duration_sec"].tolist() == [12.0, 12.0]
    assert results[0].night["ODI3_per_recording_hour"] == pytest.approx(60.0)
    assert results[0].night["ODI4_per_recording_hour"] == pytest.approx(60.0)
    assert results[0].night["ODI3_per_valid_spo2_hour"] == pytest.approx(60.0)
    assert results[0].night["ODI4_per_valid_spo2_hour"] == pytest.approx(60.0)
    assert "ODI3_recording" not in results[0].night
    assert "ODI4_recording" not in results[0].night


def test_spo2_desaturation_outputs_sleep_denominator_odi_when_stage_source_is_available(tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    signal = np.array([96] * 5 + [92] * 12 + [96] * 43, dtype=np.float32)
    np.savez(npz_path, spo2=signal)
    analyzer = Spo2DesaturationAnalyzer(
        AnalyzerConfig(
            name="spo2_desaturation",
            type="spo2_desaturation",
            input_channels=["spo2"],
            drop_thresholds=[3],
            min_duration_sec=10,
            stage_source="stage5_model",
        )
    )
    stage_epoch = pd.DataFrame({"record_id": ["rec1", "rec1"], "token_idx": [0, 1], "stage5_model_pred": [0, 2]})

    results = analyzer.run(
        [_spo2_record(npz_path)],
        _spo2_context(tmp_path),
        prior_results=[AnalyzerResult("stage5_model", "rec1", epoch=stage_epoch)],
    )
    assert results[0].night["ODI3_per_sleep_hour"] == pytest.approx(120.0)


def test_spo2_desaturation_fails_when_stage_denominator_missing(tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    signal = np.array([96] * 5 + [92] * 12 + [96] * 43, dtype=np.float32)
    np.savez(npz_path, spo2=signal)
    analyzer = Spo2DesaturationAnalyzer(
        AnalyzerConfig(
            name="spo2_desaturation",
            type="spo2_desaturation",
            input_channels=["spo2"],
            drop_thresholds=[3],
            min_duration_sec=10,
            stage_source="stage5_model",
        )
    )

    with pytest.raises(ValueError, match="stage_source"):
        analyzer.run([_spo2_record(npz_path)], _spo2_context(tmp_path), prior_results=[])


def test_event_related_hypoxic_burden_uses_pred_event_fields(tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    signal = np.array([96] * 20 + [90] * 20 + [95] * 20, dtype=np.float32)
    np.savez(npz_path, spo2=signal)
    source_events = pd.DataFrame({"record_id": ["rec1"], "onset_sec": [20.0], "offset_sec": [40.0]})
    analyzer = EventRelatedHypoxicBurdenAnalyzer(
        AnalyzerConfig(
            name="pred_hypoxic_burden",
            type="event_related_hypoxic_burden",
            input_channels=["spo2"],
            event_source="ahi_model",
        )
    )

    results = analyzer.run(
        [_spo2_record(npz_path)],
        _spo2_context(tmp_path),
        prior_results=[AnalyzerResult("ahi_model", "rec1", events=source_events)],
    )
    assert "resp_event_hypoxic_burden_pctmin" in results[0].events.columns
    assert results[0].night["resp_event_hypoxic_burden_event_count"] == 1
    assert results[0].night["resp_event_hypoxic_burden_pctmin"] == pytest.approx(140 / 60)
    assert not any(column.startswith("clinical") for column in results[0].night)


def test_event_related_hypoxic_burden_deduplicates_multi_threshold_events(tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    signal = np.array([96] * 20 + [90] * 20 + [95] * 20, dtype=np.float32)
    np.savez(npz_path, spo2=signal)
    source_events = pd.DataFrame(
        {
            "record_id": ["rec1", "rec1"],
            "onset_sec": [20.0, 20.0],
            "offset_sec": [40.0, 40.0],
            "drop_threshold_pct": [3.0, 4.0],
        }
    )
    analyzer = EventRelatedHypoxicBurdenAnalyzer(
        AnalyzerConfig(
            name="pred_hypoxic_burden",
            type="event_related_hypoxic_burden",
            input_channels=["spo2"],
            event_source="spo2_desaturation",
        )
    )

    results = analyzer.run(
        [_spo2_record(npz_path)],
        _spo2_context(tmp_path),
        prior_results=[AnalyzerResult("spo2_desaturation", "rec1", events=source_events)],
    )
    assert len(results[0].events) == 1
    assert "desaturation_area_burden_pctmin" in results[0].events.columns
    assert results[0].night["desaturation_area_burden_event_count"] == 1
    assert results[0].night["desaturation_area_burden_pctmin"] == pytest.approx(140 / 60)


def test_event_related_hypoxic_burden_fails_when_source_result_is_missing(tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, spo2=np.ones(60, dtype=np.float32) * 96)
    analyzer = EventRelatedHypoxicBurdenAnalyzer(
        AnalyzerConfig(
            name="pred_hypoxic_burden",
            type="event_related_hypoxic_burden",
            input_channels=["spo2"],
            event_source="ahi_model",
        )
    )

    with pytest.raises(ValueError, match="event_source"):
        analyzer.run([_spo2_record(npz_path)], _spo2_context(tmp_path), prior_results=[])


def test_event_related_hypoxic_burden_allows_real_empty_source_events(tmp_path: Path):
    npz_path = tmp_path / "rec1.npz"
    np.savez(npz_path, spo2=np.ones(60, dtype=np.float32) * 96)
    analyzer = EventRelatedHypoxicBurdenAnalyzer(
        AnalyzerConfig(
            name="pred_hypoxic_burden",
            type="event_related_hypoxic_burden",
            input_channels=["spo2"],
            event_source="ahi_model",
        )
    )
    source_events = pd.DataFrame(columns=["record_id", "onset_sec", "offset_sec"])

    results = analyzer.run(
        [_spo2_record(npz_path)],
        _spo2_context(tmp_path),
        prior_results=[AnalyzerResult("ahi_model", "rec1", events=source_events)],
    )
    assert results[0].night["resp_event_hypoxic_burden_event_count"] == 0
    assert results[0].night["resp_event_hypoxic_burden_pctmin"] == 0.0
    assert results[0].night["resp_event_hypoxic_burden_pctmin_per_recording_hour"] == 0.0


def test_load_records_preserves_npz_paths_and_manifest_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.savez(data_dir / "rec1.npz", ppg=np.ones(2, dtype=np.float32))
    index = tmp_path / "index.csv"
    index.write_text(
        "path,duration,split,source,patient_id,age,note\n" "data/rec1.npz,60,test,unit,p001,55,do-not-export\n"
    )
    monkeypatch.chdir(tmp_path)

    records = load_records(
        DataConfig(
            backend="npz",
            index=index,
            split=["test"],
            path_column="path",
            duration_column="duration",
            split_column="split",
            token_sec=30,
            max_tokens=1535,
            record_id_columns=["source", "patient_id"],
            metadata_columns=["age"],
        )
    )
    manifest = records_to_frame(records, metadata_columns=["age"])

    assert records[0].raw_path == "data/rec1.npz"
    assert records[0].path == Path("data/rec1.npz")
    assert records[0].path_exists is True
    assert manifest.loc[0, "raw_path"] == "data/rec1.npz"
    assert manifest.loc[0, "resolved_path"] == "data/rec1.npz"
    assert bool(manifest.loc[0, "path_exists"]) is True
    assert manifest.loc[0, "age"] == 55
    assert "patient_id" not in manifest.columns
    assert "note" not in manifest.columns


def test_load_records_reads_kaldi_manifest(tmp_path: Path):
    root = _write_tiny_kaldi_manifest(tmp_path)

    records = load_records(
        DataConfig(
            backend="kaldi",
            index=None,
            split=["test"],
            path_column="path",
            duration_column="duration",
            split_column="split",
            token_sec=30,
            max_tokens=2,
            kaldi_data_root=root,
            kaldi_manifest=root / "manifest.json",
        )
    )

    assert [record.record_id for record in records] == ["sample_a", "sample_b"]
    assert records[0].duration_sec == 60
    assert records[0].metadata["sample_key"] == "sample_a"


@pytest.mark.parametrize("sample_key", ["site/sample_a", "site\\sample_a", ".", ".."])
def test_load_records_rejects_unsafe_kaldi_sample_key_record_ids(tmp_path: Path, sample_key: str):
    root = _write_tiny_kaldi_manifest(tmp_path)
    split_manifest = root / "manifests" / "test.csv"
    pd.DataFrame([_kaldi_row(sample_key)]).to_csv(split_manifest, index=False)

    with pytest.raises(ValueError, match="path-safe sleep2stat record_id"):
        load_records(
            DataConfig(
                backend="kaldi",
                index=None,
                split=["test"],
                path_column="path",
                duration_column="duration",
                split_column="split",
                token_sec=30,
                max_tokens=2,
                kaldi_data_root=root,
                kaldi_manifest=root / "manifest.json",
            )
        )


@pytest.mark.parametrize("patient_id", [".", ".."])
def test_load_records_rejects_unsafe_npz_record_id_columns(tmp_path: Path, patient_id: str):
    index = tmp_path / "index.csv"
    index.write_text(f"path,duration,split,patient_id\n/tmp/a.npz,60,test,{patient_id}\n")

    with pytest.raises(ValueError, match="path-safe sleep2stat record_id"):
        load_records(
            DataConfig(
                backend="npz",
                index=index,
                split=["test"],
                path_column="path",
                duration_column="duration",
                split_column="split",
                token_sec=30,
                max_tokens=1535,
                record_id_columns=["patient_id"],
            )
        )


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
                path_column="path",
                duration_column="duration",
                split_column="split",
                token_sec=30,
                max_tokens=1535,
                record_id_columns=["source", "patient_id"],
            )
        )


def test_load_records_fallback_ids_use_original_index_rows(tmp_path: Path):
    index = tmp_path / "index.csv"
    index.write_text("path,duration,split,source\n" "/tmp/train.npz,60,train,unit\n" "/tmp/test.npz,60,test,unit\n")

    all_records = load_records(
        DataConfig(
            backend="npz",
            index=index,
            split=["train", "test"],
            path_column="path",
            duration_column="duration",
            split_column="split",
            token_sec=30,
            max_tokens=1535,
        )
    )
    test_records = load_records(
        DataConfig(
            backend="npz",
            index=index,
            split=["test"],
            path_column="path",
            duration_column="duration",
            split_column="split",
            token_sec=30,
            max_tokens=1535,
        )
    )

    assert [record.record_id for record in all_records] == ["train__row0", "test__row1"]
    assert [record.record_id for record in test_records] == ["test__row1"]


def test_kaldi_dataset_routing_filters_to_pending_sample_keys(tmp_path: Path):
    root = _write_tiny_kaldi_manifest(tmp_path)
    split_manifest = root / "manifests" / "test.csv"
    rows = pd.read_csv(split_manifest)
    bad_row = _kaldi_row("sample_bad")
    bad_row["token_end"] = 5
    bad_row["num_tokens"] = 5
    pd.concat([rows, pd.DataFrame([bad_row])], ignore_index=True).to_csv(split_manifest, index=False)
    context = Sleep2statContext(
        config=SimpleNamespace(
            data=DataConfig(
                backend="kaldi",
                index=None,
                split=["test"],
                path_column="path",
                duration_column="duration",
                split_column="split",
                token_sec=30,
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
        channel_specs={"ppg": ChannelSpec(source="ppg", sfreq=100, kind="ppg", input_dim=3000)},
        batch_size=1,
        num_workers=0,
        context=context,
    )

    assert datasets[0].channel_input_dims["ppg"] == 2
    assert [str(sample.id) for sample in datasets[0].data] == ["sample_b"]
    assert "sample_bad" in pd.read_csv(split_manifest)["sample_key"].tolist()


def test_sleep2stat_kaldi_downstream_rejects_embedding_export_manifest(tmp_path: Path):
    root = _write_tiny_kaldi_manifest(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["embedding_kind"] = "token"
    manifest_path.write_text(json.dumps(manifest))
    context = Sleep2statContext(
        config=SimpleNamespace(
            data=DataConfig(
                backend="kaldi",
                index=None,
                split=["test"],
                path_column="path",
                duration_column="duration",
                split_column="split",
                token_sec=30,
                kaldi_data_root=root,
                kaldi_manifest=manifest_path,
                max_tokens=2,
            ),
            signals=SignalsConfig(channels={}),
        ),
        device="cpu",
        num_workers=0,
    )

    with pytest.raises(ValueError, match="already-tokenized backbone embeddings"):
        _build_kaldi_datasets(
            records=[
                SleepRecord(
                    record_id="sample_a",
                    path=Path("/original/sample_a.npz"),
                    split="test",
                    source="unit",
                    duration_sec=60,
                    token_sec=30,
                    max_tokens=2,
                    metadata={"sample_key": "sample_a"},
                )
            ],
            channel_specs={"ppg": ChannelSpec(source="ppg", sfreq=100, kind="ppg", input_dim=3000)},
            batch_size=1,
            num_workers=0,
            context=context,
        )


class _FakeHypnogram:
    def __init__(self, labels, proba=None):
        self._labels = pd.Series(labels)
        self.proba = None if proba is None else pd.DataFrame(proba)

    def as_int(self):
        mapping = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "REM": 4}
        return self._labels.map(lambda label: mapping.get(str(label).strip().upper(), -1)).astype(np.int16)


def _fake_mne_module():
    class FakeRawArray:
        def __init__(self, data, info, verbose=False):
            self.data = data
            self.info = info

        def get_data(self, units=None):
            if units is None:
                return self.data
            return self.data * 1_000_000

    return SimpleNamespace(
        create_info=lambda ch_names, sfreq, ch_types: {"ch_names": ch_names, "sfreq": sfreq, "ch_types": ch_types},
        io=SimpleNamespace(RawArray=FakeRawArray),
    )


def _yasa_context(tmp_path: Path, *, include_probabilities: bool = True) -> Sleep2statContext:
    return Sleep2statContext(
        config=SimpleNamespace(
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


def _spo2_context(tmp_path: Path) -> Sleep2statContext:
    return Sleep2statContext(
        config=SimpleNamespace(
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
            signals=SignalsConfig(
                channels={
                    "spo2": ChannelSpec(source="spo2", sfreq=1, kind="spo2", input_dim=30),
                }
            ),
            outputs=SimpleNamespace(include_probabilities=True),
        ),
        device="cpu",
        num_workers=0,
    )


def _spo2_record(path: Path) -> SleepRecord:
    return SleepRecord(
        record_id="rec1",
        path=path,
        split="test",
        source="unit",
        duration_sec=60,
        token_sec=30,
        max_tokens=2,
        metadata={},
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
