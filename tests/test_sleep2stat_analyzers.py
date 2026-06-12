from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sleep2stat.analyzers.model_downstream import (
    Sleep2vecDownstreamAnalyzer,
    decode_ahi_logits,
    decode_classification_logits,
    decode_regression_logits,
)
from sleep2stat.config import AnalyzerConfig, ChannelSpec, SignalsConfig
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import SleepRecord


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

    class FakeEvalModel:
        def __call__(self, batch):
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
                metadata={"sex": 1},
            )
        ],
        context,
    )

    assert calls == {"on_load_checkpoint": 1, "load_state_dict": 1, "eval_model": 1}
    assert failures == []
    assert results[0].night["sex_model_pred"] == 1


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
