import pandas as pd
import pytest

from sleep2stat.config import ReducerConfig
from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.io.records import SleepRecord
from sleep2stat.reducers.hypnogram_stats import HypnogramStatsReducer
from sleep2stat.reducers.stage_agreement import StageAgreementReducer
from sleep2stat.reducers.transition_stats import TransitionStatsReducer


def _record() -> SleepRecord:
    return SleepRecord(
        record_id="rec1",
        path="rec1.npz",
        split="test",
        source="unit",
        duration_sec=210,
        token_sec=30,
        max_tokens=7,
        metadata={},
    )


def test_hypnogram_stats_from_epoch_predictions():
    frame = pd.DataFrame(
        {
            "record_id": ["rec1"] * 7,
            "path": ["rec1.npz"] * 7,
            "token_idx": list(range(7)),
            "stage5_model_pred": [0, 1, 2, 2, 3, 4, 0],
        }
    )
    reducer = HypnogramStatsReducer(ReducerConfig(name="stage5_stats", type="hypnogram_stats", source="stage5_model"))

    output = reducer.reduce([_record()], [AnalyzerResult("stage5_model", "rec1", epoch=frame)], None)

    stats = output[0].night
    assert stats["stage5_model_TIB_min"] == 3.5
    assert stats["stage5_model_TST_min"] == 2.5
    assert stats["stage5_model_SOL_min"] == 0.5
    assert stats["stage5_model_REM_latency_min"] == 2.0
    assert stats["stage5_model_pct_N2"] == pytest.approx(0.4)


def test_stage_agreement_reducer_outputs_accuracy_and_kappa():
    left = pd.DataFrame(
        {
            "record_id": ["rec1"] * 4,
            "path": ["rec1.npz"] * 4,
            "token_idx": [0, 1, 2, 3],
            "stage5_model_pred": [0, 1, 2, 4],
        }
    )
    right = pd.DataFrame(
        {
            "record_id": ["rec1"] * 4,
            "path": ["rec1.npz"] * 4,
            "token_idx": [0, 1, 2, 3],
            "reference_stage5_pred": [0, 2, 2, 4],
        }
    )
    reducer = StageAgreementReducer(
        ReducerConfig(
            name="stage_model_reference",
            type="stage_agreement",
            left="stage5_model",
            right="reference_stage5",
        )
    )

    output = reducer.reduce(
        [_record()],
        [
            AnalyzerResult("stage5_model", "rec1", epoch=left),
            AnalyzerResult("reference_stage5", "rec1", epoch=right),
        ],
        None,
    )

    assert output[0].night["stage_model_reference_accuracy"] == 0.75
    assert "stage_model_reference_kappa" in output[0].night
    assert output[0].night["stage_model_reference_overlap_coverage"] == 1.0


def test_stage_agreement_reports_partial_overlap_coverage():
    left = pd.DataFrame(
        {
            "record_id": ["rec1"] * 4,
            "path": ["rec1.npz"] * 4,
            "token_idx": [0, 1, 2, 3],
            "stage5_model_pred": [0, 1, 2, 4],
        }
    )
    right = pd.DataFrame(
        {
            "record_id": ["rec1"] * 2,
            "path": ["rec1.npz"] * 2,
            "token_idx": [0, 1],
            "reference_stage5_pred": [0, 1],
        }
    )
    reducer = StageAgreementReducer(
        ReducerConfig(
            name="stage_model_reference",
            type="stage_agreement",
            left="stage5_model",
            right="reference_stage5",
        )
    )

    output = reducer.reduce(
        [_record()],
        [
            AnalyzerResult("stage5_model", "rec1", epoch=left),
            AnalyzerResult("reference_stage5", "rec1", epoch=right),
        ],
        None,
    )

    assert output[0].night["stage_model_reference_overlap_epoch_count"] == 2
    assert output[0].night["stage_model_reference_overlap_coverage"] == 0.5


def test_transition_stats_entropy_uses_transition_counts_only():
    frame = pd.DataFrame(
        {
            "record_id": ["rec1"] * 4,
            "path": ["rec1.npz"] * 4,
            "token_idx": [0, 1, 2, 3],
            "stage5_model_pred": [0, 1, 2, 3],
        }
    )
    reducer = TransitionStatsReducer(
        ReducerConfig(name="stage5_transition", type="transition_stats", source="stage5_model")
    )

    output = reducer.reduce([_record()], [AnalyzerResult("stage5_model", "rec1", epoch=frame)], None)

    assert output[0].night["stage5_model_stage_shift_index"] == 1.0
    assert output[0].night["stage5_model_transition_entropy"] == pytest.approx(1.0986122886681096)
