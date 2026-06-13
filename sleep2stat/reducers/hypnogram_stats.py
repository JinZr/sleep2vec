from __future__ import annotations

import numpy as np

from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import SleepRecord
from sleep2stat.reducers.base import BaseReducer
from sleep2stat.registry import register_reducer


@register_reducer("yasa_hypnogram_stats")
@register_reducer("hypnogram_stats")
class HypnogramStatsReducer(BaseReducer):
    def reduce(
        self,
        records: list[SleepRecord],
        results: list[AnalyzerResult],
        context: Sleep2statContext,
    ) -> list[AnalyzerResult]:
        source = self.config.source
        output = []
        for result in results:
            if result.name != source or result.epoch is None or result.epoch.empty:
                continue
            pred_col = f"{source}_pred"
            if pred_col not in result.epoch.columns:
                continue
            stages = result.epoch[pred_col].to_numpy()
            token_sec = _token_sec_for(records, result.record_id)
            output.append(
                AnalyzerResult(
                    self.config.name,
                    result.record_id,
                    night=_hypnogram_stats(stages, token_sec=token_sec, prefix=str(source)),
                )
            )
        return output


def _hypnogram_stats(stages, *, token_sec: int, prefix: str) -> dict[str, float]:
    raw_values = np.asarray(stages)
    # Clinically, TIB follows the full hypnogram/recording span; scored_TIB keeps the valid-stage denominator explicit.
    scored = np.isin(raw_values, [0, 1, 2, 3, 4])
    values = raw_values[scored].astype(np.int64)
    epoch_min = token_sec / 60.0
    recording_min = float(raw_values.size * epoch_min)
    unscored_count = int(raw_values.size - values.size)
    valid_stage_ratio = float(values.size / raw_values.size) if raw_values.size else np.nan
    if values.size == 0:
        return {
            f"{prefix}_recording_duration_min": recording_min,
            f"{prefix}_scored_TIB_min": 0.0,
            f"{prefix}_TIB_min": recording_min,
            f"{prefix}_TST_min": 0.0,
            f"{prefix}_unscored_epoch_count": unscored_count,
            f"{prefix}_valid_stage_epoch_ratio": valid_stage_ratio,
            f"{prefix}_SE_ratio": np.nan,
            f"{prefix}_SE_pct": np.nan,
            f"{prefix}_SE": np.nan,
        }

    sleep = np.isin(values, [1, 2, 3, 4])
    scored_tib_min = float(values.size * epoch_min)
    tst_min = float(sleep.sum() * epoch_min)
    if sleep.any():
        first_sleep = int(np.argmax(sleep))
        last_sleep = int(len(sleep) - 1 - np.argmax(sleep[::-1]))
        sleep_period = values[first_sleep : last_sleep + 1]
        # YASA/AASM-style WASO is wake within SPT; terminal wake and onset-to-end wake are separate outputs.
        waso_spt_min = float((sleep_period == 0).sum() * epoch_min)
        terminal_wake_min = float((values[last_sleep + 1 :] == 0).sum() * epoch_min)
        waso_to_end_min = float((values[first_sleep:] == 0).sum() * epoch_min)
        sol_min = float(first_sleep * epoch_min)
    else:
        first_sleep = None
        sleep_period = np.asarray([], dtype=np.int64)
        waso_spt_min = 0.0
        terminal_wake_min = 0.0
        waso_to_end_min = 0.0
        sol_min = np.nan

    rem_positions = np.where(values == 4)[0]
    rem_latency_min = np.nan
    if first_sleep is not None and rem_positions.size:
        rem_latency_min = float(max(0, int(rem_positions[0]) - first_sleep) * epoch_min)

    # Keep SE_ratio as 0-1 for modeling tables; SE_pct is the report-facing clinical percentage.
    se_ratio = float(tst_min / recording_min) if recording_min > 0 else np.nan
    stats = {
        f"{prefix}_recording_duration_min": recording_min,
        f"{prefix}_scored_TIB_min": scored_tib_min,
        f"{prefix}_TIB_min": recording_min,
        f"{prefix}_TST_min": tst_min,
        f"{prefix}_unscored_epoch_count": unscored_count,
        f"{prefix}_valid_stage_epoch_ratio": valid_stage_ratio,
        f"{prefix}_WASO_SPT_min": waso_spt_min,
        f"{prefix}_WASO_min": waso_spt_min,
        f"{prefix}_terminal_wake_after_last_sleep_min": terminal_wake_min,
        f"{prefix}_WASO_after_sleep_onset_to_recording_end_min": waso_to_end_min,
        f"{prefix}_SE_ratio": se_ratio,
        f"{prefix}_SE_pct": se_ratio * 100.0 if not np.isnan(se_ratio) else np.nan,
        f"{prefix}_SE": se_ratio,
        f"{prefix}_SOL_min": sol_min,
        f"{prefix}_REM_latency_min": rem_latency_min,
        f"{prefix}_stage_shift_rate_per_sleep_hour": _stage_shift_index(sleep_period, tst_min),
        f"{prefix}_stage_shift_index": _stage_shift_index(sleep_period, tst_min),
        f"{prefix}_sleep_to_wake_transition_index": _sleep_to_wake_transition_index(sleep_period, tst_min),
        f"{prefix}_SFI_yasa_like": _sleep_to_wake_transition_index(sleep_period, tst_min),
        f"{prefix}_sleep_bout_count": _sleep_bout_count(values),
        f"{prefix}_mean_sleep_bout_min": _mean_sleep_bout_min(values, epoch_min),
    }
    for stage_id, label in ((1, "N1"), (2, "N2"), (3, "N3"), (4, "REM")):
        minutes = float((values == stage_id).sum() * epoch_min)
        stats[f"{prefix}_{label}_min"] = minutes
        stats[f"{prefix}_pct_{label}"] = float(minutes / tst_min) if tst_min > 0 else np.nan
        stats[f"{prefix}_{label}_ratio_TST"] = float(minutes / tst_min) if tst_min > 0 else np.nan
        stats[f"{prefix}_{label}_pct_TST"] = float(minutes / tst_min * 100.0) if tst_min > 0 else np.nan
    return stats


def _stage_shift_index(sleep_period: np.ndarray, tst_min: float) -> float:
    if sleep_period.size < 2 or tst_min <= 0:
        return np.nan
    transitions = int(np.sum(sleep_period[1:] != sleep_period[:-1]))
    return float(transitions / (tst_min / 60.0))


def _sleep_to_wake_transition_index(sleep_period: np.ndarray, tst_min: float) -> float:
    if sleep_period.size < 2 or tst_min <= 0:
        return np.nan
    transitions = int(np.sum((sleep_period[:-1] != 0) & (sleep_period[1:] == 0)))
    return float(transitions / (tst_min / 60.0))


def _sleep_bout_count(values: np.ndarray) -> int:
    sleep = np.isin(values, [1, 2, 3, 4])
    starts = sleep & np.concatenate(([True], ~sleep[:-1]))
    return int(starts.sum())


def _mean_sleep_bout_min(values: np.ndarray, epoch_min: float) -> float:
    sleep = np.isin(values, [1, 2, 3, 4])
    lengths = []
    start = None
    for idx, is_sleep in enumerate(sleep.tolist() + [False]):
        if is_sleep and start is None:
            start = idx
        elif not is_sleep and start is not None:
            lengths.append(idx - start)
            start = None
    return float(np.mean(lengths) * epoch_min) if lengths else np.nan


def _token_sec_for(records: list[SleepRecord], record_id: str) -> int:
    for record in records:
        if record.record_id == record_id:
            return record.token_sec
    return 30
