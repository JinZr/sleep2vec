from __future__ import annotations

import numpy as np

from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import SleepRecord
from sleep2stat.reducers.base import BaseReducer
from sleep2stat.registry import register_reducer


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
    values = np.asarray(stages)
    valid = values >= 0
    values = values[valid].astype(np.int64)
    epoch_min = token_sec / 60.0
    if values.size == 0:
        return {f"{prefix}_TIB_min": 0.0, f"{prefix}_TST_min": 0.0, f"{prefix}_SE": np.nan}

    sleep = values != 0
    tib_min = float(values.size * epoch_min)
    tst_min = float(sleep.sum() * epoch_min)
    if sleep.any():
        first_sleep = int(np.argmax(sleep))
        last_sleep = int(len(sleep) - 1 - np.argmax(sleep[::-1]))
        sleep_period = values[first_sleep : last_sleep + 1]
        waso_min = float((sleep_period == 0).sum() * epoch_min)
        sol_min = float(first_sleep * epoch_min)
    else:
        first_sleep = None
        sleep_period = np.asarray([], dtype=np.int64)
        waso_min = 0.0
        sol_min = np.nan

    rem_positions = np.where(values == 4)[0]
    rem_latency_min = np.nan
    if first_sleep is not None and rem_positions.size:
        rem_latency_min = float(max(0, int(rem_positions[0]) - first_sleep) * epoch_min)

    stats = {
        f"{prefix}_TIB_min": tib_min,
        f"{prefix}_TST_min": tst_min,
        f"{prefix}_WASO_min": waso_min,
        f"{prefix}_SE": float(tst_min / tib_min) if tib_min > 0 else np.nan,
        f"{prefix}_SOL_min": sol_min,
        f"{prefix}_REM_latency_min": rem_latency_min,
        f"{prefix}_SFI": _stage_fragmentation_index(sleep_period, tst_min),
    }
    for stage_id, label in ((1, "N1"), (2, "N2"), (3, "N3"), (4, "REM")):
        minutes = float((values == stage_id).sum() * epoch_min)
        stats[f"{prefix}_{label}_min"] = minutes
        stats[f"{prefix}_pct_{label}"] = float(minutes / tst_min) if tst_min > 0 else np.nan
    return stats


def _stage_fragmentation_index(sleep_period: np.ndarray, tst_min: float) -> float:
    if sleep_period.size < 2 or tst_min <= 0:
        return np.nan
    transitions = int(np.sum(sleep_period[1:] != sleep_period[:-1]))
    return float(transitions / (tst_min / 60.0))


def _token_sec_for(records: list[SleepRecord], record_id: str) -> int:
    for record in records:
        if record.record_id == record_id:
            return record.token_sec
    return 30
