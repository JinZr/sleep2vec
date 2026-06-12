from __future__ import annotations

import numpy as np

from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import SleepRecord
from sleep2stat.reducers.base import BaseReducer
from sleep2stat.registry import register_reducer

STAGE_LABELS = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


@register_reducer("transition_stats")
class TransitionStatsReducer(BaseReducer):
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
            output.append(
                AnalyzerResult(
                    self.config.name,
                    result.record_id,
                    night=_transition_stats(result.epoch[pred_col].to_numpy(), prefix=str(source)),
                )
            )
        return output


def _transition_stats(stages, *, prefix: str) -> dict[str, float]:
    values = np.asarray(stages)
    values = values[values >= 0].astype(np.int64)
    stats: dict[str, float] = {}
    if values.size < 2:
        stats[f"{prefix}_transition_entropy"] = np.nan
        stats[f"{prefix}_stage_shift_index"] = 0.0
        return stats
    transitions = values[1:] != values[:-1]
    stats[f"{prefix}_stage_shift_index"] = float(transitions.sum() / max(1, values.size - 1))
    transition_counts: dict[str, float] = {}
    for left, right in zip(values[:-1], values[1:]):
        if left in STAGE_LABELS and right in STAGE_LABELS:
            key = f"{prefix}_transition_{STAGE_LABELS[int(left)]}_to_{STAGE_LABELS[int(right)]}"
            transition_counts[key] = transition_counts.get(key, 0.0) + 1.0
    stats.update(transition_counts)
    raw_counts = np.asarray(list(transition_counts.values()), dtype=np.float64)
    raw_counts = raw_counts[raw_counts > 0]
    prob = raw_counts / raw_counts.sum() if raw_counts.size else raw_counts
    stats[f"{prefix}_transition_entropy"] = float(-(prob * np.log(prob)).sum()) if prob.size else np.nan
    return stats
