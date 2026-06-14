from __future__ import annotations

from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import SleepRecord
from sleep2stat.reducers.base import BaseReducer
from sleep2stat.registry import register_reducer


@register_reducer("respiratory_stats")
class RespiratoryStatsReducer(BaseReducer):
    def reduce(
        self,
        records: list[SleepRecord],
        results: list[AnalyzerResult],
        context: Sleep2statContext,
    ) -> list[AnalyzerResult]:
        # The v0.1 AHI analyzer already emits model-derived respiratory night stats.
        source = self.config.source
        return [
            AnalyzerResult(self.config.name, result.record_id, night=dict(result.night or {}))
            for result in results
            if result.name == source and result.night is not None
        ]
