from __future__ import annotations

import numpy as np

from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import SleepRecord
from sleep2stat.reducers.base import BaseReducer
from sleep2stat.registry import register_reducer


@register_reducer("event_density")
class EventDensityReducer(BaseReducer):
    def reduce(
        self,
        records: list[SleepRecord],
        results: list[AnalyzerResult],
        context: Sleep2statContext,
    ) -> list[AnalyzerResult]:
        source = self.config.source
        record_by_id = {record.record_id: record for record in records}
        output = []
        for result in results:
            if result.name != source or result.events is None:
                continue
            record = record_by_id.get(result.record_id)
            hours = record.duration_sec / 3600.0 if record and record.duration_sec > 0 else 0.0
            count = int(len(result.events))
            output.append(
                AnalyzerResult(
                    self.config.name,
                    result.record_id,
                    night={
                        f"{source}_event_count": count,
                        f"{source}_event_density_per_hour": float(count / hours) if hours > 0 else np.nan,
                    },
                )
            )
        return output
