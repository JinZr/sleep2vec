from __future__ import annotations

from sleep2stat.config import ReducerConfig
from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import SleepRecord


class BaseReducer:
    def __init__(self, config: ReducerConfig):
        self.config = config

    def reduce(
        self,
        records: list[SleepRecord],
        results: list[AnalyzerResult],
        context: Sleep2statContext,
    ) -> list[AnalyzerResult]:
        raise NotImplementedError
