from __future__ import annotations

from sleep2stat.config import AnalyzerConfig
from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import SleepRecord


class BaseAnalyzer:
    def __init__(self, config: AnalyzerConfig):
        self.config = config

    def prepare(self, context: Sleep2statContext) -> None:
        return None

    def run(
        self,
        records: list[SleepRecord],
        context: Sleep2statContext,
        prior_results: list[AnalyzerResult] | None = None,
    ) -> list[AnalyzerResult]:
        raise NotImplementedError

    def close(self) -> None:
        return None
