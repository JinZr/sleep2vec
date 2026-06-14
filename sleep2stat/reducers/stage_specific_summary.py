from __future__ import annotations

import pandas as pd

from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import SleepRecord
from sleep2stat.reducers.base import BaseReducer
from sleep2stat.registry import register_reducer

STAGE_LABELS = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


@register_reducer("stage_specific_summary")
class StageSpecificSummaryReducer(BaseReducer):
    def reduce(
        self,
        records: list[SleepRecord],
        results: list[AnalyzerResult],
        context: Sleep2statContext,
    ) -> list[AnalyzerResult]:
        stage_source = str(self.config.options.get("stage_source", ""))
        if not stage_source:
            raise ValueError("stage_specific_summary requires options.stage_source.")
        source_results = {
            result.record_id: result.epoch
            for result in results
            if result.name == self.config.source and result.epoch is not None and not result.epoch.empty
        }
        stage_results = {
            result.record_id: result.epoch
            for result in results
            if result.name == stage_source and result.epoch is not None and not result.epoch.empty
        }
        output = []
        stage_col = f"{stage_source}_pred"
        for record_id, frame in source_results.items():
            stage = stage_results.get(record_id)
            if stage is None:
                raise ValueError(
                    f"stage_specific_summary stage_source {stage_source!r} has no epoch result for {record_id!r}."
                )
            if stage_col not in stage.columns:
                raise ValueError(
                    f"stage_specific_summary stage_source {stage_source!r} is missing column {stage_col!r} "
                    f"for {record_id!r}."
                )
            # Join by token_idx rather than timestamps; both frames are already
            # epoch-indexed, and this avoids rounding differences in start/end seconds.
            merged = frame.merge(stage[["token_idx", stage_col]], on="token_idx", how="inner")
            night = _stage_numeric_means(str(self.config.source), merged, stage_col)
            if night:
                output.append(AnalyzerResult(self.config.name, record_id, night=night))
        return output


def _stage_numeric_means(source: str, frame: pd.DataFrame, stage_col: str) -> dict[str, float]:
    output = {}
    numeric_columns = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in {"token_idx", "start_sec", "end_sec", stage_col}
    ]
    for stage_id, label in STAGE_LABELS.items():
        group = frame[frame[stage_col] == stage_id]
        if group.empty:
            continue
        for column in numeric_columns:
            if column == stage_col:
                continue
            value = group[column].mean()
            if pd.notna(value):
                metric = column.removeprefix(f"{source}_")
                output[f"{source}_{label}_{metric}_mean"] = float(value)
    return output
