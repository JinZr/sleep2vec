from __future__ import annotations

import numpy as np
import pandas as pd

from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.io.records import SleepRecord


class StageSourceResolver:
    def __init__(self, records: list[SleepRecord], results: list[AnalyzerResult] | None = None):
        self.records = {record.record_id: record for record in records}
        self.epochs: dict[tuple[str, str], pd.DataFrame] = {}
        for result in results or []:
            if result.epoch is not None and not result.epoch.empty:
                self.epochs[(result.record_id, result.name)] = result.epoch

    def get_epoch_stage(self, record_id: str, source_name: str) -> pd.DataFrame | None:
        frame = self.epochs.get((record_id, source_name))
        if frame is None or f"{source_name}_pred" not in frame.columns:
            return None
        return frame

    def get_stage_mask(self, record_id: str, source_name: str, stage_id: int) -> np.ndarray | None:
        stages = self._stage_values(record_id, source_name)
        if stages is None:
            return None
        return stages == int(stage_id)

    def get_tst_hours(self, record_id: str, source_name: str) -> float | None:
        stages = self._stage_values(record_id, source_name)
        if stages is None:
            return None
        token_sec = self.records.get(record_id).token_sec if record_id in self.records else 30
        sleep_epochs = int(np.sum(np.isin(stages, [1, 2, 3, 4])))
        return float(sleep_epochs * token_sec / 3600.0)

    def get_denominator_hours(self, record_id: str, source_name: str) -> dict[str, float] | None:
        stages = self._stage_values(record_id, source_name)
        if stages is None:
            return None
        token_sec = self.records.get(record_id).token_sec if record_id in self.records else 30
        hours_per_epoch = token_sec / 3600.0
        # AHI-style denominators are in hours: all sleep for AHI, and REM/NREM for
        # stage-specific respiratory-event rates.
        return {
            "sleep": float(np.sum(np.isin(stages, [1, 2, 3, 4])) * hours_per_epoch),
            "rem": float(np.sum(stages == 4) * hours_per_epoch),
            "nrem": float(np.sum(np.isin(stages, [1, 2, 3])) * hours_per_epoch),
        }

    def get_stage_minutes(self, record_id: str, source_name: str) -> dict[str, float] | None:
        stages = self._stage_values(record_id, source_name)
        if stages is None:
            return None
        token_sec = self.records.get(record_id).token_sec if record_id in self.records else 30
        minutes_per_epoch = token_sec / 60.0
        # YASA event summaries report densities per stage minute, so keep minutes
        # here instead of sharing the hour-denominator helper above.
        return {
            "N1": float(np.sum(stages == 1) * minutes_per_epoch),
            "N2": float(np.sum(stages == 2) * minutes_per_epoch),
            "N3": float(np.sum(stages == 3) * minutes_per_epoch),
            "REM": float(np.sum(stages == 4) * minutes_per_epoch),
            "NREM": float(np.sum(np.isin(stages, [1, 2, 3])) * minutes_per_epoch),
            "N2N3": float(np.sum(np.isin(stages, [2, 3])) * minutes_per_epoch),
            "sleep": float(np.sum(np.isin(stages, [1, 2, 3, 4])) * minutes_per_epoch),
        }

    def stage_at_seconds(self, record_id: str, source_name: str, seconds: np.ndarray) -> np.ndarray | None:
        frame = self.get_epoch_stage(record_id, source_name)
        if frame is None:
            return None
        pred_col = f"{source_name}_pred"
        token_sec = self.records.get(record_id).token_sec if record_id in self.records else 30
        token_idx = np.floor(np.asarray(seconds, dtype=float) / token_sec).astype(int)
        by_token = dict(zip(frame["token_idx"].astype(int), frame[pred_col].astype(int)))
        return np.asarray([by_token.get(int(idx), -1) for idx in token_idx], dtype=np.int64)

    def _stage_values(self, record_id: str, source_name: str) -> np.ndarray | None:
        frame = self.get_epoch_stage(record_id, source_name)
        if frame is None:
            return None
        return frame[f"{source_name}_pred"].to_numpy(dtype=np.int64)
