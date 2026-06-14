from __future__ import annotations

import numpy as np
import pandas as pd

from data.utils import load_npz
from sleep2stat.analyzers.base import BaseAnalyzer
from sleep2stat.core.artifacts import AnalyzerResult, FailureRecord
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import SleepRecord
from sleep2stat.registry import register_analyzer


@register_analyzer("npz_stage_reference")
class NpzStageReferenceAnalyzer(BaseAnalyzer):
    def run(
        self,
        records: list[SleepRecord],
        context: Sleep2statContext,
        prior_results: list[AnalyzerResult] | None = None,
    ) -> tuple[list[AnalyzerResult], list[FailureRecord]]:
        key = str(self.config.stage_key)
        results: list[AnalyzerResult] = []
        failures: list[FailureRecord] = []
        for record in records:
            try:
                with load_npz(str(record.path)) as npz:
                    if key not in npz:
                        raise KeyError(f"NPZ key {key!r} not found.")
                    values = np.asarray(npz[key]).reshape(-1)
                n_tokens = min(values.shape[0], int(record.duration_sec // record.token_sec), record.max_tokens)
                token_idx = np.arange(n_tokens, dtype=np.int64)
                start_sec = token_idx * record.token_sec
                frame = pd.DataFrame(
                    {
                        "record_id": record.record_id,
                        "path": str(record.path),
                        "token_idx": token_idx,
                        "start_sec": start_sec.astype(np.float32),
                        "end_sec": (start_sec + record.token_sec).astype(np.float32),
                        f"{self.config.name}_pred": values[:n_tokens].astype(np.int64),
                    }
                )
                results.append(AnalyzerResult(self.config.name, record.record_id, epoch=frame))
            except Exception as exc:
                failures.append(
                    FailureRecord(
                        record_id=record.record_id,
                        analyzer=self.config.name,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
        return results, failures
