from __future__ import annotations

import numpy as np
import pandas as pd

from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import SleepRecord
from sleep2stat.reducers.base import BaseReducer
from sleep2stat.registry import register_reducer


@register_reducer("stage_agreement")
class StageAgreementReducer(BaseReducer):
    def reduce(
        self,
        records: list[SleepRecord],
        results: list[AnalyzerResult],
        context: Sleep2statContext,
    ) -> list[AnalyzerResult]:
        left = self.config.left
        right = self.config.right
        output = []
        left_results = _epoch_results_by_record(results, str(left))
        right_results = _epoch_results_by_record(results, str(right))
        for record_id, left_frame in left_results.items():
            right_frame = right_results.get(record_id)
            if right_frame is None:
                continue
            merged = left_frame.merge(
                right_frame,
                on=["record_id", "path", "token_idx"],
                suffixes=("_left", "_right"),
            )
            left_col = f"{left}_pred"
            right_col = f"{right}_pred"
            if left_col not in merged.columns or right_col not in merged.columns:
                continue
            metrics = _agreement_metrics(
                merged[left_col].to_numpy(),
                merged[right_col].to_numpy(),
                prefix=self.config.name,
            )
            left_valid = int((left_frame[left_col].to_numpy() >= 0).sum()) if left_col in left_frame.columns else 0
            right_valid = int((right_frame[right_col].to_numpy() >= 0).sum()) if right_col in right_frame.columns else 0
            overlap_valid = int(((merged[left_col].to_numpy() >= 0) & (merged[right_col].to_numpy() >= 0)).sum())
            metrics[f"{self.config.name}_overlap_epoch_count"] = overlap_valid
            metrics[f"{self.config.name}_left_epoch_count"] = left_valid
            metrics[f"{self.config.name}_right_epoch_count"] = right_valid
            metrics[f"{self.config.name}_overlap_coverage"] = (
                float(overlap_valid / max(left_valid, right_valid)) if max(left_valid, right_valid) > 0 else np.nan
            )
            output.append(AnalyzerResult(self.config.name, record_id, night=metrics))
        return output


def _epoch_results_by_record(results: list[AnalyzerResult], name: str) -> dict[str, pd.DataFrame]:
    return {
        result.record_id: result.epoch
        for result in results
        if result.name == name and result.epoch is not None and not result.epoch.empty
    }


def _agreement_metrics(left, right, *, prefix: str) -> dict[str, float]:
    left_values = np.asarray(left).reshape(-1)
    right_values = np.asarray(right).reshape(-1)
    valid = (left_values >= 0) & (right_values >= 0)
    left_values = left_values[valid].astype(np.int64)
    right_values = right_values[valid].astype(np.int64)
    if left_values.size == 0:
        return {
            f"{prefix}_accuracy": np.nan,
            f"{prefix}_macro_f1": np.nan,
            f"{prefix}_kappa": np.nan,
            f"{prefix}_disagreement_rate": np.nan,
        }
    accuracy = float(np.mean(left_values == right_values))
    return {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_macro_f1": _macro_f1(left_values, right_values),
        f"{prefix}_kappa": _cohen_kappa(left_values, right_values),
        f"{prefix}_disagreement_rate": float(1.0 - accuracy),
        f"{prefix}_N1_disagreement_rate": _stage_disagreement(left_values, right_values, stage=1),
        f"{prefix}_REM_disagreement_rate": _stage_disagreement(left_values, right_values, stage=4),
    }


def _macro_f1(left: np.ndarray, right: np.ndarray) -> float:
    labels = sorted(set(left.tolist()) | set(right.tolist()))
    f1s = []
    for label in labels:
        tp = np.sum((left == label) & (right == label))
        fp = np.sum((left != label) & (right == label))
        fn = np.sum((left == label) & (right != label))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(np.mean(f1s)) if f1s else np.nan


def _cohen_kappa(left: np.ndarray, right: np.ndarray) -> float:
    labels = sorted(set(left.tolist()) | set(right.tolist()))
    observed = np.mean(left == right)
    expected = 0.0
    total = float(left.size)
    for label in labels:
        expected += (np.sum(left == label) / total) * (np.sum(right == label) / total)
    if expected == 1.0:
        return 1.0
    return float((observed - expected) / (1.0 - expected))


def _stage_disagreement(left: np.ndarray, right: np.ndarray, *, stage: int) -> float:
    mask = (left == stage) | (right == stage)
    if not mask.any():
        return np.nan
    return float(np.mean(left[mask] != right[mask]))
