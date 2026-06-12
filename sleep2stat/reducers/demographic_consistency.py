from __future__ import annotations

import math

from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import SleepRecord
from sleep2stat.reducers.base import BaseReducer
from sleep2stat.registry import register_reducer


@register_reducer("demographic_consistency")
class DemographicConsistencyReducer(BaseReducer):
    def reduce(
        self,
        records: list[SleepRecord],
        results: list[AnalyzerResult],
        context: Sleep2statContext,
    ) -> list[AnalyzerResult]:
        by_record = _night_by_record(results)
        output = []
        for record in records:
            merged = dict(by_record.get(record.record_id, {}))
            warnings = []
            age_pred_name = self.config.age_prediction
            if age_pred_name:
                age_pred = merged.get(f"{age_pred_name}_pred")
                age_meta = record.metadata.get(self.config.metadata_age_column)
                if age_pred is not None and age_meta is not None:
                    try:
                        age_meta_float = float(age_meta)
                        if math.isfinite(age_meta_float):
                            merged["age_metadata"] = age_meta_float
                            merged["age_abs_error"] = abs(float(age_pred) - age_meta_float)
                            if merged["age_abs_error"] >= 20:
                                warnings.append("age_prediction_metadata_gap_ge_20")
                    except (TypeError, ValueError):
                        pass
            sex_pred_name = self.config.sex_prediction
            if sex_pred_name:
                sex_pred = merged.get(f"{sex_pred_name}_pred")
                sex_meta = _encode_sex(record.metadata.get(self.config.metadata_sex_column))
                if sex_pred is not None and sex_meta is not None:
                    merged["sex_metadata"] = sex_meta
                    merged["sex_model_metadata_match"] = int(sex_pred) == int(sex_meta)
                    prob_male = merged.get(f"{sex_pred_name}_prob_male")
                    if merged["sex_model_metadata_match"] is False and prob_male is not None:
                        confidence = max(float(prob_male), 1.0 - float(prob_male))
                        if confidence >= 0.9:
                            warnings.append("high_confidence_sex_metadata_conflict")
            merged["demographic_warning_count"] = len(warnings)
            output.append(AnalyzerResult(self.config.name, record.record_id, night=merged, warnings=warnings))
        return output


def _night_by_record(results: list[AnalyzerResult]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for result in results:
        if result.night is None:
            continue
        merged.setdefault(result.record_id, {}).update(result.night)
    return merged


def _encode_sex(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"male", "m", "1", "1.0", "x"}:
            return 1
        if text in {"female", "f", "0", "0.0"}:
            return 0
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number in {0, 1} else None
