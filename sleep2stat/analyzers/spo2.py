from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data.utils import load_npz
from sleep2stat.analyzers.base import BaseAnalyzer
from sleep2stat.core.artifacts import AnalyzerResult
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.core.stage_sources import StageSourceResolver
from sleep2stat.io.records import SleepRecord
from sleep2stat.registry import register_analyzer


@register_analyzer("spo2_summary")
class Spo2SummaryAnalyzer(BaseAnalyzer):
    def run(
        self,
        records: list[SleepRecord],
        context: Sleep2statContext,
        prior_results: list[AnalyzerResult] | None = None,
    ) -> list[AnalyzerResult]:
        results: list[AnalyzerResult] = []
        for record in records:
            signal, sfreq, valid = _spo2_signal(record, context, self.config)
            results.append(
                AnalyzerResult(self.config.name, record.record_id, night=_spo2_summary(signal, sfreq, valid))
            )
        return results


@register_analyzer("spo2_desaturation")
class Spo2DesaturationAnalyzer(BaseAnalyzer):
    def run(
        self,
        records: list[SleepRecord],
        context: Sleep2statContext,
        prior_results: list[AnalyzerResult] | None = None,
    ) -> list[AnalyzerResult]:
        results: list[AnalyzerResult] = []
        drops = self.config.drop_thresholds
        min_duration = float(self.config.min_duration_sec)
        max_duration = self.config.max_duration_sec
        resolver = StageSourceResolver(records, prior_results or [])
        for record in records:
            signal, sfreq, valid = _spo2_signal(record, context, self.config)
            events = _desaturation_events(
                record,
                self.config.name,
                signal,
                sfreq,
                valid,
                drops=drops,
                min_duration_sec=min_duration,
                max_duration_sec=max_duration,
            )
            results.append(
                AnalyzerResult(
                    self.config.name,
                    record.record_id,
                    events=events,
                    night=_odi_stats(record, events, drops, valid, sfreq, resolver, self.config.stage_source),
                )
            )
        return results


@register_analyzer("event_related_hypoxic_burden")
class EventRelatedHypoxicBurdenAnalyzer(BaseAnalyzer):
    def run(
        self,
        records: list[SleepRecord],
        context: Sleep2statContext,
        prior_results: list[AnalyzerResult] | None = None,
    ) -> list[AnalyzerResult]:
        results: list[AnalyzerResult] = []
        event_source = self.config.event_source
        if not event_source:
            raise ValueError("event_related_hypoxic_burden requires event_source.")
        events_by_record = _events_by_record(prior_results or [], event_source)
        for record in records:
            source_events = events_by_record.get(record.record_id)
            if source_events is None:
                raise ValueError(
                    f"event_source {event_source!r} produced no event result for record {record.record_id!r}."
                )
            if source_events.empty:
                results.append(
                    AnalyzerResult(self.config.name, record.record_id, night=_empty_burden(record, event_source))
                )
                continue
            signal, sfreq, valid = _spo2_signal(record, context, self.config)
            events, night = _event_related_burden(
                record,
                self.config.name,
                event_source,
                source_events,
                signal,
                sfreq,
                valid,
            )
            results.append(AnalyzerResult(self.config.name, record.record_id, events=events, night=night))
        return results


def _spo2_signal(record: SleepRecord, context: Sleep2statContext, config) -> tuple[np.ndarray, float, np.ndarray]:
    channel_name = config.input_channels[0] if config.input_channels else None
    if channel_name is None:
        raise ValueError(f"Analyzer {config.name!r} requires an SpO2 input channel.")
    spec = context.config.signals.channels[channel_name]
    with load_npz(str(record.path)) as npz:
        if spec.source not in npz:
            raise KeyError(f"NPZ key {spec.source!r} not found for SpO2 channel {channel_name!r}.")
        signal = np.asarray(npz[spec.source], dtype=np.float64).reshape(-1) * float(spec.scale)
    valid = np.isfinite(signal)
    artifact = dict(config.artifact or {})
    min_value = artifact.get("valid_min")
    max_value = artifact.get("valid_max")
    if min_value is not None:
        valid &= signal >= float(min_value)
    if max_value is not None:
        valid &= signal <= float(max_value)
    max_change = artifact.get("max_abs_change_per_sec")
    if max_change is not None and signal.size > 1 and spec.sfreq > 0:
        # Drop single-sample oximetry jumps from nadir/T90/event logic; interpolation is intentionally not applied.
        jump = np.abs(np.diff(signal)) * float(spec.sfreq) > float(max_change)
        jump_artifact = np.zeros(signal.size, dtype=bool)
        jump_artifact[1:] |= jump
        valid &= ~jump_artifact
    return signal, float(spec.sfreq), valid


def _spo2_summary(signal: np.ndarray, sfreq: float, valid: np.ndarray) -> dict[str, float]:
    cleaned = signal[valid]
    artifact_pct = float(1.0 - valid.mean()) if valid.size else 0.0
    if cleaned.size == 0:
        return {
            "spo2_mean": np.nan,
            "spo2_median": np.nan,
            "spo2_nadir": np.nan,
            "spo2_t90_min": 0.0,
            "spo2_t90_ratio_recording": 0.0,
            "spo2_t90_pct_recording": 0.0,
            "spo2_t88_min": 0.0,
            "spo2_artifact_pct": artifact_pct,
        }
    recording_min = signal.size / sfreq / 60.0 if sfreq > 0 else 0.0
    # T90/T88 ignore invalid samples in the numerator, while the denominator stays
    # the full recording span.  The paired artifact_pct keeps that choice visible.
    t90_min = float(np.sum(valid & (signal < 90.0)) / sfreq / 60.0) if sfreq > 0 else 0.0
    t88_min = float(np.sum(valid & (signal < 88.0)) / sfreq / 60.0) if sfreq > 0 else 0.0
    return {
        "spo2_mean": float(np.mean(cleaned)),
        "spo2_median": float(np.median(cleaned)),
        "spo2_nadir": float(np.min(cleaned)),
        "spo2_t90_min": t90_min,
        # Ratio stays 0-1; pct is 0-100 for clinical table/report readability.
        "spo2_t90_ratio_recording": float(t90_min / recording_min) if recording_min > 0 else np.nan,
        "spo2_t90_pct_recording": float(t90_min / recording_min * 100.0) if recording_min > 0 else np.nan,
        "spo2_t88_min": t88_min,
        "spo2_artifact_pct": artifact_pct,
    }


def _desaturation_events(
    record: SleepRecord,
    analyzer_name: str,
    signal: np.ndarray,
    sfreq: float,
    valid: np.ndarray,
    *,
    drops: list[float],
    min_duration_sec: float,
    max_duration_sec: float | None,
) -> pd.DataFrame:
    rows = []
    for drop in drops:
        rows.extend(
            _desaturation_rows(
                record,
                analyzer_name,
                signal,
                sfreq,
                valid,
                drop=float(drop),
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
            )
        )
    return pd.DataFrame(rows)


def _desaturation_rows(
    record: SleepRecord,
    analyzer_name: str,
    signal: np.ndarray,
    sfreq: float,
    valid: np.ndarray,
    *,
    drop: float,
    min_duration_sec: float,
    max_duration_sec: float | None,
) -> list[dict[str, Any]]:
    if sfreq <= 0 or signal.size == 0:
        return []
    rows = []
    baseline = None
    start = None
    nadir = None
    event_baseline = None
    for idx, value in enumerate(signal):
        if not valid[idx]:
            # Bad oximetry samples terminate a candidate event; we do not bridge them
            # because nadir and area should be tied to measured SpO2.
            if start is not None:
                rows.extend(
                    _close_desat(
                        record,
                        analyzer_name,
                        drop,
                        start,
                        idx,
                        sfreq,
                        event_baseline,
                        nadir,
                        signal,
                        min_duration_sec,
                    )
                )
            start = None
            baseline = None
            continue
        # The local baseline is the running maximum since the last reset.  A
        # desaturation starts when the current sample is drop points below it.
        baseline = float(value) if baseline is None else max(float(baseline), float(value))
        if start is None and baseline - float(value) >= drop:
            start = idx
            nadir = float(value)
            event_baseline = float(baseline)
        elif start is not None:
            nadir = min(float(nadir), float(value))
            duration = (idx - start) / sfreq
            recovered = float(value) >= float(event_baseline) - 1.0
            timed_out = max_duration_sec is not None and duration >= float(max_duration_sec)
            if recovered or timed_out:
                rows.extend(
                    _close_desat(
                        record,
                        analyzer_name,
                        drop,
                        start,
                        idx,
                        sfreq,
                        event_baseline,
                        nadir,
                        signal,
                        min_duration_sec,
                    )
                )
                start = None
                baseline = float(value)
    if start is not None:
        rows.extend(
            _close_desat(
                record,
                analyzer_name,
                drop,
                start,
                signal.size,
                sfreq,
                event_baseline,
                nadir,
                signal,
                min_duration_sec,
            )
        )
    for event_idx, row in enumerate(rows):
        row["event_id"] = f"{record.record_id}__{analyzer_name}__{int(drop)}pct__{event_idx}"
    return rows


def _close_desat(
    record,
    analyzer_name,
    drop,
    start,
    end,
    sfreq,
    baseline,
    nadir,
    signal,
    min_duration_sec,
) -> list[dict[str, Any]]:
    duration = (end - start) / sfreq
    if duration < min_duration_sec:
        return []
    segment = np.asarray(signal[start:end], dtype=float)
    if segment.size:
        drop_curve = np.maximum(float(baseline) - segment, 0.0)
        # Integrate percent drop over seconds; pct-min below is this same area
        # divided by 60, not a separate event-count metric.
        area_pctsec = float(drop_curve.sum() / sfreq)
        nadir_idx = int(np.nanargmin(segment))
        time_to_nadir = float(nadir_idx / sfreq)
    else:
        area_pctsec = 0.0
        time_to_nadir = np.nan
    return [
        {
            "record_id": record.record_id,
            "path": str(record.path),
            "analyzer": analyzer_name,
            "event_type": f"pred_spo2_desaturation_{int(drop)}pct",
            "onset_sec": float(start / sfreq),
            "offset_sec": float(end / sfreq),
            "duration_sec": float(duration),
            "drop_threshold_pct": float(drop),
            "spo2_baseline": float(baseline),
            "spo2_nadir": float(nadir),
            "spo2_drop_pct": float(baseline - nadir),
            "spo2_desat_area_pctsec": area_pctsec,
            "spo2_desat_area_pctmin": float(area_pctsec / 60.0),
            "spo2_time_to_nadir_sec": time_to_nadir,
            "spo2_recovery_duration_sec": float(duration - time_to_nadir) if not np.isnan(time_to_nadir) else np.nan,
        }
    ]


def _odi_stats(
    record: SleepRecord,
    events: pd.DataFrame,
    drops: list[float],
    valid: np.ndarray,
    sfreq: float,
    resolver: StageSourceResolver,
    stage_source: str | None,
) -> dict[str, float]:
    recording_hours = record.duration_sec / 3600.0 if record.duration_sec > 0 else 0.0
    valid_spo2_hours = float(valid.sum() / sfreq / 3600.0) if sfreq > 0 else 0.0
    # ODI is a desaturation count per hour.  We emit each denominator explicitly,
    # because recording time, valid-SpO2 time, and sleep time answer different questions.
    stage_denominators = resolver.get_denominator_hours(record.record_id, stage_source) if stage_source else None
    if stage_source and stage_denominators is None:
        raise ValueError(
            f"spo2_desaturation stage_source {stage_source!r} has no denominator for {record.record_id!r}."
        )
    output: dict[str, float] = {}
    for drop in drops:
        count = int(np.sum(events.get("drop_threshold_pct", pd.Series(dtype=float)) == float(drop)))
        output[f"ODI{int(drop)}_per_recording_hour"] = _rate(count, recording_hours)
        output[f"ODI{int(drop)}_per_valid_spo2_hour"] = _rate(count, valid_spo2_hours)
        if stage_denominators is not None:
            output[f"ODI{int(drop)}_per_sleep_hour"] = _rate(count, stage_denominators["sleep"])
        output[f"spo2_desaturation_{int(drop)}pct_event_count"] = count
    return output


def _event_related_burden(
    record: SleepRecord,
    analyzer_name: str,
    event_source: str,
    source_events: pd.DataFrame,
    signal: np.ndarray,
    sfreq: float,
    valid: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    burden_prefix = _burden_prefix(event_source)
    rows = []
    burdens = []
    drops = []
    # Multiple upstream event views can describe the same interval.  Burden should
    # integrate each physiologic interval once, not once per duplicate row.
    unique_events = source_events.drop_duplicates(subset=["onset_sec", "offset_sec"]).reset_index(drop=True)
    for event_idx, row in unique_events.iterrows():
        onset = float(row.get("onset_sec", 0.0))
        offset = float(row.get("offset_sec", onset))
        # Use a pre-event maximum as baseline, then integrate the fall over the
        # event plus a short recovery window.  This measures hypoxic burden, not ODI.
        pre_left = max(0, int((onset - 120.0) * sfreq))
        left = max(0, int(onset * sfreq))
        right = min(signal.size, int((offset + 60.0) * sfreq))
        if right <= left or sfreq <= 0:
            continue
        pre = signal[pre_left:left][valid[pre_left:left]]
        segment = signal[left:right][valid[left:right]]
        if pre.size == 0 or segment.size == 0:
            continue
        baseline = float(np.nanmax(pre))
        drop = float(max(0.0, baseline - np.nanmin(segment)))
        burden = float(np.sum(np.maximum(baseline - segment, 0.0)) / sfreq / 60.0)
        burdens.append(burden)
        drops.append(drop)
        rows.append(
            {
                "record_id": record.record_id,
                "path": str(record.path),
                "event_id": f"{record.record_id}__{analyzer_name}__{event_idx}",
                "analyzer": analyzer_name,
                "event_type": "pred_event_related_spo2_drop",
                "onset_sec": onset,
                "offset_sec": offset,
                "duration_sec": max(0.0, offset - onset),
                f"{burden_prefix}_spo2_baseline": baseline,
                f"{burden_prefix}_spo2_drop": drop,
                f"{burden_prefix}_pctmin": burden,
            }
        )
    hours = record.duration_sec / 3600.0 if record.duration_sec > 0 else 0.0
    total_burden = float(np.sum(burdens)) if burdens else 0.0
    night = {
        f"{burden_prefix}_event_count": int(len(rows)),
        f"{burden_prefix}_spo2_drop_mean": float(np.mean(drops)) if drops else np.nan,
        f"{burden_prefix}_spo2_drop_p95": float(np.percentile(drops, 95)) if drops else np.nan,
        f"{burden_prefix}_pctmin": total_burden,
        f"{burden_prefix}_pctmin_per_recording_hour": _rate(total_burden, hours),
    }
    return pd.DataFrame(rows), night


def _empty_burden(record: SleepRecord, event_source: str = "") -> dict[str, float]:
    hours = record.duration_sec / 3600.0 if record.duration_sec > 0 else 0.0
    burden_prefix = _burden_prefix(event_source)
    return {
        f"{burden_prefix}_event_count": 0,
        f"{burden_prefix}_spo2_drop_mean": np.nan,
        f"{burden_prefix}_spo2_drop_p95": np.nan,
        f"{burden_prefix}_pctmin": 0.0,
        f"{burden_prefix}_pctmin_per_recording_hour": 0.0 if hours > 0 else np.nan,
    }


def _burden_prefix(event_source: str) -> str:
    if event_source == "spo2_desaturation":
        # Desaturation-source area is not respiratory-event hypoxic burden, even though both use pct-min units.
        return "desaturation_area_burden"
    return "resp_event_hypoxic_burden"


def _rate(numerator: float, denominator_hours: float) -> float:
    return float(numerator / denominator_hours) if denominator_hours > 0 else np.nan


def _events_by_record(results: list[AnalyzerResult], source: str) -> dict[str, pd.DataFrame]:
    output = {}
    for result in results:
        if result.name == source and result.events is not None:
            output[result.record_id] = result.events
    return output
