from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np
import pandas as pd

from data.utils import load_npz
from sleep2stat.analyzers.base import BaseAnalyzer
from sleep2stat.config import ChannelSpec
from sleep2stat.core.artifacts import AnalyzerResult, FailureRecord
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.core.stage_sources import StageSourceResolver
from sleep2stat.io.records import SleepRecord
from sleep2stat.registry import register_analyzer

STAGE_LABEL_TO_ID = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "REM": 4}
STAGE_ID_TO_LABEL = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
DEFAULT_BANDS = {
    "delta": (0.5, 4, "Delta"),
    "theta": (4, 8, "Theta"),
    "alpha": (8, 12, "Alpha"),
    "sigma": (12, 16, "Sigma"),
    "beta": (16, 30, "Beta"),
}


class _YasaBaseAnalyzer(BaseAnalyzer):
    def prepare(self, context: Sleep2statContext) -> None:
        self._mne = importlib.import_module("mne")
        self._yasa = importlib.import_module("yasa")

    def _build_raw(self, record: SleepRecord, context: Sleep2statContext):
        specs = {name: context.config.signals.channels[name] for name in self.config.input_channels}
        sfreqs = {float(spec.sfreq) for spec in specs.values()}
        if len(sfreqs) != 1:
            raise ValueError(f"YASA analyzer {self.config.name!r} requires all input channels to share sfreq.")

        arrays = []
        ch_names = []
        ch_types = []
        with load_npz(str(record.path)) as npz:
            for name, spec in specs.items():
                if spec.source not in npz:
                    raise KeyError(f"NPZ key {spec.source!r} not found for YASA channel {name!r}.")
                arrays.append(np.asarray(npz[spec.source], dtype=np.float64).reshape(-1) * float(spec.scale))
                ch_names.append(spec.mne_name or name)
                ch_types.append(_mne_channel_type(spec))
        length = min((array.shape[0] for array in arrays), default=0)
        if length <= 0:
            raise ValueError(f"No signal samples available for YASA analyzer {self.config.name!r}.")
        data = np.stack([array[:length] for array in arrays], axis=0)
        info = self._mne.create_info(ch_names=ch_names, sfreq=sfreqs.pop(), ch_types=ch_types)
        return self._mne.io.RawArray(data, info, verbose=False), dict(zip(self.config.input_channels, ch_names))


@register_analyzer("yasa_stage")
class YasaStageAnalyzer(_YasaBaseAnalyzer):
    def run(
        self,
        records: list[SleepRecord],
        context: Sleep2statContext,
        prior_results: list[AnalyzerResult] | None = None,
    ) -> tuple[list[AnalyzerResult], list[FailureRecord]]:
        results: list[AnalyzerResult] = []
        failures: list[FailureRecord] = []
        include_probabilities = bool(context.config.outputs.include_probabilities)
        for record in records:
            try:
                raw, mne_names = self._build_raw(record, context)
                stager = self._yasa.SleepStaging(
                    raw,
                    eeg_name=_first_kind_channel(self.config.input_channels, context, mne_names, "eeg"),
                    eog_name=_first_kind_channel(self.config.input_channels, context, mne_names, "eog"),
                    emg_name=_first_kind_channel(self.config.input_channels, context, mne_names, "emg"),
                    metadata=_yasa_metadata(record),
                )
                # YASA v0.7 returns a Hypnogram object; stages and probabilities live on that object.
                hypnogram = stager.predict()
                labels = hypnogram.as_int().to_numpy()
                probabilities = None if hypnogram.proba is None else pd.DataFrame(hypnogram.proba)
                n_tokens = min(labels.shape[0], int(record.duration_sec // record.token_sec), record.max_tokens)
                frame = _epoch_base_frame(record, n_tokens)
                stage_ids = np.asarray([_stage_id(label) for label in labels[:n_tokens]], dtype=np.int64)
                frame[f"{self.config.name}_pred"] = stage_ids
                frame[f"{self.config.name}_label"] = [
                    STAGE_ID_TO_LABEL.get(int(value), "UNKNOWN") for value in stage_ids
                ]
                arrays = {}
                if probabilities is not None and n_tokens > 0:
                    proba_frame = probabilities.iloc[:n_tokens].reset_index(drop=True)
                    numeric = proba_frame.select_dtypes(include=[np.number])
                    if not numeric.empty:
                        frame[f"{self.config.name}_confidence"] = numeric.max(axis=1).to_numpy()
                        if include_probabilities:
                            for column in numeric.columns:
                                frame[f"{self.config.name}_prob_{_stage_prob_label(column)}"] = numeric[
                                    column
                                ].to_numpy()
                            arrays[f"{self.config.name}_probabilities"] = numeric.to_numpy(dtype=np.float32)
                results.append(AnalyzerResult(self.config.name, record.record_id, epoch=frame, arrays=arrays))
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


@register_analyzer("yasa_bandpower")
class YasaBandpowerAnalyzer(_YasaBaseAnalyzer):
    def run(
        self,
        records: list[SleepRecord],
        context: Sleep2statContext,
        prior_results: list[AnalyzerResult] | None = None,
    ) -> tuple[list[AnalyzerResult], list[FailureRecord]]:
        results: list[AnalyzerResult] = []
        failures: list[FailureRecord] = []
        options = self.config.outputs
        by_epoch = bool(options["by_epoch"])
        by_stage = bool(options["by_stage"])
        by_night = bool(options["by_night"])
        relative = bool(options["relative"])
        stage_source = self.config.stage_source
        bands = _band_specs(options.get("bands"))
        stage_results = _stage_results_by_record(prior_results or [], str(stage_source)) if stage_source else {}
        for record in records:
            try:
                raw, _ = self._build_raw(record, context)
                epoch = _epoch_bandpower(
                    self._yasa,
                    raw,
                    record,
                    self.config.name,
                    bands=bands,
                    relative=relative,
                )
                night = {}
                if by_night:
                    night.update(_night_bandpower_means(self.config.name, epoch))
                if by_stage:
                    if not stage_source:
                        raise ValueError("yasa_bandpower stage_source is required when by_stage=true.")
                    stage = stage_results.get(record.record_id)
                    if stage is None:
                        raise ValueError(
                            f"yasa_bandpower stage_source {stage_source!r} has no "
                            f"epoch result for {record.record_id!r}."
                        )
                    night.update(_stage_bandpower_means(self.config.name, epoch, stage, str(stage_source)))
                results.append(
                    AnalyzerResult(
                        self.config.name,
                        record.record_id,
                        epoch=epoch if by_epoch else None,
                        night=night or None,
                    )
                )
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


class _YasaEventAnalyzer(_YasaBaseAnalyzer):
    detector_name = ""
    event_type = ""

    def run(
        self,
        records: list[SleepRecord],
        context: Sleep2statContext,
        prior_results: list[AnalyzerResult] | None = None,
    ) -> tuple[list[AnalyzerResult], list[FailureRecord]]:
        results: list[AnalyzerResult] = []
        failures: list[FailureRecord] = []
        resolver = StageSourceResolver(records, prior_results or [])
        for record in records:
            try:
                raw, _ = self._build_raw(record, context)
                frame = _call_yasa_event_detector(
                    self._yasa,
                    self.detector_name,
                    raw,
                    stage_source=self.config.stage_source,
                    stages=self.config.stages,
                    resolver=resolver,
                    record=record,
                )
                events = _yasa_event_frame(record, self.config.name, self.event_type, frame)
                night = _event_night_summary(
                    record,
                    self.config.name,
                    self.event_type,
                    events,
                    resolver,
                    self.config.stage_source,
                )
                results.append(AnalyzerResult(self.config.name, record.record_id, events=events, night=night))
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


@register_analyzer("yasa_spindles")
class YasaSpindlesAnalyzer(_YasaEventAnalyzer):
    detector_name = "spindles_detect"
    event_type = "yasa_spindle"


@register_analyzer("yasa_slowwaves")
class YasaSlowWavesAnalyzer(_YasaEventAnalyzer):
    detector_name = "sw_detect"
    event_type = "yasa_slowwave"


@register_analyzer("yasa_rem")
class YasaRemAnalyzer(_YasaEventAnalyzer):
    detector_name = "rem_detect"
    event_type = "yasa_rem"


@register_analyzer("yasa_hrv_stage")
class YasaHrvStageAnalyzer(_YasaBaseAnalyzer):
    def run(
        self,
        records: list[SleepRecord],
        context: Sleep2statContext,
        prior_results: list[AnalyzerResult] | None = None,
    ) -> tuple[list[AnalyzerResult], list[FailureRecord]]:
        results: list[AnalyzerResult] = []
        failures: list[FailureRecord] = []
        resolver = StageSourceResolver(records, prior_results or [])
        if not self.config.stage_source:
            return [], [
                FailureRecord(
                    record_id=record.record_id,
                    analyzer=self.config.name,
                    error_type="ValueError",
                    message="yasa_hrv_stage requires stage_source.",
                )
                for record in records
            ]
        for record in records:
            try:
                raw, _ = self._build_raw(record, context)
                data, sfreq, _ = _raw_data(raw)
                seconds = np.arange(data.shape[1], dtype=np.float64) / sfreq
                hypno = resolver.stage_at_seconds(record.record_id, self.config.stage_source, seconds)
                if hypno is None:
                    raise ValueError(f"stage_source {self.config.stage_source!r} not found for {record.record_id!r}.")
                frame = _call_yasa_hrv_stage(self._yasa, raw, hypno)
                results.append(
                    AnalyzerResult(self.config.name, record.record_id, night=_flatten_hrv(self.config.name, frame))
                )
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


def _mne_channel_type(spec: ChannelSpec) -> str:
    kind = spec.kind.lower()
    if kind in {"eeg", "eog", "emg", "ecg", "misc"}:
        return kind
    return "misc"


def _first_kind_channel(
    input_channels: list[str],
    context: Sleep2statContext,
    mne_names: dict[str, str],
    kind: str,
) -> str | None:
    for channel in input_channels:
        if context.config.signals.channels[channel].kind.lower() == kind:
            return mne_names[channel]
    return None


def _yasa_metadata(record: SleepRecord) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {}
    age = _finite_float(record.metadata.get("age"))
    if age is not None:
        metadata["age"] = age
    sex = record.metadata.get("sex")
    if sex is not None:
        male = _sex_to_male(sex)
        if male is not None:
            metadata["male"] = male
    return metadata or None


def _sex_to_male(value: Any) -> bool | None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"m", "male", "1", "true"}:
            return True
        if normalized in {"f", "female", "0", "false"}:
            return False
        numeric = _finite_float(normalized)
    else:
        numeric = _finite_float(value)
    if numeric is None:
        return None
    if numeric == 0:
        return False
    if numeric == 1:
        return True
    return None


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _stage_id(label: Any) -> int:
    if isinstance(label, str):
        return STAGE_LABEL_TO_ID.get(label.strip().upper(), -1)
    try:
        value = int(label)
    except (TypeError, ValueError):
        return -1
    return value if value in STAGE_ID_TO_LABEL else -1


def _stage_prob_label(column: Any) -> str:
    if isinstance(column, str):
        return "REM" if column.upper() == "R" else _slug(column)
    try:
        value = int(column)
    except (TypeError, ValueError):
        return _slug(column)
    return STAGE_ID_TO_LABEL.get(value, str(value))


def _epoch_base_frame(record: SleepRecord, n_tokens: int) -> pd.DataFrame:
    token_idx = np.arange(n_tokens, dtype=np.int64)
    start_sec = token_idx * record.token_sec
    return pd.DataFrame(
        {
            "record_id": record.record_id,
            "path": str(record.path),
            "token_idx": token_idx,
            "start_sec": start_sec.astype(np.float32),
            "end_sec": (start_sec + record.token_sec).astype(np.float32),
            "is_padding": False,
        }
    )


def _flatten_bandpower(analyzer_name: str, frame: pd.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {}
    if frame.empty:
        return output
    for row_idx, row in frame.reset_index(drop=True).iterrows():
        channel = row.get("Chan", row.get("channel", row.get("ch_name", row_idx)))
        channel_slug = _slug(channel)
        for column, value in row.items():
            numeric = _finite_float(value)
            if numeric is None:
                continue
            output[f"{analyzer_name}_{channel_slug}_{_slug(column)}"] = numeric
    return output


def _band_specs(raw: Any) -> list[tuple[float, float, str]]:
    if raw is None:
        raw = ["delta", "theta", "alpha", "sigma", "beta"]
    specs = []
    for item in raw:
        if isinstance(item, str):
            key = item.strip().lower()
            if key not in DEFAULT_BANDS:
                raise ValueError(f"Unsupported YASA bandpower band: {item!r}.")
            specs.append(DEFAULT_BANDS[key])
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            specs.append((float(item[0]), float(item[1]), str(item[2])))
        else:
            raise ValueError(f"Invalid YASA bandpower band spec: {item!r}.")
    return specs


def _epoch_bandpower(
    yasa_module: Any,
    raw: Any,
    record: SleepRecord,
    analyzer_name: str,
    *,
    bands: list[tuple[float, float, str]],
    relative: bool,
) -> pd.DataFrame:
    data, sfreq, ch_names = _raw_data(raw, units="uV")
    samples_per_epoch = int(round(float(sfreq) * record.token_sec))
    n_tokens = min(int(record.duration_sec // record.token_sec), record.max_tokens)
    n_tokens = min(n_tokens, data.shape[1] // samples_per_epoch if samples_per_epoch > 0 else 0)
    frame = _epoch_base_frame(record, n_tokens)
    suffix = "rel" if relative else "abs"
    band_names = [_slug(label).lower() for _, _, label in bands]
    for band_name in band_names:
        frame[f"{analyzer_name}_{band_name}_{suffix}"] = np.nan
    for token_idx in range(n_tokens):
        # Run bandpower on the same token windows used by staging.  Later reducers
        # can then join by token_idx without doing floating-point time matching.
        left = token_idx * samples_per_epoch
        right = left + samples_per_epoch
        segment = data[:, left:right]
        bandpower = _call_bandpower(yasa_module, segment, sfreq, ch_names, bands, relative)
        means = _band_means(bandpower, band_names)
        for band_name, value in means.items():
            frame.loc[token_idx, f"{analyzer_name}_{band_name}_{suffix}"] = value
    return frame


def _raw_data(raw: Any, *, units: str | None = None) -> tuple[np.ndarray, float, list[str]]:
    if hasattr(raw, "get_data"):
        if units == "uV":
            # MNE Raw stores Volts internally, while YASA array APIs expect microvolts.
            data = raw.get_data(units={"eeg": "uV", "eog": "uV", "emg": "uV", "ecg": "uV"})
        else:
            data = raw.get_data()
    else:
        data = raw.data
    info = raw.info
    sfreq = float(info["sfreq"] if isinstance(info, dict) else info["sfreq"])
    ch_names = None
    if isinstance(info, dict):
        ch_names = info.get("ch_names")
    if ch_names is None:
        ch_names = getattr(raw, "ch_names", [f"ch{idx}" for idx in range(np.asarray(data).shape[0])])
    return np.asarray(data, dtype=np.float64), sfreq, list(ch_names)


def _call_bandpower(
    yasa_module: Any,
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    bands: list[tuple[float, float, str]],
    relative: bool,
) -> pd.DataFrame:
    try:
        return pd.DataFrame(yasa_module.bandpower(data, sf=sfreq, ch_names=ch_names, bands=bands, relative=relative))
    except TypeError:
        return pd.DataFrame(yasa_module.bandpower(data))


def _band_means(frame: pd.DataFrame, band_names: list[str]) -> dict[str, float]:
    output = {}
    if frame.empty:
        return output
    columns = {_slug(column).lower(): column for column in frame.columns}
    for band_name in band_names:
        column = columns.get(band_name)
        if column is None:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            output[band_name] = float(values.mean())
    return output


def _night_bandpower_means(analyzer_name: str, epoch: pd.DataFrame) -> dict[str, float]:
    output = {}
    prefix = f"{analyzer_name}_"
    for column in epoch.columns:
        if not column.startswith(prefix) or not column.endswith(("_rel", "_abs")):
            continue
        values = pd.to_numeric(epoch[column], errors="coerce")
        if values.notna().any():
            output[f"{column}_mean"] = float(values.mean())
    return output


def _stage_bandpower_means(
    analyzer_name: str,
    epoch: pd.DataFrame,
    stage: pd.DataFrame,
    stage_source: str,
) -> dict[str, float]:
    pred_col = f"{stage_source}_pred"
    if pred_col not in stage.columns:
        raise ValueError(f"stage_source {stage_source!r} epoch output does not contain {pred_col!r}.")
    # Stage-specific band means use the stage assigned to the same token window.
    # This intentionally differs from event summaries, which are staged by onset.
    merged = epoch.merge(stage[["token_idx", pred_col]], on="token_idx", how="left")
    output = {}
    band_columns = [
        column
        for column in epoch.columns
        if column.startswith(f"{analyzer_name}_") and column.endswith(("_rel", "_abs"))
    ]
    for stage_id, stage_label in STAGE_ID_TO_LABEL.items():
        group = merged[merged[pred_col] == stage_id]
        if group.empty:
            continue
        for column in band_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            if values.notna().any():
                band = column.removeprefix(f"{analyzer_name}_").removesuffix("_rel").removesuffix("_abs")
                output[f"{analyzer_name}_{stage_label}_{band}_mean"] = float(values.mean())
    return output


def _stage_results_by_record(results: list[AnalyzerResult], stage_source: str) -> dict[str, pd.DataFrame]:
    output = {}
    for result in results:
        if result.name == stage_source and result.epoch is not None and not result.epoch.empty:
            output[result.record_id] = result.epoch
    return output


def _call_yasa_event_detector(
    yasa_module: Any,
    detector_name: str,
    raw: Any,
    *,
    stage_source: str | None,
    stages: list[str],
    resolver: StageSourceResolver,
    record: SleepRecord,
) -> pd.DataFrame:
    detector = getattr(yasa_module, detector_name)
    hypno = None
    if stages:
        if not stage_source:
            raise ValueError(f"{detector_name} stage filtering requires stage_source.")
        data, sfreq, _ = _raw_data(raw)
        seconds = np.arange(data.shape[1], dtype=np.float64) / sfreq
        stage = resolver.stage_at_seconds(record.record_id, stage_source, seconds)
        if stage is None:
            raise ValueError(f"stage_source {stage_source!r} not found for {record.record_id!r}.")
        allowed = {_stage_id(stage) for stage in stages}
        keep = (stage >= 0) & np.isin(stage, list(allowed))
        hypno = np.where(keep, stage, 0).astype(np.int64)
    if detector_name == "rem_detect":
        # YASA rem_detect is defined on paired LOC/ROC EOG arrays, not on Raw plus channel names.
        data, sfreq, _ = _raw_data(raw, units="uV")
        if data.shape[0] != 2:
            raise ValueError("yasa_rem requires exactly two EOG input channels.")
        result = detector(data[0], data[1], sfreq, hypno=hypno)
    else:
        try:
            result = detector(raw, hypno=hypno) if hypno is not None else detector(raw)
        except TypeError:
            data, sfreq, ch_names = _raw_data(raw, units="uV")
            kwargs = {"sf": sfreq, "ch_names": ch_names}
            if hypno is not None:
                kwargs["hypno"] = hypno
            result = detector(data, **kwargs)
    summary = getattr(result, "summary", None)
    if callable(summary):
        result = summary()
    return pd.DataFrame(result)


def _yasa_event_frame(record: SleepRecord, analyzer_name: str, event_type: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for event_idx, row in frame.reset_index(drop=True).iterrows():
        onset = _first_numeric(row, "Start", "start", "Onset", "onset", "onset_sec")
        duration = _first_numeric(row, "Duration", "duration", "duration_sec")
        end = _first_numeric(row, "End", "end", "offset_sec")
        if onset is None:
            continue
        if duration is None and end is not None:
            duration = float(end) - float(onset)
        if duration is None:
            duration = 0.0
        event = {
            "record_id": record.record_id,
            "path": str(record.path),
            "event_id": f"{record.record_id}__{analyzer_name}__{event_idx}",
            "analyzer": analyzer_name,
            "event_type": event_type,
            "onset_sec": float(onset),
            "offset_sec": float(onset) + float(duration),
            "duration_sec": float(duration),
        }
        for source, target in (
            ("Confidence", "confidence"),
            ("Prob", "confidence"),
            ("Amplitude", "amplitude"),
            ("Frequency", "frequency"),
            ("RelPower", "relative_power"),
            ("RMS", "rms"),
            ("Slope", "slope"),
            ("Stage", "stage"),
        ):
            if source in row and not _is_missing(row[source]):
                event[f"{analyzer_name}_{target}"] = row[source]
        rows.append(event)
    return pd.DataFrame(rows)


def _event_night_summary(
    record: SleepRecord,
    analyzer_name: str,
    event_type: str,
    events: pd.DataFrame,
    resolver: StageSourceResolver | None = None,
    stage_source: str | None = None,
) -> dict[str, float]:
    hours = record.duration_sec / 3600.0 if record.duration_sec > 0 else 0.0
    count = int(len(events))
    output = {
        f"{analyzer_name}_event_count": count,
        f"{analyzer_name}_event_density_per_hour": float(count / hours) if hours > 0 else np.nan,
    }
    if not events.empty and "duration_sec" in events.columns:
        output[f"{analyzer_name}_duration_mean_sec"] = float(
            pd.to_numeric(events["duration_sec"], errors="coerce").mean()
        )
    if resolver is not None and stage_source:
        stage_minutes = resolver.get_stage_minutes(record.record_id, stage_source)
        if stage_minutes is not None:
            density_events = events
            stage_col = f"{analyzer_name}_stage"
            if not events.empty and stage_col not in events.columns:
                # Stage-specific density is event count per minute in that sleep stage, so each event needs a stage.
                if "onset_sec" not in events.columns:
                    raise ValueError(f"YASA event table for {analyzer_name!r} is missing 'onset_sec'.")
                stages = resolver.stage_at_seconds(record.record_id, stage_source, events["onset_sec"].to_numpy())
                if stages is None:
                    raise ValueError(f"stage_source {stage_source!r} not found for {record.record_id!r}.")
                density_events = events.copy()
                density_events[stage_col] = stages
            # Mirrors YASA DetectionResults.summary(grp_stage=True): stage event count divided by stage minutes.
            output.update(_stage_event_densities(analyzer_name, event_type, density_events, stage_minutes))
    return output


def _stage_event_densities(
    analyzer_name: str,
    event_type: str,
    events: pd.DataFrame,
    stage_minutes: dict[str, float],
) -> dict[str, float]:
    output: dict[str, float] = {}
    # These denominators mirror the usual physiologic context of each detector:
    # spindles in N2/N3, slow waves in N3/NREM, and REM events in REM.
    if event_type == "yasa_spindle":
        output[f"{analyzer_name}_spindle_density_per_min_N2"] = _stage_density(
            events,
            analyzer_name,
            "N2",
            stage_minutes,
        )
        n2n3_count = _stage_event_count(events, analyzer_name, {"N2", "N3"})
        output[f"{analyzer_name}_spindle_density_per_min_N2N3"] = _count_per_min(
            n2n3_count,
            stage_minutes.get("N2N3", 0.0),
        )
    elif event_type == "yasa_slowwave":
        output[f"{analyzer_name}_slowwave_density_per_min_N3"] = _stage_density(
            events,
            analyzer_name,
            "N3",
            stage_minutes,
        )
        nrem_count = _stage_event_count(events, analyzer_name, {"N1", "N2", "N3"})
        output[f"{analyzer_name}_slowwave_density_per_min_NREM"] = _count_per_min(
            nrem_count,
            stage_minutes.get("NREM", 0.0),
        )
    elif event_type == "yasa_rem":
        output[f"{analyzer_name}_rapid_eye_movement_density_per_min_REM"] = _stage_density(
            events,
            analyzer_name,
            "REM",
            stage_minutes,
        )
    return output


def _stage_density(
    events: pd.DataFrame,
    analyzer_name: str,
    stage: str,
    stage_minutes: dict[str, float],
) -> float:
    return _count_per_min(_stage_event_count(events, analyzer_name, {stage}), stage_minutes.get(stage, 0.0))


def _stage_event_count(events: pd.DataFrame, analyzer_name: str, stages: set[str]) -> int:
    stage_col = f"{analyzer_name}_stage"
    if events.empty:
        return 0
    if stage_col not in events.columns:
        raise ValueError(f"YASA event table for {analyzer_name!r} is missing {stage_col!r}.")
    labels = events[stage_col].map(lambda value: STAGE_ID_TO_LABEL.get(_stage_id(value), _slug(value)))
    return int(labels.isin(stages).sum())


def _count_per_min(count: int, minutes: float) -> float:
    return float(count / minutes) if minutes > 0 else np.nan


def _call_yasa_hrv_stage(yasa_module: Any, raw: Any, hypno: np.ndarray) -> pd.DataFrame:
    hrv_stage = getattr(yasa_module, "hrv_stage")
    data, sfreq, _ = _raw_data(raw)
    result = hrv_stage(data[0], sfreq, hypno=hypno)
    if isinstance(result, tuple):
        result = result[0]
    return pd.DataFrame(result)


def _flatten_hrv(analyzer_name: str, frame: pd.DataFrame) -> dict[str, float]:
    output = {}
    if frame.empty:
        return output
    data = frame.reset_index()
    stage_col = None
    for candidate in ("Stage", "stage", "values"):
        if candidate in data.columns:
            stage_col = candidate
            break
    if stage_col is None:
        return output
    # YASA hrv_stage returns per-epoch rows, often indexed by integer stage values.
    ignored = {"stage", "values", "epoch", "start", "duration", "index"}
    metric_columns = [column for column in data.columns if str(column).lower() not in ignored]
    for stage_value, group in data.groupby(stage_col):
        stage_id = _stage_id(stage_value)
        stage = STAGE_ID_TO_LABEL.get(stage_id, _slug(stage_value))
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            if values.notna().any():
                output[f"{analyzer_name}_{stage}_{_slug(column)}"] = float(values.mean())
    return output


def _first_numeric(row: pd.Series, *names: str) -> float | None:
    for name in names:
        if name in row:
            value = _finite_float(row[name])
            if value is not None:
                return value
    return None


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _slug(value: Any) -> str:
    text = str(value).strip()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in cleaned.split("_") if part) or "value"
