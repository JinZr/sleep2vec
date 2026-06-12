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
from sleep2stat.io.records import SleepRecord
from sleep2stat.registry import register_analyzer

STAGE_LABEL_TO_ID = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "REM": 4}
STAGE_ID_TO_LABEL = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


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
                labels = np.asarray(stager.predict()).reshape(-1)
                probabilities = _predict_proba(stager)
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
    ) -> tuple[list[AnalyzerResult], list[FailureRecord]]:
        results: list[AnalyzerResult] = []
        failures: list[FailureRecord] = []
        for record in records:
            try:
                raw, _ = self._build_raw(record, context)
                bandpower = pd.DataFrame(self._yasa.bandpower(raw))
                night = _flatten_bandpower(self.config.name, bandpower)
                results.append(AnalyzerResult(self.config.name, record.record_id, night=night))
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
        return None
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


def _predict_proba(stager: Any) -> pd.DataFrame | None:
    predict_proba = getattr(stager, "predict_proba", None)
    if predict_proba is None:
        return None
    proba = predict_proba()
    if proba is None:
        return None
    return pd.DataFrame(proba)


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


def _slug(value: Any) -> str:
    text = str(value).strip()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in cleaned.split("_") if part) or "value"
