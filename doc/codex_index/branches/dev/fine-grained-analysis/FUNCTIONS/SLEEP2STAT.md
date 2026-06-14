# sleep2stat Function Index

## `StageSourceResolver.get_denominator_hours`

- File: `sleep2stat/core/stage_sources.py`
- Signature: `get_denominator_hours(record_id: str, source_name: str) -> dict[str, float] | None`
- Purpose and contract: return sleep, REM, and NREM denominator hours from an epoch stage source.
- Reuse guidance: use for AHI, ODI, and other hour-denominator metrics that need sleep-stage context.
- Duplication risk: do not recalculate sleep/REM/NREM masks independently in analyzers.

## `StageSourceResolver.get_stage_minutes`

- File: `sleep2stat/core/stage_sources.py`
- Signature: `get_stage_minutes(record_id: str, source_name: str) -> dict[str, float] | None`
- Purpose and contract: return stage-minute denominators for YASA-style density calculations.
- Reuse guidance: use for per-minute N2, N2N3, N3, NREM, and REM event densities.

## `decode_ahi_logits`

- File: `sleep2stat/analyzers/model_downstream.py`
- Signature: `decode_ahi_logits(...) -> list[AnalyzerResult]`
- Purpose and contract: convert model AHI logits into second alignment, event rows, and model-derived
  respiratory night statistics.
- Important outputs: event-rate fields must distinguish model-covered and recording denominators; clinical
  `pred_ahi` is emitted only when a sleep denominator is available.
- Reuse guidance: keep all sleep2stat model-AHI field naming here.

## `_spo2_signal`

- File: `sleep2stat/analyzers/spo2.py`
- Signature: `_spo2_signal(record, context, config) -> tuple[np.ndarray, float, np.ndarray]`
- Purpose and contract: load and scale SpO2, then build the validity mask used by SpO2 analyzers.
- Reuse guidance: put SpO2 artifact validity rules here so summary, ODI, and burden use the same mask.

## `_odi_stats`

- File: `sleep2stat/analyzers/spo2.py`
- Signature: `_odi_stats(record, events, drops, valid, sfreq, resolver, stage_source) -> dict[str, float]`
- Purpose and contract: summarize desaturation counts using explicit recording, valid-SpO2, and optional
  sleep denominators.

## `HypnogramStatsReducer.reduce`

- File: `sleep2stat/reducers/hypnogram_stats.py`
- Signature: `reduce(records, results, context) -> list[AnalyzerResult]`
- Purpose and contract: summarize stage predictions into sleep architecture metrics using explicit
  ratio/percent and SPT-vs-recording naming.

## `TransitionStatsReducer.reduce`

- File: `sleep2stat/reducers/transition_stats.py`
- Signature: `reduce(records, results, context) -> list[AnalyzerResult]`
- Purpose and contract: summarize adjacent-stage transitions as counts, entropy, and change fraction.
- Duplication risk: do not reuse `stage_shift_index`; it collides with hypnogram per-hour semantics.

## `_event_night_summary`

- File: `sleep2stat/analyzers/yasa.py`
- Signature: `_event_night_summary(record, analyzer_name, event_type, events, resolver=None, stage_source=None) -> dict[str, float]`
- Purpose and contract: summarize YASA event tables and, when stage context exists, emit YASA-style
  event density per stage minute.
