# Reuse Guide

## sleep2stat Hotspots

- Reuse `StageSourceResolver` for sleep, REM, NREM, and stage-minute denominators. Do not duplicate
  stage denominator logic inside analyzers or reducers.
- Reuse `AnalyzerResult` for analyzer and reducer outputs. Add fields to existing `night`, `epoch`,
  `second`, or `events` tables instead of inventing a parallel result object.
- Keep model-derived AHI postprocessing in `decode_ahi_logits`; do not add a second AHI naming path in
  reducers or CLI plotting.
- Keep SpO2 validity filtering in `_spo2_signal`; summary, desaturation, and burden analyzers should share
  the same valid mask.
- Keep YASA event table flattening in `_yasa_event_frame` and night-level event summary in
  `_event_night_summary`.
- Keep plot field fallback in `sleep2stat/plot.py`; do not write compatibility columns into generated
  result tables solely for plotting.

## Duplication Risks

- `stage_shift_index` is ambiguous: hypnogram uses transitions per TST hour, while transition stats use
  adjacent-epoch fraction. Use explicit output names.
- `pct_*` is ambiguous unless the field is explicitly percent. Prefer `*_ratio_TST` for 0-1 values and
  `*_pct_TST` for 0-100 values.
- AHI, ODI, burden, and event density fields must encode their denominator.
