# Module Map

| Responsibility | Canonical files | Notes |
| --- | --- | --- |
| Record and result contracts | `sleep2stat/io/records.py`, `sleep2stat/core/artifacts.py` | Preserve append-style sidecar outputs; do not mutate source NPZ/index data. |
| Stage source reuse | `sleep2stat/core/stage_sources.py` | Central place for stage denominators and stage-at-time lookup. |
| Model downstream outputs | `sleep2stat/analyzers/model_downstream.py` | Owns model-derived stage/AHI/age/sex analyzer rows and AHI postprocessing names. |
| SpO2 analyzers | `sleep2stat/analyzers/spo2.py` | Owns SpO2 summary, desaturation events, ODI, and event-related burden naming. |
| YASA analyzers | `sleep2stat/analyzers/yasa.py` | Owns YASA stage, bandpower, spindle, slow-wave, REM, and HRV summaries. |
| Night reducers | `sleep2stat/reducers/` | Owns derived table-level statistics such as hypnogram, transition, agreement, density, and demographics. |
| Plot field selection | `sleep2stat/plot.py` | Prefer explicit denominator/unit fields; keep legacy fallback only for old result bundles. |
| Tests | `tests/test_sleep2stat_*.py` | Pin public output-field contracts and config validation. |
