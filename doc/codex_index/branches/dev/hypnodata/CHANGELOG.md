# Changelog

## 2026-06-16

- Simplified raw-signal candidates to ordered exact-label strings and removed
  regex, priority, and adapter scoring from the indexed channel-resolution
  contract.
- Documented built-in hypnodata AHI output: `signals.ahi` writes `ah_event`,
  scalar `ahi`, and scalar `tst` for downstream AHI finetune recipes.
- Refreshed scoped branch index after adding hypnodata annotation/event
  materialization support.
- Refreshed annotation config guidance to use second-based output-grid fields
  (`epoch_sec`, `interval_sec`, `window_sec`) instead of annotation
  `target_sfreq`.
- Documented adapter-owned annotation sources and annotation-only signal kinds:
  `stage`, `event_table`, `event_dense`, and `event_anchor`.
- Indexed standard event-row helpers, table/dense/anchor materializers, and
  stage-aware event filtering.
- Initialized scoped branch index for `dev/hypnodata`.
- Indexed branch-local `hypnodata` config, preprocessing, pipeline, manifest,
  docs, and tests.
- Documented structured preprocess steps:
  - `filter` through NeuroKit2
  - `notch` through SciPy `iirnotch` and `filtfilt`
- Documented that fixed internal steps such as finite checks and common-duration
  truncation are not YAML preprocess entries.
