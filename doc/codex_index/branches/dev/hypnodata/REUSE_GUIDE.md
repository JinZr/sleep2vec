# Reuse Guide

## Canonical Implementations

| Responsibility | Canonical implementation | Do not bypass with |
| --- | --- | --- |
| Load hypnodata YAML | `hypnodata.config.load_config` | Hand-written YAML parsing in CLI or pipeline code |
| Parse signal candidates | `hypnodata.config._build_candidates` | Re-reading ordered label lists outside config parsing |
| Parse structured preprocess steps | `hypnodata.config._build_preprocess_steps` plus `FilterStep` and `NotchStep` | String step aliases or legacy `not_implemented` markers |
| Parse annotation-only signal declarations | `hypnodata.config._build_signals` | Source-specific `signals.<name>.annotation` blocks |
| Derive configured output frequency | `hypnodata.config.declared_target_sfreq` | Recomputing `1 / epoch_sec`, `1 / interval_sec`, or `1 / window_sec` in manifests or pipeline code |
| Read standard event CSV rows | `hypnodata.annotations.read_event_csv` | Hand-written `Type/Start/Duration` parsing in adapters |
| Materialize annotation arrays | `hypnodata.annotations.materialize_event_table`, `materialize_dense_events`, `materialize_anchor_events`, and stage readers | Per-adapter label matrix builders |
| Filter events by sleep stage | `hypnodata.annotations.filter_events_to_sleep_stages` | Per-adapter sleep-overlap filtering loops |
| Materialize built-in AHI output | `hypnodata.annotations.materialize_ahi_from_events` | Ad hoc AHI/TST calculations in adapters |
| Apply filter/notch steps | `hypnodata.preprocess.preprocess_signal` | Per-adapter filter code or pipeline-local signal math |
| Align signal duration | `hypnodata.preprocess.truncate_to_common` | Independent truncation in backends or manifests |
| Resolve channels | `hypnodata.channels.resolve_channels` | New label matching or priority logic |
| Write NPZ records | `hypnodata.backends.write_npz_record` | Direct `np.savez` calls from pipeline branches |
| Write public manifests | `hypnodata.manifests.write_manifests` | Ad hoc CSV writers |
| Run record conversion | `hypnodata.pipeline.run_pipeline` | Separate orchestration entrypoints |

## Structured Preprocess Rules

- Use `FilterStep(method, order, lowcut, highcut)` for NeuroKit2 filtering.
- Use `NotchStep(freq, q)` for explicit SciPy notch filtering.
- Reject bare string steps at the config boundary.
- Keep `finite_check` and `truncate_to_common` as fixed internal steps.

## Annotation Rules

- Keep annotation sources adapter-owned; core YAML only declares canonical
  outputs under `signals`.
- Empty `candidates` are reserved for `kind: stage`, `event_table`,
  `event_dense`, `event_anchor`, and built-in `ahi`.
- Use `epoch_sec`, `interval_sec`, or `window_sec` for annotation output grids;
  do not use `target_sfreq` for annotation-only signals.
- Use standard event rows `[type, start_sec, duration_sec]` before producing
  table, dense, or anchor outputs.
- `signals.ahi` is the only built-in clinical summary output in hypnodata; it
  writes `ah_event`, scalar `ahi`, and scalar `tst` for AHI finetune data.
- Hypnodata still must not compute ODI, T90, hypoxic burden, sleep efficiency,
  or unrelated downstream summaries.

## Duplication Risks

1. Do not add center-specific filter defaults by `kind`; filter semantics must
   stay explicit in YAML.
2. Do not copy `/Users/zrjin/git/wuji` worker-state patterns into hypnodata.
   Borrow signal-processing ideas only.
3. Do not copy `/Users/zrjin/git/wuji` event worker output builders into each
   adapter. Use the canonical annotation materializers.
4. Do not write Kaldi directly from hypnodata. Use
   `preprocess/convert_npz_to_kaldi.py` on the standardized NPZ manifest.
5. Do not add `duration_sec` aliases to `record_manifest.csv`; downstream uses
   `duration`.
6. Do not introduce `schema_version` or `version_schema` in hypnodata configs.
