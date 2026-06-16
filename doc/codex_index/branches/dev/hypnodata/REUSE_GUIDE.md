# Reuse Guide

## Canonical Implementations

| Responsibility | Canonical implementation | Do not bypass with |
| --- | --- | --- |
| Load hypnodata YAML | `hypnodata.config.load_config` | Hand-written YAML parsing in CLI or pipeline code |
| Parse signal candidates | `hypnodata.config._build_candidates` | Re-reading candidate dicts outside config parsing |
| Parse structured preprocess steps | `hypnodata.config._build_preprocess_steps` plus `FilterStep` and `NotchStep` | String step aliases or legacy `not_implemented` markers |
| Apply filter/notch steps | `hypnodata.preprocess.preprocess_signal` | Per-adapter filter code or pipeline-local signal math |
| Align signal duration | `hypnodata.preprocess.truncate_to_common` | Independent truncation in backends or manifests |
| Resolve channels | `hypnodata.channels.resolve_channels` | New label/regex matching loops |
| Write NPZ records | `hypnodata.backends.write_npz_record` | Direct `np.savez` calls from pipeline branches |
| Write public manifests | `hypnodata.manifests.write_manifests` | Ad hoc CSV writers |
| Run record conversion | `hypnodata.pipeline.run_pipeline` | Separate orchestration entrypoints |

## Structured Preprocess Rules

- Use `FilterStep(method, order, lowcut, highcut)` for NeuroKit2 filtering.
- Use `NotchStep(freq, q)` for explicit SciPy notch filtering.
- Reject bare string steps at the config boundary.
- Keep `finite_check` and `truncate_to_common` as fixed internal steps.

## Duplication Risks

1. Do not add center-specific filter defaults by `kind`; filter semantics must
   stay explicit in YAML.
2. Do not copy `/Users/zrjin/git/wuji` worker-state patterns into hypnodata.
   Borrow signal-processing ideas only.
3. Do not write Kaldi directly from hypnodata. Use
   `preprocess/convert_npz_to_kaldi.py` on the standardized NPZ manifest.
4. Do not add `duration_sec` aliases to `record_manifest.csv`; downstream uses
   `duration`.
5. Do not introduce `schema_version` or `version_schema` in hypnodata configs.
