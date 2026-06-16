# Changelog

## 2026-06-16

- Initialized scoped branch index for `dev/hypnodata`.
- Indexed branch-local `hypnodata` config, preprocessing, pipeline, manifest,
  docs, and tests.
- Documented structured preprocess steps:
  - `filter` through NeuroKit2
  - `notch` through SciPy `iirnotch` and `filtfilt`
- Documented that fixed internal steps such as finite checks and common-duration
  truncation are not YAML preprocess entries.
