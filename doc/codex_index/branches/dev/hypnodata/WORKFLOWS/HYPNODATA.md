# Hypnodata Workflows

## Add Or Change Config Fields

1. Update dataclasses and parsing in `hypnodata/config.py`.
2. Reject unknown or duplicate semantic fields at the config boundary.
3. Update `tests/test_hypnodata_config.py`.
4. Update `configs/hypnodata/README.md` and `configs/hypnodata/toy_edf_npz.yaml`
   when the user-facing schema changes.

## Add Or Change Signal Preprocessing

1. Parse user-facing YAML into typed steps in `hypnodata/config.py`.
2. Execute parsed steps in `hypnodata/preprocess.py`.
3. Keep execution order explicit: scale, polarity, structured preprocess list,
   target-rate resampling, finite check, then common-duration truncation.
4. Record real executed steps in `ProcessedSignal.steps`.
5. Add or update direct signal tests in `tests/test_hypnodata_preprocess.py`.
6. Add or update pipeline manifest tests when step names or output metadata
   change.

## Validate Hypnodata Changes

Use the smallest relevant set first, then the full hypnodata glob:

```bash
conda run -n exp python -m pytest -q tests/test_hypnodata_config.py tests/test_hypnodata_preprocess.py tests/test_hypnodata_pipeline_npz.py
conda run -n exp python -m pytest -q tests/test_hypnodata_*.py
conda run -n exp flake8 hypnodata tests/test_hypnodata_*.py
PYTHONPYCACHEPREFIX=/tmp/sleep2vec_pycache conda run -n exp python -m compileall hypnodata tests
git diff --check
```

## Downstream Compatibility

- Use `tests/test_hypnodata_downstream_sleep2stat.py` for sleep2stat record
  loading compatibility.
- Use `tests/test_hypnodata_downstream_presets.py` for preset mask filtering.
- Use `tests/test_hypnodata_downstream_kaldi_converter.py` for NPZ-to-Kaldi
  compatibility.

Hypnodata should not write Kaldi archives or preset pickle files directly.
