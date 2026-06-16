# Module Map

## Natural Edit Boundaries

| Boundary | Primary files | Responsibility | Reuse guidance |
| --- | --- | --- | --- |
| Config schema | `hypnodata/config.py` | Strict YAML parsing, typed config dataclasses, signal candidates, structured preprocess steps | Add schema checks here, not in pipeline code |
| Discovery and adapters | `hypnodata/discovery.py`, `hypnodata/adapters.py`, `hypnodata/records.py` | Convert glob/CSV/custom sources into `RecordTask` objects and center-specific hooks | Keep center-specific logic in adapters |
| EDF and channel resolution | `hypnodata/edf.py`, `hypnodata/channels.py` | Inventory raw EDF files, read raw signals, select canonical channels | Reuse `resolve_channels`; do not duplicate matching logic |
| Signal preprocessing | `hypnodata/preprocess.py` | Scale, polarity, structured filter/notch, resample, finite check, common truncation | Extend `preprocess_signal` or `truncate_to_common`; do not add parallel preprocess runners |
| Pipeline orchestration | `hypnodata/pipeline.py` | Per-record execution, resume/overwrite/dry-run/crash behavior, QC/failure accounting, progress writes | Keep orchestration here and signal math in `preprocess.py` |
| Output manifests | `hypnodata/manifests.py`, `hypnodata/backends.py`, `hypnodata/status.py` | NPZ path layout, CSV/JSON manifests, mask columns, progress JSON | Preserve downstream names: `path`, `duration`, mask columns |
| Example config and docs | `configs/hypnodata/` | User-facing contract examples and boundaries | Keep examples aligned with `load_config` |
| Contract tests | `tests/test_hypnodata_*.py`, `tests/hypnodata_test_helpers.py` | Pin schema, preprocessing, pipeline outputs, adapter hooks, downstream compatibility, progress/resume behavior | Add focused tests near the owning contract |

## Dependency Flow

- `pipeline.py` depends on config, discovery, adapters, EDF, channel selection,
  preprocessing, backends, manifests, QC, and status.
- `preprocess.py` depends on parsed `FilterStep` / `NotchStep` objects from
  `config.py`; it does not parse YAML mappings.
- `manifests.py` depends on `HypnodataConfig` for signal metadata and mask
  columns, but does not run preprocessing.

## Ownership Notes

- YAML semantic changes belong in `hypnodata/config.py` with config tests.
- Signal math changes belong in `hypnodata/preprocess.py` with direct unit
  tests.
- Output column or manifest semantics belong in `hypnodata/manifests.py` and
  pipeline tests.
- Downstream compatibility with sleep2stat, presets, and Kaldi conversion is
  pinned by dedicated `tests/test_hypnodata_downstream_*.py` files.
