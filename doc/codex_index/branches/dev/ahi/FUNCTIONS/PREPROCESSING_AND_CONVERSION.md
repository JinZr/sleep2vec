# Preprocessing And Conversion

## `preprocess.save_dataset_presets.main`

- File: `preprocess/save_dataset_presets.py`
- Signature: `main() -> None`
- Purpose and contract: canonical preset-generation CLI. It resolves dataset name, split variants, metadata variants, and output paths, then instantiates `PSGPretrainDataset` with `save_preset_path` so preset validation and writing happen through the dataset layer.
- Important inputs/outputs: CLI args in; preset pickle files out.
- Side effects: creates parent directories and writes preset files unless `--dry-run` is set.
- Key callers/callees: called from `__main__`; callees include `_infer_dataset_name`, `_resolve_meta_names`, `_render_output_path`, and `PSGPretrainDataset`.
- Reuse guidance: use this CLI path to generate preset pickles.
- Duplication risk notes: do not pickle `SampleIndex` lists directly in one-off scripts unless the preset schema is intentionally changing.

## `preprocess.save_dataset_presets._infer_dataset_name`

- File: `preprocess/save_dataset_presets.py`
- Signature: `_infer_dataset_name(index_paths: list[Path]) -> str`
- Purpose and contract: derive a stable dataset stem from one input CSV path, or return `multi` when multiple CSVs are combined.
- Important inputs/outputs: index path list in, dataset name string out.
- Side effects: none.
- Key callers/callees: caller is `main`.
- Reuse guidance: use for preset output naming.
- Duplication risk notes: dataset-stem cleanup rules belong here.

## `preprocess.save_dataset_presets._render_output_path`

- File: `preprocess/save_dataset_presets.py`
- Signature: `_render_output_path(output_template: str, dataset_name: str, split: str, n_tokens: int, meta_data_name: str | None) -> Path`
- Purpose and contract: render one preset output path from the template fields supported by the CLI.
- Important inputs/outputs: template inputs in, expanded `Path` out.
- Side effects: none.
- Key callers/callees: caller is `main`.
- Reuse guidance: use this helper if preset naming behavior changes.
- Duplication risk notes: template-field validation should stay centralized here.

## `preprocess.merge_dataset_presets.main`

- File: `preprocess/merge_dataset_presets.py`
- Signature: `main() -> None`
- Purpose and contract: merge multiple preset pickle files into one by concatenating their top-level lists.
- Important inputs/outputs: input preset paths and output path in; merged preset file out.
- Side effects: writes one pickle file.
- Key callers/callees: callees include `_load_preset`, `_validate_items`, `_flatten`.
- Reuse guidance: use for preset concatenation after separate generation passes.
- Duplication risk notes: this utility trusts all list items to be schema-compatible.

## `preprocess.split_index_by_dataset.get_channel_mask_columns`

- File: `preprocess/split_index_by_dataset.py`
- Signature: `get_channel_mask_columns(df: pd.DataFrame) -> list[str]`
- Purpose and contract: detect channel presence columns by the `_mask` suffix, excluding `stage_mask`.
- Important inputs/outputs: dataframe in, mask-column list out.
- Side effects: none.
- Key callers/callees: caller is `main`; downstream caller is `compute_available_channels`.
- Reuse guidance: this is the canonical `_mask` discovery rule for split preparation.
- Duplication risk notes: keep aligned with `mask_missing_stats.py`.

## `preprocess.split_index_by_dataset.compute_available_channels`

- File: `preprocess/split_index_by_dataset.py`
- Signature: `compute_available_channels(df: pd.DataFrame, mask_cols: list[str]) -> pd.Series`
- Purpose and contract: count present channels by summing rows where mask columns equal numeric `1`.
- Important inputs/outputs: dataframe and mask columns in, availability count series out.
- Side effects: none.
- Key callers/callees: caller is `main`.
- Reuse guidance: use when filtering rows by minimum channel availability before preset generation.
- Duplication risk notes: presence semantics must match `mask_missing_stats.py`.

## `preprocess.split_index_by_dataset.assign_splits`

- File: `preprocess/split_index_by_dataset.py`
- Signature: `assign_splits(df: pd.DataFrame, group_key: pd.Series, seed: int, shuffle: bool) -> tuple[pd.Series, dict[str, dict[str, int]]]`
- Purpose and contract: assign train/val/test splits per dataset group with capped 10% val and test allocation.
- Important inputs/outputs: grouped dataframe and RNG settings in; split series plus per-group counts out.
- Side effects: none.
- Key callers/callees: caller is `main`.
- Reuse guidance: this is the canonical split-allocation policy.
- Duplication risk notes: if allocation rules change, update this helper and downstream documentation together.

## `preprocess.split_index_by_dataset.main`

- File: `preprocess/split_index_by_dataset.py`
- Signature: `main() -> None`
- Purpose and contract: read a source CSV, optionally filter by minimum available channels, mark external datasets, assign splits, and write a new CSV.
- Important inputs/outputs: input/output CSV paths in; rewritten CSV out.
- Side effects: writes CSV and prints summary statistics.
- Key callers/callees: callees include `get_channel_mask_columns`, `compute_available_channels`, `compute_external_mask`, and `assign_splits`.
- Reuse guidance: use as the standard split-preparation CLI.
- Duplication risk notes: this is the source of truth for split assignment before preset creation.

## `preprocess.mask_missing_stats.main`

- File: `preprocess/mask_missing_stats.py`
- Signature: `main() -> None`
- Purpose and contract: stream large CSVs, compute missing-channel statistics under the rule `present iff numeric value == 1`, and write four report CSVs.
- Important inputs/outputs: input CSV path and output prefix in; four output CSVs out.
- Side effects: writes CSV reports and prints a human-readable summary.
- Key callers/callees: caller is `__main__`; helper `_prefix_path` normalizes the output prefix.
- Reuse guidance: use for mask-coverage inspection instead of writing one-off analysis scripts.
- Duplication risk notes: its `_mask` semantics must stay aligned with split filtering.

## `preprocess.watchpat_zzp_to_edf.convert_zzp_to_edf`

- File: `preprocess/watchpat_zzp_to_edf.py`
- Signature: `convert_zzp_to_edf(input_path, output_path, backend, include_internal_1hz, include_pulse_rate, json_summary_path, verbose)`
- Purpose and contract: top-level single-file WatchPAT conversion entrypoint.
- Important inputs/outputs: `.zzp` input plus output controls in; EDF file and optional JSON summary out.
- Side effects: reads archives, writes EDF and optional JSON, may depend on optional external libraries.
- Key callers/callees: callers are `main` and batch-conversion helpers; callees include `decode_sleep_dat`, `infer_channel_mapping`, `build_signals`, `write_edf`, and `build_summary`.
- Reuse guidance: use this function for programmatic conversion instead of duplicating decode/write orchestration.
- Duplication risk notes: this file already centralizes many heuristics; avoid scattering WatchPAT assumptions elsewhere.

## `preprocess.watchpat_zzp_to_edf.decode_sleep_dat`

- File: `preprocess/watchpat_zzp_to_edf.py`
- Signature: `decode_sleep_dat(sleep_dat, metadata, verbose)`
- Purpose and contract: decode Sleep.dat payload structure into timing markers, inferred stream layout, and sample arrays suitable for later channel mapping.
- Important inputs/outputs: raw Sleep.dat bytes and metadata in; decoded structure out.
- Side effects: none beyond compute.
- Key callers/callees: caller is `convert_zzp_to_edf`; callees include `_find_valid_markers`, `_estimate_pre_marker_frames`, `infer_stream_layout`, and `_parse_full_second_segment`.
- Reuse guidance: use when conversion work needs decoded Sleep.dat signals before EDF packaging.
- Duplication risk notes: Sleep.dat layout inference is specialized and should not be reimplemented casually.

## `preprocess.watchpat_zzp_to_edf.infer_stream_layout`

- File: `preprocess/watchpat_zzp_to_edf.py`
- Signature: `infer_stream_layout(payload, marker_positions, probe_seconds: int = 20) -> StreamLayout`
- Purpose and contract: infer high-rate and low-rate channel layout heuristically from timing markers and frame counts.
- Important inputs/outputs: raw payload and marker positions in; `StreamLayout` out.
- Side effects: none.
- Key callers/callees: caller is `decode_sleep_dat`.
- Reuse guidance: use for WatchPAT frame-layout inference only.
- Duplication risk notes: layout heuristics are domain-specific; if they change, keep all downstream channel-mapping assumptions in sync.

## `preprocess.watchpat_zzp_to_edf.infer_channel_mapping`

- File: `preprocess/watchpat_zzp_to_edf.py`
- Signature: `infer_channel_mapping(frames, spo2, low_rate_channels)`
- Purpose and contract: infer semantic channel mapping from decoded frame arrays and low-rate channels.
- Important inputs/outputs: decoded frames in; mapping object out.
- Side effects: none.
- Key callers/callees: caller is `convert_zzp_to_edf`.
- Reuse guidance: use only within the WatchPAT conversion pipeline.
- Duplication risk notes: exact physiological mapping heuristics are specialized and partially heuristic; avoid cloning them outside this module.

## `preprocess.watchpat_zzp_to_edf.build_signals`

- File: `preprocess/watchpat_zzp_to_edf.py`
- Signature: `build_signals(decoded, mapping, include_internal_1hz, include_pulse_rate)`
- Purpose and contract: convert decoded samples plus inferred mapping into EDF-ready `SignalSpec` records.
- Important inputs/outputs: decoded conversion state in; signal list out.
- Side effects: may derive additional pulse-rate signals.
- Key callers/callees: caller is `convert_zzp_to_edf`; may call `derive_pulse_rate`.
- Reuse guidance: use for EDF packaging after decode/mapping.
- Duplication risk notes: keep signal labeling and sample-rate decisions centralized here.
