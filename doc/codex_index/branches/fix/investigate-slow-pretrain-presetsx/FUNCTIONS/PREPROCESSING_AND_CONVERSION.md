# Preprocessing And Conversion

## `preprocess.save_dataset_presets.main`

- File: `preprocess/save_dataset_presets.py`
- Signature: `main() -> None`
- Purpose and contract: canonical preset-generation CLI. It resolves dataset name, split variants, metadata variants, channel validation policy, worker budget, and output paths, then instantiates `PSGPretrainDataset` with `save_preset_path` so preset validation and writing happen through the dataset layer, including built-in AHI rejection of samples whose tokenized `ah_event` labels are all ignore-valued. When `--num-workers` is omitted, it preserves automatic validation-worker selection by passing `filter_max_workers=None` in both single-preset and multi-preset builds. When `--num-workers` is explicitly set, that same value is forwarded to each job's inner validation path.
- Important inputs/outputs: CLI args in; preset pickle files out. `--num-workers` is optional; when set, it controls both outer job concurrency and each job's `filter_max_workers` value.
- Side effects: creates parent directories, may create a temporary filtered CSV, writes preset files unless `--dry-run` is set.
- Key callers/callees: called from `__main__`; callees include `_load_preset_build_block`, `_resolve_validation_channels`, `_build_preset_job`, and `PSGPretrainDataset`.
- Reuse guidance: use this CLI path to generate preset pickles.
- Duplication risk notes: do not pickle `SampleIndex` lists directly in one-off scripts unless the preset schema is intentionally changing.

## `preprocess.save_dataset_presets._load_preset_build_block`

- File: `preprocess/save_dataset_presets.py`
- Signature: `_load_preset_build_block(config_data: dict[str, Any]) -> tuple[list[str] | None, int | None]`
- Purpose and contract: parse the optional YAML `preset_build` block and require both `required_channels` and `min_channels` when it is present.
- Important inputs/outputs: raw config mapping in; validated `required_channels` / `min_channels` pair out.
- Side effects: none.
- Key callers/callees: callers are `_resolve_channels_and_dims` and `main`.
- Reuse guidance: keep YAML-driven preset channel policy here.
- Duplication risk notes: do not interpret `preset_build` ad hoc in callers.

## `preprocess.save_dataset_presets._resolve_validation_channels`

- File: `preprocess/save_dataset_presets.py`
- Signature: `_resolve_validation_channels(*, model_channels, channel_input_dims, preset_required_channels, selected_channels) -> tuple[list[str], dict[str, int]]`
- Purpose and contract: derive the effective preset-validation channel set from YAML model channels, optional `preset_build.required_channels`, optional CLI `--channels`, and built-in validation channels such as `stage5` and `ahi`. When `ahi` is selected, `stage5` is appended automatically so built-in AHI presets validate the sleep-stage stream required by final runtime metrics.
- Important inputs/outputs: configured channel sources in; ordered channel list plus resolved input dims out.
- Side effects: none.
- Key callers/callees: callers are `_resolve_channels_and_dims` and `main`.
- Reuse guidance: use this helper for every preset-builder channel-selection change.
- Duplication risk notes: built-in validation-channel handling belongs here, not in wrapper scripts.

## `preprocess.save_dataset_presets._filter_index_df_for_required_channels`

- File: `preprocess/save_dataset_presets.py`
- Signature: `_filter_index_df_for_required_channels(df: pd.DataFrame, required_channels: list[str]) -> pd.DataFrame`
- Purpose and contract: prefilter one index dataframe to rows whose required channel-mask columns are all present, using built-in mask mappings such as `stage_mask` for `stage5` and `ah_event_mask` for `ahi`. For built-in AHI preset generation this means strict mode requires both `ah_event_mask` and `stage_mask`.
- Important inputs/outputs: dataframe plus required channels in; filtered dataframe out.
- Side effects: none.
- Key callers/callees: caller is `_build_preset_job`; callee is `normalize_mask_frame`.
- Reuse guidance: use this when missing channels are disallowed for preset generation.
- Duplication risk notes: mask-column mapping must stay aligned with built-in channel specs.

## `preprocess.save_dataset_presets._build_preset_job`

- File: `preprocess/save_dataset_presets.py`
- Signature: `_build_preset_job(*, output_path, index_paths, channel_names, channel_input_dims, split, meta_data_name, n_tokens, stride_tokens, mask_rate, allow_missing_channels, min_channels, batch_size, shuffle, filter_max_workers) -> tuple[Path, int]`
- Purpose and contract: execute one preset-build unit, optionally materializing a temporary required-mask-filtered CSV before delegating to `PSGPretrainDataset`. `filter_max_workers=None` intentionally leaves sample validation on the dataset-side automatic worker default.
- Important inputs/outputs: one job specification in; output path plus dataset length out.
- Side effects: may create and remove a temporary CSV, writes the preset pickle, and restores original `metadata["source"]` after filtered builds.
- Key callers/callees: caller is `main`; callees include `_resolve_single_index_path`, `_filter_index_df_for_required_channels`, `_restore_preset_source`, and `PSGPretrainDataset`.
- Reuse guidance: keep per-job preset side effects here.
- Duplication risk notes: temporary filtered-index handling should not be reimplemented in parallel job wrappers.

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
- Key callers/callees: callees include `_load_preset`, `_validate_items`, and `_flatten`.
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
