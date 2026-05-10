# Preprocessing And Conversion

## `preprocess.save_dataset_presets.main`

- File: `preprocess/save_dataset_presets.py`
- Signature: `main() -> None`
- Purpose and contract: canonical preset-generation CLI. It resolves channels from YAML, applies optional `preset_build` policy, plans output paths, optionally prefilters the source CSV by required masks, then instantiates `PSGPretrainDataset` so validation and writing happen through the dataset layer.
- Important inputs/outputs: CLI args in; preset pickle files out.
- Side effects: creates parent directories, may write temporary filtered CSVs, and writes preset files unless `--dry-run` is set.
- Key callers/callees: called from `__main__`; callees include `_load_model_channels`, `_load_preset_build_block`, `_resolve_validation_channels`, `_resolve_effective_min_channels`, `_build_preset_job`, and `PSGPretrainDataset`.
- Reuse guidance: use this CLI path to generate preset pickles.
- Duplication risk notes: do not pickle `SampleIndex` lists directly in one-off scripts unless the preset schema is intentionally changing.

## `preprocess.save_dataset_presets._load_preset_build_block`

- File: `preprocess/save_dataset_presets.py`
- Signature: `_load_preset_build_block(config_data: dict[str, Any]) -> tuple[list[str] | None, int | None]`
- Purpose and contract: parse the optional YAML `preset_build` contract and require that `required_channels` and `min_channels` appear together when the block is present.
- Important inputs/outputs: raw config mapping in; `(required_channels, min_channels)` out.
- Side effects: none.
- Key callers/callees: callers are `main`, `_resolve_channels_and_dims`, and `utils/check_configs.py`.
- Reuse guidance: use this helper rather than reading `preset_build` ad hoc in new tooling.
- Duplication risk notes: field validation belongs here.

## `preprocess.save_dataset_presets._resolve_validation_channels`

- File: `preprocess/save_dataset_presets.py`
- Signature: `_resolve_validation_channels(*, model_channels: list[str], channel_input_dims: dict[str, int], preset_required_channels: list[str] | None, selected_channels: list[str] | None) -> tuple[list[str], dict[str, int]]`
- Purpose and contract: choose the effective validation-channel set from YAML or CLI, auto-inject `stage5` when built-in `ahi` is requested, and provide effective input dims for both declared and built-in channels.
- Important inputs/outputs: model channels, dims, and optional overrides in; ordered channel list plus effective dims out.
- Side effects: none.
- Key callers/callees: callers are `main`, `_resolve_channels_and_dims`, and `utils/check_configs.py`.
- Reuse guidance: use this helper whenever preset-generation code needs to reason about strict required channels.
- Duplication risk notes: built-in channel admission rules should stay centralized here.

## `preprocess.save_dataset_presets._resolve_effective_min_channels`

- File: `preprocess/save_dataset_presets.py`
- Signature: `_resolve_effective_min_channels(*, channel_names: Sequence[str], cli_min_channels: int, preset_min_channels: int | None) -> int`
- Purpose and contract: resolve the effective minimum-channel requirement, forcing full-channel admission for built-in `ahi`.
- Important inputs/outputs: channel list and min-channel candidates in, effective minimum out.
- Side effects: none.
- Key callers/callees: callers are `main` and `utils/check_configs.py`.
- Reuse guidance: use this helper instead of open-coding `ahi`-specific minimum logic.
- Duplication risk notes: `ahi` full-channel enforcement belongs here.

## `preprocess.save_dataset_presets._filter_index_df_for_required_channels`

- File: `preprocess/save_dataset_presets.py`
- Signature: `_filter_index_df_for_required_channels(df: pd.DataFrame, required_channels: list[str]) -> pd.DataFrame`
- Purpose and contract: strictly prefilter an index dataframe by required mask columns, using generic mask lookup plus built-in channel mask rules.
- Important inputs/outputs: dataframe and required channels in, filtered dataframe out.
- Side effects: none.
- Key callers/callees: caller is `_build_preset_job`; callee is `normalize_mask_frame`.
- Reuse guidance: use this when strict preset generation should fail before runtime sample validation.
- Duplication risk notes: required-mask semantics must stay aligned with split-time mask normalization.

## `preprocess.save_dataset_presets._build_preset_job`

- File: `preprocess/save_dataset_presets.py`
- Signature: `_build_preset_job(*, output_path: Path, index_paths: list[str], channel_names: list[str], channel_input_dims: dict[str, int], split: str, meta_data_name: str | None, n_tokens: int, stride_tokens: int, mask_rate: float, allow_missing_channels: bool, min_channels: int, batch_size: int, shuffle: bool, filter_max_workers: int | None) -> tuple[Path, int]`
- Purpose and contract: execute one preset-build unit, including optional strict CSV prefiltering and post-build restoration of the original `source` field.
- Important inputs/outputs: preset-build job inputs in, `(output_path, dataset_len)` out.
- Side effects: may write a temporary CSV, instantiates `PSGPretrainDataset`, and writes a preset pickle.
- Key callers/callees: caller is `main`; callees include `_filter_index_df_for_required_channels`, `_restore_preset_source`, and `PSGPretrainDataset`.
- Reuse guidance: use this helper for parallel preset-build orchestration.
- Duplication risk notes: source restoration after strict prefiltering is easy to forget; keep it centralized here.

## `preprocess.save_dataset_presets._infer_dataset_name` and `_render_output_path`

- File: `preprocess/save_dataset_presets.py`
- Signatures:
  - `_infer_dataset_name(index_paths: list[Path]) -> str`
  - `_render_output_path(output_template: str, dataset_name: str, split: str, n_tokens: int, meta_data_name: str | None) -> Path`
- Purpose and contract: derive stable dataset stems and output paths from CLI inputs and supported template fields.
- Important inputs/outputs: input paths/template values in, dataset name or rendered `Path` out.
- Side effects: none.
- Key callers/callees: callers are `main` and the preset-build loop.
- Reuse guidance: use for preset output naming.
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

## `preprocess.convert_npz_to_kaldi.convert`

- File: `preprocess/convert_npz_to_kaldi.py`; package-local mirrors: `sleep2vec2/preprocess/convert_npz_to_kaldi.py`, `sleep2expert/preprocess/convert_npz_to_kaldi.py`
- Signature: `convert(args: argparse.Namespace) -> Path`
- Purpose and contract: convert CSV-indexed NPZ token windows into split-specific Kaldi ark/scp files plus `manifest.json` format v2 and `manifests/{split}.csv`, preserving sample metadata, cropped `token_start`/`token_end`, available-channel lists, and built-in `ahi`/`tst` metadata when requested. Optional `--ark-shards` writes numbered shard ark/scp files while preserving one aggregate `{channel}.scp` for runtime readers; `--num-workers` parallelizes record conversion while keeping Kaldi writes deterministic on the main thread.
- Important inputs/outputs: index CSV(s), config YAML, output directory, selected channels, windowing settings, missing-channel policy, optional ark shard count, worker count, and optional path-prefix maps in; Kaldi manifest paths out.
- Side effects: reads NPZ files, writes sorted split-specific Kaldi ark/scp files or shard files plus aggregate scps and manifests, and imports optional `kaldi_native_io` only when conversion runs.
- Key callers/callees: called by `main`; callees include `_build_channel_registry`, `load_npz`, `load_builtin_ahi_metadata`, `window`, and `normalize_mask_frame`.
- Reuse guidance: use this CLI to build Kaldi roots for pretrain, finetune, and inference instead of creating ad hoc ark/scp writers. Standalone variants should use their package-local converter.
- Duplication risk notes: standalone variants must use their package-local converter so tokenizer/extractor semantics come from the same recipe namespace.

## `preprocess.split_index_by_dataset.normalize_mask_frame`

- File: `preprocess/split_index_by_dataset.py`
- Signature: `normalize_mask_frame(df: pd.DataFrame, mask_cols: list[str]) -> pd.DataFrame`
- Purpose and contract: normalize raw mask columns to boolean presence using the repository truthy set (`1`, `1.0`, `true`, `t`, `yes`).
- Important inputs/outputs: dataframe and mask column names in, normalized boolean frame out.
- Side effects: none.
- Key callers/callees: callers are `compute_available_channels`, `find_missing_global_pair_coverage`, and `save_dataset_presets._filter_index_df_for_required_channels`.
- Reuse guidance: this is the canonical `_mask` normalization rule.
- Duplication risk notes: keep aligned with `mask_missing_stats.py`.

## `preprocess.split_index_by_dataset.assign_splits_by_dataset`

- File: `preprocess/split_index_by_dataset.py`
- Signature: `assign_splits_by_dataset(df: pd.DataFrame, seed: int, shuffle: bool, n_val: int = 20, n_test: int = 20) -> tuple[pd.Series, dict[str, dict[str, int]]]`
- Purpose and contract: assign train/val/test splits independently per dataset group using the current fixed-count policy.
- Important inputs/outputs: dataframe and RNG settings in; split series plus per-dataset counts out.
- Side effects: none.
- Key callers/callees: caller is `main`; callee is `split_sizes`.
- Reuse guidance: this is the canonical split-allocation policy.
- Duplication risk notes: if allocation rules change, update this helper and documentation together.

## `preprocess.split_index_by_dataset.find_missing_global_pair_coverage`

- File: `preprocess/split_index_by_dataset.py`
- Signature: `find_missing_global_pair_coverage(df: pd.DataFrame, split: pd.Series, mask_cols: list[str]) -> dict[str, list[str]]`
- Purpose and contract: detect feasible modality pairs that exist globally but are missing from `val` or `test`.
- Important inputs/outputs: dataframe, assigned splits, and mask columns in; missing-pair summary out.
- Side effects: none.
- Key callers/callees: caller is `main`; callee is `normalize_mask_frame`.
- Reuse guidance: use this helper when split-generation tooling needs pair-coverage diagnostics.
- Duplication risk notes: pair-coverage logic should stay centralized here.

## `preprocess.split_index_by_dataset.main`

- File: `preprocess/split_index_by_dataset.py`
- Signature: `main() -> None`
- Purpose and contract: read a source CSV, optionally filter by minimum available channels, mark external datasets, assign splits, optionally check pair coverage, and write a new CSV.
- Important inputs/outputs: input/output CSV paths in; rewritten CSV out.
- Side effects: writes CSV and prints summary statistics or warnings.
- Key callers/callees: callees include `get_channel_mask_columns`, `compute_available_channels`, `compute_external_mask`, `assign_splits_by_dataset`, and `find_missing_global_pair_coverage`.
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

## `preprocess.watchpat_zzp_to_edf.infer_stream_layout`, `infer_channel_mapping`, and `build_signals`

- File: `preprocess/watchpat_zzp_to_edf.py`
- Signatures:
  - `infer_stream_layout(payload, marker_positions, probe_seconds: int = 20) -> StreamLayout`
  - `infer_channel_mapping(frames, spo2, low_rate_channels)`
  - `build_signals(decoded, mapping, include_internal_1hz, include_pulse_rate)`
- Purpose and contract: infer WatchPAT frame layout, map decoded channels to semantic signals, and convert them into EDF-ready `SignalSpec` records.
- Important inputs/outputs: decoded payload state in, layout/mapping/signals out.
- Side effects: `build_signals` may derive additional pulse-rate signals.
- Key callers/callees: caller chain is `convert_zzp_to_edf` -> `decode_sleep_dat` / `infer_channel_mapping` / `build_signals`.
- Reuse guidance: keep these heuristics inside the WatchPAT conversion pipeline.
- Duplication risk notes: signal layout and physiological channel mapping are specialized and partially heuristic; avoid cloning them outside this module.
