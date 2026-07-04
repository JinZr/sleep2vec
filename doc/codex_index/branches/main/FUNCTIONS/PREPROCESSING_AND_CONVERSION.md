# Preprocessing And Conversion

## `utils.check_configs.check_config_file`

- File: `utils/check_configs.py`
- Signature: `check_config_file(path: Path) -> None`
- Purpose and contract: validate one repository YAML file through the correct config parser. Model configs route to the root or variant config/preset helpers; sleep2stat-shaped YAML and files under `configs/sleep2stat/` route to `sleep2stat.config.load_config`.
- Important inputs/outputs: config path in; raises on invalid config and returns `None` on success.
- Side effects: imports the selected parser modules.
- Key callers/callees: caller is `utils.check_configs.main`; callees include `sleep2vec.config.load_*`, package-local variant config helpers, preset-build helpers, and `sleep2stat.config.load_config`.
- Reuse guidance: use this checker for repo config validation instead of writing shell loops over config directories.
- Duplication risk notes: do not duplicate sleep2stat schema checks here; the branch is only a router to `sleep2stat.config.load_config`.

## `preprocess.save_dataset_presets.main`

- File: `preprocess/save_dataset_presets.py`
- Signature: `main() -> None`
- Purpose and contract: canonical preset-generation CLI. It resolves channels and optional NPZ-key aliases from YAML, applies optional `preset_build` policy, loads survival preset-build sidecars when `finetune.task.type=survival`, plans output paths, optionally prefilters the source CSV by required masks, then instantiates `PSGPretrainDataset` so validation and writing happen through the dataset layer.
- Important inputs/outputs: CLI args in; preset pickle files out.
- Side effects: creates parent directories, may write temporary filtered CSVs, and writes preset files unless `--dry-run` is set.
- Key callers/callees: called from `__main__`; callees include `_load_model_channels`, `_load_model_channel_aliases`, `_load_preset_build_block`, `_load_survival_build_config`, `_resolve_validation_channels`, `_resolve_effective_min_channels`, `_build_preset_job`, and `PSGPretrainDataset`.
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

## `preprocess.save_dataset_presets._load_model_channel_aliases`

- File: `preprocess/save_dataset_presets.py`; package-local mirrors: `sleep2vec2/preprocess/save_dataset_presets.py`, `sleep2expert/preprocess/save_dataset_presets.py`
- Signature: `_load_model_channel_aliases(config_data: dict[str, Any]) -> dict[str, list[str]]`
- Purpose and contract: parse optional YAML `model.channels[*].aliases` as NPZ input-key fallbacks for preset raw validation only.
- Important inputs/outputs: raw config mapping in; mapping of configured channel name to alias NPZ keys out.
- Side effects: none.
- Key callers/callees: caller is `main`; consumers pass the result into `_build_preset_job` and `PSGPretrainDataset`.
- Reuse guidance: use this helper when preset generation needs the same YAML alias semantics as runtime NPZ loading.
- Duplication risk notes: aliases do not affect mask columns, Kaldi manifests, or converter output channel names.

## `preprocess.save_dataset_presets._load_survival_build_config`

- File: `preprocess/save_dataset_presets.py`; package-local mirrors: `sleep2vec2/preprocess/save_dataset_presets.py`, `sleep2expert/preprocess/save_dataset_presets.py`
- Signature: `_load_survival_build_config(config_data: dict[str, Any]) -> tuple[Any | None, int | None]`
- Purpose and contract: derive the survival sidecar config and output dimension for preset generation from YAML, requiring `finetune.survival` only when `finetune.task.type=survival`.
- Important inputs/outputs: raw config mapping in; typed survival config plus output dim, or `(None, None)` for non-survival configs.
- Side effects: none.
- Key callers/callees: caller is `main`; consumers pass the result into `_build_preset_job` and `PSGPretrainDataset`.
- Reuse guidance: use this helper when preset generation or config checks need survival sidecar awareness.
- Duplication risk notes: do not silently accept `finetune.survival` on non-survival tasks.

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
- Signature: `_build_preset_job(*, output_path: Path, index_paths: list[str], channel_names: list[str], channel_input_dims: dict[str, int], split: str, meta_data_name: str | None, n_tokens: int, stride_tokens: int, mask_rate: float, allow_missing_channels: bool, min_channels: int, batch_size: int, shuffle: bool, filter_max_workers: int | None, survival_label_config: Any | None = None, survival_output_dim: int | None = None, channel_aliases: Mapping[str, Sequence[str]] | None = None) -> tuple[Path, int]`
- Purpose and contract: execute one preset-build unit, including optional strict CSV prefiltering and post-build restoration of the original `source` field.
- Important inputs/outputs: preset-build job inputs plus optional survival sidecar config/output dim in, `(output_path, dataset_len)` out.
- Side effects: may write a temporary CSV, instantiates `PSGPretrainDataset`, and writes a preset pickle.
- Key callers/callees: caller is `main`; callees include `_filter_index_df_for_required_channels`, `_restore_preset_source`, and `PSGPretrainDataset`.
- Reuse guidance: use this helper for parallel preset-build orchestration.
- Duplication risk notes: source restoration after strict prefiltering is easy to forget; keep it centralized here.

## `preprocess.save_dataset_presets._preset_manifest_payload`

- File: `preprocess/save_dataset_presets.py`
- Signature: `_preset_manifest_payload(*, output_path: Path, config_path: Path, index_paths: list[Path], dataset_name: str, split: str, n_tokens: int, stride_tokens: int, channels: list[str], allow_missing_channels: bool, min_channels: int, meta_data_name: str | None, sample_count: int) -> dict`
- Purpose and contract: build the sidecar manifest payload for a generated preset pickle, including input paths, split/window policy, required channels, sample count, available-channel counts, and source counts.
- Important inputs/outputs: generated preset path and build metadata in; JSON-serializable manifest mapping out.
- Side effects: reads the generated preset pickle to summarize `payload["available_channels"]` and metadata source counts.
- Key callers/callees: caller is `main`; callee is `_summarize_preset_items`.
- Reuse guidance: use this helper when preset manifests need the same schema as the CLI sidecars.
- Duplication risk notes: sidecar manifest schema should stay aligned with `doc/agent_contracts/run_manifest.md`.

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

## `preprocess.convert_npz_to_kaldi.convert`

- File: `preprocess/convert_npz_to_kaldi.py`; package-local mirrors: `sleep2vec2/preprocess/convert_npz_to_kaldi.py`, `sleep2expert/preprocess/convert_npz_to_kaldi.py`
- Signature: `convert(args: argparse.Namespace) -> Path`
- Purpose and contract: convert CSV-indexed NPZ token windows into split-specific Kaldi ark/scp files plus `manifest.json` format v2. Before opening writers, it preflights index-derived sample keys after split filtering, per-split overlap stride selection, survival key preservation, and mask-based availability filtering, rejecting duplicate keys before partial ark/scp output is created. When overlapping windows are requested, `val`/`test` rows are retained but use non-overlapping stride unless `--include-overlap-eval-splits` is passed. The converter defaults to `--compress-ark`, compressing non-built-in signal channels in the `train` split with Kaldi `CompressedMatrixWriter` and `CompressionMethod.kTwoByteAuto`, while built-in `stage5` and `ahi` plus non-train splits stay uncompressed `FloatMatrix` entries. `--no-compress-ark` forces all channel ark files to `FloatMatrix`.
- Important inputs/outputs: index CSV(s), config YAML, output directory, optional split filter, selected channels, windowing settings, missing-channel policy, shard count, worker count, path-prefix maps, survival key column, and compression switch in; Kaldi data root with split manifests, channel scps, ark files, and manifest channel `ark_storage` metadata out.
- Side effects: reads NPZ files and writes sorted split-specific Kaldi ark/scp files or shard files plus aggregate scps and streamed split manifests; imports optional `kaldi_native_io` only after index preflight passes.
- Key callers/callees: called from `__main__`; callees include `_resolve_channels`, `_validate_unique_sample_keys`, `_convert_record`, `kaldi_native_io.FloatMatrixWriter`, and `kaldi_native_io.CompressedMatrixWriter`.
- Reuse guidance: use this CLI to build Kaldi roots for pretrain, finetune, and inference instead of creating ad hoc ark/scp writers. Standalone variants should keep their package-local mirrors behaviorally aligned.
- Duplication risk notes: keep root, `sleep2vec2`, and `sleep2expert` converter storage semantics in sync when changing writer behavior.

## `preprocess.mask_missing_stats.main`

- File: `preprocess/mask_missing_stats.py`
- Signature: `main() -> None`
- Purpose and contract: stream large CSVs, compute missing-channel statistics with split-compatible truthy mask values (`1`, `1.0`, `true`, `t`, `yes`), and write four report CSVs.
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

## `utils.cut_ukb_sleep_with_asleep.main`

- File: `utils/cut_ukb_sleep_with_asleep.py`
- Signature: `main()`
- Purpose and contract: standalone CLI for cutting nightly UKB `.cwa` accelerometer segments with the external pip-installed `asleep` package. It recursively finds `.cwa` files, mirrors the input tree under the output directory, runs asleep parsing and sleep-window detection, keeps only the longest sleep block per UK local noon-to-noon interval, and writes per-night compressed NPZ files plus CSV manifests.
- Important inputs/outputs: input `.cwa` file or directory and output directory in; per-night `.npz` files with local `time` and raw `device_time`, per-file `night_sleep_blocks.csv`, asleep cache files, and root `manifest.csv` out. `--time-shift auto` keeps asleep on device time and applies dynamic `Europe/London` offsets per timestamp; explicit numeric shifts keep the fixed-shift legacy path.
- Side effects: creates output directories, may download asleep model weights through asleep when requested, and may remove per-file asleep caches when `--remove-cache` is passed.
- Key callers/callees: called from `__main__`; callees include `asleep.get_sleep.get_parsed_data`, `asleep.get_sleep.transform_data2model_input`, and `asleep.get_sleep.get_sleep_windows`.
- Reuse guidance: use this utility for UKB CWA night extraction instead of adding sleep2vec-dependent cutting scripts.
- Duplication risk notes: this is intentionally outside the sleep2vec preset and Kaldi conversion contracts; downstream conversion should consume its NPZ manifest rather than mixing asleep logic into `preprocess/save_dataset_presets.py`.

## `utils.parse_ukb_annotations_by_person.main`

- File: `utils/parse_ukb_annotations_by_person.py`
- Signature: `main() -> None`
- Purpose and contract: parse UK Biobank annotation exports into a derived bundle with dataset metadata, column/coding tables, withdrawal lists, per-participant JSON files, manifest, and README.
- Important inputs/outputs: raw UKB root or `annotations/` directory plus output root in; `datasets/<dataset_id>/` CSV/JSONL metadata, `participants/<eid_prefix>/<eid>/<dataset_id>.json`, `withdrawals/withdrawn_eids.csv`, `manifest.json`, and README out.
- Side effects: creates derived output directories, writes many CSV/JSON/README files, and deletes stale participant JSON files for the current dataset before rewriting.
- Key callers/callees: called from `__main__`; callees include `resolve_annotation_root`, `parse_html_dictionary`, `parse_r_codings`, `build_dataset_outputs`, `parse_withdrawals`, `parse_participants`, `write_manifest`, and `write_readme`.
- Reuse guidance: use this parser when downstream pipelines need stable ML feature names and participant-level JSON rather than re-parsing `.tab` exports manually.
- Duplication risk notes: UDI parsing, HTML dictionary parsing, coding-table extraction, and retry-on-EIO participant writes are centralized here.

## `utils.parse_ukb_annotations_by_person.build_dataset_outputs`

- File: `utils/parse_ukb_annotations_by_person.py`
- Signature: `build_dataset_outputs(annotation_root: Path, output_root: Path, tab_path: Path) -> tuple[dict, list[dict]]`
- Purpose and contract: derive dataset-level schema artifacts for one UKB `.tab` file from companion dictionary/coding files when available, falling back to header-only metadata when not.
- Important inputs/outputs: annotation root, output root, and `.tab` path in; dataset metadata plus normalized column records out.
- Side effects: writes `columns.csv`, `columns.jsonl`, `fields.csv`, `codings.csv`, and `missing_codings.csv`.
- Key callers/callees: caller is `main`; callees include `parse_html_dictionary`, `parse_header_only`, `parse_r_codings`, `normalize_columns`, and `build_field_summary`.
- Reuse guidance: use for dataset-level schema generation before participant JSON parsing.
- Duplication risk notes: dictionary/coding fallback behavior belongs here.

## `utils.parse_ukb_annotations_by_person.parse_participants`

- File: `utils/parse_ukb_annotations_by_person.py`
- Signature: `parse_participants(output_root: Path, tab_path: Path, dataset_id: str, columns: list[dict], withdrawn_eids: set[str], exclude_withdrawn: bool, limit_rows: int | None) -> dict`
- Purpose and contract: stream one `.tab` file and write per-participant JSON values using the normalized feature names from the dataset schema.
- Important inputs/outputs: output root, `.tab` path, dataset id, normalized columns, withdrawals, and optional row limit in; participant metadata summary out.
- Side effects: removes stale participant JSON files for that dataset and writes `participants/<eid_prefix>/<eid>/<dataset_id>.json`.
- Key callers/callees: caller is `main`; callees include `validate_tab_header`, `participant_path`, and `write_participant_json`.
- Reuse guidance: use this when regenerating participant JSON for a parsed dataset.
- Duplication risk notes: missing-value omission, withdrawal handling, and EIO retry behavior should stay centralized here.

## `utils.collect_ukb_demographics.main`

- File: `utils/collect_ukb_demographics.py`
- Signature: `main() -> None`
- Purpose and contract: collect `eid`, sex, and age fields from UKB-style participant JSON files, recording source keys for each derived value.
- Important inputs/outputs: JSON root, output CSV path, optional glob pattern, and optional dedupe-by-eid flag in; one-row-per-JSON or one-row-per-eid CSV out.
- Side effects: reads JSON files recursively and writes the destination CSV.
- Key callers/callees: called from `__main__`; callees include `extract_record`, `get_values`, `first_present`, `find_fallback_key`, and `compute_age_from_birth_and_assessment`.
- Reuse guidance: use after `parse_ukb_annotations_by_person.py` or equivalent UKB JSON export when age/sex covariates are needed.
- Duplication risk notes: source-key provenance for sex and age should remain here instead of being reconstructed in spreadsheets.

## `utils.collect_ukb_demographics.extract_record`

- File: `utils/collect_ukb_demographics.py`
- Signature: `extract_record(path: Path, root: Path) -> dict[str, str]`
- Purpose and contract: extract one demographic row from a UKB-style participant JSON, including `eid`, `dataset_id`, sex code/label, age, JSON path, relative path, and source fields.
- Important inputs/outputs: JSON path and scan root in; CSV row dictionary out.
- Side effects: reads one JSON file.
- Key callers/callees: caller is `main`; callees include `load_json`, `get_values`, `first_present`, `find_fallback_key`, and `compute_age_from_birth_and_assessment`.
- Reuse guidance: use for programmatic demographic extraction when the CLI is too coarse.
- Duplication risk notes: primary key preference and fallback source tracking belong here.

## `utils.fix_kaldi_index.fix_index`

- File: `utils/fix_kaldi_index.py`
- Signature: `fix_index(df: pd.DataFrame, *, source_field: str) -> tuple[pd.DataFrame, int]`
- Purpose and contract: ensure each row produces a unique Kaldi sample-key prefix by assigning a synthesized `session_id` only to duplicate `(source, record-key)` rows.
- Important inputs/outputs: index dataframe and source-field name in; fixed dataframe plus changed-row count out.
- Side effects: none.
- Key callers/callees: caller is `main`; callees include `_key_prefix`, `_unique_session_id`, `_source_prefix`, and `_record_key_from_row`.
- Reuse guidance: use before `convert_npz_to_kaldi.py` when duplicate sample keys would otherwise fail converter preflight.
- Duplication risk notes: keep sample-key sanitization aligned with converter key construction.

## `utils.fix_kaldi_index.main`

- File: `utils/fix_kaldi_index.py`
- Signature: `main(argv: Sequence[str] | None = None) -> int`
- Purpose and contract: CLI wrapper around `fix_index`; overwrites the input CSV with a `.backup` by default or writes a separate output path when requested.
- Important inputs/outputs: `--index`, optional `--output`, and `--source-field` in; repaired CSV out.
- Side effects: may create `--index.backup`, creates output parent directories, and writes CSV.
- Key callers/callees: called from `__main__`; callee is `fix_index`.
- Reuse guidance: use this CLI for local Kaldi index repair instead of editing `session_id` by hand.
- Duplication risk notes: this is a data-prep repair utility, not a runtime fallback for duplicate keys.

## `utils.match_case_controls.main`

- File: `utils/match_case_controls.py`
- Signature: `main(argv=None) -> None`
- Purpose and contract: build matched case-control cohorts from a flat CSV with exact matching, hard calipers, propensity-score features, optional genetic-style weight search, and balance diagnostics.
- Important inputs/outputs: input CSV, matched output path, case/id/covariate settings, exact columns, calipers, ratio, minimum controls, optional quality gates, and output paths in; matched, unmatched cases, excluded rows, case match counts, and balance CSVs out.
- Side effects: writes all requested/default CSV outputs and exits nonzero for configured quality failures such as unmatched cases or SMD threshold violations.
- Key callers/callees: called from `__main__`; callees include `prepare_rows`, `build_design_matrices`, `estimate_propensity_scores`, `optimize_weight_matrix`, `match_cases`, `compute_balance`, and `write_csv`.
- Reuse guidance: use this script for reproducible cohort matching rather than maintaining matching notebooks.
- Duplication risk notes: case/control string handling, NA-token preservation, caliper semantics, and balance output schema are covered by tests and should not be cloned.

## `utils.match_case_controls.match_cases`

- File: `utils/match_case_controls.py`
- Signature: `match_cases(df: pd.DataFrame, *, is_case: pd.Series, id_col: str, exact_cols: list[str], calipers: dict[str, float], ratio: int, min_controls_per_case: int, propensity: pd.Series, distance_features: pd.DataFrame, weight_matrix: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int], list[str]]`
- Purpose and contract: greedily match ordered cases to unused controls subject to exact constraints, calipers, ratio, and minimum-control policy.
- Important inputs/outputs: cleaned cohort data, case mask, matching constraints, propensity scores, distance features, and weight matrix in; matched rows, unmatched cases, case counts, matched indices, and matched roles out.
- Side effects: none.
- Key callers/callees: callers are `main` and `optimize_weight_matrix`; callees include `order_cases_like_matchit`, `passes_exact`, `passes_calipers`, and `matching_distance`.
- Reuse guidance: use for deterministic matching once features and weights have been prepared.
- Duplication risk notes: non-reuse of controls and partial-vs-unmatched status logic belongs here.

## `utils.match_case_controls.compute_balance`

- File: `utils/match_case_controls.py`
- Signature: `compute_balance(encoded: pd.DataFrame, matched_indices: list[int], matched_roles: list[str], *, before_is_case: pd.Series, metadata: list[tuple[str, str, str]], max_smd: float | None) -> pd.DataFrame`
- Purpose and contract: compute before/after standardized mean differences for encoded covariates and mark whether each after-match value passes an optional SMD threshold.
- Important inputs/outputs: encoded covariate matrix, matched row indices/roles, pre-match case mask, design metadata, and optional max SMD in; balance dataframe out.
- Side effects: none.
- Key callers/callees: caller is `main`; callees include `smd_denominator` and `smd`.
- Reuse guidance: use for post-match diagnostics that need the same balance schema as the CLI.
- Duplication risk notes: pre-match denominator policy is tested here.

## `utils.match_case_controls.optimize_weight_matrix`

- File: `utils/match_case_controls.py`
- Signature: `optimize_weight_matrix(df: pd.DataFrame, *, is_case: pd.Series, id_col: str, exact_cols: list[str], calipers: dict[str, float], ratio: int, min_controls_per_case: int, propensity: pd.Series, distance_features: pd.DataFrame, base_matrix: np.ndarray, encoded_balance: pd.DataFrame, metadata: list[tuple[str, str, str]], seed: int, genetic_maxiter: int, genetic_popsize: int) -> np.ndarray`
- Purpose and contract: optionally search feature weights with SciPy differential evolution to improve match balance while penalizing unmatched or shortfall cases.
- Important inputs/outputs: matching inputs, base covariance/identity matrix, encoded balance features, metadata, and search controls in; weighted matrix out.
- Side effects: may run a SciPy optimizer.
- Key callers/callees: caller is `main`; callees include `weighted_matrix`, `match_cases`, and `matching_objective`.
- Reuse guidance: use when the default equal-weight distance is not sufficient for balance.
- Duplication risk notes: optimization objective and penalties are part of the matching contract; avoid ad hoc external tuning loops.
