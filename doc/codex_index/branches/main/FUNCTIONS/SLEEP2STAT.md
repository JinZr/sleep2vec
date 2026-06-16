# sleep2stat

This catalog covers `sleep2stat/`, a derived-analysis runtime for per-record and cohort sleep statistics. It is not a trainer. It should reuse existing model, dataset, Kaldi, and agent-tooling contracts instead of creating parallel runtime paths.

## `sleep2stat.config.load_config`

- File: `sleep2stat/config.py`
- Signature: `load_config(path: str | Path) -> Sleep2statConfig`
- Purpose and contract: parse and validate sleep2stat YAML with strict top-level blocks: `run`, `data`, `signals`, `analyzers`, `reducers`, and `outputs`. Semantic data fields such as backend, path/duration/split columns, token length, and max tokens must be explicit in YAML; config-level `run.overwrite` is not supported.
- Important inputs/outputs: config path in; frozen `Sleep2statConfig` dataclass out.
- Side effects: reads YAML.
- Key callers/callees: callers include `sleep2stat.cli.main`, `agent_tools.configs.sleep2stat_config_summary`, and `utils.check_configs.check_config_file`; callees include `_build_run_config`, `_build_data_config`, `_build_signals_config`, `_build_analyzers`, `_validate_backend_analyzer_support`, `_build_reducers`, `_validate_reducer_references`, and `_build_outputs`.
- Reuse guidance: use this as the only schema validation path for sleep2stat configs.
- Duplication-risk notes: do not duplicate stage-source ordering, YASA/Kaldi support checks, analyzer names, reducer names, required sleep2stat fields, or global-table validation in agent tooling.

## `sleep2stat.cli.main`

- File: `sleep2stat/cli.py`
- Signature: `main(argv: list[str] | None = None) -> int`
- Purpose and contract: dispatch sleep2stat subcommands: `validate-config`, `run`, `summarize`, `resume-status`, `repair`, `cohort-finalize`, `plot-record`, and `plot-cohort`; `validate-config --check-records` performs record metadata preflight, and `summarize --num-workers` controls parallel per-record sidecar reads during table rebuild.
- Important inputs/outputs: optional argv in; process exit code out.
- Side effects: prints run paths, status summaries, or summary rows; runs analysis bundles, rebuilds global tables during summarize, repairs status files, finalizes merged cohort tables, and writes plot files through `sleep2stat.plot`.
- Key callers/callees: called by `sleep2stat/__main__.py`; delegates to `load_config`, `load_records`, `run_pipeline`, `_summarize`, `resume_status`, `repair_run_status`, `cohort_finalize`, `plot_record`, and `plot_cohort`.
- Reuse guidance: generated scripts should call this CLI rather than importing pipeline internals directly.
- Duplication-risk notes: keep CLI behavior thin; command planning and safety checks belong in `agent_tools`.

## `sleep2stat.status.resume_status`

- File: `sleep2stat/status.py`
- Signature: `resume_status(run_dir: Path) -> dict[str, Any]`
- Purpose and contract: inspect a run directory by comparing `record_manifest.csv`, per-record `_SUCCESS.json`, `status/failures.csv`, `status/progress.json`, and optional `status/pid.json`.
- Important inputs/outputs: run directory in; structured counts and status such as `running`, `stale_running`, `interrupted`, `completed`, or `completed_with_failures` out.
- Side effects: none.
- Key callers/callees: called by CLI `resume-status` and `repair`; uses writer completion-marker naming.
- Reuse guidance: use this for resume diagnostics instead of manually comparing progress, PIDs, logs, and success markers.
- Duplication-risk notes: do not add separate shard-status scripts that reinterpret the bundle layout.

## `sleep2stat.finalize.cohort_finalize`

- File: `sleep2stat/finalize.py`
- Signature: `cohort_finalize(output_run_dir: Path, input_run_dirs: list[Path]) -> dict[str, Any]`
- Purpose and contract: merge completed/interrupted sleep2stat run tables into a cohort-level finalized bundle, keeping later duplicate `record_id` rows and dropping failures resolved by successful night-stat rows.
- Important inputs/outputs: output run directory and ordered input run directories in; merged manifest/progress metadata out.
- Side effects: writes merged `record_manifest.csv`, `tables/night_stats.csv`, `status/failures.csv`, `status/progress.json`, and `run_manifest.json`.
- Key callers/callees: called by CLI `cohort-finalize`; optional plotting remains delegated to `plot_cohort`.
- Reuse guidance: use for post-shard or fixed-cohort merge/finalization instead of one-off pandas combine scripts.
- Duplication-risk notes: this command does not read raw NPZs, rerun analyzers, or copy per-record sidecars.

## `sleep2stat.io.records.load_records`

- File: `sleep2stat/io/records.py`
- Signature: `load_records(data_cfg: DataConfig, *, split_override: list[str] | None = None, limit: int | None = None) -> list[SleepRecord]`
- Purpose and contract: build `SleepRecord` objects from NPZ index CSV rows or Kaldi manifest split CSV rows.
- Important inputs/outputs: sleep2stat data config, optional split override, and optional record limit in; record list out.
- Side effects: reads CSV and Kaldi manifest JSON files.
- Key callers/callees: caller is `run_pipeline`; callees are `_load_npz_records`, `_load_kaldi_records`, `_record_id`, `_validate_record_id_segment`, and `_validate_unique_record_ids`.
- Reuse guidance: use this whenever sleep2stat needs record discovery, split filtering, path preservation, or record-id semantics.
- Duplication-risk notes: do not read sleep2stat indexes directly in analyzers; path-safe `record_id` and duplicate-id checks belong here.

## `sleep2stat.io.records.records_to_frame`

- File: `sleep2stat/io/records.py`
- Signature: `records_to_frame(records: list[SleepRecord], metadata_columns: list[str] | None = None) -> pd.DataFrame`
- Purpose and contract: serialize record metadata into `record_manifest.csv` rows.
- Important inputs/outputs: records and optional metadata-column allowlist in; dataframe out.
- Side effects: none.
- Key callers/callees: caller is `AnalysisBundleWriter.write_record_manifest`.
- Reuse guidance: use for manifest rows so `raw_path`, `resolved_path`, `path_exists`, timing, and scalar metadata stay aligned.
- Duplication-risk notes: avoid separate manifest row builders in CLI or tests.

## `sleep2stat.core.pipeline.run_pipeline`

- File: `sleep2stat/core/pipeline.py`
- Signature: `run_pipeline(config: Sleep2statConfig, args: argparse.Namespace)`
- Purpose and contract: execute a sleep2stat run from config and runtime args. For non-model configs, `--num-workers > 1` uses an internal single-machine record-level `splitN`; configs with `sleep2vec_downstream` keep the canonical model/DataLoader path.
- Important inputs/outputs: validated config plus CLI args in; run directory path out.
- Side effects: writes output directories, progress, record manifests, per-record sidecars, global tables, failure CSVs, and run manifest.
- Key callers/callees: caller is `sleep2stat.cli.main`; callees include `load_records`, `AnalysisBundleWriter`, `create_analyzer`, `create_reducer`, `_chunk_size`, `_record_chunks`, and the internal record split helpers.
- Reuse guidance: this is the canonical execution loop for sleep2stat analysis bundles.
- Duplication-risk notes: dry-run, skip-existing, chunk-level failure handling, reducer fallback, and completion markers must not be reimplemented in agent scripts.

## `sleep2stat.core.artifacts.AnalyzerResult`

- File: `sleep2stat/core/artifacts.py`
- Signature: `AnalyzerResult(name: str, record_id: str, epoch: pd.DataFrame | None = None, second: pd.DataFrame | None = None, events: pd.DataFrame | None = None, night: dict[str, Any] | None = None, arrays: dict[str, np.ndarray] = field(default_factory=dict), warnings: list[str] = field(default_factory=list))`
- Purpose and contract: carry one analyzer or reducer output for one record across the pipeline and into bundle writers.
- Important inputs/outputs: analyzer name, record id, optional epoch/second/events/night tables, optional arrays, and warnings in; structured pipeline result out.
- Side effects: none.
- Key callers/callees: produced by analyzers and reducers; consumed by `run_pipeline`, `StageSourceResolver`, and `AnalysisBundleWriter.collect_tables`.
- Reuse guidance: add fields to the existing `epoch`, `second`, `events`, `night`, or `arrays` slots instead of creating a parallel result object.
- Duplication-risk notes: new output surfaces should remain compatible with writer table collection and per-record sidecars.

## `sleep2stat.registry.register_analyzer` and `create_analyzer`

- File: `sleep2stat/registry.py`
- Signatures:
  - `register_analyzer(name: str)`
  - `create_analyzer(config: AnalyzerConfig)`
- Purpose and contract: register analyzer classes by YAML type name and instantiate configured analyzers.
- Important inputs/outputs: type names and `AnalyzerConfig` in; analyzer instance out.
- Side effects: registration mutates `ANALYZER_REGISTRY` at import time.
- Key callers/callees: `sleep2stat/analyzers/__init__.py` imports analyzer modules for side effects; `run_pipeline` calls `create_analyzer`.
- Reuse guidance: add new analyzer types by decorating a class with `@register_analyzer`.
- Duplication-risk notes: avoid type dispatch in `run_pipeline`.

## `sleep2stat.registry.register_reducer` and `create_reducer`

- File: `sleep2stat/registry.py`
- Signatures:
  - `register_reducer(name: str)`
  - `create_reducer(config: ReducerConfig)`
- Purpose and contract: register reducer classes by YAML type name and instantiate configured reducers.
- Important inputs/outputs: type names and `ReducerConfig` in; reducer instance out.
- Side effects: registration mutates `REDUCER_REGISTRY` at import time.
- Key callers/callees: `sleep2stat/reducers/__init__.py` imports reducer modules for side effects; `run_pipeline` calls `create_reducer`.
- Reuse guidance: add reducer behavior through this registry.
- Duplication-risk notes: keep reducer type support synchronized with `SUPPORTED_REDUCER_TYPES` in `sleep2stat.config`.

## `sleep2stat.core.stage_sources.StageSourceResolver`

- File: `sleep2stat/core/stage_sources.py`
- Signature: `StageSourceResolver(records: list[SleepRecord], results: list[AnalyzerResult] | None = None)`
- Purpose and contract: index epoch-stage analyzer results and provide stage lookup plus denominator helpers.
- Important inputs/outputs: records and prior analyzer results in; stage frames, masks, TST hours, denominator hours, stage minutes, or stage-at-second arrays out.
- Side effects: none.
- Key callers/callees: callers include model AHI decoding, YASA event analyzers, YASA HRV, SpO2 ODI, and reducers needing stage context.
- Reuse guidance: use `get_denominator_hours` for sleep/REM/NREM hour metrics and `get_stage_minutes` for YASA-style stage event densities.
- Duplication-risk notes: do not recalculate stage masks independently inside analyzers.

## `sleep2stat.analyzers.model_downstream.Sleep2vecDownstreamAnalyzer`

- File: `sleep2stat/analyzers/model_downstream.py`
- Signature: `Sleep2vecDownstreamAnalyzer(config: AnalyzerConfig)`
- Purpose and contract: run a trained `sleep2vec`, `sleep2vec2`, or `sleep2expert` downstream checkpoint over sleep2stat records and convert logits into epoch, second, event, or night outputs.
- Important inputs/outputs: analyzer config with namespace, label name, finetune config, checkpoint path, input channels, optional scalar threshold, and explicit AHI postprocess controls in; analyzer results and per-record failures out.
- Side effects: imports namespace-local model modules, loads a checkpoint, builds datasets/loaders, moves batches to the configured device, and may create temporary filtered Kaldi manifests.
- Key callers/callees: instantiated through registry; uses `_build_datasets`, `_build_kaldi_datasets`, namespace-local `apply_finetune_config`, namespace-local `Sleep2vecFinetuning`, `_resolve_threshold`, `_decode_batch`, and logit decoders.
- Reuse guidance: use this analyzer for model-derived stage, age, sex, or AHI outputs inside sleep2stat.
- Duplication-risk notes: do not bypass the finetune config/model path or feed embedding-export Kaldi manifests into this analyzer.

## `sleep2stat.analyzers.model_downstream._build_datasets`

- File: `sleep2stat/analyzers/model_downstream.py`
- Signature: `_build_datasets(*, records: list[SleepRecord], channel_specs: dict[str, ChannelSpec], batch_size: int, num_workers: int, context: Sleep2statContext) -> list[DefaultDataset]`
- Purpose and contract: build model-analyzer datasets for NPZ or Kaldi sleep2stat records while preserving the existing runtime batch contract.
- Important inputs/outputs: records, sleep2stat channel specs, loader controls, and context in; one or more datasets out.
- Side effects: for Kaldi, delegates to temporary filtered manifest creation.
- Key callers/callees: caller is `Sleep2vecDownstreamAnalyzer.run`; callees are `_Sleep2statDataset` for NPZ and `_build_kaldi_datasets` for Kaldi.
- Reuse guidance: route model-analyzer data loading through this helper.
- Duplication-risk notes: keep raw-signal widths and Kaldi manifest embedding-width handling here.

## `sleep2stat.analyzers.model_downstream.decode_classification_logits`

- File: `sleep2stat/analyzers/model_downstream.py`
- Signature: `decode_classification_logits(analyzer_name: str, logits: torch.Tensor, batch: dict[str, Any], record_by_path: dict[str, SleepRecord], *, record_by_id: dict[str, SleepRecord] | None = None, include_probabilities: bool, include_logits: bool) -> list[AnalyzerResult]`
- Purpose and contract: convert scalar or sequence classification logits into night or epoch analyzer results.
- Important inputs/outputs: logits and batch metadata in; analyzer results with predictions, confidence, optional probabilities, and optional raw logits out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecDownstreamAnalyzer._decode_batch`.
- Reuse guidance: use for stage and scalar classification sleep2stat outputs.
- Duplication-risk notes: keep class probability naming here so writers and plots see stable fields.

## `sleep2stat.analyzers.model_downstream.decode_regression_logits`

- File: `sleep2stat/analyzers/model_downstream.py`
- Signature: `decode_regression_logits(analyzer_name: str, logits: torch.Tensor, batch: dict[str, Any], record_by_path: dict[str, SleepRecord], *, record_by_id: dict[str, SleepRecord] | None = None) -> list[AnalyzerResult]`
- Purpose and contract: convert regression logits into night-level predictions and optional metadata error fields.
- Important inputs/outputs: logits and batch metadata in; analyzer results out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecDownstreamAnalyzer._decode_batch`.
- Reuse guidance: use for scalar regression outputs such as age.
- Duplication-risk notes: do not duplicate age metadata error calculation in reducers.

## `sleep2stat.analyzers.model_downstream.decode_ahi_logits`

- File: `sleep2stat/analyzers/model_downstream.py`
- Signature: `decode_ahi_logits(analyzer_name: str, logits: torch.Tensor, batch: dict[str, Any], record_by_path: dict[str, SleepRecord], *, record_by_id: dict[str, SleepRecord] | None = None, threshold: float, threshold_source: str = "config", include_probabilities: bool = True, min_event_duration_sec: int = 10, merge_tolerance_sec: int = 3, denominator_stage_source: str | None = None, output_second_alignment: bool = True, output_event_alignment: bool = True, stage_resolver: StageSourceResolver | None = None) -> list[AnalyzerResult]`
- Purpose and contract: convert AHI logits into second alignment, predicted respiratory events, and model-derived night statistics.
- Important inputs/outputs: logits, batch, threshold, postprocess controls, optional stage source in; analyzer results with second/event/night tables out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecDownstreamAnalyzer._decode_batch`; callees include `_events_from_prob`, `_event_rate`, and `StageSourceResolver`.
- Reuse guidance: keep all model-AHI field naming, threshold metadata, event extraction, and clinical sleep-denominator `pred_ahi` logic here.
- Duplication-risk notes: model-hour and recording-hour event rates are QC denominators; only sleep-denominator outputs should be treated as clinical AHI.

## `sleep2stat.analyzers.npz_stage_reference.NpzStageReferenceAnalyzer`

- File: `sleep2stat/analyzers/npz_stage_reference.py`
- Signature: `NpzStageReferenceAnalyzer(config: AnalyzerConfig)`
- Purpose and contract: read a stage vector from each NPZ record and expose it as an epoch analyzer result.
- Important inputs/outputs: configured NPZ stage key and records in; epoch frame out.
- Side effects: reads NPZ files.
- Key callers/callees: uses `data.utils.load_npz`; reducers and `StageSourceResolver` consume its epoch output.
- Reuse guidance: use for reference hypnograms in sleep2stat bundles.
- Duplication-risk notes: do not create parallel NPZ stage readers in reducers.

## `sleep2stat.analyzers.yasa.YasaStageAnalyzer`

- File: `sleep2stat/analyzers/yasa.py`
- Signature: `YasaStageAnalyzer(config: AnalyzerConfig)`
- Purpose and contract: run YASA sleep staging on configured NPZ channels and emit epoch stage predictions plus optional probabilities.
- Important inputs/outputs: YASA-compatible channels and metadata in; epoch result and optional probability array out.
- Side effects: imports `mne` and `yasa`, reads NPZ signals, creates MNE Raw arrays.
- Key callers/callees: extends `_YasaBaseAnalyzer`; uses `_build_raw`, `_first_kind_channel`, `_yasa_metadata`, `_stage_id`, and `_epoch_base_frame`.
- Reuse guidance: use this for YASA-derived stage sources.
- Duplication-risk notes: YASA stage metadata encoding belongs in `_yasa_metadata`.

## `sleep2stat.analyzers.yasa.YasaBandpowerAnalyzer`

- File: `sleep2stat/analyzers/yasa.py`
- Signature: `YasaBandpowerAnalyzer(config: AnalyzerConfig)`
- Purpose and contract: compute YASA bandpower by epoch, by night, and optionally by stage source.
- Important inputs/outputs: EEG-like raw channel data, explicit output mode controls, configured bands, relative/absolute controls, and optional top-level `stage_source` in; epoch and night outputs out.
- Side effects: imports YASA/MNE and reads NPZ signals.
- Key callers/callees: uses `_epoch_bandpower`, `_band_specs`, `_night_bandpower_means`, and `_stage_bandpower_means`.
- Reuse guidance: use this for spectral microstructure outputs instead of adding bandpower reducers.
- Duplication-risk notes: stage-specific band means join by `token_idx`; do not use floating-time joins.

## `sleep2stat.analyzers.yasa._YasaEventAnalyzer`

- File: `sleep2stat/analyzers/yasa.py`
- Signature: `_YasaEventAnalyzer(config: AnalyzerConfig)`
- Purpose and contract: shared implementation for YASA spindle, slow-wave, and REM event analyzers.
- Important inputs/outputs: configured detector, optional stage filtering, records, context, and prior results in; events and night summaries out.
- Side effects: imports YASA/MNE and reads NPZ signals.
- Key callers/callees: base for `YasaSpindlesAnalyzer`, `YasaSlowWavesAnalyzer`, and `YasaRemAnalyzer`; uses `_call_yasa_event_detector`, `_yasa_event_frame`, and `_event_night_summary`.
- Reuse guidance: add closely related YASA event detectors through this base when their output contract matches.
- Duplication-risk notes: REM detection requires paired EOG arrays; keep that special case inside `_call_yasa_event_detector`.

## `sleep2stat.analyzers.yasa._event_night_summary`

- File: `sleep2stat/analyzers/yasa.py`
- Signature: `_event_night_summary(record: SleepRecord, analyzer_name: str, event_type: str, events: pd.DataFrame, resolver: StageSourceResolver | None = None, stage_source: str | None = None) -> dict[str, float]`
- Purpose and contract: summarize YASA event tables as counts, recording-hour densities, duration means, and optional stage-minute densities.
- Important inputs/outputs: event dataframe plus optional stage resolver in; night-stat mapping out.
- Side effects: none.
- Key callers/callees: called by `_YasaEventAnalyzer.run`; uses `_stage_event_densities`.
- Reuse guidance: use for YASA-like event summaries requiring stage-specific density fields.
- Duplication-risk notes: stage-specific density is events per stage minute; keep that denominator separate from AHI hour denominators.

## `sleep2stat.analyzers.yasa.YasaHrvStageAnalyzer`

- File: `sleep2stat/analyzers/yasa.py`
- Signature: `YasaHrvStageAnalyzer(config: AnalyzerConfig)`
- Purpose and contract: run YASA HRV-by-stage using a configured stage source and flatten stage-level HRV outputs into night stats.
- Important inputs/outputs: raw channel data, stage source, records, context, and prior results in; night stats out.
- Side effects: imports YASA/MNE and reads NPZ signals.
- Key callers/callees: uses `StageSourceResolver.stage_at_seconds`, `_call_yasa_hrv_stage`, and `_flatten_hrv`.
- Reuse guidance: use when HRV outputs need stage-specific context.
- Duplication-risk notes: this analyzer requires a stage source; do not silently fall back to whole-record HRV.

## `sleep2stat.analyzers.spo2._spo2_signal`

- File: `sleep2stat/analyzers/spo2.py`
- Signature: `_spo2_signal(record: SleepRecord, context: Sleep2statContext, config) -> tuple[np.ndarray, float, np.ndarray]`
- Purpose and contract: load configured SpO2 signal, apply scaling, and create the validity mask used by all SpO2 analyzers.
- Important inputs/outputs: record, context, and analyzer config in; signal, sample rate, and validity mask out.
- Side effects: reads NPZ files.
- Key callers/callees: used by `Spo2SummaryAnalyzer`, `Spo2DesaturationAnalyzer`, and `EventRelatedHypoxicBurdenAnalyzer`.
- Reuse guidance: keep SpO2 artifact and validity rules here.
- Duplication-risk notes: do not let individual SpO2 analyzers implement their own artifact filters.

## `sleep2stat.analyzers.spo2.Spo2SummaryAnalyzer`

- File: `sleep2stat/analyzers/spo2.py`
- Signature: `Spo2SummaryAnalyzer(config: AnalyzerConfig)`
- Purpose and contract: compute night-level SpO2 summary metrics such as mean, median, nadir, T90/T88, and artifact percentage.
- Important inputs/outputs: records and context in; night-stat analyzer results out.
- Side effects: reads NPZ signals.
- Key callers/callees: uses `_spo2_signal` and `_spo2_summary`.
- Reuse guidance: use for whole-record oximetry summaries.
- Duplication-risk notes: T90/T88 numerator excludes invalid samples while denominator stays the recording span; keep that visible through explicit field names.

## `sleep2stat.analyzers.spo2.Spo2DesaturationAnalyzer`

- File: `sleep2stat/analyzers/spo2.py`
- Signature: `Spo2DesaturationAnalyzer(config: AnalyzerConfig)`
- Purpose and contract: detect SpO2 desaturation events and compute ODI fields with explicit denominators.
- Important inputs/outputs: explicit drop thresholds, explicit minimum duration, optional maximum duration, optional stage source, records, context, and prior results in; event rows and night stats out.
- Side effects: reads NPZ signals.
- Key callers/callees: uses `_spo2_signal`, `_desaturation_events`, `_desaturation_rows`, `_close_desat`, and `_odi_stats`.
- Reuse guidance: use this for ODI and desaturation event outputs.
- Duplication-risk notes: ODI per recording hour, per valid-SpO2 hour, and optional per sleep hour answer different questions; keep fields separate.

## `sleep2stat.analyzers.spo2.EventRelatedHypoxicBurdenAnalyzer`

- File: `sleep2stat/analyzers/spo2.py`
- Signature: `EventRelatedHypoxicBurdenAnalyzer(config: AnalyzerConfig)`
- Purpose and contract: integrate SpO2 burden around upstream event-source intervals.
- Important inputs/outputs: configured `event_source`, SpO2 signal, records, context, and prior results in; burden events and night stats out.
- Side effects: reads NPZ signals.
- Key callers/callees: uses `_events_by_record`, `_spo2_signal`, `_event_related_burden`, `_empty_burden`, and `_burden_prefix`.
- Reuse guidance: use for respiratory-event hypoxic burden or desaturation-source burden summaries.
- Duplication-risk notes: duplicate upstream intervals are deduplicated by onset/offset before integration.

## `sleep2stat.reducers.hypnogram_stats.HypnogramStatsReducer.reduce`

- File: `sleep2stat/reducers/hypnogram_stats.py`
- Signature: `reduce(records: list[SleepRecord], results: list[AnalyzerResult], context: Sleep2statContext) -> list[AnalyzerResult]`
- Purpose and contract: summarize epoch-stage predictions into sleep architecture metrics.
- Important inputs/outputs: records and epoch analyzer results in; night analyzer results out.
- Side effects: none.
- Key callers/callees: calls `_hypnogram_stats`; registered as `hypnogram_stats`.
- Reuse guidance: use this reducer for architecture metrics from model or YASA stage sources.
- Duplication-risk notes: TIB, scored TIB, recording duration, SPT-WASO, and TST composition have explicit canonical field names; do not emit ambiguous aliases.

## `sleep2stat.reducers.transition_stats.TransitionStatsReducer.reduce`

- File: `sleep2stat/reducers/transition_stats.py`
- Signature: `reduce(records: list[SleepRecord], results: list[AnalyzerResult], context: Sleep2statContext) -> list[AnalyzerResult]`
- Purpose and contract: summarize adjacent-stage transitions as counts, entropy, and change fraction.
- Important inputs/outputs: epoch stage results in; night transition stats out.
- Side effects: none.
- Key callers/callees: calls `_transition_stats`.
- Reuse guidance: use for transition matrices and entropy-like stage-fragmentation fields.
- Duplication-risk notes: adjacent-epoch change fraction is distinct from hypnogram stage shifts per TST hour.

## `sleep2stat.reducers.stage_agreement.StageAgreementReducer.reduce`

- File: `sleep2stat/reducers/stage_agreement.py`
- Signature: `reduce(records: list[SleepRecord], results: list[AnalyzerResult], context: Sleep2statContext) -> list[AnalyzerResult]`
- Purpose and contract: compare two epoch-stage sources on shared `(record_id, path, token_idx)` rows.
- Important inputs/outputs: left and right stage source names in config plus epoch results in; agreement metrics out.
- Side effects: none.
- Key callers/callees: calls `_epoch_results_by_record` and `_agreement_metrics`.
- Reuse guidance: use for model-vs-reference or model-vs-YASA stage agreement.
- Duplication-risk notes: overlap coverage is part of the contract; do not report accuracy without the overlap counts.

## `sleep2stat.reducers.stage_specific_summary.StageSpecificSummaryReducer.reduce`

- File: `sleep2stat/reducers/stage_specific_summary.py`
- Signature: `reduce(records: list[SleepRecord], results: list[AnalyzerResult], context: Sleep2statContext) -> list[AnalyzerResult]`
- Purpose and contract: compute per-stage means for numeric epoch columns from another analyzer.
- Important inputs/outputs: source analyzer name and `options.stage_source` in config plus epoch results in; night stats out.
- Side effects: none.
- Key callers/callees: calls `_stage_numeric_means`.
- Reuse guidance: use for stage-stratified summaries of epoch-level numeric outputs.
- Duplication-risk notes: joins are by `token_idx`, not floating timestamps.

## `sleep2stat.reducers.event_density.EventDensityReducer.reduce`

- File: `sleep2stat/reducers/event_density.py`
- Signature: `reduce(records: list[SleepRecord], results: list[AnalyzerResult], context: Sleep2statContext) -> list[AnalyzerResult]`
- Purpose and contract: compute generic whole-recording event count and event density per recording hour from event analyzer results.
- Important inputs/outputs: source event analyzer name plus records/results in; night stats out.
- Side effects: none.
- Key callers/callees: called by `run_pipeline` through reducer registry.
- Reuse guidance: use only for generic recording-hour event density.
- Duplication-risk notes: analyzers with sleep-stage denominators should compute staged rates upstream.

## `sleep2stat.reducers.demographic_consistency.DemographicConsistencyReducer.reduce`

- File: `sleep2stat/reducers/demographic_consistency.py`
- Signature: `reduce(records: list[SleepRecord], results: list[AnalyzerResult], context: Sleep2statContext) -> list[AnalyzerResult]`
- Purpose and contract: compare model age/sex predictions against configured record metadata fields and emit warning counts.
- Important inputs/outputs: records, previous night stats, configured prediction names, and metadata field names in; night stats and warnings out.
- Side effects: none.
- Key callers/callees: calls `_night_by_record` and `_encode_sex`.
- Reuse guidance: use for sleep2stat demographic plausibility checks.
- Duplication-risk notes: sex encoding deliberately treats unknown, non-binary, and invalid values as missing.

## `sleep2stat.io.writers.AnalysisBundleWriter`

- File: `sleep2stat/io/writers.py`
- Signature: `AnalysisBundleWriter(config: Sleep2statConfig)`
- Purpose and contract: own all sleep2stat output bundle writes and resumable run bookkeeping, including optional parallel reads when rebuilding global tables from per-record sidecars.
- Important inputs/outputs: validated config in; methods write record manifests, progress, failures, run manifests, per-record sidecars, global table shards, summary tables, and completion markers.
- Side effects: creates directories, copies config, writes YAML/JSON/CSV/NPZ files, writes run PID/progress JSON atomically, and rebuilds tables from shards or sidecars; it does not delete existing run directories.
- Key callers/callees: used by `run_pipeline` and CLI summarize; key methods include `prepare`, `filter_records_for_run`, `write_record_manifest`, `write_progress`, `write_failures`, `write_chunk`, `write_completion_markers`, `rebuild_global_tables`, and `write_run_manifest`.
- Reuse guidance: use this writer for every sleep2stat output-contract change.
- Duplication-risk notes: skip-existing, config fingerprint validation, failure merging, sidecar completeness, no-overwrite output handling, and global table rebuilds belong here.

## `sleep2stat.plot.plot_record`

- File: `sleep2stat/plot.py`
- Signature: `plot_record(run_dir: Path, record_id: str) -> list[Path]`
- Purpose and contract: render plots for one completed per-record sleep2stat output directory using the stable per-record `events.csv(.gz)` sidecar for events.
- Important inputs/outputs: run directory and record id in; list of created plot paths out.
- Side effects: reads per-record sidecars and writes PNG plots.
- Key callers/callees: called by `sleep2stat.cli.main`; uses `_read_table`, `_plot_hypnogram_overlay`, and `_plot_ahi_spo2_trace`.
- Reuse guidance: use for record-level visualization from bundle outputs.
- Duplication-risk notes: plots should read bundle contracts, not analyzer internals.

## `sleep2stat.plot.plot_cohort`

- File: `sleep2stat/plot.py`
- Signature: `plot_cohort(run_dir: Path, *, group_column: str = "source", stage_source: str | None = None, adjust_covariates: list[str] | None = None) -> list[Path]`
- Purpose and contract: render cohort-level respiratory and microstructure panels, plus sleep architecture and optional harmonization diagnostics when a concrete stage source is supplied.
- Important inputs/outputs: run directory, grouping field, optional concrete stage source, and optional covariates in; plot paths out.
- Side effects: reads global tables and writes PNG plots.
- Key callers/callees: called by CLI and agent-generated `plot-cohort` commands; uses `_load_cohort_frame`, `_select_stage_source`, metric-spec helpers, and plotting helpers.
- Reuse guidance: use for cohort visualization after a bundle completes or after `summarize` rebuilds global tables.
- Duplication-risk notes: plot reads canonical bundle fields only; command generation should pass explicit recipe values through to the CLI without adding stage-source inference.
