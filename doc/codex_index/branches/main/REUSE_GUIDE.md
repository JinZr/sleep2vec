# Reuse Guide

This page answers the practical question: when you need to add or change behavior, which implementation should you reuse first?

## Highest-Value Reuse Hotspots

| Responsibility | Canonical implementation to reuse | Why it is canonical | Do not bypass with |
| --- | --- | --- | --- |
| Pretrain YAML parsing | `sleep2vec.config.load_pretrain_config` | Enforces required model/loss/data blocks and typed config-bundle creation, including `adapt` | Hand-written YAML parsing in `pretrain.py` or `adapt.py` |
| Finetune YAML parsing | `sleep2vec.config.load_finetune_config` | Centralizes task, head, layer-mix, LoRA/DoRA hyperparameters, evaluation-visualization, and model-averaging parsing | Entry-point specific YAML parsing |
| Config-only model loading | `sleep2vec.config.load_model_config` | Smallest schema-only loader when callers need channel/backbone structure without a full runtime bundle | Ad hoc YAML slicing |
| Task semantics | `sleep2vec.common.apply_task_flags` plus `get_task_label_source_name`, `get_task_auxiliary_label_source_names`, and `remap_stage_labels` | Single source for built-in `stage3`/`stage4`/`stage5`/`ahi`/`sex`/`age` semantics | Duplicated task/type/monitor/remap logic |
| Finetune CLI normalization | `sleep2vec.common.apply_finetune_config` | Binds YAML into `args`, enforces channel parity, and derives task flags | Re-copying YAML fields in `finetune.py` or `infer.py` |
| Data-backend normalization | `sleep2vec.common.apply_data_backend_args` | Centralizes `npz` vs `kaldi`, `kaldi_data_root`, `kaldi_manifest`, and preset rejection for Kaldi runs | Backend-specific checks scattered through entrypoints |
| Run artifact persistence | `sleep2vec.common.persist_run_config_and_args` | Single helper for root-level and phase-scoped config / CLI snapshots | Entry-point-local file copying |
| Registry-backed construction | `sleep2vec.builders.*`, `sleep2vec.registry.*` | All config-backed model assembly flows through here | Direct instantiation scattered across callers |
| Tokenizer instantiation | `sleep2vec.modules.tokenizers.build_tokenizer_from_channel` and `build_tokenizer_mapping` | Guarantees channel config is respected | Manual tokenizer maps |
| Projection creation | `sleep2vec.modules.projection.build_projection_head` | Central toggle for enabled/disabled projection heads | Ad hoc `SimCLRProjectionHead(...)` calls |
| Backbone encode path | `Sleep2vecPretrainModel._token_embeddings_to_hidden` | Single place that projects tokens, adds CLS, builds masks, runs the encoder, and optionally exposes hidden states | Re-creating encoder + CLS plumbing in downstream code |
| Adaptation freeze policy | `Sleep2vecPretrainModel.apply_adaptation_freeze_policy` and `get_adaptation_param_groups` | Encodes stage1/stage2 trainability boundaries for encoder, projection, legacy channels, and new channels | New per-phase freeze helpers outside the backbone |
| Downstream feature path | `Sleep2vecDownstreamModel.forward` | Central path for per-modality encoding, temporal aggregation, channel fusion, layer mix, and head invocation | Parallel forward paths in trainer code |
| Pretrained backbone loading | `Sleep2vecDownstreamModel.load_pretrained_backbone` | Encodes prefix handling, EMA fallback, and CLS mismatch warnings | Custom checkpoint slicing logic |
| Pretrain init loading for adapt/resume | `sleep2vec.checkpoints.load_pretrain_init_weights` | Shared loader for `model.` vs averaged-model prefixes with explicit load reporting | Custom state-dict prefix stripping |
| LoRA insertion | `Sleep2vecDownstreamModel.freeze_backbone_and_insert_lora` and package-local variant mirrors | Centralizes freeze policy, LoRA/DoRA hyperparameters, adapter insertion, and separate-adapter trainability | Direct `peft` calls in trainer code |
| Pretrain data loaders | `sleep2vec.utils.get_pretrain_dataloader` | Owns missing-channel mode, sequential pair-eval validation, worker defaults, and sampler choice | Building `PSGPretrainDataset` loaders manually in entrypoints |
| Finetune/infer data loaders | `sleep2vec.utils._build_finetune_loader` and `get_finetune_dataloaders` | Own split/source choices, built-in sequence label channels, and AHI auxiliary `stage5` injection | Hand-rolled finetune loader creation |
| Dataset backend dispatch | `sleep2vec.utils._dataset_class_for_args` | Chooses `PSGPretrainDataset` or `KaldiPSGDataset` from normalized `args.data_backend` | Entry-point-specific dataset class branching |
| Sample validation | `data.utils.filter_valid_sample_indices` | Produces `payload["available_channels"]`, validates built-in AHI samples, and drops broken samples early | Custom preset-building loops |
| Built-in AHI metadata loading | `data.utils.load_builtin_ahi_metadata` | Single contract for `ah_event`, scalar `ahi`, and scalar `tst` | Custom scalar parsing in dataset or metrics code |
| Runtime batch assembly | `DefaultDataset.dataloader` | Single source for collate-time NPZ reads, tokenization, metadata packing, `token_start`, `w/h`, and sampler choice | New collate functions outside `data/default_dataset.py` |
| Runtime token storage hooks | `DefaultDataset._get_available_channels_for_src` and `_load_tokens_for_src` | Extension points that let `KaldiPSGDataset` reuse collate semantics without NPZ reads | Separate Kaldi collate functions |
| Kaldi matrix reads | `data.kaldi_io.KaldiReaderPool` | Owns sorted `scp:` reader construction, process-local reader reopening, and shape checks | Direct `kaldi_native_io` readers in datasets |
| Kaldi runtime dataset | `data.kaldi_psg_dataset.KaldiPSGDataset` | Reuses `DefaultDataset` batch contract from `manifest.json` format v2 and split CSVs | New dataset classes that bypass `SampleIndex` payload semantics |
| Missing-channel training batches | `PairFirstBatchSampler` | Canonical train-time sampler for pair-first missing-channel pretraining/adaptation | Ad hoc pair scheduling loops |
| Missing-channel homogeneous eval/train fallback | `AvailableChannelsBucketBatchSampler` | Canonical bucketed sampler when pair-first is not active | New bucket logic in entrypoints |
| Checkpoint averaging | `sleep2vec.checkpoints.select_checkpoints` and `average_checkpoints` | Encodes epoch-first selection plus fallback to mtime | Local checkpoint averaging scripts |
| Generic downstream metric reduction | `sleep2vec.metrics.compute_downstream_metrics` | Single metric reducer for classification recall/specificity, regression, multilabel AHI pointwise, and stage remap outputs | Per-stage custom metric calculations |
| AHI threshold search and event metrics | `sleep2vec.metrics.compute_ahi_event_metrics`, `select_best_ahi_threshold`, and the prepared-record helpers | Single contract for validation threshold search, record merging, and event/summary metrics | New AHI evaluation branches in trainers or scripts |
| Result CSV output | `sleep2vec.results.save_result_csv` | Preserves rank-zero gating, lockfile semantics, schema expansion, and standard metadata columns | One-off CSV writers or the removed `metrics.save_result_csv` path |
| Inference output paths and manifest | `sleep2vec.results.prepare_inference_result_paths`, `make_prediction_run_id`, and `save_inference_manifest` | Centralizes `results/inference/<namespace>/<label>/<prediction_run_id>/`, checkpoint tags, overview paths, and manifest schema | Rebuilding inference output folders in `infer.py` or trainers |
| Inference W&B artifacts | `sleep2vec.infer._log_inference_outputs_to_wandb` | Logs metrics plus `prediction_row_count` and uploads the metrics, predictions, manifest, and overview files together | W&B upload logic in result writers or variant-specific one-offs |
| Inference prediction rows | `sleep2vec.sleep2vec_inference.extract_prediction_records`, `build_prediction_rows`, and `build_ahi_prediction_rows` | Converts model outputs into path-level rows for classification, regression, multilabel, and AHI tasks while deduplicating `(path, token_start)` windows | Saving raw batch logits directly from trainer loops |
| Token embedding extraction | `sleep2vec.extract_embeddings.run_extraction` plus package-local variant mirrors | Loads pretrain or finetune backbone checkpoints strictly, reuses runtime loaders, preserves configured finetune adapters, and writes selected-layer token states to NPZ or Kaldi manifests | Ad hoc scripts that duplicate tokenizer/CLS/layer-selection or checkpoint-prefix handling |
| Prediction CSV output | `sleep2vec.results.save_prediction_csv` | Preserves rank-zero gating, lockfile semantics, empty-header output, list serialization, and metadata parity with metrics CSVs | Ad hoc per-path CSV writers |
| Downstream eval plotting | `sleep2vec.visualization.downstream_eval.DownstreamEvalVisualizer` | Centralizes confusion matrix, ROC, regression scatter, and AHI summary scatter logging | WandB logging logic inside trainer steps |
| Preset generation | `preprocess/save_dataset_presets.py` | Canonical CLI path that exercises `PSGPretrainDataset` and YAML-driven `preset_build` side effects | External scripts that pickle `SampleIndex` lists directly |
| Kaldi conversion | `preprocess/convert_npz_to_kaldi.py` plus package-local mirrors | Canonical NPZ-to-Kaldi root writer with split manifests, sorted scps, sharding, and semantic compression policy | One-off ark/scp writers |
| Split generation | `preprocess/split_index_by_dataset.py` | Canonical dataset-group split policy, mask normalization, and optional global pair-coverage checks | Manual split assignment notebooks |
| Config validation | `utils/check_configs.py` | Canonical repo policy check for config-loader compatibility and `preset_build` strictness | One-off shell loops or YAML linters without repo semantics |
| sleep2stat YAML parsing | `sleep2stat.config.load_config` | Enforces strict run/data/signals/analyzers/reducers/outputs schema, analyzer/reducer references, backend support, and stage-source ordering | Ad hoc YAML parsing in `sleep2stat.cli`, agent tooling, or recipes |
| sleep2stat record loading | `sleep2stat.io.records.load_records` | Single NPZ/Kaldi `SleepRecord` loader with split filtering, path preservation, path-safe record ids, and duplicate-id rejection | Direct pandas reads in analyzers or writers |
| sleep2stat execution loop | `sleep2stat.core.pipeline.run_pipeline` | Owns dry-run, single-use output checks, chunking, analyzer preparation, reducer execution, progress, and manifest writes | A parallel execution manager in `agent_tools` or scripts |
| sleep2stat result object | `sleep2stat.core.artifacts.AnalyzerResult` | Single carrier for analyzer/reducer `epoch`, `second`, `events`, `night`, `arrays`, and warnings | Parallel result DTOs or writer-specific row objects |
| sleep2stat analyzer/reducer construction | `sleep2stat.registry.create_analyzer` and `create_reducer` | Keeps analyzer/reducer type dispatch centralized through registration side effects | Type-name `if`/`elif` trees outside the registry |
| sleep2stat stage denominators | `sleep2stat.core.stage_sources.StageSourceResolver` | Single source for stage epoch lookup, sleep/REM/NREM hour denominators, stage-minute denominators, and onset-time stage assignment | Recomputing sleep masks independently in model, YASA, or SpO2 analyzers |
| sleep2stat model analyzer data path | `Sleep2vecDownstreamAnalyzer` plus `_build_datasets` / `_build_kaldi_datasets` | Reuses namespace-local finetune config/model code and root `DefaultDataset` / `KaldiPSGDataset` batch contracts | Passing raw arrays directly into finetune models or accepting embedding-export Kaldi manifests |
| sleep2stat AHI decoding | `sleep2stat.analyzers.model_downstream.decode_ahi_logits` | Centralizes threshold resolution, second alignment, event extraction, model/recording/sleep denominators, and clinical `pred_ahi` naming | Respiratory-event postprocessing in reducers or plot code |
| sleep2stat YASA event summaries | `sleep2stat.analyzers.yasa._event_night_summary` | Mirrors YASA event-count and stage-density semantics while using configured stage sources | Stage-density calculations copied into reducers |
| sleep2stat SpO2 loading and ODI | `sleep2stat.analyzers.spo2._spo2_signal` and `_odi_stats` | Centralizes SpO2 scaling, artifact masking, valid-SpO2 denominators, recording denominators, and optional sleep denominators | Independent oximetry artifact filters per analyzer |
| sleep2stat bundle writing | `sleep2stat.io.writers.AnalysisBundleWriter` | Owns single-use per-record sidecars, global shards, cumulative summaries, progress, and manifest schema | Writing bundle files directly from analyzers or CLI code |
| sleep2stat plotting | `sleep2stat.plot.plot_record` and `plot_cohort` | Reads completed bundle contracts and canonical cohort metric fields for record/cohort visualizations | Plot scripts that inspect analyzer internals directly or reintroduce legacy field fallbacks |
| Agent workflow support | `agent_tools.plans`, `agent_tools.recipes`, `agent_tools.decisions`, `agent_tools.hparam`, `agent_tools.experiments`, `agent_tools.adaptive_hparam`, and `agent_tools.progress` | Centralizes context bundles, recipe loading, plan generation, stop-and-consult gates, hparam orchestration, experiment monitoring, and machine-readable progress | A second training entrypoint or natural-language-only policy |
| WatchPAT conversion | `preprocess.watchpat_zzp_to_edf.convert_zzp_to_edf` | Single entrypoint for `.zzp` decoding and EDF writing | Parallel conversion scripts |
| UKB asleep night cutting | `utils/cut_ukb_sleep_with_asleep.py` | Standalone utility that mirrors UKB `.cwa` input trees and saves longest sleep block per asleep noon-to-noon interval | New sleep2vec-dependent cutting scripts |
| UKB annotation parsing | `utils/parse_ukb_annotations_by_person.py` | Converts UKB export bundles into derived dataset metadata, codings, withdrawals, manifest, and per-participant JSON files | One-off parsers that lose UDI/feature-name provenance |
| UKB demographic collection | `utils/collect_ukb_demographics.py` | Reads UKB-style participant JSON trees and extracts age/sex with source columns | Manual spreadsheet joins for sex/age fields |
| Kaldi index repair | `utils/fix_kaldi_index.py` | Assigns stable unique `session_id` values so converter sample keys are unique before ark/scp writing | Editing CSV rows by hand after converter failures |
| Case-control matching | `utils/match_case_controls.py` | MatchIt-style CSV utility with exact constraints, calipers, propensity features, optional genetic weight search, and balance outputs | Untracked notebooks for cohort matching |
| Standalone dense variant | `sleep2vec2/*` package-local implementations | Maintains behavior parity while keeping imports under `sleep2vec2`, including data/preprocess and LoRA/DoRA mirrors | Cross-namespace shortcuts through root `sleep2vec`, `data`, or `preprocess` |
| Standalone MoE config | `sleep2expert.config.MoeConfig`, `_validate_moe_config`, and `_build_finetune_moe_tuning_config` | Single source for MoE schema, router groups, finetune modes, LR scales including `lora`, and unsupported regularization/router-LoRA checks | Reading `model.backbone.moe` as loose dicts |
| Sparse MoE routing | `sleep2expert.backbones.roformer.moe.TopKRouter` and `SparseMoEFFN` | Canonical router/expert implementation for learned, random, hard-modality, and hard-group modes | Router branches outside the standalone RoFormer layers |
| MoE regularization | `sleep2expert.losses.moe_regularization.compute_moe_regularization` and `compute_downstream_moe_regularization` | Centralizes load balance, modality balance, route consistency, router z-loss, entropy, and downstream-supported subset | Trainer-local MoE loss calculations |
| MoE checkpoint expansion | `sleep2expert.checkpoints.initialize_moe_from_dense_if_possible` | Clones compatible dense FFN weights into MoE experts and rejects incomplete or shape-incompatible states | Ad hoc state-dict rewrites before load |
| Routing export | `sleep2expert.routing_analysis.run_routing_analysis` | Loads a finetune or pretrained MoE model, reads `last_moe_aux`, writes CSV rows, and optionally renders heatmaps | Separate scripts that inspect router tensors manually |

## Reuse Rules By Change Type

### If you are changing config semantics

- Reuse `load_pretrain_config`, `load_finetune_config`, `apply_finetune_config`, and `persist_run_config_and_args`.
- Keep parse-time validation in `sleep2vec/config.py`.
- Keep CLI mutation and built-in task semantics in `sleep2vec/common.py`.
- Do not move semantic validation into entrypoints or tests.

### If you are changing model construction or task outputs

- Reuse `build_encoder_factory`, `build_tokenizers_and_dim`, `build_projection`, `build_cls_embedding`, `build_temporal_aggregator`, and `build_channel_aggregator`.
- Reuse `remap_stage_labels` instead of open-coding stage3/stage4 merges.
- Register new implementations through registries instead of branching on names in runtime code.

### If you are changing batch or sampler behavior

- Reuse `filter_valid_sample_indices` for preset validation and `DefaultDataset.dataloader` for collate-time semantics.
- Reuse `PairFirstBatchSampler` or `AvailableChannelsBucketBatchSampler` instead of adding new sampler logic in `sleep2vec/utils.py`.
- Preserve `payload["available_channels"]` if missing-channel support is involved.
- For Kaldi, override the storage hooks through `KaldiPSGDataset`; keep the batch shape produced by `DefaultDataset.dataloader`.

### If you are changing AHI behavior

- Keep built-in AHI sample validation in `data.utils.load_builtin_ahi_metadata` and `filter_valid_sample_indices`.
- Keep AHI loader semantics in `_build_finetune_loader`.
- Keep threshold fitting and record merging in `sleep2vec.metrics` and `Sleep2vecFinetuning`.
- Do not create a second results/threshold path in entrypoints.

### If you are changing runtime orchestration

- Keep trainer, callback, wandb, checkpoint, and phase-transition behavior in `pretrain.py`, `adapt.py`, `finetune.py`, `infer.py`, or the Lightning modules.
- Reuse `persist_run_config_and_args`, `save_result_csv`, and checkpoint helpers instead of duplicating serialization and output logic.

### If you are changing preprocessing or config policy

- Prefer composing the existing CLI utilities:
  - `split_index_by_dataset.py`
  - `mask_missing_stats.py`
  - `save_dataset_presets.py`
  - `merge_dataset_presets.py`
  - `utils/check_configs.py`
- Only touch `watchpat_zzp_to_edf.py` for WatchPAT-specific conversion work.
- Use `utils/cut_ukb_sleep_with_asleep.py` for UKB `.cwa` night extraction with the external `asleep` package; keep it independent of sleep2vec runtime/data imports.
- Use `utils/parse_ukb_annotations_by_person.py` before downstream scripts need stable UKB feature names or participant JSON.
- Use `utils/collect_ukb_demographics.py` to derive `eid`, age, and sex tables from those JSON trees.
- Use `utils/fix_kaldi_index.py` when duplicate Kaldi sample keys come from repeated source/session path prefixes.
- Use `utils/match_case_controls.py` for reproducible case-control cohort matching and balance diagnostics.

### If you are adding agent-facing workflow support

- Reuse `agent_tools.plans.build_context`, `agent_tools.plans.build_plan`, `agent_tools.recipes`, and `agent_tools.decisions.evaluate_consultation_gates` for context, plan, recipe, and stop-and-consult behavior.
- Reuse `agent_tools.hparam`, `agent_tools.experiments`, `agent_tools.adaptive_hparam`, and `agent_tools.progress` for launch/monitor/rank/export, W&B/checkpoint collection, adaptive tuning, and progress reporting.
- Do not create a second training entrypoint.
- Do not parse W&B logs when run manifests are available.
- Keep context gathering lightweight and free of Torch/Lightning imports.
- Enforce `NEEDS_USER_INPUT` for high-impact ambiguous decisions before writing runnable scripts.
- Preserve recipe `variant` routing so generated commands call `sleep2vec`, `sleep2vec2`, or `sleep2expert` package-local entrypoints as requested.

### If you are changing sleep2stat

- Reuse `sleep2stat.config.load_config` for all structural config checks, including agent-facing summaries and `utils/check_configs.py`.
- Reuse `load_records` for NPZ/Kaldi record discovery; do not let analyzers read index CSVs directly.
- Reuse `AnalyzerResult` for analyzer and reducer outputs; add fields to existing tables instead of inventing parallel result objects.
- Reuse `StageSourceResolver` whenever a metric needs TST, sleep-hour, REM/NREM-hour, or stage-minute denominators.
- Reuse `AnalysisBundleWriter` for output files, single-use output checks, and global table rebuilds.
- Keep model-derived respiratory event semantics in `decode_ahi_logits`; reducers should consume analyzer outputs rather than reinterpret logits.
- Keep `task=sleep2stat` variantless in `agent_tools`; generated commands should call the existing `sleep2stat` CLI only.

### If you are changing standalone variants

- Keep `sleep2vec2` and `sleep2expert` imports package-local.
- Mirror root data/preprocess behavior only when the contract is meant to stay identical.
- Keep standalone RoFormer attention-backend changes aligned across `sleep2vec2` and `sleep2expert`.
- Keep LoRA/DoRA config and downstream insertion behavior aligned with root; for `sleep2expert`, use `finetune.moe_tuning.lr_scales.lora` for adapter optimizer grouping.
- Use variant-specific tests such as `tests/test_sleep2vec2_namespace.py`, `tests/test_sleep2expert_namespace.py`, `tests/test_variant_data_protocol.py`, and the Kaldi backend parity tests to guard namespace drift.
- For `sleep2expert` MoE behavior, route schema changes through `sleep2expert.config`, routing changes through `sleep2expert.backbones.roformer.moe`, and export changes through `sleep2expert.routing_analysis`.

## Major Duplication Risks

1. `Sleep2vecPretrainModel` construction is config-only. Do not reintroduce manual channel, hidden-size, projection, or encoder-factory constructor branches.
2. `_contrastive_accuracy` is still duplicated in both shipped contrastive loss modules.
3. Warmup-plus-cosine optimizer scheduling now exists in pretrain, finetune, and adaptation. Avoid creating a fourth copy unless the schedule contract truly changes.
4. Available-channel resolution is duplicated between `sleep2vec.utils`, `data.utils`, sampler initialization, and `DefaultDataset` internals. Avoid creating another interpretation.
5. `_mask` truthiness now matters in both split preparation and strict preset prefiltering. Keep `normalize_mask_frame` semantics aligned everywhere.
6. AHI evaluation is split into pointwise training reduction and event-based validation/test reduction. Do not create a third metric path.
7. Config folder names are not authoritative semantics. Inspect actual `finetune.task`, `model.cls`, and `preset_build` fields before assuming behavior from file names.
8. The old `sleep2vec.metrics.save_result_csv` location is stale. New code should reuse `sleep2vec.results.save_result_csv`.
9. Kaldi support reuses the same `DefaultDataset` batch contract through storage hooks. Avoid adding a second collate path.
10. `sleep2vec2` and `sleep2expert` are package-local mirrors; cross-namespace imports are regressions unless a test explicitly permits them.
11. MoE routing aux is transient in `last_moe_aux`; persistent analysis should go through `sleep2expert.routing_analysis`.
12. Inference metrics, overview rows, predictions, manifests, and W&B artifacts share one `prediction_run_id`; update `sleep2vec.results`, `sleep2vec.sleep2vec_inference`, and inference W&B logging together instead of adding a parallel export path.
13. Binary `specificity` and stage alias `spec` intentionally have different averaging semantics for two-class stage collapses; use `compute_downstream_metrics` instead of recalculating them outside `metrics.py`.
14. sleep2stat model-hour, recording-hour, and sleep-hour respiratory denominators have distinct meanings; keep field names explicit and do not collapse them into one AHI column.
15. sleep2stat YASA and SpO2 analyzers are NPZ/raw-signal analyzers; only `sleep2vec_downstream` currently supports Kaldi-backed records.

## Known Non-Reuse Zones

- `preprocess/preprocess_pipeline.ipynb` is workflow history, not canonical library code.
- Package-local preprocessing notebooks under variants are workflow history, not canonical reusable implementations.
- Test helper functions are scaffolding only; reuse the product functions they exercise instead.
