# Reuse Guide

This page answers the practical question: when you need to add or change behavior, which implementation should you reuse first?

## Highest-Value Reuse Hotspots

| Responsibility | Canonical implementation to reuse | Why it is canonical | Do not bypass with |
| --- | --- | --- | --- |
| Pretrain YAML parsing | `sleep2vec.config.load_pretrain_config` | Enforces required model/loss/data blocks and typed config-bundle creation, including `adapt` | Hand-written YAML parsing in `pretrain.py` or `adapt.py` |
| Finetune YAML parsing | `sleep2vec.config.load_finetune_config` | Centralizes task, head, layer-mix, LoRA, evaluation-visualization, and model-averaging parsing | Entry-point specific YAML parsing |
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
| LoRA insertion | `Sleep2vecDownstreamModel.freeze_backbone_and_insert_lora` | Centralizes freeze policy and adapter insertion | Direct `peft` calls in trainer code |
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
| Generic downstream metric reduction | `sleep2vec.metrics.compute_downstream_metrics` | Single metric reducer for classification, regression, multilabel AHI pointwise, and stage remap outputs | Per-stage custom metric calculations |
| AHI threshold search and event metrics | `sleep2vec.metrics.compute_ahi_event_metrics`, `select_best_ahi_threshold`, and the prepared-record helpers | Single contract for validation threshold search, record merging, and event/summary metrics | New AHI evaluation branches in trainers or scripts |
| Result CSV output | `sleep2vec.results.save_result_csv` | Preserves rank-zero gating, lockfile semantics, schema expansion, and standard metadata columns | One-off CSV writers or the removed `metrics.save_result_csv` path |
| Downstream eval plotting | `sleep2vec.visualization.downstream_eval.DownstreamEvalVisualizer` | Centralizes confusion matrix, ROC, regression scatter, and AHI summary scatter logging | WandB logging logic inside trainer steps |
| Preset generation | `preprocess/save_dataset_presets.py` | Canonical CLI path that exercises `PSGPretrainDataset` and YAML-driven `preset_build` side effects | External scripts that pickle `SampleIndex` lists directly |
| Kaldi conversion | `preprocess/convert_npz_to_kaldi.py` plus package-local mirrors | Canonical NPZ-to-Kaldi root writer with split manifests, sorted scps, sharding, and semantic compression policy | One-off ark/scp writers |
| Split generation | `preprocess/split_index_by_dataset.py` | Canonical dataset-group split policy, mask normalization, and optional global pair-coverage checks | Manual split assignment notebooks |
| Config validation | `utils/check_configs.py` | Canonical repo policy check for config-loader compatibility and `preset_build` strictness | One-off shell loops or YAML linters without repo semantics |
| WatchPAT conversion | `preprocess.watchpat_zzp_to_edf.convert_zzp_to_edf` | Single entrypoint for `.zzp` decoding and EDF writing | Parallel conversion scripts |
| Standalone dense variant | `sleep2vec2/*` package-local implementations | Maintains behavior parity while keeping imports under `sleep2vec2`, including data/preprocess mirrors | Cross-namespace shortcuts through root `sleep2vec`, `data`, or `preprocess` |
| Standalone MoE config | `sleep2expert.config.MoeConfig`, `_validate_moe_config`, and `_build_finetune_moe_tuning_config` | Single source for MoE schema, router groups, finetune modes, LR scales, and unsupported regularization checks | Reading `model.backbone.moe` as loose dicts |
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

### If you are changing standalone variants

- Keep `sleep2vec2` and `sleep2expert` imports package-local.
- Mirror root data/preprocess behavior only when the contract is meant to stay identical.
- Use variant-specific tests such as `tests/test_sleep2vec2_namespace.py`, `tests/test_sleep2expert_namespace.py`, `tests/test_variant_data_protocol.py`, and the Kaldi backend parity tests to guard namespace drift.
- For `sleep2expert` MoE behavior, route schema changes through `sleep2expert.config`, routing changes through `sleep2expert.backbones.roformer.moe`, and export changes through `sleep2expert.routing_analysis`.

## Major Duplication Risks

1. `Sleep2vecPretrainModel` still contains a legacy non-config code path with hardcoded tokenizer wiring. Treat the config-backed path as authoritative.
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

## Known Non-Reuse Zones

- `preprocess/preprocess_pipeline.ipynb` is workflow history, not canonical library code.
- Package-local preprocessing notebooks under variants are workflow history, not canonical reusable implementations.
- Test helper functions are scaffolding only; reuse the product functions they exercise instead.
