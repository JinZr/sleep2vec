# Reuse Guide

This page answers the practical question: when you need to add or change behavior, which implementation should you reuse first?

## Highest-Value Reuse Hotspots

| Responsibility | Canonical implementation to reuse | Why it is canonical | Do not bypass with |
| --- | --- | --- | --- |
| Pretrain YAML parsing | `sleep2vec.config.load_pretrain_config` | Enforces required model/loss/data blocks and typed config bundle creation | Hand-written YAML parsing in `pretrain.py` |
| Finetune YAML parsing | `sleep2vec.config.load_finetune_config` | Centralizes task, head, layer-mix, LoRA, and model-averaging parsing | Entry-point specific YAML parsing |
| Built-in task semantics | `sleep2vec.common.apply_task_flags` plus the built-in task helper family in `sleep2vec.common` | Single source for built-in labels, supported AHI monitor-switch semantics, label sources, class labels, stage remaps, and auxiliary label channels | Re-copying label semantics into `finetune.py`, `infer.py`, or trainer code |
| Finetune CLI normalization | `sleep2vec.common.apply_finetune_config` | Binds YAML into `args`, enforces channel parity, and applies built-in task semantics | Partial YAML binding in entrypoints |
| Registry-backed construction | `sleep2vec.builders.*`, `sleep2vec.registry.*` | All config-backed model assembly flows through here | Direct instantiation scattered across callers |
| Tokenizer instantiation | `sleep2vec.modules.tokenizers.build_tokenizer_from_channel` and `build_tokenizer_mapping` | Guarantees channel config is respected | Manual tokenizer maps |
| Projection creation | `sleep2vec.modules.projection.build_projection_head` | Central toggle for enabled/disabled projection heads | Ad hoc `SimCLRProjectionHead(...)` calls |
| Backbone encode path | `Sleep2vecPretrainModel._token_embeddings_to_hidden` | Single place that projects tokens, adds CLS, builds masks, runs encoder, and optionally exposes hidden states | Re-creating encoder + CLS plumbing in downstream code |
| Downstream feature path | `Sleep2vecDownstreamModel.forward` | Central path for per-modality encoding, temporal aggregation, channel fusion, layer mix, and head invocation | Parallel forward paths in trainer code |
| Pretrained backbone loading | `Sleep2vecDownstreamModel.load_pretrained_backbone` | Encodes prefix handling, EMA fallback, and CLS mismatch warnings | Custom checkpoint slicing logic |
| LoRA insertion | `Sleep2vecDownstreamModel.freeze_backbone_and_insert_lora` | Centralizes freeze policy and adapter insertion | Direct `peft` calls in trainer code |
| Pretrain data loaders | `sleep2vec.utils.get_pretrain_dataloader` | Owns missing-channel mode, pair-first training, and validation loader construction | Building `PSGPretrainDataset` loaders manually in entrypoints |
| Finetune data loaders | `sleep2vec.utils.get_finetune_dataloaders` and `_build_finetune_loader` | Own split/source choices and built-in sequence pseudo-channel behavior (`stage5`, `ahi`) plus the built-in AHI summary metadata (`ahi`, `tst`) and required auxiliary `stage5` stream for final evaluation | Hand-rolled finetune loader creation |
| Sample validation | `data.utils.filter_valid_sample_indices` | Produces `payload["available_channels"]`, persists built-in AHI scalars, and drops broken or all-ignored built-in AHI samples early | Custom preset-building loops |
| Runtime batch assembly | `DefaultDataset.dataloader` | Single source for collate-time NPZ reads, tokenization, metadata packing, `w/h` matrices, pair tagging, and ignore-value padding for runtime label channels | New collate functions outside `data/default_dataset.py` |
| Missing-channel training batches | `PairFirstBatchSampler` | Canonical train-time sampler for pair-first missing-channel pretraining | Ad hoc pair scheduling loops |
| Validation pair filtering | `sleep2vec.utils._filter_dataset_for_pair_support` | Canonical filter for per-pair validation loaders | Duplicated support checks |
| Distributed AHI finetune progress bar | `sleep2vec.callbacks.progress_bar.build_distributed_ahi_progress_bar` | Keeps the built-in batch-level progress bar while skipping the rank-zero-only train-epoch-end UI work that skews later hooks under DDP | Re-embedding custom progress-bar subclasses in `finetune.py` or adding post-hoc barrier callbacks |
| Checkpoint averaging | `sleep2vec.checkpoints.select_checkpoints` and `average_checkpoints` | Encodes epoch-first selection plus fallback to mtime | Local checkpoint averaging scripts |
| Non-AHI downstream metric reduction | `sleep2vec.metrics.compute_downstream_metrics` | Canonical reducer for multiclass classification and regression outputs | Per-task custom metric calculations in trainer code |
| AHI token-level pointwise metrics | `sleep2vec.metrics.compute_ahi_pointwise_metrics` | Keeps token-level AHI binary metrics namespaced when a caller explicitly needs array-based pointwise summaries | Reusing generic binary metric names in ad hoc callers |
| AHI final validation/test/infer metrics | `sleep2vec.metrics.compute_ahi_event_metrics` | Centralizes event matching, validation threshold fitting, TST gating, and the split between detection-style event stats and NPZ-aligned scalar AHI summaries | Re-deriving AHI event logic inside `Sleep2vecFinetuning` |
| Result CSV output | `sleep2vec.results.save_result_csv` | Preserves accumulated rows in one CSV, stamps each row with `experiment_version`, keeps rank-zero-only write behavior under DDP, and serializes concurrent writers with a file lock | One-off CSV writers |
| Preset validation channel resolution | `preprocess.save_dataset_presets._resolve_validation_channels` | Owns YAML-vs-built-in channel selection, including built-in `stage5` / `ahi` validation channels and automatic `ahi -> stage5` expansion | Duplicated channel subset logic in wrapper scripts |
| AHI preset admission threshold | `preprocess.save_dataset_presets._resolve_effective_min_channels` | Forces built-in `ahi` presets to require every requested validation channel before serializing windows | Ad hoc `min_channels` overrides in preset scripts |
| Preset required-mask prefilter | `preprocess.save_dataset_presets._filter_index_df_for_required_channels` | Applies strict mask-based CSV prefiltering with built-in `stage_mask` / `ah_event_mask` handling when missing channels are disallowed | Manual CSV filtering before preset generation |
| Preset generation | `preprocess.save_dataset_presets.main` and `_build_preset_job` | Canonical CLI path that exercises `PSGPretrainDataset` and preset side effects | External scripts that pickle `SampleIndex` lists directly |
| Split generation | `preprocess/split_index_by_dataset.py` | Canonical dataset-group split policy | Manual split assignment notebooks |
| Missing-mask statistics | `preprocess/mask_missing_stats.py` | Canonical `_mask == 1` presence semantics | New mask-summary scripts with different conventions |
| WatchPAT conversion | `preprocess/watchpat_zzp_to_edf.convert_zzp_to_edf` | Single entrypoint for `.zzp` decoding and EDF writing | Parallel conversion scripts |

## Reuse Rules By Change Type

### If you are changing config semantics

- Reuse `load_pretrain_config`, `load_finetune_config`, `apply_finetune_config`, and `apply_task_flags`.
- Keep parse-time validation in `sleep2vec/config.py`.
- Keep CLI mutation and built-in task attribute derivation in `sleep2vec/common.py`.
- Do not move semantic validation into entrypoints or tests.

### If you are changing built-in sequence task behavior

- Reuse the built-in task helper family in `sleep2vec.common`.
- Keep sleep-stage remapping in `remap_stage_labels`.
- Keep raw `ahi` handling separate from sleep-stage remaps; `ahi` consumes runtime `batch["tokens"]["ahi"]` built from NPZ `ah_event`, requires `batch["tokens"]["stage5"]` as an auxiliary runtime stream for final masking, uses scalar NPZ summaries `ahi` and `tst`, and always runs the full validation event-eval path so the fitted `ahi_eval_threshold` can be persisted into checkpoints.
- Do not invent a second task registry in entrypoints or trainer code.

### If you are changing model construction

- Reuse `build_encoder_factory`, `build_tokenizers_and_dim`, `build_projection`, `build_cls_embedding`, `build_temporal_aggregator`, and `build_channel_aggregator`.
- Register new implementations through registries instead of branching on names in runtime code.

### If you are changing batch or sampler behavior

- Reuse `filter_valid_sample_indices` for preset validation and `DefaultDataset.dataloader` for collate-time semantics.
- Reuse `PairFirstBatchSampler` or `AvailableChannelsBucketBatchSampler` instead of adding new sampler logic in `sleep2vec/utils.py`.
- Preserve `payload["available_channels"]` if missing-channel support is involved.

### If you are changing runtime orchestration

- Keep trainer/callback/wandb/checkpoint behavior in `pretrain.py`, `finetune.py`, `infer.py`, or the Lightning modules.
- Keep reusable callback implementations in `sleep2vec/callbacks/`; entrypoints should only decide when to install them.
- Reuse `dump_cli_args_yaml`, `save_result_csv`, and checkpoint helpers instead of duplicating serialization and output logic.
- Keep experiment-row tagging and rank-zero gating inside `sleep2vec.results.save_result_csv`; do not scatter per-entrypoint CSV-write guards or ad hoc version columns. The current writer is single-node scoped; do not infer multi-node correctness from it.
- Reuse `sleep2vec.distributed.is_rank_zero_process` only for single-node process-level env-based rank checks; keep `trainer.is_global_zero` branches local to Lightning runtime code.
- Reuse `sleep2vec.distributed.is_torch_distributed_ready` and `get_rank_world_size` for generic torch.distributed readiness / `(rank, world_size)` fallbacks; do not keep cloning `dist.is_available() and dist.is_initialized()` or local `_get_dist_info()` helpers.
- For `ahi`, reuse `compute_ahi_event_metrics` for val/test/infer event evaluation and checkpoint-threshold reuse. Train-time AHI pointwise metrics should stay on the reduced confusion-count path inside `Sleep2vecFinetuning` instead of rebuilding epoch-wide token arrays; log accuracy/precision/recall/F1 from globally reduced counts and keep train ROC-AUC unsupported.

### If you are changing preprocessing

- Prefer composing the existing CLI utilities:
  - `split_index_by_dataset.py`
  - `mask_missing_stats.py`
  - `save_dataset_presets.py`
  - `merge_dataset_presets.py`
- Keep built-in validation-channel logic in `save_dataset_presets.py`.
- Only touch `watchpat_zzp_to_edf.py` for WatchPAT-specific conversion work.

## Major Duplication Risks

1. `Sleep2vecPretrainModel` still contains a legacy non-config code path with hardcoded tokenizer wiring. Treat the config-backed path as authoritative.
2. `_contrastive_accuracy` is duplicated in both contrastive loss modules.
3. Warmup-plus-cosine optimizer scheduling is duplicated between the pretrain and finetune Lightning modules.
4. Run-artifact writing (`config.yaml`, `cli_args.yaml`) is duplicated between `pretrain.py` and `finetune.py`; reuse `dump_cli_args_yaml` rather than creating new serializers.
5. Available-channel resolution is duplicated between `sleep2vec.utils`, `DefaultDataset`, and preset prefilter helpers. Avoid creating a fourth implementation.
6. `_mask` column detection currently exists in more than one preprocessing script; keep semantics aligned if editing either location.
7. Config folder names are not authoritative semantics. Inspect actual `model.cls`, `model.head`, and built-in task flags before assuming behavior from directory names.

## Known Non-Reuse Zones

- `preprocess/preprocess_pipeline.ipynb` is workflow history, not canonical library code.
- Variant directories on this branch are not active code reuse targets.
- Test helper functions are scaffolding only; reuse the product functions they exercise instead.
