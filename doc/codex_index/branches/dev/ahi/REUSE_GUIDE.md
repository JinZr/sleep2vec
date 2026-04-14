# Reuse Guide

This page answers the practical question: when you need to add or change behavior, which implementation should you reuse first?

## Highest-Value Reuse Hotspots

| Responsibility | Canonical implementation to reuse | Why it is canonical | Do not bypass with |
| --- | --- | --- | --- |
| Pretrain YAML parsing | `sleep2vec.config.load_pretrain_config` | Enforces required model/loss/data blocks and typed config bundle creation | Hand-written YAML parsing in `pretrain.py` |
| Finetune YAML parsing | `sleep2vec.config.load_finetune_config` | Centralizes task, head, layer-mix, LoRA, and model-averaging parsing | Entry-point specific YAML parsing |
| Finetune CLI normalization | `sleep2vec.common.apply_finetune_config` | Binds YAML into `args`, enforces channel parity, derives task flags | Re-copying YAML fields in `finetune.py` or `infer.py` |
| Task semantics | `sleep2vec.common.apply_task_flags` | Single source for built-in labels, monitor selection, seq-label flags, and custom-task validation | Duplicated task/type/monitor logic |
| Registry-backed construction | `sleep2vec.builders.*`, `sleep2vec.registry.*` | All config-backed model assembly flows through here | Direct instantiation scattered across callers |
| Tokenizer instantiation | `sleep2vec.modules.tokenizers.build_tokenizer_from_channel` and `build_tokenizer_mapping` | Guarantees channel config is respected | Manual tokenizer maps |
| Projection creation | `sleep2vec.modules.projection.build_projection_head` | Central toggle for enabled/disabled projection heads | Ad hoc `SimCLRProjectionHead(...)` calls |
| Backbone encode path | `Sleep2vecPretrainModel._token_embeddings_to_hidden` | Single place that projects tokens, adds CLS, builds masks, runs encoder, and optionally exposes hidden states | Re-creating encoder + CLS plumbing in downstream code |
| Downstream feature path | `Sleep2vecDownstreamModel.forward` | Central path for per-modality encoding, temporal aggregation, channel fusion, layer mix, and head invocation | Parallel forward paths in trainer code |
| Pretrained backbone loading | `Sleep2vecDownstreamModel.load_pretrained_backbone` | Encodes prefix handling, EMA fallback, and CLS mismatch warnings | Custom checkpoint slicing logic |
| LoRA insertion | `Sleep2vecDownstreamModel.freeze_backbone_and_insert_lora` | Centralizes freeze policy and adapter insertion | Direct `peft` calls in trainer code |
| Pretrain data loaders | `sleep2vec.utils.get_pretrain_dataloader` | Owns missing-channel mode, per-pair validation loaders, and validation pair filtering | Building `PSGPretrainDataset` loaders manually in entrypoints |
| Finetune data loaders | `sleep2vec.utils.get_finetune_dataloaders` | Owns split/source choices and built-in seq pseudo-channel behavior (`stage5`, `ahi`) | Hand-rolled finetune loader creation |
| Sample validation | `data.utils.filter_valid_sample_indices` | Produces `payload["available_channels"]` and drops broken samples early | Custom preset-building loops |
| Runtime batch assembly | `DefaultDataset.dataloader` | Single source for collate-time NPZ reads, tokenization, metadata packing, `w/h` matrices, and sampler choice | New collate functions outside `data/default_dataset.py` |
| Missing-channel training batches | `PairFirstBatchSampler` | Canonical train-time sampler for pair-first missing-channel pretraining | Ad hoc pair scheduling loops |
| Validation pair filtering | `sleep2vec.utils._filter_dataset_for_pair_support` | Canonical filter for per-pair validation loaders | Duplicated support checks |
| Checkpoint averaging | `sleep2vec.checkpoints.select_checkpoints` and `average_checkpoints` | Encodes epoch-first selection plus fallback to mtime | Local checkpoint averaging scripts |
| Downstream metric reduction | `sleep2vec.metrics.compute_downstream_metrics` | Single metric reducer for classification, seq multi-label, and regression outputs | Per-stage custom metric calculations |
| Result CSV output | `sleep2vec.metrics.save_result_csv` | Preserves standard columns and append behavior | One-off CSV writers |
| Preset generation | `preprocess/save_dataset_presets.py` | Canonical CLI path that exercises `PSGPretrainDataset` and preset side effects | External scripts that pickle `SampleIndex` lists directly |
| Split generation | `preprocess/split_index_by_dataset.py` | Canonical dataset-group split policy | Manual split assignment notebooks |
| Missing-mask statistics | `preprocess/mask_missing_stats.py` | Canonical `_mask == 1` presence semantics | New mask-summary scripts with different conventions |
| WatchPAT conversion | `preprocess/watchpat_zzp_to_edf.convert_zzp_to_edf` | Single entrypoint for `.zzp` decoding and EDF writing | Parallel conversion scripts |

## Reuse Rules By Change Type

### If you are changing config semantics

- Reuse `load_pretrain_config`, `load_finetune_config`, and `apply_finetune_config`.
- Keep parse-time validation in `sleep2vec/config.py`.
- Keep CLI mutation in `sleep2vec/common.py`.
- Do not move semantic validation into entrypoints or tests.

### If you are changing model construction

- Reuse `build_encoder_factory`, `build_tokenizers_and_dim`, `build_projection`, `build_cls_embedding`, `build_temporal_aggregator`, and `build_channel_aggregator`.
- Register new implementations through registries instead of branching on names in runtime code.

### If you are changing batch or sampler behavior

- Reuse `filter_valid_sample_indices` for preset validation and `DefaultDataset.dataloader` for collate-time semantics.
- Reuse `PairFirstBatchSampler` or `AvailableChannelsBucketBatchSampler` instead of adding new sampler logic in `sleep2vec/utils.py`.
- Preserve `payload["available_channels"]` if missing-channel support is involved.

### If you are changing runtime orchestration

- Keep trainer/callback/wandb/checkpoint behavior in `pretrain.py`, `finetune.py`, `infer.py`, or the Lightning modules.
- Reuse `dump_cli_args_yaml`, `save_result_csv`, and checkpoint helpers instead of duplicating serialization and output logic.

### If you are changing preprocessing

- Prefer composing the existing CLI utilities:
  - `split_index_by_dataset.py`
  - `mask_missing_stats.py`
  - `save_dataset_presets.py`
  - `merge_dataset_presets.py`
- Only touch `watchpat_zzp_to_edf.py` for WatchPAT-specific conversion work.

## Major Duplication Risks

1. `Sleep2vecPretrainModel` still contains a legacy non-config code path with hardcoded tokenizer wiring. Treat the config-backed path as authoritative.
2. `_contrastive_accuracy` is duplicated in both contrastive loss modules.
3. Warmup-plus-cosine optimizer scheduling is duplicated between the pretrain and finetune Lightning modules.
4. Run-artifact writing (`config.yaml`, `cli_args.yaml`) is duplicated between `pretrain.py` and `finetune.py`; reuse `dump_cli_args_yaml` rather than creating new serializers.
5. Available-channel resolution is duplicated between `sleep2vec.utils` and `DefaultDataset` internals. Avoid creating a third implementation.
6. `_mask` column detection currently exists in more than one preprocessing script; keep semantics aligned if editing either location.
7. Config folder names are not authoritative semantics. Inspect actual `model.cls` and `model.head` fields before assuming behavior from directory names.

## Known Non-Reuse Zones

- `preprocess/preprocess_pipeline.ipynb` is workflow history, not canonical library code.
- Variant directories on this branch are not active code reuse targets.
- Test helper functions are scaffolding only; reuse the product functions they exercise instead.
