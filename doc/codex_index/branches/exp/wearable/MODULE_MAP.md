# Module Map

## Top-Level Areas

| Area | Role | Key Files | Notes |
| --- | --- | --- | --- |
| `sleep2vec/` | Runtime entrypoints, config parsing, model construction, Lightning modules, checkpoints, metrics | `pretrain.py`, `adapt.py`, `finetune.py`, `infer.py`, `config.py`, `common.py`, `pretrain_model.py`, `downstream_model.py`, `sleep2vec_modelling.py`, `sleep2vec_adaptation.py`, `sleep2vec_finetuning.py`, `checkpoints.py` | Main source of truth for training and evaluation behavior |
| `data/` | Sample indexing, dataset filtering, collate path, metadata handling, samplers | `default_dataset.py`, `psg_pretrain_dataset.py`, `samplers.py`, `utils.py`, `metadata.py`, `channel_selection.py` | Branch changes here are tightly coupled to wearable and missing-channel behavior |
| `preprocess/` | CSV splitting, preset generation, preset merging, missing-channel statistics, WatchPAT conversion | `save_dataset_presets.py`, `merge_dataset_presets.py`, `split_index_by_dataset.py`, `mask_missing_stats.py`, `watchpat_zzp_to_edf.py` | `save_dataset_presets.py` is now YAML-aware and should stay aligned with runtime channel contracts |
| `configs/` | Recipe definitions | `sleep2vec_dense_pretrain.yaml`, `sleep2vec_dense_adapt_ppg_actigraphy.yaml`, `sleep2vec_large_pretrain.yaml`, downstream configs under `cls_emb/` and `token_emb/` | Architecture and task selection live here; CLI should not duplicate them |
| `tests/` | Contract coverage | `test_config_loading.py`, `test_adapt.py`, `test_adapt_pair_schedule_callback.py`, `test_adaptation.py`, `test_bucket_sampler.py`, `test_generic_channel_dataset.py`, `test_save_dataset_presets.py`, `test_pair_first_sampler.py`, `test_checkpoints.py` | Strongest branch-specific safety net is around adaptation and heterogeneous-channel loaders |
| `utils/` | Repository helpers | `style_check.sh` is the main operational helper mentioned in repo docs | No branch-specific product logic found here |

## Dependency Map

### Config and runtime plumbing

- `sleep2vec/config.py` parses YAML into dataclasses.
- `sleep2vec/common.py` applies parsed config onto CLI namespace and persists run snapshots.
- Entry points (`pretrain.py`, `adapt.py`, `finetune.py`, `infer.py`) call into `config.py` and `common.py` before they touch the model or data stack.

### Dataset assembly

- `sleep2vec/utils.py` is the runtime loader factory layer.
- `sleep2vec/utils.get_pretrain_dataloader` and `sleep2vec/utils.get_finetune_dataloaders` create `PSGPretrainDataset`.
- `data/psg_pretrain_dataset.py` translates CSV rows or preset payloads into `SampleIndex`.
- `data/default_dataset.py` owns filtering, few-shot selection, collate-time tensor materialization, metadata packaging, and batch-sampler selection.
- `data/samplers.py` owns the custom batch samplers that Lightning must not re-shard a second time.

### Model construction

- `sleep2vec/builders.py` and `sleep2vec/registry.py` are the canonical extension surface for backbones, tokenizers, projection heads, and model averagers.
- `sleep2vec/pretrain_model.py` builds the shared tokenizer -> encoder -> projection path.
- `sleep2vec/downstream_model.py` reuses `Sleep2vecPretrainModel` and attaches downstream heads, layer mixing, and LoRA.

### Training modules

- `sleep2vec/sleep2vec_modelling.py` wraps contrastive pretraining.
- `sleep2vec/sleep2vec_adaptation.py` subclasses the pretraining module for staged adaptation.
- `sleep2vec/sleep2vec_finetuning.py` wraps downstream task training and evaluation.

### Evaluation and artifacts

- `sleep2vec/checkpoints.py` is the canonical checkpoint averaging and checkpoint-init loader.
- `sleep2vec/metrics.py` is the canonical metric aggregation and results-CSV writer.
- `sleep2vec/callbacks/pair_acc_logger.py` is the canonical pair-level visibility hook for pretraining and adaptation.

## Canonical Files By Concern

### YAML and task semantics

- Canonical: `sleep2vec/config.py`, `sleep2vec/common.py`
- Avoid duplicating: ad-hoc YAML reads in entrypoints or preprocess scripts beyond `save_dataset_presets.py`

### Channel and input-dimension semantics

- Canonical: `sleep2vec/common.channel_input_dims_from_model_config`, `data.psg_pretrain_dataset._build_channel_registry`
- Avoid duplicating: hard-coded channel registries in runtime code

### Missing-channel pretraining

- Canonical: `sleep2vec/utils.get_pretrain_dataloader`, `data.default_dataset.DefaultDataset.dataloader`, `data.samplers.PairFirstBatchSampler`, `data.samplers.AvailableChannelsBucketBatchSampler`
- Avoid duplicating: custom pair sampling inside a new loader or callback

### Adaptation phase semantics

- Canonical: `sleep2vec/adapt.py`, `sleep2vec/sleep2vec_adaptation.py`
- Avoid duplicating: stage directory naming, checkpoint reuse, or pair-schedule logic elsewhere

### Downstream checkpoint init and averaging

- Canonical: `sleep2vec/checkpoints.py`
- Avoid duplicating: direct state-dict filtering in finetune or infer code

## Tracked Absences

- `sleep2vec2/`: no tracked source files on this branch
- `sleep2vec_moe/`: no tracked source files on this branch
- `sleep2vec_hires/`: no tracked source files on this branch

These directories should be treated as unavailable for branch-local reuse analysis.
