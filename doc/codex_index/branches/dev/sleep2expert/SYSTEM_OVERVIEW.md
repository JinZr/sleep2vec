# System Overview

## Repository Shape

The repository is a config-driven multimodal sleep modeling system with five operational layers:

1. Schema and task semantics: `sleep2vec/config.py`, `sleep2vec/common.py`
2. Construction and extension: `sleep2vec/registry.py`, `sleep2vec/builders.py`, `sleep2vec/backbones/`, `sleep2vec/modules/`, `sleep2vec/cls/`, `sleep2vec/downstreams/`
3. Model and trainer runtime: `sleep2vec/pretrain_model.py`, `sleep2vec/downstream_model.py`, `sleep2vec/sleep2vec_modelling.py`, `sleep2vec/sleep2vec_finetuning.py`, `sleep2vec/sleep2vec_adaptation.py`
4. Data and preprocessing: `data/`, `preprocess/`, `sleep2vec/utils.py`, plus mirrored `sleep2vec2/data/` and `sleep2vec2/preprocess/`
5. Runtime support and tooling: `sleep2vec/checkpoints.py`, `sleep2vec/results.py`, `sleep2vec/distributed.py`, `sleep2vec/visualization/`, `utils/check_configs.py`

Top-level behavior is not encoded in YAML alone. YAML defines model, loss, task, head, evaluation-visualization, and adapt references; entrypoints still inject runtime-only values such as learning rate, devices, checkpoint paths, diagnostics mode, and experiment naming.

## Primary Entrypoints

- `python -m sleep2vec.pretrain`
- `python -m sleep2vec.adapt`
- `python -m sleep2vec.finetune`
- `python -m sleep2vec.infer`
- `python utils/check_configs.py`
- Preprocessing CLIs under `preprocess/`
- `python -m sleep2vec2.pretrain`, `adapt`, `finetune`, and `infer` for the standalone mirror recipe
- Preprocessing CLIs under `sleep2vec2.preprocess`

## Runtime Stack

### Pretrain

`pretrain.py` performs the orchestration:

1. Parse CLI.
2. Load YAML with `load_pretrain_config`.
3. Copy model-derived channel metadata into `args`.
4. Build one training loader plus one validation loader whose batch sampler iterates channel pairs through `get_pretrain_dataloader`.
5. Instantiate `Sleep2vecPretraining`, which owns a `Sleep2vecPretrainModel`, contrastive loss, diagnostics hooks, and optional model averager.
6. Configure Lightning callbacks:
   - `ModelCheckpoint`
   - `EarlyStopping`
   - `LearningRateMonitor`
   - `PairAccLoggerCallback`
7. Persist `config.yaml` and `cli_args.yaml`.
8. Train with `trainer.fit(...)`.

The monitored validation metric is `val_contrastive_acc`.

### Adapt

`adapt.py` is a staged pretrain variant for adding modalities to an existing backbone:

1. Parse CLI, including `--phase stage1|stage2`.
2. Load pretrain-style YAML and require a top-level `adapt` block.
3. Derive initial pair probabilities from `adapt.new_channels` and `adapt.stage2.pair_schedule`.
4. Build missing-channel-aware train/validation loaders with `get_pretrain_dataloader`.
5. Resolve or validate the experiment directory and phase-specific checkpoint layout via `_resolve_adapt_run_artifacts`.
6. Persist root-level and phase-specific `config*.yaml` / `cli_args*.yaml`.
7. Instantiate `Sleep2vecAdaptation`, which wraps `Sleep2vecPretraining` but applies adaptation freeze policy and stage-specific optimizer grouping.
8. Train with optional `AdaptPairScheduleCallback` during stage 2.

Stage transitions are strict: `--ckpt-path` resumes within the same phase only, while `--pretrained-backbone-path` is used for weight initialization and for stage1 -> stage2 transitions.

### Finetune

`finetune.py` normalizes downstream configuration and launches supervised training:

1. Parse CLI.
2. Call `apply_finetune_config(args)`.
3. That call loads YAML via `load_finetune_config`, validates task semantics, converts configured data paths to `Path`, and enforces `data.data_channel_names == model.channels`.
4. Build train/val/test loaders via `get_finetune_dataloaders`.
5. Instantiate `Sleep2vecFinetuning`, which owns:
   - a `Sleep2vecPretrainModel` backbone
   - a `Sleep2vecDownstreamModel` head stack
   - optional pretrained-backbone loading
   - optional LoRA insertion
   - optional model averaging
   - optional downstream evaluation visualization hooks
6. Run `trainer.fit(...)`.
7. Test the best checkpoint or requested checkpoint with `trainer.test(...)`.
8. Append evaluation metrics to a CSV via `sleep2vec.results.save_result_csv`.

Built-in task semantics now include `stage3`, `stage4`, `stage5`, `ahi`, `sex`, and `age`.

### Inference

`infer.py` reuses the finetune configuration path:

1. Parse CLI and validate explicit checkpoint path unless the alias is `best` or `last`.
2. Call `apply_finetune_config(args)`.
3. Build a single evaluation loader with `_build_inference_loader`.
4. Instantiate `Sleep2vecFinetuning`.
5. Optionally select and average checkpoints with `select_checkpoints` and `average_checkpoints`.
6. Run `trainer.test(...)`.
7. Optionally append metrics to `results.csv`.

AHI inference deliberately rejects checkpoint averaging because the fitted `ahi_eval_threshold` is checkpoint-specific.

## Core Data Contract

The runtime assumes a batch dictionary with these keys:

| Key | Producer | Consumer | Notes |
| --- | --- | --- | --- |
| `id` | `DefaultDataset.dataloader` | logging only | Sample ids |
| `length` | `DefaultDataset.dataloader` | backbone, downstream heads, losses | Token count per sample before padding |
| `token_start` | `DefaultDataset.dataloader` | AHI event aggregation | Preserves window offset for later record merging |
| `tokens` | `DefaultDataset.dataloader` | backbone, finetune loss, AHI eval | Channel-name keyed padded token tensors |
| `mlm_mask` | `DefaultDataset.dataloader` | pretrain backbone | Channel-name keyed boolean mask tensors |
| `metadata` | `DefaultDataset.dataloader` | finetune loss, metrics, negative weighting | Includes `age`, `sex`, `source`, `path`, and requested labels; built-in AHI also backfills `ahi` and `tst` |
| `w` | `DefaultDataset.dataloader` | `WeightedInfoNCELoss` | Negative-sample weight matrix |
| `h` | `DefaultDataset.dataloader` | `WeightedInfoNCELoss` | Same-path hardness mask |
| `pair` | `DefaultDataset.dataloader` | callbacks/logging | Present for pair-first or sequential pair-eval batches |

### Important Invariants

- All configured channels must share one tokenizer output dimension.
- `model.cls.downstream='cls'` requires a real CLS strategy such as `bert`.
- Built-in sequence tasks are `stage3`, `stage4`, `stage5`, and `ahi` only.
- `stage3` and `stage4` are runtime remaps over raw `stage5` tokens; their source labels still come from `stage5`.
- Built-in `ahi` requires NPZ key `ah_event` plus scalar NPZ keys `ahi` and `tst`; evaluation also expects `stage5` tokens as an auxiliary label source.
- Non-sequence metadata classification remains binary-only.
- Pair-first missing-channel pretraining requires `payload["available_channels"]` on every retained sample.
- Weighted InfoNCE requires both `w` and `h` in the batch.

## Construction Model

The canonical creation path is:

1. Parse YAML into dataclasses with `load_pretrain_config`, `load_finetune_config`, or `load_model_config`.
2. Bind runtime args with `apply_model_config_args` or `apply_finetune_config`.
3. Resolve extensions through registries:
   - backbone registry
   - tokenizer registry
   - projection registry
   - model averager registry
4. Build the encoder factory, tokenizer mapping, and projection head via `builders.py`.
5. Build CLS behavior via `build_cls_embedding`.
6. Build downstream temporal/channel/head composition through `sleep2vec/downstreams/`.

## Preprocessing And Config Validation

The preprocessing surface is split between reusable CLIs and one notebook:

- `split_index_by_dataset.py`: assign `train/val/test/external`, normalize mask truthiness, optionally enforce global pair coverage
- `mask_missing_stats.py`: summarize `_mask` coverage
- `save_dataset_presets.py`: build preset pickles through `PSGPretrainDataset`, including YAML-driven `preset_build` validation
- `merge_dataset_presets.py`: concatenate multiple preset pickles
- `watchpat_zzp_to_edf.py`: convert WatchPAT `.zzp` archives to EDF and optional JSON summary
- `preprocess_pipeline.ipynb`: manual, dataset-specific workflow history

The canonical preset path is:

`CSV split prep -> optional mask analysis -> preset generation -> optional preset merge`

`utils/check_configs.py` is the branch-local tooling path for validating:

- runtime config loader compatibility
- shared tokenizer-dimension parity
- YAML `preset_build` strictness
- repo-specific `ppg_*finetune*.yaml` contracts

## Outputs And Side Effects

- Pretrain checkpoints: `log-pretrain/<run>/checkpoints/`
- Adapt checkpoints:
  - stage1: `log-adapt/<run>/checkpoints/`
  - stage2: `log-adapt/<run>/checkpoints.stage2/`
- Finetune checkpoints: `log-finetune/<version>/checkpoints/`
- Per-run copied configs: `config.yaml`, plus phase-specific `config.stage1.yaml` / `config.stage2.yaml` for adaptation
- Per-run CLI snapshot: `cli_args.yaml`, plus phase-specific adaptation snapshots
- Downstream results table: caller-specified CSV via `sleep2vec.results.save_result_csv`
- Visualization side effects:
  - pair-accuracy heatmaps
  - layer-mix heatmaps and tables
  - confusion matrices, ROC curves, and regression scatter plots
- Preprocessing outputs:
  - preset pickles
  - split CSVs
  - mask statistics CSVs
  - EDF files and optional JSON summaries

## Variant State On This Branch

`sleep2vec2/` is active on this branch as a standalone mirror of the base recipe. It carries package-local copies of the base runtime, `data/`, and `preprocess/`, duplicated YAMLs under `configs/sleep2vec2/`, and a copied standalone RoFormer implementation under `sleep2vec2/backbones/roformer/`.

`sleep2vec_moe/` and `sleep2vec_hires/` remain branch-state placeholders with no tracked source files here.
