# System Overview

## Repository Shape

The repository is a config-driven multimodal sleep modeling system with four main operational layers:

1. Schema and runtime binding: `sleep2vec/config.py`, `sleep2vec/common.py`
2. Construction and extension: `sleep2vec/registry.py`, `sleep2vec/builders.py`, `sleep2vec/backbones/`, `sleep2vec/modules/`, `sleep2vec/cls/`, `sleep2vec/downstreams/`
3. Model and trainer runtime: `sleep2vec/pretrain_model.py`, `sleep2vec/downstream_model.py`, `sleep2vec/sleep2vec_modelling.py`, `sleep2vec/sleep2vec_finetuning.py`, CLI entrypoints
4. Data and preprocessing: `data/`, `preprocess/`, `sleep2vec/utils.py`

The top-level product behavior is not encoded in YAML alone. YAML defines model, loss, task, head, and data references; CLI entrypoints still inject runtime-only values such as learning rate, devices, checkpoint paths, diagnostics mode, and experiment naming.

## Primary Entrypoints

- `python -m sleep2vec.pretrain`
- `python -m sleep2vec.finetune`
- `python -m sleep2vec.infer`
- Preprocessing CLIs under `preprocess/`

## Runtime Stack

### Pretrain

`pretrain.py` performs the orchestration:

1. Parse CLI.
2. Load YAML with `load_pretrain_config`.
3. Copy selected config fields back into `args`.
4. Build train loader plus one validation loader per channel pair through `get_pretrain_dataloader`.
5. Instantiate `Sleep2vecPretraining`, which owns a `Sleep2vecPretrainModel`, contrastive loss, diagnostics hooks, and optional model averager.
6. Configure Lightning callbacks:
   - `ModelCheckpoint`
   - `EarlyStopping`
   - `LearningRateMonitor`
   - `PairAccLoggerCallback`
7. Persist `config.yaml` and `cli_args.yaml` next to checkpoints.
8. Train with `trainer.fit(...)`.

The monitored validation metric is `val_contrastive_acc`.

### Finetune

`finetune.py` follows a different normalization path:

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
6. Run `trainer.fit(...)`.
7. Test the best checkpoint or requested checkpoint with `trainer.test(...)`.
8. Append evaluation metrics to a CSV with `save_result_csv`.

### Inference

`infer.py` reuses the finetune configuration path:

1. Parse CLI.
2. Validate explicit checkpoint path unless the alias is `best` or `last`.
3. Call `apply_finetune_config(args)`.
4. Build a single eval loader with `_build_inference_loader`.
5. Instantiate `Sleep2vecFinetuning`.
6. Optionally select and average checkpoints with `select_checkpoints` and `average_checkpoints`.
7. Run `trainer.test(...)`.
8. Optionally write `results.csv`.

This is evaluation-only orchestration; checkpoint selection and averaging live in `sleep2vec/checkpoints.py`, not in the trainer classes.

## Core Data Contract

The runtime assumes a batch dictionary with these keys:

| Key | Producer | Consumer | Notes |
| --- | --- | --- | --- |
| `id` | `DefaultDataset.dataloader` | logging only | Sample ids |
| `length` | `DefaultDataset.dataloader` | backbone, downstream, losses | Token count per sample before padding |
| `tokens` | `DefaultDataset.dataloader` | backbone, finetune loss | Channel-name keyed token tensors |
| `mlm_mask` | `DefaultDataset.dataloader` | pretrain backbone | Channel-name keyed boolean mask tensors |
| `metadata` | `DefaultDataset.dataloader` | finetune loss, metrics, negative weighting | Includes `age`, `sex`, `source`, `path`, and requested labels |
| `w` | `DefaultDataset.dataloader` | `WeightedInfoNCELoss` | Negative-sample weight matrix |
| `h` | `DefaultDataset.dataloader` | `WeightedInfoNCELoss` | Same-path hardness mask |
| `pair` | `DefaultDataset.dataloader` | callbacks/logging | Present for pair-first batches |

### Important Invariants

- All configured channels must share one tokenizer output dimension.
- `model.cls.downstream='cls'` requires a real CLS strategy such as `bert`.
- Built-in sequence tasks are `stage3`, `stage4`, `stage5`, and `ahi`.
- `stage3`, `stage4`, and `stage5` consume raw `batch["tokens"]["stage5"]`; `ahi` consumes raw `batch["tokens"]["ahi"]`.
- `ahi` targets are per-token `[30]` binary slices padded with `-1.0`; loss and metrics flatten only valid positions.
- Non-built-in sequence labels are unsupported, and non-built-in metadata classification remains binary-only.
- Pair-first missing-channel pretraining requires `payload["available_channels"]` to be present on every sample.
- Weighted InfoNCE requires both `w` and `h` in the batch.

## Construction Model

The canonical creation path is:

1. Parse YAML into dataclasses with `load_pretrain_config` or `load_finetune_config`.
2. Validate or bind runtime args with `apply_finetune_config` when applicable.
3. Resolve extensions through registries:
   - backbone registry
   - tokenizer registry
   - projection registry
   - model averager registry
4. Build the encoder factory, tokenizer mapping, and projection head via `builders.py`.
5. Build CLS behavior via `build_cls_embedding`.
6. Build downstream temporal/channel/head composition through `sleep2vec/downstreams/`.

## Preprocessing System

The preprocessing surface is split between reusable CLIs and one notebook:

- `split_index_by_dataset.py`: assign `train/val/test/external`
- `mask_missing_stats.py`: summarize `_mask` coverage
- `save_dataset_presets.py`: build preset pickles by instantiating `PSGPretrainDataset`
- `merge_dataset_presets.py`: concatenate multiple preset pickles
- `watchpat_zzp_to_edf.py`: convert WatchPAT `.zzp` archives to EDF and optional JSON summary
- `preprocess_pipeline.ipynb`: manual, dataset-specific workflow history

The canonical preset path is:

`CSV split prep -> optional mask analysis -> preset generation -> optional preset merge`

## Outputs And Side Effects

- Pretrain checkpoints: `log-pretrain/<run>/checkpoints/`
- Finetune checkpoints: `log-finetune/<version>/checkpoints/`
- Per-run copied configs: `config.yaml`
- Per-run CLI snapshot: `cli_args.yaml`
- Downstream results table: caller-specified CSV via `save_result_csv`
- Preprocessing outputs:
  - preset pickles
  - split CSVs
  - mask statistics CSVs
  - EDF files and optional JSON summaries

## Variant State On This Branch

`sleep2vec2/`, `sleep2vec_moe/`, and `sleep2vec_hires/` exist as directories but contain no tracked source files on `main`. Treat them as branch-state notes, not active extension targets, unless tracked code appears in a later commit.
