# System Overview

## Core Shape

`sleep2vec` is a config-driven training stack for multi-channel sleep signals.

- YAML under `configs/` selects model structure, tokenizers, projection head, downstream head, and loss behavior.
- CLI flags set runtime knobs such as devices, learning rate, checkpoint paths, diagnostics, and worker counts.
- `data/` turns CSV rows or preset pickles into `SampleIndex` windows, then materializes token tensors inside the dataloader collate path.
- `sleep2vec/pretrain_model.py` is the shared backbone wrapper used by both contrastive pretraining and downstream finetuning.
- `sleep2vec/sleep2vec_modelling.py`, `sleep2vec/sleep2vec_adaptation.py`, and `sleep2vec/sleep2vec_finetuning.py` are the Lightning modules that attach training logic around the shared model.

## Main Runtime Flows

### Pretrain

`sleep2vec/pretrain.py` loads a pretrain YAML bundle, copies the resolved config and CLI args into the run directory, builds missing-channel-aware dataloaders when requested, constructs `Sleep2vecPretraining`, and trains with Lightning callbacks for checkpoints, early stopping, LR monitoring, and pair-accuracy logging.

### Adapt

`sleep2vec/adapt.py` is a branch-specific entrypoint for staged modality adaptation.

- Stage 1 focuses on new modalities and optionally the shared projection.
- Stage 2 reuses the stage-1 experiment directory, switches checkpoint output to `checkpoints.stage2`, unfreezes more of the backbone, and changes pair sampling over training progress.
- `sleep2vec/sleep2vec_adaptation.py` owns the phase-specific optimizer groups and the training-pair schedule callback.

### Finetune

`sleep2vec/finetune.py` loads a finetune YAML, applies built-in or YAML-defined task semantics, builds train/val/test loaders, constructs `Sleep2vecFinetuning`, optionally loads pretrained backbone weights, and writes a stable `best.ckpt` copy after training.

### Inference

`sleep2vec/infer.py` reuses the finetune config path, optionally averages several checkpoints, runs test-only evaluation, optionally logs to W&B, and appends summary metrics to a CSV.

## Data Contracts

The branch now relies on YAML `model.channels` as the authoritative declaration for runtime channels and per-token input widths.

- `sleep2vec/common.py` copies those channel names and `input_dim` values into argparse state.
- `data/psg_pretrain_dataset.py` refuses to build non-`stage5` channels without explicit `channel_input_dims`.
- `preprocess/save_dataset_presets.py` also requires YAML `model.channels` and rejects unknown `--channels`.

Missing-channel pretraining is handled by three linked components:

1. `data.utils.filter_valid_sample_indices` records `payload["available_channels"]` when filtering preset candidates.
2. `data.default_dataset.DefaultDataset.dataloader` changes collate behavior when `allow_missing_channels=True`.
3. `data.samplers.PairFirstBatchSampler` or `data.samplers.AvailableChannelsBucketBatchSampler` shape batch composition to prevent pair collapse.

## Model Stack

### Shared backbone

`sleep2vec/pretrain_model.py` owns:

- channel tokenizer construction
- encoder factory use
- optional projection head
- CLS handling strategy
- masking and token-to-hidden conversion
- adaptation freeze-state helpers used by the branch adaptation flow

### Downstream head path

`sleep2vec/downstream_model.py` wraps the shared backbone with:

- temporal aggregation
- channel aggregation inside the selected head
- optional layer mix over encoder hidden states
- optional LoRA insertion into the encoder
- checkpoint-init logic for downstream training

### Metrics and callbacks

- `sleep2vec/metrics.py` computes downstream metrics and appends results to CSV.
- `sleep2vec/callbacks/pair_acc_logger.py` tracks pair-level validation accuracy and train-time pair-sampling diagnostics.

## Branch-Specific Additions

Compared with `main`, this branch adds a first-class wearable adaptation path:

- config schema: `AdaptConfig`, stage-specific LR scales, and pair-schedule validation
- runtime: `sleep2vec.adapt`
- training module: `Sleep2vecAdaptation`
- configs: wearable adaptation YAMLs plus larger pretrain recipes
- tests: adapt runtime, pair schedule, generic channel datasets, bucket samplers, and preset generation

## Variant Status

`sleep2vec2/`, `sleep2vec_moe/`, and `sleep2vec_hires/` are named directories in the tree, but this branch does not have tracked source files inside them. Any parity assumptions for those variants are `unknown` here.
