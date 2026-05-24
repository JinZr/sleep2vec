# Adapt Workflow

## Purpose

Run staged modality adaptation on top of a pretrain-style backbone, with explicit phase boundaries for new-modality warmup and later joint refinement.

## Entry Command

Canonical entrypoint: `python -m sleep2vec.adapt --config ... --phase stage1|stage2 --version-name ...`

Primary code path:

1. `sleep2vec.adapt.sleep2vec_adapt`
2. `sleep2vec.config.load_pretrain_config`
3. `sleep2vec.common.apply_model_config_args`
4. `sleep2vec.common.apply_data_backend_args`
5. `sleep2vec.sleep2vec_adaptation.initial_pair_probs_for_phase`
6. `sleep2vec.utils.get_pretrain_dataloader`
7. `sleep2vec.adapt._resolve_adapt_run_artifacts`
8. `sleep2vec.sleep2vec_adaptation.Sleep2vecAdaptation`
9. Lightning `trainer.fit(...)`

## Detailed Flow

1. Parse CLI.
   - Requires `--phase stage1|stage2`.
   - Requires `--config` to point at a pretrain-style YAML with a top-level `adapt` block.
2. Load and validate config.
   - `load_pretrain_config` parses `model`, `loss`, `data`, optional `model_averaging`, and `adapt`.
   - `adapt.new_channels` must be present in `model.channels`.
   - Stage-2 pair schedule must end at `until=1.0`.
3. Bind model-derived fields into `args`.
   - `channel_names`
   - `channel_input_dims`
   - `backbone_arch`
   - data-backend fields from YAML
   - initial `train_pair_probs`
4. Build loaders.
   - Reuses the missing-channel pretrain dataloader path.
   - Uses `PSGPretrainDataset` for `npz` or `KaldiPSGDataset` for `kaldi`.
   - Usually depends on `PairFirstBatchSampler` for training.
   - Validation uses sequential pair evaluation.
5. Resolve run artifacts and checkpoint policy.
   - `--ckpt-path` resumes exactly inside the same phase and run directory.
   - `--pretrained-backbone-path` initializes weights.
   - Stage2 transition requires `--pretrained-backbone-path` to point at a prior adapt stage1 checkpoint.
   - Fresh stage2 transition refuses to reuse a non-empty `checkpoints.stage2/` directory.
6. Persist run metadata.
   - Root run files: `config.yaml`, `cli_args.yaml` on fresh runs
   - Phase-scoped files: `config.stage1.yaml`, `cli_args.stage1.yaml`, `config.stage2.yaml`, `cli_args.stage2.yaml`
7. Instantiate `Sleep2vecAdaptation`.
   - Inherits the pretrain loop from `Sleep2vecPretraining`.
   - Loads init weights via `load_pretrain_init_weights` when appropriate.
   - Applies adaptation freeze policy to the backbone.
8. Train.
   - Stage1 optimizes new-modality groups, optionally the shared projection.
   - Stage2 optimizes encoder/CLS, shared legacy projection, and new modalities with scaled learning rates.
   - `AdaptPairScheduleCallback` updates pair probabilities during stage2.

## Important Runtime Decisions

- Adaptation uses pretrain-style contrastive loss and logging, not downstream finetune loss code.
- Kaldi-backed adaptation is controlled by YAML data backend fields; legacy preset pickle paths are not valid with Kaldi.
- Pair scheduling is part of the runtime contract; do not move it into ad hoc sampler wrappers.
- The checkpoint directory name is phase-specific:
  - stage1: `checkpoints/`
  - stage2: `checkpoints.stage2/`
- Reusing a stage1 checkpoint as `--ckpt-path` for stage2 is invalid; stage transitions go through `--pretrained-backbone-path`.

## Outputs

- Stage1 checkpoints under `log-adapt/<run>/checkpoints/`
- Stage2 checkpoints under `log-adapt/<run>/checkpoints.stage2/`
- Root and phase-specific copied configs / CLI snapshots
- W&B run under project `sleep2vec-adapt`

## Edit Hotspots

- Change phase-transition or resume semantics: `sleep2vec/adapt.py`
- Change freeze policy or optimizer groups: `sleep2vec/sleep2vec_adaptation.py`, `sleep2vec/pretrain_model.py`
- Change pair-schedule behavior: `sleep2vec/sleep2vec_adaptation.py`, `data/samplers.py`, `sleep2vec/callbacks/pair_acc_logger.py`
- Change missing-channel or data-backend loader policy: `sleep2vec/common.py`, `sleep2vec/utils.py`, `data/default_dataset.py`, `data/utils.py`, `data/kaldi_psg_dataset.py`
