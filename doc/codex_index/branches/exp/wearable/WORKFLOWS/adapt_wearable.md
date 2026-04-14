# Workflow: Adapt Wearable

## Branch-Specific Purpose

This branch adds a two-phase adaptation workflow for introducing new modalities such as `ppg` and `actigraphy_vm` while reusing a pretrained backbone.

## Runtime Sequence

1. CLI enters `sleep2vec.adapt.sleep2vec_adapt`.
2. `load_pretrain_config` parses pretrain-style YAML and requires a top-level `adapt` block.
3. `apply_model_config_args` copies channel names and widths into `args`.
4. `initial_pair_probs_for_phase` computes the starting train-pair distribution for stage 1 or the first stage-2 schedule point.
5. `get_pretrain_dataloader` builds train and per-pair validation loaders.
6. `_resolve_adapt_run_artifacts` decides whether this is:
   - exact phase resume via `--ckpt-path`
   - stage-1 -> stage-2 transition via `--pretrained-backbone-path`
   - fresh run
7. `persist_run_config_and_args` writes root and phase-specific snapshots.
8. `Sleep2vecAdaptation` loads init weights when appropriate, applies phase-specific freeze policy, and configures phase-specific optimizer groups.
9. Stage 2 attaches `AdaptPairScheduleCallback`, which updates the pair sampler each epoch.

## Checkpoint Semantics

### Exact resume

- flag: `--ckpt-path`
- contract: checkpoint must already belong to the same adapt phase
- effect: trainer resumes optimizer/scheduler/epoch state

### Phase transition

- flag: `--pretrained-backbone-path`
- stage-1 input: base pretrain checkpoint
- stage-2 input: prior adapt stage-1 checkpoint
- effect: model weights initialize from checkpoint, but trainer state resets; stage 2 writes under `checkpoints.stage2`

## Pair-Schedule Semantics

- `adapt.stage2.pair_schedule` is expressed in training-progress fractions
- each point supplies `until` and `new_pair_ratio`
- the final point must end at `until=1.0`
- `build_new_modality_pair_probs` splits pair probability mass between new-modality pairs and legacy pairs

## Main Reuse Points

- run-state machine: `sleep2vec.adapt._resolve_adapt_run_artifacts`
- schedule math: `sleep2vec.sleep2vec_adaptation.build_new_modality_pair_probs`
- optimizer grouping: `sleep2vec.sleep2vec_adaptation.Sleep2vecAdaptation.configure_optimizers`
- config persistence: `sleep2vec.common.persist_run_config_and_args`

## Main Failure Checks

- `adapt.new_channels` missing from `model.channels`
- stage-2 checkpoint passed from the wrong phase
- stage-2 pair schedule not ending at `1.0`
- missing `payload["available_channels"]` in presets while missing-channel training is active
