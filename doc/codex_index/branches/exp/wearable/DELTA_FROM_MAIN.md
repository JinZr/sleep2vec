# Delta From `main`

## Baseline Status

- Branch: `exp/wearable`
- HEAD: `75f98ef35fd16000302291205429ea3f6d556788`
- `main`: `9abfaa6cec09c26798cf960c0e15c0a0ca846093`
- Merge base: `825a30433e1f3d4cfcf6e4338cde5c29426411f3`
- Commits ahead of `main`: `19`
- Main handbook availability: unavailable in this checkout because `doc/codex_index/branches/main/` has no files

## Headline Differences

### New runtime path: staged adaptation

Added:

- `sleep2vec/adapt.py`
- `sleep2vec/sleep2vec_adaptation.py`
- `tests/test_adapt.py`
- `tests/test_adapt_pair_schedule_callback.py`
- `tests/test_adaptation.py`

Effect:

- the branch adds a dedicated two-phase adaptation flow for newly introduced modalities
- phase transition semantics are explicit: stage 2 reuses the stage-1 experiment directory but writes checkpoints under `checkpoints.stage2`
- train pair distribution can be scheduled over training progress

### Config schema expanded for wearable adaptation

Changed:

- `sleep2vec/config.py`
- `sleep2vec/common.py`
- `tests/test_config_loading.py`

Effect:

- YAML can now include a top-level `adapt:` block
- `adapt.new_channels` must match `model.channels`
- stage-2 pair schedules are validated and must end at `until=1.0`
- task/config helpers still own downstream task semantics, so branch changes remain centralized

### Channel declarations moved harder toward YAML

Changed:

- `preprocess/save_dataset_presets.py`
- `data/psg_pretrain_dataset.py`
- `sleep2vec/common.py`
- `sleep2vec/utils.py`
- `tests/test_generic_channel_dataset.py`
- `tests/test_save_dataset_presets.py`

Effect:

- preset generation now requires YAML `model.channels`
- non-`stage5` runtime channels require explicit `channel_input_dims`
- wearable channels such as `ppg` and `actigraphy_vm` can be added through YAML without editing a hard-coded registry

### Missing-channel training path strengthened

Changed:

- `data/default_dataset.py`
- `data/psg_pretrain_dataset.py`
- `data/samplers.py`
- `sleep2vec/utils.py`
- `tests/test_bucket_sampler.py`
- `tests/test_pair_first_sampler.py`

Effect:

- training can use pair-first sampling with per-pair probability control
- validation can filter samples to scheduled pairs when channels are missing
- bucketed batching avoids montage mixing collapse when missing-channel training is enabled
- eval loader sharding behavior was tightened through sampler changes

### Runtime and checkpoint behavior adjusted

Changed:

- `sleep2vec/pretrain.py`
- `sleep2vec/finetune.py`
- `sleep2vec/infer.py`
- `sleep2vec/checkpoints.py`
- `sleep2vec/pretrain_model.py`
- `sleep2vec/downstream_model.py`
- `tests/test_checkpoints.py`

Effect:

- pretrain and finetune persist run config snapshots alongside artifacts
- finetune writes a stable `best.ckpt` copy after training
- inference supports checkpoint averaging via canonical helpers
- pretrain-model init loading is shared across pretrain, adapt, and downstream flows

### New and updated configs

Added or expanded:

- `configs/sleep2vec_dense_adapt_ppg_actigraphy.yaml`
- `configs/sleep2vec_dense_adapt_ppg_actigraphy_cls.yaml`
- `configs/sleep2vec_large_adapt_ppg_actigraphy.yaml`
- `configs/sleep2vec_large_adapt_ppg_actigraphy_cls.yaml`
- `configs/sleep2vec_large_pretrain.yaml`
- `configs/sleep2vec_large_pretrain_cls.yaml`

Effect:

- the branch documents large-model pretraining and wearable adaptation recipes directly in YAML

## Notable Cleanups Relative To Earlier Logic

- legacy `train_pair_sampling` flags are no longer part of the active runtime path
- dataset code now accesses explicit dataset attributes instead of relying on fallback lookups in the touched paths

## Areas With No Branch-Local Source Delta

- `sleep2vec2/`: no tracked source files
- `sleep2vec_moe/`: no tracked source files
- `sleep2vec_hires/`: no tracked source files

These areas are effectively outside branch-local code analysis for this handbook.

## Stale Entries Removed

- none; this is the first branch-local handbook for `exp/wearable`

## Unresolved Ambiguities

- without a populated `main` handbook, document-to-document stale-entry comparison is not possible
- variant-package parity versus other branches is `unknown`
