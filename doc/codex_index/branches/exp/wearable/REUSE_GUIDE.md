# Reuse Guide

## Highest-Value Reuse Hotspots

### 1. YAML contracts and task semantics

Reuse these before adding new config parsing:

- `sleep2vec.config.load_pretrain_config`
- `sleep2vec.config.load_finetune_config`
- `sleep2vec.config.validate_model_config`
- `sleep2vec.common.apply_model_config_args`
- `sleep2vec.common.apply_finetune_config`
- `sleep2vec.common.apply_task_flags`

Why these are canonical:

- they already enforce required-vs-optional YAML structure
- they keep built-in task semantics (`stage5`, `sex`, `age`) in one place
- they are the only branch code paths that understand `adapt.*`

Duplication risk:

- High. Re-parsing YAML in entrypoints or tests will drift quickly from runtime behavior.

### 2. Channel declarations and dataset widths

Reuse these before touching channel handling:

- `sleep2vec.common.channel_input_dims_from_model_config`
- `data.psg_pretrain_dataset._build_channel_registry`
- `preprocess.save_dataset_presets._resolve_channels_and_dims`

Why these are canonical:

- the branch moved channel names and `input_dim` values to YAML `model.channels`
- runtime and preprocess now share the same contract

Do not do this:

- introduce a second hard-coded channel registry
- infer input widths from file contents at runtime

### 3. Missing-channel pretraining and pair scheduling

Reuse these before changing batch composition:

- `sleep2vec.utils.get_pretrain_dataloader`
- `data.default_dataset.DefaultDataset.dataloader`
- `data.utils.filter_valid_sample_indices`
- `data.samplers.PairFirstBatchSampler`
- `data.samplers.AvailableChannelsBucketBatchSampler`
- `sleep2vec.sleep2vec_adaptation.build_new_modality_pair_probs`

Why these are canonical:

- the collate path and batch samplers are coupled through `payload["available_channels"]`
- train-time pair distribution is controlled by the sampler, not by the Lightning module
- adaptation stage 2 depends on `set_pair_probs()` support in the train batch sampler

Duplication risk:

- Very high. Parallel implementations of pair selection or availability bucketing will diverge from validation behavior.

### 4. Adaptation run directory and checkpoint semantics

Reuse these before changing adaptation workflows:

- `sleep2vec.adapt._resolve_adapt_run_artifacts`
- `sleep2vec.common.persist_run_config_and_args`
- `sleep2vec.sleep2vec_adaptation.initial_pair_probs_for_phase`
- `sleep2vec.sleep2vec_adaptation.AdaptPairScheduleCallback`

Why these are canonical:

- they encode the branch contract that `--ckpt-path` resumes the same phase, while `--pretrained-backbone-path` handles stage transitions
- they preserve stage-specific snapshots (`config.stage1.yaml`, `cli_args.stage1.yaml`, and stage-2 siblings) without clobbering root files

Do not do this:

- hand-roll stage1/stage2 directory logic in shell scripts
- treat a stage-1 checkpoint as interchangeable with a stage-2 resume checkpoint

### 5. Shared backbone and downstream wrapping

Reuse these before building new model paths:

- `sleep2vec.pretrain_model.Sleep2vecPretrainModel`
- `sleep2vec.downstream_model.Sleep2vecDownstreamModel`
- `sleep2vec.builders.*`
- `sleep2vec.registry.*`

Why these are canonical:

- they centralize tokenizer construction, encoder instantiation, projection logic, CLS handling, layer mix, LoRA insertion, and downstream head construction

Duplication risk:

- High. New model entrypoints should compose these pieces rather than recreating them.

### 6. Checkpoint averaging and init loading

Reuse these before loading weights manually:

- `sleep2vec.checkpoints.load_pretrain_init_weights`
- `sleep2vec.checkpoints.extract_pretrain_init_state_dict`
- `sleep2vec.checkpoints.select_checkpoints`
- `sleep2vec.checkpoints.average_checkpoints`

Why these are canonical:

- they already know about `ema_model.` vs `model.` prefixes
- they already define the fallback from epoch ordering to modification time for inference-time averaging

### 7. Preset preparation

Reuse these before adding another data-prep helper:

- `preprocess.split_index_by_dataset.main`
- `preprocess.mask_missing_stats.main`
- `preprocess.save_dataset_presets.main`
- `preprocess.merge_dataset_presets.main`

Preferred sequence:

1. split CSVs
2. inspect channel-missing statistics if needed
3. build presets from YAML channel declarations
4. merge presets only when multiple sources must be combined

## Test Anchors To Update When Contracts Change

- Config/task semantics: `tests/test_config_loading.py`, `tests/test_common_finetune_apply.py`, `tests/test_metadata_task_validation.py`
- Adapt runtime and schedule logic: `tests/test_adapt.py`, `tests/test_adapt_pair_schedule_callback.py`, `tests/test_adaptation.py`
- Missing-channel sampling and collate rules: `tests/test_pair_first_sampler.py`, `tests/test_bucket_sampler.py`, `tests/test_generic_channel_dataset.py`, `tests/test_pretrain_pair_filtering.py`
- Preset generation: `tests/test_save_dataset_presets.py`
- Checkpoint selection and init loading: `tests/test_checkpoints.py`

## Known Non-Reuse Areas On This Branch

- `sleep2vec2/`, `sleep2vec_moe/`, and `sleep2vec_hires/` cannot be treated as reusable branch-local sources because this branch has no tracked source files there.
