# sleep2wave Diffusion Workflow

## Purpose

Train a latent diffusion transformer that generates target modality latents from available condition modality latents.

## Canonical Path

1. Train or provide a compatible sleep2wave autoencoder checkpoint.
2. Load `stage: diffusion` config with `load_sleep2wave_config`.
3. Build train and val split DataLoaders through `train_diffusion.build_dataloader`; cache-only validation can reuse train rows when the cache artifact has no val split.
4. `Sleep2WaveDiffusionLightning` loads the autoencoder checkpoint, or uses an existing latent cache for translation/partial-full-only training.
5. `Sleep2WaveTaskSampler` samples phase-appropriate tasks.
6. Restoration/imputation tasks apply task-aware waveform corruptions from `training.corruptions` before autoencoder encoding.
7. `Sleep2WaveDiffusionTransformer` predicts target noise.
8. Validation logs epoch losses and task-family waveform examples when W&B is active.
9. Save epoch checkpoints and `last.ckpt`.

## Config Contract

Required blocks:

- `data`
- `modalities`
- `diffusion`
- `training`
- `sampler`
- `export`

Important constraints:

- `training.phase` must be 1 through 5.
- `data.context_epochs` must match `diffusion.context_epochs`.
- `diffusion.autoencoder_checkpoint` is required for waveform-to-latent training.
- `diffusion.latent_cache_path` can replace the autoencoder checkpoint only for translation/partial-full task mixes.
- `training.phase_checkpoint` initializes the diffusion transformer from a previous Sleep2Wave phase while keeping the current config.
- CLI `--resume-from-checkpoint` is reserved for Lightning crash recovery.
- `diffusion.beta_schedule` is currently `cosine`.
- `diffusion.prediction_type` is currently `epsilon`.
- `diffusion.task_attention_mask` is currently `directional`.
- `training.replay.enabled` selects replay-style default task mixtures when no explicit `task_mix` is provided; replay defaults train restoration and imputation before adding translation, two-condition, and partial-full tasks.
- `training.condition_counts` controls translation and partial-full condition-set sizes; partial-full samples among configured counts that fit the available modalities.
- `training.restoration_condition_counts` controls restoration/imputation condition-set sizes; the target modality is always included and extra modalities act as clean auxiliary context.
- `diffusion.validation_examples` controls W&B validation example count and target-modality candidates for diffusion phases; examples use the configured sampler and are logged per active task family.
- Tiny and medium diffusion phase recipes use `training.corruptions.*.by_modality` for physiologic restoration/imputation corruptions, and selected entries can define weighted `choices`.
- `diffusion.condition_dropout` preserves partial-full coverage by moving dropped condition modalities into the target set.

## Command

```bash
python -m sleep2wave.train_diffusion \
  --config configs/sleep2wave/sleep2wave_diffusion_tiny_phase1.yaml \
  --version-name diffusion-smoke \
  --accelerator cpu \
  --devices 1 \
  --num-workers 0 \
  --seed 0
```

Build a latent cache:

```bash
python -m sleep2wave.cache_latents \
  --config configs/sleep2wave/sleep2wave_diffusion_medium_phase2.yaml \
  --autoencoder-ckpt checkpoints/sleep2wave_autoencoder_medium.ckpt \
  --output-dir outputs/sleep2wave_latent_cache
```

## Edit Hotspots

- Task semantics: `sleep2wave/diffusion/tasks.py`
- Attention mask semantics: `sleep2wave/diffusion/task_masks.py`
- Model shape and embeddings: `sleep2wave/diffusion/model.py`
- Schedule and samplers: `sleep2wave/diffusion/schedule.py`, `sleep2wave/diffusion/samplers.py`
- Training step: `sleep2wave/diffusion/lightning.py`
- Latent cache: `sleep2wave/diffusion/latent_cache.py`, `sleep2wave/cache_latents.py`
- Curriculum: `sleep2wave/training/phase_schedule.py`, `sleep2wave/training/task_sampler.py`

## Tests

```bash
python3.10 -m pytest -q \
  tests/test_sleep2wave_diffusion_model_shapes.py \
  tests/test_sleep2wave_diffusion_losses.py \
  tests/test_sleep2wave_diffusion_task_masks.py \
  tests/test_sleep2wave_diffusion_tasks.py \
  tests/test_sleep2wave_diffusion_schedule.py \
  tests/test_sleep2wave_diffusion_sampler.py \
  tests/test_sleep2wave_diffusion_train_smoke.py \
  tests/test_sleep2wave_phase_schedule.py \
  tests/test_sleep2wave_task_sampler.py
```
