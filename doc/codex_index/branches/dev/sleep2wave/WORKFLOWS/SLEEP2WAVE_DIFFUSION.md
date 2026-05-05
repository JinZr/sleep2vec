# sleep2wave Diffusion Workflow

## Purpose

Train a latent diffusion transformer that generates target modality latents from available condition modality latents.

## Canonical Path

1. Train or provide a compatible sleep2wave autoencoder checkpoint.
2. Load `stage: diffusion` config with `load_sleep2wave_config`.
3. Build train split DataLoader through `train_diffusion.build_dataloader`.
4. `Sleep2WaveDiffusionLightning` loads the autoencoder checkpoint.
5. `Sleep2WaveTaskSampler` samples phase-appropriate tasks.
6. `Sleep2WaveDiffusionTransformer` predicts target noise.
7. Save epoch checkpoints and `last.ckpt`.

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
- `diffusion.autoencoder_checkpoint` is required.
- `diffusion.beta_schedule` is currently `cosine`.
- `diffusion.prediction_type` is currently `epsilon`.
- `diffusion.task_attention_mask` is currently `directional`.

## Command

```bash
python -m sleep2wave.train_diffusion \
  --config configs/sleep2wave/sleep2wave_diffusion_tiny.yaml \
  --version-name diffusion-smoke \
  --accelerator cpu \
  --devices 1 \
  --num-workers 0 \
  --seed 0
```

## Edit Hotspots

- Task semantics: `sleep2wave/diffusion/tasks.py`
- Attention mask semantics: `sleep2wave/diffusion/task_masks.py`
- Model shape and embeddings: `sleep2wave/diffusion/model.py`
- Schedule and samplers: `sleep2wave/diffusion/schedule.py`, `sleep2wave/diffusion/samplers.py`
- Training step: `sleep2wave/diffusion/lightning.py`
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
