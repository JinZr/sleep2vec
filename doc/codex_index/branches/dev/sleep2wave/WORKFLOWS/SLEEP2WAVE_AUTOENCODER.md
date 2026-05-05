# Sleep2Wave Autoencoder Workflow

## Purpose

Train modality-specific waveform autoencoders that produce one latent vector per 30-second epoch.

## Canonical Path

1. Load `stage: autoencoder` config with `load_sleep2wave_config`.
2. Build train split DataLoader through `train_autoencoder.build_dataloader`.
3. Train `Sleep2WaveAutoencoderLightning`.
4. Save epoch checkpoints and `last.ckpt`.

## Config Contract

Required blocks:

- `data`
- `modalities`
- `autoencoder`
- `training`
- `export`

Current autoencoder architecture constraints:

- `encoder_type: conv1d_epoch`
- `decoder_type: convtranspose1d_epoch`
- `one_latent_per_epoch: true`
- `modality_specific: true`

## Command

```bash
python -m sleep2wave.train_autoencoder \
  --config configs/sleep2wave/sleep2wave_autoencoder_tiny.yaml \
  --version-name autoencoder-smoke \
  --accelerator cpu \
  --devices 1 \
  --num-workers 0
```

## Outputs

`train_autoencoder.py` writes:

- copied config and CLI args through `persist_run_config_and_args`
- Lightning checkpoints under `<export.output_dir>/<version-name>/checkpoints/`
- `last.ckpt`

## Edit Hotspots

- Model architecture: `sleep2wave/autoencoders/model.py`
- Loss behavior: `sleep2wave/autoencoders/losses.py`
- Lightning training: `sleep2wave/autoencoders/lightning.py`
- CLI/trainer wiring: `sleep2wave/train_autoencoder.py`

## Tests

```bash
python3.10 -m pytest -q \
  tests/test_sleep2wave_autoencoder_model.py \
  tests/test_sleep2wave_autoencoder_losses.py \
  tests/test_sleep2wave_autoencoder_train_smoke.py
```
