# sleep2wave Autoencoder Workflow

## Purpose

Train modality-specific waveform autoencoders that produce channel-specific temporal latent maps per 30-second epoch.

## Canonical Path

1. Load `stage: autoencoder` config with `load_sleep2wave_config`.
2. Build train split DataLoader through `train_autoencoder.build_dataloader`.
3. Train and validate `Sleep2WaveAutoencoderLightning`.
4. Save epoch checkpoints and `last.ckpt`.

## Config Contract

Required blocks:

- `data`
- `modalities`
- `autoencoder`
- `training`
- `export`

`data.backend` defaults to `npz`, so existing tiny/medium configs and `sleep2wave_train.sh` continue to use `preset_path` or `index`. For opt-in Kaldi runs, set `data.backend: kaldi` with `kaldi_data_root`, `kaldi_manifest`, and matching `context_epochs`; the CLI command is unchanged.

Current autoencoder architecture constraints:

- `encoder_type: temporal_conv`
- `decoder_type: temporal_conv`
- `latent_frames_per_epoch.high_frequency: 60`
- `latent_frames_per_epoch.low_frequency: 30`
- `channel_specific: true`

Current latent contract:

- High-frequency modalities: `[B, E, C, 60, D]`
- Low-frequency modalities: `[B, E, C, 30, D]`
- Reconstructions preserve the input signal shape, including channel dimension for `[B, E, C, S]` inputs.

Optional logging config:

- `training.validation.interval_steps`
- `training.validation.max_batches_per_modality`
- `training.validation.examples.num_examples`
- `training.validation.examples.modalities`

If omitted, validation logs one W&B clean/reconstruction waveform example for every configured modality and runs capped validation every 1000 training steps.

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
- W&B validation waveform examples when a run is active

## Edit Hotspots

- Model architecture: `sleep2wave/autoencoders/model.py`
- Loss behavior: `sleep2wave/autoencoders/losses.py`
- Lightning training: `sleep2wave/autoencoders/lightning.py`
- CLI/trainer wiring: `sleep2wave/train_autoencoder.py`
- Data backend loading: `sleep2wave/data/generative_dataset.py`

## Tests

```bash
python3.10 -m pytest -q \
  tests/test_sleep2wave_autoencoder_model.py \
  tests/test_sleep2wave_autoencoder_losses.py \
  tests/test_sleep2wave_autoencoder_train_smoke.py
```
