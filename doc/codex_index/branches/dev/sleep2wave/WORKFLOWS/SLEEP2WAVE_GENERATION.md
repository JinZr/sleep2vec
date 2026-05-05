# sleep2wave Generation Workflow

## Purpose

Generate target PSG modalities from incomplete recordings using a trained sleep2wave autoencoder and diffusion checkpoint.

## Canonical Path

1. Load `stage: inference` config with `load_sleep2wave_config`.
2. Build a `GenerationTask` from CLI task and modality arguments.
3. Load autoencoder and diffusion checkpoints.
4. Iterate `Sleep2WaveGenerativeDataset` windows.
5. Sample target latents with DDIM or DDPM.
6. Decode latents to waveform space.
7. Fuse overlapping windows and masks.
8. Compute uncertainty.
9. Write generation artifacts.

## Command

```bash
python -m sleep2wave.generate \
  --config configs/sleep2wave/sleep2wave_generate_tiny.yaml \
  --diffusion-ckpt checkpoints/sleep2wave_diffusion.ckpt \
  --task translation \
  --condition-modalities eeg eog \
  --target-modalities airflow \
  --output-dir outputs/sleep2wave_generate_run \
  --device cpu
```

## Artifact Contract

`write_generation_artifacts` writes:

- `generated.npz`
- `uncertainty.npz`
- `masks.npz`
- `metadata.jsonl`
- `config.yaml`
- `cli_args.yaml`
- `manifest.json`

`build_generation_manifest` records provenance as `generated_decision_support_not_acquired_clinical_channels` and `clinical_use: decision_support_only`.

## Current Limitations

- `validate_single_night` requires all generation windows to belong to one subject/night/path.
- `--preset-path` and `--index` are mutually exclusive, and exactly one source must be available through CLI or config.
- DDPM sampling requires sampler steps equal to diffusion steps.

## Edit Hotspots

- CLI orchestration: `sleep2wave/generate.py`
- Sampling: `sleep2wave/diffusion/samplers.py`
- Sliding-window fusion: `sleep2wave/inference/sliding_window.py`
- Uncertainty: `sleep2wave/inference/uncertainty.py`
- Artifact schema: `sleep2wave/export/artifacts.py`, `sleep2wave/export/manifest.py`

## Tests

```bash
python3.10 -m pytest -q \
  tests/test_sleep2wave_generate_cli.py \
  tests/test_sleep2wave_export_artifacts.py \
  tests/test_sleep2wave_sliding_window.py
```
