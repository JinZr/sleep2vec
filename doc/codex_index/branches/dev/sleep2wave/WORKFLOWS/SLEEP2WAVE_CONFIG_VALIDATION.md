# Sleep2Wave Config Validation Workflow

## Purpose

Validate both Sleep2Wave generative-stage configs and package-local legacy-style runtime configs.

## Canonical Paths

- Generative configs: `sleep2wave.generative.config.load_sleep2wave_config`
- Legacy-style Sleep2Wave runtime configs: `sleep2wave.config.load_pretrain_config` or `sleep2wave.config.load_finetune_config`
- Repo-wide validation: `utils/check_configs.py`

## Config Types

### Generative stage configs

Files with:

- `recipe: sleep2wave`
- `stage: autoencoder`, `diffusion`, `inference`, or `evaluation`

These are routed directly to `load_sleep2wave_config`.

### Package-local runtime configs

Any non-generative YAML under `configs/sleep2wave/` should be routed through package-local `sleep2wave.config` and `sleep2wave.preprocess.save_dataset_presets` helpers.

## Required Checks

- Stage configs must contain required blocks and must not contain disallowed blocks.
- `modalities` must match `sleep2wave.data.modalities`.
- Diffusion and inference configs must keep `data.context_epochs == diffusion.context_epochs`.
- `sampler.steps` must be compatible with diffusion steps and DDPM/DDIM rules.
- Sleep2Wave finetune configs must keep LoRA flags disabled unless the standalone RoFormer support changes.

## Commands

```bash
python utils/check_configs.py configs/sleep2wave
python3.10 -m pytest -q tests/test_check_configs.py tests/test_sleep2wave_generative_config.py tests/test_sleep2wave_namespace.py
```

## Edit Hotspots

- Stage schema: `sleep2wave/generative/config.py`
- Runtime config schema: `sleep2wave/config.py`
- Repo routing: `utils/check_configs.py`
- Tiny recipes: `configs/sleep2wave/*.yaml`
