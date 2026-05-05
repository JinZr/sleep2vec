# System Overview

## Branch Architecture

`dev/sleep2wave` keeps the existing `sleep2vec` runtime and adds a standalone `sleep2wave` package.

The branch has two active code surfaces:

- Inherited surface: `sleep2vec/`, top-level `data/`, top-level `preprocess/`, and base `configs/` behave as documented in `doc/codex_index/branches/main/`.
- Branch surface: `sleep2wave/`, `configs/sleep2wave/`, sleep2wave tests, and `utils/check_configs.py` additions implement and validate the new sleep2wave workflow.

## sleep2wave Runtime Layers

### Package-Local Sleep2Vec Mirror

`sleep2wave` contains package-local copies of the core `sleep2vec` runtime families:

- `sleep2wave.config`, `common`, `builders`, `registry`
- `sleep2wave.pretrain`, `adapt`, `finetune`, `infer`
- `sleep2wave.pretrain_model`, `downstream_model`, `sleep2vec_modelling`, `sleep2vec_finetuning`, `sleep2vec_adaptation`
- `sleep2wave.data.*`, `losses`, `metrics`, `callbacks`, `visualization`

The package-local mirror exists so sleep2wave can evolve without importing the base `sleep2vec`, top-level `data`, or top-level `preprocess` namespaces. `tests/test_sleep2wave_namespace.py` checks this namespace boundary.

### Generative sleep2wave Stack

The new generative stack is separate from the copied pretrain/finetune runtime:

- `sleep2wave.generative.config.load_sleep2wave_config` parses stage-specific `recipe: sleep2wave` YAML.
- `sleep2wave.data.generative_dataset.Sleep2WaveGenerativeDataset` produces clean and observed waveform windows plus availability, quality, and corruption masks.
- `sleep2wave.autoencoders.model.Sleep2WaveAutoencoder` encodes each modality independently into one latent per epoch.
- `sleep2wave.diffusion.model.Sleep2WaveDiffusionTransformer` denoises latent targets conditioned on available modality latents.
- `sleep2wave.generate.run_generation` samples, decodes, fuses sliding windows, and writes artifacts.
- `sleep2wave.evaluate_generation.run_evaluation` scores generated artifacts against waveform, feature, event, efficiency, and downstream metric families.

## Stage Semantics

`load_sleep2wave_config` supports four stages:

| Stage | Required blocks | Disallowed blocks | Primary entrypoint |
| --- | --- | --- | --- |
| `autoencoder` | `data`, `modalities`, `autoencoder`, `training`, `export` | `diffusion`, `sampler`, `evaluation` | `python -m sleep2wave.train_autoencoder` |
| `diffusion` | `data`, `modalities`, `diffusion`, `training`, `sampler`, `export` | `autoencoder`, `evaluation` | `python -m sleep2wave.train_diffusion` |
| `inference` | `data`, `modalities`, `diffusion`, `sampler`, `export` | `training`, `autoencoder`, `evaluation` | `python -m sleep2wave.generate` |
| `evaluation` | `modalities`, `evaluation`, `export` | `data`, `training`, `autoencoder`, `diffusion`, `sampler`, `initialization` | `python -m sleep2wave.evaluate_generation` |

## Modality Contract

Canonical sleep2wave modalities live in `sleep2wave.data.modalities`:

- high-frequency, 128 Hz: `eeg`, `eog`, `emg`, `ecg`
- low-frequency, 4 Hz: `airflow`, `belt`, `spo2`, `ibi`, `resp`
- epoch duration: 30 seconds
- frames per epoch: 3840 for high-frequency modalities and 120 for low-frequency modalities

Configs must use canonical names. Dataset/index loading may resolve known aliases such as `eeg_original`, `resp_nasal_original`, `heartbeat`, and `breath`.

## Data Contract

sleep2wave generative presets are pickled `list[SampleIndex]` objects with payload fields including:

- `sleep2wave_schema_version`
- `available_channels`
- `availability_mask_keys`
- `quality_mask_keys`
- `canonical_channel_map`
- `epoch_sec`
- `sample_rates`
- `frames_per_epoch`
- `subject_id`, `night_id`, `night_epoch_count`

`build_sample_indices_from_frame` and `build_sleep2wave_presets` are the canonical ways to build this schema. `Sleep2WaveGenerativeDataset` can load either a preset pickle or an index CSV, but each config/data source must choose exactly one.

## Model Flow

Autoencoder training:

1. `Sleep2WaveGenerativeDataset` returns `clean_signals`.
2. `Sleep2WaveAutoencoder` reconstructs each modality and returns one latent per epoch.
3. `Sleep2WaveAutoencoderLoss` combines masked waveform L1/L2 and spectral losses.

Diffusion training:

1. `Sleep2WaveDiffusionLightning` loads a frozen autoencoder checkpoint.
2. `Sleep2WaveTaskSampler` samples a generation task from the configured phase schedule.
3. The autoencoder encodes clean and observed signals into latents.
4. Noise is added to target latents through `build_diffusion_schedule`.
5. `Sleep2WaveDiffusionTransformer` predicts target noise under a directional task attention mask.

Generation:

1. `generate.py` loads an inference config, autoencoder checkpoint, and diffusion checkpoint.
2. It builds a `GenerationTask`, collects sliding windows, and samples target latents with DDIM or DDPM.
3. It decodes generated latents, fuses overlapping windows, computes uncertainty, and writes NPZ/JSON/YAML artifacts.

Evaluation:

1. `evaluate_generation.py` validates generated artifact files.
2. It loads generated means, masks, optional references, optional baselines, event JSON, and downstream metrics.
3. It writes `metrics.json` and `metrics.csv`.

## Branch Constraints

- sleep2wave finetune configs reject LoRA flags. `sleep2wave.config.load_finetune_config` raises if `freeze_backbone_and_insert_lora`, `insert_lora`, or `separate_adapters` is true.
- sleep2wave uses a standalone RoFormer implementation. `sleep2wave.checkpoints.load_pretrain_init_weights` rejects legacy HF-style RoFormer checkpoint keys when loading into the standalone target.
- `utils/check_configs.py` routes `configs/sleep2wave/**` through package-local `sleep2wave` config and preset helpers.
- `sleep2vec2/`, `sleep2vec_moe/`, and `sleep2vec_hires/` have no tracked source files on this branch.
