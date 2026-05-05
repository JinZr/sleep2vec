# Reuse Guide

## Highest-Value Reuse Hotspots

| Responsibility | Canonical implementation to reuse | Why it is canonical | Do not bypass with |
| --- | --- | --- | --- |
| Base `sleep2vec` contracts | `doc/codex_index/branches/main/` plus unchanged base source | The base runtime is inherited on this branch | Re-indexing base behavior from memory |
| sleep2wave generative YAML parsing | `sleep2wave.generative.config.load_sleep2wave_config` | Enforces stage-specific required and disallowed blocks | Hand-written YAML parsing in entrypoints |
| sleep2wave modality schema | `sleep2wave.data.modalities` | Single source for modality order, sample rates, aliases, and frames per epoch | New local modality lists |
| Index to generative windows | `build_sample_indices_from_frame` and `build_sample_indices_from_index` | Builds schema-versioned `SampleIndex` payloads and validates split boundaries | Manual `SampleIndex` pickling |
| Generative dataset loading | `Sleep2WaveGenerativeDataset` | Handles preset/index source choice, signal slicing, availability/quality masks, corruption, and metadata | Entry-point-local NPZ readers |
| Generative batch collation | `collate_sleep2wave_generative` | Pads channel dimensions and stacks signal/mask dictionaries consistently | Ad hoc dict stacking |
| Split-safe preset CLI | `sleep2wave.preprocess.build_sleep2wave_presets.build_sleep2wave_presets` | Canonical write path for sleep2wave generative preset pickles | External scripts that skip schema validation |
| Index validation | `sleep2wave.preprocess.validate_sleep2wave_index.validate_sleep2wave_index` | Fast boundary check for required columns, masks, split leakage, and zero-modality rows | Partial CSV checks |
| Autoencoder model | `Sleep2WaveAutoencoder` | Encodes and decodes modality-specific one-latent-per-epoch waveform branches | Per-modality model copies |
| Autoencoder loss | `compute_autoencoder_loss` / `Sleep2WaveAutoencoderLoss` | Applies availability and quality masks to waveform and spectral losses | Trainer-local loss math |
| Diffusion task semantics | `build_generation_task` and `validate_generation_task` | Centralizes restoration/imputation/translation/partial_full constraints | Branching on task strings in model or sampler code |
| Directional attention masks | `build_directional_task_attention_mask` | Canonical condition/target token mask builder | Manual transformer masks |
| Diffusion transformer | `Sleep2WaveDiffusionTransformer.from_config` | Constructs model from typed `DiffusionConfig` | Direct constructor calls from YAML dicts |
| Diffusion schedule | `build_diffusion_schedule` | Single cosine schedule implementation and validation | Recomputing beta/alpha arrays locally |
| Diffusion samplers | `build_sampler`, `DDIMSampler`, `DDPMSampler` | Encodes DDIM/DDPM step constraints and generated-latent shape | Sampling loops inside generation CLI |
| Training task sampling | `Sleep2WaveTaskSampler` and `build_phase_schedule` | Central phase/task mix and availability-aware task selection | Random task code in Lightning modules |
| Autoencoder checkpoint loading | `load_sleep2wave_autoencoder_checkpoint` | Loads only compatible sleep2wave autoencoder weights | Raw `torch.load` in diffusion/generation code |
| Sleep2Vec2 initialization | `load_sleep2vec2_initialization` | Groups compatible checkpoint keys and reports loaded/missing/skipped state | Unstructured state-dict copying |
| Generation runtime | `run_generation` | Owns task creation, checkpoint loading, window collection, fusion, uncertainty, and artifact export | Separate generation scripts |
| Artifact writing | `write_generation_artifacts` and `build_generation_manifest` | Defines generated artifact file schema and provenance metadata | One-off NPZ/JSON writers |
| Sliding-window fusion | `fuse_overlapping_windows` and `fuse_mask_windows` | Canonical mean/median/uncertainty and mask fusion behavior | Local overlap loops |
| Uncertainty summary | `compute_uncertainty` | Produces mean/std/sample-count/high-uncertainty outputs for artifacts | Recomputing summaries in export code |
| Generation evaluation | `run_evaluation` plus `evaluation/*.py` metric families | Canonical metric orchestration and output writer | Notebook-only metric calculations |
| Config validation | `utils/check_configs.py` | Routes base vs sleep2wave configs to the right package-local loaders | YAML-only linters |

## Reuse Rules By Change Type

### Changing sleep2wave config semantics

- Reuse `load_sleep2wave_config` for `recipe: sleep2wave` configs.
- Keep stage required/disallowed block policy in `sleep2wave/generative/config.py`.
- Add or update `tests/test_sleep2wave_generative_config.py` for schema behavior.
- For non-generative `configs/sleep2wave` recipes, keep `sleep2wave.config` package-local and preserve LoRA rejection unless explicitly changing that contract.

### Changing modality or data semantics

- Start with `sleep2wave.data.modalities`.
- Reuse `build_sample_indices_from_frame` for schema-versioned `SampleIndex` creation.
- Reuse `Sleep2WaveGenerativeDataset` for waveform loading and masks.
- Keep `tests/test_sleep2wave_modalities.py`, `tests/test_sleep2wave_generative_dataset.py`, and `tests/test_sleep2wave_preprocess_contract.py` aligned.

### Changing autoencoder behavior

- Reuse `Sleep2WaveAutoencoder` and `Sleep2WaveAutoencoderLoss`.
- Keep one latent per epoch unless the config contract is intentionally changed.
- Check `train_autoencoder.py`, `autoencoders/lightning.py`, autoencoder checkpoint loading, and autoencoder tests together.

### Changing diffusion behavior

- Reuse `GenerationTask`, `TokenLayout`, `build_directional_task_attention_mask`, `Sleep2WaveDiffusionTransformer`, and `build_sampler`.
- Keep task-family validation outside the transformer forward path by using `validate_generation_task`.
- Update task, mask, schedule, sampler, model-shape, and diffusion-smoke tests together.

### Changing generation artifacts

- Reuse `run_generation`, `write_generation_artifacts`, and `build_generation_manifest`.
- Keep `evaluate_generation.py` compatible with any artifact schema change.
- Update `tests/test_sleep2wave_generate_cli.py`, `tests/test_sleep2wave_export_artifacts.py`, and `tests/test_sleep2wave_evaluate_cli.py`.

### Changing config validation

- Extend `CONFIG_VARIANTS` only for real package-local config roots.
- For `recipe: sleep2wave` configs, route through `load_sleep2wave_config`.
- For legacy-style runtime configs under `configs/sleep2wave`, route through package-local `sleep2wave.config` and `sleep2wave.preprocess.save_dataset_presets`.

## Major Duplication Risks

1. `sleep2wave` intentionally mirrors much of `sleep2vec`; keep branch-specific changes package-local and avoid reintroducing base imports.
2. The sleep2wave modality list appears in configs, tests, model construction, and metrics. Treat `sleep2wave.data.modalities` as canonical.
3. `build_sample_indices_from_frame` and `Sleep2WaveGenerativeDataset` both understand preset payloads. Do not create a third schema interpretation.
4. Diffusion tasks, task masks, task sampler, and generation CLI must agree on restoration/imputation auxiliary-token behavior.
5. Artifact schema is shared by generation, export, evaluation, and tests. Do not change one without the others.
6. DDIM/DDPM sampler constraints are enforced in both config parsing and sampler constructors. Keep error behavior aligned.
7. LoRA is available in the base runtime but intentionally rejected in `sleep2wave.config` for now.

## Known Non-Reuse Zones

- `sleep2wave/preprocess/preprocess_pipeline.ipynb` is workflow history.
- Tracked font assets are packaging assets, not reusable logic.
- `sleep2wave/training/replay_buffer.py` should not be treated as active replay training until the Lightning path uses it.
