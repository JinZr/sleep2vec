# Module Map

## Natural Edit Boundaries

| Boundary | Primary files | Responsibility | Key dependencies | Extension points | Notes |
| --- | --- | --- | --- | --- | --- |
| Inherited base runtime | `sleep2vec/`, top-level `data/`, top-level `preprocess/` | Existing pretrain/adapt/finetune/infer behavior | Main branch docs | Base task/config/runtime edits | Source is unchanged relative to branch intent; consult `branches/main/` first |
| Package-local runtime mirror | `sleep2wave/config.py`, `common.py`, `builders.py`, `registry.py`, `pretrain.py`, `adapt.py`, `finetune.py`, `infer.py` | sleep2wave-local copy of base model/task/runtime contracts | local `sleep2wave.*` imports | sleep2wave-specific runtime divergence | Keep namespace-local; do not import base `sleep2vec`, top-level `data`, or top-level `preprocess` |
| Standalone backbone | `sleep2wave/backbones/`, `sleep2wave/backbones/roformer/` | Build standalone RoFormer or HF BERT encoder factories | Torch, optional Transformers | Backbone implementation and config compatibility | Legacy HF RoFormer checkpoint keys are intentionally rejected |
| Generative config schema | `sleep2wave/generative/config.py`, `configs/sleep2wave/*.yaml` | Strict stage-specific sleep2wave YAML parsing | `sleep2wave.data.modalities` | New stage fields or generation policy | Canonical parser for `recipe: sleep2wave` configs |
| sleep2wave modality and dataset contract | `sleep2wave/data/modalities.py`, `generative_dataset.py`, `generative_batch.py`, `quality.py`, `corruptions.py`, `derivations.py` | Canonical modalities, index/preset windows, signal slicing, masks, corruption, and collation | pandas, NumPy, Torch | New modalities, masks, derivations, or data layout | Preset schema version lives here |
| sleep2wave preprocessing | `sleep2wave/preprocess/build_sleep2wave_presets.py`, `validate_sleep2wave_index.py`, `derive_sleep2wave_channels.py`, copied split/preset utilities | Build and validate generative presets and indexes | `sleep2wave.data.generative_dataset`, pandas | New index columns or preset schema | `watchpat_zzp_to_edf.py` remains conversion-specific |
| Autoencoder model | `sleep2wave/autoencoders/model.py`, `losses.py`, `lightning.py`, `checkpoints.py` | Modality-specific waveform autoencoding and training loop | Torch, Lightning | Autoencoder architecture or losses | One latent per 30-second epoch is the current contract |
| Diffusion model | `sleep2wave/diffusion/model.py`, `schedule.py`, `samplers.py`, `tasks.py`, `task_masks.py`, `losses.py`, `lightning.py`, `latent_cache.py` | Latent diffusion transformer, task semantics, directional masks, sampling, cache loading, training loop | autoencoder checkpoint or latent cache, Torch, Lightning | Task families, mask semantics, sampler behavior | DDPM requires full schedule length; DDIM can use sparse steps |
| Task curriculum | `sleep2wave/training/phase_schedule.py`, `task_sampler.py`, `replay_buffer.py`, `logging.py` | Phase-to-task mix defaults, random task sampling, run naming | diffusion tasks, availability masks | New phase schedule or sampling policy | Phases 1-5 are diffusion-only |
| sleep2wave generation runtime | `sleep2wave/generate.py`, `generate_batch.py`, `inference/sliding_window.py`, `inference/uncertainty.py`, `export/artifacts.py`, `export/manifest.py` | CLI generation, batch per-night wrapper, checkpoint loading, sliding-window fusion, artifact export | autoencoder, diffusion sampler, config | Artifact schema, fusion mode, uncertainty policy | `generate.py` remains one subject/night per run |
| sleep2wave evaluation runtime | `sleep2wave/evaluate_generation.py`, `evaluation/*.py` | Evaluate generated artifacts across metric families | NumPy, JSON, generated artifact schema | Metric families or output schema | Writes both JSON and CSV metrics |
| Config validation tooling | `utils/check_configs.py`, `tests/test_check_configs.py` | Repo-wide YAML validation with sleep2wave variant routing | config loaders, importlib | New config variant roots | `CONFIG_VARIANTS` maps `configs/sleep2wave` to package-local loaders |
| Tests | `tests/test_sleep2wave_*.py` plus modified `tests/test_check_configs.py` | Pin sleep2wave namespace, config, data, model, runtime, artifact, and metrics contracts | pytest, Torch, pandas, yaml | Contract coverage | Use these before calling branch changes complete |

## Internal Dependency Flow

- `utils/check_configs.py` -> `sleep2wave.generative.config.load_sleep2wave_config` for `recipe: sleep2wave` stage configs
- `utils/check_configs.py` -> `sleep2wave.config` and `sleep2wave.preprocess.save_dataset_presets` for non-generative `configs/sleep2wave`
- `train_autoencoder.py` -> `load_sleep2wave_config` -> `Sleep2WaveGenerativeDataset` -> `Sleep2WaveAutoencoderLightning`
- `train_diffusion.py` -> `load_sleep2wave_config` -> `Sleep2WaveGenerativeDataset` -> `Sleep2WaveDiffusionLightning`
- `Sleep2WaveDiffusionLightning` -> autoencoder checkpoint or `Sleep2WaveLatentCacheDataset` -> `Sleep2WaveTaskSampler` -> `Sleep2WaveDiffusionTransformer`
- `generate.py` -> `load_sleep2wave_config` -> autoencoder checkpoint + diffusion checkpoint -> `build_sampler` -> sliding-window fusion -> artifact export
- `evaluate_generation.py` -> generated artifact files -> metric family modules -> `metrics.json` and `metrics.csv`
- `build_sleep2wave_presets.py` -> `build_sample_indices_from_frame` -> schema-versioned `SampleIndex` list

## Ownership Notes

- Config-stage edits belong in `sleep2wave/generative/config.py` and matching `configs/sleep2wave/*.yaml` tests.
- Legacy-style pretrain/finetune config edits inside `sleep2wave.config` should also check namespace parity tests and LoRA rejection behavior.
- Modality additions or sample-rate changes are cross-cutting: update `modalities.py`, generative config validation, dataset slicing, metrics assumptions, tiny configs, and tests.
- Generation artifact changes should keep `generate.py`, `export/artifacts.py`, `export/manifest.py`, `evaluate_generation.py`, and artifact tests aligned.
- Task semantics changes should keep `diffusion/tasks.py`, `diffusion/task_masks.py`, `training/phase_schedule.py`, `training/task_sampler.py`, and diffusion tests aligned.

## Ambiguities Worth Remembering

- `sleep2wave/preprocess/preprocess_pipeline.ipynb` is workflow history, not canonical library code.
- Cache-only diffusion is intentionally limited to translation and partial-full task mixes because restoration/imputation need waveform-level corruption before encoding.
- `sleep2wave` package-local runtime mirrors the base runtime, but the branch-specific reason to edit it should be explicit before diverging from base behavior.
