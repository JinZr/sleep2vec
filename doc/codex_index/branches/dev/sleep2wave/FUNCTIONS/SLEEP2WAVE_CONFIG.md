# sleep2wave Config Functions

## Generative config bundle family

- File: `sleep2wave/generative/config.py`
- Dataclasses:
  - `DataConfig`, `ModalitiesConfig`, `AutoencoderLossConfig`, `AutoencoderValidationExamplesConfig`, `AutoencoderConfig`
  - `DiffusionValidationExamplesConfig`, `TransformerConfig`, `EmbeddingsConfig`, `DiffusionConfig`
  - `ReplayConfig`, `CorruptionChoiceConfig`, `CorruptionSpecConfig`, `CorruptionPolicyConfig`, `TrainingCorruptionsConfig`, `TrainingConfig`, `InferenceConfig`, `SamplerConfig`
  - `InitializationConfig`, `ExportConfig`, `EvaluationConfig`, `Sleep2WaveConfig`
- Purpose and contract: define the typed in-memory schema for `recipe: sleep2wave` YAML stages.
- Important inputs/outputs: YAML mappings in; frozen dataclass tree out.
- Side effects: none beyond reading YAML through `load_sleep2wave_config`.
- Reuse guidance: extend these dataclasses before adding ad hoc entrypoint fields.
- Duplication-risk notes: stage semantics and modality validation must stay centralized here.

## `sleep2wave.generative.config.load_sleep2wave_config`

- File: `sleep2wave/generative/config.py`
- Signature: `load_sleep2wave_config(path: str | Path) -> Sleep2WaveConfig`
- Purpose and contract: parse a strict sleep2wave YAML file with `recipe: sleep2wave` and one of `autoencoder`, `diffusion`, `inference`, or `evaluation` stages.
- Important inputs/outputs: config path in; `Sleep2WaveConfig` out.
- Side effects: reads YAML from disk.
- Key callers/callees: callers include `train_autoencoder.py`, `train_diffusion.py`, `generate.py`, `evaluate_generation.py`, `utils/check_configs.py`, and tests. Callees include `_load_data`, `_load_modalities`, `_load_autoencoder`, `_load_diffusion`, `_load_training`, `_load_sampler`, `_load_initialization`, `_load_export`, and `_load_evaluation`.
- Reuse guidance: use this for every sleep2wave stage config instead of parsing YAML in entrypoints.
- Duplication-risk notes: do not repeat required/disallowed stage block logic elsewhere.

## Stage loaders

- File: `sleep2wave/generative/config.py`
- Functions:
  - `_load_data(raw) -> DataConfig`
  - `_load_modalities(raw) -> ModalitiesConfig`
  - `_load_autoencoder(raw, configured_modalities) -> AutoencoderConfig`
  - `_load_diffusion(raw) -> DiffusionConfig`
  - `_load_training(raw) -> TrainingConfig`
  - `_load_inference(raw) -> InferenceConfig`
  - `_load_sampler(raw, diffusion_cfg) -> SamplerConfig`
  - `_load_initialization(raw) -> InitializationConfig | None`
  - `_load_evaluation(raw) -> EvaluationConfig`
- Purpose and contract: parse individual schema blocks and reject unsupported fields, including autoencoder and diffusion validation example logging config, task-aware training/inference corruption specs with optional weighted choices, restoration condition-count sampling, evaluation corruption mask policy, and phase checkpoint paths.
- Important inputs/outputs: raw mapping in; typed config out.
- Side effects: none.
- Key callers/callees: called by `load_sleep2wave_config`.
- Reuse guidance: keep validation close to each block when adding fields.
- Duplication-risk notes: `sampler.steps` and DDPM constraints are validated here and in sampler constructors; keep them aligned.

## `sleep2wave.config.load_pretrain_config`

- File: `sleep2wave/config.py`
- Signature: `load_pretrain_config(path: str | Path) -> PretrainConfigBundle`
- Purpose and contract: package-local parser for legacy-style Sleep2Vec pretrain/adapt YAML under the sleep2wave namespace.
- Important inputs/outputs: YAML path in; local `PretrainConfigBundle` out.
- Side effects: reads YAML only.
- Key callers/callees: used by package-local pretrain/adapt code and `utils/check_configs.py` for non-generative `configs/sleep2wave` recipes.
- Reuse guidance: use for package-local pretrain/adapt config behavior.
- Duplication-risk notes: keep local imports under `sleep2wave.*`.

## `sleep2wave.config.load_finetune_config`

- File: `sleep2wave/config.py`
- Signature: `load_finetune_config(path: str | Path) -> FinetuneConfigBundle`
- Purpose and contract: package-local parser for finetune/infer YAML; rejects LoRA flags for the standalone RoFormer path.
- Important inputs/outputs: YAML path in; local `FinetuneConfigBundle` out.
- Side effects: reads YAML only.
- Key callers/callees: used by package-local finetune/infer code, `utils/check_configs.py`, and `tests/test_sleep2wave_namespace.py`.
- Reuse guidance: use for every non-generative sleep2wave finetune config.
- Duplication-risk notes: do not bypass its LoRA rejection with direct `LoraConfig` construction.

## `sleep2wave.config.validate_model_config`

- File: `sleep2wave/config.py`
- Signature: `validate_model_config(model_cfg: ModelConfig) -> int`
- Purpose and contract: enforce shared tokenizer output dimensions plus CLS and head option validity.
- Important inputs/outputs: `ModelConfig` in; shared feature dimension out.
- Side effects: none.
- Key callers/callees: builders, tests, and config checks.
- Reuse guidance: call before constructing package-local tokenizers or heads from config.
- Duplication-risk notes: tokenizer parity checks should not be reimplemented in tooling.

## `utils.check_configs.CONFIG_VARIANTS`

- File: `utils/check_configs.py`
- Value: `{"sleep2wave": ("sleep2wave.config", "sleep2wave.preprocess.save_dataset_presets")}`
- Purpose and contract: route configs under `configs/sleep2wave` to package-local config and preset helpers.
- Important inputs/outputs: config path in; variant-specific tool bundle out through `_load_config_tools`.
- Side effects: imports package modules dynamically.
- Key callers/callees: `check_config_file`.
- Reuse guidance: add variant roots here only when they have package-local loaders.
- Duplication-risk notes: do not special-case sleep2wave config paths in shell scripts.

## `utils.check_configs._validate_sleep2wave_generative_config`

- File: `utils/check_configs.py`
- Signature: `_validate_sleep2wave_generative_config(path: Path) -> None`
- Purpose and contract: validate `recipe: sleep2wave` stage configs by calling `sleep2wave.generative.config.load_sleep2wave_config`.
- Important inputs/outputs: path in; raises on invalid config.
- Side effects: dynamic import and YAML read.
- Key callers/callees: called by `check_config_file` when `_is_sleep2wave_generative_config` is true.
- Reuse guidance: keep generative-stage validation delegated to the canonical parser.
- Duplication-risk notes: avoid duplicating sleep2wave stage validation in `utils/check_configs.py`.

## Tests

- `tests/test_sleep2wave_generative_config.py`
- `tests/test_sleep2wave_namespace.py`
- `tests/test_check_configs.py`
