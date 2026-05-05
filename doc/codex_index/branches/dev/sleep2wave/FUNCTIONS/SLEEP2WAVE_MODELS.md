# Sleep2Wave Model Functions

## `sleep2wave.autoencoders.model.Sleep2WaveAutoencoder`

- File: `sleep2wave/autoencoders/model.py`
- Signature: `Sleep2WaveAutoencoder(*, latent_dim: int, encoder_type: str = "conv1d_epoch", decoder_type: str = "convtranspose1d_epoch", modalities: Sequence[str] = CANONICAL_MODALITIES)`
- Purpose and contract: build modality-specific waveform autoencoders and return one latent vector per epoch for each modality.
- Important inputs/outputs: `clean_signals` dict in; `Sleep2WaveAutoencoderOutput(latents, reconstructions)` out.
- Side effects: module construction and forward computation.
- Key callers/callees: `Sleep2WaveAutoencoderLightning`, diffusion autoencoder loading, generation decoding.
- Reuse guidance: use this for all Sleep2Wave waveform latent encoding and decoding.
- Duplication-risk notes: do not create separate per-modality autoencoder classes outside this module.

## `sleep2wave.autoencoders.model.Sleep2WaveAutoencoder.decode_latents`

- File: `sleep2wave/autoencoders/model.py`
- Signature: `decode_latents(self, latents: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]`
- Purpose and contract: decode `[B, E, D]` latent tensors into modality waveform reconstructions.
- Important inputs/outputs: modality latent dict in; modality waveform dict out.
- Side effects: forward computation.
- Key callers/callees: `generate._decode_generated_latents`.
- Reuse guidance: use for generation instead of manually calling branch decoders.

## `sleep2wave.autoencoders.losses.compute_autoencoder_loss`

- File: `sleep2wave/autoencoders/losses.py`
- Signature: `compute_autoencoder_loss(reconstructions, targets, *, availability_mask=None, quality_mask=None, config: AutoencoderLossConfig) -> dict[str, torch.Tensor]`
- Purpose and contract: compute masked waveform L1, waveform L2, spectral, and total autoencoder losses.
- Important inputs/outputs: reconstruction/target dicts plus masks in; loss dict out.
- Side effects: none.
- Key callers/callees: `Sleep2WaveAutoencoderLoss.forward`, autoencoder Lightning training.
- Reuse guidance: keep loss changes here rather than inside the Lightning module.

## `sleep2wave.autoencoders.lightning.Sleep2WaveAutoencoderLightning`

- File: `sleep2wave/autoencoders/lightning.py`
- Signature: `Sleep2WaveAutoencoderLightning(config: Sleep2WaveConfig)`
- Purpose and contract: Lightning wrapper for autoencoder training, checkpoint config persistence, optimizer grouping, and train-step logging.
- Important inputs/outputs: typed autoencoder-stage config in; training loss out.
- Side effects: logs metrics and writes config into checkpoints.
- Key callers/callees: `train_autoencoder.train_autoencoder`.
- Reuse guidance: use for autoencoder training; keep trainer wiring in `train_autoencoder.py`.

## `sleep2wave.diffusion.tasks.GenerationTask`

- File: `sleep2wave/diffusion/tasks.py`
- Dataclass fields: `task_type`, `condition_modalities`, `target_modalities`, `use_auxiliary_token`, `allow_target_target_attention`
- Purpose and contract: represent one restoration, imputation, translation, or partial_full generation task.
- Reuse guidance: pass tasks through `validate_generation_task` or construct with `build_generation_task`.

## `sleep2wave.diffusion.tasks.build_generation_task`

- File: `sleep2wave/diffusion/tasks.py`
- Signature: `build_generation_task(task_type: str, *, condition_modalities: Sequence[str], target_modalities: Sequence[str], auxiliary_restoration_token: bool = False, allow_target_target_attention: bool = True) -> GenerationTask`
- Purpose and contract: normalize and validate generation task semantics.
- Important inputs/outputs: symbolic task fields in; `GenerationTask` out.
- Side effects: none.
- Key callers/callees: task sampler and generation CLI.
- Reuse guidance: use instead of constructing `GenerationTask` directly.

## `sleep2wave.diffusion.task_masks.TokenLayout`

- File: `sleep2wave/diffusion/task_masks.py`
- Signature: `TokenLayout(modalities: tuple[str, ...] = CANONICAL_MODALITIES, context_epochs: int = 15, include_aux: bool = True)`
- Purpose and contract: map modality/epoch pairs to transformer token positions.
- Important inputs/outputs: modality and context definitions in; token index helpers out.
- Side effects: none.
- Key callers/callees: `Sleep2WaveDiffusionTransformer`.
- Reuse guidance: use for token position math instead of recomputing offsets.

## `sleep2wave.diffusion.task_masks.build_directional_task_attention_mask`

- File: `sleep2wave/diffusion/task_masks.py`
- Signature: `build_directional_task_attention_mask(task: GenerationTask, layout: TokenLayout, *, availability_mask: dict[str, torch.Tensor] | None = None, batch_size: int = 1) -> TaskAttentionMask`
- Purpose and contract: build condition/target/active token masks and blocked attention matrix for directional generation.
- Important inputs/outputs: task/layout/availability in; `TaskAttentionMask` out.
- Side effects: none.
- Key callers/callees: `Sleep2WaveDiffusionTransformer.forward`.
- Reuse guidance: keep task attention policy here.

## `sleep2wave.diffusion.model.Sleep2WaveDiffusionTransformer`

- File: `sleep2wave/diffusion/model.py`
- Signature: `Sleep2WaveDiffusionTransformer(*, latent_dim, hidden_size, num_layers, num_heads, mlp_ratio, diffusion_steps, context_epochs, modalities=CANONICAL_MODALITIES, use_diffusion_step_embedding=True, use_modality_embedding=True, use_epoch_position_embedding=True, use_sleep_night_position_embedding=True, use_availability_embedding=True, use_quality_embedding=True, include_aux=True)`
- Purpose and contract: denoise target modality latents conditioned on available modality latents and task masks.
- Important inputs/outputs: noisy target latents, timesteps, task, condition latents, masks, night position in; `Sleep2WaveDiffusionOutput(predicted_noise, task_mask)` out.
- Side effects: forward computation.
- Key callers/callees: diffusion Lightning and samplers; callees include `build_directional_task_attention_mask`.
- Reuse guidance: construct from config through `from_config` when possible.
- Duplication-risk notes: do not create task-specific forward paths outside the task/mask abstractions.

## `sleep2wave.diffusion.schedule.build_diffusion_schedule`

- File: `sleep2wave/diffusion/schedule.py`
- Signature: `build_diffusion_schedule(num_steps: int, beta_schedule: str = "cosine") -> DiffusionSchedule`
- Purpose and contract: build validated cosine beta schedule and derived alpha arrays.
- Important inputs/outputs: step count and schedule name in; `DiffusionSchedule` out.
- Side effects: none.
- Reuse guidance: use for training and inference sampling.

## `sleep2wave.diffusion.samplers.build_sampler`

- File: `sleep2wave/diffusion/samplers.py`
- Signature: `build_sampler(config: SamplerConfig, *, diffusion_steps: int, beta_schedule: str) -> BaseDiffusionSampler`
- Purpose and contract: construct a DDIM or DDPM sampler from typed config.
- Important inputs/outputs: sampler config and diffusion schedule settings in; sampler out.
- Side effects: none.
- Key callers/callees: `generate._collect_generation_windows`.
- Reuse guidance: use this instead of branching on sampler name in generation code.

## `sleep2wave.diffusion.lightning.Sleep2WaveDiffusionLightning`

- File: `sleep2wave/diffusion/lightning.py`
- Signature: `Sleep2WaveDiffusionLightning(config: Sleep2WaveConfig, *, seed: int = 0)`
- Purpose and contract: Lightning wrapper for latent diffusion training with frozen autoencoder encoding, sampled task curriculum, condition dropout, and AdamW optimization.
- Important inputs/outputs: diffusion-stage config in; training loss out.
- Side effects: loads autoencoder checkpoint, logs metrics, saves config into checkpoint.
- Key callers/callees: `train_diffusion.train_diffusion`.
- Reuse guidance: keep diffusion train-step behavior here and trainer setup in `train_diffusion.py`.

## `sleep2wave.training.phase_schedule.build_phase_schedule`

- File: `sleep2wave/training/phase_schedule.py`
- Signature: `build_phase_schedule(phase: int, task_mix: dict[str, float] | None = None) -> PhaseSchedule`
- Purpose and contract: resolve diffusion phase 1-5 into a validated task mix.
- Important inputs/outputs: phase and optional override in; `PhaseSchedule` out.
- Side effects: none.
- Key callers/callees: `Sleep2WaveTaskSampler`.
- Reuse guidance: use for curriculum changes instead of hardcoding task weights elsewhere.

## `sleep2wave.training.task_sampler.Sleep2WaveTaskSampler`

- File: `sleep2wave/training/task_sampler.py`
- Signature: `Sleep2WaveTaskSampler(*, modalities=CANONICAL_MODALITIES, phase: int, task_mix=None, condition_counts=None, auxiliary_restoration_token=True, seed=0)`
- Purpose and contract: sample availability-aware generation tasks from the phase schedule.
- Important inputs/outputs: optional availability mask in; `GenerationTask` out.
- Side effects: uses internal seeded random generator.
- Key callers/callees: `Sleep2WaveDiffusionLightning.training_step`.
- Reuse guidance: use for all diffusion task sampling.

## `sleep2wave.initialization.sleep2vec2.load_sleep2vec2_initialization`

- File: `sleep2wave/initialization/sleep2vec2.py`
- Signature: `load_sleep2vec2_initialization(module: nn.Module, checkpoint_path: str | Path, config: InitializationConfig, *, target_groups: set[str], device: str | torch.device = "cpu") -> Sleep2Vec2InitializationReport`
- Purpose and contract: selectively load compatible Sleep2Vec2 checkpoint groups into Sleep2Wave modules and report loaded, missing, skipped, and incompatible keys.
- Important inputs/outputs: module/checkpoint/config/group set in; report out.
- Side effects: reads checkpoint and loads module weights.
- Key callers/callees: `train_autoencoder.py`, `train_diffusion.py`, tests.
- Reuse guidance: use this for supported initialization groups instead of manual state-dict slicing.

## Tests

- `tests/test_sleep2wave_autoencoder_model.py`
- `tests/test_sleep2wave_autoencoder_losses.py`
- `tests/test_sleep2wave_autoencoder_train_smoke.py`
- `tests/test_sleep2wave_diffusion_model_shapes.py`
- `tests/test_sleep2wave_diffusion_losses.py`
- `tests/test_sleep2wave_diffusion_task_masks.py`
- `tests/test_sleep2wave_diffusion_tasks.py`
- `tests/test_sleep2wave_diffusion_schedule.py`
- `tests/test_sleep2wave_diffusion_sampler.py`
- `tests/test_sleep2wave_diffusion_train_smoke.py`
- `tests/test_sleep2wave_phase_schedule.py`
- `tests/test_sleep2wave_task_sampler.py`
- `tests/test_sleep2wave_sleep2vec2_init.py`
