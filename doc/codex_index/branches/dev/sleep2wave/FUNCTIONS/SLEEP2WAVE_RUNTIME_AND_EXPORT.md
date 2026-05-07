# sleep2wave Runtime And Export Functions

## `sleep2wave.train_autoencoder.build_dataloader`

- File: `sleep2wave/train_autoencoder.py`
- Signature: `build_dataloader(config, *, num_workers: int, split: str = "train")`
- Purpose and contract: build a split-specific DataLoader for autoencoder-stage configs.
- Important inputs/outputs: typed config in; DataLoader out.
- Side effects: dataset reads preset/index and NPZ files during iteration.
- Key callers/callees: `train_autoencoder`.
- Reuse guidance: keep autoencoder data-loader setup here.

## `sleep2wave.train_autoencoder.train_autoencoder`

- File: `sleep2wave/train_autoencoder.py`
- Signature: `train_autoencoder(args: argparse.Namespace) -> Path`
- Purpose and contract: run sleep2wave autoencoder training/validation and write a run directory with checkpoints and config snapshots.
- Important inputs/outputs: CLI args with config and version name in; run directory path out.
- Side effects: creates output/checkpoint directories, persists config/args, initializes W&B logger, trains/validates with Lightning, logs validation waveform examples, writes `last.ckpt`.
- Key callers/callees: `main`; callees include `load_sleep2wave_config`, `persist_run_config_and_args`, `Sleep2WaveAutoencoderLightning`, optional `load_sleep2vec2_initialization`.
- Reuse guidance: use as the autoencoder training entrypoint.

## `sleep2wave.train_diffusion.build_dataloader`

- File: `sleep2wave/train_diffusion.py`
- Signature: `build_dataloader(config, *, num_workers: int, seed: int, split: str = "train")`
- Purpose and contract: build a split-specific DataLoader for diffusion-stage configs with available-modality bucketed waveform batches, or a latent-cache DataLoader when `diffusion.autoencoder_checkpoint` is omitted; cache-only validation falls back to train rows for train-only cache artifacts.
- Important inputs/outputs: typed config and seed in; DataLoader out.
- Side effects: dataset reads files during iteration.
- Key callers/callees: `train_diffusion`; callee `AvailableChannelsBucketBatchSampler`.
- Reuse guidance: keep diffusion train data-loader setup here.

## `sleep2wave.train_diffusion.train_diffusion`

- File: `sleep2wave/train_diffusion.py`
- Signature: `train_diffusion(args: argparse.Namespace) -> Path`
- Purpose and contract: run sleep2wave latent diffusion training/validation and write a run directory with checkpoints and config snapshots.
- Important inputs/outputs: CLI args in; run directory path out.
- Side effects: seeds Lightning, creates directories, persists config/args, optionally initializes from `training.phase_checkpoint`, initializes W&B logger, trains/validates with Lightning, logs validation waveform examples, writes `last.ckpt`.
- Key callers/callees: `main`; callees include `load_sleep2wave_config`, `Sleep2WaveDiffusionLightning`, optional `load_sleep2vec2_initialization`.
- Reuse guidance: use as the diffusion training entrypoint.

## `sleep2wave.generate._load_diffusion_model`

- File: `sleep2wave/generate.py`
- Signature: `_load_diffusion_model(checkpoint_path: Path, config: Sleep2WaveConfig, *, device: torch.device) -> Sleep2WaveDiffusionTransformer`
- Purpose and contract: load a sleep2wave diffusion checkpoint into a config-built transformer.
- Important inputs/outputs: checkpoint path and config in; eval-mode model out.
- Side effects: reads checkpoint and loads state dict.
- Key callers/callees: `run_generation`.
- Reuse guidance: use for generation-time diffusion checkpoint loading.

## `sleep2wave.generate.run_generation`

- File: `sleep2wave/generate.py`
- Signature: `run_generation(args: argparse.Namespace) -> Path`
- Purpose and contract: generate missing or target PSG modalities from incomplete recordings.
- Important inputs/outputs: CLI args in; output artifact directory path out.
- Side effects: reads config, checkpoints, preset/index and NPZ data; writes generated artifacts.
- Key callers/callees: `main`; callees include `load_sleep2wave_config`, `build_generation_task`, `_resolve_inference_corruption_specs`, `load_sleep2wave_autoencoder_checkpoint`, `_load_diffusion_model`, `_collect_generation_windows`, `_fuse_generated`, `_fuse_masks`, `compute_uncertainty`, `build_generation_manifest`, and `write_generation_artifacts`.
- Reuse guidance: use as the only generation orchestration path.
- Duplication-risk notes: keep artifact schema and evaluation compatibility here.

## `sleep2wave.generate_batch.run_batch_generation`

- File: `sleep2wave/generate_batch.py`
- Signature: `run_batch_generation(args: argparse.Namespace) -> list[Path]`
- Purpose and contract: group preset/index windows by subject/night and call the canonical one-night `run_generation` path once per group.
- Important inputs/outputs: generation CLI args in; generated artifact directories out.
- Side effects: writes temporary per-night preset pickles and generation artifacts.
- Reuse guidance: keep batch generation as a wrapper; do not duplicate artifact or sampler logic.

## `sleep2wave.generate._collect_generation_windows`

- File: `sleep2wave/generate.py`
- Signature: `_collect_generation_windows(*, config, args, model, autoencoder, sampler_config, task, device) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]], list[int], list[dict[str, Any]]]`
- Purpose and contract: iterate dataset windows, apply inference condition corruptions or externally supplied condition masks, sample target latents, decode them, collect masks, and record metadata.
- Important inputs/outputs: model/data/task context in; generated windows, mask windows, start epochs, metadata rows out.
- Side effects: reads data and runs model inference.
- Key callers/callees: `run_generation`; callee `validate_single_night` currently limits generation to one subject/night.
- Reuse guidance: change collection semantics here rather than in artifact writing.

## `sleep2wave.inference.sliding_window.fuse_overlapping_windows`

- File: `sleep2wave/inference/sliding_window.py`
- Signature: `fuse_overlapping_windows(windows: torch.Tensor, start_epochs: Sequence[int], *, mode: str = "mean", eps: float = 1e-6) -> FusedWindowTensor`
- Purpose and contract: fuse generated sliding-window tensors into contiguous epoch-indexed output.
- Important inputs/outputs: `[num_samples, windows, context_epochs, ...]` tensor and starts in; fused tensor and epoch index out.
- Side effects: none.
- Key callers/callees: `generate._fuse_generated`.
- Reuse guidance: use for all generated waveform fusion.

## `sleep2wave.inference.sliding_window.fuse_mask_windows`

- File: `sleep2wave/inference/sliding_window.py`
- Signature: `fuse_mask_windows(windows: torch.Tensor, start_epochs: Sequence[int], *, mode: str) -> FusedWindowTensor`
- Purpose and contract: fuse per-window masks with `any` or `mean` semantics.
- Important inputs/outputs: mask windows and starts in; fused mask and epoch index out.
- Side effects: none.
- Key callers/callees: `generate._fuse_masks`.
- Reuse guidance: keep mask fusion here.

## `sleep2wave.inference.uncertainty.compute_uncertainty`

- File: `sleep2wave/inference/uncertainty.py`
- Signature: `compute_uncertainty(generated: dict[str, torch.Tensor], *, high_uncertainty_threshold: float | None = None) -> dict[str, ModalityUncertainty]`
- Purpose and contract: summarize generated samples by modality with mean, std, sample count, and high-uncertainty mask.
- Important inputs/outputs: generated sample tensor dict in; uncertainty dict out.
- Side effects: none.
- Key callers/callees: `generate.run_generation`.
- Reuse guidance: use before writing uncertainty artifacts.

## `sleep2wave.export.manifest.build_generation_manifest`

- File: `sleep2wave/export/manifest.py`
- Signature: `build_generation_manifest(*, task_type, condition_modalities, target_modalities, diffusion_ckpt, autoencoder_ckpt, sampler, output_files) -> dict[str, Any]`
- Purpose and contract: build the generation artifact manifest with provenance and clinical-use metadata.
- Important inputs/outputs: generation context in; JSON-serializable manifest out.
- Side effects: none.
- Key callers/callees: `generate.run_generation`.
- Reuse guidance: update this when the artifact schema changes.

## `sleep2wave.export.artifacts.write_generation_artifacts`

- File: `sleep2wave/export/artifacts.py`
- Signature: `write_generation_artifacts(output_dir, *, generated, uncertainty, masks, metadata_rows, manifest, config_path, args) -> Path`
- Purpose and contract: write generation outputs to `generated.npz`, `uncertainty.npz`, `masks.npz`, `metadata.jsonl`, `config.yaml`, `cli_args.yaml`, and `manifest.json`.
- Important inputs/outputs: artifact payloads in; output directory path out.
- Side effects: creates directories and writes files.
- Key callers/callees: `generate.run_generation`.
- Reuse guidance: keep all generation artifact writing here.

## Package-local copied runtime entrypoints

- Files:
  - `sleep2wave/pretrain.py`
  - `sleep2wave/adapt.py`
  - `sleep2wave/finetune.py`
  - `sleep2wave/infer.py`
- Purpose and contract: run package-local equivalents of base pretrain/adapt/finetune/infer flows.
- Reuse guidance: use only when running sleep2wave-local runtime configs; for base `sleep2vec` configs use base entrypoints.
- Duplication-risk notes: do not mix base and package-local imports.

## Tests

- `tests/test_sleep2wave_autoencoder_train_smoke.py`
- `tests/test_sleep2wave_diffusion_train_smoke.py`
- `tests/test_sleep2wave_generate_cli.py`
- `tests/test_sleep2wave_export_artifacts.py`
- `tests/test_sleep2wave_sliding_window.py`
