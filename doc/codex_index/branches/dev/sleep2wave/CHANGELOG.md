# Branch Index Changelog

## 2026-05-08

- Updated after Phase 2B padded multi-channel diffusion:
  - diffusion token layout now includes modality, epoch, channel, and patch
  - `Sleep2WaveDiffusionTransformer` predicts `[B, E, C, L, D]` noise with `channel_mask`
  - diffusion configs require `embeddings.channel_position`
  - latent-cache migration remains deferred to Phase 3

## 2026-05-05

- Initialized the `dev/sleep2wave` branch index at commit `55458eba899c81026710d31c31a3143501d911bd`.
- Used `doc/codex_index/branches/main/` at baseline commit `12350da513fe1b011c8eb10671e75ca5f139857f` as the inherited base-runtime reference.
- Added branch-specific coverage for:
  - package-local `sleep2wave` runtime mirror
  - sleep2wave generative config stages
  - modality and generative dataset contracts
  - autoencoder training
  - latent diffusion training
  - generation artifact export
  - generation evaluation
  - sleep2wave config validation routing
- No stale branch-index entries were removed because this was a new branch index.
- Updated after the Sleep2Wave gap-closure implementation:
  - task-aware restoration/imputation corruptions are configured under `training.corruptions`
  - replay-enabled diffusion phase defaults include imputation tasks
  - `training.phase_checkpoint` and `--resume-from-checkpoint` separate phase continuation from crash resume
  - IBI/RESP are treated as given primary channels; sidecar derivation and `derived_path` fallback were removed
  - autoencoder decoding uses ConvTranspose1d and supports padded multi-channel inputs through `channel_mask`
  - cache-only diffusion is supported for translation/partial-full task mixes
  - medium Sleep2Wave YAMLs were added under `configs/sleep2wave`
  - generated-signal event adapters and per-night batch generation wrappers were added

## Unresolved Ambiguities

- Exact production data paths and real sleep2wave artifact conventions are unknown from tracked source alone.
