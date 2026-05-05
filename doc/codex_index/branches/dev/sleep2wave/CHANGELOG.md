# Branch Index Changelog

## 2026-05-05

- Initialized the `dev/sleep2wave` branch index at commit `55458eba899c81026710d31c31a3143501d911bd`.
- Used `doc/codex_index/branches/main/` at baseline commit `12350da513fe1b011c8eb10671e75ca5f139857f` as the inherited base-runtime reference.
- Added branch-specific coverage for:
  - package-local `sleep2wave` runtime mirror
  - Sleep2Wave generative config stages
  - modality and generative dataset contracts
  - autoencoder training
  - latent diffusion training
  - generation artifact export
  - generation evaluation
  - Sleep2Wave config validation routing
- No stale branch-index entries were removed because this was a new branch index.

## Unresolved Ambiguities

- Replay config and `sleep2wave/training/replay_buffer.py` exist, but replay is not wired into the inspected diffusion Lightning training path.
- `diffusion.latent_cache_path` exists in config, but cache-only diffusion training is rejected.
- Exact production data paths and real Sleep2Wave artifact conventions are unknown from tracked source alone.
