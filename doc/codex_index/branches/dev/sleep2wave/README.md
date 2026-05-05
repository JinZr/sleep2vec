# Dev Sleep2Wave Branch Codex Engineering Index

This directory is the branch-scoped engineering manual for `dev/sleep2wave`.

## Branch Scope

- Branch: `dev/sleep2wave`
- Indexed commit: `55458eba899c81026710d31c31a3143501d911bd`
- Main baseline commit: `12350da513fe1b011c8eb10671e75ca5f139857f`
- Generated at: `2026-05-05T09:01:33Z`
- Mode: `initialize-branch`

## Purpose

Use this index before changing branch behavior to answer:

1. Whether the behavior belongs to the inherited `sleep2vec` runtime or the additive `sleep2wave` namespace.
2. Which package-local implementation is canonical for Sleep2Wave data, model, generation, or evaluation work.
3. Which config loader and workflow exercise the change.
4. Which tests already pin the contract.

## Recommended Reading Order

1. [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md)
2. [MODULE_MAP.md](./MODULE_MAP.md)
3. [REUSE_GUIDE.md](./REUSE_GUIDE.md)
4. [DELTA_FROM_MAIN.md](./DELTA_FROM_MAIN.md)
5. Relevant page under [FUNCTIONS/](./FUNCTIONS/)
6. Relevant page under [WORKFLOWS/](./WORKFLOWS/)

For small fixes, read this page plus the single relevant function or workflow page.

## Coverage

Tracked files indexed from this branch:

- `sleep2vec/`: 78 tracked files, inherited from the main branch index
- `data/`: 6 tracked files, inherited from the main branch index
- `preprocess/`: 6 tracked files, inherited from the main branch index
- `sleep2wave/`: 144 tracked files, indexed in this branch manual
- `configs/`: 36 tracked files, including 4 `configs/sleep2wave/*.yaml`
- `tests/`: 52 tracked files, including 28 `tests/test_sleep2wave_*.py`
- `utils/`: 2 tracked files, with `utils/check_configs.py` extended for Sleep2Wave configs
- `sleep2vec2/`, `sleep2vec_moe/`, `sleep2vec_hires/`: no tracked source files on this branch

## Branch-Specific Surface

`dev/sleep2wave` adds a standalone `sleep2wave` package. It contains:

- a package-local mirror of core `sleep2vec` pretrain/adapt/finetune/infer contracts
- a standalone RoFormer backbone implementation used instead of legacy HF RoFormer checkpoint keys
- a new Sleep2Wave generative stack for waveform autoencoding, latent diffusion, generation artifacts, and evaluation
- Sleep2Wave-specific preprocessing and schema-versioned generative presets

## Deliverable Layout

- [MANIFEST.json](./MANIFEST.json)
- [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md)
- [MODULE_MAP.md](./MODULE_MAP.md)
- [REUSE_GUIDE.md](./REUSE_GUIDE.md)
- [CHANGELOG.md](./CHANGELOG.md)
- [DELTA_FROM_MAIN.md](./DELTA_FROM_MAIN.md)
- [FUNCTIONS/BASELINE_SLEEP2VEC.md](./FUNCTIONS/BASELINE_SLEEP2VEC.md)
- [FUNCTIONS/SLEEP2WAVE_CONFIG.md](./FUNCTIONS/SLEEP2WAVE_CONFIG.md)
- [FUNCTIONS/SLEEP2WAVE_GENERATIVE_DATA.md](./FUNCTIONS/SLEEP2WAVE_GENERATIVE_DATA.md)
- [FUNCTIONS/SLEEP2WAVE_MODELS.md](./FUNCTIONS/SLEEP2WAVE_MODELS.md)
- [FUNCTIONS/SLEEP2WAVE_RUNTIME_AND_EXPORT.md](./FUNCTIONS/SLEEP2WAVE_RUNTIME_AND_EXPORT.md)
- [FUNCTIONS/SLEEP2WAVE_EVALUATION.md](./FUNCTIONS/SLEEP2WAVE_EVALUATION.md)
- [WORKFLOWS/BASE_SLEEP2VEC_RUNTIME.md](./WORKFLOWS/BASE_SLEEP2VEC_RUNTIME.md)
- [WORKFLOWS/SLEEP2WAVE_CONFIG_VALIDATION.md](./WORKFLOWS/SLEEP2WAVE_CONFIG_VALIDATION.md)
- [WORKFLOWS/SLEEP2WAVE_PREPROCESSING.md](./WORKFLOWS/SLEEP2WAVE_PREPROCESSING.md)
- [WORKFLOWS/SLEEP2WAVE_AUTOENCODER.md](./WORKFLOWS/SLEEP2WAVE_AUTOENCODER.md)
- [WORKFLOWS/SLEEP2WAVE_DIFFUSION.md](./WORKFLOWS/SLEEP2WAVE_DIFFUSION.md)
- [WORKFLOWS/SLEEP2WAVE_GENERATION.md](./WORKFLOWS/SLEEP2WAVE_GENERATION.md)
- [WORKFLOWS/SLEEP2WAVE_EVALUATION.md](./WORKFLOWS/SLEEP2WAVE_EVALUATION.md)

## Reliability Notes

- Product code was not modified while building this index.
- The inherited base runtime is referenced through `doc/codex_index/branches/main/` where the branch did not change the source.
- Sleep2Wave claims in this index were checked against tracked branch code.
- Unknown behavior is marked as `unknown` rather than inferred.
