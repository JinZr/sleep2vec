# Main Branch Codex Engineering Index

This directory is the branch-scoped engineering manual for `main`. It is intended to be consulted before editing code in this repository.

## Branch Scope

- Branch: `main`
- Commit: `825a30433e1f3d4cfcf6e4338cde5c29426411f3`
- Generated at: `2026-03-25T08:31:56Z`
- Mode: `initialize-main`

## Purpose

Use this index to answer four questions before making changes:

1. Which module owns the behavior I need to change?
2. Which existing function or class should I reuse instead of reimplementing?
3. Which runtime path will exercise the change?
4. Which test or contract file already defines the expected behavior?

The index is intentionally biased toward contract-bearing, reuse-relevant APIs rather than exhaustively documenting every trivial helper.

## Recommended Reading Order

1. [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md) for the runtime architecture and batch contracts.
2. [MODULE_MAP.md](./MODULE_MAP.md) for ownership boundaries and extension points.
3. [REUSE_GUIDE.md](./REUSE_GUIDE.md) for canonical functions and duplicate-implementation traps.
4. Relevant workflow file under [WORKFLOWS/](./WORKFLOWS/).
5. Relevant family catalog under [FUNCTIONS/](./FUNCTIONS/).

## Coverage

Tracked files indexed from this branch:

- `sleep2vec/`: 58 tracked files
- `data/`: 6 tracked files
- `preprocess/`: 6 tracked files
- `configs/`: 12 tracked files
- `tests/`: 10 tracked files
- `utils/`: 1 tracked file

Branch-state coverage only:

- `sleep2vec2/`: no tracked source files on `main`
- `sleep2vec_moe/`: no tracked source files on `main`
- `sleep2vec_hires/`: no tracked source files on `main`

## Coverage Boundaries

- Indexed in detail: CLI entrypoints, config loaders, builders/registries, core model/runtime classes, dataset/sampler contracts, preprocessing CLIs, checkpoint helpers, diagnostics/visualization hooks, and the tests that pin key contracts.
- Indexed at module/workflow level only: `preprocess/preprocess_pipeline.ipynb`.
- Not indexed as source of truth: `__pycache__/`, `.DS_Store`, ignored local artifacts, and untracked experiment folders such as `index/` and `new_index/`.
- `AGENTS.md` is referenced for ownership context but is not reproduced here as an editable source of truth.

## How To Use This Index

- If you are changing YAML semantics, start with [FUNCTIONS/CONFIG_AND_REGISTRIES.md](./FUNCTIONS/CONFIG_AND_REGISTRIES.md).
- If you are changing training or inference orchestration, start with [FUNCTIONS/RUNTIME_ORCHESTRATION.md](./FUNCTIONS/RUNTIME_ORCHESTRATION.md) and the relevant workflow.
- If you are changing model forward behavior or downstream heads, start with [FUNCTIONS/MODELS_AND_HEADS.md](./FUNCTIONS/MODELS_AND_HEADS.md).
- If you are changing dataset loading, missing-channel behavior, presets, or samplers, start with [FUNCTIONS/DATASETS_AND_SAMPLERS.md](./FUNCTIONS/DATASETS_AND_SAMPLERS.md).
- If you are changing CSV splitting, preset generation, missing-mask statistics, or WatchPAT conversion, start with [FUNCTIONS/PREPROCESSING_AND_CONVERSION.md](./FUNCTIONS/PREPROCESSING_AND_CONVERSION.md).
- If you are touching visualization or diagnostics code, start with [FUNCTIONS/VISUALIZATION_AND_DIAGNOSTICS.md](./FUNCTIONS/VISUALIZATION_AND_DIAGNOSTICS.md).
- If you think you need `sleep2vec2/`, `sleep2vec_moe/`, or `sleep2vec_hires/`, read [FUNCTIONS/VARIANT_SURFACES.md](./FUNCTIONS/VARIANT_SURFACES.md) first; on this branch they are placeholders, not active tracked implementations.

## Deliverable Layout

- [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md)
- [MODULE_MAP.md](./MODULE_MAP.md)
- [REUSE_GUIDE.md](./REUSE_GUIDE.md)
- [CHANGELOG.md](./CHANGELOG.md)
- [FUNCTIONS/](./FUNCTIONS/)
- [WORKFLOWS/](./WORKFLOWS/)
- [MANIFEST.json](./MANIFEST.json)

## Reliability Notes

- Every claim in this index is grounded in the current tracked code on `main`.
- If behavior is unclear from source, the index says `unknown` rather than inferring.
- Some runtime paths were only statically inspected due environment limits; those areas are explicitly marked in the relevant pages.
