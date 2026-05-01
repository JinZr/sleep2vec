# dev/sleep2expert Branch Codex Engineering Index

This directory is the branch-scoped engineering manual for `dev/sleep2expert`. Use it before broader code changes, and use a quick consult for small localized fixes.

## Branch Scope

- Branch: `dev/sleep2expert`
- Last full refresh commit: `21bbd67bc7b69dce4b119141cd201779688f016c`
- Last full refresh at: `2026-05-01T05:06:42Z`
- Mode: `initialize-branch`

## Purpose

Use this index to answer four questions before making changes:

1. Which module owns the behavior I need to change?
2. Which existing function or class should I reuse instead of reimplementing?
3. Which runtime path will exercise the change?
4. Which test or contract file already defines the expected behavior?

The manual is intentionally biased toward contract-bearing, reuse-relevant APIs rather than exhaustively listing trivial helpers.

## Recommended Reading Order

1. [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md) for the current runtime architecture, task semantics, and batch contract.
2. [MODULE_MAP.md](./MODULE_MAP.md) for edit boundaries, ownership seams, and dependency flow.
3. [REUSE_GUIDE.md](./REUSE_GUIDE.md) for canonical implementations and duplication traps.
4. Relevant workflow file under [WORKFLOWS/](./WORKFLOWS/).
5. Relevant family catalog under [FUNCTIONS/](./FUNCTIONS/).

For small, localized fixes or routine updates, it is enough to read this page plus the single most relevant page from `REUSE_GUIDE.md`, `MODULE_MAP.md`, `WORKFLOWS/`, or `FUNCTIONS/`.

## Coverage

Tracked files indexed from this branch:

- `sleep2vec/`: 78 tracked files
- `data/`: 6 tracked files
- `preprocess/`: 6 tracked files
- `configs/`: 64 indexed files, including `configs/sleep2vec2/`
- `tests/`: 26 indexed files
- `utils/`: 2 tracked files
- `sleep2vec2/`: 97 indexed files, including 87 Python source files and 9 visualization font assets

Branch-state coverage only:

- `sleep2vec_moe/`: no tracked source files on this branch
- `sleep2vec_hires/`: no tracked source files on this branch

## Coverage Boundaries

- Indexed in detail: config loaders and task semantics, runtime entrypoints, checkpoint helpers, result writing, adaptation orchestration, backbone/downstream contracts, dataset/sampler contracts, preprocessing CLIs, downstream evaluation visualizations, and the tests that pin those contracts.
- `sleep2vec2/` is an active standalone mirror of the base recipe, summarized in the variant-surface catalog. Its key branch delta is package-local `data/`, `preprocess/`, copied YAMLs under `configs/sleep2vec2/`, copied visualization font assets, and a standalone RoFormer backbone under `sleep2vec2/backbones/roformer/`.
- Indexed at module or workflow level only: `preprocess/preprocess_pipeline.ipynb` and tracked visualization font binaries under `sleep2vec/visualization/assets/fonts/`.
- Not indexed as source of truth: `__pycache__/`, `.DS_Store`, ignored local artifacts, and untracked experiment folders such as `index/` and `new_index/`.
- `AGENTS.md` is referenced for ownership context but is not reproduced here as an editable source of truth.

## How To Use This Index

- For a small local fix, start here, then jump to one relevant section instead of doing the full handbook pass.
- For a broader behavior or contract change, follow the full reading order above.

- If you are changing YAML semantics, built-in task behavior, or config validation, start with [FUNCTIONS/CONFIG_AND_REGISTRIES.md](./FUNCTIONS/CONFIG_AND_REGISTRIES.md).
- If you are changing pretrain, adapt, finetune, inference, checkpoint, or result-export orchestration, start with [FUNCTIONS/RUNTIME_ORCHESTRATION.md](./FUNCTIONS/RUNTIME_ORCHESTRATION.md) and the relevant workflow.
- If you are changing backbone forward behavior, adaptation freeze policy, downstream heads, AHI epoch reduction, or layer mix, start with [FUNCTIONS/MODELS_AND_HEADS.md](./FUNCTIONS/MODELS_AND_HEADS.md).
- If you are changing dataset loading, built-in AHI sample validation, missing-channel behavior, or samplers, start with [FUNCTIONS/DATASETS_AND_SAMPLERS.md](./FUNCTIONS/DATASETS_AND_SAMPLERS.md).
- If you are changing CSV splitting, preset generation, preset-build strictness, missing-mask statistics, or WatchPAT conversion, start with [FUNCTIONS/PREPROCESSING_AND_CONVERSION.md](./FUNCTIONS/PREPROCESSING_AND_CONVERSION.md).
- If you are changing evaluation plots, pair-accuracy logging, or diagnostics, start with [FUNCTIONS/VISUALIZATION_AND_DIAGNOSTICS.md](./FUNCTIONS/VISUALIZATION_AND_DIAGNOSTICS.md).
- If you think you need `sleep2vec2/`, `sleep2vec_moe/`, or `sleep2vec_hires/`, read [FUNCTIONS/VARIANT_SURFACES.md](./FUNCTIONS/VARIANT_SURFACES.md) first; on this branch `sleep2vec2/` is active and the other variant roots remain placeholders.

## Deliverable Layout

- [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md)
- [MODULE_MAP.md](./MODULE_MAP.md)
- [REUSE_GUIDE.md](./REUSE_GUIDE.md)
- [CHANGELOG.md](./CHANGELOG.md)
- [FUNCTIONS/](./FUNCTIONS/)
- [WORKFLOWS/](./WORKFLOWS/)
- [MANIFEST.json](./MANIFEST.json)

Current workflow coverage:

- [WORKFLOWS/PRETRAIN.md](./WORKFLOWS/PRETRAIN.md)
- [WORKFLOWS/ADAPT.md](./WORKFLOWS/ADAPT.md)
- [WORKFLOWS/FINETUNE.md](./WORKFLOWS/FINETUNE.md)
- [WORKFLOWS/INFER_AND_CHECKPOINTS.md](./WORKFLOWS/INFER_AND_CHECKPOINTS.md)
- [WORKFLOWS/PREPROCESSING.md](./WORKFLOWS/PREPROCESSING.md)
- [WORKFLOWS/CONFIG_VALIDATION.md](./WORKFLOWS/CONFIG_VALIDATION.md)

## Reliability Notes

- Every claim in this index is grounded in the current `dev/sleep2expert` working tree.
- If behavior is unclear from source, the index says `unknown` rather than inferring.
- Some runtime paths were only statically inspected due environment limits; those areas are explicitly marked in the relevant pages.
