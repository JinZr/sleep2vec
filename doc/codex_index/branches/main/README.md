# Main Branch Codex Engineering Index

This directory is the branch-scoped engineering manual for `main`. Use it before broader code changes, and use a quick consult for small localized fixes.

## Branch Scope

- Branch: `main`
- Last refresh commit: `d95c9d45d63479b0fc28b011e138002691921104`
- Last refresh at: `2026-06-21T04:32:55Z`
- Mode: `refresh`

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

- `sleep2vec/`: 87 tracked files
- `data/`: 9 tracked files
- `preprocess/`: 7 tracked files
- `sleep2vec2/`: 110 tracked files
- `sleep2expert/`: 115 tracked files
- `sleep2stat/`: 30 tracked files
- `sleep2vec_moe/`: 0 tracked files
- `sleep2vec_hires/`: 0 tracked files
- `configs/`: 103 tracked files
- `tests/`: 98 tracked files
- `utils/`: 8 tracked files
- `agent_tools/`: 18 tracked files
- `skills/`: 37 tracked files
- `recipes/`: 25 tracked files
- `agent_policies/` and `doc/agent_contracts/`: 9 tracked files

## Coverage Boundaries

- Indexed in detail: config loaders and task semantics, runtime entrypoints, checkpoint helpers, result and prediction writing, inference W&B artifact logging, adaptation orchestration, backbone/downstream contracts, Cox survival finetuning, LSTM temporal aggregation, dataset/sampler/survival-sidecar contracts, Kaldi data-backend routing, preprocessing CLIs, standalone data utilities, `sleep2stat` analysis bundles, agent tooling and consultation gates, example config validation, standalone `sleep2vec2`/`sleep2expert` variant contracts, `sleep2expert` MoE routing/export/subnetwork surfaces, downstream evaluation metrics/visualizations, and the tests that pin those contracts.
- Indexed at module or workflow level only: `preprocess/preprocess_pipeline.ipynb`, package-local variant preprocessing notebooks, and tracked visualization font binaries under `*/visualization/assets/fonts/`.
- Outside this index scope: tracked example data scaffolding under `egs/`.
- Not indexed as source of truth: `__pycache__/`, `.DS_Store`, ignored local artifacts, and untracked experiment folders such as `index/` and `new_index/`.
- `AGENTS.md` is referenced for ownership context but is not reproduced here as an editable source of truth.

## How To Use This Index

- For a small local fix, start here, then jump to one relevant section instead of doing the full handbook pass.
- For a broader behavior or contract change, follow the full reading order above.

- If you are changing YAML semantics, built-in task behavior, or config validation, start with [FUNCTIONS/CONFIG_AND_REGISTRIES.md](./FUNCTIONS/CONFIG_AND_REGISTRIES.md).
- If you are changing pretrain, adapt, finetune, inference, checkpoint, result-export, or prediction-export orchestration, start with [FUNCTIONS/RUNTIME_ORCHESTRATION.md](./FUNCTIONS/RUNTIME_ORCHESTRATION.md) and the relevant workflow.
- If you are changing backbone forward behavior, adaptation freeze policy, downstream heads, Cox survival reduction, AHI epoch reduction, LSTM temporal aggregation, or layer mix, start with [FUNCTIONS/MODELS_AND_HEADS.md](./FUNCTIONS/MODELS_AND_HEADS.md).
- If you are changing dataset loading, survival sidecar attachment, built-in AHI sample validation, missing-channel behavior, or samplers, start with [FUNCTIONS/DATASETS_AND_SAMPLERS.md](./FUNCTIONS/DATASETS_AND_SAMPLERS.md).
- If you are changing CSV splitting, preset generation, survival preset sidecars, preset-build strictness, missing-mask statistics, WatchPAT conversion, Kaldi index repair, UKB annotation parsing, UKB demographic extraction, or case-control matching utilities, start with [FUNCTIONS/PREPROCESSING_AND_CONVERSION.md](./FUNCTIONS/PREPROCESSING_AND_CONVERSION.md).
- If you are changing evaluation plots, pair-accuracy logging, or diagnostics, start with [FUNCTIONS/VISUALIZATION_AND_DIAGNOSTICS.md](./FUNCTIONS/VISUALIZATION_AND_DIAGNOSTICS.md).
- If you are changing `sleep2stat` config parsing, records, analyzers, reducers, writers, plotting, or agent recipe routing, start with [FUNCTIONS/SLEEP2STAT.md](./FUNCTIONS/SLEEP2STAT.md) and [WORKFLOWS/SLEEP2STAT.md](./WORKFLOWS/SLEEP2STAT.md).
- If you are changing agent-facing recipes, consultation gates, command plans, hparam orchestration, experiment monitoring, or progress files, start with [FUNCTIONS/AGENT_TOOLING.md](./FUNCTIONS/AGENT_TOOLING.md) and [WORKFLOWS/AGENT_TOOLING.md](./WORKFLOWS/AGENT_TOOLING.md).
- If you think you need `sleep2vec2/` or `sleep2expert/`, read [FUNCTIONS/VARIANT_SURFACES.md](./FUNCTIONS/VARIANT_SURFACES.md) first. Both are active tracked standalone namespaces on this branch.

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
- [WORKFLOWS/VARIANTS_AND_ROUTING.md](./WORKFLOWS/VARIANTS_AND_ROUTING.md)
- [WORKFLOWS/AGENT_TOOLING.md](./WORKFLOWS/AGENT_TOOLING.md)
- [WORKFLOWS/SLEEP2STAT.md](./WORKFLOWS/SLEEP2STAT.md)

## Reliability Notes

- Every claim in this index is grounded in the current tracked code on `main`.
- If behavior is unclear from source, the index says `unknown` rather than inferring.
- Some runtime paths were only statically inspected due environment limits; those areas are explicitly marked in the relevant pages.
