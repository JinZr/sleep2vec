# Codex Engineering Index

This directory is the shared, branch-independent navigation layer for the repository. It helps an engineer find the owner of a behavior, the implementation to reuse, the runtime path that exercises it, and the tests that define its contract.

It is deliberately small. It is not a generated inventory, a branch snapshot, or a substitute for reading code.

## Sources Of Truth

When documents disagree, use this order:

1. Current source code and tests on the checked-out branch.
2. [`AGENTS.md`](../../AGENTS.md), task-specific contracts, schemas, and config documentation.
3. This navigation index.
4. Historical index content available through Git history.

The index describes stable ownership and intended reuse. Always verify behavior in the live source before editing or reporting a contract.

## Reading Guide

| Question | Start here |
| --- | --- |
| Which subsystem owns this behavior? | [`MODULE_MAP.md`](./MODULE_MAP.md) |
| Which implementation should I reuse? | [`REUSE_GUIDE.md`](./REUSE_GUIDE.md) |
| How does a command or artifact flow through the repo? | [`WORKFLOWS.md`](./WORKFLOWS.md) |
| What are the current coding and ownership rules? | [`AGENTS.md`](../../AGENTS.md) |

Small, localized fixes may go directly to the relevant source and focused tests. Consult this index when a change crosses modules, adds reusable behavior, changes a public contract or workflow, or has unclear ownership.

## Stable Coverage

The navigation covers these long-lived areas:

- root model configuration, data loading, model composition, training, inference, and result export;
- preprocessing, preset generation, and NPZ/Kaldi data backends;
- standalone `sleep2vec2` and `sleep2expert` variants;
- `sleep2stat` derived-analysis bundles;
- `agent_tools`, recipes, skills, consultation gates, and managed experiments;
- the tests that pin those contracts.

Notebooks, one-off experiment directories, generated artifacts, caches, and untracked data are outside the index. Repository utilities are listed only when they are canonical entrypoints for a durable workflow.

## Maintenance Contract

The complete index is exactly these four files:

- `README.md`
- `MODULE_MAP.md`
- `REUSE_GUIDE.md`
- `WORKFLOWS.md`

Update the shared documents only when one of these changes:

- subsystem ownership or a dependency boundary;
- a canonical implementation that other code should reuse;
- a public data, config, runtime, artifact, or agent-facing contract;
- a key workflow or top-level entrypoint.

Do not update the index for:

- a local bug fix that preserves the documented contract;
- internal refactoring that leaves ownership and workflow unchanged;
- routine test additions, formatting, or dependency bumps;
- function signatures, caller lists, file counts, commit hashes, or refresh timestamps.

Do not create feature-branch copies, `DELTA_FROM_MAIN.md`, manifests, changelogs, function catalogs, archives, or redirect files. Git history is the archive.

## Editing Rules

- Keep descriptions at the responsibility and contract level; link to authoritative source or tests.
- Prefer one canonical owner over parallel implementations or compatibility aliases.
- Remove stale entries instead of preserving historical descriptions.
- Keep variant-specific behavior package-local and label shared behavior as parity, not inheritance.
- If ownership is uncertain after source inspection, leave it undocumented until the contract is clear.
