# Agent Tools/Refractor Branch Codex Engineering Index

This directory is the branch-scoped engineering manual for `agent_tools/refractor`. It was initialized from the checked-out source rather than copied from the `main` index.

## Branch Scope

- Branch: `agent_tools/refractor`
- Indexed commit: `faefcc7f3d46cf3739ff53dd89d8749ba9d93b9e`
- Generated at: `2026-07-15T05:03:08Z`
- Mode: `repair`
- Detailed scope: `agent_tools/` and its contracts in `recipes/`, `skills/`, `agent_policies/`, `doc/agent_contracts/`, and `tests/agent_tools/`
- Context-only scope: the model, preprocessing, variant, and `sleep2stat` entrypoints invoked by generated commands

## Recommended Reading Order

1. [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md) for the orchestration and mutation boundaries.
2. [MODULE_MAP.md](./MODULE_MAP.md) for the responsibility owners inside `agent_tools/`.
3. [REUSE_GUIDE.md](./REUSE_GUIDE.md) before introducing a helper, facade, schema layer, or artifact reader.
4. [WORKFLOWS/AGENT_TOOLING.md](./WORKFLOWS/AGENT_TOOLING.md) for end-to-end recipe, plan, hparam, and experiment flows.
5. [FUNCTIONS/AGENT_TOOLING.md](./FUNCTIONS/AGENT_TOOLING.md) for reusable function contracts.
6. [DELTA_FROM_MAIN.md](./DELTA_FROM_MAIN.md) for the branch comparison.

## Ownership Summary

- `recipes.py` loads authored YAML and base-recipe overlays and rejects authored internal metadata.
- `decision_*` modules own consultation, task-specific field closure, and decision rules; `decisions.py` aggregates task-aware runtime and decision-name closure.
- `plan_*` modules own context, rendering, reusable hparam execution-identity freezing, plan compilation, and preflight; `plans.py` owns top-level/artifact closure and source-layer dispatch without adding a schema facade.
- `experiment_workspace.py` owns canonical workspace identity and `run_manifest.tsv` state.
- `experiment_io.py` owns local/SSH reads, writes, and managed-output topology checks.
- `run_artifacts.py` and `run_evidence.py` own frozen-plan/runtime evidence validation.
- `hparam_*` modules own launch, cross-plan capacity refresh, explicit queue draining, verified execution snapshots and pre-start guards, observation, selection, and postprocessing; `hparam.py` only re-exports public operations.
- `experiment_tracking.py` builds observations and reports; `experiments.py` exposes public lifecycle commands.
- `adaptive_hparam.py` composes existing plan, hparam, and workspace owners, keeping round-000 Python/commit identity stable while later operational settings remain source-controlled.

## Coverage Boundaries

The user narrowed this initialization to agent tooling. Other tracked packages are counted in [MANIFEST.json](./MANIFEST.json) and described only where they are downstream command targets. For full model/data/runtime guidance, consult the current code and the `main` branch index, then verify every claim against this branch.

Generated files, ignored run directories, W&B state, remote hosts, and untracked experiments are not indexed as source of truth. `AGENTS.md` is an ownership policy input and was not copied or edited.

## Deliverables

- [MANIFEST.json](./MANIFEST.json)
- [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md)
- [MODULE_MAP.md](./MODULE_MAP.md)
- [REUSE_GUIDE.md](./REUSE_GUIDE.md)
- [CHANGELOG.md](./CHANGELOG.md)
- [DELTA_FROM_MAIN.md](./DELTA_FROM_MAIN.md)
- [FUNCTIONS/AGENT_TOOLING.md](./FUNCTIONS/AGENT_TOOLING.md)
- [WORKFLOWS/AGENT_TOOLING.md](./WORKFLOWS/AGENT_TOOLING.md)

## Reliability Notes

- The checked-out branch is the source of truth.
- Public and important internal symbols were checked against current source signatures.
- Static inspection establishes ownership and call flow; it does not prove access to external paths, SSH hosts, W&B, GPUs, or datasets.
- When a behavior is not established by current code, this index says `unknown` rather than inferring it.
