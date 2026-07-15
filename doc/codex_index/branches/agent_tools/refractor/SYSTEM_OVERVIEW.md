# System Overview

## Repository Role

The repository's model, data, preprocessing, variant, and `sleep2stat` packages remain the runtime owners. `agent_tools` is an orchestration layer for agents: it inspects inputs, enforces stop-and-consult decisions, freezes commands and artifacts, and manages experiment evidence. It must call existing runtime entrypoints rather than create a second training or analysis runtime.

## Agent Tooling Flow

```text
authored recipe + optional user decisions
        |
        v
recipes.load_recipe_with_base
        |
        v
plans.evaluate_recipe
  -> configs / index / preset summaries
  -> decisions.evaluate_consultation_gates
        |
        v
plans.preflight_plan                (read-only)
  -> workspace containment
  -> input/config freeze checks
  -> final-test and output topology guards
        |
        v
plans.build_plan                    (mutation begins only after safe preflight)
  -> experiment_workspace
  -> plan_rendering / plan_hparam
  -> canonical runtime/preprocess/sleep2stat commands
        |
        v
hparam_runtime / experiments / adaptive_hparam
  -> run_artifacts + run_evidence
  -> experiment_workspace.merge_run_manifest
```

## Contract Layers

1. **Authored input:** recipes, base recipes, user-decision YAML, policy YAML, and runtime config YAML.
2. **Consultation:** `DecisionReport` carries `PASS`, `WARN`, `NEEDS_USER_INPUT`, or `FAIL`; exit codes are derived from blocking issues.
3. **Preflight:** `plans.preflight_plan` performs all checks that must precede workspace creation or run mutation.
4. **Frozen plan:** a successful plan records resolved recipes, configs, scripts, hashes, stable run ids, semantic names, and absolute management locations.
5. **Canonical state:** `run_manifest.tsv` is the durable lifecycle authority. Auxiliary status, metrics, checkpoint, W&B, ranking, and adaptive tables are evidence or projections.
6. **Runtime evidence:** only the exact frozen runtime directory, plan, script, PID, checkpoint directory, and runtime manifest are eligible evidence sources.

## Mutation Boundary

`evaluate_recipe` and `preflight_plan` are the reusable read-only boundaries. `build_plan` may initialize a workspace only after preflight has established enough ownership and safety. Issues marked as pre-workspace, unresolved experiment/step metadata, invalid workdirs, and managed-output topology failures must return before creating directories or files.

`doctor --output-dir` is diagnostic and may write question artifacts. `context` writes a diagnostic bundle but does not authorize runnable work. `plan`, hparam launch/queue, experiment lifecycle commands, and adaptive execution are mutation surfaces.

## Workspace and Identity

- Local experiment roots are canonicalized once by `canonical_local_experiment_root`; repository-owned locators are absolute.
- Remote roots are passed exactly as absolute paths and handled by `experiment_io` over SSH.
- `experiment.id`, `step.id`, and `run_id` form managed identity. Stable run ids use `run-NNN`; semantic parameter-derived names remain human-readable.
- `experiment.yaml`, per-step `step.yaml`, and `run_manifest.tsv` bind ownership.
- `merge_run_manifest` is the only canonical state commit. Monitoring and external tracking produce observations, then commit through this owner.

## Task Routing

- `preset_prepare` renders the existing preprocessing preset CLI.
- `finetune`, `infer`, and `evaluate` route through the package selected by `variant`.
- `hparam_tune` freezes a finite set of per-run configs and launch scripts before any launch.
- Planning freezes the target Python command and expected Git commit. Only a local `REPO_ROOT` target without a conda wrapper may use manager-interpreter/HEAD defaults; SSH, separate-workdir, and conda-wrapped targets require explicit identity. The first eligible hparam execute verifies isolated imports, repository-owned module origin, host identity, clean worktree, exact commit, the normalized `argparse` option set, and every frozen CLI argv. Each actual start repeats identity/import checks through the same wrapper and verifies the run's frozen script/config hashes immediately before `nohup`. Plans that predate frozen identity must be recreated. `hparam-run-queue` composes monitor observations with the locked launch owner, which also refreshes relevant cross-plan capacity blockers and fails queue mode on `missing_pid` blockers.
- `sleep2stat` is variantless and routes through `python -m sleep2stat` commands.
- Supported variant modules are selected in `models.module_for_variant` and `plan_rendering.variant_module`.

## External Boundaries

- W&B access lives in experiment tracking and remains external evidence, not canonical state.
- SSH operations are exact, timeout-bounded where used for consultation/progress, and isolated in path, I/O, progress, or runtime-evidence owners.
- Torch/Lightning are intentionally absent from lightweight context/config summaries.
- Dataset, checkpoint, and config semantics remain owned by their runtime packages; agent tooling may summarize or gate them but should not reimplement them.

## Test Strategy

`tests/agent_tools/` pins CLI dispatch, recipes, consultation, plan atomicity, hparam artifacts and lifecycle, experiment workspace ownership, remote/local I/O, adaptive rounds, sleep2stat routing, skills, and user decisions. Contract changes should extend the smallest owner-focused tests and then run the full agent-tools suite.
