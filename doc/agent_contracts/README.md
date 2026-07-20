# Agent Contracts

Each contract has one normative owner:

| Contract | Owner |
| --- | --- |
| Recipe workflow, effective recipe, and task/variant routing | [task_recipe.md](task_recipe.md) |
| Accepted recipe fields and finite allowlists | [task recipe schema](../../recipes/schemas/task_recipe.schema.md) |
| Explicit decision format, materialization, and precedence | [user_decisions.md](user_decisions.md) |
| Workspace layout, experiment/step ownership, and lifecycle entrypoints | [experiment_workspace.md](experiment_workspace.md) |
| Managed run identity, canonical state, reducer, commit, projections, and evidence | [run_manifest.md](run_manifest.md) |
| Diagnostic context bundles | [context_bundle.md](context_bundle.md) |
| Final and external-test gates | [external_test_locking.md](external_test_locking.md) |
| Resumable validation-to-external-test orchestration | [experiment_pipeline.md](experiment_pipeline.md) |

New runnable plans must follow the recipe and workspace contracts. Run-state consumers must follow the run-manifest contract rather than recovering state from derived artifacts. Multi-job external evaluation must additionally follow the pipeline contract.
