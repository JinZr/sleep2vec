# Agent Contracts

Use the question router before reading a whole contract. The engineering
[index](../codex_index/README.md) locates code owners; these contracts define
operation and evidence boundaries.

## Find the next action

| Question | Read |
| --- | --- |
| Taking over an experiment: what is current and what may I do? | [Takeover and continue execution](experiment_workspace.md#takeover-and-continue-execution) |
| Doctor is slow, WARN, or created no questions directory; what does that mean? | [Consultation and diagnostics](task_recipe.md#consultation-and-diagnostics) |
| Which Python does the preflight card describe? | [Identity legend](experiment_workspace.md#execution-identity-legend), then [preflight evidence](task_recipe.md#registration-preflight) |
| How do I resolve `NEEDS_USER_INPUT`? | [Decision materialization and retry](user_decisions.md) |
| Doctor passed; why can plan still fail? | [Registration preflight](task_recipe.md#registration-preflight) and [publication](experiment_workspace.md#publication-and-registration) |
| Which search sources and technical defaults are supported? | [Search space](task_recipe.md#search-space) |
| Is test-selected tuning allowed? | [Selection and test-access policy](external_test_locking.md#selection-and-test-access-policy) |
| How do I launch or queue a frozen plan? | [Hparam launch and queue](task_recipe.md#launch-and-queue), or [ordinary inference](task_recipe.md#managed-ordinary-inference) |
| When is the execution snapshot frozen and rechecked? | [Execution snapshot and launch revalidation](task_recipe.md#execution-snapshot-and-launch-revalidation) |
| What establishes Slurm job/cluster identity? | [Submission and routing](run_manifest.md#submission-and-routing) |
| Did SSH loss mean no submission, or may I stop/retry? | [Stopping and uncertain states](run_manifest.md#stopping-and-uncertain-states) |
| Can a purged job finish when accounting is disabled? | [Terminal evidence](run_manifest.md#terminal-evidence) |
| Read recorded status or refresh observations? | [Read-only status](experiment_workspace.md#read-only-status-and-advisory-actions) versus [entrypoint effects](experiment_workspace.md#lifecycle-entrypoints) |
| Why is selection/finalization blocked? | [Selection and consumers](task_recipe.md#selection-and-selected-candidate-consumers), then [finalization](experiment_workspace.md#finalization) |
| How does the next adaptive round proceed; where may the proposer write? | [Adaptive workflow](task_recipe.md#adaptive-workflow) and [proposal handshake](task_recipe.md#proposal-handshake) |
| How do I run a resumable external matrix? | [Pipeline invocation and frozen state](experiment_pipeline.md#invocation-and-frozen-state) |
| Where do meaningful observations and decisions go? | [Research log](experiment_workspace.md#research-log) |
| What runtime-update or heartbeat setup may be carried forward? | [Conditional runtime refresh](experiment_workspace.md#conditional-runtime-refresh-before-initialization), [short heartbeat maintenance](experiment_workspace.md#short-heartbeat-maintenance) |

## Contract owners

Each detailed rule has one normative owner; linked summaries do not create a
second lifecycle or authorization source.

| Contract | Owner |
| --- | --- |
| Recipe workflow, effective recipe, and task/variant routing | [task_recipe.md](task_recipe.md) |
| Accepted recipe fields and finite allowlists | [task recipe schema](../../recipes/schemas/task_recipe.schema.md) |
| Explicit decision format, materialization, and precedence | [user_decisions.md](user_decisions.md) |
| Workspace ownership, publication, takeover, status, finalization and research log | [experiment_workspace.md](experiment_workspace.md) |
| Managed run identity, canonical state, reducer, commit, projections, and evidence | [run_manifest.md](run_manifest.md) |
| Diagnostic context bundles | [context_bundle.md](context_bundle.md) |
| Final and external-test gates | [external_test_locking.md](external_test_locking.md) |
| Resumable external evaluation over registered-ranking-selected checkpoints | [experiment_pipeline.md](experiment_pipeline.md) |

New runnable plans must follow the recipe and workspace contracts. Run-state consumers must follow the run-manifest contract rather than recovering state from derived artifacts. Multi-job external evaluation must additionally follow the pipeline contract.
