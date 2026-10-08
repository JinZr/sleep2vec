# Agent Contracts

Use the question router before reading a whole contract. The engineering
[index](../code_index/README.md) locates code owners; these contracts define
operation and evidence boundaries.

## Quickstart: the first five commands

Run each through `python -m agent_tools`; every subcommand documents itself with `--help`.

| Step | Command | What it does and does not authorize |
| --- | --- | --- |
| 1 | `skills --list` | Lists the task playbooks. Read the matching `skills/<task>/SKILL.md` first. |
| 2 | `experiment-status --run-dir <experiment-root>` | On takeover, prints recorded state read-only. Then follow [takeover](experiment_workspace.md#takeover-and-continue-execution). |
| 3 | `doctor --recipe <recipe> --output-dir <dir>` | Consultation and diagnostics. Exit 0 continues, 1 fails, 2 needs [user decisions](user_decisions.md). It publishes no runnable commands. |
| 4 | `plan --recipe <recipe> --output-dir <plan-dir> [--user-decisions <decisions.yaml>]` | Publishes a frozen plan in a fresh directory inside `experiment.root`. Adaptive recipes use `hparam-adaptive-init` instead. It does not launch. |
| 5 | `hparam-launch --plan-dir <plan-dir>` | Dry run by default (`infer-launch` and `preset-launch` likewise). Add `--execute` only with explicit launch authorization. |

Monitors (`experiment-monitor`, `hparam-monitor`) refresh observations and never launch. The
[stop-and-consult policy](../../AGENTS.md#agent-stop-and-consult-policy) separates concept
planning, publication (steps 3–4) and launch (step 5).

An expected refusal (validation, malformed YAML, a missing artifact, a remote, subprocess or
scheduler failure) ends the command with exit 1 and one `error: <message>` line as the last stderr
line, also under `--json`, so stdout carries only results; embedded line breaks are folded into
` | `. Progress lines a command already printed to stderr, such as `doctor`'s phases, may precede
it. Exit 2 means
only `NEEDS_USER_INPUT` from a consultation command, or an argparse usage error (or no subcommand).
Other exceptions are bugs and keep their traceback.

## Find the next action

| Question | Read |
| --- | --- |
| Taking over an experiment: what is current and what may I do? | [Takeover and continue execution](experiment_workspace.md#takeover-and-continue-execution) |
| Doctor is slow, WARN, or created no questions directory; what does that mean? | [Consultation and diagnostics](task_recipe.md#consultation-and-diagnostics) |
| Which Python does the preflight card describe? | [Identity legend](experiment_workspace.md#execution-identity-legend), then [preflight evidence](hparam_workflow.md#registration-preflight) |
| How do I resolve `NEEDS_USER_INPUT`? | [Decision materialization and retry](user_decisions.md) |
| Doctor passed; why can plan still fail? | [Registration preflight](hparam_workflow.md#registration-preflight) and [publication](experiment_workspace.md#publication-and-registration) |
| Which search sources and technical defaults are supported? | [Search space](hparam_workflow.md#search-space) |
| Is test-selected tuning allowed? | [Selection and test-access policy](external_test_locking.md#selection-and-test-access-policy) |
| How do I launch or queue a frozen plan? | [Hparam launch and queue](hparam_workflow.md#launch-and-queue), [ordinary inference](task_recipe.md#managed-ordinary-inference), or [managed preset preparation](task_recipe.md#managed-preset-preparation) |
| When is the execution snapshot frozen and rechecked? | [Execution snapshot and launch revalidation](hparam_workflow.md#execution-snapshot-and-launch-revalidation) |
| Which `.tsv` owns lifecycle state, and what are the others? | [Table index](run_manifest.md#table-index) |
| What establishes Slurm job/cluster identity? | [Submission and routing](run_manifest.md#submission-and-routing) |
| Did SSH loss mean no submission, or may I stop/retry? | [Stopping and uncertain states](run_manifest.md#stopping-and-uncertain-states) |
| Can a purged job finish when accounting is disabled? | [Terminal evidence](run_manifest.md#terminal-evidence) |
| Read recorded status or refresh observations? | [Read-only status](experiment_workspace.md#read-only-status-and-advisory-actions) versus [entrypoint effects](experiment_workspace.md#lifecycle-entrypoints) |
| Why is selection/finalization blocked? | [Selection and consumers](hparam_workflow.md#selection-and-selected-candidate-consumers), then [finalization](experiment_workspace.md#finalization) |
| How does the next adaptive round proceed; where may the proposer write? | [Adaptive workflow](hparam_workflow.md#adaptive-workflow) and [proposal handshake](hparam_workflow.md#proposal-handshake) |
| How do I run a resumable managed evaluation pipeline? | [Pipeline invocation and frozen state](experiment_pipeline.md#invocation-and-frozen-state), then [cohort selection](experiment_pipeline.md#cohort-selection-and-report-only-boundary) when internal cohorts gate candidates. |
| Where do meaningful observations and decisions go? | [Research log](experiment_workspace.md#research-log) |
| What runtime-update or heartbeat setup may be carried forward? | [Conditional runtime refresh](experiment_workspace.md#conditional-runtime-refresh-before-initialization), [short heartbeat maintenance](experiment_workspace.md#short-heartbeat-maintenance) |

## Contract owners

Each detailed rule has one normative owner; linked summaries do not create a
second lifecycle or authorization source.

| Contract | Owner |
| --- | --- |
| Recipe workflow, effective recipe, and task/variant routing | [task_recipe.md](task_recipe.md) |
| Ordinary and adaptive hparam search, launch, selection and rounds | [hparam_workflow.md](hparam_workflow.md) |
| Accepted recipe fields and finite allowlists | [task recipe schema](../../recipes/schemas/task_recipe.schema.md) |
| Explicit decision format, materialization, and precedence | [user_decisions.md](user_decisions.md) |
| Workspace ownership, publication, takeover, status, finalization and research log | [experiment_workspace.md](experiment_workspace.md) |
| Managed run identity, canonical state, reducer, commit, projections, and evidence | [run_manifest.md](run_manifest.md) |
| Diagnostic context bundles | [context_bundle.md](context_bundle.md) |
| Final and external-test gates | [external_test_locking.md](external_test_locking.md) |
| Resumable managed evaluation over registered-ranking winners or internally gated candidates | [experiment_pipeline.md](experiment_pipeline.md) |

New runnable plans must follow the recipe and workspace contracts. Run-state consumers must follow the run-manifest contract rather than recovering state from derived artifacts. Multi-job managed evaluation must additionally follow the pipeline contract.
