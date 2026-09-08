# Skill: hyperparameter_tuning

## When to use

Use for managed `hparam_tune` planning, launch/monitor/select/evaluate, or
append-only adaptive tuning. Test-selected tuning is supported with explicit
test access; this skill does not select a scientific split for the user.

On takeover, first read [Takeover and continue execution](../../doc/agent_contracts/experiment_workspace.md#takeover-and-continue-execution).
It owns evidence-to-action decisions, rolling runtime sync and short heartbeat
maintenance. Do not infer lifecycle from history.

## Required inputs

Use the authorized experiment/step, base recipe, search domain or explicitly
requested static profile/grid, budget, selection metric/mode/split, test/final-evaluation policy and
execution identity. Read the relevant detailed owners before preparing work:

- [Search space](../../doc/agent_contracts/task_recipe.md#search-space) for the default adaptive workflow, frozen domains, initial joint points and explicitly requested static `finetune_balanced` searches.
- [Test-access policy](../../doc/agent_contracts/external_test_locking.md#selection-and-test-access-policy) for selection split, test-after-fit and unlock requirements.
- [Launch and queue](../../doc/agent_contracts/task_recipe.md#launch-and-queue) for local/SSH identity, direct/Slurm resources and capacity.
- [Adaptive workflow](../../doc/agent_contracts/task_recipe.md#adaptive-workflow) when enabled; it owns initialization, frozen Python/route/scientific identity, per-round commit provenance, strategy and budget.

## Interaction stages

Keep concept planning, publication, and launch distinct. Concept planning
discusses scientific choices, candidate scope and budget without running
`doctor` or `plan`, publishing recipe bytes, inspecting live resources or
producing runnable commands. Publication may write the authorized recipe or
decisions, run `doctor` and `plan`, and freeze artifacts, but it does not launch.
Launch requires an explicit request to execute or complete the run. One request
may authorize all three stages when it explicitly asks for design and execution;
otherwise stop at the requested stage.

## First information-gathering commands

`python -m agent_tools doctor --recipe <recipe>` evaluates consultation and
diagnostics; [read its report and exit status](../../doc/agent_contracts/task_recipe.md#consultation-and-diagnostics),
not the existence of a questions directory. `experiment-status --run-dir <root>`
reads recorded state; `context` is diagnostic-only. Monitor commands refresh
and **write** observations, although they never launch pending work.

`plan` and `hparam-adaptive-init` create managed state; they are not merely
information-gathering commands. Use them only at the appropriate preparation
stage below.

## Decision checklist

Selection split, metric and mode must come from explicit user decisions, not
agent inference recorded as `explicit_recipe`. Confirm the scientific/data,
checkpoint, test-access and runtime scope against the authorized recipe, then
let consultation validate the detailed field and resource rules.

Collect unresolved scientific decisions once. Reuse the same authorized
`decisions.yaml` for later doctor checks and fresh plan publication; resolved
plan artifacts carry those choices into launch and recovery. Ask only for new
unresolved decisions introduced by validation, input drift or an expanded
scope. If doctor exposes a new decision absent from its existing template, use a
fresh doctor output directory. Use a fresh directory as well when a currently
resolved concrete value differs from the preserved template; doctor neither
merges nor overwrites the old file. The decision file records intent but does
not itself authorize a later interaction stage.

For a new recipe with no authored search, an ordinary tuning request defaults
to terminal-only `adaptive.suggest.strategy: agent_proposal`. Static profile/grid
search requires an explicit request. Existing authored or frozen searches must
not be rewritten. Explain at the outset that the agent will use completed results
to choose later rounds; do not preplan the whole budget and call it adaptive.

Use the default 12-run search budget only when the user has not specified a total
budget. Default to `round_size: 2` and `max_rounds: 6`; a smaller concurrency or
total-run cap reduces round size to `min(2, permitted concurrent runs, total
budget)`, with `max_rounds = ceil(total budget / round_size)`. Translate GPU limits
using the authorized GPUs per run. A concurrency cap does not fix epochs. Author
these fields, the explicit objective and `replacement: {enabled: false}` in the
recipe; they are not automatic parser defaults.

Before initialization, inspect the effective base/runtime config and available
prior experiments. Choose a bounded domain and first-round points using:

- The strongest comparable result, missing evidence, and whether its best point
  touches a parameter bound or the final checkpoint. Boundary evidence motivates
  a hypothesis; it does not prove that expanding the bound will improve results.
- The important technical axes to search or hold fixed, with an evidence or
  budget reason for each fixed choice. Do not freeze a setting merely because it
  appears in the source config. Include only parameters consumed by the task and
  variant; coupled settings must produce valid complete candidate configs.
- What the initial points can distinguish within the compute authority. Prefer
  explicit `search.configurations` for the opening batch, alongside the complete
  frozen domain in `search.parameters`. Every point must cover every domain key;
  its values must lie within the envelopes, and the point count must fit all
  initial budgets. Without configurations, keep the full initial Cartesian product
  within round size. Numeric bounds and categorical choices can leave useful
  alternatives for later rounds without spending initial runs on them.
- For coupled YAML settings, declare complete mapping or list choices under a
  `yaml:/...` key, such as the whole LayerMix or adaptation block. Each choice
  replaces that value completely; a proposal selects an authorized block rather
  than editing arbitrary fields inside it. Use the source variant's valid config
  semantics, including pretrained-backbone requirements for frozen adaptation.
  Runtime values remain scalar. There is no automatic profile-to-adaptive compiler.

The templates are starting examples to adjust to the actual base config, runtime
and evidence. Choose technical values within the authorized domain without asking
for each learning rate, training length, scheduler, dropout or LoRA level. Unknown
scientific choices still require consultation. Budget/domain expansion, test
unlock, changed data/label/split/checkpoint, an existing protocol change or a later
interaction stage needs its own authority. Do not modify active frozen workflows
to accommodate a newly noticed search limitation.

## Result-to-proposal reasoning

After each round is terminal and required results are complete, read the issued
proposal input and its cited config, manifest, diagnostic and log evidence. Compare
all available completed rounds with the incumbent and revisit earlier rationale
and `RESEARCH_LOG.md`; the previous round's winner is not automatically the
workflow's best result. The issued rows identify the global incumbent and carry
accepted prior proposal rationale; inspect `checkpoint_test_results` trajectories,
`monitor_checkpoint_path` and `stop_reason` when present. Keep validation-monitor
and test-objective checkpoint identities distinct, and use test feedback only
under its frozen authorization. Use the exact evidence identities and submission format
in the [proposal handshake](../../doc/agent_contracts/task_recipe.md#proposal-handshake).

Use `training_history.observations` when present to compare the logged training
loss and validation trajectory, not just each run's best score. Its source path
and hash identify the already-synced W&B history; an empty field means these
observations are unavailable. Sparse or sampled points do not locate the actual
training endpoint. `_step` is a W&B log index; only an explicit
`trainer/global_step` is trainer-step evidence. `learning_rate_ranges` reports
observed extrema, not schedule order. Read the cited history for a needed detail
within the frozen split policy; do not infer an early-stopping cause or silently
sync new evidence while applying an already-issued proposal.

Write a concise, useful `rationale` using the existing free-text field:

1. Separate observations from explanations. Identify the compared configurations,
   metric differences, training/checkpoint evidence and failure causes. A scalar
   score or last-checkpoint winner alone does not establish underfitting,
   overfitting or an unstable optimizer. Read available trajectories when needed;
   state missing evidence instead of inventing curve shape or noise estimates.
2. State the leading explanation and plausible alternatives. Joint changes support
   a joint strategy, not the isolated effect of one parameter. Small differences
   without repeat evidence remain uncertain. Infrastructure failure is not a low
   scientific score; distinguish it from a supported infeasible configuration.
3. Explain each complete candidate point: what it changes, why that could improve
   the objective or distinguish explanations, and which observed runs support it.
   Balance improvement near the incumbent with useful exploration according to
   evidence and remaining budget; neither a fixed quota nor automatic shrinking
   around the latest winner is required. Prefer `configurations` for intentional
   joint points. Avoid repeats unless they answer an explicit reproducibility or
   unresolved-failure question within the existing contract and authority.
4. Say what result would change the current interpretation and why these points
   merit the remaining runs. Stay within the issued domain and budget; if an
   important alternative is outside them, report the concrete limitation instead
   of silently changing the workflow. On the next result, check this expectation
   and append meaningful observations or revised interpretations with
   `experiment-note`, rather than copying unchanged monitoring history.

Use [the worked result-to-proposal example](examples/result_to_proposal.md) for the
level of reasoning expected. Tool validation establishes identity, completeness
and allowed values; a non-empty rationale does not by itself establish scientific
quality. Report only the best observed candidate, with uncertainty and any
explicit test-feedback selection clearly identified.

## Stop-and-consult gates

Stop new execution for required `NEEDS_USER_INPUT`, an uncovered high-impact
choice, or an identity/result-evidence blocker. Fill emitted `decisions.yaml`
only from authorized choices and retry through the [decision contract](../../doc/agent_contracts/user_decisions.md).
Do not re-ask resolved decisions or stop monitoring already-active work just
because a later launch is blocked. Uncertain submission does not authorize
resubmission, manual manifest repair or unauthenticated cancellation.

## Canonical commands

For rolling latest-main maintenance, keep one existing checkout. Run
`python -m agent_tools runtime-sync --workdir <checkout> [--host <host>]` as the
non-mutating check; add `--execute` only when authorized to fast-forward
`origin/main`. Never clone or reset for a heartbeat. Sync and launch share a
short critical-section lock, not a whole-job lock: an older process can continue
after its spawn boundary records A while the checkout advances to B, and a later
process records B. That SHA is point-in-time provenance, not a guarantee that
checkout code bytes remain fixed for the whole job.

For explicitly static tuning, follow doctor → `plan` → `hparam-launch` dry-run →
authorized `hparam-run-queue --execute` → terminal monitoring → `hparam-select`
→ report/finalization. Once launch is explicitly authorized, continue the chosen
workflow without another question about technical levels or each execute step.
Use variant-local runtime commands generated by the planner, not hand-written
training scripts. Stop via `hparam-stop --run-id <id> --reason <text>` under the
existing [direct/Slurm evidence contract](../../doc/agent_contracts/run_manifest.md).

For the default adaptive workflow, complete doctor, `hparam-adaptive-init`,
initial launch dry-run and the authorized initial launch, then terminal monitoring
and the next-round proposals. Adaptive recipes do not enter through generic `plan`.
Follow the exact [proposal handshake](../../doc/agent_contracts/task_recipe.md#proposal-handshake):
the tool issues the input, the external agent writes only its named submission,
and the tool preflights/registers/launches. `hparam-adaptive-loop` is only for
explicit `best_neighborhood`, not an LLM driver for `agent_proposal`.
If an execute receipt is lost, a heartbeat may repeat only the exact same
proposal execute. The command returns the existing suggestion when canonical
committed evidence includes successful execute completion; incomplete,
unresolved, or conflicting state does not authorize a new launch. Monitors
remain non-launching. An older execute is not a successful replay while any
later committed proposal lacks ordered completion evidence or retains a
canonical launch failure.
Later rounds may use a newer commit while Python, route, objective, and the
scientific contract remain frozen. Treat mixed commits as recorded provenance,
not an experimental arm, and never rewrite earlier plan or snapshot bytes.

## Expected artifacts

Use the [workspace layout](../../doc/agent_contracts/experiment_workspace.md),
[canonical run evidence](../../doc/agent_contracts/run_manifest.md), and
[adaptive readiness](../../doc/agent_contracts/task_recipe.md#initialization-readiness)
as the artifact owners. Hparam execution snapshots are frozen during
registration preflight, not first execute. Output existence alone is not
completion evidence.

## Validation gates

Read [registration preflight](../../doc/agent_contracts/task_recipe.md#registration-preflight)
and [launch revalidation](../../doc/agent_contracts/task_recipe.md#execution-snapshot-and-launch-revalidation).
All candidate sources validate final config bytes through the canonical variant
owner. The frozen domain permits values; it does not certify every cross-axis
combination. Choose jointly valid points and correct rejected, unaccepted
submissions within the issued domain. Planner-local config and target CLI checks are distinct evidence, not
proof of model construction, checkpoint compatibility, forward/backward or GPU
execution. `execution.runtime_commit` is planned/baseline provenance; launch
first-fills the canonical actual `runtime_commit` from the HEAD observed between
embedded verification and direct child `Popen` in the same short lock, or from
the Slurm allocation's locked preflight/HEAD → sidecar → `srun` sequence. The
lock does not cover child lifetime. A mismatch warns without blocking. The
hparam managed-scheduler clean/importable-code, module-origin, Python/route,
frozen-argv, and artifact-hash checks remain fail-closed. In a Slurm allocation, the module-origin recheck
means the current module remains inside the current repository with the same
module name; it does not require the exact frozen origin path. Dry-run does not
replace the live eligible-execute checks.

After terminal runs, follow [selection and selected-candidate consumers](../../doc/agent_contracts/task_recipe.md#selection-and-selected-candidate-consumers)
and [finalization](../../doc/agent_contracts/experiment_workspace.md#finalization).
Use final external evaluation only under the explicit unlock. Report the best
observed candidate within the frozen domain, metric, split and budget, not a
global optimum.

For skill maintenance, use the [fixed reasoning cases and review rubric](evaluations/rubric.md)
to inspect whether changed guidance affects decisions; these are concept-planning
cases, not an experiment launcher or a numerical-answer test.

## Common failure modes

For missing decisions, slow checks, submission uncertainty, incomplete
checkpoint tests, frozen-artifact drift or proposal rejection, use the
[contract question router](../../doc/agent_contracts/README.md).
Recreate incompatible plans or request fresh proposal inputs only through the
existing contracts and authority; do not patch frozen artifacts in place.

## Relevant owners and index pages

Owners: `agent-tooling-maintainer`, `runtime-orchestrator`, `regression-guard`.
Index: [agent workflow](../../doc/codex_index/WORKFLOWS.md#agent-planning-and-managed-experiments).
