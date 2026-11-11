# Skill: hyperparameter_tuning

## When to use

Use for managed `hparam_tune` planning, launch/monitor/select/evaluate, and for
choosing a further round's points from completed results. Selection uses the
validation split; the test split is never a selection split.

On takeover, first read [Takeover and continue execution](../../doc/agent_contracts/experiment_workspace.md#takeover-and-continue-execution).
It owns evidence-to-action decisions, rolling runtime sync and short heartbeat
maintenance. Do not infer lifecycle from history.

## Required inputs

Use the authorized experiment/step, base recipe, search configurations or grid,
budget, selection metric/mode/split, test/final-evaluation policy and execution
identity. Read the relevant detailed owners before preparing work:

- [Search space](../../doc/agent_contracts/hparam_workflow.md#search-space) for grids, exact joint points and further rounds.
- [Test-access policy](../../doc/agent_contracts/external_test_locking.md#selection-and-test-access-policy) for selection split, test-after-fit and unlock requirements.
- [Launch and queue](../../doc/agent_contracts/hparam_workflow.md#launch-and-queue) for runtime identity, direct/Slurm resources and capacity.

## Interaction stages

Keep concept planning, publication and launch as the separate stages that the
[stop-and-consult policy](../../AGENTS.md#agent-stop-and-consult-policy) defines.

## First information-gathering commands

`python -m agent_tools doctor --recipe <recipe>` evaluates consultation and
diagnostics; [read its report and exit status](../../doc/agent_contracts/task_recipe.md#consultation-and-diagnostics),
not the existence of a questions directory. `experiment-status --run-dir <root>`
reads recorded state; `context` is diagnostic-only. Monitor commands refresh
and **write** observations, although they never launch pending work.

`plan` creates managed state; it is not merely an information-gathering command.
Use it only at the appropriate preparation stage below.

## Decision checklist

Selection split, metric and mode must come from explicit user decisions, not
agent inference recorded as `explicit_recipe`. Confirm the scientific/data,
checkpoint, test-access and runtime scope against the authorized recipe, then
let consultation validate the detailed field and resource rules.

Collect unresolved scientific decisions once. Reuse the same authorized
`decisions.yaml` for later doctor checks and fresh plan publication; resolved
plan artifacts carry those choices into launch and recovery. Ask only for new
unresolved decisions introduced by validation, input drift or an expanded
scope. The [decision contract](../../doc/agent_contracts/user_decisions.md) owns
doctor-directory reuse; the decision file does not authorize a later stage.

The [stop-and-consult policy](../../AGENTS.md#agent-stop-and-consult-policy) owns
the static-plan rule for a tuning request, the protection of frozen plans, and
the default search budget in
[`hparam_search_defaults`](../../agent_policies/consultation_policy.yaml).
Explain at the outset that completed results will choose later rounds; do not
spend the whole budget in the first plan by default.
A concurrency cap does not fix epochs. Author the points or grid and
`search.max_runs` in the recipe; they are not automatic parser defaults.

Before the first plan, inspect the effective base/runtime config and available
prior experiments. Choose the searched axes and first-round points using:

- The strongest comparable result, missing evidence, and whether its best point
  sits at the edge of the searched values or the final checkpoint. Edge evidence
  motivates a hypothesis; it does not prove that searching further will improve
  results.
- The important technical axes to search or hold fixed, with an evidence or
  budget reason for each fixed choice. Do not freeze a setting merely because it
  appears in the source config. Include only parameters consumed by the task and
  variant; coupled settings must produce valid complete candidate configs.
- What the points can distinguish within the compute authority. Declare either
  explicit `search.configurations` (complete, unique joint points) or a
  `search.parameters` grid expanded by Cartesian product, never both, and fit the
  run count within `search.max_runs`.
- For coupled YAML settings, declare complete mapping or list values under a
  `yaml:/...` key, such as the whole LayerMix or adaptation block. Each value
  replaces that block completely; a point selects an authorized block rather
  than editing arbitrary fields inside it. Use the source variant's valid config
  semantics, including pretrained-backbone requirements for frozen adaptation.
  Runtime values remain scalar.

Before freezing, consider plausible opening outcomes: continued improvement at
the edge of the searched values, or deterioration away from an earlier best point.
Explain which useful follow-up comparisons the first round enables and which it
deliberately leaves out. More axes or wider ranges are not inherently better; allocate room to hypotheses
that merit the remaining fits. Distinguish a measured cost limit from a provisional
compute judgment when throughput is unknown. Use the
[initial-search example](examples/initial_search_domain.md) for this reasoning;
it does not prescribe numerical ranges.

The templates are starting examples to adjust to the actual base config, runtime
and evidence. Do not modify frozen plans to accommodate a newly noticed search
limitation.

## Result-to-proposal reasoning

After a round is terminal and its required results are complete, read the
canonical results and the configs, manifests, diagnostics and logs they cite.
Compare every completed round of the step with the incumbent and revisit earlier
rationale and `RESEARCH_LOG.md`; the previous round's winner is not automatically
the best result. Inspect `stop_reason` when present.

Use available W&B history to compare the logged training loss and validation
trajectory, not just each run's best score. Sparse or sampled points do not
locate the actual training endpoint. `_step` is a W&B log index; only an
explicit `trainer/global_step` is trainer-step evidence. Learning-rate extrema
are not schedule order. Do not infer an early-stopping cause. Missing optional
history does not itself block a round; explain what can and cannot be concluded
from the available evidence.

Before planning the next round, append an `experiment-note` that cites the prior
results it relies on and gives this reasoning:

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
   merit the remaining runs. Stay within the total budget; if an important
   alternative would exceed it or change an authorized decision, report the concrete limitation instead
   of silently changing the search. On the next result, check this expectation
   and append meaningful observations or revised interpretations with
   `experiment-note`, rather than copying unchanged monitoring history.

Use [the worked result-to-proposal example](examples/result_to_proposal.md) for the
level of reasoning expected. Plan validation establishes identity, completeness
and allowed values; a well-written note does not by itself establish scientific
quality. Report only the best observed candidate, with uncertainty clearly
identified.

## Stop-and-consult gates

Stop new execution for required `NEEDS_USER_INPUT`, an uncovered high-impact
choice, or an identity/result-evidence blocker. Fill emitted `decisions.yaml`
only from authorized choices and retry through the [decision contract](../../doc/agent_contracts/user_decisions.md).
Do not re-ask resolved decisions or stop monitoring already-active work just
because a later launch is blocked. Uncertain submission does not authorize
resubmission, manual manifest repair or unauthenticated cancellation.

## Canonical commands

For rolling latest-main maintenance, keep one existing checkout. Run
`python -m agent_tools runtime-sync --workdir <checkout>` as the
non-mutating check; add `--execute` only when authorized to fast-forward
`origin/main`. Never clone or reset for a heartbeat. Sync and launch share a
short critical-section lock, not a whole-job lock: an older process can continue
after its spawn boundary records A while the checkout advances to B, and a later
process records B. That SHA is point-in-time provenance, not a guarantee that
checkout code bytes remain fixed for the whole job.

Each round follows doctor → `plan` with a fresh `--output-dir` → `hparam-launch`
dry-run → authorized `hparam-run-queue --execute` → terminal monitoring. For a
direct plan, `--execute` starts one detached queue runner that survives an SSH
drop and returns its PID and `queue_runner.log`; `hparam-monitor` and
`experiment-monitor` report whether it is alive and never start one. Read that
log before rerunning `--execute` after a runner has died. After
the last round, `hparam-select` → report/finalization. Use variant-local runtime
commands generated by the planner, not hand-written training scripts. Stop via
`hparam-stop --run-id <id> --reason <text>` under the existing
[direct/Slurm evidence contract](../../doc/agent_contracts/run_manifest.md).

A further round keeps the experiment and step, and never rewrites an earlier
plan or snapshot. It may record a newer runtime commit; treat mixed
commits as recorded provenance, not an experimental arm.

## Expected artifacts

Use the [workspace layout](../../doc/agent_contracts/experiment_workspace.md) and
[canonical run evidence](../../doc/agent_contracts/run_manifest.md) as the
artifact owners. Hparam execution snapshots are frozen during registration
preflight, not first execute. Output existence alone is not completion evidence.

## Validation gates

Read [registration preflight](../../doc/agent_contracts/hparam_workflow.md#registration-preflight)
and [launch revalidation](../../doc/agent_contracts/hparam_workflow.md#execution-snapshot-and-launch-revalidation).
All candidate sources validate final config bytes through the canonical variant
owner. Plan validation does not certify every cross-axis combination before the
config loader sees it. Choose jointly valid points and correct a rejected plan
with a fresh plan. Planner-local config and target CLI checks are distinct evidence, not
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

After terminal runs, follow [selection and selected-candidate consumers](../../doc/agent_contracts/hparam_workflow.md#selection-and-selected-candidate-consumers)
and [finalization](../../doc/agent_contracts/experiment_workspace.md#finalization).
Use final external evaluation only under the explicit unlock. Report the best
observed candidate among the searched configurations, metric, split and budget,
not a global optimum.

For skill maintenance, use the [fixed reasoning cases and review rubric](evaluations/rubric.md)
to inspect whether changed guidance affects decisions; these are concept-planning
cases, not an experiment launcher or a numerical-answer test.

## Common failure modes

For missing decisions, slow checks, submission uncertainty, incomplete
checkpoint tests or frozen-artifact drift, use the
[contract question router](../../doc/agent_contracts/README.md).
Recreate incompatible plans only through the existing contracts and authority;
do not patch frozen artifacts in place.

## Relevant owners and index pages

Owners: `agent-tooling-maintainer`, `runtime-orchestrator`, `regression-guard`.
Index: [agent workflow](../../doc/code_index/WORKFLOWS.md#agent-planning-and-managed-experiments).
