# Experiment Workspace Contract

An experiment workspace is the durable, human-readable record for related preparation, training, evaluation, and analysis steps. Heavy datasets, checkpoints, W&B files, and trainer logs remain in their runtime locations; the workspace stores frozen snapshots, indexes, events, and reports.

Read by question:

- [Ownership and paths](#ownership-and-paths)
- [Publication and registration](#publication-and-registration)
- [Takeover and continue execution](#takeover-and-continue-execution)
- [Lifecycle entrypoints](#lifecycle-entrypoints)
- [Read-only status and advisory actions](#read-only-status-and-advisory-actions)
- [Finalization](#finalization)
- [Research log](#research-log)
- [Mutation and runtime identity](#mutation-and-runtime-identity)

```text
<experiment.root>/
├── experiment.yaml
├── experiment_manifest.tsv  # optional experiment-CLI index
├── README.md
├── RESEARCH_LOG.md
├── events.jsonl
├── run_manifest.tsv
├── run_matrix.csv
├── reports/
│   ├── status.md
│   ├── ranking.csv
│   ├── hparam_selection.md
│   ├── experiment_ranking.csv
│   └── final.md
├── pipelines/<pipeline-id>/  # optional managed external-evaluation state
├── steps/<step.id>/step.yaml
└── <plan directory>/
    ├── recipe.resolved.yaml
    ├── plan.json
    ├── run_all.sh
    ├── execution_snapshot.json  # hparam registration preflight; task-specific elsewhere
    └── runs/run-000--<semantic-name>/
        ├── run.json
        ├── config.yaml
        ├── launch.sh
        └── artifacts.json
```

The pipeline owns its [internal artifact tree](experiment_pipeline.md#invocation-and-frozen-state).

## Ownership and paths

`experiment.yaml` records a stable id, title, objective, canonical root, and
explicit baseline. Experiment and step ids use lowercase letters, digits,
hyphens, and underscores. Step phase is one of `prepare`, `train`, `evaluate`,
or `analyze`. Every step file uses the shared
`{step, experiment_id, plan_controller, recipe_path, plans}` envelope.
`plan_controller` is the only owner of whether the step is `ordinary`,
`adaptive`, or `pipeline`. `experiment-register-step` initially records
`unassigned` with no recipe or plans; the first planner or pipeline freeze may
bind it to one concrete owner, and that binding cannot later change.

Existing experiment and step metadata are read through the workspace owner and
merged through their reducers. Missing files may be created only by their
designated first producer; blank, malformed, incomplete, or conflicting
metadata is never repaired by overwriting it.

`events.jsonl` is modified only through the canonical managed append owner. All
supported local and SSH writers execute on the workspace-owning host and share
the descriptor-anchored `.events.jsonl.cas.lock`; under that lock, the final
public root, parent, and prior-content check binds the namespace for the
immediately following atomic rename. Raw shell appends and concurrent lockless
workspace renames or alias changes are unsupported. An SSH transport failure
has an unknown commit outcome and must not be blindly retried.

Managed SSH I/O also requires a complete operation result, not just a zero
transport exit code: some endpoints hide failed child exit codes. Missing or
malformed write results leave the commit outcome uncertain and do not authorize
another write. A positively reported compare-and-swap conflict retains the
existing bounded fresh-read retry; it is not a transport-failure retry.

`experiment_manifest.tsv` is optional for plan-created workspaces. When present, it contains exactly one row whose experiment id and root match `experiment.yaml`.

Local recipe roots are based at the repository root; local experiment CLI roots
are based at the caller's current working directory. Both are expanded and
resolved once. Local repository-owned management locators are persisted as
absolute paths, including recipe, plan, run, config, script, artifact, report,
runtime/checkpoint, adaptive, and event paths.

SSH roots and locators remain exact remote strings. User-authored semantic data
and checkpoint paths are not normalized by this management rule.

A new plan must be contained by its experiment root and registered in its step manifest. A non-empty unmanaged root is rejected rather than adopted, and a completed experiment cannot accept another plan. Historical workspaces are not migrated or renamed.

## Publication and registration

Plan publication separates physical materialization from canonical registration.
When staging is used, frozen paths and commands name the final plan directory
while bytes are written to a hidden sibling on the same filesystem. `plan.json`
is written after the rest of the frozen bundle. A fresh destination is published
by directory rename; publication into an accepted preexisting destination moves
plan-owned entries and publishes `plan.json` last. Only after the published
bundle validates are the step manifest and planned canonical rows committed.

Publication and status share the adapter-owned deterministic plan contract for
run identities, paths, derived configs, complete executable scripts, search
combinations, and required final-evaluation snapshots. Synchronized edits to a
manifest and its artifacts cannot redefine the frozen recipe. Resolved recipes
retain the source-config snapshot and hparam recipes retain the explicit
final-evaluation config snapshot when applicable. Task-specific gates belong to
[hparam registration preflight](task_recipe.md#registration-preflight) and
[adaptive initialization readiness](task_recipe.md#initialization-readiness).

## Takeover and continue execution

Read current artifacts before acting. Research history explains **why**; the
canonical manifest establishes the recorded lifecycle, not the other way round.

| Question | Authoritative input | Use |
| --- | --- | --- |
| Which experiment and declared work? | `experiment.yaml`, registered `steps/<step.id>/step.yaml` | Confirm root, objective, step and controller ownership. |
| What is registered, active or terminal? | Canonical `run_manifest.tsv` | Read current lifecycle and frozen execution identity. |
| What will run, or already ran? | Registered frozen recipe, plan, configs, scripts and identity records | Verify exact execution content; do not reconstruct it from chat. |
| Are required results complete? | Task-required runtime/result manifests, metrics and bound audit evidence | Decide selection and finalization eligibility. |
| What decisions are authorized, and why? | Authorized recipe/decisions and their authorization record; `RESEARCH_LOG.md` when present | Recover scope and rationale, never substitute narrative for lifecycle evidence. |

The following phases are explanatory, not new state fields. Actions still use
the entrypoint-specific checks and the current authorization.

| Evidence | Conclusion and next action | Do not infer |
| --- | --- | --- |
| Prepared recipe, no registered managed plan | Complete consultation, then publish through the relevant planner/init owner. | Doctor PASS or an output directory proves registration, writability, submission or success. |
| Registered plan, no credible submission evidence | Perform that entrypoint's launch checks; launch only within existing authorization. | `planned` alone proves a historical manually wrapped job was never submitted. |
| Credible queued/running scheduler or process evidence | Continue monitoring the same frozen run. | Queue delay, missing output or SSH disconnection permits duplicate submission. |
| Training terminal, required checkpoint tests incomplete | Follow the task's remaining test phase or diagnose missing evidence. | Fit completion or a log message permits test ranking. |
| Current adaptive round terminal, required results complete, budget remains | Follow the [proposal handshake](task_recipe.md#proposal-handshake), including tool-issued input, exact proposal, preflight and authorized execute. | A generic continue instruction expands the envelope or bypasses an evidence blocker. |
| Declared work complete, no active runs, selection/report requirements met | Apply [finalization](#finalization), including controller-owned completion checks. | One completed phase closes every assigned obligation, or a result directory proves success. |

Distinguish normal waiting (queue, training, checkpoint testing) from diagnosis
(identity mismatch, unavailable state, missing required results) and a new user
decision (uncovered scientific choice or required `NEEDS_USER_INPUT`). Do not
re-ask already explicit label/split/metric/budget choices. A new launch blocker
does not cancel monitoring duties for existing work. Estimate end-to-end time
including queue delay, post-fit checkpoint tests, proposal/preflight and
controller work, not just training fit time.

An incidental question about status, GPU use or timing does not revoke unfinished
authorized work. Answer it, then continue the remaining obligations in the active
execution, or through the already authorized scheduled continuation when waiting
is required. Pause for an explicit stop/replacement instruction, a genuine blocker
or a decision outside the existing authority, not merely because the answer ended
one phase of the conversation.

Gate each dependent action on positive evidence that its producer succeeded.
A failed bundle creation must not be followed by its transfer, and an incomplete
runtime must not be used for initialization. Failure of one dependency does not
block independent, already authorized work. An outer shell/SSH exit 0, a partial
directory or a readable HEAD is not evidence of complete preparation.

Read doctor report and exit status: nonblocking WARN/exit 0 can be successful,
and `--output-dir` need not create a questions directory when no template is
needed. See [consultation and diagnostics](task_recipe.md#consultation-and-diagnostics).
If a check is slow or SSH disconnects, inspect the original operation handle,
phase logs and durable completion receipt against its frozen identity. Follow a
still-live operation rather than starting another attempt. Missing or lost
completion evidence remains uncertain even if a process disappears; do not retry
a side effect to obtain a cleaner receipt. Uncertain submission is not definite
failure; follow the [Slurm uncertainty rules](run_manifest.md#stopping-and-uncertain-states).
Managed ordinary infer/evaluate now uses the shared Slurm transaction; older
manual wrappers remain historical and are not repaired or adopted by this flow.

### Execution identity legend

| Identity | Meaning |
| --- | --- |
| Manager/controller host and Python | Where planning or a control command executes; status advice names its `control_host`. Planner-local final-config validation describes this runtime. |
| Frozen execution host/workdir/Python and planned commit | Target route whose owner-specific identity and CLI gates are verified for launch; the actual launch-boundary HEAD observation is recorded separately and is not a checkout pin. Diagnostic package metadata is not model-execution proof or a new environment policy. |
| Slurm allocation node | Where the scheduler actually runs the workload; compute hostname may differ from the submission host. |

See the [preflight evidence limits](task_recipe.md#registration-preflight) before
interpreting a card as target config validation, checkpoint compatibility or
forward/backward validation.

### Conditional runtime refresh before initialization

The default rolling-latest model uses one existing checkout. A heartbeat or
agent first inspects the trusted upstream without mutation:

```bash
python -m agent_tools runtime-sync --workdir <checkout> [--host <host>] [--python <target-python>]
```

With `--host`, the manager sends a self-contained sync program through the
selected target Python. The target checkout may therefore predate the
`runtime-sync` subcommand; it does not need to import its own `agent_tools`
before the fast-forward.

Add `--execute` only within the current update authority. Execute fetches
`origin/main` and updates the same clean checkout by fast-forward only. It never
clones another repository, resets local history, installs dependencies, rewrites
plans, or changes scientific configuration. A dirty tracked tree, untracked or
ignored importable code, malformed upstream identity, or non-fast-forward
history fails closed. An unchanged result needs no new runtime or research note.

Runtime sync and process launch share a short checkout lock. The lock covers
only the sync operation or the launch critical section that observes HEAD and
starts the child. For provenance-aware managed direct launch, embedded
verification, HEAD capture, and child `Popen` occur in that same critical
section; Slurm similarly keeps allocation preflight/HEAD, sidecar publication,
and `srun` `Popen` together. The lock is released for the child lifetime. Thus
a process whose spawn boundary observed commit A may continue while the
checkout fast-forwards to B, and a later process observes B. The canonical manifest
records both values. They are point-in-time provenance: code imported or read
after the lock is released is not guaranteed to remain entirely at A or B.
Mixed commits are not a scientific variable.

Local direct launch waits for the embedded launcher to return a definitive
pre-spawn failure or committed PID receipt; it does not apply the SSH transport
timeout while that launcher waits for the runtime lock. An SSH timeout remains
outcome-unknown evidence and is reconciled without automatic relaunch.

Commit drift does not authorize mutation of registered recipes, plans, configs,
scripts, or `execution_snapshot.json`. `execution.runtime_commit` remains the
planned/baseline commit, and a different launch-time HEAD is recorded and
warned rather than blocked. `runtime-sync` itself still requires a clean tree
without untracked or ignored importable code. Hparam, ordinary Slurm, and
pipeline managed-attempt launches retain their owner-specific managed-scheduler
gates. Managed direct launch checks stable Python/route/module origin, live argv,
clean code, and artifact hashes before starting. Commit drift remains allowed.
The Slurm allocation
recheck requires the current module to remain inside the current repository with
the same module name rather than matching the frozen origin path. Direct preset
launch retains its frozen Python, clean importable-code, lifecycle-module, and
script/config/input-hash gates, but does not claim workload module-origin or
live-argv checks. Preserve current monitoring obligations and
do not infer a retry from a lost receipt or uncertain submission.

### Fixed-commit runtime staging

Explicit fixed-commit staging remains available when a workflow specifically
requires a separate immutable checkout; it is not the rolling default. For such
an authorized replacement, use
[`utils/stage_git_runtime.py`](../../utils/stage_git_runtime.py) rather than a
chain of independent bundle, transfer and checkout commands. It prepares Git
content only: it does not select upstream commits, install dependencies, run
regressions or doctor, activate a runtime, or modify an experiment.

- `stage --source-repo PATH --commit FULL_SHA --destination PATH --evidence-dir PATH`
  requires fresh destination and evidence directories. For SSH, also provide
  `--host HOST --remote-python PATH --remote-attempt-dir PATH` explicitly.
  Bundle production and verification must succeed before transfer; the fixed
  checkout runs with detached standard streams and persistent evidence.
  Git subprocesses isolate user/system configuration, global attributes and
  hooks without changing the caller's environment. Worker handles and terminal
  receipts are published atomically without replacement, with directory fsync.
- A positive start is **not completion**: `stage` reports `started` with exit 2.
  Use `check --evidence-dir PATH` to inspect that same attempt without writing or
  restarting it. Only a matching successful terminal receipt together with the
  exact, clean, detached checkout permits `check` exit 0. The frozen worker hash
  is verified before executing the status check. Pending/unknown is exit 2;
  a verified failure or invalid evidence is nonzero. A pending `recorded_pid`
  locates the original worker; verify its current command and attempt before
  calling it live, and never treat that PID alone as cancellation authority.
- Keep the attempt and stage logs after failure or disconnection. Never rerun
  `stage` to obtain a missing receipt, reuse a destination, or turn a partial HEAD
  into success. A later successful check is independent completion evidence; it
  does not establish the return code of a previously disconnected SSH command.

Staging success does not waive the activation and consultation gates above.

### Short heartbeat maintenance

When maintenance is authorized, update the existing automation's short entry,
not a duplicate. This contract does not itself authorize an automation change.

| Keep in the entry | Link to artifacts instead |
| --- | --- |
| Stable scope, exact root/workflow entry paths, authorization locator and this takeover contract | Job lists, per-run state, metrics, checkpoint inventories and historical hashes |
| Current monitoring/continuation duties, predecessor gates, authorized update policy and valid scoped exceptions/expiry | Completed phases, resolved blocker transcripts, superseded pins and exhausted one-time repair/network permissions |
| Effective scientific, test-access, budget and runtime-provenance limits; meaningful-change reporting and whole-assignment stop conditions | Full schemas, the takeover table, adaptive handshake and unchanged polling history |

Before an authorized revision, verify entry paths, current phase, authority and
remaining obligations. Remove obsolete active blocker wording only after fresh
resolution evidence is recorded in the append-only audit; do not revive an
exhausted permission or remove a still-valid constraint merely because it is
old. A phase finishing does not justify pausing remaining duties. Record and
notify meaningful changes once, rather than keeping a diary in the prompt.

## Finalization

Completion closes the declared search domain, metric, split and budget, not
every plausible hyperparameter. Report the best observed candidate within
that frozen scope. A completed root cannot accept another plan; separately
authorized follow-on work uses a separately bound root and references prior
evidence without copying canonical lifecycle rows.

`experiment-finalize` requires registered materialized work, no active runs,
non-empty reasons for stopped rows, and a non-empty final report. For ordinary
hparam steps, `ready_to_select` advice after terminal success is not permission
to finalize. Complete [selection](task_recipe.md#selection-and-selected-candidate-consumers)
before interpreting a report as accepted final evidence. Pure
ordinary-hparam experiments may finalize directly from the verified selection
report only when every hparam step has a selected winner. Mixed experiments and
partly failed multi-step searches require a separate combined report, and
all-failed hparam experiments may close with a non-empty failure report. The
canonical selection report cannot substitute for a combined or failure report.

Status validates the frozen selection audit bytes and reconstructs step-wide
checkpoint/run ranks without reopening runtime manifests or checkpoint contents.
Finalization additionally rehashes every checkpoint named by the bound audits
on its canonical execution host, then holds the canonical run-manifest lock
while it verifies the same manifest snapshot and commits terminal metadata.

New finalization commits `status: completed` only with the canonical
`reports/final.md` path and SHA-256. When ordinary hparam selection exists, the
terminal metadata also freezes the verified selection-report SHA-256. Status
validates these bound bytes so report deletion, tampering, or a concurrent
selection commit becomes visible as corrupt terminal evidence. Historical
completed manifests without report bindings remain readable and are not
retroactively migrated. A selection report path alias or byte-identical copy
cannot substitute for the required mixed-experiment combined report or
all-failed hparam report; pure ordinary-hparam automatic finalization still
requires the canonical `reports/hparam_selection.md` path.

## Research log

`RESEARCH_LOG.md` is the append-only, human-readable record of meaningful
research actions, observations, interpretations, decisions, and conclusions.
It is chronological narrative, not a current-state summary and not a lifecycle
owner. Agents taking over an experiment read `experiment.yaml`, the research
log when present, and then the current canonical manifests before acting.

Append one entry with:

```bash
python -m agent_tools experiment-note \
  --run-dir <experiment.root> \
  --entry <entry.yaml> \
  [--remote <host>]
```

`--entry` must name an existing local YAML file, including when the managed
workspace is remote. Inline YAML/text and stdin (`-`) are not accepted. The CLI
validates this boundary before reading the experiment workspace.

The entry YAML is a closed mapping:

```yaml
id: observation-20260725-001
recorded_at: "2026-07-25T02:03:04Z"
occurred_at: "2026-07-25T01:58:00Z"  # optional
kind: observation  # action | observation | interpretation | decision | conclusion
title: Validation loss stopped improving
actor: agent:codex
source: codex-task:019f...
authority: human  # optional; required for decision
scope:             # optional
  step_id: train-model
  run_ids: [run-001]
evidence:
  - label: validation report
    locator: reports/validation.md
    sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
supersedes: []     # optional list of existing entry ids
body: |
  The validation curve plateaued after epoch 12.
```

Required fields are `id`, `recorded_at`, `kind`, `title`, `actor`, `source`,
non-empty `evidence`, and non-empty `body`. Timestamps must be UTC ISO
timestamps. `authority`, when present, is `human`, `policy`, or
`canonical_decision`; every `decision` requires it. Evidence locators are
preserved exactly. A scoped step and run must already belong to the workspace,
and `run_ids` requires `step_id`.

The caller supplies the entry id as its idempotency key. The owner normalizes
the entry and stores an entry id plus content digest in a hidden Markdown
marker. Repeating the same id and normalized content is a successful no-op;
reusing the id for different content fails. Corrections append a new entry and
list the old ids in `supersedes`; existing entries are never rewritten.
Malformed preambles, markers, digests, aliases, or lock targets fail before
append. Local and SSH writers use compare-and-swap and retry conflicts without
dropping a competing entry.

Fresh CLI- and plan-created workspaces receive the preamble. A valid historical
workspace without the file is not migrated by init, planning, or monitoring;
the first explicit `experiment-note` creates it. Completed experiments may
accept retrospective notes, but notes never authorize plans, launches,
external-test access, status changes, or finalization. Log write failure does
not rewrite or roll back canonical state. Polling without a meaningful new
fact should not create an entry.

## Lifecycle entrypoints

| Entrypoint | Effect and detailed owner |
| --- | --- |
| `doctor`, `context` | Consultation/diagnostics and diagnostic bundles, not execution authority; [diagnostic contract](task_recipe.md#consultation-and-diagnostics). |
| `runtime-sync` | Inspect `origin/main` by default; with `--execute`, fast-forward one clean existing checkout in place under the short runtime lock. It does not clone, reset, or launch work. |
| `plan` | Freeze and register recipe, config, commands, hashes and planned runs; [publication](#publication-and-registration). |
| `hparam-launch`, `hparam-run-queue` | Explicit launch/queue advancement; dry-run default; [launch and queue](task_recipe.md#launch-and-queue). CLI names preview/execute and projects recorded state counts. |
| `infer-launch`, `infer-stop` | Managed ordinary Slurm inference launch/stop; [ordinary inference](task_recipe.md#managed-ordinary-inference). |
| `preset-launch`, `preset-stop` | Execution-host local detached preset launch/stop; launch defaults to dry-run, stop requires a reason; [managed preset preparation](task_recipe.md#managed-preset-preparation). |
| `hparam-monitor`, `experiment-monitor` | Observe and write canonical evidence/projections, never launch pending work. |
| `hparam-stop` | Reasoned stop under authenticated identity rules; [prelaunch stop](task_recipe.md#launch-and-queue), [direct process evidence](run_manifest.md#pid-and-runtime-evidence), [Slurm stopping](run_manifest.md#stopping-and-uncertain-states). |
| `hparam-select` | Commit step-scoped ranking/selection; [selection and consumers](task_recipe.md#selection-and-selected-candidate-consumers). |
| `hparam-adaptive-*` | Tool-owned round publication and advancement; [adaptive workflow](task_recipe.md#adaptive-workflow). |
| `experiment-note` | Append an evidence-backed [research note](#research-log), no lifecycle change. |
| `experiment-run` | Explicit resumable [external matrix](experiment_pipeline.md); dry-run starts nothing. |
| `experiment-status` | Validate and display the [read-only snapshot](#read-only-status-and-advisory-actions). |
| `experiment-rank`, `experiment-finalize` | Write experiment ranking; commit completion only under [finalization](#finalization). |

By default `hparam-monitor` rereads canonical state and observes the frozen
current plan every `--poll-seconds` (60 seconds) until terminal; `--once` makes
one observation round. Its terminal CLI summary projects structured failure
evidence from `run_status.tsv`, not a second lifecycle interpretation or a new
log read. Raw recorded `log_tail` is printed only with `--include-log-tail`
because it may contain sensitive data. Neither continuous nor single-round
monitoring launches pending work, including pipeline jobs.

## Read-only status and advisory actions

`experiment-status` strictly validates the experiment owner, every registered
step and frozen plan control bundle, and canonical `run_manifest.tsv`, then
prints a deterministic snapshot. It may display already-recorded scheduler,
process, health, checkpoint and runtime-manifest fields but never refreshes
them. `run_status.tsv`, `launch_manifest.tsv`, events, logs, runtime manifests,
W&B and adaptive/pipeline controller state are not alternate lifecycle inputs.
Modern hparam selection reports, shared rankings and plan-local checkpoint
audits are read only as hash-bound consistency evidence for canonical selection
rows. Status does not read checkpoint contents or write workspace state.

Frozen recipe structure is validated by the same dictionary-only
`decision_rules` owner used by planning, without consultation, policy decisions,
config loading or external input/path probes. Layered recipes validate both
source layers plus their effective-only overlay. A pure adapter hook may parse
frozen config bytes solely to reproduce planner commands; source configs and
other external inputs are not reopened. Recipe input snapshots and the frozen
creator-host context reproduce run matrices, paths, derived configs, complete
scripts and final-evaluation requirements through the same `plan_contract`
owner as publication. Agreement between edited canonical rows and scripts is
insufficient if they differ from that recipe-derived contract.

Step output projects registered `plan_controller`; run output projects canonical
execution transport/host and recorded Slurm node separately. Advisory argv
arrays name `control_host` only as the control command's invocation host; see
the [identity legend](#execution-identity-legend). Suggestions do not authorize
launch or mutation.

A registered directory containing `questions.json`, `questions.md` and
`plan.blocked.md`, plus optional `decisions.yaml` and `plan.draft.json`, is a
non-runnable planning outcome and is skipped. Nested plan directories have an
exact envelope: missing required, extra or aliased entries fail closed. In a
workspace-root plan, plan-owned artifacts remain strict while canonical
workspace siblings are outside that envelope. Physical publication/validation
precedes step ownership. Historical blocked plans without `decisions.yaml`
remain readable. Plans bind the registered step's core `id`, `phase`, `purpose`;
manifest-owned `inputs` and `outputs` remain valid metadata.

Only step `plan_controller` classifies launch advice and controller-deferred
blockers; frozen recipes and canonical pipeline identity must agree with it.
Active adaptive/pipeline plans defer their own advance/finalize actions without
blocking unrelated ordinary launch advice. Status's read-set contains no
controller completion proof, so completed experiment metadata with any adaptive
or pipeline plan is corrupt under this read-set. A registered step with no
materialized plan and no canonical rows also blocks finalization; completed
metadata cannot prove the declared work was absent or finished.

These status blockers do not revoke controller finalize callbacks: an adaptive
or pipeline controller that validates its own frozen terminal state may call
`experiment-finalize`. The direct finalizer independently rejects unmaterialized
steps rather than treating status's incomplete controller read-set as proof.
A canonical `stopped` row without non-empty `stop_reason` blocks finalize advice
and makes completed metadata corrupt; the finalizer enforces the same boundary.
Valid blockers return success; corrupt control state or local/SSH read failure
returns non-zero.

## Mutation and runtime identity

Every mutation other than fresh initialization requires a parseable,
root-matching workspace owner. Managed output targets are preflighted before
mutation: existing targets must be independent regular files under valid
directory ancestry. Local and SSH uncertainty fails closed.

Ordinary output-path validation uses metadata without opening files. When
observed inode identities collide, validation holds descriptors for all paths
seen so far and checks their current identities together; this is not a
cross-file transaction snapshot. Linux uses `O_PATH` for the leaves and directory
walk, so unreadable files and search-only directories remain valid. Platforms
without `O_PATH`, including macOS, require read access for this collision check
and explicitly fail closed if it is denied. Validation never reads file contents,
changes permissions, or falls back to unpinned path observations.

Runtime identity and defaults belong to the
[non-hparam identity](task_recipe.md#non-hparam-runtime-identity) and
[hparam launch](task_recipe.md#launch-and-queue) contracts. The
[execution snapshot contract](task_recipe.md#execution-snapshot-and-launch-revalidation)
owns registration-time creation, live launch revalidation and the restricted
missing-snapshot boundary; this workspace layout does not authorize rebinding
historical plans.

Run identity, status vocabulary, reducer precedence, PID/evidence behavior, atomic commit, and projection sequencing belong to [run_manifest.md](run_manifest.md).
