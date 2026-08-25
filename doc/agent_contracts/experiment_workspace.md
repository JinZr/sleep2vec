# Experiment Workspace Contract

An experiment workspace is the durable, human-readable record for related preparation, training, evaluation, and analysis steps. Heavy datasets, checkpoints, W&B files, and trainer logs remain in their runtime locations; the workspace stores frozen snapshots, indexes, events, and reports.

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
│   ├── experiment_ranking.csv
│   └── final.md
├── pipelines/<pipeline-id>/  # optional managed external-evaluation state
│   ├── spec.source.yaml
│   ├── spec.resolved.yaml
│   ├── pipeline.json
│   ├── checkpoints.json
│   ├── preflight.json
│   ├── jobs.tsv
│   ├── execution_snapshot.json  # single-variant initial scheduler
│   ├── initial_schedulers/<variant>/execution_snapshot.json  # multi-variant initial schedulers
│   ├── results.csv
│   ├── metrics.csv
│   ├── summary.md
│   ├── final.md
│   ├── recipes/<job-id>/attempt-NNN.yaml
│   ├── plans/<job-id>/attempt-NNN/
│   ├── preflight_retries/<job-id>/attempt-NNN.json
│   ├── retry_schedulers/<job-id>/execution_snapshot.json
│   └── results/<job-id>/attempt-NNN/
├── steps/<step.id>/step.yaml
└── <plan directory>/
    ├── recipe.resolved.yaml
    ├── plan.json
    ├── run_all.sh
    ├── execution_snapshot.json  # created by the first verified execute
    └── runs/run-000--<semantic-name>/
        ├── run.json
        ├── config.yaml
        ├── launch.sh
        └── artifacts.json
```

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

`experiment_manifest.tsv` is optional for plan-created workspaces. When present, it contains exactly one row whose experiment id and root match `experiment.yaml`.

Local recipe roots are based at the repository root; local experiment CLI roots
are based at the caller's current working directory. Both are expanded and
resolved once. Local repository-owned management locators are persisted as
absolute paths, including recipe, plan, run, config, script, artifact, report,
runtime/checkpoint, adaptive, and event paths.

SSH roots and locators remain exact remote strings. User-authored semantic data
and checkpoint paths are not normalized by this management rule.

A new plan must be contained by its experiment root and registered in its step manifest. A non-empty unmanaged root is rejected rather than adopted, and a completed experiment cannot accept another plan. Historical workspaces are not migrated or renamed.

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

Hparam plan publication separates physical materialization from canonical
registration. When staging is used, every frozen path and command names the
final plan directory while all bytes are written to a hidden sibling on the
same filesystem. `plan.json` is written only after the rest of the frozen
bundle, the complete directory is published by one rename, and only then are
the step manifest and all planned `run_manifest.tsv` rows committed.
Plan publication and status validation share the adapter-owned deterministic
plan contract for run identities, paths, derived configs, complete executable
scripts, search combinations, and required final-evaluation snapshots;
synchronized edits to a manifest and its artifacts cannot redefine the frozen
recipe. Resolved recipes retain the source-config snapshot and hparam recipes
also retain the explicit final-evaluation config snapshot when applicable.

Adaptive round 000 adds one final commit boundary: its registry is validated
and its README is written after canonical plan registration, then
`adaptive/workflow.json` is created atomically as the readiness marker. Launch, queue, and monitor reject
an adaptive round while that marker is absent. A retry may accept a complete
unregistered round only when deterministic regeneration produces an identical
plan tree; incomplete, partial-canonical, or differing visible rounds remain
invalid and are not repaired in place.

## Lifecycle entrypoints

- `plan` freezes the effective recipe, configs, commands, hashes, and planned runs.
- `hparam-launch` validates frozen artifacts and explicitly starts one eligible wave; dry-run remains the default.
- `hparam-run-queue` is the explicit long-running action that repeatedly fills available capacity until every current-plan run is terminal; dry-run performs one preview and returns.
- `hparam-monitor` observes registered runs and never schedules pending work. By default it rereads the canonical manifest and observes the frozen current plan every `--poll-seconds` (60 seconds) until all of its runs are terminal; `--once` performs exactly one observation round.
- `hparam-stop` requires a reason. Direct runs verify and stop the complete
  process group before committing terminal state. Slurm runs atomically record
  nonterminal `stopping`, request time, reason, and job binding before
  `scancel`; an interrupted request is retriable only with the same reason, and
  `stopped` is committed only after the matching job is observed as
  `CANCELLED`.
- `hparam-select` writes a step-scoped ranking using the metric, mode, and split frozen in the hparam recipe. Registered plans aggregated for one step must match all three fields. Runtime manifests, physical checkpoint inventories, and hashes are read from each run's frozen local or SSH execution target; unavailable SSH evidence for a canonically successful run fails the whole selection before ranking output. For test-selected tuning it first writes an immutable plan-local checkpoint-level audit ranking, then projects the best checkpoint per run into the existing workspace ranking and canonical one-row-per-run manifest. Later compatible plans may extend the workspace ranking without rewriting or invalidating the earlier audit; downstream postprocessing requires and revalidates the frozen checkpoint hash before writing outputs.
- `hparam-adaptive-*` appends rounds and commits replacements through the canonical owner.
- `experiment-note` atomically appends one evidence-backed research-log entry and never changes lifecycle state.
- `experiment-run` is the explicit, resumable external-evaluation launcher. Dry-run starts nothing; execute waits for successful source plans, freezes checkpoints selected by each source plan's registered ranking, and manages the declared job matrix.
- `experiment-status` strictly validates the experiment owner, every registered
  step and frozen plan control bundle, and the canonical `run_manifest.tsv`,
  then prints a deterministic read-only snapshot. It never reads projections
  as lifecycle evidence, refreshes runtime observations, or writes workspace
  state. Frozen recipe structure is validated by the same dictionary-only
  `decision_rules` owner used by planning; status does not rerun consultation,
  policy decisions, config loading, or external input/path probes. Layered
  recipes validate both source layers plus any effective-only overlay produced
  after their canonical merge. The frozen config bytes may be parsed by a pure
  adapter hook solely to reproduce the planner's commands; no source config or
  other external input is reopened. The frozen plan context, not the status
  reader's repository root or Python interpreter, reproduces relative source
  paths and complete launch scripts across creator and controller hosts.
  Suggested commands are advisory argv arrays and
  do not authorize a launch or mutation.
  A registered directory containing exactly `questions.json`, `questions.md`,
  `plan.blocked.md`, and optional `plan.draft.json` is a non-runnable planning
  outcome and is skipped; missing, extra, or aliased entries fail closed. A
  plan binds to the registered step's core `id`, `phase`, and `purpose`, while
  manifest-owned `inputs` and `outputs` remain valid step metadata.
  Status classifies launch advice and controller-deferred blockers only from
  the step manifest's frozen `plan_controller`; the frozen recipe and canonical
  pipeline identity must agree with that owner and cannot replace it.
  Active adaptive and pipeline plans produce plan-scoped blockers for their
  controller-owned advance/finalize actions without blocking an unrelated
  ordinary plan launch. Because the status read-set contains no controller
  completion proof, completed experiment metadata with any adaptive or
  pipeline plan fails as corrupt canonical control state. A registered step
  with no materialized plan and no canonical rows likewise blocks finalization;
  completed metadata cannot prove that controller-deferred work was completed.
  A canonical `stopped` row without a non-empty `stop_reason` blocks the
  finalize advisory, and completed metadata containing such a row is corrupt.
  `experiment-finalize` independently enforces the same stop-reason boundary.
  Valid blockers return success; corrupt canonical control state or local/SSH
  read failure returns non-zero.
- `experiment-rank` writes experiment-wide ranking.
- `experiment-finalize` requires no active runs and a non-empty final report.

Single-round and continuous `hparam-monitor`, plus `experiment-monitor`, remain non-launching even when a
pipeline has pending jobs. Pipeline locking, frozen state, attempt isolation,
and finalization sequencing belong to
[experiment_pipeline.md](experiment_pipeline.md).

## Mutation and runtime identity

Every mutation other than fresh initialization requires a parseable,
root-matching workspace owner. Managed output targets are preflighted before
mutation: existing targets must be independent regular files under valid
directory ancestry. Local and SSH uncertainty fails closed.

Managed hparam planning freezes `execution.python` and
`execution.runtime_commit`. Only the canonical manager runtime—a local target
at `REPO_ROOT` without a conda wrapper—may omit them; planning then freezes the
current manager interpreter and manager repository HEAD. SSH targets, separate
local workdirs, and conda-wrapped targets must provide both values explicitly.

Lifecycle-owned `infer` / `evaluate` plans may declare an all-or-none local
`execution.workdir`, `execution.python`, and `execution.runtime_commit`
identity. `experiment-run` requires that identity for every attempt. Its
generated scripts verify the frozen commit before committing `running`, then
use the frozen Python for inference and every lifecycle commit. The managed
scheduler's execution snapshot and pre-start probe remain the authoritative
launch checks.

## Execution snapshot and capacity

The first `hparam-launch --execute` or `hparam-run-queue --execute` with an
eligible slot probes the exact Python executable through the configured target,
workdir, conda wrapper, and explicit environment. It requires:

- the planned Git commit and no tracked worktree changes;
- no untracked or ignored importable Python or extension-module code;
- target-reported host identity;
- a runtime module whose resolved origin is inside the verified repository;
- successful `argparse` validation of every frozen argument vector from that
  origin.

Untracked experiment artifacts and data remain allowed. The snapshot stores the
module origin, normalized supported options and their digests, every validated
argv vector, and the explicit execution environment. Rendered CLI text is not
snapshot evidence. The evidence is atomically written to
`execution_snapshot.json`; every later eligible launch wave re-probes and
requires exact equality.

Immediately before each process-group start, the same
target/env/conda/PYTHONPATH wrapper rechecks Python/version, commit, repository
root, hostname, module origin, untracked or ignored importable code, and the
selected run's frozen script/config hashes. Target and leaf `PYTHONPATH`
contain only `execution.workdir`, so another manager checkout cannot satisfy
missing imports.

For Slurm, the first execute freezes the snapshot's raw SHA-256 in every
canonical run and binds it to each queued job as a batch-script argument.
`job.sbatch` carries the plan-level snapshot path into the allocation wrapper,
which verifies those exact bytes before parsing the snapshot. The compute node
must then match the frozen Python executable and version before it may start the
leaf script. Its hostname remains allocation evidence and is not required to
equal the submission host.

Dry-run and monitor never probe or create the snapshot. A plan without frozen
Python/commit identity must be recreated. A missing snapshot may be established
only while every plan run remains `planned` or `pending` and has no committed
execution target. Once execution identity or later state exists, the plan must
be recreated instead of upgraded in place. Removed `trial_*` plans and status
files remain unmanaged and read-only.

Before calculating capacity, execute-mode launch refreshes observable active
blockers from other plans sharing the relevant target, host, and GPU pool, then
commits their status transitions. The full queue fails explicitly if a
current-plan run or relevant cross-plan capacity blocker is `missing_pid`; a
queue with no eligible slot does not probe the execution snapshot. External
datasets, drivers, and environment outside explicit `execution.env` remain
operational dependencies rather than snapshot contents.

Run identity, status vocabulary, reducer precedence, PID/evidence behavior, atomic commit, and projection sequencing belong to [run_manifest.md](run_manifest.md).
