# Run Manifest Contract

`run_manifest.tsv` is the only mutable owner of managed run lifecycle state and execution identity.

This contract owns identity, state reduction, evidence, and atomic commits.
Command choice, status advice, and finalization belong to
[the workspace contract](experiment_workspace.md); hparam selection belongs to
[the hparam workflow contract](hparam_workflow.md).

## Contents

- [Identity and frozen fields](#identity-and-frozen-fields)
- [Canonical state and projections](#canonical-state-and-projections), its [table index](#table-index), and [runtime artifact evidence](#runtime-artifact-evidence)
- [Status reducer](#status-reducer), [evidence ownership](#evidence-ownership), and [atomic commit](#atomic-commit)
- [PID and runtime evidence](#pid-and-runtime-evidence)
- [Slurm evidence](#slurm-scheduler-evidence): [frozen job script](#frozen-job-script),
  [submission and adoption](#submission-and-adoption), [monitor state table](#monitor-state-table),
  and [stopping and lost runs](#stopping-and-lost-runs)
- [Consumer requirements](#consumer-requirements)

## Identity and frozen fields

The canonical managed key is `(step_id, run_id)`. Both fields are required and one canonical table contains at most one row per key. A run uses the next stable step-local `run-NNN` id. `run_name` is human-readable, and `version` is the bounded slug of experiment id, step id, run id, and run name. Version may resolve external evidence only when a complete managed key is absent and the match is unique.

Hyperparameter run names and parameter summaries use lexicographically sorted
full parameter keys, independent of mapping insertion order. Sorting happens
before shortened field names are assigned. Nested mapping values are likewise
order-independent, while list values retain their authored order. Frozen names,
versions, paths, and hashes are not rewritten to migrate historical plans.

Plan-owned identity, semantic parameters, config/script hashes, artifact paths,
runtime/checkpoint directories, and execution identity are frozen after
registration. `planned_runtime_commit` is the plan's full lowercase
40-character baseline Git commit ID; `runtime_commit` is the corresponding
full lowercase commit ID observed under the short runtime lock
immediately before a provenance-aware direct child is spawned. A Slurm job
writes `git rev-parse HEAD` to `run_dir/runtime_commit` at allocation start and
the monitor copies it. A script-owned direct route without the
new outer receipt observes it at its own `running` boundary. Each is a trusted
first fill and immutable once non-empty. The observed SHA is point-in-time
provenance, not a guarantee that checkout code bytes remain unchanged for the
whole job.
They may differ: that difference is recorded rolling provenance, not a lifecycle
error or scientific variable. Shared execution identity otherwise consists of
target (always `local` once launched), workdir, GPUs, log path, and command. Direct execution additionally
owns the PID path, launched PID, process-group id, and OS process-start token.
Slurm execution instead freezes scheduler type and the sbatch path and
SHA-256 in the plan; the numeric scheduler job id is a trusted one-time binding.
Slurm observation adds `scheduler_raw_state`, `scheduler_reason`,
`scheduler_node`, and `scheduler_exit_code`. PID and scheduler
bindings are mutually exclusive. Only the canonical owner may perform each
trusted first fill.

Managed rows may freeze `terminal_status_owner`. This field does not change
the canonical `(step_id, run_id)` key. Older managed rows may omit it and retain
their existing lifecycle behavior; a present value is immutable and must match
all later evidence.

When present, `terminal_status_owner` is exactly `script` or `monitor`. It
selects the existing process-exit rule explicitly: script-owned runs must commit
their own terminal status, while monitor-owned runs leave confirmed-exit
inference to the monitor. Slurm runs are monitor-owned and follow the
[Slurm state table](#monitor-state-table).
Lifecycle-owned inference with explicit runtime identity uses the same frozen
runtime Python for its workload and every lifecycle commit. Its start commit
records the planned and actual commits; a difference does not bypass any other
launch gate.

Managed tables declare either one row per run or many rows per run. Both forms require complete managed identity and reject removed `trial_id` or `param.*` formats. Historical formats remain read-only and are never translated into current state.

## Canonical state and projections

`launch_manifest.tsv` and `run_status.tsv` retain their plan-local paths and fields but are written only from rows returned by a successful canonical commit. They are projections and are never read to restore lifecycle status or execution identity. Matrices, status reports, rankings, and events are also derived artifacts. `RESEARCH_LOG.md` is an append-only narrative record rather than a projection, but it likewise never owns or restores lifecycle state.

`run_matrix.csv` and `reports/run_matrix.md` project both commit fields. The
Markdown projection labels a mismatch as `different (rolling update)`; that is
the provenance warning and has no lifecycle effect.

Health labels are observational and never own lifecycle state. Managed GPU
activity is attributed to the frozen process group so DDP child processes count
as active. Without another positive progress signal, an unavailable checkpoint
probe, an unavailable GPU probe for a GPU-assigned run, or the first checkpoint
observation without a comparison baseline reports `health_unknown`;
`possibly_stalled` requires a later, comparable observation with no detected
progress. Remote artifact uncertainty preserves the last checkpoint inventory
while leaving the current health poll's `checkpoint_count` blank.

Local log evidence reads only the needed suffix of seekable UTF-8 regular logs,
expanding for long lines; other text encodings and stream inputs retain their
text-read behavior. Encountered read errors still propagate. A tail is not a
whole-file integrity check or an atomic snapshot of concurrent log writes.
Health keeps its probe order; no log evidence is cached across observations.

### Table index

Each managed `.tsv` has one role. Only `run_manifest.tsv` owns
lifecycle state; every other table is a registration record, a projection, or
observed evidence, and none of them restores run status.

| File | Location | Written by | Lifecycle role |
| --- | --- | --- | --- |
| `run_manifest.tsv` | `<experiment.root>/` | Canonical [atomic commit](#atomic-commit) from plan registration, launch, monitor and stop | Only mutable lifecycle and execution-identity owner; one row per run |
| `launch_manifest.tsv`, `run_status.tsv` | Plan directory | Rewritten from the rows of each successful canonical commit | Projections; never read to restore state |
| `experiment_manifest.tsv` | `<experiment.root>/` | `experiment-init` | Optional one-row index that must match `experiment.yaml` |
| `trial_status.tsv` | Historical trees | None | Removed format; managed commands refuse a tree that contains it |

## Runtime artifact evidence

Runtime `run_manifest.json` supplies metrics and checkpoint evidence only. It does not own lifecycle status. A truly missing runtime manifest means evidence is not yet available; an existing alias, non-regular file, invalid encoding/JSON, or non-mapping payload is corrupt.

Finetune's `multilabel_per_disease_metrics.csv` and
`survival_per_disease_metrics.csv` retain each evaluated checkpoint's rows,
identified by `ckpt_path`. The optional `--export-predictions` flag (disabled by
default) also writes each checkpoint's predictions to the run's `predictions.csv`,
recorded as `prediction_csv_path` in the terminal runtime manifest. Predictions
are written after each test to avoid retaining all checkpoints' predictions in
memory; partial files from failed or interrupted runs are not complete evidence.
Consumers of a selected model must filter these tables by the exact checkpoint
path in the selection result, not by row order or the validation-best summary.
Missing rows must not fall back to another checkpoint. Prediction output that
was not enabled is unavailable, and historical missing outputs require separate
authorized evaluation rather than inference from aggregate scores.

Finetune runtime directories are single-use: a single-process launch rejects a
non-empty `log-finetune/<version>` before persisting configuration, loading
data, or fitting. Before distributed initialization, rank zero exclusively
creates a `.distributed-preflight` launch marker only after observing an empty
directory; other ranks require its complete matching launch token and reject
anything outside the current startup files or an empty checkpoint directory.
The marker stays in the run directory so late ranks reject stale launches.
All-checkpoint result rows form one evidence matrix; runtimes evaluate
the complete declared checkpoint set and write every required run-local
prediction or per-disease artifact before appending the full matrix to the
aggregate results CSV under one lock and one atomic replacement. The successful
terminal manifest is written only after that matrix commit. A failed checkpoint
test or required artifact writer may publish its existing `status=failed`
manifest, but never a successful terminal manifest or a partial checkpoint
matrix. Atomic rewrites read historical cells as text so lexical identities
such as zero-padded experiment versions remain unchanged.

## Status reducer

The current vocabulary includes scheduled `planned`/`pending`, scheduler handoff `queued`, active `launched`/`running`/`stopping`/`lost`/`missing_pid`, and terminal `completed`/`failed`/`finished`/`launch_failed`/`stopped`/`superseded` states.

- An update without status preserves the existing status.
- Terminal status is sticky, except incoming `failed` evidence may correct `completed` or `finished`.
- Active `lost` is not sticky: later Slurm exit evidence replaces it.
- Recorded stop request, reason, and stop time remain unchanged through stale
  observations of a `stopping` or `stopped` run.
- Active status cannot regress through stale `planned` or `pending` evidence.
- `superseded` commits only when the freshly read canonical state is still `planned` or `pending`.
- Monitoring preserves finished-to-completed normalization for evidence whose script does not own terminal commits.
- Monitoring preserves a managed-script `running` state when neither PID nor W&B execution evidence exists; absence of those evidence sources is not process-exit evidence.
- A `script` path requires strict managed process-identity checks but does not by itself assign terminal-status ownership.
- A lifecycle-enabled generated script owns its terminal commit. Confirmed disappearance of its process group without a canonical `completed` or `failed` commit is `failed`, never inferred success. New hparam launch scripts are explicitly monitor-owned and append a structured shell exit code to their log: only code `0` becomes `finished`; nonzero, missing, or malformed exit evidence becomes `failed`. Historical owner-less hparam plans retain the legacy failure-marker inference.

All lifecycle callers reuse the same row reducer. They do not implement source-specific precedence.

A run canceled while still `planned` or `pending` may be `stopped` without
execution identity. This shape requires no populated execution fields, scheduler
job binding, launch time, or stop request. Slurm's frozen plan `log_path` does
not establish launch. Slurm `queued`, `running`, `stopping`, `completed`,
`finished`, and `failed` states require a scheduler job id. `lost` and `stopped`
may lack one: a job that left `squeue` before its id was recorded is `lost`,
and a reasoned stop resolves it. Cancellation records a reason and
stop time; it is not evidence of successful execution or scheduler cancellation.

## Evidence ownership

Mutation-facing evidence must first resolve to a canonical managed row. Any supplied frozen field must agree before source-specific fields are allowlisted.

- Workspace rankings and candidate tables prove scope through the validated workspace plus managed key.
- Candidate and ranking tables are completely validated before other-step or earlier-plan rows are filtered.
- Checkpoint evidence must exist as an independent regular direct child of the matched frozen checkpoint directory, revalidated before selection or finalization.
- Launch/status projections never contribute evidence.

Foreign, unmatched, incomplete, or drifting evidence fails or remains in raw inventory; it does not update canonical rows.

## Atomic commit

The workspace owner reads, reduces, and commits the complete canonical table.

`experiment-monitor` observes one owner-validated input snapshot per round. A
concurrent registration is retained by the commit but first observed next round;
the snapshot never replaces the commit owner's fresh read or concurrency checks.

- Commits hold a stable lock from canonical read through same-directory temporary write, `fsync`, `os.replace`, and run-matrix projection; a digest conflict under that lock fails without overwriting newer state.
- New keys must carry the owning experiment id.

The owner returns the rows actually committed. Callers use those rows for projections, reports, and transition events. A later projection/report failure makes the command nonzero but does not roll back canonical state; the next command may regenerate derived artifacts.

No caller reads or writes `run_manifest.tsv` directly.

## PID and runtime evidence

New managed launches create a dedicated OS session and process group. The
low-level process launcher remains compatible with legacy/internal callers that
omit planned-commit capture: their JSON receipt has exactly `pid`,
`process_group_id`, and `process_start_token`. Provenance-aware managed callers
supply the planned commit and receive the same three fields plus the fourth
`runtime_commit`, observed under the short runtime lock immediately before
process creation. Readers require all three base fields, allow only the optional
`runtime_commit` field, and require a non-empty value to be a full lowercase
40-character Git commit ID; missing base fields or additional fields are
corrupt.

For a provenance-aware managed direct launch, embedded verification, HEAD
capture, and child `Popen` are ordered inside the same short lock. The receipt is
written afterward; the lock does not cover the child lifetime.

- The PID must be the process-group leader.
- Monitoring compares the file with frozen canonical values and the live OS
  start token, so a reused PID is not accepted as the managed workload.
- Historical integer-only PID files are insufficient for script-owned process
control and fail closed.

If those canonical process fields are still blank after an unresolved launch, monitoring or stop may fill them only after the live leader command matches the frozen absolute launch script. Monitoring and stop reject a partially populated canonical process identity.

Only confirmed absence means no process identity. Other evidence is handled as
follows:

- Empty, malformed, non-positive, aliased, or invalid-encoding identity
  content is corrupt and makes launchable scheduled state non-launchable
  `missing_pid`.
- An identity naming an already-dead leader before canonical process fields
  were bound also becomes `missing_pid`; monitoring does not bind it or infer a
  terminal result.
- A path/read `OSError` while scheduled aborts before mutation.

Stop propagates identity uncertainty before signal or mutation and rejects
terminal rows before identity access. It sends `SIGTERM` to the complete process group and commits `stopped` only
after the group has exited.

When monitoring proves corrupt, partial, mismatched, or reused managed process
identity, it records `process_identity_error` with the canonical status update.
Any affected non-terminal run remains active `missing_pid`, so queues cannot
release its capacity or retry it automatically. A valid
identity whose process group is confirmed dead may still become `failed` under
the normal terminal-owner rule.

## Slurm scheduler evidence

Both hparam and ordinary infer use the same Slurm lifecycle owner. Slurm owns
queueing and job state: agent tools submit, observe, and cancel jobs but never
hold a queue for them. A Slurm run's initial canonical row contains no
execution identity or job. The planned log path is not launch evidence.

### Frozen job script

Each run owns one frozen leaf `job.sbatch`, rendered at plan time with its
SHA-256 recorded. Its job name is `{experiment_id}/{step_id}/{run_id}`. It
requests one node with one task and one GPU per frozen Lightning device, the
frozen CPUs per task, memory, walltime, and `nice`, plus `nodelist` when set. It
sets `--no-requeue` and writes `run_dir/slurm.log`.

Inside the allocation the job:

1. Writes `git rev-parse HEAD` to `run_dir/runtime_commit`.
2. Checks the frozen config and launch-script SHA-256 values.
3. Runs one labeled `srun --kill-on-bad-exit=1` step of the frozen launch script.

The step's exit code is written atomically to `run_dir/exit_code`, both at the
normal end and from the `TERM` or `INT` trap. A trapped signal writes 143 or
130, and a failed checkout, commit, or hash check writes 3. The job never
compares HEAD with `planned_runtime_commit`; a difference is recorded rolling
provenance. The commit file is point-in-time provenance only.

### Submission and adoption

`hparam-launch --execute`, `hparam-run-queue --execute`, and
`infer-launch --execute` each make one submission pass under the run lock and
return without polling. The pass runs in this order:

1. Run the pre-submit gates: the frozen script hash, empty runtime and
   checkpoint directories, and the live execution-snapshot probe.
2. Commit the execution identity of each launchable run.
3. Query `squeue --me --states=all` once. A failed query fails the command.

Then each `planned` or `pending` run is handled by the first matching case:

- **Two or more jobs share the run's name:** abort the pass before any `sbatch`
  and leave the decision to a human.
- **Exactly one job has the name:** adopt it by committing its job id, raw state,
  and mapped status.
- **No job, but `slurm.log` or `exit_code` already exists:** commit `lost` with
  the reason `job ran earlier; id unknown`.
- **Otherwise:** run `sbatch --parsable job.sbatch` with every `SBATCH_*`
  variable removed, and commit the returned positive job id as `queued`. A
  non-zero `sbatch` exit commits `launch_failed` with its error. A timeout
  raises, and the next pass adopts the job by name.

All Slurm client subprocesses run on the host where agent tools run and remove
any inherited `SLURM_CLUSTERS`. The recorded and executed submission commands
share one renderer. The job id is an immutable one-time binding.

For a valid registered ordinary inference plan, an execute-time guard failure
may commit `launch_failed` with its diagnostic, but only when all of these hold:

- It happened before the pass committed execution identity.
- A fresh canonical read under the launch lock still shows `planned` or `pending`.
- That row has no launch evidence.

Invalid or unregistered plan artifacts cannot authorize this update. A failure
after execution identity is committed never becomes an unsubmitted failure and
never authorizes a retry.

### Monitor state table

The table applies to non-terminal Slurm rows that have a job id. One
`squeue --me --states=all` snapshot serves every such row in a monitor round,
and no query runs when no row needs one. The monitor copies
`run_dir/runtime_commit` when present; a value that is not a full lowercase
40-character Git commit ID is an error. It also records the job's raw state,
reason, and node while `squeue` still lists the job.

| `run_dir/exit_code` | `squeue --me --states=all` | Stop requested | Status |
| --- | --- | --- | --- |
| `0` | any | any | `completed` |
| non-zero | any | yes / no | `stopped` / `failed`, keeping `scheduler_exit_code` |
| absent | queued state (`PENDING` and similar) | no / yes | `queued` / `stopping` |
| absent | running state (`RUNNING`, `CONFIGURING`, `COMPLETING`, `SUSPENDED`, and similar) | no / yes | `running` / `stopping` |
| absent | terminal state, or not listed | yes | `stopped` |
| absent | terminal state, or not listed | no | `lost`, with the raw state while still listed |

An `exit_code` outside 0 to 255 is an error. `lost` is active and not sticky: it
blocks finalization, and a later `exit_code` upgrades it. Labeled logs never
own lifecycle.

### Stopping and lost runs

`hparam-stop` and `infer-stop` require a reason. What a stop does depends on
the row:

- **The row has a job id:** atomically record non-terminal `stopping` with the
  reason, request time, and job binding, then call `scancel`. The monitor
  commits `stopped` according to the table. A failed or interrupted
  cancellation keeps that intent, and it may be retried only with the same
  reason.
- **The row is launchable:** look its name up in `squeue`. One match binds that
  job and follows the job-id path. No match commits a metadata-only stop.
  Several matches fail.
- **The row is `lost`, or otherwise has no job id:** commit `stopped` without
  calling `scancel`.

Nothing is ever resubmitted automatically. A `lost` run is resolved only by a
reasoned stop. A stale observation created before the stop request may update
diagnostic evidence, but it cannot erase or bypass the canonical stop intent.

## Consumer requirements

`experiment-status` consumes lifecycle only from this canonical table. Step
controller classification belongs to `steps/<step.id>/step.yaml.plan_controller`,
not a duplicate table field. The complete
[read-only status and advisory contract](experiment_workspace.md#read-only-status-and-advisory-actions)
owns frozen-plan validation, controller blockers, and the distinction from
non-launching monitors that write fresh observations.

Every hparam mutation first validates workspace ownership, step registration, frozen run hashes, the independent `recipe.resolved.yaml` byte digest recorded by `plan.json`, and equality between the two complete effective recipe copies. Missing or partial canonical state fails rather than being repaired by launch, selection, or collection.

Selected-candidate consumers obey the canonical ownership checks here and the
[selection contract](hparam_workflow.md#selection-and-selected-candidate-consumers).

