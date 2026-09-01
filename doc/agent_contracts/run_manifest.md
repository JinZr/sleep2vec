# Run Manifest Contract

`run_manifest.tsv` is the only mutable owner of managed run lifecycle state and execution identity.

This contract owns identity, state reduction, evidence, and atomic commits.
Command choice, status advice, and finalization belong to
[the workspace contract](experiment_workspace.md); hparam selection and adaptive
protocols belong to [the recipe contract](task_recipe.md).

## Contents

- [Identity and frozen fields](#identity-and-frozen-fields)
- [Canonical state and projections](#canonical-state-and-projections) and [runtime artifact evidence](#runtime-artifact-evidence)
- [Status reducer](#status-reducer), [evidence ownership](#evidence-ownership), and [atomic commit](#atomic-commit)
- [PID and runtime evidence](#pid-and-runtime-evidence)
- [Slurm evidence](#slurm-scheduler-evidence): [submission/routing](#submission-and-routing),
  [sidecar reads and monitor reuse](#sidecar-reads-and-monitor-round-reuse),
  [terminal evidence](#terminal-evidence), and [stopping/uncertainty](#stopping-and-uncertain-states)
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
registration. `planned_runtime_commit` is the plan's full baseline commit;
`runtime_commit` is the full commit observed under the short runtime lock
immediately before a provenance-aware direct child is spawned or before the
Slurm allocation wrapper spawns `srun`. A script-owned direct route without the
new outer receipt observes it at its own `running` boundary. Each is a trusted
first fill and immutable once non-empty. The observed SHA is point-in-time
provenance, not a guarantee that checkout code bytes remain unchanged for the
whole job.
They may differ: that difference is recorded rolling provenance, not a lifecycle
error or scientific variable. Shared execution identity otherwise consists of
target, host, workdir, GPUs, log path, and command. Direct execution additionally
owns the PID path, launched PID, process-group id, and OS process-start token.
Slurm execution instead freezes scheduler type, optional
`scheduler_direct_controller` topology, submit token, sbatch path/hash,
allocation-identity path, and terminal-sidecar path; the execution-snapshot
SHA-256, numeric scheduler job id, and optional cluster are trusted one-time
bindings. New Slurm plans materialize the topology as `true` or `false`; older
rows may omit it and retain the default bound-cluster routing. PID and scheduler
bindings are mutually exclusive. Only the canonical owner may perform each
trusted first fill.

Pipeline-managed inference rows may additionally freeze `pipeline_id`,
`job_id`, `attempt`, and `result_root`. Managed rows may freeze
`terminal_status_owner`. These fields do not change the canonical
`(step_id, run_id)` key. Older managed rows may omit them and retain their
existing lifecycle behavior; a present value is immutable and must match all
later evidence.

When present, `terminal_status_owner` is exactly `script`, `monitor`, or
`scheduler_sidecar`. It
selects the existing process-exit rule explicitly: script-owned runs must commit
their own terminal status, while monitor-owned runs leave confirmed-exit
inference to the monitor. Scheduler-sidecar runs require the verified atomic
sidecar under the [scheduler terminal-evidence rules](#terminal-evidence).
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
Slurm SSH health obtains the display tail and later path-mtime age in one
remote operation, preserving their independent failures. An invalid paired
response falls back once to the existing separate probes. Direct-process health
keeps its probe order; no log evidence is cached across observations.

## Runtime artifact evidence

Runtime `run_manifest.json` supplies metrics and checkpoint evidence only. It does not own lifecycle status. A truly missing runtime manifest means evidence is not yet available; an existing alias, non-regular file, invalid encoding/JSON, or non-mapping payload is corrupt.

For test-selected hparam runs, the terminal runtime manifest also contains
`test_all_checkpoints_after_fit: true` and `checkpoint_test_results`, one
`{checkpoint_path, epoch, metrics}` mapping for every regular non-alias
`epoch=*.ckpt` in the frozen checkpoint directory.
Top-level `metrics` remains the test result for the validation-best checkpoint;
checkpoint-level hparam selection uses only the complete nested evidence.
The [selection owner](task_recipe.md#selection-and-selected-candidate-consumers)
keeps a many-checkpoint audit separate from this table's one lifecycle row per
run, which projects only that run's best test-ranked checkpoint.

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

For a pipeline attempt, `result_root` is a single-use empty directory and the
runtime manifest beneath it must be unique and match the frozen inference
inputs. Pipeline projections cannot infer success from files other than that
validated manifest.

## Status reducer

The current vocabulary includes scheduled `planned`/`pending`, scheduler handoff `submitting`/`queued`, active `launched`/`running`/`stopping`/`unknown_remote`/`unknown_scheduler`/`missing_pid`, and terminal `completed`/`failed`/`finished`/`launch_failed`/`stopped`/`superseded` states.

- An update without status preserves the existing status.
- Terminal status is sticky, except incoming `failed` evidence may correct `completed` or `finished`.
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
job or cluster binding, launch time, or stop request. Slurm's frozen plan
`log_path` and preflight execution-snapshot hash do not establish launch. Slurm
`completed`, `finished`, and `failed` states, and stopped Slurm runs with launch
evidence, still require a scheduler job id. Cancellation records a reason and
stop time; it is not evidence of successful execution or scheduler cancellation.

## Evidence ownership

Mutation-facing evidence must first resolve to a canonical managed row. Any supplied frozen field must agree before source-specific fields are allowlisted.

- W&B evidence with an experiment id must match the workspace. Evidence without it may match only one unique runtime version.
- Distinct W&B run ids resolving to the same managed run are ambiguous and fail before canonical state or managed metrics are written.
- Workspace metrics, checkpoints, rankings, adaptive registries, and candidate tables prove scope through the validated workspace plus managed key.
- Candidate and ranking tables are completely validated before other-step or earlier-plan rows are filtered.
- Checkpoint evidence must exist as an independent regular direct child of the matched frozen checkpoint directory, revalidated on the run's execution host before indexing or ranking.
- Launch/status projections never contribute evidence.

Foreign, unmatched, incomplete, or drifting evidence fails or remains in raw inventory; it does not update canonical rows.

Experiment checkpoint indexing follows each row's frozen runtime/checkpoint
pair. Both may be empty for a non-checkpoint-producing run; a partial pair is
invalid. Existing checkpoint evidence must remain inside the eligible managed
keys and frozen directories.

## Atomic commit

The workspace owner reads, reduces, and commits the complete canonical table.

`experiment-monitor` observes one owner-validated input snapshot per round. A
concurrent registration is retained by the commit but first observed next round;
the snapshot never replaces the commit owner's fresh read or concurrency checks.

- Local commits hold a stable lock from canonical read through same-directory temporary write, `fsync`, `os.replace`, and run-matrix projection.
- SSH commits lock remotely, compare the expected digest, and conditionally replace a same-directory temporary file.
- An SSH conflict causes a fresh read and merge, with at most three attempts. Exhausted conflicts fail without overwriting newer state.
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
SHA; missing base fields or additional fields are corrupt.

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

- Empty, malformed, non-positive, aliased, or invalid-encoding local identity
  content is corrupt and makes launchable scheduled state non-launchable
  `missing_pid`.
- A local identity naming an already-dead leader before canonical process fields
  were bound also becomes `missing_pid`; monitoring does not bind it or infer a
  terminal result.
- Equivalent unbound remote evidence remains `unknown_remote`.
- A local path/read `OSError` while scheduled aborts before mutation. Remote
  permission, type, decoding, transport, and timeout failures produce
  recoverable `unknown_remote` monitoring evidence.
- A launch-command timeout or SSH transport return code `255` is unresolved
  rather than `launch_failed`. The attempted run remains active until identity
  monitoring proves its state, preventing a duplicate launch after transport
  uncertainty.

Stop propagates identity uncertainty before signal or mutation and rejects
terminal rows before identity access. On SSH it verifies and signals atomically.
It sends `SIGTERM` to the complete process group and commits `stopped` only
after the group has exited.

When monitoring proves corrupt, partial, mismatched, or reused managed process
identity, it records `process_identity_error` with the canonical status update.
Any affected non-terminal run remains active `missing_pid`, so queues and
pipeline runners cannot release its capacity or retry it automatically. A valid
identity whose process group is confirmed dead may still become `failed` under
the normal terminal-owner rule.

## Slurm scheduler evidence

Both hparam and ordinary infer/evaluate use the same Slurm lifecycle owner.
Ordinary inference freezes its model command in the plan and worker, but its
initial canonical row contains no transport identity, submission command, job,
or cluster. The planned log path and preflight snapshot are not launch evidence.
The worker does not write canonical status or rely on an inner EXIT trap:
`run-frozen-job` owns its allocation and terminal sidecars.

### Submission and routing

One Slurm-backed canonical run owns one frozen leaf `job.sbatch`. Before any
submission, the launcher freezes the plan-level execution snapshot's raw
SHA-256 across all canonical runs. After launch preflight, each run queries
`scontrol show config` for exactly one non-empty valid `ClusterName`, then
atomically commits that cluster, transport identity, and `submitting` before
calling `sbatch --parsable` with the snapshot digest. A failed identity query
does not submit or bind the cluster; hparam leaves that run's state unchanged.
The ordinary inference facade records an unsubmitted guard failure as described
below. Dry-run does not query or bind the cluster. A cluster without a job id
is valid only in `submitting` or
`launch_failed`. The returned positive job id is an immutable binding; a bare
job-id receipt retains the prebound cluster. Non-empty cluster bindings cannot
be changed or cleared. Follow-up `squeue`, `scontrol`, and `scancel`
commands route to the bound cluster. If submission times out, returns malformed
output, or loses SSH after possible submission, the launcher searches the exact
frozen comment token on the prebound cluster and frozen controller route;
zero or multiple matches remain unresolved and never
authorize another submission.
Bound-cluster routing uses `--clusters=<scheduler_cluster>` unless the canonical
run freezes `scheduler_direct_controller: true`. Local control transport does
not establish that topology; monitoring and stop preserve the frozen choice.
All Slurm client subprocesses on either a local or SSH submission host remove
inherited `SLURM_CLUSTERS`; `sbatch` additionally removes every `SBATCH_*`
variable. The recorded submission command and transport share one renderer.
`SLURM_CONF`, `PATH`, other environment, and the parent process remain unchanged.

For a valid registered ordinary inference plan, an execute-time guard failure
may commit `launch_failed` with its diagnostic only when a fresh canonical read
under the launch lock is still `planned`/`pending` and has no launch evidence.
Invalid or unregistered plan artifacts cannot authorize this update. Exceptions
after `submitting` or a trusted execution binding retain the shared scheduler
state and reconciliation rules; they never become an unsubmitted failure or authorize retry.

If a receipt names a different cluster, the launcher first commits the returned
job id with the original cluster, `unknown_scheduler`, and raw state
`SUBMISSION_CLUSTER_MISMATCH`, then aborts the submission batch. This quarantine
and its reason survive stale observations. Monitoring does not read Slurm
sidecars or query the scheduler for it; stop rejects before recording intent
or cancelling. Queue and direct launch both block further submissions in that
plan. There is no automatic conflict-clearing or legacy-cluster recovery path.

### Sidecar reads and monitor-round reuse

Slurm sidecar reads validate the managed path and read the same opened regular
file in one local/SSH operation. Raw `..` components, symlinks in any traversed
directory or leaf, hardlinked files, and non-regular leaves are rejected. Only
a missing open means absent evidence; read and UTF-8 failures remain errors.
Remote reads require one complete response for every requested path, including
explicit missing values; empty or partial command output is not absence. These
checks do not promote sidecar job or cluster values to canonical identity.

One monitor round may share sidecar text within one managed owner and matching
execution host, independently of scheduler job, cluster, or controller topology.
Terminal files are read first; allocation reads include only runs with missing
or empty terminal text, or a valid empty terminal mapping. Malformed or nonempty
terminal data never causes allocation prefetch. Each run still parses and
validates its own identity before scheduler queries. A future run's invalid file
does not change the current run's error order: a failed batch is discarded and
that phase uses exact per-file reads for the rest of the round. Conflicted runs
and mismatched or overridden transports are excluded from prefetch. Missing or
empty snapshots last only this round; a sidecar published afterward is observed
in the next successful round. Absence never establishes a terminal state.

Within one `hparam-monitor` or `experiment-monitor` round, runs with complete
canonical job, cluster, token, transport, and explicit controller topology may
share a successful exact-ID `squeue` query on the same frozen route. Groups
require at least two distinct job IDs. The first participating run queries only
after its sidecar validation; every run still validates its own evidence before
using the shared result. Only positive matches are reused. Missing IDs take the
normal single-job query path; failed or ambiguous batches publish no partial
results and disable sharing for that group for the rest of the round. Legacy
identities, submission-cluster conflicts, and transport overrides are excluded.
Each round starts fresh, and launch, stop, and submission reconciliation never
consume these monitor snapshots. `scheduler_observed_at` remains the per-run
observation processing time, not a promise of a separate scheduler sampling time.
Health observation reuses a successful controller fallback's details, but keeps
the fresh controller retry after accounting when the first controller query
returned no record. This changes query timing, never frozen identity or the
evidence required to commit lifecycle state.

### Terminal evidence

Live controller state comes from `squeue` and `scontrol`. If both no longer know
a bound job, monitoring queries duplicate allocation records through `sacct`
on the bound cluster when accounting is available. Accounting identity requires
exactly one JobID row whose comment matches the frozen submit token; blank,
mismatched, absent, or ambiguous comments fail without changing canonical
lifecycle. A cluster that does not retain job comments cannot provide terminal
accounting identity, so monitoring remains unchanged and fails closed. A
self-contained bootstrap acquires the runtime lock before importing the
checkout-local Slurm worker and transfers that same open lock descriptor into
the worker. The compute wrapper then verifies the frozen launch/config hashes.
Under that short lock it requires a clean importable-code state, verifies that
the named module still resolves inside the current repository, validates the
live CLI, and observes the allocation-side actual commit. It then verifies the
exact execution-snapshot bytes and requires current Python/version and module name to
match; it does not require the current module-origin path to equal the frozen
snapshot field. The wrapper writes allocation evidence and spawns `srun` before
releasing the lock. The observed SHA does not promise that checkout bytes stay
unchanged afterward. It requires
`SLURM_NTASKS` to match the frozen GPU count and the observed Python executable
and version to match the plan snapshot. It starts one foreground, labeled
`srun --kill-on-bad-exit=1 --quit-on-interrupt` step with one task per GPU and no
explicit task-level GPU binding, preserving the full allocation GPU visibility
expected by the frozen Lightning devices in every rank. Each task emits one
bounded startup-identity record before the runtime command; only global rank
zero writes the diagnostic exit marker. The compute-node hostname is observed
allocation evidence and may differ from the submission host. Only the wrapper
writes allocation and terminal JSON sidecars bound to the same token and job
id; the terminal sidecar records the aggregate step exit code. Labeled logs
never own lifecycle. Sidecar job ids are local query candidates
only; first binding requires a valid submission receipt or positive exact-token
scheduler evidence on the frozen route. Sidecar clusters never select the query
route or fill canonical identity. Query errors or missing records cannot save
sidecar-only identity: without a canonical job id the run stays `submitting`.
Legacy rows with a job id but no cluster can still finish with normal scheduler
and matching sidecar evidence, without filling cluster or enabling the missing-
evidence exception. A canonical completed or failed status
normally requires both the matching terminal sidecar and a terminal scheduler
observation; a scheduler failure overrides a zero wrapper exit code. The narrow
accounting-disabled exception requires the exact bound job to be absent from
`squeue`, explicitly invalid in `scontrol`, `sacct` to report that accounting
storage is disabled, and the atomic sidecar to match the frozen job, token, and
non-empty canonical cluster while transport and controller topology remain
explicitly frozen and match the route used for every query. SSH transport
failures remain `unknown_scheduler` even if partial output mentions disabled
accounting. The exception records raw scheduler state `MISSING`, recovers
exit zero as `completed` and non-zero as `failed`, and never infers `stopped`.
Other unavailable controller or accounting evidence, a vanished job, or an
incomplete terminal evidence pair becomes active `unknown_scheduler` rather
than inferred success or failure. An accounting terminal record is scheduler
evidence only; ordinary completion or failure still requires the matching
sidecar.

For ordinary inference, a surviving outer worker records the workload or
guard's nonzero exit without needing an inner EXIT trap. Bootstrap, log-open,
outer SIGKILL, or terminal-publication failures can leave no usable sidecar:
authenticated raw `COMPLETED`, `FAILED`, or `TIMEOUT` then remains
`unknown_scheduler` with diagnostic state, not guessed success or failure.

### Stopping and uncertain states

Before `scancel`, `hparam-stop` and `infer-stop` atomically record nonterminal
`stopping` with its reason, request time, and frozen job binding. A failed or
interrupted cancellation preserves that canonical intent and may be retried
only with the same reason. Monitoring commits `stopped` only after the same
scheduler job is observed as `CANCELLED`; this explicit stop
intent is the narrow exception that does not require a terminal sidecar because
a pending job may never start its wrapper. A cancellation signal received while
the wrapper is validating frozen identity terminates the job without starting
the leaf process. Slurm transition flags remain active;
in particular, raw `STOPPED` retains its allocation and maps to canonical
`running` or `stopping`, never terminal `stopped`. When Slurm reports raw
`REVOKED` federation sibling state, sibling-cluster rebinding is unsupported;
the run fails closed as active `unknown_scheduler` while preserving the frozen
job, cluster, scheduler reason, and any stop intent, and never authorizes
rebinding or relaunch. A stale observation that was
created before the stop request may update diagnostic evidence but cannot erase
or bypass the canonical stop intent.

## Consumer requirements

`experiment-status` consumes lifecycle only from this canonical table. Step
controller classification belongs to `steps/<step.id>/step.yaml.plan_controller`,
not a duplicate table field; pipeline identity columns must agree with it. The
complete [read-only status and advisory contract](experiment_workspace.md#read-only-status-and-advisory-actions)
owns frozen-plan validation, controller blockers, and the distinction from
non-launching monitors that write fresh observations.

Every hparam mutation first validates workspace ownership, step registration, frozen run hashes, the independent `recipe.resolved.yaml` byte digest recorded by `plan.json`, and equality between the two complete effective recipe copies. Missing or partial canonical state fails rather than being repaired by launch, selection, collection, or postprocess.

Selected-candidate consumers obey the canonical ownership checks here and the
[selection/postprocessing contract](task_recipe.md#selection-and-selected-candidate-consumers).

Adaptive lifecycle entrypoints also require the independent workflow readiness
marker and ordered evidence in [initialization readiness](task_recipe.md#initialization-readiness).

`collect-runs` requires a valid canonical table, distinguishes a header-only current table from missing/corrupt input, and cannot write to or alias the canonical manifest. Optional non-managed summaries may remain best-effort evidence.
