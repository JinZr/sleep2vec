# Run Manifest Contract

`run_manifest.tsv` is the only mutable owner of managed run lifecycle state and execution identity.

## Identity and frozen fields

The canonical managed key is `(step_id, run_id)`. Both fields are required and one canonical table contains at most one row per key. A run uses the next stable step-local `run-NNN` id. `run_name` is human-readable, and `version` is the bounded slug of experiment id, step id, run id, and run name. Version may resolve external evidence only when a complete managed key is absent and the match is unique.

Plan-owned identity, semantic parameters, config/script hashes, artifact paths, runtime/checkpoint directories, and execution identity are frozen after registration. Shared execution identity consists of target, host, workdir, GPUs, log path, and command. Direct execution additionally owns the PID path, launched PID, process-group id, and OS process-start token. Slurm execution instead freezes scheduler type, optional `scheduler_direct_controller` topology, submit token, sbatch path/hash, allocation-identity path, and terminal-sidecar path; the execution-snapshot SHA-256, numeric scheduler job id, and optional cluster are trusted one-time bindings. New Slurm plans materialize the topology as `true` or `false`; older rows may omit it and retain the default bound-cluster routing. PID and scheduler bindings are mutually exclusive. Only the canonical owner may perform each trusted first fill.

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
inference to the monitor. Scheduler-sidecar runs take terminal truth only from
their verified atomic sidecar.
Lifecycle-owned inference with explicit runtime identity uses the same frozen
runtime Python for its workload and every lifecycle commit, after its frozen
runtime-commit guard succeeds.

Managed tables declare either one row per run or many rows per run. Both forms require complete managed identity and reject removed `trial_id` or `param.*` formats. Historical formats remain read-only and are never translated into current state.

## Canonical state and projections

`launch_manifest.tsv` and `run_status.tsv` retain their plan-local paths and fields but are written only from rows returned by a successful canonical commit. They are projections and are never read to restore lifecycle status or execution identity. Matrices, status reports, rankings, and events are also derived artifacts. `RESEARCH_LOG.md` is an append-only narrative record rather than a projection, but it likewise never owns or restores lifecycle state.

Health labels are observational and never own lifecycle state. Managed GPU
activity is attributed to the frozen process group so DDP child processes count
as active. Without another positive progress signal, an unavailable checkpoint
probe, an unavailable GPU probe for a GPU-assigned run, or the first checkpoint
observation without a comparison baseline reports `health_unknown`;
`possibly_stalled` requires a later, comparable observation with no detected
progress. Remote artifact uncertainty preserves the last checkpoint inventory
while leaving the current health poll's `checkpoint_count` blank.

Runtime `run_manifest.json` supplies metrics and checkpoint evidence only. It does not own lifecycle status. A truly missing runtime manifest means evidence is not yet available; an existing alias, non-regular file, invalid encoding/JSON, or non-mapping payload is corrupt.

For test-selected hparam runs, the terminal runtime manifest also contains
`test_all_checkpoints_after_fit: true` and `checkpoint_test_results`, one
`{checkpoint_path, epoch, metrics}` mapping for every regular non-alias
`epoch=*.ckpt` in the frozen checkpoint directory.
Top-level `metrics` remains the test result for the validation-best checkpoint;
checkpoint-level hparam selection uses only the complete nested evidence. The
selector hashes each checkpoint, writes the plan-local all-checkpoint ranking,
and keeps `run_manifest.tsv` at one lifecycle row per run by projecting only
that run's best test-ranked checkpoint.

Finetune runtime directories are single-use: rank zero rejects a non-empty
`log-finetune/<version>` before persisting configuration, loading data, or
fitting. All-checkpoint result rows form one evidence matrix; runtimes evaluate
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
- Active status cannot regress through stale `planned` or `pending` evidence.
- `superseded` commits only when the freshly read canonical state is still `planned` or `pending`.
- Monitoring preserves finished-to-completed normalization for evidence whose script does not own terminal commits.
- Monitoring preserves a managed-script `running` state when neither PID nor W&B execution evidence exists; absence of those evidence sources is not process-exit evidence.
- A `script` path requires strict managed process-identity checks but does not by itself assign terminal-status ownership.
- A lifecycle-enabled generated script owns its terminal commit. Confirmed disappearance of its process group without a canonical `completed` or `failed` commit is `failed`, never inferred success. New hparam launch scripts are explicitly monitor-owned and append a structured shell exit code to their log: only code `0` becomes `finished`; nonzero, missing, or malformed exit evidence becomes `failed`. Historical owner-less hparam plans retain the legacy failure-marker inference.

All lifecycle callers reuse the same row reducer. They do not implement source-specific precedence.

## Evidence ownership

Mutation-facing evidence must first resolve to a canonical managed row. Any supplied frozen field must agree before source-specific fields are allowlisted.

- W&B evidence with an experiment id must match the workspace. Evidence without it may match only one unique runtime version.
- Distinct W&B run ids resolving to the same managed run are ambiguous and fail before canonical state or managed metrics are written.
- Workspace metrics, checkpoints, rankings, adaptive registries, and candidate tables prove scope through the validated workspace plus managed key.
- Candidate and ranking tables are completely validated before other-step or earlier-plan rows are filtered.
- Checkpoint evidence must exist as an independent regular direct child of the matched frozen checkpoint directory, revalidated on the run's execution host before indexing or ranking.
- Launch/status projections never contribute evidence.

Foreign, unmatched, incomplete, or drifting evidence fails or remains in raw inventory; it does not update canonical rows.

## Atomic commit

The workspace owner reads, reduces, and commits the complete canonical table.

- Local commits hold a stable lock from canonical read through same-directory temporary write, `fsync`, `os.replace`, and run-matrix projection.
- SSH commits lock remotely, compare the expected digest, and conditionally replace a same-directory temporary file.
- An SSH conflict causes a fresh read and merge, with at most three attempts. Exhausted conflicts fail without overwriting newer state.
- New keys must carry the owning experiment id.

The owner returns the rows actually committed. Callers use those rows for projections, reports, and transition events. A later projection/report failure makes the command nonzero but does not roll back canonical state; the next command may regenerate derived artifacts.

No caller reads or writes `run_manifest.tsv` directly.

## PID and runtime evidence

New managed launches create a dedicated OS session and process group. They
write one JSON identity file containing exactly `pid`, `process_group_id`, and
`process_start_token`.

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

## Slurm scheduler evidence

One Slurm-backed canonical run owns one frozen leaf `job.sbatch`. Before any
submission, the launcher freezes the plan-level execution snapshot's raw
SHA-256 across all canonical runs. It commits transport identity plus
`submitting`, and passes that digest as a batch-script argument to
`sbatch --parsable`. The returned positive job id and optional cluster are
immutable scheduler bindings. Follow-up `squeue`, `scontrol`, and `scancel`
commands route to the bound cluster. If submission times out, returns malformed
output, or loses SSH after possible submission, the launcher searches the exact
frozen comment token; zero or multiple matches remain unresolved and never
authorize another submission.
Before invoking `sbatch` on either a local or SSH submission host, the canonical
submission command removes every inherited `SBATCH_*` variable so ambient
client options cannot override frozen directives; other submission environment
remains available.

Live controller state comes from `squeue` and `scontrol`. If both no longer know
a bound job, monitoring queries duplicate allocation records through `sacct`
on the bound cluster when accounting is available. Accounting identity requires
exactly one JobID row whose comment matches the frozen submit token; blank,
mismatched, absent, or ambiguous comments fail without changing canonical
lifecycle. A cluster that does not retain job comments cannot provide terminal
accounting identity, so monitoring remains unchanged and fails closed. The
compute wrapper verifies
the exact execution-snapshot bytes before parsing them, then revalidates the
runtime commit, module origin, CLI, and frozen launch/config hashes. It requires
the allocation task count to match the frozen GPU count and starts one foreground
`srun` step with one task per GPU. Only the wrapper writes the allocation and
terminal JSON sidecars bound to the same token and job id; the terminal sidecar
records the aggregate step exit code. A canonical completed
or failed status requires both the matching terminal sidecar and a terminal
scheduler observation; a scheduler failure overrides a zero wrapper exit code.
If controller and accounting evidence are unavailable, a vanished job or an
incomplete terminal evidence pair becomes active `unknown_scheduler` rather
than inferred success or failure. An accounting terminal record is scheduler
evidence only; ordinary completion or failure still requires the matching
sidecar. Before `scancel`, `hparam-stop` atomically records nonterminal
`stopping` with its reason, request time, and frozen job binding. A failed or
interrupted cancellation preserves that canonical intent and may be retried
only with the same reason. Monitoring commits `stopped` only after the same
scheduler job is observed as `CANCELLED`; this explicit stop
intent is the narrow exception that does not require a terminal sidecar because
a pending job may never start its wrapper. Slurm transition flags remain active;
in particular, raw `STOPPED` retains its allocation and maps to canonical
`running` or `stopping`, never terminal `stopped`. When Slurm reports raw
`REVOKED` federation sibling state, sibling-cluster rebinding is unsupported;
the run fails closed as active `unknown_scheduler` while preserving the frozen
job, cluster, scheduler reason, and any stop intent, and never authorizes
rebinding or relaunch. A stale observation that was
created before the stop request may update diagnostic evidence but cannot erase
or bypass the canonical stop intent.

When monitoring proves corrupt, partial, mismatched, or reused managed process
identity, it records `process_identity_error` with the canonical status update.
Any affected non-terminal run remains active `missing_pid`, so queues and
pipeline runners cannot release its capacity or retry it automatically. A valid
identity whose process group is confirmed dead may still become `failed` under
the normal terminal-owner rule.

Experiment checkpoint indexing follows each row's frozen runtime/checkpoint pair. Both may be empty for a non-checkpoint-producing run; a partial pair is invalid. Existing checkpoint evidence must remain inside the eligible managed keys and frozen directories.

## Consumer requirements

Every hparam mutation first validates workspace ownership, step registration, frozen run hashes, the independent `recipe.resolved.yaml` byte digest recorded by `plan.json`, and equality between the two complete effective recipe copies. Missing or partial canonical state fails rather than being repaired by launch, selection, collection, or postprocess.

Selected-candidate postprocessing refreshes lifecycle status from the current canonical manifest rather than trusting ranking or candidate-table status. For test selection, caller-provided rank, checkpoint path, and SHA-256 must match both the frozen workspace ranking and canonical run row before top-k filtering. Physical SHA-256 revalidation then covers only candidates retained by top-k, or every candidate under `all_candidates`. `hparam-external-eval` accepts only `completed` or `finished` runs; it and `hparam-export-logits` reject SSH-owned candidates before writing outputs because these direct helpers have no remote config-staging and result-collection protocol.

An adaptive plan under `adaptive/rounds/round_NNN` is runnable only after the
root-matching `adaptive/workflow.json` commit marker exists as an independent
regular file. Planning and initialization may inspect an uncommitted plan with
an explicit internal bypass, but lifecycle entrypoints never do.

`collect-runs` requires a valid canonical table, distinguishes a header-only current table from missing/corrupt input, and cannot write to or alias the canonical manifest. Optional non-managed summaries may remain best-effort evidence.
