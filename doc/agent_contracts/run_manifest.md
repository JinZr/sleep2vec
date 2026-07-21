# Run Manifest Contract

`run_manifest.tsv` is the only mutable owner of managed run lifecycle state and execution identity.

## Identity and frozen fields

The canonical managed key is `(step_id, run_id)`. Both fields are required and one canonical table contains at most one row per key. A run uses the next stable step-local `run-NNN` id. `run_name` is human-readable, and `version` is the bounded slug of experiment id, step id, run id, and run name. Version may resolve external evidence only when a complete managed key is absent and the match is unique.

Plan-owned identity, semantic parameters, config/script hashes, artifact paths, runtime/checkpoint directories, and execution identity are frozen after registration. Execution identity consists of target, host, workdir, GPUs, PID/log paths, command, and the launched PID, process-group id, and OS process-start token. Only the canonical owner may perform each trusted first fill.

Pipeline-managed inference rows may additionally freeze `pipeline_id`,
`job_id`, `attempt`, `result_root`, and `terminal_status_owner`. These fields do
not change the canonical `(step_id, run_id)` key. Older managed rows may omit
them and retain their existing lifecycle behavior; a present value is immutable
and must match all later evidence.

When present, `terminal_status_owner` is exactly `script` or `monitor`. It
selects the existing process-exit rule explicitly: script-owned runs must commit
their own terminal status, while monitor-owned runs leave confirmed-exit
inference to the monitor.
Lifecycle-owned inference with explicit runtime identity uses the same frozen
runtime Python for its workload and every lifecycle commit, after its frozen
runtime-commit guard succeeds.

Managed tables declare either one row per run or many rows per run. Both forms require complete managed identity and reject removed `trial_id` or `param.*` formats. Historical formats remain read-only and are never translated into current state.

## Canonical state and projections

`launch_manifest.tsv` and `run_status.tsv` retain their plan-local paths and fields but are written only from rows returned by a successful canonical commit. They are projections and are never read to restore lifecycle status or execution identity. Matrices, Markdown reports, rankings, and events are also derived artifacts.

Runtime `run_manifest.json` supplies metrics and checkpoint evidence only. It does not own lifecycle status. A truly missing runtime manifest means evidence is not yet available; an existing alias, non-regular file, invalid encoding/JSON, or non-mapping payload is corrupt.

For a pipeline attempt, `result_root` is a single-use empty directory and the
runtime manifest beneath it must be unique and match the frozen inference
inputs. Pipeline projections cannot infer success from files other than that
validated manifest.

## Status reducer

The current vocabulary includes scheduled `planned`/`pending`, active `launched`/`running`/`unknown_remote`/`missing_pid`, and terminal `completed`/`failed`/`finished`/`launch_failed`/`stopped`/`superseded` states.

- An update without status preserves the existing status.
- Terminal status is sticky, except incoming `failed` evidence may correct `completed` or `finished`.
- Active status cannot regress through stale `planned` or `pending` evidence.
- `superseded` commits only when the freshly read canonical state is still `planned` or `pending`.
- Monitoring preserves finished-to-completed normalization for evidence whose script does not own terminal commits.
- Monitoring preserves a managed-script `running` state when neither PID nor W&B execution evidence exists; absence of those evidence sources is not process-exit evidence.
- A `script` path requires strict managed process-identity checks but does not by itself assign terminal-status ownership.
- A lifecycle-enabled generated script owns its terminal commit. Confirmed disappearance of its process group without a canonical `completed` or `failed` commit is `failed`, never inferred success. Hparam launch scripts do not own terminal commits, so their monitor infers `finished` or `failed` from confirmed process exit and log evidence.

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
- A launch-command timeout is unresolved rather than `launch_failed`. The
  attempted run remains active until identity monitoring proves its state,
  preventing a duplicate launch after a transport timeout.

Stop propagates identity uncertainty before signal or mutation and rejects
terminal rows before identity access. On SSH it verifies and signals atomically.
It sends `SIGTERM` to the complete process group and commits `stopped` only
after the group has exited.

When monitoring proves corrupt, partial, mismatched, or reused managed process
identity, it records `process_identity_error` with the canonical status update.
Pipeline runners treat that evidence as a permanent automatic-retry blocker.
A lifecycle-owned active run whose identity file disappears becomes
`missing_pid`, not retryable `failed`; a valid identity whose process group is
confirmed dead may still become `failed` under the normal terminal-owner rule.

Experiment checkpoint indexing follows each row's frozen runtime/checkpoint pair. Both may be empty for a non-checkpoint-producing run; a partial pair is invalid. Existing checkpoint evidence must remain inside the eligible managed keys and frozen directories.

## Consumer requirements

Every hparam mutation first validates workspace ownership, step registration, frozen hashes, and equality between the complete effective recipe in `plan.json` and `recipe.resolved.yaml`. Missing or partial canonical state fails rather than being repaired by launch, selection, collection, or postprocess.

`collect-runs` requires a valid canonical table, distinguishes a header-only current table from missing/corrupt input, and cannot write to or alias the canonical manifest. Optional non-managed summaries may remain best-effort evidence.
