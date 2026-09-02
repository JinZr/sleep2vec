# External Evaluation Pipeline Contract

`experiment-run` is the explicit, resumable owner for the standard workflow
that waits for managed hparam sources, freezes checkpoints selected by their registered rankings,
runs an external-evaluation matrix, and finalizes the experiment. It is not a
general command DAG and is not a monitoring command.

## Invocation and frozen state

```bash
python -m agent_tools experiment-run \
  --run-dir <experiment-root> \
  --spec <external-matrix-v1.yaml> \
  --unlock-final-test \
  [--execute | --dry-run] \
  [--resume] \
  [--poll-seconds 60]
```

Dry-run is the default and starts no process. Execute mode holds one exclusive
runner lock beside `pipelines/<pipeline-id>/` and atomically freezes the source
and parsed spec, their SHA-256 identity, source-plan identities, external-preset
hashes, checkpoint selection, job/attempt mapping, and runtime identity under
that directory. Once state exists, execution requires `--resume --execute`; any
frozen spec, source-plan, preset, config, checkpoint, or artifact drift fails
closed. When an eligible attempt reaches the shared managed scheduler, its
separate Python, route, clean/importable-code, module-origin, live-argv, and
artifact-hash launch gates also fail closed. Those are launch gates, not a claim
that every pipeline read or controller transition re-probes the runtime. A
rolling checkout's commit advance is recorded per attempt and does not rewrite
frozen pipeline or snapshot bytes.

The workspace links to this pipeline-owned subtree:

```text
pipelines/<pipeline-id>/
├── spec.source.yaml
├── spec.resolved.yaml
├── pipeline.json
├── checkpoints.json
├── preflight.json
├── jobs.tsv
├── execution_snapshot.json  # single-variant initial scheduler
├── initial_schedulers/<variant>/execution_snapshot.json
├── results.csv
├── metrics.csv
├── summary.md
├── final.md
├── recipes/<job-id>/attempt-NNN.yaml
├── plans/<job-id>/attempt-NNN/
├── preflight_retries/<job-id>/attempt-NNN.json
├── retry_schedulers/<job-id>/execution_snapshot.json
└── results/<job-id>/attempt-NNN/
```

Multi-variant initial schedulers use the variant-specific snapshot paths.

The v1 spec is a closed contract for one canonical `evaluate` step. It declares
the pipeline and experiment ids, planned/baseline runtime commit, GPU
concurrency, at most two attempts per logical job, checkpoint sources and
policy, external jobs, and whether finalization is enabled. Task, variant,
label, config, and inference module are derived from each frozen source plan.
The spec may assert those values but cannot define a second semantic source.

The closed v1 sections are:

- `pipeline`: `kind: external_matrix`, matching experiment id, one `evaluate`
  step, and `finalize: true`;
- `runtime`: absolute workdir, target Python and full planned/baseline runtime
  commit, accelerator, device, FP32 precision, batch size 128, and seed;
- `execution`: GPU pool, one GPU per run, bounded concurrency, and exactly two
  maximum attempts;
- `evaluation_policy`: `external_test_locked: false` and
  `final_test_unlocked: true`;
- `checkpoint_policy`: one checkpoint, no model averaging, forbidden state-key
  prefixes, and AHI-threshold enforcement;
- `checkpoint_sources`: absolute managed plan, frozen selection metric/mode, and
  optional task/variant/label assertions;
- `jobs`: stable id, source id, cohort, modality, absolute inference preset,
  workers, and optional task/variant/label assertions.

## Source and checkpoint gates

All source plans must belong to the workspace and have only terminal runs
before checkpoint selection. A source is ready when at least one run completed
successfully; failed or stopped runs remain recorded but do not block selection
from the successful per-run winners. A source with no successful run fails.
Selection reuses the managed hparam-ranking and candidate-resolution owner and
requires an exact metric and mode match. The pipeline freezes the selected
score, config, checkpoint path, and content hashes before any external job
starts.
Schema v1 has no remote source-artifact staging boundary, so it accepts only
local source plans and rejects an SSH-owned source before creating pipeline
state or other outputs.
If interruption leaves `checkpoints.json` before its hash reaches pipeline
state, resume reruns the hparam-ranking owner and accepts the orphan only
when every selected field still matches; it never adopts the file by hashing
its current bytes alone.

Checkpoint policy is evaluated before the matrix is launched. It can require a
non-averaged config, reject EMA state keys, require `avg_ckpts=1`, and require
checkpoint-owned AHI threshold evidence for AHI jobs. External results never
participate in checkpoint selection or tuning.

Every declared inference recipe passes the normal `doctor` and `plan` gates,
including the explicit final-test unlock and preset validation, before the
first job starts. The complete initial matrix preflight is committed in
`preflight.json` before any attempt plan is built. Retry preflight evidence is
also committed before its attempt is materialized, so an interrupted runner can
resume without reusing or overwriting an attempt directory. One failed initial
preflight blocks the entire launch wave; a failed retry preflight does not stop
already-independent jobs, but it prevents finalization.

## Managed attempts and results

Each logical job is registered as a standard managed run. Physical GPUs are
assigned through `CUDA_VISIBLE_DEVICES`; the child inference command receives
package-local logical device 0. Every attempt has a new, empty `result_root`,
which is passed to the package-local inference entrypoint through
`--results-root`.
Every attempt freezes the spec's runtime workdir, Python, and planned/baseline
commit. The runtime Python field is one executable name or path without
whitespace, arguments, or `~` shorthand. At start, the generated script records
the actual commit in the canonical run row and uses the same frozen Python for
inference and all lifecycle commits. For this managed direct attempt, the actual
commit is observed between embedded verification and child `Popen` in the same
short runtime lock. The lock does not cover inference lifetime or guarantee
stable checkout bytes throughout inference. A
planned/actual commit mismatch is warning provenance, not a launch blocker.
Missing Python, dirty or shadowing importable code, module-origin drift,
incompatible frozen argv, or frozen artifact-hash drift still prevents the
managed attempt launch. Any `launched` process evidence already committed by
the scheduler remains canonical evidence for monitor reconciliation under the
lifecycle-owner rules.
Exactly one valid `run_manifest.json` must be discoverable below that root and
must agree with the frozen checkpoint, config, preset, split, and runtime
inputs. Its `metrics_csv_path` and `prediction_csv_path` must identify existing,
independent regular files below the result root. Per-disease metric tables remain
optional when the corresponding task emits no per-disease rows.

Only a canonical runtime `failed` or `launch_failed` attempt may receive one
fresh retry. A `completed` or `finished` lifecycle commit whose result manifest
fails verification retains that canonical lifecycle status, while the jobs
projection records the validation error and marks the logical job failed
without retrying it.
Missing or uncertain process identity, a still-live process, or an explicitly
stopped run is not retried automatically. Unsafe process-identity evidence is
persisted on the canonical run, so a runner interruption cannot turn that
blocker into a retry. Independent jobs may continue after another job fails,
but a failed logical job blocks finalization.

If a current attempt has canonical status `missing_pid`, or an unfinished
pipeline shares execution capacity with another such run, scheduling commits
the latest observations without starting another process, syncs the attempt
table, and persists the pipeline as `blocked` with the blocking run identity.
After that identity is resolved manually, `--resume --execute` may continue the
same frozen pipeline. An already-terminal matrix does not need that capacity
and may still aggregate and complete.

`run_manifest.tsv` remains the only lifecycle owner. Pipeline status, job
tables, and reports are projections. The optional pipeline fields carried by a
managed row are defined in [run_manifest.md](run_manifest.md).

## Completion and finalization

The pipeline succeeds only when every declared logical job has one verified
successful attempt and no attempt remains active. The user-selected `--spec`
defines the job set; schema v1 does not impose a fixed matrix size. Every job
frozen from that spec must complete (N/N). The pipeline then writes
`results.csv`, `metrics.csv`, `summary.md`, and `final.md` from all scalar
manifest metrics, preserving non-finite values explicitly, including the
frozen checkpoint, preset, actual runtime commit, and result path for each job;
the planned/baseline commit remains in the frozen spec and canonical manifest.
Mixed commits are provenance only, not a pipeline factor.

The report is committed before `experiment-finalize`; the terminal experiment
commit is the final mutation. A partial matrix, invalid manifest, exhausted
retry, active run, or failed report prevents finalization.
