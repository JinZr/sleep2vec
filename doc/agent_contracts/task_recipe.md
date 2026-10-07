# Task Recipe Contract

Task recipes under `recipes/` bind one task to an experiment and step. The
accepted fields and finite allowlists are defined in the
[task recipe schema](../../recipes/schemas/task_recipe.schema.md).

This contract owns authored and effective inputs, consultation, and runtime
routing. Ordinary and adaptive hparam protocols belong to
[hparam_workflow.md](hparam_workflow.md). Workspace publication, takeover, status,
and finalization belong to [experiment_workspace.md](experiment_workspace.md);
canonical lifecycle and scheduler evidence belong to [run_manifest.md](run_manifest.md).

## Contents

- [Authored-input closure](#authored-input-closure) and [effective recipe](#effective-recipe)
- [Consultation and diagnostics](#consultation-and-diagnostics)
- [Experiment binding](#experiment-binding) and [task/variant routing](#task-and-variant-routing)
- [Non-hparam runtime identity](#non-hparam-runtime-identity) and [managed ordinary inference](#managed-ordinary-inference)
- [Hparam workflow](hparam_workflow.md) (separate contract)

## Authored-input closure

Recipe shape is validated before config inspection and before any workspace,
script, manifest, or event is created. Validation is task-aware and owner-based:

- `experiment_workspace` owns `experiment` and `step`;
- task decision owners own `inputs`, `evaluation_policy`, `execution`, `search`,
  and `adaptive`;
- renderer mappings own runtime and preset CLI fields;
- `plans` owns top-level routing and artifacts.

This does not add a second schema registry or general recipe facade.

Unknown and task-inapplicable fields fail with their original field path and
source layer. For hparam recipes, the base finetune source and local tuning
overlay are validated independently before merged semantics run. Raw authored
`_...` fields are reserved and rejected.

## Effective recipe

Planning produces one effective recipe:

```text
recipe fields + recipe decisions + explicit user decisions
  -> materialized recipe
  -> cheap authored-input checks and static ownership consultation
  -> config summary and consultation
  -> frozen plan and resolved recipe
```

Materialization follows these rules:

- Recipe decisions with a task-owned canonical field are written into that
  field first. Explicit user decisions may then override both the canonical
  field and effective decision mapping before config inspection and
  consultation are rerun.
- A layered hparam recipe takes its task only from the local overlay or an
  explicit user decision. The base finetune task cannot become the effective
  tuning task.
- Policy-only decisions remain under `decisions` rather than creating inert
  recipe sections.
- For finetune and hparam tasks, an explicit `required_channels` decision must
  match `preset_build.required_channels` in the selected config.
- For preset preparation, a config `preset_build` block exclusively owns
  `required_channels` and `min_channels`. Matching recipe decisions retain
  provenance, but the effective preset omits the duplicate CLI fields. Without
  `preset_build`, those decisions materialize into the preset CLI fields.
- For hparam, `inputs.ckpt_path` is reserved for the selected final-evaluation
  checkpoint and is not rendered into tuning finetune commands.
- Empty or null rendered decisions remain unresolved instead of falling back
  to older canonical values. Explicit `pretrained_backbone_path: null` retains
  its established train-without-pretraining meaning.

Before config or data reads, the effective recipe and its retained source layers
must be serializable by the existing frozen-JSON writer. YAML dates/timestamps
must be quoted when a string is intended; unsupported values are not silently
converted. Task-owned checks also reject known hard input errors at this point,
including malformed hparam search spaces. These checks use the effective values
after user overrides and do not turn missing decisions into new hard failures.
Config-dependent profile expansion and full consultation still run afterward.

Static experiment/step consultation also runs before config or data reads. A
layered hparam recipe must supply its own local ownership; complete base-recipe
metadata does not satisfy that requirement. Missing or unresolved ownership
returns the existing `NEEDS_USER_INPUT` questions without probing config, data,
workspace identity, or runtime. Legal user decisions still materialize first;
ownership is filled in recipe fields, not new decision aliases. Base-task
consultation with `require_experiment=False` and standalone diagnostics keep
their existing scope.

Plan preflight also compares an existing `experiment.yaml` with the effective
experiment identity before config/data inspection. Doctor does not add this
workspace read. An absent manifest is not a reservation or authorization to
register: final workspace validation and registration locks remain authoritative.

`plan.json` and `recipe.resolved.yaml` must contain the same complete effective
recipe. Retained base/local recipe copies are source audit only; launch,
selection, adaptive, and postprocess consumers read the effective recipe.
For hparam plans, top-level `plan.json.resolved_recipe_sha256` binds the exact
bytes of `recipe.resolved.yaml`; consumers verify that digest before parsing the
recipe and then verify complete semantic equality between the two copies.
Frozen recipes containing trusted `_base_recipe` or `_local_recipe` metadata
are consumed through `run_artifacts.read_hparam_plan`; they are not re-entered
through the authored-recipe loader.

Decision-file behavior and precedence belong to [user_decisions.md](user_decisions.md).

## Consultation and diagnostics

Run `doctor` or `plan` consultation before generating runnable experiment
commands. Missing, ambiguous, conflicting, or `ASK_USER` high-impact decisions
require user input, not an inferred value recorded as explicit. For hparam
selection this includes the split, metric, and mode. Resolve generated questions
through [user decisions](user_decisions.md); a blocked-plan retry uses a fresh
output directory. [Context bundles](context_bundle.md) are diagnostic-only and
do not authorize runnable commands.

`doctor` emits its PID and synchronous phase on stderr before potentially slow
probes. For an unblocked hparam recipe it separately reports the target host,
actual Python executable/version, and installed PyTorch Lightning distribution
version without importing Lightning. Runtime-card probe failure does not change
the consultation result or introduce a dependency-version gate. Manager, target,
and allocation identities are distinct; see the
[execution identity legend](experiment_workspace.md#execution-identity-legend).

Slurm task diagnostics may inspect version, priority, backfill, accounting,
partition, reservation, and one literal fixed node's configured CPU, memory,
and GPU capacity through read-only `scontrol` queries. Fixed-node capacity is
an empty-node theoretical limit, not current availability: one run that cannot
fit is a failure, while a larger co-resident batch or unavailable capacity
evidence is a warning. Blank, comma-separated, and bracket-expression
`nodelist` values leave capacity unknown. This time-stamped advice does not
change the frozen scheduler request or make unavailable accounting or capacity
a registration blocker. `nice=0` is the highest unprivileged nice setting; no
user-side option guarantees first priority.

Within one `doctor` or `plan` consultation, index checks reuse the accepted
config summary and subject keys from successful, complete local survival or
multilabel sidecar validation. This is one validation view, not a cross-call
cache: later invocations reread inputs, and registration and launch retain
independent checks. Failed or deferred validation keeps its existing path, and
config-byte drift checks remain in force. Full subject key sets stay in memory,
not reports or frozen artifacts.

Read the completed command's report and exit code together. For a normally
completed `doctor` or `plan` consultation, PASS or nonblocking WARN returns 0,
FAIL returns 1, and NEEDS_USER_INPUT returns 2; FAIL takes precedence when
blocking issues are mixed. These are consultation-result codes, not a universal
CLI error protocol. Argument, input, or runtime errors may instead return
nonzero with a stderr diagnostic or traceback and no report. Normal doctor
progress also uses stderr, so stderr output alone is not a failure signal.
An absent or incomplete report, or an exit code inconsistent with its result,
does not establish successful consultation. Inspect the original error; do not
invent missing decisions or continue execution from that evidence.

PASS or a nonblocking WARN does not guarantee that `--output-dir` creates a directory:
doctor writes questions/templates only when its output contract requires them.
Doctor also does not establish workspace writability, plan registration,
submission, or completed results. If a check is slow or SSH disconnects, first
establish the fate of that operation; connection loss alone does not authorize
a duplicate check or launch. Continue through the
[takeover flow](experiment_workspace.md#takeover-and-continue-execution).

## Experiment binding

Every runnable recipe declares complete `experiment` metadata (`id`, `title`,
`objective`, `root`, and `baseline`) and a `step` (`id`, `phase`, and
`purpose`). A hparam recipe declares its own binding rather than inheriting it
from its base finetune recipe.

The plan directory must be inside `experiment.root`. Workspace layout, path
canonicalization, step registration, and lifecycle ownership belong to
[experiment_workspace.md](experiment_workspace.md).

## Task and variant routing

| task | accepted variant | generated runtime |
| --- | --- | --- |
| `sleep2stat` | omitted or `null` | `python -m sleep2stat` |
| `preset_prepare` | `sleep2vec`, `sleep2vec2`, `sleep2expert` | package-local `preprocess/save_dataset_presets.py` |
| `finetune`, `hparam_tune` | `sleep2vec`, `sleep2vec2`, `sleep2expert`, `sex_age_baseline` | `<variant>.finetune` |
| `infer`, `evaluate` | `sleep2vec`, `sleep2vec2`, `sleep2expert`, `sex_age_baseline` | `<variant>.infer` |
| `embedding_extraction` | `sleep2vec`, `sleep2vec2`, `sleep2expert` | `<variant>.extract_embeddings` |

Preset preparation routes each variant to its package-local script. Variant
scripts reject root-only `manifest_output` and `write_sidecar_manifest` fields
instead of falling back to the root runtime. `sex_age_baseline` does not own
preset generation.

`pretrain` and `adapt` have direct runtime skills and CLIs but are not runnable
task-recipe values because agent tools have no renderer for them. Missing or
unsupported routing blocks command generation.

### Whole-night embedding extraction

The `embedding_extraction` task intentionally exposes only local whole-night
NPZ-index export. It accepts pretrain and finetune model YAML, freezes the
validated config bytes, and routes the generated command to the selected
package-local extractor in the current checkout. It does not accept an
`execution` block, config-window mode, presets, Kaldi, dataset source overrides,
or a configurable batch size.

The recipe supplies explicit `config`, `ckpt_path`, non-empty `data_index`, and
`eval_split`; unique model `channels`; `embedding_kind: both`, `layer_index: -1`,
`output_format: npz`, `sequence_mode: whole-night`, and `max_source_tokens` in
`[1, 4095]`; optional `device` and `num_workers` in `[0, 8]`; and an absolute,
fresh `embedding_dir` with `overwrite: false`. Test rows additionally require
`external_test_locked: false` and `final_test_unlocked: true`.

The model config must use a RoFormer backbone and
`model.cls.embedding_type: bert`. A finetune config must not set
`data.finetune_preset_path`, its effective `data.data_channel_names` must match
`model.channels`, and every selected index row must satisfy the effective
`train_dataset_names` or `test_dataset_names` filter when that filter is non-empty.
Rows without a non-empty `source` use the authored index path as their source,
matching the package-local dataset loader.

Planning shares the runtime's static index validator for required and unique
columns, non-empty split selection, duplicate paths, finite 30-second-aligned
durations, token cap, and NPZ existence. The embedding directory and plan
directory may not contain one another, and embedding output may not occupy the
experiment-managed `plans`, `reports`, or `steps` namespaces. The package-local extractor remains
authoritative for model, checkpoint, and dataset loading semantics. Its terminal
NPZ manifest binds the config, checkpoint, extractor, and index hashes; there is
no Kaldi/preset hash promise in this task.
The plan also freezes checkpoint and index CSV hashes and verifies those external
inputs before committing the run to `running`; referenced NPZ contents remain
runtime-owned and are not hashed during planning.

### Runtime paths and data inputs

Except for `embedding_extraction`, runnable non-hparam scripts use an explicit
absolute `execution.workdir` for cwd and PYTHONPATH, otherwise `REPO_ROOT`.
Relative runtime-semantic dataset
and checkpoint paths are validated from that same cwd while their authored
strings remain unchanged. Runtime `~` home-directory shorthand is rejected;
use an absolute or workdir-relative path.

Local relative `inputs.config` values remain planning-source locators under
`REPO_ROOT`. The planner freezes their bytes and gives the runtime a plan-local
absolute config path.
Successful plans also freeze an internal `_plan_context` containing the creator
home, manager Python, and repository root. Registered-plan recompilation uses
that exact context for relative source locators and implicit script defaults;
it never substitutes the status reader's host environment.

- Generic and variant-local Kaldi inference requires a `kaldi_data_root`
  directory plus a `kaldi_manifest` file and rejects NPZ preset overrides.
- NPZ finetune/inference may consume a frozen preset without reopening survival
  sidecars or the multilabel `label_index` / `has_label_index`. Multilabel runs
  still read `disease_columns_index` to name per-disease metrics and prediction
  columns, so it stays required. `preset_prepare` always validates the sidecar
  files needed to build that preset.
- Checkpoint and pretrained-backbone inputs must be files.
- Checkpoint averaging rejects AHI. `avg_ckpts` must be a positive integer,
  `best`/`last` aliases require an explicit `avg_ckpt_dir`, and any explicit
  averaging directory is validated from the runtime cwd.

## Non-hparam runtime identity

`preset_prepare`, `infer`, and `evaluate` accept `execution.python` and
`execution.runtime_commit`. Declaring either turns the otherwise-common
`execution.workdir` into an all-or-none local/default-local runtime identity.
Python is one executable name or path without whitespace, arguments, or `~`
shorthand; the commit is a full 40-character planned/baseline Git commit ID.
Authored hexadecimal may use either case; the resolved recipe freezes it in
lowercase.
Other non-hparam tasks reject Python and commit identity rather than silently
rendering commands that ignore them.

When the identity is present, the resolved recipe and plan freeze those planned
bytes and use the same frozen Python for the workload and all `running` /
`completed` / `failed` commits. A provenance-aware managed launcher observes
HEAD under the short runtime lock immediately before spawning its child; a
direct script without that outer launcher observes HEAD at its own `running`
boundary. The canonical start commit records that point-in-time value. A
planned/actual mismatch does not block execution or rewrite the plan, and the
observation does not promise that checkout bytes stay unchanged for the whole
job. Use an absolute Python path for independence from the launcher's PATH; an
explicit executable name remains PATH-resolved. Route-specific launch gates are
defined below: direct preset scripts do not acquire the hparam managed-scheduler
module-origin or live-argv contract.
`execution.target` and `execution.host` on other non-hparam tasks remain
path-validation context; they do not provide a generic SSH launcher. Ordinary
Slurm inference uses the managed exception below; the direct-script identity
and lifecycle rules above remain unchanged.

New `preset_prepare` recipes without Python/commit identity freeze the planning
interpreter (`sys.executable`), manager Git HEAD, and `REPO_ROOT` workdir before
command generation; that HEAD is planned/baseline provenance rather than a
permanent checkout pin. This default applies only to local/default-local
execution at the exact manager checkout with no remote path context. A separate
workdir or remote path context requires a complete explicit local identity; SSH
is not a preset launcher. An unavailable manager commit fails before workspace
creation. Partial authored identities are rejected, not filled with defaults.
Historical registered preset plans without identity retain their original
commands and are never rebound or migrated by readers.

### Managed preset preparation

New effective `preset_prepare` recipes freeze `execution.scheduler.type: direct`
and an explicit script terminal-status owner. Plan and launch on the execution
host; this does not add recipe-driven SSH execution. The registered plan keeps
its variant-local preset command and frozen planned runtime identity. Its top-level
`run.sh` delegates to `preset-launch --plan-dir <plan>`: both default to dry-run,
and execution requires `--execute`. Do not launch the worker `launch.sh`
separately or add a background SSH shell wrapper.

The launcher validates the registered plan and frozen inputs, then records the
execution identity and launch attempt before starting a detached process.
The launch command rechecks the frozen worker script and config hashes before
spawning, including changes made after the manager's initial validation, and
requires a clean importable-code state and the lifecycle module in the current
repository. Under the short runtime lock it observes HEAD immediately before
spawning and adds that value to the new managed PID receipt. It does not perform
the hparam workload module-origin or live-argv checks.
Stdin is closed to input; stdout and stderr share the run's persistent
`stdout.log`. The process has its own session and a recorded PID, process group,
and start token. Loss of the launching connection or an incomplete receipt does
not authorize another launch. These controls preserve the preset runtime
contract above; they do not create the hparam module/host execution snapshot.
The launcher runs on the shared direct managed scheduler under the run lock.
A dry run refreshes the manifest projection and status report without recording
launch evidence, and a completed experiment refuses launch. A pre-launch guard
failure such as the runtime identity probe records `launch_failed` with its
reason; a row that already has launch evidence or a PID receipt is never
launched again.

The worker remains responsible for `running`, `completed`, and `failed` commits.
Use `experiment-monitor` to observe the existing run; it neither launches work
nor infers successful completion from a log or a vanished process. Use
`preset-stop --plan-dir <plan> --reason <reason>` for a reasoned, identity-checked
stop. An uncertain stop remains `stopping` with its original reason; monitoring
does not clear that intent. A later explicit stop may confirm the recorded
process group has exited, but cannot replace its identity or launch again.
Historical registered plans without the direct scheduler declaration keep
their original bytes and interpretation; the new launcher does not migrate or
restart them.

### Managed ordinary inference

Ordinary `infer` and `evaluate` plans may declare `execution.scheduler.type:
slurm`. They reuse the existing single-node Slurm resource fields and protected
environment rules in [Launch and queue](hparam_workflow.md#launch-and-queue), with explicit
`execution.workdir`, `execution.python`, and a full 40-character planned/baseline
`execution.runtime_commit`, frozen in lowercase. Submission may use `target: local`
or `target: ssh`;
`scheduler.direct_controller` independently selects controller routing. Paths
must already be available on the execution host; planning does not upload a
runtime or input bundle.

`gpus_per_run: N` freezes allocation-local `runtime.devices: [0, ..., N-1]`.
Conflicting devices or CPU execution settings are rejected.
`sex_age_baseline` supports multi-GPU DDP. Checkpoint choice, averaging, split,
and external-test authorization retain their existing consultation rules.

The registered plan contains one run, a top-level manager `run.sh`, a frozen
worker `launch.sh`, and `job.sbatch`. Use `infer-launch --plan-dir <plan>` for
a dry-run and add `--execute` only when execution is authorized; `run.sh`
delegates to the same operation and also defaults to dry-run. Do not submit or
run the worker separately. Its exact model command stays in the frozen plan,
while the canonical submission command is bound by the shared launch
transaction. Registration and dry-run do not bind a job, cluster, or execution
identity.

Use `experiment-monitor` to refresh scheduler evidence and `experiment-status`
for read-only advice. `infer-stop --plan-dir <plan> --reason <reason>` acts on
the unique run through the shared Slurm stop transaction. Repeated execute
does not resubmit a queued, active, terminal, or uncertain run. The
[Slurm evidence contract](run_manifest.md#slurm-scheduler-evidence) owns terminal
and lost-receipt rules, including failures before the workload starts. This
does not extend managed evaluation pipelines or migrate historical manually
wrapped inference plans.
