# Task Recipe Contract

Task recipes under `recipes/` bind one task to an experiment and step. The
accepted fields and finite allowlists are defined in the
[task recipe schema](../../recipes/schemas/task_recipe.schema.md).

This contract owns authored and effective inputs, consultation, runtime routing,
and ordinary/adaptive hparam protocols. Workspace publication, takeover, status,
and finalization belong to [experiment_workspace.md](experiment_workspace.md);
canonical lifecycle and scheduler evidence belong to [run_manifest.md](run_manifest.md).

## Contents

- [Authored-input closure](#authored-input-closure) and [effective recipe](#effective-recipe)
- [Consultation and diagnostics](#consultation-and-diagnostics)
- [Experiment binding](#experiment-binding) and [task/variant routing](#task-and-variant-routing)
- [Non-hparam runtime identity](#non-hparam-runtime-identity) and [managed ordinary inference](#managed-ordinary-inference)
- [Hparam search space](#search-space), [registration preflight](#registration-preflight), and [launch/queue](#launch-and-queue)
- [Execution snapshot and launch revalidation](#execution-snapshot-and-launch-revalidation)
- [Selection and selected-candidate consumers](#selection-and-selected-candidate-consumers)
- [Adaptive workflow](#adaptive-workflow): [initialization readiness](#initialization-readiness),
  [frozen round identity](#frozen-round-identity), [strategy/budget](#strategy-and-budget),
  and [proposal handshake](#proposal-handshake)

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
partition, and reservation capabilities through read-only `scontrol` queries.
This time-stamped advice does not change the frozen scheduler request or make
unavailable accounting a registration blocker. `nice=0` is the highest
unprivileged nice setting; no user-side option guarantees first priority.

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
  or multilabel sidecars. `preset_prepare` always validates the sidecar files
  needed to build that preset.
- Checkpoint and pretrained-backbone inputs must be files.
- Checkpoint averaging rejects AHI and `sex_age_baseline`. `avg_ckpts` must be a
  positive integer, `best`/`last` aliases require an explicit `avg_ckpt_dir`,
  and any explicit averaging directory is validated from the runtime cwd.

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
environment rules in [Launch and queue](#launch-and-queue), with explicit
`execution.workdir`, `execution.python`, and a full 40-character planned/baseline
`execution.runtime_commit`, frozen in lowercase. Submission may use `target: local`
or `target: ssh`;
`scheduler.direct_controller` independently selects controller routing. Paths
must already be available on the execution host; planning does not upload a
runtime or input bundle.

`gpus_per_run: N` freezes allocation-local `runtime.devices: [0, ..., N-1]`.
Conflicting devices or CPU execution settings are rejected;
`sex_age_baseline` supports only one GPU. Checkpoint choice, averaging, split,
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

## Hparam workflow

Search-space generation does not choose or revoke scientific test policy; read
[test-access policy](external_test_locking.md#selection-and-test-access-policy)
before selecting a search. The selection split, metric, and mode must come from
explicit user authorization, not an agent inference relabeled as `explicit_recipe`.

### Search space

- Search keys are explicit `runtime.<name>` fields or
  `yaml:/json/pointer/path` config overrides. Removed bare or `param.*` forms
  are rejected rather than translated.
- An explicit search requires positive `search.max_runs`, uses `method: grid`,
  and is exactly one of:
  - `search.parameters`: a per-key candidate mapping expanded by Cartesian
    product, with axes in lexicographic full-key order and each candidate
    value list in its authored order;
  - `search.configurations`: complete joint configuration points expanded
    in their authored list order, one run per point. Mapping key order does
    not affect a point's run name or parameter summary.
- Both shapes use the same key rules and apply `[:max_runs]` prefix truncation
  after expansion. Reordering parameter mapping keys does not change run ids
  or the selected grid prefix.
- `search.profile: finetune_balanced` is the alternative authored intent for
  `sleep2vec` and `sleep2vec2` finetuning labels `ahi`, `arousal`, `stage4`,
  `age`, and `sex`. It is mutually exclusive with authored `parameters` or
  `configurations`. The hparam adapter resolves config facts and materializes
  `method: grid`, a default `max_runs: 12`, and deterministic complete joint
  configurations before consultation. An explicit budget override must be in
  `[4, 32]` and cover every generated level.
- The profile compiler, owned by
  `agent_tools/domain/finetune_hparam_profile.py`, searches bounded technical
  levels for learning rate, weight decay, the full LayerMix block,
  supported dropout fields, full/head-only/LoRA adaptation arms when a
  pretrained backbone is passed to tuning, and positive
  scalar `pos_weight` when it exists. A complete explicit
  `finetune.layer_mix` mapping and an explicit `finetune.lora` control mapping
  containing both control booleans are required. Omitted non-control LoRA
  fields retain canonical variant loader defaults; the exact generated config
  and hashes plus runtime/repository identity preserve provenance. Disabled
  source LayerMix must already use `layer_indices:
  null` and `shared_across_modalities: false`; single-channel source LayerMix
  must also disable sharing. Multi-channel enabled levels cover both shared
  and unshared atomic mappings without spending budget on inert duplicates. It keeps
  batch size, epochs, patience, aggregation, EMA, pretrained checkpoint,
  channels, and class weights frozen. Its first point exactly matches the
  source runtime and all active source config mappings. A zero source weight
  decay uses profile-owned `1e-5` and `1e-4` positive anchors so the family is
  genuinely searched. The remaining points
  include normalized LayerMix off, synchronized dropout, and full/head-only/
  LoRA arms when eligible before greedily covering missing levels and then
  missing pairs with stable tie breaking instead of truncating a Cartesian
  prefix.
- The authored profile remains the only generation intent. The resolved
  recipe and plan freeze its exact configurations, config digest, runtime/repo
  identity, budget, searched-family coverage, metric, and split. Reports may
  claim only the best observed candidate within that frozen search domain,
  metric, split, and budget. This is an inventory of registered profile axes,
  not a claim that arbitrary or unknown config fields were searched; only axes
  with more than one candidate are reported as searched.
- The first profile version does not support adaptive tuning. Existing
  explicit searches and historical frozen plans are unchanged; profile
  expansion is not retroactively required by registered-plan readers.
- Adaptive source recipes must declare `search.parameters`, which supplies the
  envelope and neighborhood source. `search.configurations` appears only in
  derived rounds and static plans.

While creating a new hparam recipe, an explicit request to tune selects the
unique supported `finetune_balanced` profile only when no authored parameters,
configurations, or adaptive search exists. Existing explicit/adaptive search
always wins. That request covers the profile's deterministic technical levels
and default 12-run search budget when experiment, step, config, label,
selection split/metric/mode, test policy, host, and runtime identity are already
unambiguous. An authored budget override or expansion needs separate authority.
Without `inputs.pretrained_backbone_path`, LoRA adaptation stays fixed;
`inputs.ckpt_path` is final-evaluation-only, not a tuning backbone.

The tuning request alone does not authorize publication or launch. Publication
may carry the authorized scope through doctor and plan. Once launch is explicitly
authorized, continue through launch dry-run, queue execute, terminal monitoring,
selection, final report, and finalization without asking again about tool-owned
technical levels or execution. This does not authorize test unlock, changed
label/split/checkpoint/data, or an adaptive round. Report only the best observed
candidate within the frozen domain, metric, split, and budget, never a global optimum.

### Registration preflight

New ordinary hparam plans and adaptive rounds are fully materialized in a
temporary directory on the final destination filesystem. The staged frozen
bundle is recompiled and validated with the same reader used after publication;
each final candidate config is then checked with the recipe variant's canonical
`load_finetune_config` and `validate_model_config` in the planner's current Python
and code environment. This applies equally to profile expansions, explicit
configurations, parameter grids, and accepted agent proposals after all YAML
overrides. Consultation checks captured input bytes; registration independently
checks the exact frozen candidate bytes. Launch, including dry-run, repeats the
frozen config check for prospective `planned`/`pending` runs before submission;
active/terminal-only calls and monitoring do not load candidate configs.
Identical config bytes share one canonical load only within each validation
boundary, never across calls or rounds. These checks do not reread full sidecar
tables for each candidate. Runtime overrides remain CLI arguments, not YAML
model-config fields, and their per-combination checks are not deduplicated.

Direct registration also checks frozen candidate bytes unless it follows the
existing trusted staged-publication path. That internal path still strictly
rereads frozen artifacts; it is not a persistent validation certificate or a
general bypass for callers. The common publication/registration boundary is
owned by [the workspace contract](experiment_workspace.md#publication-and-registration).

The tool also validates managed output topology on the frozen execution host
and inspects the target Python, repository commit, module origin, supported CLI
options, and every final argv. Only a complete pass permits atomic publication,
step/controller registration, canonical run-row merge, and the `plan_created`
event. Pipeline-derived attempts use the same target and topology checks once
per scheduler group before any attempt in that group is published or
registered.

`agent_tools plan --validate-only` exposes that exact staged path for an
ordinary `hparam_tune` recipe, then discards the staging directory. PASS, FAIL,
and `NEEDS_USER_INPUT` do not create a plan, workspace, question/event file,
step, or canonical run row. Other tasks are rejected because this first
validate-only contract is hparam-specific.

Every new hparam `plan.md` includes a human-readable registration-preflight
card. Target Python, runtime commit, module origin, and the argv digest come
from the frozen execution snapshot. The card separately reports target CLI
argv checks and planner-local final-config checks, including total runs and
unique config bytes. Target CLI preflight proves argument parsing, not config
execution on that target. Neither check proves model construction, checkpoint
compatibility, forward/backward, GPU execution, or complete candidate-specific
dataset validation; those operations are not performed. The
sex-age canonical finetune wrapper uses `load_config(validate_sidecars=False)`.
The card distinguishes the control transport from the validated preflight host and shows the actual
Python executable/version reported by that target. Variant, runtime module,
actual config loader, architecture, and channels come from each final generated
config and are grouped with their run IDs. The card is a projection for audit;
it names the frozen scheduler and, for Slurm, the single-node task/GPU/rank
topology plus per-task resources and controller routing. The topology is
derived from the same normalized scheduler request used by the launcher. The
frozen generated config bytes and hashes remain semantic authority. This
deterministic preflight does not inspect free bytes, estimate checkpoint storage,
or turn unavailable Slurm accounting into a plan blocker; accounting capability
remains a time-stamped `doctor` diagnostic.

Rendering this audit projection does not rescan data indexes or label sidecars;
consultation and final-evaluation validation remain separate gates.

### Launch and queue

The optional `execution` block configures the managed launcher.

- `execution.python` names one target Python executable without whitespace,
  arguments, or `~` shorthand. `execution.runtime_commit` names a full
  40-character planned/baseline Git commit ID; launch records the actual commit
  separately. Authored hexadecimal is case-insensitive and freezes lowercase.
- Conda wrapping belongs in `execution.conda_env`, not `execution.python`.
- Omitted `execution.scheduler` resolves to `{type: direct}`. Direct runs use
  the existing `gpu_pool` / `gpus_per_run` / `max_concurrent` process model.
- `execution.scheduler.type: slurm` keeps `target` as local/SSH control
  transport and submits every frozen run as its own single-node allocation. It
  requires `partition`, `cpus_per_task`, `memory`, and `walltime`; optional scheduler
  fields are `nice`, `nodelist`, and boolean `direct_controller`. The latter
  defaults to false: follow-up commands route to the bound cluster with
  `--clusters`. Set it to true only when the submission endpoint already talks
  directly to that controller and federation routing is unavailable. This
  topology is never inferred from `target`. `gpus_per_run` is a positive YAML integer,
  defaults to one, and becomes the allocation GPU count plus logical
  `runtime.devices=[0, ..., N-1]`. The allocation wrapper starts one foreground
  `srun` step containing exactly N tasks, one per GPU. `cpus_per_task` applies
  to each task, so the total CPU request is `N * cpus_per_task`; `memory` is
  the allocation's whole-node memory limit. `sex_age_baseline` rejects N > 1
  because that runtime does not implement DDP.
  Slurm recipes reject `gpu_pool`, `max_concurrent`, `conda_env`, locally
  authored `runtime.devices`, unknown scheduler fields, and arbitrary sbatch
  arguments. They also reject `SLURM_*`, `RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
  `MASTER_*`, and `CUDA_VISIBLE_DEVICES` entries in `execution.env` because
  allocation identity, distributed rank identity, and GPU isolation belong to
  the scheduler. The allocation wrapper removes ambient generic distributed
  rank and rendezvous variables before starting `srun`, so Slurm task identity
  remains canonical.
- Only the canonical manager runtime—a local target at `REPO_ROOT` without a
  conda wrapper—may omit Python and commit identity. Planning then freezes the
  current manager interpreter and baseline repository HEAD. SSH targets,
  separate local workdirs, and conda-wrapped targets must author both values
  explicitly.
- With direct scheduling, `hparam-launch` starts one capacity-limited wave and
  `hparam-run-queue --execute` keeps filling that capacity. With Slurm, the
  launch owner submits every launchable leaf job without applying host-global
  GPU capacity to scheduler allocations.
- Generated leaf scripts use only `execution.workdir` on `PYTHONPATH`;
  `execution.env.PYTHONPATH` is rejected rather than merged.

Preview with `hparam-launch` before execute and inspect frozen identity,
scheduler/resources, GPU assignment, W&B project/group, log and lifecycle
identity paths, and test policy. Dry-run is the default; `--execute` is the
launch action. It does not grant scientific or test-access decisions.

Before calculating direct capacity, execute-mode launch refreshes observable
active blockers from other plans sharing the relevant target, host, and GPU
pool, then commits their status transitions. The full queue fails if a
current-plan run or relevant cross-plan capacity blocker is `missing_pid`.
A queue with no eligible slot does not probe the execution snapshot.
`hparam-monitor` never starts pending work: by default it rereads and observes
the current plan every 60 seconds until terminal; `--poll-seconds` changes the
interval and `--once` performs one round. `--health` adds progress evidence.
See [workspace command side effects](experiment_workspace.md#lifecycle-entrypoints)
for observation writes and explicit raw-log display.

`hparam-stop` with a non-empty reason can cancel a canonical `planned` or
`pending` run before any execution identity is bound. It rereads the canonical
row under the same lock as launch, records `stopped`, the reason, and stop time,
then updates the existing projections and `run_stopped` event. This applies to
direct and Slurm plans without process, SSH, sidecar, or scheduler probes.
Dry-run preview identity does not count; Slurm's plan-owned log path and
preflight snapshot also do not prove submission. Any partial or complete launch
identity, launch time, or existing stop request keeps the authenticated runtime
stop path mandatory. Already terminal runs still reject another stop request.
Post-commit publication failures propagate without rolling back the canonical
stop; `hparam-stop` does not resume publication for terminal runs.
Canceled runs retain their frozen artifacts and cannot be launched later.

### Execution snapshot and launch revalidation

Registration preflight records verified Python/version, host, repository and
commit, module origin, explicit-environment digest, normalized supported-option
digest, and exact validated argv digest in `execution_snapshot.json` before a
new hparam plan is published. The verified target records its current commit;
a difference from the planned/baseline commit is warning provenance, not a
publication blocker. The target must have no tracked worktree changes and no
untracked or ignored importable Python or extension-module code. Untracked
experiment artifacts and data remain allowed.
The runtime module must resolve inside that repository, and every frozen argv
must pass its actual `argparse` implementation; rendered CLI text is not evidence.
The snapshot stores the explicit execution environment, normalized supported
options and digests, and every validated argv vector.

Eligible execute waves live-reprobe the same target, using the configured
target/workdir/conda/explicit-environment wrapper. Commit and derived CLI-option
inventory may advance with a rolling checkout; launch records the actual commit
and still validates every frozen argv against the live parser. Python, route,
repository/module origin, explicit environment, clean importable code, and
frozen artifact hashes remain fail-closed. The registered plan and
`execution_snapshot.json` bytes are never rewritten to match a later checkout.
Direct execution needs a capacity-eligible candidate and no missing-PID blocker;
Slurm needs launchable rows. Output topology is rechecked before process start
or submission. Dry-run and monitor do not probe or create the execution snapshot;
the separate dry-run candidate-config check remains planner-local.
External datasets, drivers, and environment outside explicit `execution.env`
remain operational dependencies rather than snapshot contents. The eligible
execute-wave probe above owns live frozen-argv compatibility. Before a direct
managed child is spawned, one short lock covers the embedded launch verification
of Python/version, repository root, hostname, module origin, tracked and
untracked or ignored importable code, and frozen script/config hashes, followed
by HEAD capture and `Popen`. A small Slurm bootstrap preserves terminal evidence
when the checkout worker cannot import or start. The allocation worker takes the
short lock for preflight, HEAD snapshot, allocation-sidecar publication, and
`srun` `Popen`. The locks are released after spawn;
these are point-in-time launch observations, not a promise that checkout code
bytes remain unchanged throughout the job.
Target and leaf `PYTHONPATH` contain only `execution.workdir`; another manager
checkout cannot satisfy missing imports.

For Slurm execute, the launcher freezes the snapshot's raw SHA-256 in every
canonical run before submission and passes it to the batch job. Allocation
verification is detailed in [terminal evidence](run_manifest.md#terminal-evidence).
The shared lifecycle owner also defines
[submission/routing](run_manifest.md#submission-and-routing),
[sidecar reads and monitor reuse](run_manifest.md#sidecar-reads-and-monitor-round-reuse),
and [stopping/uncertain states](run_manifest.md#stopping-and-uncertain-states).

Supported historical boundary: a missing snapshot can be established only
while every run is `planned` or `pending` with no committed execution target.
Once execution identity or later state exists, recreate the plan rather than
upgrading it. Plans lacking frozen Python/commit identity also require recreation;
removed `trial_*` plans and status files remain unmanaged and read-only.

### Selection and selected-candidate consumers

Run `hparam-select` only after every managed run is terminal; manifest,
checkpoint inventory, and physical hash evidence come only from successful
canonical runs on their frozen local or SSH execution target. Compatible plans
in one step must agree on selection metric, mode, and split. Selection replaces
current-plan keys in shared `reports/ranking.csv`, reranks the complete step,
and writes deterministic `reports/hparam_selection.md`.
Unavailable SSH evidence for a canonically successful run fails selection
before ranking output. Once canonical selection rows bind checkpoint hashes,
selector re-entry may only reproduce the same score and checkpoint evidence.
Deleting `ranking.csv` does not authorize replacement from changed runtime
evidence; the projection may be rebuilt only from unchanged canonical selection.
Validation-selected tuning resolves a fixed epoch checkpoint rather than a
moving best/last alias.

For test-selected tuning, complete finite checkpoint-level evidence for the
frozen `test_*` metric globally ranks every compatible regular non-alias saved
checkpoint before choosing the best per run. The immutable plan-local
`checkpoint_test_ranking.csv` has plan-local `rank`; the workspace ranking
retains one row per run and records each winner's global position as
`checkpoint_rank`. This audit does not add lifecycle rows. Selection binds the
exact overall winner path/SHA-256, each contributing plan's checkpoint-ranking
path/hash, and the selection report's path/hash in the canonical manifest.
The report records metric/mode/split, evaluated count, winner
run/checkpoint/score, parameters, search overrides, frozen config/script
paths/hashes, and ranking path. Runtime evidence shape remains owned by
[the run manifest contract](run_manifest.md#runtime-artifact-evidence).
Later compatible plans may extend the workspace ranking without rewriting or
invalidating an earlier plan's frozen audit.

Selected-candidate consumers refresh lifecycle from the current canonical
manifest, not a ranking or candidate-table status. For test selection,
caller-provided rank, checkpoint path, and SHA-256 must match both frozen
workspace ranking and canonical row before top-k filtering. Physical hash
revalidation then covers retained candidates only, or every candidate under
`all_candidates`; generated external-evaluation scripts recheck that hash when
executed. `hparam-external-eval` accepts only `completed` or `finished` runs.
It and `hparam-export-logits` reject SSH-owned candidates before writing because
these direct helpers lack remote config-staging and result-collection protocols.
Read-only candidate resolution is owned by `hparam_selection`: when canonical
selection exists it verifies `reports/ranking.csv`, preserves canonical rank,
and resolves top-K/all over successful per-run winners only. Direct analysis
helpers may still consume an explicit one-winner-per-run candidate table before
canonical selection for validation-selected plans; that path validates the
table without publishing lifecycle state. Test-selected plans require canonical
selection and raw multi-epoch checkpoint rankings are not postprocess inputs.

Status validates bound audits and reconstructs the global checkpoint/run
ranking; finalization rehashes every audited checkpoint on its frozen target.
The detailed [status read-set](experiment_workspace.md#read-only-status-and-advisory-actions)
and [final report acceptance](experiment_workspace.md#finalization) have one
workspace owner, including mixed/all-failed cases and historical completed-read
compatibility. Final test authorization belongs to
[external_test_locking.md](external_test_locking.md); managed evaluation pipelines
belong to [experiment_pipeline.md](experiment_pipeline.md).

## Adaptive workflow

The optional `adaptive` block defines append-only rounds bounded by
`adaptive.max_runs_total`.

An authored recipe with `adaptive.enabled=true` must enter through
`hparam-adaptive-init`. The generic `plan` command rejects it before workspace
mutation; only the adaptive workflow owner may materialize round plans.

### Initialization readiness

Round 000 has a final readiness boundary after canonical plan registration.
The owner validates the initial registry, reconciles `plan_created`, and writes
the matching README. It then atomically creates root-matching
`adaptive/workflow.json` as an independent regular file, binding the validated
registry and README hashes to
the no-clobber marker commit, and records `adaptive_init` afterward. Consumers
require the exact ordered events as well as the marker; launch, queue, and
monitor cannot treat partial initialization as runnable. Only internal
planning/initialization inspection may explicitly bypass this readiness check.

Existing recovery is narrow: a complete unregistered round is reusable only
when deterministic regeneration yields an identical tree. Under the round
publication lock, initialization rereads canonical state and may repair only a
missing or malformed initial registry; a valid registry with different frozen
rows, an incomplete round, partial canonical state, or a differing visible
tree is rejected rather than repaired in place.

After readiness, verify the initial plan and launch dry-run before the
authorized initial execute. For takeover and predecessor obligations, use the
single [workspace continuation flow](experiment_workspace.md#takeover-and-continue-execution);
the exact next-round protocol is [below](#proposal-handshake).

### Frozen round identity

- Initialization resolves `execution.python`, `execution.runtime_commit`,
  `adaptive.objective_metric`, and `adaptive.objective_mode` for round 000. The
  Python, objective, route, and scientific contract are workflow-wide frozen;
  round 000's commit is baseline provenance. The frozen route includes the
  scheduler type and, for Slurm, `partition`, `nodelist`, and
  `direct_controller`; scheduler resource limits such as CPU, memory, walltime,
  and nice remain operational inputs.
- Later rounds re-read the mutable source recipe, reject conflicting frozen
  Python, route, objective, or scientific values, and may plan from a newer
  runtime commit. Other operational execution fields, including concurrency,
  GPU allocation, and `env`, remain source-controlled subject to normal
  preflight. Each round plan remains immutable.
- The source recipe and every suggestion pass read-only preflight before digest,
  suggestion, or event artifacts are written. Earlier round plans, configs,
  logs, and checkpoints are not rewritten.

Each run records its planned/baseline and actual commit through the canonical
manifest. Mixed commits across rounds are provenance only, not an A/B arm or
scientific search variable. A later runtime does not rewrite prior plans or
snapshots. Adaptive commands append experiment events and create registry,
digest, suggestion, and round artifacts; they do not make those artifacts
alternate lifecycle owners.

### Strategy and budget

Control flags must be YAML booleans; run, round, and poll budgets must be positive
YAML integers; replacement grace and margin must be finite and non-negative.
Test or external objectives require explicit
`adaptive.test_feedback_for_selection=true`; a `test_*` objective also needs
`test_after_fit=true`.

When `selection_split=test`, adaptive `test_*` objectives, including one distinct
from the selection metric, reduce complete `checkpoint_test_results` by the
frozen objective mode and bind the selected checkpoint path/epoch. Only
canonically `completed` or `finished` runs are eligible for checkpoint-objective
ranking and incumbency. Validation/run-level objectives such as `val_*` and
`best_model_score` retain top-level evidence. A running test-checkpoint objective
cannot trigger metric-based retirement, while independent log failures can.

`adaptive.suggest.strategy` defaults to `agent_proposal`; the only other value
is explicit `best_neighborhood`. An enabled proposal workflow requires a
non-blank string `adaptive.objective_metric` and non-empty
`adaptive.objective_mode`, `adaptive.round_size`, `adaptive.max_rounds`, and
`adaptive.max_runs_total`. Missing, null, or blank required values return
`NEEDS_USER_INPUT` (exit 2) before workspace mutation; a non-string objective metric fails the
recipe contract.

`agent_proposal` is terminal-only, so `adaptive.replacement` must be omitted or
exactly `{enabled: false}`. Optional `adaptive.suggest.bounds` may authorize a
closed interval for numeric search parameters:

- keys must be a subset of `search.parameters`;
- integer-valued grids require integer endpoints;
- grids containing a float accept finite integer or float endpoints;
- categorical or mixed grids do not accept bounds.

Without explicit bounds, numeric parameters use their original minimum and
maximum, and categorical proposals remain within the original choices. A
disabled adaptive block starts no suggestion protocol. Active-round replacement
and automatic neighborhood suggestions require explicit `best_neighborhood`.
Each confirmed replacement start grants one retirement credit. A durable Slurm
`stopping` request reserves one credit in plan order but does not release
capacity; only scheduler-confirmed `stopped` may precede a capacity-dependent
replacement launch.

### Proposal handshake

Agent proposals use two phases.

1. A proposal-free `hparam-adaptive-step` monitors and digests the current
   round. It returns `waiting_for_round_terminal` while a run is active;
   otherwise it writes an immutable
   `adaptive/proposal_inputs/round_NNN--<id12>.json` snapshot.
2. An external agent writes only the exact
   `adaptive/proposal_submissions/round_NNN--<id12>.json` path named by that
   snapshot. It must not create or replace proposal inputs or tool lifecycle
   state. `hparam-adaptive-step --proposal <path>` previews the proposal; adding
   `--execute` applies it through normal preflight, registration, and launch.

Proposal-input schema v2 requires `input.source_config_sha256`. The request id
binds those exact config bytes with complete digest rows, source recipe, frozen
execution identity, remaining budget, and parameter envelopes. The tool records
an `agent_proposal_requested` issuance containing the request id, source and
target rounds, exact paths, and complete snapshot-file hash. If a crash leaves
the exact snapshot without its issuance, retry appends the missing event. One
matching event is idempotent; duplicate or conflicting records fail. Input v1
must be regenerated, while proposal submissions remain schema v1.

Phase two requires one matching issuance and exact snapshot bytes before it
trusts bounds or budget. It reconstructs the complete input from current recipe,
workflow, round, manifest/registry, and runtime evidence, then repeats that
validation after candidate preflight and before lifecycle mutation. Config-byte
or canonical-state drift therefore fails. If refreshed base and local layers
offset one another without changing the effective snapshot, the candidate is
rebuilt and preflighted from that refreshed pair.

Execute copies validated source-config bytes to the next round's
`source_config.yaml` and initially materializes from the validated in-memory
proposal without re-reading the mutable config. Once an accepted suggestion is
published, its exact candidate bytes remain authoritative while an unregistered
round is recovered, even if the mutable source runtime commit advances. Recovery
rebuilds and compares every other candidate field before reusing that commit.
Validation resolves the specific issued snapshot rather than `_latest_digest`,
so later digest refreshes do not invalidate otherwise unchanged evidence. Failed
uncommitted launch attempts are never reused; a later request may bind the same
terminal source round to a higher fresh target round.

After a target round is canonically committed, repeating the exact same
`hparam-adaptive-step --proposal <path> --execute` is idempotent. The retry
returns the already-published suggestion only when the immutable input and
proposal, acceptance artifact and event, suggestion hash, registered plan, and
`launch_round` evidence all agree, and the original command wrote its successful
completion event after launch and replacement handling finished. It does not
stage, register, emit, or launch again. A preview remains subject to the live
round binding, while an incomplete target registry or any uncommitted,
conflicting, or uncertain launch state continues to fail closed.
Acceptance, launch, and completion events must occur in that order. Replaying
an older proposal also requires every later committed agent-proposal round to
have the same ordered completion evidence and no canonical `launch_failed` row.

A proposal changes only the search space and submits exactly one of:

- `parameters`: the complete per-key candidate mapping, budgeted by Cartesian
  product;
- `configurations`: complete joint points, budgeted by point count, with every
  point covering exactly the snapshot keys, satisfying all envelopes, and
  remaining unique.

The submission must cite evidence run IDs from the issued snapshot and fit both
`round_size` and the remaining total-run budget. Use joint configurations for
specific paired settings; per-key values instead authorize their Cartesian
product. During phase two, do not regenerate a digest, choose a latest digest,
or invoke `hparam-suggest`. Changed config bytes or canonical workflow evidence
require a fresh issued snapshot, not editing the old input or reusing an
uncommitted target round.

Snapshots do not encode expansion mode, so either shape may answer any
snapshot. Task, variant, data, objective, budget, execution identity,
replacement policy, commands, and run state remain tool-owned. Direct
`hparam-suggest` and `hparam-adaptive-loop` do not support `agent_proposal`; the
external agent drives the handshake through `hparam-adaptive-step`.
