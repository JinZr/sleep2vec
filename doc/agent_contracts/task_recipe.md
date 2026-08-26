# Task Recipe Contract

Task recipes under `recipes/` bind one task to an experiment and step. The
accepted fields and finite allowlists are defined in the
[task recipe schema](../../recipes/schemas/task_recipe.schema.md).

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

## Non-hparam inference runtime identity

Only `infer` / `evaluate` accept `execution.python` and
`execution.runtime_commit`. Declaring either turns the otherwise-common
`execution.workdir` into an all-or-none local/default-local runtime identity.
Python is one executable name or path without whitespace, arguments, or `~`
shorthand; the commit is a lowercase 40-character Git commit SHA. Other
non-hparam tasks reject Python and commit identity rather than silently
rendering commands that ignore them.

When the identity is present, the resolved recipe and plan freeze it. The
generated script enters that workdir, verifies its Git HEAD before the first
lifecycle mutation, and uses the same frozen Python for inference and all
`running` / `completed` / `failed` commits. A missing interpreter or commit
mismatch fails before `running` is committed and before inference starts.
`execution.target` and `execution.host` on other non-hparam tasks remain
path-validation context; they do not provide a generic SSH launcher.

## Hparam workflow

### Search space

- Search keys are explicit `runtime.<name>` fields or
  `yaml:/json/pointer/path` config overrides. Removed bare or `param.*` forms
  are rejected rather than translated.
- An explicit search requires positive `search.max_runs`, uses `method: grid`,
  and is exactly one of:
  - `search.parameters`: a per-key candidate mapping expanded by Cartesian
    product;
  - `search.configurations`: complete joint configuration points expanded
    verbatim, one run per point.
- Both shapes use the same key rules and `[:max_runs]` prefix truncation.
- `search.profile: finetune_balanced` is the alternative authored intent for
  `sleep2vec` and `sleep2vec2` finetuning labels `ahi`, `arousal`, and
  `stage4`. It is mutually exclusive with authored `parameters` or
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
  not a claim that arbitrary or unknown config fields were searched.
- The first profile version does not support adaptive tuning. Existing
  explicit searches and historical frozen plans are unchanged; profile
  expansion is not retroactively required by registered-plan readers.
- Adaptive source recipes must declare `search.parameters`, which supplies the
  envelope and neighborhood source. `search.configurations` appears only in
  derived rounds and static plans.

### Registration preflight

New ordinary hparam plans and adaptive rounds are fully materialized in a
temporary directory on the final destination filesystem. The staged frozen
bundle is recompiled and validated with the same reader used after publication;
the tool then validates managed output topology on the frozen execution host
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
card. Target Python, runtime commit, module origin, run/argv counts, and the
argv digest come from the frozen execution snapshot. Variant, runtime module,
actual config loader, architecture, and channels come from each final generated
config and are grouped with their run IDs. The card is a projection for audit;
the frozen generated config bytes and hashes remain semantic authority. This
deterministic preflight does not inspect free bytes, estimate checkpoint storage,
or turn unavailable Slurm accounting into a plan blocker; accounting capability
remains a time-stamped `doctor` diagnostic.

### Managed launcher

The optional `execution` block configures the managed launcher.

- `execution.python` names one target Python executable without whitespace,
  arguments, or `~` shorthand. `execution.runtime_commit` names the full
  expected Git commit.
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
- Slurm plans warn that priority remains cluster-managed. `doctor` may inspect
  version, priority, backfill, accounting, partition, and reservation
  capabilities through read-only `scontrol` queries. Advice never changes the
  frozen scheduler request: `nice=0` is the highest unprivileged nice setting,
  and no user-side option guarantees first priority.
- Only the canonical manager runtime—a local target at `REPO_ROOT` without a
  conda wrapper—may omit Python and commit identity. Planning then freezes the
  current manager interpreter and repository HEAD. SSH targets, separate local
  workdirs, and conda-wrapped targets must author both values explicitly.
- With direct scheduling, `hparam-launch` starts one capacity-limited wave and
  `hparam-run-queue --execute` keeps filling that capacity. With Slurm, the
  launch owner submits every launchable leaf job without applying host-global
  GPU capacity to scheduler allocations.
- Generated leaf scripts use only `execution.workdir` on `PYTHONPATH`;
  `execution.env.PYTHONPATH` is rejected rather than merged.

Registration preflight records verified Python/version, host, repository and
commit, module origin, explicit-environment digest, normalized supported-option
digest, and exact validated argv digest in `execution_snapshot.json` before a
new hparam plan is published. Every launch wave live-reprobes the same target
and must match that frozen snapshot; it also rechecks the managed output
topology before starting a process or submitting `sbatch`. Immediately before
each managed process starts, the same target/env/conda/PYTHONPATH wrapper
rechecks Python/version, commit, repository root, hostname, module origin,
untracked or ignored importable code, and the run's frozen script/config hashes.
For Slurm, the allocation wrapper also requires `SLURM_NTASKS` to match the
frozen `gpus_per_run`, then compares its observed Python executable and version
with the plan-level execution snapshot before starting the leaf script through
one `srun --kill-on-bad-exit=1 --quit-on-interrupt` child without explicit
task-level GPU binding. This preserves the complete allocated GPU visibility
expected by the frozen Lightning device list in every externally launched rank.
The launcher freezes the snapshot's raw SHA-256 in every canonical run and
passes it as a batch-script argument, so the allocation verifies the exact
snapshot bytes before parsing them. Compute-node hostname is observed evidence
and may differ from the submission host.
Plans lacking frozen Python or commit identity must be recreated rather than
upgraded in place.

Each Slurm run additionally freezes `job.sbatch`, its hash, a deterministic
submit token, log path, allocation-identity path, and terminal-sidecar path.
Only the allocation wrapper writes the two scheduler sidecars; ranks share the
Slurm log, and only global rank zero writes the diagnostic exit marker. The
terminal sidecar records the aggregate `srun` exit code.
Submission commits `submitting` before `sbatch`; a timeout or SSH disconnect is
reconciled by the exact token and is never retried blindly. Monitoring uses
`squeue`/`scontrol` for controller state, then queries the exact bound-cluster
`sacct` allocation when the job has aged out of the controller. It requires
`--clusters=<scheduler_cluster>` for bound-cluster routing unless the canonical
run freezes `scheduler_direct_controller: true`; local transport alone does not
establish direct-controller topology. New Slurm plans freeze the recipe's
`direct_controller` choice in each canonical run so all monitoring and stop
paths preserve it without reinterpreting transport. Terminal truth normally
requires both a terminal scheduler observation and the matching atomic terminal
sidecar. The narrow accounting-disabled exception additionally requires the
exact bound job to be absent from `squeue`, explicitly invalid in `scontrol`,
and `sacct` to report disabled accounting, then uses a sidecar matching the
frozen job, token, and non-empty canonical cluster, with frozen transport and
controller topology matching the actual query route, to recover exit zero as
`completed` or non-zero as `failed` while retaining raw state `MISSING`. Other
incomplete terminal evidence is `unknown_scheduler`. Stop first records the
frozen scheduler job id, nonterminal `stopping`, request time, and reason in the
canonical manifest, then uses that job id with `scancel`, not PID evidence. An
interrupted or failed cancellation keeps the request recoverable; the same
reason may retry it, while a different reason cannot overwrite it. Only a
matching scheduler `CANCELLED` observation commits canonical `stopped`, even
when cancellation prevented the wrapper from writing a terminal sidecar. A
cancellation signal received while the allocation wrapper is still validating
frozen identity terminates the job without starting the leaf process. Live
Slurm transition flags remain active; raw `STOPPED` retains its allocation and
is not canonical `stopped`. When Slurm reports raw `REVOKED` federation sibling
state, sibling-cluster rebinding is unsupported; the run fails closed as active
`unknown_scheduler`, the frozen job, cluster, scheduler reason, and stop intent
remain canonical, and the run is not relaunched.
The submission command strips every ambient `SBATCH_*` variable on the local or
SSH submission host before invoking `sbatch`, while preserving ordinary runtime
environment such as `PATH` and Slurm client configuration.

Frozen per-run execution identity and its canonical owner are defined in
[run_manifest.md](run_manifest.md).

### Adaptive rounds and strategy

The optional `adaptive` block defines append-only rounds bounded by
`adaptive.max_runs_total`.

An authored recipe with `adaptive.enabled=true` must enter through
`hparam-adaptive-init`. The generic `plan` command rejects it before workspace
mutation; only the adaptive workflow owner may materialize round plans.

- Control flags must be YAML booleans; run, round, and poll budgets must be
  positive YAML integers; replacement grace and margin values must be finite
  and non-negative.
- Test or external objectives require explicit test-feedback authorization.
- When `selection_split=test`, adaptive `test_*` objectives—including a `test_*` objective distinct from the frozen
  selection metric—reduce the complete `checkpoint_test_results` by the frozen objective mode and bind the selected
  checkpoint path and epoch. Only canonically `completed` or `finished` runs are eligible for checkpoint-objective
  ranking and incumbency. Validation/run-level objectives such as `val_*` and `best_model_score` retain top-level
  evidence. A running test-checkpoint objective cannot trigger metric-based retirement, while independent log failures can.
- Initialization resolves `execution.python`, `execution.runtime_commit`,
  `adaptive.objective_metric`, and `adaptive.objective_mode` once for round 000
  and stores them as workflow-wide frozen values.
- Later rounds re-read the mutable source recipe, reject conflicting execution
  identity or objective values, and carry the frozen values forward. Other
  operational execution fields, including concurrency, GPU allocation, and
  `env`, remain source-controlled subject to normal preflight. Each round plan
  remains immutable.
- The source recipe and every suggestion pass read-only preflight before digest,
  suggestion, or event artifacts are written. Earlier round plans, configs,
  logs, and checkpoints are not rewritten.

`adaptive.suggest.strategy` defaults to `agent_proposal`; the only other value
is explicit `best_neighborhood`. An enabled proposal workflow requires a
non-blank string `adaptive.objective_metric` and non-empty
`adaptive.objective_mode`, `adaptive.round_size`, `adaptive.max_rounds`, and
`adaptive.max_runs_total`. Missing, null, or blank required values stop
consultation before workspace mutation; a non-string objective metric fails the
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

### Agent proposal handshake

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
`source_config.yaml` and materializes from the validated in-memory proposal; it
does not re-read the mutable config or suggestion. Validation resolves the
specific issued snapshot rather than `_latest_digest`, so later digest refreshes
do not invalidate otherwise unchanged evidence. Failed uncommitted launch
attempts are never reused; a later request may bind the same terminal source
round to a higher fresh target round.

A proposal changes only the search space and submits exactly one of:

- `parameters`: the complete per-key candidate mapping, budgeted by Cartesian
  product;
- `configurations`: complete joint points, budgeted by point count, with every
  point covering exactly the snapshot keys, satisfying all envelopes, and
  remaining unique.

Snapshots do not encode expansion mode, so either shape may answer any
snapshot. Task, variant, data, objective, budget, execution identity,
replacement policy, commands, and run state remain tool-owned. Direct
`hparam-suggest` and `hparam-adaptive-loop` do not support `agent_proposal`; the
external agent drives the handshake through `hparam-adaptive-step`.

### Ranking and final evaluation

`reports/ranking.csv` is shared across plans in the same step. Runnable hparam
plans in that step must use the same selection metric and mode. Selection
replaces current-plan keys, reranks the complete step, and writes deterministic
`reports/hparam_selection.md`. The canonical manifest binds that report's path
and hash to the selected rows; the report records metric/mode/split, evaluated
count, winner run/checkpoint/score, parameter summary, search overrides, frozen
config/script paths and hashes, and ranking path. For test-selected tuning,
canonical rows additionally bind each registered plan's complete
`checkpoint_test_ranking.csv` path and SHA-256. Status validates every bound
audit and reconstructs the global checkpoint/run ranking; finalize rehashes
every audited checkpoint on its frozen execution target. `experiment-status`
therefore advances a terminal ordinary hparam step to `ready_to_select`; a
successful selection normally writes the deterministic report and advances it
to `ready_to_finalize`, while a missing or invalid derived report/ranking is
`ready_to_report`. A verified selection report may serve directly as the final
report only when every ordinary materialized plan in the experiment is a hparam
plan and every hparam step has a selected winner. Mixed experiments and partly
failed multi-step searches require a separate non-empty combined report.
All-failed hparam steps skip selection and require a non-empty failure report;
the canonical selection report cannot substitute for either report type.
Historical completed experiments are not retroactively required to carry the
new selection-report binding.

Candidate ownership, frozen-field validation, checkpoint evidence, managed run
identity, status, atomic commit, and projections belong to
[run_manifest.md](run_manifest.md). Final external-test generation follows
[external_test_locking.md](external_test_locking.md); managed multi-source
external matrices also follow [experiment_pipeline.md](experiment_pipeline.md).
