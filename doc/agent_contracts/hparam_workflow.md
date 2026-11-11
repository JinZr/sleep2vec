# Hparam Workflow Contract

This contract owns the `hparam_tune` protocol: search space, registration
preflight, launch and queue, execution snapshot, selection, and further rounds.
Authored and effective recipe inputs, consultation, and
task routing belong to [task_recipe.md](task_recipe.md); accepted fields belong
to the [task recipe schema](../../recipes/schemas/task_recipe.schema.md).
Workspace publication and finalization belong to
[experiment_workspace.md](experiment_workspace.md); canonical lifecycle and
scheduler evidence belong to [run_manifest.md](run_manifest.md).

## Contents

- [Search space](#search-space), [registration preflight](#registration-preflight), and [launch/queue](#launch-and-queue)
- [Execution snapshot and launch revalidation](#execution-snapshot-and-launch-revalidation)
- [Selection and selected-candidate consumers](#selection-and-selected-candidate-consumers)

## Ordinary hparam workflow

Search-space generation does not choose or revoke scientific test policy; read
[test-access policy](external_test_locking.md#selection-and-test-access-policy)
before selecting a search. The selection split, metric, and mode must come from
explicit user authorization, not an agent inference relabeled as `explicit_recipe`.

### Search space

- Search keys are explicit `runtime.<name>` fields or
  `yaml:/json/pointer/path` config overrides. Removed bare or `param.*` forms
  are rejected rather than translated. JSON Pointer escapes use only `~0` and
  `~1`. List indices use canonical non-negative integers (`0`, `1`, ...), with
  no leading zero, sign or whitespace aliases; numeric dictionary keys remain
  literal strings.
- An explicit search requires positive `search.max_runs` and uses `method: grid`.
  It declares exactly one of these mutually exclusive shapes:
  - `search.parameters`: a per-key candidate mapping expanded by Cartesian
    product, with axes in lexicographic full-key order and each candidate
    value list in its authored order;
  - `search.configurations`: complete joint configuration points expanded
    in their authored list order, one run per point. Mapping key order does
    not affect a point's run name or parameter summary.
- Both shapes use the same key rules and apply `[:max_runs]` prefix truncation
  after expansion. Reordering
  parameter mapping keys does not change run ids or the selected grid prefix. A
  concrete `hparam_search_space` decision replaces the authored search shape,
  including any configurations.
- For `sleep2vec`, `sleep2vec2`, `sleep2expert`, and `sex_age_baseline`,
  managed finetune runtime fields and explicit search keys include
  `lr_scheduler`, `lr_decay_floor`, `lr_decay_shape`, `lr_decay_ratio`,
  `lr_plateau_factor`, and `lr_plateau_patience` under `runtime`. They retain the training CLI meanings:
  `decay|wsd|plateau`, a per-group relative floor, `cosine|linear`, a WSD-only
  final decay fraction, and Plateau-only factor/patience. Use complete joint
  configurations with `null` resets for inapplicable fields when switching
  scheduler families. Candidate validation checks each effective combination
  before publishing a runnable plan; WSD's exact phase lengths additionally
  require the training runtime's optimizer-update count.
- A `yaml:/...` value may be a scalar or a complete mapping or list block. An
  override replaces the whole addressed value, so replacing `finetune.tuning`
  removes source `groups` overrides absent from the selected block. Every
  generated config still passes the canonical variant loader; a block value does
  not authorize arbitrary new fields inside it or bypass config semantics.

Planning validates the concrete points through canonical config and runtime
preflight before publication or launch. Coupled axes may have valid paired
points and invalid cross-pairs, so author a `parameters` grid only when its whole
Cartesian product is valid; otherwise use explicit `configurations`.

An ordinary tuning request uses a static `hparam_tune` plan. Existing authored
or frozen searches must not be rewritten. The agent authors either explicit
`search.configurations` (complete, unique points) or a `search.parameters` grid
after reviewing the effective source config, available prior results, useful
fixed/search axes and whether the incumbent sits at the edge of the searched
values; no compiler chooses the points. A
source value is not by itself a reason to freeze a technical axis, and a
concurrency cap does not fix epochs. See the
[result-to-proposal reasoning](../../skills/hyperparameter_tuning/SKILL.md#result-to-proposal-reasoning).

If the user did not specify a total budget, use the default search budget in
[`hparam_search_defaults`](../../agent_policies/consultation_policy.yaml) across
all rounds. It is an agent authoring default, not a parser fallback: write each
plan's explicit `search.max_runs` before consultation. Prefer explicit
`search.configurations` to spend each round on informative comparisons.
Templates are starting examples whose points must be adjusted to the actual
experiment.

A further round is a new ordinary plan with a fresh `--output-dir`; the frozen
plans of earlier rounds are never rewritten. Keep it in the same experiment and
step, so `hparam-select` ranks every round of the step together under one
selection policy. Author its own configurations or grid and keep the
cumulative run count within the total budget. Record the evidence for the round first with an `experiment-note` entry
that cites the prior results it relies on.

`inputs.ckpt_path` is final-evaluation-only, not a tuning backbone.

The tuning request alone does not authorize publication or launch. Publication
may carry the authorized scope through doctor and `plan`; launch follows the
plan/queue sequence once it is explicitly authorized. This does not authorize
test unlock, changed label/split/checkpoint/data, or expansion beyond the
authorized total budget. Report only the best observed candidate among the
searched configurations, metric, split and budget, never a global optimum.

### Full-model weights-only initialization

Optional `inputs.init_checkpoint_path` is supported for `sleep2vec2` finetune
and hparam recipes. Supply an absolute full finetune checkpoint path and publish
locally on its execution host (`execution.target: local`, including Slurm).
The plan freezes its SHA-256 in the resolved recipe and each run's input
snapshots; registered-plan reads and the leaf script verify those bytes.
`inputs.ckpt_path` remains final-evaluation-only for hparam and is not passed
to training. Do not combine initialization with a pretrained-backbone input
or a direct runtime resume checkpoint. An explicit null pretrained-backbone
decision means no backbone-only initialization, not scratch training when this
full-model source is present.

`--init-checkpoint-path` loads the complete Lightning state dictionary strictly,
including backbone, trained head, LayerMix and buffers; missing/extra keys fail.
It requires the saved model configuration (including tokenizer semantics) and
task, LayerMix, loss, covariate and enabled LoRA configurations to match. Trainer epoch,
optimizer, scheduler and callbacks are not restored. The new trainability
policy and runtime schedule apply. Source checkpoint task metadata does not
prove arbitrary same-shaped label identity; the recipe author must retain the
authorized label. Structural model/head/LayerMix changes are unsupported; use an
explicit compatible search.
Doctor and registration do not construct the model or certify checkpoint
semantic compatibility; the runtime load performs that check.

### Registration preflight

New hparam plans are fully materialized in a
temporary directory on the final destination filesystem. The staged frozen
bundle is recompiled and validated with the same reader used after publication;
each final candidate config is then checked with the recipe variant's canonical
`load_finetune_config` and `validate_model_config` in the planner's current Python
and code environment. This applies equally to explicit configurations and
parameter grids after all YAML overrides. Consultation checks captured input bytes; registration independently
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
event.

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
dataset validation; those operations are not performed.
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
  the allocation's whole-node memory limit. `sex_age_baseline` uses the same
  allocation-local device contract for DDP training and inference.
  Registration preflight reports both each job's resources and the aggregate
  request across all actually planned allocations if active together. This
  deterministic plan report uses the frozen run count and does not query live
  node capacity; live fixed-node feasibility remains a `doctor` diagnostic.
  Neither report changes scheduler resources, and
  Slurm launch still submits every frozen leaf job for scheduler queuing.
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
Python callers may pass `managed_scheduler.SchedulerHooks` to
`hparam_runtime.launch_hparam_runs` to replace only the execution-snapshot
check, launch-command, and process-start effects; the CLI has no such option,
and manifest merges, events, status reports, and frozen-run validation are
unchanged.

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
moving best/last alias. When no fixed checkpoint exists for the best epoch, for
example after early stopping before the first periodic save, a completed or
skipped-test run manifest binds the `best-epoch=` file it names; selection
freezes its SHA-256 like any other checkpoint.

Selection binds the winner checkpoint path/SHA-256 and the selection report's
path/hash in the canonical manifest. The report records metric/mode/split,
evaluated count, winner run/checkpoint/score, parameters, search overrides,
frozen config/script paths/hashes, and ranking path. Runtime evidence shape
remains owned by [the run manifest contract](run_manifest.md#runtime-artifact-evidence).
The test split is never a selection split; recipe validation rejects
`selection_split: test`.

The selected winner's frozen checkpoint path is the only selected-candidate
input to [final external-test evaluation](external_test_locking.md#final-external-test-boundary);
there is no other selected-candidate consumer.

Status validates the canonical ranking; finalization rehashes every ranked
checkpoint on its frozen target.
The detailed [status read-set](experiment_workspace.md#read-only-status-and-advisory-actions)
and [final report acceptance](experiment_workspace.md#finalization) have one
workspace owner, including mixed/all-failed cases and historical completed-read
compatibility. Final test authorization belongs to
[external_test_locking.md](external_test_locking.md).
