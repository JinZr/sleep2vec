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
- For hparam, `inputs.ckpt_path` is reserved for the selected final-evaluation
  checkpoint and is not rendered into tuning finetune commands.
- Empty or null rendered decisions remain unresolved instead of falling back
  to older canonical values. Explicit `pretrained_backbone_path: null` retains
  its established train-without-pretraining meaning.

`plan.json` and `recipe.resolved.yaml` must contain the same complete effective
recipe. Retained base/local recipe copies are source audit only; launch,
selection, adaptive, and postprocess consumers read the effective recipe.
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

Preset preparation routes each variant to its package-local script. Variant
scripts reject root-only `manifest_output` and `write_sidecar_manifest` fields
instead of falling back to the root runtime. `sex_age_baseline` does not own
preset generation.

`pretrain` and `adapt` have direct runtime skills and CLIs but are not runnable
task-recipe values because agent tools have no renderer for them. Missing or
unsupported routing blocks command generation.

### Runtime paths and data inputs

Runnable non-hparam scripts use an explicit absolute `execution.workdir` for
cwd and PYTHONPATH, otherwise `REPO_ROOT`. Relative runtime-semantic dataset
and checkpoint paths are validated from that same cwd while their authored
strings remain unchanged. Runtime `~` home-directory shorthand is rejected;
use an absolute or workdir-relative path.

Local relative `inputs.config` values remain planning-source locators under
`REPO_ROOT`. The planner freezes their bytes and gives the runtime a plan-local
absolute config path.

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
- `search.max_runs` is a required positive budget, and `method` remains `grid`.
- The search space is exactly one of:
  - `search.parameters`: a per-key candidate mapping expanded by Cartesian
    product;
  - `search.configurations`: complete joint configuration points expanded
    verbatim, one run per point.
- Both shapes use the same key rules and `[:max_runs]` prefix truncation.
- Adaptive source recipes must declare `search.parameters`, which supplies the
  envelope and neighborhood source. `search.configurations` appears only in
  derived rounds and static plans.

### Managed launcher

The optional `execution` block configures the managed launcher.

- `execution.python` names one target Python executable without whitespace,
  arguments, or `~` shorthand. `execution.runtime_commit` names the full
  expected Git commit.
- Conda wrapping belongs in `execution.conda_env`, not `execution.python`.
- Only the canonical manager runtime—a local target at `REPO_ROOT` without a
  conda wrapper—may omit Python and commit identity. Planning then freezes the
  current manager interpreter and repository HEAD. SSH targets, separate local
  workdirs, and conda-wrapped targets must author both values explicitly.
- `hparam-launch` starts one capacity-limited wave.
  `hparam-run-queue --execute` uses monitor observations and the same locked
  launch owner until the current plan is terminal.
- Generated leaf scripts use only `execution.workdir` on `PYTHONPATH`;
  `execution.env.PYTHONPATH` is rejected rather than merged.

The first eligible execute atomically records verified Python/version, host,
repository and commit, module origin, explicit-environment digest, normalized
supported-option digest, and exact validated argv digest in
`execution_snapshot.json`. Later launch waves must match it. Immediately before
each managed process starts, the same target/env/conda/PYTHONPATH wrapper
rechecks Python/version, commit, repository root, hostname, module origin,
untracked or ignored importable code, and the run's frozen script/config hashes.
Plans lacking frozen Python or commit identity must be recreated rather than
upgraded in place.

Frozen per-run execution identity and its canonical owner are defined in
[run_manifest.md](run_manifest.md).

### Adaptive rounds and strategy

The optional `adaptive` block defines append-only rounds bounded by
`adaptive.max_runs_total`.

- Control flags must be YAML booleans; run, round, and poll budgets must be
  positive YAML integers; replacement grace and margin values must be finite
  and non-negative.
- Test or external objectives require explicit test-feedback authorization.
- Initialization resolves `execution.python` and `execution.runtime_commit`
  once for round 000 and stores them as workflow-wide execution identity.
- Later rounds re-read the mutable source recipe, reject conflicting identity,
  and carry the frozen identity forward. Other operational execution fields,
  including concurrency, GPU allocation, and `env`, remain source-controlled
  subject to normal preflight. Each round plan remains immutable.
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
replaces current-plan keys and reranks the complete step.

Candidate ownership, frozen-field validation, checkpoint evidence, managed run
identity, status, atomic commit, and projections belong to
[run_manifest.md](run_manifest.md). Final external-test generation follows
[external_test_locking.md](external_test_locking.md); managed multi-source
external matrices also follow [experiment_pipeline.md](experiment_pipeline.md).
