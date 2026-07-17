# Task Recipe Contract

Task recipes under `recipes/` bind one task to an experiment and step. The accepted fields and finite allowlists are defined in the [task recipe schema](../../recipes/schemas/task_recipe.schema.md).

## Authored-input closure

Recipe shape is validated before config inspection and before any workspace, script, manifest, or event is created. Validation is task-aware and owner-based: `experiment_workspace` owns `experiment` and `step`; task decision owners own `inputs`, `evaluation_policy`, `execution`, `search`, and `adaptive`; renderer mappings own runtime and preset CLI fields; and `plans` owns top-level routing and artifacts. This does not add a second schema registry or general recipe facade.

Unknown and task-inapplicable fields fail with their original field path and source layer. For hparam recipes, the base finetune source and local tuning overlay are validated independently before merged semantics run. Raw authored `_...` fields are reserved and rejected.

## Effective recipe

Planning produces one effective recipe:

```text
recipe fields + recipe decisions + explicit user decisions
  -> materialized recipe
  -> config summary and consultation
  -> frozen plan and resolved recipe
```

Recipe decisions with a task-owned canonical field are written into that field first, then explicit user decisions may override both the canonical field and effective decision mapping before config inspection and consultation are rerun. A layered hparam recipe takes its task only from the local overlay or explicit user decision; the base finetune task cannot become the effective tuning task. Policy-only decisions remain under `decisions` rather than creating inert recipe sections. For finetune and hparam tasks, an explicit `required_channels` decision must match `preset_build.required_channels` in the selected config. For hparam, `inputs.ckpt_path` is reserved for the selected final-evaluation checkpoint and is not rendered into tuning finetune commands. Empty or null rendered decisions remain unresolved instead of falling back to older canonical values; explicit `pretrained_backbone_path: null` retains its established train-without-pretraining meaning. `plan.json` and `recipe.resolved.yaml` must contain the same complete effective recipe. Retained base/local recipe copies are source audit only; launch, selection, adaptive, and postprocess consumers read the effective recipe. Frozen recipes containing trusted `_base_recipe` or `_local_recipe` metadata are consumed through `run_artifacts.read_hparam_plan`; they are not re-entered through the authored-recipe loader.

Decision-file behavior and precedence belong to [user_decisions.md](user_decisions.md).

## Experiment binding

Every runnable recipe declares complete `experiment` metadata (`id`, `title`, `objective`, `root`, and `baseline`) and a `step` (`id`, `phase`, and `purpose`). A hparam recipe declares its own binding rather than inheriting it from its base finetune recipe.

The plan directory must be inside `experiment.root`. Workspace layout, path canonicalization, step registration, and lifecycle ownership belong to [experiment_workspace.md](experiment_workspace.md).

## Task and variant routing

| task | accepted variant | generated runtime |
| --- | --- | --- |
| `sleep2stat` | omitted or `null` | `python -m sleep2stat` |
| `preset_prepare` | `sleep2vec`, `sleep2vec2`, `sleep2expert` | package-local `preprocess/save_dataset_presets.py` |
| `finetune`, `hparam_tune` | `sleep2vec`, `sleep2vec2`, `sleep2expert`, `sex_age_baseline` | `<variant>.finetune` |
| `infer`, `evaluate` | `sleep2vec`, `sleep2vec2`, `sleep2expert`, `sex_age_baseline` | `<variant>.infer` |

Preset preparation routes each variant to its package-local script. Variant scripts reject root-only `manifest_output` and `write_sidecar_manifest` fields instead of falling back to the root runtime. `sex_age_baseline` does not own preset generation.

`pretrain` and `adapt` have direct runtime skills and CLIs but are not runnable task-recipe values because agent tools have no renderer for them. Missing or unsupported routing blocks command generation.

Runnable non-hparam scripts enter `REPO_ROOT` for cwd and PYTHONPATH. User-authored semantic dataset and checkpoint paths retain their supplied meaning.

## Hparam workflow

Search keys are explicit `runtime.<name>` fields or `yaml:/json/pointer/path` config overrides. `search.max_runs` is a required positive budget. Removed bare or `param.*` forms are rejected rather than translated. The search space is either `search.parameters` (a per-key candidate mapping expanded by Cartesian product) or `search.configurations` (an explicit list of joint configuration points, each a complete `{key: value}` mapping expanded verbatim, one run per point) -- the two are mutually exclusive. Both shapes use the same key rules and the same `[:max_runs]` prefix truncation, and `method` remains `grid`. Adaptive source recipes must declare `search.parameters` (it is the envelope and neighborhood source for both suggest strategies); `search.configurations` appears in derived round recipes and static plans only.

The optional `execution` block configures the managed launcher. `execution.python` names the target Python command and `execution.runtime_commit` names the full expected Git commit. They may be omitted only for the canonical manager runtime: a local target at `REPO_ROOT` without a conda wrapper. In that case planning freezes the current manager interpreter and manager repository HEAD. SSH targets, separate local workdirs, and conda-wrapped targets must author both values explicitly. `hparam-launch` explicitly starts one capacity-limited wave, while `hparam-run-queue --execute` uses monitor observations and the same locked launch owner until the current plan is terminal. Generated leaf scripts use only `execution.workdir` on `PYTHONPATH`, and `execution.env.PYTHONPATH` is rejected rather than silently merged. The first eligible execute atomically records the verified target Python/version, host identity, runtime repository root and clean expected commit, repository-owned runtime module origin, explicit-environment digest, normalized supported-option digest, and exact validated argv digest in `execution_snapshot.json`; later launch waves must match it. Immediately before each `nohup`, the same target/env/conda/PYTHONPATH wrapper rechecks Python/version, commit, repository root, hostname, module origin, untracked or ignored importable code, and that run's frozen script/config hashes. Plans lacking frozen `execution.python` or `execution.runtime_commit` must be recreated rather than upgraded in place. Frozen per-run execution identity and its canonical owner are defined in [run_manifest.md](run_manifest.md).

The optional `adaptive` block defines append-only rounds bounded by `adaptive.max_runs_total`. Test/external objectives require explicit test feedback authorization. Initialization resolves `execution.python` and `execution.runtime_commit` once for round 000 and stores only those two fields as workflow-wide execution identity. Later rounds re-read the mutable source recipe, reject conflicting identity, and carry the frozen identity forward without resolving the manager interpreter or HEAD again. Other execution fields remain source-controlled, so capacity and environment settings such as `max_concurrent`, GPU allocation, and `env` may change between rounds subject to normal preflight; each resulting round plan remains immutable. The current source recipe and each generated suggestion must pass read-only preflight before digest, suggestion, or event artifacts are written. Earlier round plans, configs, logs, and checkpoints are not rewritten.

`adaptive.suggest.strategy` is `best_neighborhood` by default and otherwise accepts only `agent_proposal`. The default strategy preserves the existing active-round replacement flow. `agent_proposal` must explicitly declare `adaptive.objective_metric` as a non-blank string and non-empty values for `adaptive.objective_mode`, `adaptive.round_size`, `adaptive.max_rounds`, and `adaptive.max_runs_total`; omitted, null, or blank required values stop consultation before workspace mutation, while a non-string objective metric fails the recipe contract. It is terminal-only, so `adaptive.replacement` must be omitted or exactly `{enabled: false}`; replacement fields must not be silently ignored. Its optional `adaptive.suggest.bounds` mapping may authorize a wider or narrower closed interval for numeric search parameters. Bounds keys must be a subset of `search.parameters`; integer-valued grids require integer endpoints, numeric grids containing a float accept finite integer or float endpoints, and categorical or mixed grids do not accept bounds. An unbounded numeric parameter falls back to the original candidate minimum and maximum, while categorical proposals remain a subset of the original choices.

Agent proposals use a two-phase file handshake. A proposal-free `hparam-adaptive-step` monitors and digests the current round, returns `waiting_for_round_terminal` while any run is active, and otherwise writes an immutable `adaptive/proposal_inputs/round_NNN--<id12>.json` evidence snapshot. Proposal-input schema v2 requires `input.source_config_sha256`, the SHA-256 of the exact `inputs.config` file bytes. The canonical request id binds that hash together with the complete digest rows, source recipe and frozen execution identity, remaining budget, and parameter envelopes. The tool records an `agent_proposal_requested` issuance containing the request id, source and target rounds, exact input and submission paths, and the complete snapshot-file hash. Proposal-input v1 is not accepted and must be regenerated; proposal submissions remain schema v1.

An external agent may write only the exact `adaptive/proposal_submissions/round_NNN--<id12>.json` path named by the tool-issued snapshot; it must not create or replace proposal inputs or tool lifecycle state. `hparam-adaptive-step --proposal <path>` previews without lifecycle mutation, and adding `--execute` applies the proposal through the normal preflight, registration, and launch path. Phase two requires an unambiguous matching issuance and exact snapshot bytes before trusting its bounds or budget, then reconstructs the complete input from current canonical recipe, workflow, round, manifest/registry, and runtime evidence and requires it to match the snapshot. This live validation is repeated after candidate preflight and before lifecycle mutation, so config-byte or canonical-state drift is rejected. Execute copies the validated source-config bytes to the new round's `source_config.yaml` and materializes the round recipe from the already validated in-memory proposal; it does not re-read the mutable source config or suggestion. Submission validation resolves the specific issued snapshot rather than consulting `_latest_digest`; later digest refreshes do not invalidate otherwise unchanged canonical evidence. Failed uncommitted launch attempts are never reused, so a later request may bind the same terminal source round to a higher fresh target round. A proposal changes exactly one thing -- the search space -- and submits it as exactly one of `parameters` (the complete per-key candidate mapping, budgeted by its Cartesian product) or `configurations` (an explicit list of joint configuration points, budgeted by point count; every point must cover exactly the snapshot's parameter keys, every value must satisfy its envelope, and duplicate points are rejected). Snapshots do not encode the expansion mode, so either shape may answer any snapshot. Task, variant, data, objective, budget, execution identity, replacement policy, commands, and run state remain tool-owned. Direct `hparam-suggest` and `hparam-adaptive-loop` calls do not support `agent_proposal`; the external agent drives each handshake through `hparam-adaptive-step`.

`reports/ranking.csv` is shared across plans in the same step. Runnable hparam plans in that step must use the same selection metric and mode; selection replaces current-plan keys and reranks the complete step. Candidate ownership, frozen-field validation, and checkpoint evidence are defined in [run_manifest.md](run_manifest.md). Final external-test generation follows [external_test_locking.md](external_test_locking.md).

Managed run identity, status, atomic commit, projections, and evidence belong exclusively to [run_manifest.md](run_manifest.md).
