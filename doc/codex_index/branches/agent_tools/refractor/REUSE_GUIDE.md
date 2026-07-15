# Reuse Guide

## Canonical Implementations

| Need | Reuse | Avoid |
|---|---|---|
| Load an authored recipe and optional base | `recipes.load_recipe_with_base` | Command-specific YAML loaders or merge rules |
| Close task-recipe fields | Existing owners reached by `plans.evaluate_recipe`: `experiment_metadata_issues`, `consultation_contract_issues`, task/hparam/execution rules, and artifact checks | A schema registry, reflection scan, or second recipe facade |
| Parse authoritative YAML | `experiment_workspace.read_managed_yaml_mapping` | `yaml.safe_load` at managed ownership boundaries |
| Obtain lightweight config facts | `configs.config_summary` | Importing training models or inferring semantics from filenames |
| Summarize index/preset state | `index_csv.index_summary`, `presets.preset_summary` | Shell-only CSV/pickle probing |
| Resolve consultation status | `decisions.evaluate_consultation_gates` | Duplicated task policy in templates or renderers |
| Prove a plan is safe without writes | `plans.preflight_plan` | Calling `build_plan` speculatively or recreating partial checks |
| Generate a managed plan | `plans.build_plan` | Writing executable commands directly from recipes |
| Route a task to a command | `plans._commands_for_recipe` plus `plan_rendering` | A second dispatcher in CLI, skills, or adaptive code |
| Render runtime flags | `plan_rendering.runtime_cli_args`, `infer_runtime_cli_args`, task-specific argument helpers | Repeated flag maps in plans or postprocessing |
| Initialize/bind a workspace | `experiment_workspace.ensure_experiment_workspace` | Ad hoc directory trees or inferred ownership |
| Read/update canonical run state | `read_run_manifest`, `merge_run_manifest` | Treating status/ranking/W&B tables as authoritative |
| Validate managed output topology | `experiment_io.validate_managed_output_paths` | Per-command symlink/hard-link/containment checks |
| Read strict local/remote tables | `experiment_io.read_rows_at` | Separate SSH TSV parsers |
| Consume a frozen hparam plan | `run_artifacts.read_hparam_plan` | Reading `plan.json` alone or trusting mutable recipe input |
| Locate runtime evidence | `run_artifacts.find_run_manifest` and `run_evidence` | Recursive artifact discovery or log-name inference |
| Launch/drain/monitor/stop hparam runs | `launch_hparam_runs`, `run_hparam_queue`, `monitor_hparam_runs`, and `stop_hparam_run` from `hparam.py` | Invoking private launch shell fragments or adding scheduling to monitor |
| Manage experiment lifecycle | public functions in `experiments.py` | Mutating `experiment.yaml` or tables directly |
| Compose adaptive rounds | `adaptive_hparam` | A separate tuning registry or plan compiler |
| Expose long-running progress | `progress.write_progress` / `read_progress` | New status filenames or unbounded remote log polling |

## High-Risk Reuse Hotspots

### Recipe and decision changes

- Put validation in the owner that consumes the field: experiment/step metadata in `experiment_workspace`, path behavior in `decision_paths`, ordinary task rules in `decision_rules`, hparam/search/adaptive rules in `decision_hparam`, artifacts and orchestration in `plans`.
- Reuse `plan_rendering.FINETUNE_RUNTIME_FIELDS`, `INFER_RUNTIME_FIELDS`, `SLEEP2STAT_RUNTIME_FIELDS`, and `PRESET_FIELDS`; they are derived beside the field-to-flag mappings that consume accepted values.
- Keep `decisions.py` as the aggregate facade. Do not create a general schema registry, a parallel recipe object model, or task-specific validator facades unless the existing owner cannot express the current contract.
- Preserve `load_recipe_with_base` provenance and merge behavior. Base, local, and effective recipe views have different purposes; do not collapse them into a second loader API.
- Consume frozen hparam recipes through `run_artifacts.read_hparam_plan`; do not pass trusted `_base_recipe`/`_local_recipe` artifacts back through the authored-input loader.
- User decisions should materialize through `_materialize_decisions` before consultation, not by rewriting YAML externally.

### Planning and rendering

- `preflight_plan` is the reusable no-write gate. Any adjacent workflow that would create a directory, stop/supersede a run, or launch a process must call it first.
- `_commands_for_recipe` is the finite task dispatcher. Put reusable quoting and flag construction in `plan_rendering`; put hparam grid/freeze/final-test compilation in `plan_hparam`.
- Generated commands must call canonical package entrypoints and respect recipe `variant`. `sleep2stat` stays variantless.
- `script_lines` owns repository cwd/PYTHONPATH and optional managed lifecycle wrapping. Do not duplicate its shell prelude.

### Workspace and evidence

- Canonicalize local management roots at public boundaries, but do not normalize semantic dataset/checkpoint paths that contracts say to pass through.
- Use `read_managed_yaml_mapping` for authoritative YAML because it rejects unsafe duplicate-key/alias shapes. Use `read_step_manifest` and `merge_step_manifest` for every step producer.
- `run_manifest.tsv` is canonical. Writers produce source-allowlisted observations and commit through `merge_run_manifest`; reports and matrices use the returned canonical rows.
- Verify ownership before filtering external evidence. W&B, checkpoint, metrics, ranking, adaptive registry, and status rows must match a managed `(step_id, run_id)` and frozen fields.
- `read_hparam_plan` validates plan, recipe, workspace, row identity, frozen paths, and hashes. Every hparam mutator must cross it.
- `plan_hparam.freeze_hparam_execution` is the reusable target-identity owner. Only a local `REPO_ROOT` target without a conda wrapper may default to the manager interpreter and manager HEAD; SSH, separate-workdir, and conda-wrapped targets must author Python and full commit identity.
- Adaptive initialization calls that owner once, stores only Python/commit as workflow identity, and overlays them onto later mutable source recipes. Do not freeze capacity, GPU, environment, or other operational execution fields at workflow scope, and do not resolve manager HEAD again for later rounds.
- `run_hparam_queue` is the explicit full-plan scheduler and composes observation with repeated locked launch waves. Keep `monitor_hparam_runs` observation-only. Execute-mode launch owns `execution_snapshot.json`, probes only for an eligible launch, and must bind the planned Python/commit and exact CLI argv to a module origin inside the verified repository. Each actual start must reuse the same target wrapper to recheck identity/import safety plus the selected run's script/config hashes immediately before `nohup`. Plans missing frozen identity or a post-start snapshot are recreated rather than upgraded.
- Capacity accounting across plans belongs to the locked launch owner. It refreshes relevant observable blockers before slot calculation; queue mode requests explicit failure when a current or cross-plan blocker is `missing_pid` instead of adding an external polling loop.
- `read_pid` can prove process identity against the frozen script. Stop operations must not signal a PID inferred only from a filename.

## Duplication Traps

- A second YAML parser or base-recipe merger.
- A second task-to-module or task-to-command switch.
- Global field allowlists disconnected from actual renderers/owners.
- A parallel run status reducer or lifecycle table.
- Recursive adoption of historical or unmanaged run directories.
- Compatibility readers for removed `trial_*` artifacts.
- A second local/remote file abstraction.
- Metrics selected from test/external data without the existing unlock and selection gates.
- Hparam postprocessing that bypasses frozen selection metric, checkpoint ownership, or final-test lock.

## Change Checklist

1. Identify the existing responsibility owner in [MODULE_MAP.md](./MODULE_MAP.md).
2. Search this guide and [FUNCTIONS/AGENT_TOOLING.md](./FUNCTIONS/AGENT_TOOLING.md) for an implementation to reuse.
3. Keep authored-input validation before config reads and workspace mutation when it affects plan safety.
4. Extend the smallest owner-focused test file; add observable renderer/runtime assertions for accepted fields.
5. Run targeted tests, then `tests/agent_tools`, skill validation, format/lint checks, and `git diff --check`.
