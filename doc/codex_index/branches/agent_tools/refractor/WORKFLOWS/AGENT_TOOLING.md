# Agent Tooling Workflows

## 1. Doctor and Recipe Consultation

1. `cli._cmd_doctor` calls `plans.evaluate_recipe`.
2. `recipes.load_recipe_with_base` reads the local recipe and optional base overlay while reserving authored `_...` names.
3. `plans` validates top-level/artifact fields and dispatches each source layer to the existing experiment, decision, execution, hparam, and renderer-backed field owners.
4. Closure failure returns before config reads; issue evidence preserves the original field path and `base`, `local`, `effective`, or `user` source.
5. `_materialize_decisions` applies recipe and user decisions only to canonical fields owned by the resolved task; policy-only evidence remains under `decisions`.
6. `configs.config_summary` and `plan_context` gather only the facts needed by policy.
7. `decisions.evaluate_consultation_gates` resolves task/variant/high-impact decisions and delegates semantic checks to `decision_paths`, `decision_rules`, and `decision_hparam`.
8. `DecisionReport.exit_code` maps `FAIL` to 1, unresolved user input to 2, and nonblocking results to 0.
9. When `--output-dir` is supplied, doctor may write question artifacts; this diagnostic output is not a workspace plan.

Contract: authored-input and semantic failures that make a workspace unsafe must be represented before plan mutation. Doctor and plan should share evaluation behavior rather than maintaining separate validators.

Primary tests: recipe, consultation-policy, user-decision, and CLI-contract tests under `tests/agent_tools/`.

## 2. Diagnostic Context

1. `plans.build_context` constructs a synthetic recipe from explicit task/config/label/variant arguments.
2. User-decision document/name/entry closure runs before any bundle output, then valid decisions are materialized and the same consultation facade runs.
3. Repository, skill, config, index, preset, expected-artifact, and validation-command summaries are assembled.
4. The output directory receives `context.json`, `context.md`, and either questions/blocked output or diagnostic command/validation scripts.

Contract: context is an information-gathering surface, not the managed recipe-plan API. Recipe-backed execution planning remains in `build_plan`; do not add a parallel `context --recipe` flow.

## 3. Managed Plan Generation

### Read-only phase

1. `plans.build_plan` canonicalizes the requested local output root.
2. `plans.preflight_plan` calls `evaluate_recipe`.
3. Preflight checks experiment/step resolution, workspace containment, a locally freezable config, hparam/final-test rules, supported command routing, and every planned output path.
4. Output guards reject symlinks, hard links, directory targets, aliases, and paths outside the managed root.

### Mutation phase

1. Pre-workspace failures return immediately with no workspace artifacts.
2. Other unresolved reports may initialize the bound workspace only when ownership is sufficiently established, then write questions and a blocked plan.
3. A passing report calls `ensure_experiment_workspace`.
4. Ordinary tasks freeze the config and launch script, write `plan.json`, `plan.md`, `recipe.resolved.yaml`, and per-run artifacts, then register one canonical run.
5. `hparam_tune` delegates finite grid expansion and frozen plan creation to `plan_hparam.write_hparam_plan`.
6. `merge_run_manifest` commits planned rows and refreshes the derived run matrix.

Contract: `preflight_plan` is reusable and read-only; `build_plan` is the mutation boundary. Generated commands route through `_commands_for_recipe`, `plan_rendering`, and existing runtime/preprocessing/sleep2stat modules.

Primary tests: plan-atomicity, recipe command rendering, CLI contract, experiment workspace, and task-specific plan tests.

## 4. Hparam Plan, Launch, and Observation

### Plan compilation

1. `plan_hparam.freeze_hparam_execution` requires explicit `execution.python` and full `execution.runtime_commit` for SSH targets, separate local workdirs, and conda-wrapped targets. Only a local `REPO_ROOT` target without a conda wrapper may omit them and freeze the manager interpreter and manager repository HEAD.
2. `plan_hparam.hparam_combos` expands the declared finite search space.
3. YAML-pointer overrides are validated against the source config before per-run configs are written.
4. Every run receives stable `run-NNN` identity, semantic name, version, frozen config, launch script, hashes, runtime/checkpoint directories, and managed parameters. Hparam `inputs.ckpt_path` belongs only to final evaluation and is not rendered into these tuning commands.
5. The plan, resolved recipe/base source, launch/final-eval artifacts, and canonical manifest rows are written before execution.

### Launch

1. `hparam_runtime.launch_hparam_runs` first calls `run_artifacts.read_hparam_plan`.
2. It locks the canonical manifest, validates the complete launch output topology, and derives GPU groups from frozen execution settings.
3. Dry-run records replayable launch evidence without probing Python, Git, SSH runtime capabilities, or creating an execution snapshot.
4. The first execute with an eligible slot probes the matching target wrapper, isolated workdir import root, Python/version, target hostname, runtime repository root, clean worktree, exact commit, repository-owned module origin, normalized `argparse` option set, explicit environment digest, and every frozen argv vector before `execution_snapshot.json` is atomically written or any run execution identity/process is committed. Argparse validation requires the same resolved module origin; rendered CLI text is not snapshot evidence.
5. Later eligible launch waves re-probe and require exact snapshot equality. Immediately before every `nohup`, the same target/env/conda/PYTHONPATH wrapper rechecks Python/version, commit, repository root, hostname, module origin, untracked or ignored importable code, and that run's script/config hashes. Plans without frozen Python/commit identity must be recreated. A missing snapshot may be established only before any run has committed an execution target or advanced beyond `planned`/`pending`; a full-capacity wave does not probe it.
6. Before computing capacity, execute-mode launch refreshes observable active rows from other plans that share the relevant target, SSH host, and GPU pool, commits status transitions, and excludes blockers that became terminal.
7. `run_hparam_queue` is a separate explicit action: dry-run performs one preview; execute records monitor-owned status transitions, calls the same locked launch owner, exits only when all current-plan runs are terminal, and fails on non-retriable `missing_pid` in either the current plan or a relevant cross-plan capacity blocker.
8. Interrupted launch artifacts are reconciled against known started run keys and canonical state.

### Monitor and stop

1. `monitor_hparam_runs` validates the plan and observes only its declared runs.
2. `run_evidence.status_row` combines PID/process/log/progress/runtime/checkpoint evidence.
3. Observations are written plan-locally and committed through `merge_run_manifest`.
4. `stop_hparam_run` requires a non-empty reason and uses `read_pid(..., expected_script=...)` before signaling.

Contract: `run_manifest.tsv` is durable state. `launch_manifest.tsv`, `run_status.tsv`, and `execution_snapshot.json` are plan-local evidence, not competing state authorities. Monitor remains observation-only. Historical `trial_*` formats are read-only and current-format plans that predate frozen execution identity are recreated, not upgraded in place.

Primary tests: hparam runtime, run-artifact, run-evidence, experiment-workspace, and remote evidence tests.

## 5. Hparam Selection and Postprocessing

1. `hparam_selection.select_hparam_candidates` loads the validated plan and uses only the metric/mode frozen in `evaluation_policy`.
2. It follows registered step plans, verifies every run against canonical identity, reads only exact runtime manifests, and validates checkpoint ownership.
3. It writes workspace `reports/ranking.csv`, preserves valid rows for other steps, commits score/rank/checkpoint observations, and appends an event.
4. `scan_hparam_checkpoints` provides a separate many-checkpoints-per-run diagnostic ranking.
5. `generate_external_eval` requires explicit final-test unlock and writes selected-candidate external inference artifacts.
6. `export_hparam_logits` may export validation only without unlock; test export requires unlock unless explicitly skipped.
7. Threshold and ensemble operations consume managed exported predictions rather than reopening selection semantics.

Contract: selection uses validation-approved frozen policy. Test/external data must not feed candidate selection. Postprocessing validates selected rows and owner plans before writing any output.

## 6. Experiment Lifecycle and Tracking

### Initialization and steps

1. `experiments.init_experiment` parses an explicit spec, binds the exact root, rejects non-empty unmanaged/conflicting roots, and creates canonical manifests.
2. `register_experiment_step` validates complete step metadata and merges the canonical step envelope.

### External observations

1. `sync_wandb_runs` reads canonical rows first, obtains W&B payloads, proves ownership, writes W&B/metric evidence, and commits only source-allowlisted observations.
2. `index_checkpoints` validates metrics and creates managed checkpoint rows.
3. `monitor_experiment` builds local/remote run observations and commits them before writing its report.
4. `rank_experiment_candidates` validates metric/checkpoint rows against canonical runs and writes an experiment-wide report.

### Completion

1. `finalize_experiment` requires at least one managed run.
2. Every run must already be terminal.
3. A non-empty final report is copied to `reports/final.md`.
4. Only then is `experiment.yaml` marked completed and an event appended.

Contract: tracking produces evidence; it does not independently own lifecycle state. External rows must prove experiment/run ownership and frozen-field agreement before filtering or aggregation.

Primary tests: local/remote experiment lifecycle, workspace, tracking, W&B ownership, checkpoint, monitor, and ranking tests.

## 7. Adaptive Hparam Workflow

1. `init_adaptive_workflow` validates adaptive settings and budget, then calls `preflight_plan` for round 000 before creating workflow artifacts.
2. After passing preflight, it calls `freeze_hparam_execution` once, writes a legal hparam local overlay rather than a flattened base/effective recipe, creates round 000 through `build_plan`, and stores only the resolved Python/commit pair as `adaptive/workflow.json.execution_identity` alongside the registry, README, and events.
3. `adaptive_step` and standalone suggestion first preflight the current mutable source before digest, event, or suggestion mutation, reject an explicitly conflicting Python/commit pair, and overlay the round-000 identity. Other execution fields remain source-controlled, allowing preflight-approved capacity, GPU, environment, and similar operational changes in later round plans.
4. `digest_hparam_run` monitors the current round, records managed metrics/checkpoint/log evidence, and updates the incumbent table.
5. `suggest_next_round` ranks finite objective values, preflights the generated local overlay through a temporary authored recipe, and only then writes the suggestion plus rationale.
6. `adaptive_step` preflights the complete next round before execute-mode replacement actions.
7. Only when the replacement round fits remaining budget may execute mode stop/supersede eligible current runs, build/register the next plan, and launch it.
8. `adaptive_loop` repeats steps until budget or no-progress termination; dry-run stops after one step.

Contract: adaptive code composes existing plan, hparam, evidence, and workspace owners. Python/commit identity is workflow-stable from round 000, while other execution settings are frozen independently by each immutable round plan. Adaptive code must not maintain a second canonical status table or retire current work before a viable replacement round is validated.

Primary tests: adaptive workflow plus hparam runtime, selection, preflight, and workspace tests.

## 8. Sleep2stat Recipe Routing

1. `configs.config_summary` recognizes sleep2stat-shaped YAML and delegates structural validation to `sleep2stat.config.load_config`.
2. `decision_rules.sleep2stat_issues` handles agent-specific high-impact choices and path intent.
3. `_commands_for_recipe` renders existing `sleep2stat validate-config`, `run`, optional `summarize`, and optional plotting commands.
4. Dry-run recipes omit post-run summarize/plot commands because no completed bundle exists.

Contract: sleep2stat remains variantless. Its runtime/schema belongs to `sleep2stat`; agent tooling only gates and renders the existing CLI.

## Verification Gate

For agent-tool contract changes, use the smallest relevant test files first, then:

```bash
PYTHONPYCACHEPREFIX=/tmp/sleep2vec_pycache python3 -m compileall agent_tools tests
python3 -m pytest -q tests/agent_tools
python -m agent_tools skills --validate
git diff --check
```

If the repository's expected Python/pytest environment differs, report that explicitly rather than treating an unavailable command as a pass.
