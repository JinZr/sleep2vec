# Skill: hyperparameter_tuning

## When to use
Use for `hparam_tune` recipes that generate validation-only run plans, orchestrate active launch/monitor/select/evaluate, or run append-only adaptive external-optimized tuning through `agent_tools`.

## Required inputs
Requires experiment metadata, a named step, base recipe, search method, search parameters, budget, selection metric/mode/split, external-test lock policy, and final evaluation policy. Active orchestration additionally uses `execution:` fields. A local target at `REPO_ROOT` without a conda wrapper may omit Python/commit identity; SSH targets, separate local workdirs, and conda-wrapped targets require explicit `execution.python` and full `execution.runtime_commit`. Adaptive tuning additionally requires `adaptive.enabled=true`; if optimizing test/external metrics, `adaptive.test_feedback_for_selection=true` must be explicit. Omitting `adaptive.suggest.strategy` selects `agent_proposal`; agent proposals additionally require `adaptive.objective_metric` as an explicit non-blank string plus explicit non-empty `adaptive.objective_mode`, `adaptive.round_size`, `adaptive.max_rounds`, and `adaptive.max_runs_total`, are terminal-only, and require replacement to be omitted or exactly disabled. Use explicit `best_neighborhood` only for automatic neighborhood suggestions or active-round replacement.

## First information-gathering commands
- `python -m agent_tools doctor --recipe <recipe>`
- `python -m agent_tools plan --recipe <recipe> --output-dir <dir>`
- `python -m agent_tools hparam-launch --plan-dir <dir>`
- `python -m agent_tools hparam-run-queue --plan-dir <dir>`
- `python -m agent_tools hparam-monitor --run-dir <dir> --once`
- `python -m agent_tools hparam-checkpoint-scan --run-dir <dir> --metric <metric> --mode max|min`
- `python -m agent_tools hparam-adaptive-init --recipe <recipe> --output-dir <dir>`
- `python -m agent_tools hparam-adaptive-step --workflow-dir <dir>`
- `python -m agent_tools hparam-adaptive-step --workflow-dir <dir> --proposal <submission.json>`

## Decision checklist
Confirm validation-only selection for static tuning, namespaced `runtime.*` or `yaml:/...` search keys, generated config directory, unique version names, execution target and required Python/commit identity, GPU assignment, W&B project/group, log/PID locations, max concurrency, final-test unlock state, and whether the run is explicitly external-optimized adaptive tuning. For `agent_proposal`, also confirm the numeric authorization bounds and that terminal-only rounds, disabled replacement, and an external two-phase agent driver are intended.

## Stop-and-consult gates
The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

Stop and consult the user if:

- The search space is missing.
- The search budget is missing.
- Search keys are bare names instead of `runtime.*` or `yaml:/...`.
- `selection_metric` or `selection_mode` is missing.
- `selection_split` is test.
- `external_test_locked` is missing.
- Any tuning run would evaluate the external test split.
- Final test evaluation is requested without explicit unlock.
- `execution.target=ssh` is requested without a host.
- The target is SSH, a separate local workdir, or conda-wrapped, but `execution.python` or full `execution.runtime_commit` is missing.
- The user asks to stop or kill jobs that were not launched through the recorded hparam manifest/PID files.
- Candidate selection would use test metrics instead of validation metrics.
- Adaptive tuning uses `test_*` or `external_*` objective metrics without `adaptive.test_feedback_for_selection=true`.
- Adaptive output would overwrite an older round instead of appending a new round/event.
- `agent_proposal` omits or leaves null/empty any explicit objective, direction, round-size, round-budget, or total-run-budget field; consultation must stop with exit code 2 before workspace mutation.
- `agent_proposal` provides a non-string `adaptive.objective_metric`; the recipe contract must fail before workspace mutation.
- `agent_proposal` is requested with active replacement, invalid numeric bounds, or an expectation that `hparam-adaptive-loop` will invoke an LLM.

## Canonical commands
Generate shell scripts that call the recipe variant's `finetune` module for managed runs. After `agent_tools plan`, preview one launch wave with `hparam-launch`; dry-run is the default, and `--execute` is required to start jobs. Use `hparam-run-queue --execute` when the authorized action is to keep filling capacity until the full current plan is terminal. Monitor with `hparam-monitor`; it updates status but never fills free slots. Add `--health` when remote/GPU/IO/log/progress evidence is needed. Stop only with `hparam-stop --run-id <id> --reason <text>`, rank validation candidates with `hparam-select`, and use the same variant's `infer` module only through explicitly unlocked final evaluation commands.

For adaptive tuning, initialize with `hparam-adaptive-init`, then use `hparam-adaptive-step` or, for `best_neighborhood` only, `hparam-adaptive-loop`. Initialization freezes round 000 `execution.python` and `execution.runtime_commit` as the workflow identity. Later rounds must keep that identity but may take updated operational execution fields such as `max_concurrent`, GPU allocation, and `env` from the source recipe after preflight. Adaptive commands append to the experiment-level `events.jsonl` and create `run_registry.tsv`, digests, suggestions, and per-round plans; they must not rewrite previous round plans, scripts, configs, logs, or checkpoints.

For `agent_proposal`, call `hparam-adaptive-step --workflow-dir <dir>` first. It monitors and digests the current round; while runs remain active it returns `waiting_for_round_terminal`, and after the round is terminal it writes and records an immutable `adaptive/proposal_inputs/round_NNN--<id12>.json`. Proposal-input schema v2 requires `input.source_config_sha256`, which binds the exact `inputs.config` file bytes into the request id. Give that tool-issued snapshot to the external agent, which may write only the exact `proposal_submissions` path named inside; it must not create or modify proposal inputs, events, digests, manifests, registries, or other lifecycle state. Preview with `hparam-adaptive-step --workflow-dir <dir> --proposal <submission.json>`; add `--execute` to register and launch the validated next round. Phase two verifies the recorded issuance and complete snapshot-file hash, reconstructs the input from current canonical state, and repeats that validation after candidate preflight before mutation. Execute freezes the verified config bytes as the next round's `source_config.yaml` and materializes from the validated in-memory proposal, so later edits to the source config or suggestion do not affect the plan. Regenerate v1 input snapshots; proposal submissions remain schema v1.

The submission carries exactly one of `parameters` (per-key candidate lists, budgeted by Cartesian product) or `configurations` (explicit joint configuration points, one run per point, budgeted by point count); either way it must use every original search key (per mapping or per point), cite evidence run ids from the snapshot, stay inside numeric bounds or categorical choices, avoid duplicate points, and fit both `round_size` and remaining total budget. Prefer `configurations` when proposing specific joint settings -- two points launch exactly two runs instead of their cross product. Do not call `hparam-suggest`, regenerate a digest, or select a latest digest during phase two. If config bytes or canonical workflow evidence change, request a new snapshot. If a launch attempt creates an uncommitted round, request a new snapshot for the next fresh target round instead of reusing that round number.

## Expected artifacts
Experiment manifest, step manifest, resolved recipe, plan JSON/Markdown, `run_matrix.csv`, per-run directories named `run-NNN--<semantic-name>`, frozen config and launch snapshots, `run_all.sh`, first-execute `execution_snapshot.json`, optional `final_external_test.sh`, `launch_manifest.tsv`, `run_status.tsv`, experiment-level `events.jsonl`, `reports/ranking.csv`, optional `threshold_summary.csv`, optional `ensemble_summary.csv`, and optional `external_eval_manifest.tsv` plus copied external-test configs. Adaptive workflows also write `adaptive/run_registry.tsv`, `adaptive/incumbents.tsv`, `adaptive/digests/`, `adaptive/suggestions/`, and `adaptive/rounds/round_*/`. Agent-proposal workflows additionally write tool-owned immutable `adaptive/proposal_inputs/` snapshots, their issuance events, and a bound `source_config.yaml` in each accepted round; the external agent supplies matching files only under `adaptive/proposal_submissions/`.

## Validation gates
Run doctor first; plan generation must block with exit code 2 when consultation is required. Before executing jobs, run `hparam-launch` without `--execute` and inspect the manifest for target Python/commit, conda env, GPU assignment, W&B project/group, log path, PID path, and external-test lock behavior. Only the canonical local manager runtime may receive planner-frozen identity defaults. The first eligible execute must atomically create a verified execution snapshot before starting a process; Python/import/host failure, a module origin outside the verified repository, runtime commit or worktree drift, or rejection of any frozen CLI argument vector must stop execution. Each actual start must repeat the target identity/import check through the same wrapper and verify the frozen script/config hashes immediately before the managed process starts.

## Common failure modes
Missing experiment ownership, missing search values, run count exceeding budget, test split selection, missing final unlock, missing base recipe, invalid execution target, missing explicit identity for a non-manager runtime, stale or missing PID files, a `missing_pid` run in the current plan or a relevant cross-plan capacity blocker, changed frozen snapshots or script/config hashes, target Python/host/commit/module-origin/CLI drift, a plan without frozen Python/commit identity, a started plan without `execution_snapshot.json`, moving `best-epoch=*.ckpt` aliases used for selection, adaptive test feedback without explicit flag, and W&B summaries missing while run manifests are available. Agent-proposal failures additionally include active replacement, nonterminal source runs, a v1 or unissued input snapshot, an unknown or stale request id, a conflicting issuance record, a submission at the wrong path, modified snapshot bytes, source config/recipe/execution-identity or canonical evidence drift, missing or extra parameter keys, values outside the authorized envelope, unknown evidence run ids, duplicate configuration points, and a Cartesian product or configuration count beyond round or remaining total budget. Recreate plans and proposal inputs that predate their current frozen contracts; do not patch them in place.

## Relevant owners and index pages
Owners: `runtime-orchestrator`, `regression-guard`, `agent-tooling-maintainer`. Index: finetune and agent tooling workflows.
