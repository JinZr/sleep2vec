# Skill: hyperparameter_tuning

## When to use
Use for `hparam_tune` recipes that generate validation-only trial plans and optionally orchestrate active launch, monitoring, selection, thresholding, and final external-test evaluation through `agent_tools`.

## Required inputs
Requires base recipe, search method, search parameters, budget, selection metric/mode/split, external-test lock policy, and final evaluation policy. Active orchestration additionally requires an optional `execution:` block for target, host, workdir, conda env, GPU pool, max concurrency, W&B project/group, log dir, pid dir, and environment variables.

## First information-gathering commands
- `python -m agent_tools doctor --recipe <recipe>`
- `python -m agent_tools plan --recipe <recipe> --output-dir <dir>`
- `python -m agent_tools hparam-launch --plan-dir <dir>`
- `python -m agent_tools hparam-monitor --run-dir <dir> --once`

## Decision checklist
Confirm validation-only selection, namespaced `runtime.*` or `yaml:/...` search keys, generated config directory, unique version names, execution target, GPU assignment, W&B project/group, log/PID locations, max concurrency, and final-test unlock state.

## Stop-and-consult gates
The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

Stop and consult the user if:

- The search space is missing.
- The search budget is missing.
- Search keys are bare names instead of `runtime.*` or `yaml:/...`.
- `selection_metric` or `selection_mode` is missing.
- `selection_split` is test.
- `external_test_locked` is missing.
- Any trial command would evaluate the external test split.
- Final test evaluation is requested without explicit unlock.
- `execution.target=ssh` is requested without a host.
- The user asks to stop or kill jobs that were not launched through the recorded hparam manifest/PID files.
- Candidate selection would use test metrics instead of validation metrics.

## Canonical commands
Generate shell scripts that call the recipe variant's `finetune` module for trials. For active orchestration, run `hparam-launch` after `agent_tools plan`; dry-run is the default, and `--execute` is required to start jobs. Monitor with `hparam-monitor`, stop only with `hparam-stop --trial-id <id>`, rank validation candidates with `hparam-select`, tune binary thresholds with `hparam-threshold`, and average saved probabilities with `hparam-ensemble`. Use the same variant's `infer` module only through `hparam-external-eval --unlock-final-test` or an explicitly unlocked final evaluation script.

## Expected artifacts
Plan JSON/Markdown, trial scripts, `trials.csv`, generated config copies, `run_all.sh`, optional `final_external_test.sh`, `launch_manifest.tsv`, `trial_status.tsv`, `logs/<trial_id>.log`, `pids/<trial_id>.pid`, `candidate_ranking.csv`, optional `threshold_summary.csv`, optional `ensemble_summary.csv`, and optional `external_eval_manifest.tsv` plus copied external-test configs.

## Validation gates
Run doctor first; plan generation must block with exit code 2 when consultation is required. Before executing jobs, run `hparam-launch` without `--execute` and inspect the manifest for conda env, GPU assignment, W&B project/group, log path, PID path, and external-test lock behavior.

## Common failure modes
Missing search values, trial count exceeding budget, test split selection, missing final unlock, missing base recipe, invalid execution target, stale PID files, moving `best-epoch=*.ckpt` aliases used for selection, and W&B summaries missing while run manifests are available.

## Relevant owners and index pages
Owners: `runtime-orchestrator`, `regression-guard`, `agent-tooling-maintainer`. Index: finetune and agent tooling workflows.
