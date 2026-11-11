# Skill: inference_evaluation

## When to use
Use for `infer` tasks that evaluate a downstream checkpoint.

## Required inputs
Requires config YAML, label name, checkpoint path, eval split, backend inputs, and final external-test unlock state.

## First information-gathering commands
- `python -m agent_tools config-summary --config <config> --json`
- `python -m agent_tools doctor --recipe <recipe>`

## Decision checklist
Confirm checkpoint identity, eval split, averaging policy, external-test unlock, variant, output directory, and any separate workdir. Workdir alone changes cwd/PYTHONPATH; managed runtime provenance requires explicit workdir, Python, and a full planned/baseline commit.

## Stop-and-consult gates
The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

## Canonical commands
Use the recipe `variant` to choose the module: `python -m sleep2vec.infer`, `python -m sleep2vec2.infer`, `python -m sleep2expert.infer`, or `python -m sex_age_baseline.infer`. When runtime identity is declared, use its frozen Python. `sex_age_baseline.infer` takes the same options as `sleep2vec.infer`, including checkpoint averaging and `--override-dataset-names`; it fails at launch when `--pretrained-backbone-path` is set.

For rolling latest-main maintenance, use one existing checkout. Run
`python -m agent_tools runtime-sync --workdir <checkout>` first;
add `--execute` only to perform an authorized clean fast-forward of
`origin/main`. Do not clone or reset. The short sync/launch lock is released
while a job runs, so a process whose spawn boundary observed A may continue
while a later process observes B. That SHA does not guarantee stable checkout
bytes for the whole evaluation.

For an ordinary registered Slurm plan, follow [Managed ordinary inference](../../doc/agent_contracts/task_recipe.md#managed-ordinary-inference): `infer-launch --plan-dir <plan>` defaults to dry-run; add `--execute` only within existing authorization. Monitor through `experiment-monitor`, inspect with `experiment-status`, and stop through `run-stop --plan-dir <plan> --run-id <run-id> --reason <reason>`. Do not hand-submit `job.sbatch` or invoke the frozen worker separately.

To evaluate one checkpoint on several cohorts, publish one ordinary plan per cohort (`doctor` → `plan` → `infer-launch`) and summarize the results with the experiment's own analysis script.

## Expected artifacts
Run-local metrics CSV, prediction CSV, overview row, and `run_manifest.json`.

## Validation gates
Run config summary, verify checkpoint path, and avoid test split unless final evaluation is explicitly unlocked. `execution.runtime_commit` is planned/baseline provenance; a provenance-aware direct launcher orders embedded verification, actual-HEAD capture, and child `Popen` in the same short lock, while a Slurm job writes `git rev-parse HEAD` to `run_dir/runtime_commit` when its allocation starts. The lock does not cover child lifetime. A mismatch is recorded and warned, not blocked or treated as a scientific variable. Direct generated scripts retain their frozen Python and emitted input-hash checks; they do not imply the managed-scheduler module-origin or live-argv gate. Ordinary Slurm inference runs the managed-scheduler clean/importable-code, module-origin, Python/route, frozen-argv, and artifact-hash gates before submission; in the allocation the job rechecks only its frozen config and launch-script hashes and writes its exit code to `run_dir/exit_code`. A Slurm run without that file after its job leaves the queue is `lost`: it is resolved only by a reasoned `run-stop` and never authorizes an automatic rerun.

## Common failure modes
Missing checkpoint, AHI checkpoint averaging, missing fitted threshold, backend mismatch, locked external test, missing Python/route identity, frozen direct-input drift, or managed-scheduler clean-code, module-origin, frozen-argv, and artifact-hash failures.

## Relevant owners and index pages
Owners: `runtime-orchestrator`, `agent-tooling-maintainer`. Index: infer/checkpoints workflow.
