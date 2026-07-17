# Skill: inference_evaluation

## When to use
Use for `infer` or `evaluate` tasks that evaluate a downstream checkpoint.

## Required inputs
Requires config YAML, label name, checkpoint path, eval split, backend inputs, and final external-test unlock state.

## First information-gathering commands
- `python -m agent_tools config-summary --config <config> --json`
- `python -m agent_tools doctor --recipe <recipe>`

## Decision checklist
Confirm checkpoint identity, eval split, averaging policy, external-test unlock, variant, and output directory.

## Stop-and-consult gates
The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

## Canonical commands
Use the recipe `variant` to choose the module: `python -m sleep2vec.infer`, `python -m sleep2vec2.infer`, or `python -m sleep2expert.infer`.

## Expected artifacts
Run-local metrics CSV, prediction CSV, overview row, and `run_manifest.json`.

## Validation gates
Run config summary, verify checkpoint path, and avoid test split unless final evaluation is explicitly unlocked.

## Common failure modes
Missing checkpoint, AHI checkpoint averaging, missing fitted threshold, backend mismatch, and locked external test.

## Relevant owners and index pages
Owners: `runtime-orchestrator`, `agent-tooling-maintainer`. Index: infer/checkpoints workflow.
