# Skill: inference_evaluation

## When to use
Use for `infer` or `evaluate` tasks that evaluate a downstream checkpoint.

## Required inputs
Requires config YAML, label name, checkpoint path, eval split, backend inputs, and final external-test unlock state.

## First information-gathering commands
- `python -m agent_tools config-summary --config <config> --json`
- `python -m agent_tools doctor --recipe <recipe>`

## Decision checklist
Confirm checkpoint identity, eval split, averaging policy, external-test unlock, variant, output directory, and any separate workdir. Workdir alone changes cwd/PYTHONPATH; an immutable runtime identity requires explicit workdir, Python, and full commit identity.

## Stop-and-consult gates
The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

## Canonical commands
Use the recipe `variant` to choose the module: `python -m sleep2vec.infer`, `python -m sleep2vec2.infer`, or `python -m sleep2expert.infer`. When runtime identity is declared, use its frozen Python.

## Expected artifacts
Run-local metrics CSV, prediction CSV, overview row, and `run_manifest.json`.

## Validation gates
Run config summary, verify checkpoint path, and avoid test split unless final evaluation is explicitly unlocked. For explicit runtime identity, verify the commit guard and that inference and lifecycle commits use the same frozen Python.

## Common failure modes
Missing checkpoint, AHI checkpoint averaging, missing fitted threshold, backend mismatch, locked external test, and missing or drifted runtime identity.

## Relevant owners and index pages
Owners: `runtime-orchestrator`, `agent-tooling-maintainer`. Index: infer/checkpoints workflow.
