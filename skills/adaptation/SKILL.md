# Skill: adaptation

## When to use
Use for `adapt` tasks that run stage1/stage2 modality adaptation.

## Required inputs
Requires adapt config YAML, phase, pretrained or resume checkpoint policy, backend inputs, version name, and runtime settings.

## First information-gathering commands
- `python -m agent_tools config-summary --config <config> --json`
- `python utils/check_configs.py <config>`

## Decision checklist
Confirm phase, `adapt.new_channels`, stage transition checkpoint, pair schedule, and checkpoint directory policy.

## Stop-and-consult gates
The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

## Canonical commands
Use the recipe `variant` to choose the module: `python -m sleep2vec.adapt`, `python -m sleep2vec2.adapt`, or `python -m sleep2expert.adapt`.

## Expected artifacts
`log-adapt/<run>/checkpoints/` or `checkpoints.stage2/`, root and phase-specific config/CLI snapshots.

## Validation gates
Run config checks and optional diagnostics smoke.

## Common failure modes
Wrong phase checkpoint type, non-empty stage2 directory, missing new-channel availability, and backend mismatch.

## Relevant owners and index pages
Owners: `runtime-orchestrator`, `model-integration`, `agent-tooling-maintainer`. Index: adapt workflow.
