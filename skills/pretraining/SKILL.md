# Skill: pretraining

## When to use
Use for `pretrain` tasks that launch contrastive pretraining.

## Required inputs
Requires config YAML, data backend inputs, version name, device/runtime settings, and preset or Kaldi manifest inputs.

## First information-gathering commands
- `python -m agent_tools config-summary --config <config> --json`
- `python utils/check_configs.py <config>`

## Decision checklist
Confirm backend, checkpoint initialization, missing-channel policy, runtime devices, and output naming.

## Stop-and-consult gates
The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

## Canonical commands
Use the recipe `variant` to choose the module: `python -m sleep2vec.pretrain`, `python -m sleep2vec2.pretrain`, or `python -m sleep2expert.pretrain`.

## Expected artifacts
`log-pretrain/<run>/checkpoints/`, copied `config.yaml`, `cli_args.yaml`, and W&B logs if enabled.

## Validation gates
Run config checks and optional diagnostics smoke with `--print-diagnostics`.

## Common failure modes
Missing preset, Kaldi manifest mismatch, missing channels, W&B auth, and GPU precision incompatibility.

## Relevant owners and index pages
Owners: `runtime-orchestrator`, `model-integration`, `agent-tooling-maintainer`. Index: pretrain workflow.
