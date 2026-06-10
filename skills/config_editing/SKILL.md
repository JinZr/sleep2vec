# Skill: config_editing

## When to use
Use for `config_edit` tasks that change YAML semantics or recipes.

## Required inputs
Requires target config, intended task, explicit high-impact decisions, and validation command.

## First information-gathering commands
- `python -m agent_tools config-summary --config <config> --json`
- `python utils/check_configs.py <config>`

## Decision checklist
Confirm task semantics, channels, backend, monitor, preset policy, and owner review.

## Stop-and-consult gates
The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

## Canonical commands
Use `python utils/check_configs.py <config>` after changes.

## Expected artifacts
Updated YAML and validation output.

## Validation gates
Run config checks and targeted tests for changed contracts.

## Common failure modes
Folder-name assumptions, missing `preset_build`, channel parity mismatch, and monitor mismatch.

## Relevant owners and index pages
Owners: `config-task-contract`, `agent-tooling-maintainer`. Index: config validation workflow.
