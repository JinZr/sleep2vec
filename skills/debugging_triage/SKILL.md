# Skill: debugging_triage

## When to use
Use for `debug` tasks that collect failure context before changing code or rerunning long jobs.

## Required inputs
Requires error text, config path, command or run directory, and relevant artifacts.

## First information-gathering commands
- `python -m agent_tools repo-summary --json`
- `python -m agent_tools config-summary --config <config> --json`

## Decision checklist
Identify whether the failure is config, data, checkpoint, runtime, or environment related.

## Stop-and-consult gates
The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

## Canonical commands
Use read-only summaries first, then targeted tests or smoke commands.

## Expected artifacts
Context bundle and question list when decisions are ambiguous.

## Validation gates
Run the smallest command that reproduces or explains the failure.

## Common failure modes
Missing paths, stale generated artifacts, unsupported backend, and environment dependency gaps.

## Relevant owners and index pages
Owner: `agent-tooling-maintainer`. Index: shared `doc/codex_index/README.md`.
