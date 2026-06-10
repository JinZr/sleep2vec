# Skill: index_maintenance

## When to use
Use for `index_maintain` tasks that update branch-specific Codex index docs.

## Required inputs
Requires current branch, touched files, new contracts, and verification state.

## First information-gathering commands
- `git branch --show-current`
- `git rev-parse HEAD`
- `rg --files doc/codex_index/branches/<branch>`

## Decision checklist
Update only the relevant index files for the changed contracts unless a full refresh is required.

## Stop-and-consult gates
The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

## Canonical commands
Use normal file inspection and targeted edits under `doc/codex_index/branches/<branch>/`.

## Expected artifacts
Updated branch index README, manifest, module map, reuse guide, workflows, or function catalogs.

## Validation gates
Review links and ensure new docs point at real files.

## Common failure modes
Stale branch metadata, stale deleted file references, and over-broad refreshes.

## Relevant owners and index pages
Owner: `agent-tooling-maintainer`. Index: branch README and manifest.
