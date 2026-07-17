# Skill: index_maintenance

## When to use
Use for `index_maintain` tasks that update the shared Codex navigation docs.

## Required inputs
Requires touched files, changed ownership or contracts, and verification state.

## First information-gathering commands
- `rg --files doc/codex_index`
- `git diff --name-only`
- Inspect the changed source, tests, and relevant shared index page.

## Decision checklist
Update the index only for changes to ownership, canonical reusable implementations, public contracts, key workflows, or top-level entrypoints. Select the owning page: `MODULE_MAP.md` for boundaries, `REUSE_GUIDE.md` for canonical implementations, and `WORKFLOWS.md` for execution flow. Use `README.md` only for navigation policy.

## Stop-and-consult gates
Stop and ask the user if the owning module or public contract cannot be determined from source, tests, `AGENTS.md`, or dedicated contract documents.

## Canonical commands
Use normal file inspection and targeted edits under `doc/codex_index/`. Never create branch copies, `branches/` trees, manifests, changelogs, function catalogs, or delta files.

## Expected artifacts
One or more targeted updates to `README.md`, `MODULE_MAP.md`, `REUSE_GUIDE.md`, or `WORKFLOWS.md`.

## Validation gates
Confirm `rg --files doc/codex_index` lists exactly the four shared documents, review links against real files, search for references to removed branch-scoped index paths, and run `git diff --check`.

## Common failure modes
Copying source-level detail into the index, updating it for a local fix, stale deleted-file references, and recreating branch-specific metadata.

## Relevant owners and index pages
Owner: `agent-tooling-maintainer`. Index: shared `README.md`.
