# Delta From Main

## Comparison Point

- Branch: `agent_tools/refractor`
- Branch commit: `40b8b7b811a68369e739e9d2f717651652853ace`
- Local `main` commit at generation time: `40b8b7b811a68369e739e9d2f717651652853ace`
- Ahead/behind: `0 / 0`

## Code Delta

There is no tracked code delta from local `main` at this initialization point. `git diff main...HEAD` is empty. This branch index therefore records the same committed implementation while giving agent tooling a dedicated, branch-resolvable handbook location before further work.

## Index Delta

The existing `main` index manifest still identifies an earlier refresh commit. This branch index was independently checked against current source and records the current commit. It is intentionally narrower: agent tooling and its recipe, skill, policy, contract, and test boundaries are indexed in detail; model/data/runtime packages are included only as downstream context.

## Stale or Removed Entries

- No stale branch-specific symbols existed because this is a new index.
- No code symbols were removed relative to local `main`.
- Historical `trial_*` experiment artifacts remain explicitly read-only in current code and are not documented as supported management APIs.

## Refresh Rule

After branch changes, compare the checked-out source to this manifest, update affected function/workflow ownership, and record a new entry in [CHANGELOG.md](./CHANGELOG.md). Do not infer branch behavior from the `main` handbook alone.
