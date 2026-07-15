# Delta From Main

## Comparison Point

- Branch: `agent_tools/refractor`
- Branch commit: `faefcc7f3d46cf3739ff53dd89d8749ba9d93b9e`
- Local `main` commit at generation time: `40b8b7b811a68369e739e9d2f717651652853ace`
- Ahead/behind: `1 / 0`

## Code Delta

The branch is one commit ahead of local `main`. That commit closes authored recipe fields in their existing owners, preserves hparam source-layer semantics, and strengthens adaptive preflight ordering. The current refresh also indexes the working-tree queue-drain and verified-target snapshot changes being validated on this branch.

## Index Delta

The existing `main` index manifest still identifies an earlier refresh commit. This branch index was independently checked against current source and records the current commit. It is intentionally narrower: agent tooling and its recipe, skill, policy, contract, and test boundaries are indexed in detail; model/data/runtime packages are included only as downstream context.

## Stale or Removed Entries

- Removed stale rendered-CLI snapshot wording; current runtime evidence binds normalized supported options and exact validated argv vectors to a module origin inside the verified repository, then repeats identity/import and frozen artifact checks immediately before process start.
- No code symbols were removed relative to local `main`.
- Historical `trial_*` experiment artifacts remain explicitly read-only in current code and are not documented as supported management APIs.
- Current-format plans without frozen Python/commit identity, and already-started plans without an execution snapshot, have no compatibility upgrade path and must be recreated.

## Refresh Rule

After branch changes, compare the checked-out source to this manifest, update affected function/workflow ownership, and record a new entry in [CHANGELOG.md](./CHANGELOG.md). Do not infer branch behavior from the `main` handbook alone.
