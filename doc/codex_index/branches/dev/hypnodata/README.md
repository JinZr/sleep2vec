# Dev Hypnodata Branch Codex Engineering Index

This directory is the scoped branch index for `dev/hypnodata`. It covers the
new `hypnodata` ingestion layer and the config/preprocessing contract changed on
this branch.

## Branch Scope

- Branch: `dev/hypnodata`
- Last refresh commit: `edd0d205f3cf197e0c3e0f8f7c970430b3d94c32`
- Last refresh at: `2026-06-16T02:48:06Z`
- Mode: `initialize-branch`
- Baseline consulted: `doc/codex_index/branches/main/`

## Purpose

Use this index before changing `hypnodata` behavior. It answers:

1. Which module owns discovery, config parsing, preprocessing, output writing,
   and downstream manifests?
2. Which implementation should be extended instead of duplicated?
3. Which tests pin the public NPZ and manifest contract?

## Coverage

Indexed in detail:

- `hypnodata/`
- `configs/hypnodata/`
- `tests/test_hypnodata_*.py`
- `tests/hypnodata_test_helpers.py`

This scoped index does not rebuild the full `main` manual for model training,
datasets, variants, agent tooling, or sleep2stat internals. Use
`doc/codex_index/branches/main/` for those areas.

## Recommended Reading Order

1. [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md)
2. [MODULE_MAP.md](./MODULE_MAP.md)
3. [REUSE_GUIDE.md](./REUSE_GUIDE.md)
4. [FUNCTIONS/HYPNODATA.md](./FUNCTIONS/HYPNODATA.md)
5. [WORKFLOWS/HYPNODATA.md](./WORKFLOWS/HYPNODATA.md)

## Reliability Notes

- Claims are grounded in the current checked-out `dev/hypnodata` code.
- Where `main` has no matching `hypnodata` implementation, this index names the
  branch-local implementation as canonical.
- External `/Users/zrjin/git/wuji` code was used as reference only; it is not a
  runtime dependency.
