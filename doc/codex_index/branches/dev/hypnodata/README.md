# Dev Hypnodata Branch Codex Engineering Index

This directory is the scoped branch index for `dev/hypnodata`. It covers the
new `hypnodata` ingestion layer and the config, preprocessing, annotation, and
manifest contracts changed on this branch.

## Branch Scope

- Branch: `dev/hypnodata`
- Last refresh commit: `296dff6bedc7ee144c5ba76676f2751bcd59a07e`
- Last refresh at: `2026-06-16T05:52:59Z`
- Mode: `refresh`
- Baseline consulted: `doc/codex_index/branches/main/`

## Purpose

Use this index before changing `hypnodata` behavior. It answers:

1. Which module owns discovery, config parsing, preprocessing, output writing,
   and downstream manifests?
2. Which implementation should be extended instead of duplicated?
3. Which tests pin the public NPZ, annotation, and manifest contracts?

## Coverage

Indexed in detail:

- `hypnodata/`
- `configs/hypnodata/`
- `tests/hypnodata/test_hypnodata_*.py`
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
