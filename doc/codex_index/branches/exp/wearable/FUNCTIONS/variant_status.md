# Functions: Variant Status

## Summary

This branch contains directories named:

- `sleep2vec2/`
- `sleep2vec_moe/`
- `sleep2vec_hires/`

But `git ls-files` reports no tracked source files inside any of them on `exp/wearable`.

## What Was Observed

- only local `__pycache__` directories are present in the working tree
- no `.py`, config, or test files are tracked under these variant roots on this branch
- because of that, branch-local function indexing for these packages is not possible

## Reuse Guidance

- Treat variant behavior as `unknown` for this branch.
- Do not assume parity with `main` or with other branches.
- If a future task needs variant work, refresh the index after checking out a branch that actually tracks those sources.

## Stale-Reference Rule

If later updates add tracked files under these directories, this page should be replaced by real module/function pages and the `MANIFEST.json` variant status should be updated.
