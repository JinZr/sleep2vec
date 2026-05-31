# `dev/wrist2vec` Codex Engineering Index

This directory is the branch-scoped engineering manual for `dev/wrist2vec`.

## Branch Scope

- Branch: `dev/wrist2vec`
- Mode: `bootstrap`
- Purpose: track the side-by-side `wrist2vec` namespace fork while keeping `sleep2vec` as the behavior source of truth, plus the separate `wrist2vec_flex` recipe when source-aware flexible-input work intentionally diverges.

## Recommended Reading Order

1. This file for branch intent.
2. [REUSE_GUIDE.md](./REUSE_GUIDE.md) for the namespace-copy rules.
3. The current `sleep2vec/` source tree and top-level `configs/` recipes for implementation truth.

## Coverage

- `wrist2vec/`: side-by-side namespace copy of the base `sleep2vec/` runtime surface; keep committed baseline behavior stable.
- `wrist2vec_flex/`: standalone flexible-input recipe for source-aware channels, variable downstream channels, and flex-specific experiment logging.
- `configs/wrist2vec_*.yaml`: side-by-side user-facing recipe copies.
- `configs/wrist2vec_flex/`: flex-specific YAML recipes; do not mix them with baseline wrist configs.
- `README.md`, `pyproject.toml`, and focused tests/documentation that make the new namespace discoverable.

## Reliability Notes

- This bootstrap index is intentionally minimal. Use the tracked `sleep2vec/` code as the authoritative behavior reference.
- If behavior questions arise, inspect the matching `sleep2vec` implementation first, then verify the corresponding `wrist2vec` copy stayed aligned.
