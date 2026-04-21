# Reuse Guide

This branch exists to add a parallel `wrist2vec` namespace, not to redesign the runtime.

## Reuse Rules

- Reuse the existing `sleep2vec` implementation shape as the baseline for every `wrist2vec` module.
- Prefer mechanical namespace renames over behavioral edits.
- Keep shared helpers under `data/`, `preprocess/`, and `utils/` unchanged unless they block the new `wrist2vec` entrypoints.
- When a `wrist2vec` file needs a bug fix during this fork, patch the equivalent `sleep2vec` file first if the bug affects both namespaces; only fork behavior intentionally.

## Canonical Sources

| Responsibility | Canonical source to mirror | Do not replace with |
| --- | --- | --- |
| Runtime behavior | `sleep2vec/` module with the same role | New wrist-specific logic |
| Example YAML contracts | Matching top-level file in `configs/` | Newly invented recipe structure |
| CLI/docs wording | Existing `README.md` command surfaces | Drifted wrist-only instructions |

## Non-Goals For This Branch

- No new model, loss, data, or checkpoint semantics.
- No variant-tree fork for `sleep2vec_moe/`, `sleep2vec_hires/`, or `sleep2vec2/`.
- No refactor of shared preprocessing or dataset code just to make naming cleaner.
