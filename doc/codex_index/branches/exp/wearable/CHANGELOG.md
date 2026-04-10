# Branch Index Changelog

## 2026-04-10 17:43 +0800

- refreshed handbook snapshot to `c6cee5cef93c4228b2b11cfd79a83becdbe059a0`
- recorded dirty working-tree status and noted that untracked files were excluded from branch-canonical indexing
- updated config, data-loading, finetune, metrics, and reuse docs to describe built-in sleep-staging tasks `stage3`, `stage4`, and `stage5`
- documented that `stage3` and `stage4` reuse raw `stage5` token labels plus runtime remapping rather than introducing new input channels

Stale entries removed: stage5-only downstream task descriptions. This was a `refresh` run.

## 2026-04-09 16:44 +0800

- initialized branch handbook for `exp/wearable`
- created required root files: `README.md`, `MANIFEST.json`, `SYSTEM_OVERVIEW.md`, `MODULE_MAP.md`, `REUSE_GUIDE.md`, `CHANGELOG.md`, `DELTA_FROM_MAIN.md`
- created grouped function references under `FUNCTIONS/`
- created grouped workflow references under `WORKFLOWS/`
- documented branch delta from `main` using `git diff main...HEAD`
- recorded that `sleep2vec2/`, `sleep2vec_moe/`, and `sleep2vec_hires/` have no tracked source files on this branch

Stale entries removed: none. This was an `initialize-branch` run.
