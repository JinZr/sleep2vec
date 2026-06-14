# dev/fine-grained-analysis Codex Engineering Index

This is the branch-specific engineering index for `dev/fine-grained-analysis`.

## Branch Scope

- Branch: `dev/fine-grained-analysis`
- Baseline commit inspected: `835cf7fcdf6de6646997cb8ea271f6edd55531ea`
- Last refresh at: `2026-06-13T16:22:35Z`
- Mode: `initialize-branch`

## How To Use This Index

- Use the main index under `doc/codex_index/branches/main/` for unchanged core `sleep2vec`,
  `sleep2vec2`, `sleep2expert`, data, preprocessing, runtime, and agent-tooling contracts.
- Use this branch index for `sleep2stat` package behavior and public output-table contracts introduced
  on this branch.
- For `sleep2stat` changes, start with [REUSE_GUIDE.md](./REUSE_GUIDE.md), then read
  [FUNCTIONS/SLEEP2STAT.md](./FUNCTIONS/SLEEP2STAT.md) and [WORKFLOWS/SLEEP2STAT.md](./WORKFLOWS/SLEEP2STAT.md).

## Coverage

- Indexed in detail: `sleep2stat/`, `configs/sleep2stat/`, and the matching `tests/test_sleep2stat_*`
  contracts.
- Baseline only: inherited main-branch code outside `sleep2stat`; consult the main branch index for those
  modules before editing them.
- Not indexed: local experiment outputs, untracked data indexes, and generated result bundles.
