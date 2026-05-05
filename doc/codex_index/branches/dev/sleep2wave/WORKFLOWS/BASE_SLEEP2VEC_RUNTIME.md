# Base Sleep2Vec Runtime Workflow

## Purpose

Use this page when a change on `dev/sleep2wave` touches the inherited base `sleep2vec` runtime rather than the additive Sleep2Wave generative stack.

## Canonical Reference

The source for base runtime guidance is still the main branch index:

- `doc/codex_index/branches/main/WORKFLOWS/PRETRAIN.md`
- `doc/codex_index/branches/main/WORKFLOWS/ADAPT.md`
- `doc/codex_index/branches/main/WORKFLOWS/FINETUNE.md`
- `doc/codex_index/branches/main/WORKFLOWS/INFER_AND_CHECKPOINTS.md`
- `doc/codex_index/branches/main/WORKFLOWS/PREPROCESSING.md`
- `doc/codex_index/branches/main/WORKFLOWS/CONFIG_VALIDATION.md`

## Branch Rule

- For base configs and base source, use `sleep2vec.*`, top-level `data.*`, and top-level `preprocess.*`.
- For package-local Sleep2Wave configs or runtime, use `sleep2wave.*`.
- Do not mix base and package-local imports in new Sleep2Wave code.

## Validation

Base workflow validation remains the same as main unless the branch changes base source:

- `python -m sleep2vec.pretrain ...`
- `python -m sleep2vec.finetune ...`
- `python -m sleep2vec.infer ...`
- `python -m sleep2vec.adapt ...`
- `python utils/check_configs.py`

For Sleep2Wave package-local runtime changes, add the relevant `tests/test_sleep2wave_*.py` coverage instead of relying only on base tests.
