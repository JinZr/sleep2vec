# Delta From `main`

## Scope

This page records branch-local differences that matter for reuse and review on `fix/result-writing-issue`.

## Branch-specific differences

- `sleep2vec.finetune.supervised` now tracks whether a W&B run already existed before the finetune call, finishes only the run created during this call, and downgrades teardown failures to warnings when a primary train/test error is already active.
- `tests/test_ahi_event_metrics.py` adds regression coverage for:
  - finishing an owned W&B run after CSV export
  - leaving a preexisting W&B run open
  - preserving the primary runtime error when W&B cleanup also fails

## Reuse impact

- Reuse the ownership-aware W&B cleanup path in `sleep2vec.finetune.supervised` rather than adding broader `wandb.finish()` calls in other entrypoints.
