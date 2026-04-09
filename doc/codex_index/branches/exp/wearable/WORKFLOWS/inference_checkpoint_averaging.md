# Workflow: Inference And Checkpoint Averaging

## Runtime Sequence

1. CLI enters `sleep2vec.infer.run_inference`.
2. `apply_finetune_config` resolves the same downstream model and task semantics used for finetuning.
3. `_build_inference_loader` builds a single evaluation dataloader for the requested split and dataset list.
4. `Sleep2vecFinetuning` is constructed in test-only mode.
5. If `--avg-ckpts > 1`, the runtime:
   - resolves checkpoint candidates with `select_checkpoints`
   - averages them with `average_checkpoints`
   - loads the averaged state directly into the model
6. Lightning `test()` runs on the selected or averaged checkpoint state.
7. Optional W&B logging is initialized only on rank zero.
8. Metrics are optionally appended to a CSV with `save_result_csv`.

## Canonical Checkpoint Rules

- `--ckpt-path best|last` requires `--avg-ckpt-dir` when checkpoint averaging is requested
- `select_checkpoints` prefers epoch ordering when filenames encode epochs
- when epoch parsing is not possible, checkpoint selection falls back to modification time
- averaged checkpoint loading uses non-strict `load_state_dict` so older checkpoints can tolerate limited schema drift

## Main Reuse Points

- config application: `sleep2vec.common.apply_finetune_config`
- loader build: `sleep2vec.utils._build_finetune_loader`
- checkpoint helpers: `sleep2vec.checkpoints.select_checkpoints`, `sleep2vec.checkpoints.average_checkpoints`
- results export: `sleep2vec.metrics.save_result_csv`

## Common Failure Checks

- wrong `label_name` relative to the finetune YAML
- `best` or `last` averaging without `--avg-ckpt-dir`
- checkpoint directory contains fewer checkpoints than requested
- CPU inference combined with bf16 precision; the runtime already falls back to `32`
