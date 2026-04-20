# Infer And Checkpoints Workflow

## Purpose

Evaluate a downstream checkpoint on a selected split, optionally average multiple checkpoints first.

## Entry Command

Canonical entrypoint: `python -m sleep2vec.infer --config ... --ckpt-path ... --label-name ...`

Primary code path:

1. `sleep2vec.infer.parse_args`
2. `sleep2vec.infer.run_inference`
3. `sleep2vec.common.apply_finetune_config`
4. `sleep2vec.infer._build_inference_loader`
5. optional `sleep2vec.checkpoints.select_checkpoints`
6. optional `sleep2vec.checkpoints.average_checkpoints`
7. Lightning `trainer.test(...)`

## Detailed Flow

1. Parse CLI and validate checkpoint path.
   - Real file path must exist.
   - Aliases `best` and `last` are allowed.
2. Bind finetune YAML into `args`.
3. Build a single evaluation loader.
   - `eval_split` selects `train`, `val`, or `test`.
   - `override_dataset_names` can replace configured dataset lists.
4. Instantiate `Sleep2vecFinetuning`.
5. Optionally average checkpoints.
   - `avg_ckpts == 1`: use `ckpt_path` directly.
   - `avg_ckpts > 1`: choose candidate files from `avg_ckpt_dir` or checkpoint parent.
   - When using `best` or `last` with averaging, `avg_ckpt_dir` is required because the alias is not a concrete file.
   - Built-in `ahi` rejects averaging because the stored threshold is checkpoint-specific.
6. Run evaluation.
7. Optionally append metrics to a CSV through `sleep2vec.results.save_result_csv`.

## Checkpoint Selection Policy

- Prefer epoch ordering when file names encode epoch.
- Fall back to modification time ordering when epoch numbers are absent.
- Reject missing checkpoint directories, empty directories, and insufficient candidate counts.

## Checkpoint Averaging Policy

- Supports raw tensor dicts, Lightning `state_dict`, and `model` wrappers.
- Floating-point tensors are averaged arithmetically.
- Non-floating tensors are integer-divided after summation.
- Missing keys across checkpoints raise immediately.

## Important Runtime Decisions

- `infer.py` is the reviewed place that handles CPU precision fallback for `bf16`.
- Inference can initialize W&B separately from training and only on rank zero.
- Metric computation remains inside `Sleep2vecFinetuning`, so inference reuses finetune epoch-reduction logic rather than a separate evaluation module.
- AHI inference requires `ahi_eval_threshold` to be present in the checkpoint state.

## Edit Hotspots

- Change checkpoint averaging semantics: `sleep2vec/checkpoints.py`
- Change eval loader or dataset override behavior: `sleep2vec/infer.py`, `sleep2vec/utils.py`
- Change inference-only logging/W&B behavior: `sleep2vec/infer.py`
