# Infer And Checkpoints Workflow

## Purpose

Evaluate a downstream checkpoint on a selected split, optionally average multiple checkpoints first, and automatically write aggregate metrics, per-path prediction details, a run manifest, and a global overview row.

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
8. `sleep2vec.results.prepare_inference_result_paths`
9. `sleep2vec.results.save_result_csv`
10. `sleep2vec.sleep2vec_inference.build_prediction_rows` / `build_ahi_prediction_rows`
11. `sleep2vec.results.save_prediction_csv`
12. `sleep2vec.results.save_inference_manifest`

## Detailed Flow

1. Parse CLI and validate checkpoint path.
   - Real file path must exist.
   - Aliases `best` and `last` are allowed.
2. Bind finetune YAML into `args`.
3. Build a single evaluation loader.
   - `eval_split` selects `train`, `val`, or `test`.
   - `override_dataset_names` can replace configured dataset lists.
   - `data_backend=npz` can use `--inference-preset-path` to override YAML `data.finetune_preset_path`.
   - `data_backend=kaldi` uses `kaldi_data_root` and `kaldi_manifest`; `--inference-preset-path` is rejected.
4. Instantiate `Sleep2vecFinetuning`.
5. Optionally average checkpoints.
   - `avg_ckpts == 1`: use `ckpt_path` directly.
   - `avg_ckpts > 1`: choose candidate files from `avg_ckpt_dir` or checkpoint parent.
   - When using `best` or `last` with averaging, `avg_ckpt_dir` is required because the alias is not a concrete file.
   - Built-in `ahi` rejects averaging because the stored threshold is checkpoint-specific.
6. Create automatic inference output paths under `results/inference`.
   - Run directory: `results/inference/{namespace}/{label_name}/{prediction_run_id}/`.
   - Run-local metrics: `metrics__{label_name}__{eval_split}__{ckpt_tag}.csv`.
   - Run-local predictions: `predictions__{label_name}__{eval_split}__{ckpt_tag}.csv`.
   - Run manifest: `run_manifest.json`.
   - Global overview: `results/inference/overview.csv`.
7. Run evaluation.
8. Append one aggregate metrics row to the run-local metrics CSV and one matching row to the global overview CSV through `sleep2vec.results.save_result_csv`.
9. Append per-path prediction details to the run-local prediction CSV through `sleep2vec.results.save_prediction_csv`.
10. Write `run_manifest.json` through `sleep2vec.results.save_inference_manifest`.
   - `prediction_run_id` is shared by the aggregate row, overview row, detailed rows, manifest, and run directory name.
   - Checkpoint-tagged file names use checkpoint payload `epoch` / `global_step` when available, falling back to checkpoint filename parsing.

## Checkpoint Selection Policy

- Prefer epoch ordering when file names encode epoch.
- Fall back to modification time ordering when epoch numbers are absent.
- Reject missing checkpoint directories, empty directories, and insufficient candidate counts.
- Automatic result organization records the selected checkpoint paths in the run manifest when checkpoint averaging is active.

## Checkpoint Averaging Policy

- Supports raw tensor dicts, Lightning `state_dict`, and `model` wrappers.
- Floating-point tensors are averaged arithmetically.
- Non-floating tensors are integer-divided after summation.
- Missing keys across checkpoints raise immediately.

## Important Runtime Decisions

- `infer.py` is the reviewed place that handles CPU precision fallback for `bf16`.
- Backend-specific loader construction is still delegated through `_build_finetune_loader`.
- Inference can initialize W&B separately from training and only on rank zero.
- Inference result export is automatic; users no longer need to provide result or prediction CSV paths for the top-level inference entrypoints.
- Metric computation remains inside `Sleep2vecFinetuning`, so inference reuses finetune epoch-reduction logic rather than a separate evaluation module.
- Detailed prediction export keeps row extraction and aggregation in package-local `sleep2vec_inference.py` modules while `Sleep2vecFinetuning` still owns target extraction, DDP gathering, metric reduction, and AHI threshold reuse.
- AHI inference requires `ahi_eval_threshold` to be present in the checkpoint state.
- `sleep2vec2` and `sleep2expert` mirror this prediction-export contract through package-local `infer.py`, `results.py`, and `sleep2vec_inference.py` implementations.

## Edit Hotspots

- Change checkpoint averaging semantics: `sleep2vec/checkpoints.py`
- Change eval loader, data-backend, or dataset override behavior: `sleep2vec/infer.py`, `sleep2vec/common.py`, `sleep2vec/utils.py`, `data/kaldi_psg_dataset.py`
- Change inference-only logging/W&B, automatic result organization, or prediction-export behavior: package-local `infer.py`, `results.py`, `sleep2vec_inference.py`, and the thin `sleep2vec_finetuning.py` lifecycle glue.
