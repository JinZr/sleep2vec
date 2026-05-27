# Infer And Checkpoints Workflow

## Purpose

Evaluate a downstream checkpoint on a selected split, optionally average multiple checkpoints first, and write automatic metrics, prediction, overview, and manifest artifacts.

## Entry Command

Canonical entrypoint: `python -m sleep2vec.infer --config ... --ckpt-path ... --label-name ...`

Primary code path:

1. `sleep2vec.infer.parse_args`
2. `sleep2vec.infer.run_inference`
3. `sleep2vec.common.apply_finetune_config`
4. `sleep2vec.infer._build_inference_loader`
5. optional `sleep2vec.checkpoints.select_checkpoints`
6. optional `sleep2vec.checkpoints.average_checkpoints`
7. `sleep2vec.results.prepare_inference_result_paths`
8. Lightning `trainer.test(...)`
9. `sleep2vec.results.save_result_csv`
10. `sleep2vec.results.save_prediction_csv`
11. `sleep2vec.results.save_inference_manifest`

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
6. Prepare inference output paths.
   - `prediction_run_id` includes timestamp, namespace, experiment version, label, split, checkpoint tag, and a short hash.
   - run-local outputs live under `results/inference/<namespace>/<label>/<prediction_run_id>/`.
   - `overview.csv` lives at `results/inference/overview.csv`.
7. Run evaluation.
8. Write metrics, predictions, overview, and manifest outputs.
   - `metrics__<label>__<split>__<ckpt_tag>.csv`: run-local metrics.
   - `predictions__<label>__<split>__<ckpt_tag>.csv`: one row per path, with list values serialized for CSV.
   - `overview.csv`: append-only cross-run summary.
   - `run_manifest.json`: machine-readable run metadata, paths, checkpoint identity, runtime settings, metrics, and prediction-row count.
9. If W&B is enabled, log the metrics plus `prediction_row_count` and upload the metrics, predictions, manifest, and overview files as one `inference-<prediction_run_id>` artifact.

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
- Backend-specific loader construction is still delegated through `_build_finetune_loader`.
- Inference can initialize W&B separately from training and only on rank zero; W&B artifact logging happens after local output files are written.
- Metric computation remains inside `Sleep2vecFinetuning`, so inference reuses finetune epoch-reduction logic rather than a separate evaluation module.
- Prediction row extraction is split out to `sleep2vec/sleep2vec_inference.py`; CSV writing and run metadata stay in `sleep2vec/results.py`.
- Non-AHI prediction export deduplicates repeated `(path, token_start)` records before building per-path rows.
- AHI inference requires `ahi_eval_threshold` to be present in the checkpoint state.

## Edit Hotspots

- Change checkpoint averaging semantics: `sleep2vec/checkpoints.py`
- Change eval loader, data-backend, or dataset override behavior: `sleep2vec/infer.py`, `sleep2vec/common.py`, `sleep2vec/utils.py`, `data/kaldi_psg_dataset.py`
- Change inference-only logging/W&B behavior: `sleep2vec/infer.py` plus package-local variant mirrors when parity is required
- Change prediction row extraction: `sleep2vec/sleep2vec_inference.py`, `sleep2vec/sleep2vec_finetuning.py`
- Change inference output layout, metadata, or CSV/manifest writes: `sleep2vec/results.py`
