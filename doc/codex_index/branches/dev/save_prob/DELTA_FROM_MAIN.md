# Delta From Main

## Branch

- Branch: `dev/save_prob`
- Baseline source: `doc/codex_index/branches/main/`
- Initialized at commit: `d68f7a3ac6bea8f932bed2d3e04705b466c36d8d`

## Runtime Export Changes

- Package-local `infer.py` files in `sleep2vec`, `sleep2vec2`, and `sleep2expert` automatically create inference result run directories under `results/inference/{namespace}/{label_name}/{prediction_run_id}`.
- Package-local `results.prepare_inference_result_paths` creates the shared `prediction_run_id`, run-local metrics/prediction paths, global `overview.csv` path, checkpoint tags, and manifest path.
- Package-local `results.save_result_csv` remains the aggregate metrics writer and now records run-directory, checkpoint, task-family, and output-path metadata for both run-local metrics and global overview rows.
- Package-local `results.save_prediction_csv` is the detailed per-path prediction writer and records the same run metadata used by aggregate rows.
- Package-local `results.save_inference_manifest` writes the self-contained `run_manifest.json` for each inference run.
- Package-local `sleep2vec_inference.py` modules build detailed prediction rows while `Sleep2vecFinetuning` preserves target extraction, stage remapping, DDP gathering, and AHI threshold semantics.

## Scope Notes

- This branch index was initialized from `main`; prediction CSV export now covers root `sleep2vec` and the active standalone `sleep2vec2` / `sleep2expert` recipe namespaces.
- Variant namespaces keep package-local implementations instead of importing root `sleep2vec` export helpers.
