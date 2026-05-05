# Baseline Sleep2Vec Functions

## Scope

The base `sleep2vec/`, top-level `data/`, and top-level `preprocess/` source surfaces are inherited on `dev/sleep2wave`. Use the main branch function catalogs for unchanged base behavior:

- `doc/codex_index/branches/main/FUNCTIONS/CONFIG_AND_REGISTRIES.md`
- `doc/codex_index/branches/main/FUNCTIONS/RUNTIME_ORCHESTRATION.md`
- `doc/codex_index/branches/main/FUNCTIONS/MODELS_AND_HEADS.md`
- `doc/codex_index/branches/main/FUNCTIONS/DATASETS_AND_SAMPLERS.md`
- `doc/codex_index/branches/main/FUNCTIONS/PREPROCESSING_AND_CONVERSION.md`
- `doc/codex_index/branches/main/FUNCTIONS/VISUALIZATION_AND_DIAGNOSTICS.md`

## Branch-Specific Reuse Rule

When editing base behavior, consult the main index first. When editing Sleep2Wave behavior, prefer the package-local `sleep2wave.*` implementation and this branch index. Do not import base `sleep2vec`, top-level `data`, or top-level `preprocess` from new `sleep2wave` code unless the namespace boundary is intentionally changed.

## Package-Local Mirror

The copied `sleep2wave` runtime includes local equivalents of base config, registry, builders, model, downstream, dataset, metric, checkpoint, callback, visualization, and entrypoint modules. The package-local copies should be treated as active source, not thin re-exports.

Important local equivalents:

- `sleep2wave.config.load_pretrain_config`
- `sleep2wave.config.load_finetune_config`
- `sleep2wave.common.apply_finetune_config`
- `sleep2wave.pretrain.sleep2vec_pretrain`
- `sleep2wave.adapt.sleep2vec_adapt`
- `sleep2wave.finetune.supervised`
- `sleep2wave.infer.run_inference`
- `sleep2wave.pretrain_model.Sleep2vecPretrainModel`
- `sleep2wave.downstream_model.Sleep2vecDownstreamModel`
- `sleep2wave.data.default_dataset.DefaultDataset`
- `sleep2wave.data.samplers.PairFirstBatchSampler`
- `sleep2wave.metrics.compute_downstream_metrics`
- `sleep2wave.results.save_result_csv`

## Notable Branch Divergences

- `sleep2wave.config.load_finetune_config` rejects LoRA flags because the standalone RoFormer path does not support LoRA yet.
- `sleep2wave.checkpoints.load_pretrain_init_weights` rejects legacy HF-style RoFormer checkpoint keys when loading into the standalone RoFormer target.
- `utils/check_configs.py` routes configs under `configs/sleep2wave` through package-local config and preset helpers.

## Tests

- `tests/test_sleep2wave_namespace.py`
- `tests/test_sleep2wave_roformer_parity.py`
- `tests/test_check_configs.py`
