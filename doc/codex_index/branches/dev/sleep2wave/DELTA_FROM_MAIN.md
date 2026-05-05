# Delta From Main

## Baseline

- Branch: `dev/sleep2wave`
- Indexed commit: `55458eba899c81026710d31c31a3143501d911bd`
- Main baseline: `12350da513fe1b011c8eb10671e75ca5f139857f`
- Merge base: `12350da513fe1b011c8eb10671e75ca5f139857f`

## Summary

The branch is primarily additive. It adds a standalone `sleep2wave` namespace, four sleep2wave configs, sleep2wave tests, and config-check routing. Existing base `sleep2vec`, top-level `data`, and top-level `preprocess` source remains covered by the main branch index.

## Added Surfaces

- `sleep2wave/`: 144 tracked files
- `configs/sleep2wave/`: 4 tracked YAML recipes
- `tests/test_sleep2wave_*.py`: 28 tracked test files

## Modified Surfaces

- `pyproject.toml`: adds `sleep2wave` to `known_first_party`.
- `utils/check_configs.py`: adds `CONFIG_VARIANTS` routing for `configs/sleep2wave`, detects `recipe: sleep2wave` generative configs, and validates them through `sleep2wave.generative.config.load_sleep2wave_config`.
- `tests/test_check_configs.py`: adds acceptance checks for sleep2wave generative configs.

## Runtime Meaning

The branch introduces sleep2wave as a branch-local package rather than extending the base `sleep2vec` namespace. New work should be explicit about which side it touches:

- Base sleep representation learning: keep using `sleep2vec/` and the main index guidance.
- sleep2wave standalone runtime or generation: use `sleep2wave/`, `configs/sleep2wave/`, and this branch index.

## New Workflow Paths

- Autoencoder: `python -m sleep2wave.train_autoencoder --config configs/sleep2wave/sleep2wave_autoencoder_tiny.yaml ...`
- Diffusion: `python -m sleep2wave.train_diffusion --config configs/sleep2wave/sleep2wave_diffusion_tiny_phase1.yaml ...`
- Generation: `python -m sleep2wave.generate --config configs/sleep2wave/sleep2wave_generate_tiny.yaml ...`
- Evaluation: `python -m sleep2wave.evaluate_generation --config configs/sleep2wave/sleep2wave_eval_tiny.yaml ...`

## New Tests

Key sleep2wave test groups:

- namespace and standalone RoFormer parity: `tests/test_sleep2wave_namespace.py`, `tests/test_sleep2wave_roformer_parity.py`
- generative config: `tests/test_sleep2wave_generative_config.py`
- data and preprocessing: `tests/test_sleep2wave_modalities.py`, `tests/test_sleep2wave_generative_dataset.py`, `tests/test_sleep2wave_preprocess_contract.py`
- autoencoder: `tests/test_sleep2wave_autoencoder_*.py`
- diffusion: `tests/test_sleep2wave_diffusion_*.py`, `tests/test_sleep2wave_phase_schedule.py`, `tests/test_sleep2wave_task_sampler.py`
- generation/export/evaluation: `tests/test_sleep2wave_generate_cli.py`, `tests/test_sleep2wave_export_artifacts.py`, `tests/test_sleep2wave_evaluate_cli.py`, `tests/test_sleep2wave_*metrics.py`, `tests/test_sleep2wave_sliding_window.py`

## Stale Main Entries

No main-index entries were copied into this branch directory as branch truth. Base-runtime guidance is referenced rather than duplicated where source did not change.
