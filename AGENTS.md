# Repository Guidelines

## Project Structure & Module Organization
- `sleep2vec/` is the core library and CLI entrypoints (e.g., `pretrain.py`, `finetune.py`, `infer.py`).
- `configs/` holds YAML recipes that define model/loss/head settings.
- `data/` contains dataset loaders, samplers, and metadata helpers.
- `preprocess/` hosts scripts for building index CSVs and preset pickles.
- `utils/` contains helper scripts such as repo-wide style checks; `doc/` stores assets.

## Architecture Overview
- Config-driven design: model/loss/head selection lives in YAML under `configs/`, while schedule and runtime options are passed on the CLI.
- Modular registries enable plug-in components (backbones, tokenizers, projections, losses, and downstream heads).
- Training flow: pretrain (`sleep2vec.pretrain`) builds the backbone; finetune (`sleep2vec.finetune`) attaches a downstream head; infer (`sleep2vec.infer`) evaluates checkpoints without training.

## Build, Test, and Development Commands
Install dependencies (select the correct PyTorch wheel for your CUDA version):
```bash
pip install -r requirements.txt
```
Common entrypoints:
```bash
python -m sleep2vec.pretrain --config configs/sleep2vec_dense_pretrain.yaml ...
python -m sleep2vec.finetune --config configs/sleep2vec_dense_finetune_cls.yaml ...
python -m sleep2vec.infer --config configs/sleep2vec_dense_finetune_cls.yaml ...
```
Formatting/linting:
```bash
bash utils/style_check.sh
```

## Coding Style & Naming Conventions
- Python formatting is enforced by Black (line length 120), isort (Black profile), and Flake8.
- Use 4-space indentation; follow snake_case for functions/variables/modules and PascalCase for classes.
- Keep architecture and loss choices in YAML under `configs/`; training hyperparameters stay on the CLI.

## Testing Guidelines
- There is no dedicated test suite checked in; use diagnostics and short runs for sanity checks.
- Quick smoke test example:
```bash
python -m sleep2vec.pretrain --config configs/sleep2vec_dense_pretrain.yaml \
  --print-diagnostics --diagnostics-steps 5 --precision 32 --devices 0
```
- If adding tests, place them under `tests/` with `test_*.py` naming.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative sentences with an initial capital; include issue/PR refs when relevant (e.g., "Add diagnostics hooks (#17)").
- PRs should summarize the change, list commands run, and include artifacts for behavior changes (e.g., W&B run link or `results.csv`).

## Configuration & Secrets
- W&B login is required by default; set `WANDB_API_KEY` or use `WANDB_MODE=offline`.
- Keep dataset paths and preset pickles in config/CLI (not hardcoded); document any new index CSV columns.

## W&B Naming (Recipe-Driven)
- W&B naming must follow the active recipe definition, not a single global project name.
- Derive a recipe tag from the config/entrypoint family (for example: `base`, `hires`, or another recipe namespace) and include it in both project and run names.
- Recommended pattern:
  - project: `sleep2vec-<recipe>-<stage>` where `<stage>` is `pretrain`, `finetune`, or `infer`
  - run name prefix: `s2v-<recipe>-<stage>-`
- Keep recipe namespaces separated so dashboards, sweeps, and reports never mix runs from different recipes.
- Current hires mapping follows this policy:
  - pretrain: `sleep2vec-hires-pretrain`, `s2v-hires-pretrain-*`
  - finetune: `sleep2vec-hires-finetune`, `s2v-hires-finetune-*`
  - infer: `sleep2vec-hires-infer` (or override still including `hires`)

## Config Strictness Policy
- Follow “let it crash” for model/data semantics: missing or inconsistent YAML fields that affect model shape, task semantics, or evaluation should raise immediately.
- Defaults are acceptable only for optimization/logging/runtime convenience (e.g., epochs, lr, batch size, W&B metadata).
- When adding new config fields, mark explicitly whether they are required or optional and enforce it in config parsing.
