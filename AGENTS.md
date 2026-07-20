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
- For small special-case handling changes, patch the canonical code path in place instead of adding a helper or wrapper; when the exception is not obvious, leave a brief comment noting the intention.
- Keep architecture and loss choices in YAML under `configs/`; training hyperparameters stay on the CLI.

## Code Design Simplicity Policy
- Start from the existing contract and canonical code path. Change the narrowest owner that already handles the behavior before adding new entrypoints, helpers, wrappers, or schema branches.
- Keep one canonical spelling and location for each concept. Do not add alternate config fields, nested aliases, source/value dictionaries, plural variants, or duplicate output names for the same semantic value.
- Remove obsolete semantic entrypoints and aliases from the canonical behavior. Reject legacy fields at user-input boundaries only when accepting them would create duplicate semantic sources, hidden fallbacks, silently ignored analysis intent, or incorrect outputs. Do not add bespoke validation for inert values or malformed values that the canonical code path will naturally reject.
- Treat paths and external identifiers exactly as the contract states. If a value should be passed through as provided, do not expand, resolve, normalize, infer a base directory, or silently rewrite it.
- Make semantic choices explicit and fail fast at boundaries. Defaults are acceptable for operational convenience, but model/data semantics, label sources, thresholds, stage sources, and output contracts should not be guessed.
- Name outputs by their real unit and denominator. If both ratio and percent are emitted, both names must say so; do not also emit ambiguous historical aliases.
- Let tests pin removals as well as additions. When deleting legacy behavior, test that hidden fallbacks and ambiguous legacy entrypoints fail validation or disappear from outputs; do not add tests that require special rejection of ordinary bad values unless they would otherwise produce misleading results.

## Runtime Failure & Output Directory Policy
- Prefer `let it fail` for repository-owned analysis and derived-artifact pipelines. If an analyzer, reducer, parser, or writer raises, let the command fail with a non-zero exit instead of converting it into a partial-success bundle.
- Do not add resume, repair, skip-existing, overwrite, or partial-rebuild protocols unless the user explicitly asks for them and the runtime need is concrete. For normal research runs, rerun with a fresh output directory or manually clear the failed output.
- `experiment-run` is the explicit resumable exception for managed external evaluation. It preserves frozen pipeline state and creates a fresh, empty result root for every attempt; resume never overwrites or reinterprets a prior attempt directory.
- Treat committed run directories as single-use artifacts. A non-empty output directory should generally fail before expensive work starts unless the tool has an explicit append-only contract.
- Write terminal manifests only after the run has completed successfully. Interrupted or failed directories should be considered invalid rather than interpreted through sidecars, shards, or partial tables.
- Keep summary commands read-only unless their contract explicitly says they repair or rebuild data. A `summarize`-style command should inspect committed outputs, not infer completion from partial intermediate files.

## Codex Index Usage Policy
- `doc/codex_index/` is a shared navigation layer, not a branch-specific or exhaustive source of truth. Source code, tests, `AGENTS.md`, and dedicated contract documents remain authoritative.
- Small localized fixes and routine updates may inspect source and tests directly without consulting the index.
- Consult the shared index when a change crosses modules, adds a reusable implementation, or has unclear ownership. Start with `README.md`, then use `MODULE_MAP.md`, `REUSE_GUIDE.md`, or `WORKFLOWS.md` as needed.
- Before adding a function, method, helper, wrapper, or utility, search the source and `REUSE_GUIDE.md` for an existing implementation with the same responsibility. Prefer minimally extending the canonical owner.
- Never create feature-branch index copies, `branches/` trees, or `DELTA_FROM_MAIN.md` files.
- Update the shared index only when a change affects ownership, a canonical reusable implementation, a public contract, a key workflow, or a top-level entrypoint.
- Do not update it for local fixes, routine refactors, or inventories of signatures, callers, file counts, commits, or branch metadata.
- Keep the index limited to `README.md`, `MODULE_MAP.md`, `REUSE_GUIDE.md`, and `WORKFLOWS.md`.

## Testing Guidelines
- Use targeted pytest files for contract changes and smoke commands for runtime changes.
- There is a checked `tests/` suite; prefer the smallest relevant test set for the ownership boundary touched.
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

## Agent Stop-And-Consult Policy
Agents must not silently guess high-impact experiment decisions.

Before generating runnable commands for preset preparation, finetuning, inference, evaluation, or hyper-parameter tuning, run the relevant agent consultation checks through `agent_tools doctor` or `agent_tools plan`. `agent_tools context` is diagnostic-only and does not authorize runnable commands.

If the tool returns `NEEDS_USER_INPUT`, stop and ask the user the generated questions. Do not run training. Do not generate executable scripts. Do not evaluate external test data.

Generated runtime commands must respect recipe `variant`; do not route `sleep2vec2` or `sleep2expert` recipes through root `sleep2vec` entrypoints.

High-impact decisions include label selection, split policy, external-test locking, checkpoint selection, pretrained-backbone choice, preset regeneration, overwrite behavior, required channels, selection metric, metric direction, and hyper-parameter search budget.

## Experiment Management Policy

- Every new runnable recipe and plan must declare an `experiment` with `id`, `title`, `objective`, `root`, and `baseline`, plus a `step` with `id`, `phase`, and `purpose`.
- Runnable plans must be written inside `experiment.root`. Use the workspace as the human entry point; keep heavyweight datasets, checkpoints, W&B files, and trainer logs in their canonical locations and record links to them.
- Runs must have both a stable `run-NNN` id and a semantic parameter-derived name. Do not create new `trial_000`-style artifact names that require opening configs to understand.
- Freeze the resolved recipe, generated config, launch command, and hashes before execution. Do not silently rewrite a planned run after launch.
- Monitoring may update status and reports but must not start pending runs. Launching is an explicit action.
- `experiment-run --execute` is an explicit launching action: it may wait for successful managed training sources and then fill its frozen external-evaluation matrix. `hparam-monitor` and `experiment-monitor` remain non-launching.
- External-evaluation pipelines must select and freeze checkpoints from validation evidence before reading external metrics. Finalization requires every declared external job to have one verified successful manifest and no active attempt.
- Stopping a run requires a recorded reason. Finalization requires no active runs and a non-empty final report.
- This policy applies to new work. Do not migrate or rename historical experiment trees unless the user explicitly asks.

## Config Strictness Policy
- Follow “let it crash” for model/data semantics: missing or inconsistent YAML fields that affect model shape, task semantics, or evaluation should raise immediately.
- Defaults are acceptable only for optimization/logging/runtime convenience (e.g., epochs, lr, batch size, W&B metadata).
- When adding new config fields, mark explicitly whether they are required or optional and enforce it in config parsing.

## Codex Subagent Operating Model
- `AGENTS.md` defines subagent ownership and routing policy, but does not automatically spawn or schedule subagents. The parent Codex agent must still choose which subagent(s) to invoke.
- Split work by cross-file contracts and runtime coupling, not by raw file count.
- When a task crosses multiple ownership boundaries, assign one lead owner and request review from the adjacent owner instead of letting two agents edit the same contract blindly.
- If a task touches `sleep2vec_moe/` or `sleep2vec2/`, treat variant validation as mandatory before calling the work complete.

### Default Subagent Catalog

#### `data-contract-guardian`
- Owns: `data/default_dataset.py`, `data/psg_pretrain_dataset.py`, `data/utils.py`, `data/samplers.py`, `data/channel_selection.py`, `data/metadata.py`.
- Responsibilities: sample index semantics, `available_channels`, `source` and metadata filtering, token-window validity, few-shot behavior, pair-first batching, collate invariants.
- Invoke when: changing dataset fields, preset payload shape, filtering rules, missing-channel handling, sampler behavior, or data/label loading semantics.
- Must not be split from: pair-first and available-channel tests in `tests/data/test_pair_first_sampler.py`, `tests/data/test_bucket_sampler.py`, and `tests/data/test_data_utils.py`.
- Verification gate:
```bash
PYTHONPYCACHEPREFIX=/tmp/sleep2vec_pycache python3 -m compileall data tests
python3.10 -m pytest -q tests/data/test_pair_first_sampler.py tests/data/test_bucket_sampler.py tests/data/test_data_utils.py
```

#### `config-task-contract`
- Owns: `sleep2vec/config.py`, `sleep2vec/common.py`, `sleep2vec/builders.py`, `sleep2vec/registry.py`, `sleep2vec/backbones/encoder_factory.py`, `sleep2vec/modules/tokenizers.py`, `sleep2vec/modules/projection.py`.
- Responsibilities: YAML schema strictness, built-in task semantics, channel parity checks, registry wiring, builder contracts, required vs optional config fields.
- Invoke when: adding config fields, changing task semantics, changing registry names, changing builder signatures, or changing YAML validation behavior.
- Must not be split from: `tests/config/test_config_loading.py`, `tests/config/test_common_finetune_apply.py`, `tests/config/test_metadata_task_validation.py`, `tests/config/test_registries_and_builders.py`.
- Verification gate:
```bash
PYTHONPYCACHEPREFIX=/tmp/sleep2vec_pycache python3 -m compileall sleep2vec tests
python3.10 -m pytest -q \
  tests/config/test_config_loading.py \
  tests/config/test_common_finetune_apply.py \
  tests/config/test_metadata_task_validation.py \
  tests/config/test_registries_and_builders.py
```

#### `model-integration`
- Owns: `sleep2vec/pretrain_model.py`, `sleep2vec/downstream_model.py`, `sleep2vec/sleep2vec_modelling.py`, `sleep2vec/sleep2vec_finetuning.py`, `sleep2vec/losses/`, `sleep2vec/downstreams/`, `sleep2vec/cls/`, `sleep2vec/modules/layer_mix.py`, `sleep2vec/visualization/layer_mix.py`.
- Responsibilities: tokenization-to-backbone flow, CLS semantics, layer-mix behavior, downstream head contracts, pretrained-backbone loading, LoRA insertion, loss/head interaction.
- Invoke when: changing forward shapes, CLS or token-mask semantics, head interfaces, loss output contracts, layer-mix logic, or pretrained weight loading behavior.
- Must not be split from: `sleep2vec/config.py` task/model semantics when interface changes are involved; request review from `config-task-contract` in that case.
- Verification gate:
```bash
PYTHONPYCACHEPREFIX=/tmp/sleep2vec_pycache python3 -m compileall sleep2vec tests
python3.10 -m pytest -q \
  tests/models/test_losses.py \
  tests/visualization/test_layer_mix_visualization.py \
  tests/config/test_registries_and_builders.py
```

#### `runtime-orchestrator`
- Owns: `sleep2vec/pretrain.py`, `sleep2vec/finetune.py`, `sleep2vec/infer.py`, `sleep2vec/checkpoints.py`, `sleep2vec/metrics.py`, `sleep2vec/callbacks/pair_acc_logger.py`, `sleep2vec/diagnostics.py`.
- Responsibilities: trainer wiring, W&B behavior, checkpoint naming and averaging, inference execution, results CSV schema, diagnostics mode, runtime callbacks and logging.
- Invoke when: changing CLI flags, trainer strategies, checkpoint alias behavior, runtime logging, diagnostics flow, or evaluation/export behavior.
- Must not be split from: `tests/runtime/test_checkpoints.py` and config/task guard tests when runtime flags or monitor names change.
- Verification gate:
```bash
PYTHONPYCACHEPREFIX=/tmp/sleep2vec_pycache python3 -m compileall sleep2vec tests
python3.10 -m pytest -q \
  tests/runtime/test_checkpoints.py \
  tests/config/test_common_finetune_apply.py \
  tests/config/test_metadata_task_validation.py
```
- Smoke gate:
```bash
WANDB_MODE=offline python3.10 -m sleep2vec.pretrain \
  --config configs/sleep2vec_dense_pretrain.yaml \
  --version-name runtime-smoke \
  --print-diagnostics --diagnostics-steps 5 --precision 32 --devices 0
```

#### `preset-pipeline`
- Owns: `preprocess/save_dataset_presets.py`, `preprocess/merge_dataset_presets.py`, `preprocess/split_index_by_dataset.py`, `preprocess/mask_missing_stats.py`, `preprocess/preprocess_pipeline.ipynb`.
- Responsibilities: index splitting, preset generation, preset merging, missing-channel statistics, preprocessing documentation and reproducible data-prep flows.
- Invoke when: changing preset pickle format, split policy, preprocessing CLI shape, or dataset preparation workflow.
- Must not be split from: `data-contract-guardian` when a preprocessing change affects runtime `SampleIndex` payload semantics.
- Verification gate:
```bash
PYTHONPYCACHEPREFIX=/tmp/sleep2vec_pycache python3 -m compileall preprocess data
python preprocess/save_dataset_presets.py --help
python preprocess/merge_dataset_presets.py --help
python preprocess/split_index_by_dataset.py --help
python preprocess/mask_missing_stats.py --help
```

#### `variant-maintainer`
- Owns: `sleep2vec_moe/`, `configs_moe/`, and variant-specific `sleep2vec2/` files, especially `sleep2vec2/backbones/encoder_factory.py` and `sleep2vec2/backbones/roformer/`.
- Responsibilities: MoE config/task extensions, router/expert behavior, MoE callbacks and logging, base-to-variant parity checks, symlinked variant safety.
- Invoke when: changing any shared base contract that might affect `sleep2vec_moe` or `sleep2vec2`, or when editing variant-only files directly.
- Must not be skipped for: backbone API changes, CLS/task config changes, checkpoint-loading changes, callback/logging changes, or tokenizer/projection interface changes.
- Verification gate:
```bash
PYTHONPYCACHEPREFIX=/tmp/sleep2vec_pycache python3 -m compileall sleep2vec_moe sleep2vec2
python3.10 -m pytest -q tests/runtime/test_checkpoints.py tests/config/test_config_loading.py
```

#### `agent-tooling-maintainer`
- Owns: `skills/`, `agent_tools/`, `recipes/`, `agent_policies/`, `doc/agent_contracts/`, and agent-facing workflow examples.
- Responsibilities: task playbooks, machine-readable recipe and pipeline schemas, context bundle generation, command-plan generation, managed scheduling, skill validation, run-manifest conventions, agent documentation consistency, stop-and-consult policy enforcement, and external-test unlock checks.
- Invoke when: adding or changing agent skills, recipe or pipeline schemas, context-gathering tools, run-plan generators, consultation policies, user-decision files, managed experiment pipelines, or agent-facing documentation.
- Must not be split from: `runtime-orchestrator` when the change affects training/inference command semantics; `preset-pipeline` when the change affects preset preparation; `regression-guard` when adding new agent-tool contracts.
- Verification gate:
```bash
PYTHONPYCACHEPREFIX=/tmp/sleep2vec_pycache python3 -m compileall agent_tools tests
python3 -m pytest -q tests/agent_tools/test_agent_tools_*.py tests/agent_tools/test_agent_consultation_policy.py tests/agent_tools/test_agent_user_decisions.py tests/agent_tools/test_agent_plan_blocks_on_ambiguity.py
python -m agent_tools skills --validate
```

#### `regression-guard`
- Owns: the `tests/` strategy rather than a product module.
- Responsibilities: add or extend tests when a contract changes, identify missing coverage, and prevent silent regressions in data/config/runtime/variant flows.
- Invoke when: a change modifies an existing contract, fixes a regression, or introduces a new CLI/YAML/runtime branch.
- Default expectation: every non-trivial contract change should either update an existing test or document why no test was added.

### Routing Rules
- If a task changes `data/` semantics or sampler behavior, route first to `data-contract-guardian`.
- If a task changes YAML fields, task semantics, or registries, route first to `config-task-contract`.
- If a task changes model forward paths, head/loss contracts, LoRA, or layer-mix behavior, route first to `model-integration`.
- If a task changes entrypoints, checkpoints, metrics, callbacks, diagnostics, or inference/export behavior, route first to `runtime-orchestrator`.
- If a task changes preprocessing scripts or preset generation logic, route first to `preset-pipeline`.
- If a task changes agent-facing skills, recipes, context bundles, command plans, consultation gates, or user-decision schemas, route first to `agent-tooling-maintainer`.
- If a task touches `sleep2vec_moe/`, `configs_moe/`, or `sleep2vec2/`, require `variant-maintainer` review before completion.
- If a task changes a contract already covered by tests, or should be covered but is not, involve `regression-guard`.

### Do Not Split
- Keep `data/default_dataset.py`, `data/psg_pretrain_dataset.py`, `data/utils.py`, and `data/samplers.py` under one owner for a single change.
- Keep `sleep2vec/config.py`, `sleep2vec/common.py`, and `sleep2vec/builders.py` under one owner for a single change.
- Keep `sleep2vec/pretrain_model.py`, `sleep2vec/downstream_model.py`, `sleep2vec/sleep2vec_modelling.py`, and `sleep2vec/sleep2vec_finetuning.py` under one owner for a single change.
- Keep `sleep2vec/pretrain.py`, `sleep2vec/finetune.py`, `sleep2vec/infer.py`, `sleep2vec/checkpoints.py`, and `sleep2vec/metrics.py` under one owner for a single change.

### Handoff Format
- Every subagent report should include:
  - scope touched
  - files changed or reviewed
  - contract assumptions
  - verification commands run
  - blockers or unverified parts
- If verification could not run because `python3.10` or `pytest` is unavailable, state that explicitly instead of implying the gate passed.
