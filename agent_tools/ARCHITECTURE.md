# agent_tools architecture

`agent_tools` is a reusable research-experiment agent-control framework with a
sleep2vec domain adapter layer on top. This document is the human-readable
companion to `layering.py` (the machine-readable module partition the layering
guard test reads). Keep them in sync — `test_agent_layering.py` fails if the
mixed-module set drifts.

## Layers

Import direction is strictly one-way: **L2 → L1 → L0**, with `domain/` a
L0-level domain leaf.

| Layer | Contents | Role |
|---|---|---|
| **L0 leaves** | models, decision_models, transport, manifests, schema_map, gpu_rules, repo, plan_rendering, decision_paths, decision_hparam, plan_hparam, adaptive_proposals, experiment_workspace, experiment_io, managed_scheduler, ... | No intra-package deps beyond other L0 leaves; the reusable primitives. |
| **L1 `adapters/`** | `base` (TaskAdapter protocol), `registry` (get_adapter / all_adapters / composite_adapter), 6 per-task plugins, `config_providers` | Generic plugin skeleton + domain plugins. Kernel dispatches through the registry and never hardcodes task names. |
| **L2 kernel** | configs, decision_rules, decisions, plan_context, plans, experiment_pipeline | Orchestration over lower-layer owners and adapter declarations; authored task recipes remain governed by schema_map. |
| **`domain/`** | sidecar_summaries, finetune_summary, sex_age_summary, presets, index_csv | sleep2vec-specific summaries/validators. L0-level leaves that must not be aggregated in `domain/__init__` (would trigger a partial-import cycle via configs). |

## Module ownership

Mirrors the three frozensets in `layering.py`.

### Kernel — reusable (26, zero domain signal)
decision_models, transport, manifests, schema_map, gpu_rules, repo,
experiment_io, experiment_workspace, experiment_tracking, experiments,
run_artifacts, run_evidence, hparam, hparam_runtime, hparam_selection,
adaptive_hparam, adaptive_proposals, recipes, progress, markdown, skills,
decisions, plans, decision_rules, managed_scheduler, experiment_pipeline.

These must stay domain-free — the layering guard allows them **no** domain
imports.

`adaptive_proposals` owns the pure snapshot, parameter-envelope, and external
submission-validation contract. `adaptive_hparam` owns the surrounding digest,
preflight, round registration, launch, and lifecycle orchestration.

`managed_scheduler` owns the reusable GPU-capacity, process-observation,
execution-snapshot, and process-start primitives shared by managed launchers.
`experiment_pipeline` owns the strict validation-to-external-test state machine
and exposes it through the `experiments` facade.

### Domain — sleep2vec-specific
`domain/` (sidecar_summaries, finetune_summary, sex_age_summary, presets,
index_csv), the top-level `index_csv` re-export shim, and the per-task adapters
sleep2stat / preset_prepare / finetune / infer_evaluate / hparam_tune.

### Mixed bridges (9) — generic orchestration with tolerated domain coupling
| Module | Domain coupling (tolerated) |
|---|---|
| `models` | Hardcodes `SUPPORTED_VARIANTS` (incl. `sex_age_baseline`) and `VARIANTLESS_TASKS`. Root anchor imported everywhere. |
| `configs` | Thin shell, but hard-imports `domain.finetune_summary` + `adapters.sleep2stat`. |
| `plan_rendering` | `preset_cli_args` (sleep preset fields) + `if variant != "sex_age_baseline"` branch. |
| `decision_paths` | survival / multilabel / sex_age sidecar validation (highest domain signal among L0). |
| `decision_hparam` | hparam decision contract; depends on decision_paths' multilabel. |
| `plan_hparam` | hparam plan materialization over the domain-aware primitives. |
| `plan_context` | Imports `domain.presets` + `index_csv`. |
| `hparam_postprocess` | kaldi / label_name / torch logit post-processing. |
| `cli` | Forwards domain commands. |

The guard freezes this set: adding a module here requires updating `layering.py`
and this table, so a module can't silently slide into "mixed".

## CLI command triage (33 subcommands)

`cli_contract` freezes the 33 names; this is the ownership read.

- **Kernel (22)**: repo-summary, collect-runs, hparam-launch, hparam-run-queue,
  hparam-monitor, hparam-stop, hparam-select, hparam-checkpoint-scan,
  hparam-digest, hparam-suggest, hparam-adaptive-init, hparam-adaptive-step,
  hparam-adaptive-loop, progress, experiment-init, experiment-register-step,
  experiment-finalize, experiment-wandb-sync, experiment-index-checkpoints,
  experiment-monitor, experiment-rank, experiment-run.
- **Domain (7)**: config-summary, index-summary, preset-summary,
  hparam-external-eval, hparam-export-logits, hparam-threshold, hparam-ensemble.
- **Mixed (4)**: skills, doctor, context, plan — generic orchestration whose
  domain knowledge is injected through adapters.

## Reverse edges (kernel/mixed → domain)

All current reverse edges are top-level imports, tolerated and frozen by
`KNOWN_DOMAIN_IMPORT_EXEMPTIONS`. The guard fails on any new one.
The same guard scans every `adapters/` module and rejects imports into the
`L2_MODULES` orchestration set, including multi-dot relative spellings.

| Source → Target | Layer | Why tolerated | Future removal |
|---|---|---|---|
| `configs → domain.finetune_summary` | L2 → domain | configs shell delegates the generic finetune summary body | Would need a registry/provider indirection for the finetune-family summary |
| `configs → adapters.sleep2stat` | L2 → L1 | frozen `sleep2stat_config_summary` re-export path | Drop when the re-export is retired |
| `plan_context → domain.presets` | L2 → domain | preset summary in plan context | Route through an adapter hook |
| `plan_context → index_csv` | L2 → domain(shim) | index summary in plan context | Route through an adapter hook |
| `cli → domain.presets` | mixed → domain | `preset-summary` command | Domain CLI split |
| `cli → index_csv` | mixed → domain(shim) | `index-summary` command | Domain CLI split |
| `index_csv → domain.index_csv` | shim → domain | frozen top-level import path | Drop when the shim is retired |
| `domain.index_csv → configs` | domain → L2 | index_csv is a config-summary consumer, not a leaf; configs never imports it back, so the edge is one-way | Would need index summary to take config_summary as an argument |

Legal edges outside the reverse-edge table:
- `adapters/config_providers → domain.sex_age_summary` — L1 → L0, a legal
  direction.
- `markdown → decisions`, `experiment_workspace → experiment_io` — core → core.

## Frozen surfaces

- `cli_contract`: 33 subcommand names + argument contracts + task/variant
  routing matrix + the `cli.export_hparam_logits` attribute name (a monkeypatch
  anchor).
- The adapter-boundary guard's `KERNEL_MODULES` file list (7 modules, resolved
  as `Path(agent_tools.__file__).parent / name`).
- Frozen re-exports: `index_csv.index_summary`,
  `configs.sleep2stat_config_summary`, `configs.load_yaml`,
  `recipes.recipe_name`, `experiment_io.SSH_TIMEOUT_SECONDS`.
- External importers: 22+ preprocess/util scripts import `agent_tools.progress`;
  `agent_tools.models` is imported outside the package too. Moving either would
  break them, so they stay at the package top level.
