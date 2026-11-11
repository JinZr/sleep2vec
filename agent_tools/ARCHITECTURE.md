# agent_tools architecture

`agent_tools` is a reusable research-experiment agent-control framework with a
sleep2vec domain adapter layer on top. This document is the human-readable
companion to `layering.py` (the machine-readable module partition the layering
guard test reads). Keep them in sync — `test_agent_layering.py` fails if the
mixed-module set drifts.

## Layers

Imports point down the layers, **L2 → L1 → L0**, with `domain/` a L0-level
domain leaf. The bridges below the registry sit between L1 and L0: adapters
import them, and `configs` in turn reads `adapters/config_providers`, so the
order is a DAG rather than a strict ladder.

| Layer | Contents | Role |
|---|---|---|
| **L0 leaves** | models, decision_models, transport, manifests, schema_map, gpu_rules, repo, plan_rendering, plan_contract, decision_paths, experiment_workspace, experiment_io, execution_snapshot, managed_scheduler, ... | No intra-package deps beyond other L0 leaves; the reusable primitives. |
| **Bridges below the registry** | configs, plan_hparam, decision_hparam | Mixed bridges that never import the registry or L2. `adapters/hparam_tune` imports both hparam bridges, plan_hparam loads its config summary through configs, and configs reaches the adapters only through `adapters/config_providers`, which pairs the sleep2stat adapter's config-shape summary. |
| **L1 `adapters/`** | `base` (TaskAdapter protocol), `registry` (get_adapter / all_adapters / composite_adapter / SUPPORTED_TASKS), 6 per-task plugins, `config_providers` | Generic plugin skeleton + domain plugins. The package `__init__` re-exports nothing: import from `adapters.base` and `adapters.registry`. Kernel dispatches through the registry and never hardcodes task names; `run_artifacts` reads it to validate registered plans and owns hparam plan registration. |
| **L2 kernel** | decision_rules, decisions, plan_context, plans | Orchestration over lower-layer owners and adapter declarations; authored task recipes remain governed by schema_map. |
| **`domain/`** | sidecar_summaries, finetune_summary, sex_age_summary, presets, index_csv | sleep2vec-specific summaries/validators. L0-level leaves that must not be aggregated in `domain/__init__` (would trigger a partial-import cycle via configs). |

`test_agent_layering.py` requires the intra-package import graph, counting the
implicit package `__init__` edges, to be acyclic with no deferred
(function-local) imports.

## Module ownership

Mirrors the three frozensets in `layering.py`.

Each module's own docstring is the local authority on what that module owns;
`test_agent_layering.py` requires a non-empty docstring, but does not check its
ownership claims or accuracy. This section is the
**cross-module** map — which owner sits behind which facade, and which pairs
split a concern between them — that no single docstring can state. When the two
disagree, the docstring is next to the code and wins; fix this document.

### Kernel — reusable (30, zero domain signal)
decision_models, transport, python_programs, manifests, schema_map, gpu_rules, repo,
runtime_lock, runtime_sync,
experiment_io, research_log, experiment_workspace, experiment_sources,
experiment_tracking, experiments,
run_artifacts, run_evidence, hparam_runtime, hparam_selection,
recipes, progress,
markdown, skills,
decisions, plans, plan_contract, decision_rules, execution_snapshot, managed_scheduler, slurm.

These must stay domain-free — the layering guard allows them **no** domain
imports.

`python_programs` owns the canonical loader for embedded kernel program source
files used by transport and experiment-control modules.

`runtime_sync` owns dry-run inspection and clean `origin/main` fast-forward of
one existing checkout. It shares `runtime_lock` with launch paths so an update
cannot cross the short HEAD-observation-to-process-start critical section.

`decision_rules` owns the pure, dictionary-only recipe structure contract:
registered task and variant, top-level closure, runtime, task sections,
execution, and artifacts. `plans` adds authored/base/local layer orchestration
and policy-dependent decision validation; `run_artifacts` reuses the structural
owner for frozen registered plans without consultation, config/path probes, or
live observation. Hparam plan registration also lives in `run_artifacts` and,
unlike those reads, does probe: `preflight_hparam_plan` checks a staged plan's
candidate configs and output paths through `plan_hparam`, probes the execution
target and freezes that snapshot into the plan, and `commit_hparam_plan`
registers the published plan's run-manifest rows. `plans` calls both directly;
adapters carry no registration hooks.

`plan_contract` and adapter `compile_plan_contract()` hooks own deterministic
run identities, paths, derived config bytes, complete executable scripts, and
final-evaluation requirements shared by plan publication and registered-plan
validation. Task adapters may derive that contract only from the frozen recipe,
including its strict creator-host plan context and input snapshots, and
plan-owned config bytes. The hparam adapter's hook delegates to
`plan_hparam.compile_hparam_plan_contract`, which the hparam reader in
`run_artifacts` calls directly.

`experiment_workspace` owns managed manifests, the run-manifest lock, and the
one-way step `plan_controller` binding that makes a step ordinary when its
first plan registers. `research_log` owns append-only semantic note validation, rendering, and
CAS publication behind the workspace facade. `experiment_sources` owns
checkpoint evidence validation; `experiment_tracking` re-exports it and owns
projections, status advice, and hparam ranking validation. Recipe fields are only
consistency guards, while `run_manifest.tsv` remains the lifecycle owner.

`managed_scheduler` owns backend selection plus the reusable direct
GPU-capacity/process lifecycle and Slurm submit/observe lifecycle shared by
managed launchers. `execution_snapshot` owns the frozen execution snapshot those
launches check their target against: its shape, the target probe that produces
it, the frozen-versus-live comparison, and its atomic write. Plan registration
and the Slurm compute wrapper use it directly rather than through the scheduler.
`slurm` owns scheduler resource/script contracts, CLI
transport, machine-readable job and sidecar identity parsing, state
normalization, compute-wrapper execution, and cancellation primitives.

### Domain — sleep2vec-specific
`domain/` (sidecar_summaries, finetune_summary, sex_age_summary, presets,
index_csv) and the per-task adapters
sleep2stat / preset_prepare / finetune / infer / hparam_tune /
embedding_extraction.

### Mixed bridges (8) — generic orchestration with tolerated domain coupling
| Module | Domain coupling (tolerated) |
|---|---|
| `models` | Hardcodes `SUPPORTED_VARIANTS` (incl. `sex_age_baseline`) and `VARIANTLESS_TASKS`. Root anchor imported everywhere. |
| `configs` | Thin shell, but hard-imports `domain.finetune_summary`. |
| `plan_rendering` | `preset_cli_args` (sleep preset fields). |
| `decision_paths` | survival / multilabel sidecar validation and the sex_age_baseline pretrained-backbone rejection (highest domain signal among L0). |
| `decision_hparam` | hparam decision contract; depends on decision_paths' multilabel and sex_age_baseline backbone checks. |
| `plan_hparam` | hparam plan materialization over the domain-aware primitives. |
| `plan_context` | Imports `domain.index_csv`. |
| `cli` | Forwards domain commands. |

The guard freezes this set: adding a module here requires updating `layering.py`
and this table, so a module can't silently slide into "mixed".

## CLI command triage (23 subcommands)

`test_agent_tools_cli_contract.py` freezes the command names and checks this
three-way ownership partition against the parser, including counts and duplicates.

- **Kernel (17)**: repo-summary, runtime-sync, hparam-launch, infer-launch, infer-stop,
  preset-launch, preset-stop,
  hparam-run-queue,
  hparam-monitor, hparam-stop, hparam-select,
  progress, experiment-init, experiment-note,
  experiment-finalize, experiment-monitor, experiment-status.
- **Domain (3)**: config-summary, index-summary, preset-summary.
- **Mixed (3)**: skills, doctor, plan — generic orchestration whose
  domain knowledge is injected through adapters.

## Reverse edges (kernel/mixed → domain)

All current reverse edges are top-level imports, tolerated and frozen by
`KNOWN_DOMAIN_IMPORT_EXEMPTIONS`. The guard fails on any new one.
The same guard scans every `adapters/` module and rejects imports into the
`L2_MODULES` orchestration set, including multi-dot relative spellings.

| Source → Target | Layer | Why tolerated | Future removal |
|---|---|---|---|
| `configs → domain.finetune_summary` | mixed → domain | configs shell delegates the generic finetune summary body | Would need a `config_providers` entry for the finetune-family summary |
| `plan_context → domain.index_csv` | L2 → domain | index summary at plan time; task-independent with one implementation, adapters vary only its inputs via `index_summary_inputs_override` | None planned: a per-task hook would only forward to the same function |
| `cli → domain.presets` | mixed → domain | `preset-summary` command | Domain CLI split |
| `cli → domain.index_csv` | mixed → domain | `index-summary` command | Domain CLI split |
| `domain.index_csv → configs` | domain → mixed | index_csv is a config-summary consumer, not a leaf; configs never imports it back, so the edge is one-way | Would need index summary to take config_summary as an argument |

Legal edges outside the reverse-edge table:
- `adapters/config_providers → domain.sex_age_summary` — L1 → L0, a legal
  direction.
- `configs → adapters/config_providers → adapters/sleep2stat` — bridge → L1:
  configs reads the config-summary tables, never the registry.
- `plan_hparam → configs` — bridge → bridge: the hparam plan loads its config
  summary through the recipe-level loader.
- `run_artifacts → plan_hparam` — kernel → bridge: hparam plan registration
  checks candidate configs and output paths through plan_hparam.
- `markdown → decisions`, `experiment_workspace → experiment_io` — core → core.

## Frozen surfaces

- `cli_contract`: subcommand names and ownership groups + argument contracts + task/variant
  routing matrix + the `cli.export_hparam_logits` attribute name (a monkeypatch
  anchor) + self-documentation (every subcommand carries a `help=` summary and a
  `description=`; every option carries `help=`). Register subcommands through
  `cli._command()`, which sets the summary and description from one string.
- The adapter-boundary guard's `KERNEL_MODULES` file list (7 modules, resolved
  as `Path(agent_tools.__file__).parent / name`).
- Module ownership docstrings: every `.py` in the package carries one, with a
  non-empty summary line. A new module without one fails
  `test_every_module_has_nonempty_docstring`; content accuracy needs code review.
- No compatibility re-exports: import a name from the module that defines it.
  The former `index_csv` shim and the `configs`, `recipes`, `experiment_io` and
  `adapters` package re-exports were removed, and `test_agent_layering.py`
  rejects their return.
- External importers: 22+ preprocess/util scripts import `agent_tools.progress`;
  `agent_tools.models` is imported outside the package too. Moving either would
  break them, so they stay at the package top level.

## Complexity ceiling

`agent_tools` is gated at mccabe complexity 25, owned by
`utils/complexity_check.py` and run from both `utils/style_check.sh` and the
style_check workflow. It is not a repo-wide `.flake8` setting: the model and
training packages are out of scope by not being named, rather than by a
`per-file-ignores` entry, so nothing outside `agent_tools` is silently
unchecked. The checker spells the ceiling value; nothing else does.

Three checks, mirroring the mypy ledger ratchet in `utils/type_check.py`:

1. **The ceiling** — C901 over `agent_tools`.
2. **The suppression ledger** — `SUPPRESSION_LEDGER` is empty. Re-running with
   noqa disabled checks that no function is silently exempt and that the live
   violation identities match the ledger. Stale annotations and ledger entries
   fail the gate.
3. **The embedded programs** — `python_program_sources/*.py.src` are fragments
   assembled by `python_programs.source()` and run through `python -c`. They do
   not lint standalone (names resolve only once concatenated) and flake8's
   directory walk never sees them, so every registered program is assembled and
   checked at the same ceiling with noqa disabled. `PROGRAM_LEDGER` is also
   empty; no assembled function or top-level block is exempt.

Every probe runs `--isolated`. Reading `.flake8` would let one
`per-file-ignores` entry blind both the gate and the ledger audit at once —
file-level blindness arriving through the config instead of a comment.

Both ledgers are shrink-only: a new function or program over 25 branches is a
design signal, not a lint to suppress.

## Randomized test order

`requirements-agent-tools.txt` pins `pytest-randomly` for the agent-tool kernel
and recipe test jobs. CI prints `GITHUB_RUN_NUMBER` as the seed and passes it to
pytest. Retries keep the same seed and randomized collection order; parallel
execution order may vary. Both jobs use `utils/test_agent_tools.py`, which keeps
random ordering and disables reseeding between test phases.

Use the printed seed and the same test selection to reproduce the collection
order. The full suite also needs the domain packages excluded from kernel CI;
with those installed, a serial check is:

```bash
conda run -n exp python utils/test_agent_tools.py -q --randomly-seed=1234
```

The entrypoint defaults only the test subprocess to `DS_ACCELERATOR=cpu`, because
DeepSpeed's seed hook can initialize an accelerator during collection in the full
`exp` environment. An explicit accelerator environment value is preserved.
For a serial fixed-order control, disable the plugin and omit its seed options:

```bash
conda run -n exp python utils/test_agent_tools.py -q -p no:randomly
```

The compact `-pno:randomly` spelling and disabling the plugin through
`PYTEST_ADDOPTS` are also supported. Remove any `--randomly-seed` option from
`PYTEST_ADDOPTS` for the fixed-order run as well.
