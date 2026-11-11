# agent_tools architecture

`agent_tools` is the experiment-control package that agents drive: consultation,
plan publication, managed launches, monitoring, and experiment workspaces. The
sleep2vec specifics live in `domain/` and the per-task adapters. Each module's
docstring is the local authority on what it owns; this page is the cross-module
map. When the two disagree, the docstring wins; fix this page.

## Structural rules

- The intra-package import graph is acyclic and has no function-local imports
  (`tests/agent_tools/test_agent_import_graph.py`). An adapter therefore never
  imports a module that dispatches through the adapter registry.
- Import a name from the module that defines it; packages re-export nothing.
- `progress` and `models` are imported from outside the package, so they stay at
  the package top level.
- The CLI is self-documenting: register subcommands through `cli._command()`.
  `test_agent_tools_cli_contract.py` freezes the subcommand set and requires
  `help=` on every subcommand and option.

## Module ownership

Entry and shared vocabulary:

- `cli`: the `python -m agent_tools` subcommand surface; `run-stop` routes by plan task to the
  hparam, inference, and preset stop owners.
- `models`: repository paths, variant and task constants, YAML loading, and config-summary types.
- `recipes`: recipe, layer, user-decision, and consultation-policy YAML loaders.
- `skills`: validation of `skills/` playbooks against `skills/manifest.yaml`.
- `repo`: read-only repository orientation for `repo-summary`.

Consultation and plans:

- `decision_models`: statuses, issues, reports, and resolved user decisions.
- `decision_rules`: the pure recipe structure contract, decided from the recipe mapping alone.
- `decision_paths`: input path, sidecar, and execution-contract validation.
- `decision_hparam`: what a well-formed hparam search declaration is.
- `decisions`: the consultation gate that folds every check into one `DecisionReport`.
- `markdown`: consultation reports rendered as questions and Markdown.
- `schema_map`: the one decision-field to recipe/config path mapping.
- `configs`: config path to structured summary, including the shape-claimed sleep2stat and `sex_age_baseline` families.
- `plan_context`: plan-time index summary checks and plan Markdown.
- `plans`: `doctor` reports and frozen plan publication; `preflight_plan` is the one validation pass both share.
- `plan_contract`: run identities, paths, config bytes, and scripts that plan publication produces.
- `plan_hparam`: hparam plan materialization and the final-evaluation config snapshot.
- `plan_rendering`: recipe fields rendered into argv and launch scripts.
- `adapters/`: `base` declares the `TaskAdapter` protocol, `registry` maps task names to adapters, and one module per task carries that task's declarations and hooks.
- `domain/`: sleep2vec summaries of finetune and `sex_age_baseline` configs, survival and multilabel sidecars, presets, and index CSVs.

Launch and observation:

- `managed_scheduler`: backend selection, the direct GPU-capacity and process lifecycle, and Slurm submit, adopt, and stop.
- `slurm`: the `job.sbatch` and resource contracts and the `sbatch`, `squeue`, and `scancel` primitives.
- `execution_snapshot`: the frozen execution snapshot, its target probe, and the frozen-versus-live check.
- `gpu_rules`: partitioning a GPU pool into launch groups.
- `run_evidence`: observation of one direct run and identity-verified process-group stop.
- `hparam_runtime`: launch, queue runner, and stop for a registered hparam plan; observation is `experiments.monitor_experiment` or each launch pass.
- `hparam_selection`: candidate ranking on the frozen selection split.
- `run_artifacts`: registered-plan and run-artifact reads, and hparam plan registration.
- `runtime_lock`, `runtime_sync`: the per-checkout runtime lock, and the clean `origin/main` fast-forward taken under it.
- `transport`: local shell quoting and the bounded `bash -lc` runner.
- `python_programs`: the embedded `python -c` programs, assembled from `python_program_sources/` fragments.
- `process_identity`: the process start token and group-liveness probe, imported by `run_evidence` and embedded verbatim in the launch wrapper so both sides compute the token from one source.
- `progress`: the run-side `status/progress.json` and `status/events.jsonl`.

Experiment workspace:

- `experiments`: the facade for the experiment lifecycle commands, including `experiment-monitor`, the one observe command.
- `experiment_workspace`: `run_manifest.tsv`, the only lifecycle owner, with its lock and managed manifests.
- `experiment_io`: atomic replace, a sidecar `flock` per file, and `locked_append` under it, plain reads, and the output-path preflight.
- `research_log`: append-only research notes.
- `experiment_tracking`: checkpoint evidence validation, status projections, monitor results, and hparam ranking validation.
- `manifests`: deterministic JSON, TSV, and text serializers.

Gates owned elsewhere: the complexity ceiling in `utils/complexity_check.py`, the
`Any` ledger in `utils/type_check.py`, and the test entrypoint in
`utils/test_agent_tools.py`.
