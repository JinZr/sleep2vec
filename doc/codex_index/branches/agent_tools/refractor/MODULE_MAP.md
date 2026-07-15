# Module Map

## Public Surfaces and Responsibility Owners

| Area | Modules | Canonical responsibility | Main dependents |
|---|---|---|---|
| CLI | `__main__.py`, `cli.py` | Parse commands, call public Python entrypoints, render output, map exceptions/reports to exit codes | End users and agents invoking `python -m agent_tools` |
| Shared models | `decision_models.py`, `models.py` | Decision DTOs/status merge, repository path helpers, variant module routing, JSON conversion | Decisions, plans, CLI |
| Recipe input | `recipes.py` | Managed YAML reads, reserved authored-field rejection, base overlay/provenance, closed user-decision reads, policy reads, recipe naming | Plans and adaptive workflows |
| Consultation facade | `decisions.py` | Source resolution, task/variant support, aggregate status, task-aware runtime/decision-name closure, base-finetune recursion | `plans.evaluate_recipe`, `plans.build_context` |
| Decision owners | `decision_paths.py`, `decision_rules.py`, `decision_hparam.py` | Non-hparam execution closure/path checks, ordinary task section closure/rules, hparam section closure/search/execution/adaptive rules | `decisions.evaluate_consultation_gates`, `plans.evaluate_recipe` |
| Config and input summaries | `configs.py`, `index_csv.py`, `presets.py`, `repo.py`, `skills.py` | Lightweight, agent-readable facts without model loading | CLI, context, consultation |
| Plan facade | `plans.py` | Source-aware top-level/artifact closure, recipe evaluation, no-write preflight, workspace mutation boundary, plan/context output, task dispatch | CLI and adaptive workflow |
| Plan owners | `plan_context.py`, `plan_rendering.py`, `plan_hparam.py` | Context summaries/docs, shared accepted runtime/preset field-to-CLI mappings, hparam grid/final-test/frozen plan compilation | `plans.py`, decisions, postprocessing |
| Generic manifests | `manifests.py`, `markdown.py` | Local JSON/TSV/text primitives and decision-report rendering | Most orchestration modules |
| Workspace authority | `experiment_workspace.py` | Canonical root and identity, authoritative YAML, step/run manifests, status reducer, run matrix, events | Plans, hparam, experiments, adaptive |
| Local/remote I/O | `experiment_io.py` | Local/SSH reads and writes, strict tables, atomic replacement, output path topology, event append | Workspace and experiment lifecycle |
| Frozen/runtime evidence | `run_artifacts.py`, `run_evidence.py` | Validate hparam plans and run rows; inspect exact runtime manifests, PIDs, logs, progress, GPUs, checkpoints | Hparam, adaptive, experiment tracking |
| Hparam facade | `hparam.py` | Re-export supported public hparam operations only | CLI |
| Hparam owners | `hparam_runtime.py`, `hparam_selection.py`, `hparam_postprocess.py` | Launch/explicit queue/verified target snapshot/monitor/stop, candidate/checkpoint selection, final evaluation/logit/threshold/ensemble outputs | CLI, adaptive |
| Experiment facade | `experiments.py` | Initialize/register/finalize/sync/index/monitor/rank public commands | CLI |
| Experiment observations | `experiment_tracking.py` | W&B, checkpoint, metric, monitor, ranking observations and reports | `experiments.py` |
| Adaptive workflow | `adaptive_hparam.py` | Compose plans, monitoring, selection evidence, budget, replacement, and round registry | CLI |
| Progress | `progress.py` | Stable `status/progress.json`, event append, local/remote read and display | CLI and long-running tools |

## Dependency Direction

```text
cli
  -> plans / hparam facade / experiments / adaptive_hparam / summaries

plans
  -> recipes -> experiment_workspace.read_managed_yaml_mapping
  -> decisions -> decision_models + decision_paths/rules/hparam
  -> plan_context + plan_rendering + plan_hparam
  -> experiment_workspace + run_artifacts

hparam facade
  -> hparam_runtime / hparam_selection / hparam_postprocess
      -> run_artifacts + run_evidence
      -> experiment_workspace + experiment_io

experiments
  -> experiment_tracking
  -> experiment_workspace + experiment_io

adaptive_hparam
  -> plans + hparam_runtime
  -> run_artifacts + run_evidence
  -> experiment_workspace
```

Leaf responsibility modules should not import their public facades. In particular, decision owners should not import private helpers from `decisions.py`; hparam owners should not route through CLI; tracking code should return observations rather than mutate canonical run state independently.

## Edit Routing

| Change | Lead owner | Adjacent review/tests |
|---|---|---|
| Authored recipe loading or base merge | `recipes.py` | `plans.py`, recipe and plan-atomicity tests |
| Recipe/user-decision field contract | Existing section owner in `decision_*`, `experiment_workspace.py`, or `plans.py` | recipe, user-decision, CLI contract, plan-atomicity tests |
| Task or variant consultation | `decisions.py` plus relevant `decision_*` owner | runtime orchestrator when generated flags change |
| Command flags or task dispatch | `plan_rendering.py` and `_commands_for_recipe` in `plans.py` | runtime command tests and variant validation |
| Grid expansion/frozen hparam plan | `plan_hparam.py` | `run_artifacts.py`, hparam runtime tests |
| Run identity or lifecycle state | `experiment_workspace.py` | all hparam/experiment/adaptive consumers |
| Local/SSH artifact safety | `experiment_io.py` | workspace and remote-I/O tests |
| Runtime evidence or process identity | `run_artifacts.py` / `run_evidence.py` | monitor, stop, adaptive, experiment tests |
| Hparam launch/queue/stop/monitor | `hparam_runtime.py` | frozen-plan, target-preflight, queue-drain, and run-evidence tests |
| Candidate selection/checkpoint ranking | `hparam_selection.py` | experiment tracking and postprocess consumers |
| W&B/checkpoint observations | `experiment_tracking.py` | `experiments.py` and ownership tests |
| Adaptive round ordering/budget | `adaptive_hparam.py` | preflight, hparam runtime, workspace tests |

## Contract Files

- `recipes/schemas/task_recipe.schema.md`: authored recipe documentation.
- `doc/agent_contracts/task_recipe.md`: task recipe behavior and lifecycle expectations.
- `doc/agent_contracts/user_decisions.md`: decision-file input.
- `doc/agent_contracts/experiment_management.md`: workspace and managed run state.
- `agent_policies/consultation_policy.yaml`: high-impact decision questions and task applicability.
- `skills/manifest.yaml` and per-skill files: agent-facing workflow guidance.

## Test Map

- Recipe/consultation: `test_agent_recipe_closure.py`, `test_agent_tools_recipes.py`, `test_agent_consultation_policy.py`, `test_agent_user_decisions.py`.
- Preflight and mutation ordering: `test_agent_plan_blocks_on_ambiguity.py`, `test_agent_tools_cli_contract.py`.
- Workspace/state: `test_agent_tools_experiment_workspace.py`, `test_agent_tools_experiments.py`, `test_agent_tools_remote_experiments.py`.
- Hparam: `test_agent_tools_hparam_runtime.py`, `test_agent_tools_hparam_selection.py`, `test_agent_tools_hparam_postprocess.py`, `test_agent_tools_adaptive_hparam.py`.
- Task-specific plans: `test_agent_tools_sleep2stat.py`, `test_agent_sex_age_baseline.py`, and recipe command-rendering tests.
- Skills and summaries: `test_agent_tools_skills.py`, index/config/preset summary tests.

Exact test filenames should be confirmed with `rg --files tests/agent_tools` before invoking a narrowed subset.
