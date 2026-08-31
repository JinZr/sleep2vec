# Reuse Guide

Use this guide to find the existing owner before adding a helper, wrapper, schema branch, or parallel artifact path. The named symbols are navigation anchors, not a frozen API inventory; verify them in current source.

## Default Rule

Change the narrowest owner that already handles the behavior. Reuse public facades at subsystem boundaries and edit responsibility-specific modules behind them. A new implementation is justified only when the existing owner cannot satisfy the requested contract without taking on an unrelated responsibility.

## Canonical Implementations

| Responsibility | Reuse | Do not duplicate in |
| --- | --- | --- |
| Load model and task YAML | `load_pretrain_config`, `load_finetune_config` in [`sleep2vec/config.py`](../../sleep2vec/config.py) | entrypoints or tests |
| Bind finetune/runtime state | `apply_finetune_config` and related binders in [`sleep2vec/common.py`](../../sleep2vec/common.py) | trainer modules |
| Persist run config and CLI state | `persist_run_config_and_args` in [`sleep2vec/common.py`](../../sleep2vec/common.py) | each entrypoint |
| Construct registered components | factories in [`sleep2vec/builders.py`](../../sleep2vec/builders.py) and [`sleep2vec/registry.py`](../../sleep2vec/registry.py) | name-based runtime branches |
| Build the pretrained feature path | `Sleep2vecPretrainModel` in [`sleep2vec/pretrain_model.py`](../../sleep2vec/pretrain_model.py) | downstream heads |
| Build downstream features and heads | `Sleep2vecDownstreamModel` and [`sleep2vec/downstreams/`](../../sleep2vec/downstreams/) | finetune/infer entrypoints |
| Temporal pooling | `build_temporal_aggregator` and `mean`, `attn`, `lstm` implementations in [`sleep2vec/downstreams/temporal_aggregation/`](../../sleep2vec/downstreams/temporal_aggregation/) | task-specific trainers |
| Validate preset samples | `filter_valid_sample_indices` in [`data/utils.py`](../../data/utils.py) | preprocessing scripts or samplers |
| Validate whole-night NPZ indexes and build embedding sample keys | `validate_whole_night_index` and `build_embedding_sample_key` in [`data/whole_night_index.py`](../../data/whole_night_index.py) | variant extractors or agent adapters |
| Select batch channels and collate | `DefaultDataset` in [`data/default_dataset.py`](../../data/default_dataset.py) | NPZ/Kaldi-specific collate functions |
| Pair-first and pair-eval scheduling | samplers in [`data/samplers.py`](../../data/samplers.py) | entrypoints or callbacks |
| Kaldi matrix access | `KaldiPSGDataset` and `KaldiReaderPool` in [`data/`](../../data/) | a second dataset stack |
| Ordinary metadata encoding | [`data/metadata.py`](../../data/metadata.py) | task heads |
| Survival sidecars | load, attach, and stack owners in [`data/survival.py`](../../data/survival.py) | preset, trainer, or inference-local parsers |
| Multilabel sidecars | load, attach, and stack owners in [`data/multilabel.py`](../../data/multilabel.py) | preset, trainer, or inference-local parsers |
| Pretrain/finetune loader assembly | `get_pretrain_dataloader`, `get_finetune_dataloaders` in [`sleep2vec/utils.py`](../../sleep2vec/utils.py) | CLI entrypoints |
| Checkpoint initialization and averaging | [`sleep2vec/checkpoints.py`](../../sleep2vec/checkpoints.py) | inference or variant scripts |
| Shared downstream metrics and event primitives | [`sleep2vec/metrics/core.py`](../../sleep2vec/metrics/core.py) | result writers or task-specific metrics |
| AHI event metrics and threshold protocol | [`sleep2vec/metrics/ahi.py`](../../sleep2vec/metrics/ahi.py) | finetune, inference, result writers, or plots |
| Arousal event metrics and threshold protocol | [`sleep2vec/metrics/arousal.py`](../../sleep2vec/metrics/arousal.py) | finetune, inference, result writers, or plots |
| Task-aware epoch reduction | `Sleep2vecFinetuning` in [`sleep2vec/sleep2vec_finetuning.py`](../../sleep2vec/sleep2vec_finetuning.py) | a separate inference evaluator |
| Result and inference artifact paths | [`sleep2vec/results.py`](../../sleep2vec/results.py) | task-specific CSV writers |
| Prediction row extraction | [`sleep2vec/sleep2vec_inference.py`](../../sleep2vec/sleep2vec_inference.py) | result serialization code |
| Split-mask truthiness | `normalize_mask_frame` in [`preprocess/split_index_by_dataset.py`](../../preprocess/split_index_by_dataset.py) | each preprocessing command |
| Preset generation | [`preprocess/save_dataset_presets.py`](../../preprocess/save_dataset_presets.py) | notebooks or runtime datasets |
| NPZ-to-Kaldi conversion | [`preprocess/convert_npz_to_kaldi.py`](../../preprocess/convert_npz_to_kaldi.py) | backend runtime code |
| Repository config policy | [`utils/check_configs.py`](../../utils/check_configs.py) | YAML loaders unless it is runtime semantics |
| sleep2stat config | `load_config` in [`sleep2stat/config.py`](../../sleep2stat/config.py) | agent summaries or CLI-local validators |
| sleep2stat record discovery | `load_records` in [`sleep2stat/io/records.py`](../../sleep2stat/io/records.py) | analyzers |
| Stage-derived denominators | `StageSourceResolver` in [`sleep2stat/core/stage_sources.py`](../../sleep2stat/core/stage_sources.py) | individual analyzers/reducers |
| Model-derived AHI decoding | `decode_ahi_logits` in [`sleep2stat/analyzers/model_downstream.py`](../../sleep2stat/analyzers/model_downstream.py) | reducers or plotting |
| Analysis bundle output | `AnalysisBundleWriter` in [`sleep2stat/io/writers.py`](../../sleep2stat/io/writers.py) | analyzers or CLI branches |
| Analysis plotting | `plot_record`, `plot_cohort` in [`sleep2stat/plot.py`](../../sleep2stat/plot.py) | scripts that inspect analyzer internals |
| Static recipe structure | `recipe_structure_issues` in [`agent_tools/decision_rules.py`](../../agent_tools/decision_rules.py) for registered task/variant and section closure | planner- or status-local copies of recipe structure validation |
| Automatic finetune hparam profile | `compile_finetune_balanced_profile` in [`agent_tools/domain/finetune_hparam_profile.py`](../../agent_tools/domain/finetune_hparam_profile.py), invoked through the hparam adapter binding hook | kernel branches, skills, or templates that duplicate technical candidate generation |
| Hparam selection lifecycle and report | canonical selection fields written by [`agent_tools/hparam_selection.py`](../../agent_tools/hparam_selection.py), with checkpoint-test validation and best-candidate ordering in [`agent_tools/checkpoint_test_results.py`](../../agent_tools/checkpoint_test_results.py) and projection validation/rendering in [`agent_tools/experiment_tracking.py`](../../agent_tools/experiment_tracking.py) | event-, ranking-, status-, or finalizer-local lifecycle state |
| Frozen plan semantics | [`agent_tools/plan_contract.py`](../../agent_tools/plan_contract.py), `compile_plan_contract` adapter hooks, and hparam compilers in [`agent_tools/plan_hparam.py`](../../agent_tools/plan_hparam.py) for creator-host context, recipe-derived run matrices, configs, complete executable scripts, and final-evaluation requirements | writer/status copies, reader-host defaults, or validation derived from mutable manifests |
| Hparam candidate validation and registration preflight | planner-local canonical variant validation of final config bytes, staged bundle checks and truthful config/argv provenance in [`agent_tools/plan_hparam.py`](../../agent_tools/plan_hparam.py); target identity and argv evidence in `managed_scheduler.inspect_execution_target`; path topology in `experiment_io.validate_managed_output_paths` | proposal-specific model schemas, caller-local YAML summaries, argv parsers, path checkers, or separate preflight state |
| Agent consultation and fillable decision projection | `evaluate_consultation_gates` and `user_decision_template` through [`agent_tools/decisions.py`](../../agent_tools/decisions.py) | command renderers, output writers, or duplicate task decision allowlists |
| Agent context and plan publication | `build_context`, `build_plan`, and `preflight_plan` through [`agent_tools/plans.py`](../../agent_tools/plans.py); deferred controllers must pair `plan_publication_lock` with `publish_staged_plan_locked`; `plan_tree_sha256` in [`agent_tools/run_artifacts.py`](../../agent_tools/run_artifacts.py) compares staged plans deterministically | skills, adapters, or adaptive/pipeline callers |
| Agent task extension | adapter protocol/registry in [`agent_tools/adapters/`](../../agent_tools/adapters/) | kernel task-name branches |
| Recipe loading and layer merge | `load_recipe_with_base` and `merge_recipe_layers` in [`agent_tools/recipes.py`](../../agent_tools/recipes.py) | individual commands or alternate base/local merge logic |
| Managed workspace identity and plan registration | canonical read/merge/CAS owners, the shared plan-registration lock, and frozen-row recovery validation in [`agent_tools/experiment_workspace.py`](../../agent_tools/experiment_workspace.py), with semantic research-log validation and append publication in [`agent_tools/research_log.py`](../../agent_tools/research_log.py) | hparam, planning, or monitoring-local locks, lifecycle reducers, tables, manifest writers, or Markdown appenders |
| Local/SSH managed I/O | [`agent_tools/experiment_io.py`](../../agent_tools/experiment_io.py), including strict control-bundle reads, descriptor-validated output reads, and positive remote operation results that distinguish missing files, conflicts, and failures | each experiment command |
| Managed direct process identity and stopping | [`agent_tools/run_evidence.py`](../../agent_tools/run_evidence.py) through [`agent_tools/hparam.py`](../../agent_tools/hparam.py) | PID-only checks or caller-local signals |
| Managed log evidence | [`agent_tools/run_evidence.py`](../../agent_tools/run_evidence.py) owns bounded local tails and the paired Slurm SSH health tail/age probe | monitor-local log readers or cross-observation caches |
| Managed scheduler lifecycle | [`agent_tools/managed_scheduler.py`](../../agent_tools/managed_scheduler.py) with Slurm primitives in [`agent_tools/slurm.py`](../../agent_tools/slurm.py); shared scheduler queries stay within one round and frozen route, never launch/stop/reconciliation | hparam- or pipeline-local capacity, scheduler commands, observation, reconciliation, snapshot, start, or stop implementations |
| Registered plan reads | `is_registered_blocked_plan` for exact nested local/SSH blocked-plan envelopes and strict plan-owned artifacts in root-resident workspaces, `read_registered_plan` for materialized control bundles reusing the static recipe structure owner, and `read_hparam_plan` for stronger local launch-time validation in [`agent_tools/run_artifacts.py`](../../agent_tools/run_artifacts.py) | status, launcher, or postprocess-specific parsing |
| Public hparam operations | [`agent_tools/hparam.py`](../../agent_tools/hparam.py) facade with responsibility modules behind it | direct private cross-module imports |
| Adaptive agent proposal validation | [`agent_tools/adaptive_proposals.py`](../../agent_tools/adaptive_proposals.py) for canonical snapshots, parameter envelopes, and submission validation; [`agent_tools/adaptive_hparam.py`](../../agent_tools/adaptive_hparam.py) for orchestration | provider callbacks, latest-digest lookup during apply, or lifecycle mutation in the proposal kernel |
| Experiment evidence acquisition | W&B and local/SSH checkpoint readers in [`agent_tools/experiment_sources.py`](../../agent_tools/experiment_sources.py), consumed by [`agent_tools/experiment_tracking.py`](../../agent_tools/experiment_tracking.py) | status-, ranking-, or CLI-local source readers |
| Public experiment operations | [`agent_tools/experiments.py`](../../agent_tools/experiments.py) facade, including canonical-only `experiment_status`, with I/O/tracking owners behind it | skills or CLI handlers |
| Resumable external evaluation | [`agent_tools/experiment_pipeline.py`](../../agent_tools/experiment_pipeline.py) through the `experiments` facade, with terminal result validation and aggregation in [`agent_tools/experiment_pipeline_results.py`](../../agent_tools/experiment_pipeline_results.py) | shell loops that wait for training, select checkpoints, launch inference, interpret terminal result manifests, or finalize |
| Index/config/preset summaries | [`agent_tools/domain/`](../../agent_tools/domain/) through stable top-level facades | shell parsing templates |
| MoE routing and experts | [`sleep2expert/backbones/roformer/moe.py`](../../sleep2expert/backbones/roformer/moe.py) | trainer-local routing branches |
| MoE regularization | [`sleep2expert/losses/moe_regularization.py`](../../sleep2expert/losses/moe_regularization.py) | pretrain/finetune loops |
| Compact MoE artifacts | [`sleep2expert/export_subnetwork.py`](../../sleep2expert/export_subnetwork.py) | manual checkpoint surgery |

## Guidance By Change Type

### Config or task semantics

- Put structural and semantic validation in the package-local `config.py`.
- Put built-in label interpretation and CLI binding in `common.py`.
- Keep optimization/logging convenience defaults distinct from required model/data semantics.
- Validate checked-in recipe policy through `utils/check_configs.py`; do not turn repository naming conventions into runtime schema.

### Model construction or outputs

- Register and build components through existing registries and factories.
- Keep tokenization-to-encoder flow in the pretrain model and temporal/channel/head flow in the downstream model.
- Keep layer mix and LoRA/DoRA insertion in downstream composition, not entrypoints.
- Keep loss and epoch-reduction semantics in Lightning modules and metric owners.

### Data, presets, or samplers

- Preserve `SampleIndex` plus the `DefaultDataset` batch contract across NPZ and Kaldi.
- Preserve `payload["available_channels"]` when missing-channel support is active.
- Use pair-first, sequential-pair, or available-channel bucket samplers according to their existing contracts.
- Keep sidecar column order, keys, and masks explicit; regenerate presets when attached semantic labels change.
- Keep storage differences behind dataset hooks instead of adding backend-specific collation.

### AHI, arousal, survival, or multilabel behavior

- Keep AHI event metrics and threshold fitting in the package-local `metrics/ahi.py`; finetuning owns only stage reduction and distributed coordination.
- Keep arousal event matching, window merging, threshold fitting, and ArI summaries in the package-local `metrics/arousal.py`; finetuning owns only stage reduction and distributed coordination.
- Keep survival and multilabel sidecar parsing in `data.survival` and `data.multilabel`.
- Aggregate repeated windows by the configured subject key for subject-level loss/metrics.
- Keep prediction rows traceable to path/window and disease-column order.
- Apply shared contract changes deliberately to root, `sleep2vec2`, and `sleep2expert`.

### Runtime or artifacts

- Keep trainer, callback, phase, W&B, and test orchestration in the relevant entrypoint and Lightning module.
- Use checkpoint helpers for initialization, aliases, selection, and averaging.
- Use result owners for output directories, CSV schemas, prediction ids, and manifests.
- Use inference `--results-root` to isolate a managed attempt; consume its unique terminal manifest instead of scanning shared default outputs.
- Let analysis/export failures terminate instead of emitting partial-success bundles.

### Preprocessing

- Compose split, mask, preset, merge, and Kaldi conversion commands rather than creating an all-in-one alternate pipeline.
- Treat `preprocess/preprocess_pipeline.ipynb` as history, not reusable implementation.
- Keep standalone data utilities in `utils/` independent from the training runtime when they only prepare external cohorts or files.
- Mirror converter/preset behavior into a variant only when that package contract is intentionally standalone.

### sleep2stat

- Validate all configs through `sleep2stat.config.load_config`.
- Let record loaders own discovery and analyzers own raw/model extraction.
- Reuse `StageSourceResolver` for sleep/stage denominators and onset-stage assignment.
- Let reducers consume analyzer results instead of reinterpreting logits or raw arrays.
- Let `AnalysisBundleWriter` enforce single-use output and terminal manifests.

### Agent tooling

- Treat `decisions.py`, `plans.py`, `hparam.py`, and `experiments.py` as public facades.
- Extend tasks through adapters and declarations, not sleep-specific kernel branches or a new runtime;
  use existing validators and package-local entrypoints. Follow
  [`ARCHITECTURE.md`](../../agent_tools/ARCHITECTURE.md) and its layering test.
- Keep task input checks in adapters and shared consultation in `decisions.py` / `plans.py`;
  [consultation and diagnostics](../agent_contracts/task_recipe.md#consultation-and-diagnostics)
  owns decision gates and per-call validation reuse.
- Reuse the profile compiler, `plan_contract`, adapter compilation hooks, and `plan_hparam`
  for [search expansion](../agent_contracts/task_recipe.md#search-space) and
  [frozen candidate/registration checks](../agent_contracts/task_recipe.md#registration-preflight),
  not caller-local schemas or frozen-plan validators.
- Keep manifest/CAS and `plan_controller` binding in `experiment_workspace`, evidence acquisition
  in `experiment_sources`, and projections/status in `experiment_tracking` behind `experiments`;
  see [canonical state](../agent_contracts/run_manifest.md#canonical-state-and-projections)
  and the [status read-set](../agent_contracts/experiment_workspace.md#read-only-status-and-advisory-actions).
- Reuse `hparam_selection` and `checkpoint_test_results` for
  [selection and selected-candidate consumers](../agent_contracts/task_recipe.md#selection-and-selected-candidate-consumers).
  Append semantic notes through `experiments.append_experiment_note` and `research_log`,
  not direct Markdown writes; see [research-log ownership](../agent_contracts/experiment_workspace.md#research-log).
- Reuse `managed_scheduler` for backend lifecycle and `slurm` for resource, command, state,
  and sidecar primitives; [run-manifest evidence](../agent_contracts/run_manifest.md#slurm-scheduler-evidence)
  owns their interpretation. Keep external-matrix policy and orchestration in `experiment_pipeline`,
  with reduction in `experiment_pipeline_results`; see [pipeline gates](../agent_contracts/experiment_pipeline.md#invocation-and-frozen-state).
- Reuse `python_programs.source` and `transport.remote_python_program_command` for embedded kernels;
  keep byte-preserving sources under `agent_tools/python_program_sources`, not inline copies.
- Keep proposal snapshots/envelopes in `adaptive_proposals` and preflight/lifecycle orchestration
  in `adaptive_hparam`; see the [adaptive workflow](../agent_contracts/task_recipe.md#adaptive-workflow)
  and [proposal handshake](../agent_contracts/task_recipe.md#proposal-handshake).

### Standalone variants

- Keep imports, config loaders, data/preprocess modules, metrics, results, and runtime package-local.
- Use variant tests to distinguish required parity from intentional variant behavior.
- Route MoE schema to `sleep2expert.config`, execution to its RoFormer MoE modules, and export to routing/subnetwork owners.
- Use `sleep2expert.export_subnetwork` when a compact artifact is required; inference route filters do not compact a checkpoint.

## Non-Reuse Zones

- Notebooks and historical experiment scripts are context, not canonical libraries.
- Test helpers are scaffolding; reuse the product implementation they exercise.
- Generated configs, context bundles, manifests, reports, and run directories are artifacts, not templates for new code.
- Git history preserves removed index detail; do not reintroduce historical aliases into the shared navigation layer.
