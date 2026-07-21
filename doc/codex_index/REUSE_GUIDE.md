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
| Select batch channels and collate | `DefaultDataset` in [`data/default_dataset.py`](../../data/default_dataset.py) | NPZ/Kaldi-specific collate functions |
| Pair-first and pair-eval scheduling | samplers in [`data/samplers.py`](../../data/samplers.py) | entrypoints or callbacks |
| Kaldi matrix access | `KaldiPSGDataset` and `KaldiReaderPool` in [`data/`](../../data/) | a second dataset stack |
| Ordinary metadata encoding | [`data/metadata.py`](../../data/metadata.py) | task heads |
| Survival sidecars | load, attach, and stack owners in [`data/survival.py`](../../data/survival.py) | preset, trainer, or inference-local parsers |
| Multilabel sidecars | load, attach, and stack owners in [`data/multilabel.py`](../../data/multilabel.py) | preset, trainer, or inference-local parsers |
| Pretrain/finetune loader assembly | `get_pretrain_dataloader`, `get_finetune_dataloaders` in [`sleep2vec/utils.py`](../../sleep2vec/utils.py) | CLI entrypoints |
| Checkpoint initialization and averaging | [`sleep2vec/checkpoints.py`](../../sleep2vec/checkpoints.py) | inference or variant scripts |
| Downstream metric definitions | [`sleep2vec/metrics.py`](../../sleep2vec/metrics.py) | result writers or plots |
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
| Agent consultation | `evaluate_consultation_gates` through [`agent_tools/decisions.py`](../../agent_tools/decisions.py) | command renderers |
| Agent context and plan | `build_context`, `build_plan`, `preflight_plan` through [`agent_tools/plans.py`](../../agent_tools/plans.py) | skills or adapters |
| Agent task extension | adapter protocol/registry in [`agent_tools/adapters/`](../../agent_tools/adapters/) | kernel task-name branches |
| Recipe loading | [`agent_tools/recipes.py`](../../agent_tools/recipes.py) | individual commands |
| Managed workspace identity | canonical read/merge/CAS owners in [`agent_tools/experiment_workspace.py`](../../agent_tools/experiment_workspace.py) | hparam, planning, or monitoring-local tables and manifest writers |
| Local/SSH managed I/O | [`agent_tools/experiment_io.py`](../../agent_tools/experiment_io.py) | each experiment command |
| Managed process identity and stopping | [`agent_tools/run_evidence.py`](../../agent_tools/run_evidence.py) through [`agent_tools/hparam.py`](../../agent_tools/hparam.py) | PID-only checks or caller-local signals |
| Managed GPU scheduling and process launch | [`agent_tools/managed_scheduler.py`](../../agent_tools/managed_scheduler.py) | hparam- or pipeline-local capacity, observation, snapshot, and process-start implementations |
| Frozen hparam plan reads | `read_hparam_plan` in [`agent_tools/run_artifacts.py`](../../agent_tools/run_artifacts.py) | launcher/postprocess-specific parsing |
| Public hparam operations | [`agent_tools/hparam.py`](../../agent_tools/hparam.py) facade with responsibility modules behind it | direct private cross-module imports |
| Adaptive agent proposal validation | [`agent_tools/adaptive_proposals.py`](../../agent_tools/adaptive_proposals.py) for canonical snapshots, parameter envelopes, and submission validation; [`agent_tools/adaptive_hparam.py`](../../agent_tools/adaptive_hparam.py) for orchestration | provider callbacks, latest-digest lookup during apply, or lifecycle mutation in the proposal kernel |
| Public experiment operations | [`agent_tools/experiments.py`](../../agent_tools/experiments.py) facade with I/O/tracking owners behind it | skills or CLI handlers |
| Resumable external evaluation | [`agent_tools/experiment_pipeline.py`](../../agent_tools/experiment_pipeline.py) through the `experiments` facade | shell loops that wait for training, select checkpoints, launch inference, or finalize |
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

### AHI, survival, or multilabel behavior

- Keep AHI event metrics and threshold fitting in the shared metric/finetune path.
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
- Extend tasks through adapters and declarations; keep the reusable kernel free of new sleep-specific branches.
- Run consultation before runnable plans and stop on `NEEDS_USER_INPUT`.
- Treat `run_manifest.tsv` as authoritative managed state; mirrors and reports are projections.
- Reuse `managed_scheduler` for capacity and process lifecycle primitives; keep pipeline policy in `experiment_pipeline`.
- Use `experiment-run` for validation-selected external matrices. Monitor commands remain non-launching.
- Keep external-agent suggestions inside the `adaptive_proposals` snapshot/envelope contract; let `adaptive_hparam` own preflight and lifecycle changes.
- Generate calls to existing model, preprocess, baseline, and sleep2stat entrypoints rather than adding an agent runtime.
- Follow [`agent_tools/ARCHITECTURE.md`](../../agent_tools/ARCHITECTURE.md) and its layering test for kernel/domain boundaries.

### Standalone variants

- Keep imports, config loaders, data/preprocess modules, metrics, results, and runtime package-local.
- Use variant tests to distinguish required parity from intentional variant behavior.
- Route MoE schema to `sleep2expert.config`, execution to its RoFormer MoE modules, and export to routing/subnetwork owners.
- Use `sleep2expert.export_subnetwork` when a compact artifact is required; inference route filters do not compact a checkpoint.

## Major Duplication Risks

- Keep model construction, checkpoint selection/averaging, metrics, and result
  writing in their existing builders and artifact owners.
- Preserve one `DefaultDataset` batch contract across NPZ and Kaldi. Do not
  reinterpret channel availability, mask truthiness, sidecars, or disease order
  in callers.
- Keep root, `sleep2vec2`, and `sleep2expert` package-local. Use the MoE owners
  for routing, regularization, and compact export instead of checkpoint surgery.
- Reuse sleep2stat stage-source, reducer, and bundle-writer owners; analyzers
  return results rather than writing bundle artifacts directly.
- Extend agent tasks through adapters and canonical workspace owners. Do not
  parse managed manifests in callers or create an agent-specific trainer,
  preprocessing engine, or analysis executor.
- Use the managed external pipeline for wait/select/launch/finalize loops;
  monitoring remains observational.

## Non-Reuse Zones

- Notebooks and historical experiment scripts are context, not canonical libraries.
- Test helpers are scaffolding; reuse the product implementation they exercise.
- Generated configs, context bundles, manifests, reports, and run directories are artifacts, not templates for new code.
- Git history preserves removed index detail; do not reintroduce historical aliases into the shared navigation layer.
