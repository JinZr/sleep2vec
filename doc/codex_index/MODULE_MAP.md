# Module Map

This map records stable ownership boundaries. It is a routing aid; inspect the linked source and focused tests before changing behavior.

## System Shape

The main runtime is config-driven:

1. YAML loaders define model, data, task, and output semantics.
2. CLI entrypoints bind runtime options and delegate construction.
3. dataset owners turn NPZ or Kaldi records into one shared batch contract.
4. model builders compose tokenizers, projections, encoders, aggregation, and heads.
5. Lightning modules own loss, optimization, metric reduction, and task-specific evaluation.
6. result owners write checkpoints, metrics, predictions, manifests, and visualizations.

Preprocessing produces the index, preset, or Kaldi artifacts consumed by the dataset layer. `sleep2stat` consumes raw signals and model outputs to create analysis bundles. `agent_tools` plans, explicitly launches, and monitors these existing entrypoints; it does not replace their runtimes, and monitor commands never launch pending work.

## Ownership Boundaries

| Area | Canonical location | Responsibility | Change here when |
| --- | --- | --- | --- |
| Config schema | [`sleep2vec/config.py`](../../sleep2vec/config.py) | Typed YAML loading, required semantic fields, model/data/task blocks | YAML meaning or validation changes |
| Built-in task binding | [`sleep2vec/common.py`](../../sleep2vec/common.py) | Apply config to CLI state, built-in label semantics, data-backend binding, run snapshots | a task flag, label source, or CLI/config bridge changes |
| Registries and construction | [`sleep2vec/registry.py`](../../sleep2vec/registry.py), [`sleep2vec/builders.py`](../../sleep2vec/builders.py) | Construct registered backbone, tokenizer, projection, CLS, aggregation, and head components | a construction contract or registry changes |
| Dataset index and collation | [`data/default_dataset.py`](../../data/default_dataset.py) | `SampleIndex`, filtering, channel selection, token loading, padding, metadata stacking, sampler attachment | batch or sample semantics change |
| Preset/index loading | [`data/psg_pretrain_dataset.py`](../../data/psg_pretrain_dataset.py), [`data/utils.py`](../../data/utils.py) | Build and validate windowed sample indexes, attach runtime metadata, preserve available channels | preset payload or sample validity changes |
| Metadata and sidecars | [`data/metadata.py`](../../data/metadata.py), [`data/survival.py`](../../data/survival.py), [`data/multilabel.py`](../../data/multilabel.py) | Encode ordinary metadata and attach/stack subject-level survival or multilabel vectors | label-sidecar shape or metadata tensorization changes |
| Sampling | [`data/samplers.py`](../../data/samplers.py) | Pair-first training, sequential pair evaluation, available-channel buckets, distributed sharding | batch scheduling or sampler invariants change |
| Storage backends | [`data/kaldi_io.py`](../../data/kaldi_io.py), [`data/kaldi_psg_dataset.py`](../../data/kaldi_psg_dataset.py) | Read Kaldi manifests/scp matrices through dataset storage hooks | Kaldi discovery or matrix access changes |
| Pretrain model | [`sleep2vec/pretrain_model.py`](../../sleep2vec/pretrain_model.py) | Channel tokenization, projection, CLS insertion, encoder forward, adaptation parameter groups | backbone feature flow changes |
| Downstream model | [`sleep2vec/downstream_model.py`](../../sleep2vec/downstream_model.py), [`sleep2vec/downstreams/`](../../sleep2vec/downstreams/) | Temporal aggregation, channel fusion, heads, layer mix, pretrained loading, adapters | downstream feature or head composition changes |
| Training modules | [`sleep2vec/sleep2vec_modelling.py`](../../sleep2vec/sleep2vec_modelling.py), [`sleep2vec/sleep2vec_finetuning.py`](../../sleep2vec/sleep2vec_finetuning.py), [`sleep2vec/sleep2vec_adaptation.py`](../../sleep2vec/sleep2vec_adaptation.py) | Losses, optimizer groups, schedules, epoch reduction, AHI/survival/multilabel behavior, adaptation phases | training or evaluation semantics change |
| Runtime entrypoints | [`sleep2vec/pretrain.py`](../../sleep2vec/pretrain.py), [`sleep2vec/adapt.py`](../../sleep2vec/adapt.py), [`sleep2vec/finetune.py`](../../sleep2vec/finetune.py), [`sleep2vec/infer.py`](../../sleep2vec/infer.py) | CLI parsing, trainer wiring, run directories, checkpoint/test orchestration | command or phase behavior changes |
| Runtime artifacts | [`sleep2vec/checkpoints.py`](../../sleep2vec/checkpoints.py), [`sleep2vec/results.py`](../../sleep2vec/results.py), [`sleep2vec/sleep2vec_inference.py`](../../sleep2vec/sleep2vec_inference.py) | Checkpoint selection/averaging, result paths, metrics/prediction rows, inference manifests | artifact layout or checkpoint policy changes |
| Metrics and diagnostics | [`sleep2vec/metrics.py`](../../sleep2vec/metrics.py), [`sleep2vec/diagnostics.py`](../../sleep2vec/diagnostics.py), [`sleep2vec/visualization/`](../../sleep2vec/visualization/) | Metric definitions, diagnostics hooks, evaluation plots | reported metric or visualization contracts change |
| Preprocessing | [`preprocess/`](../../preprocess/) | Split indexes, inspect masks, generate/merge presets, convert NPZ to Kaldi, convert WatchPAT archives | reproducible data-prep behavior changes |
| Config policy check | [`utils/check_configs.py`](../../utils/check_configs.py) | Validate checked-in YAMLs through runtime loaders plus repository recipe policy | repo-wide config policy changes |
| `sleep2stat` | [`sleep2stat/`](../../sleep2stat/) | Strict analysis config, record loading, analyzers, reducers, bundle writing, plots | derived-analysis behavior changes |
| Agent control plane | [`agent_tools/`](../../agent_tools/), [`agent_tools/ARCHITECTURE.md`](../../agent_tools/ARCHITECTURE.md) | Consultation, context, plans, managed experiments, hparam lifecycle, progress, summaries | agent-facing contracts or experiment ownership changes |
| Managed scheduling | [`agent_tools/managed_scheduler.py`](../../agent_tools/managed_scheduler.py) | Shared GPU capacity, process observation, execution snapshots, and process starts for managed launchers | hparam and external-pipeline scheduling behavior changes |
| External evaluation pipeline | [`agent_tools/experiment_pipeline.py`](../../agent_tools/experiment_pipeline.py), [`agent_tools/experiments.py`](../../agent_tools/experiments.py) | Wait for successful managed sources, freeze validation checkpoints, run isolated external jobs, summarize, and finalize | `experiment-run`, external-matrix, retry, or pipeline-resume behavior changes |
| Adaptive proposal boundary | [`agent_tools/adaptive_proposals.py`](../../agent_tools/adaptive_proposals.py), [`agent_tools/adaptive_hparam.py`](../../agent_tools/adaptive_hparam.py) | Immutable evidence snapshots and bounded external-agent proposals, followed by tool-owned preflight, registration, and launch | adaptive strategy, proposal protocol, or parameter-envelope semantics change |
| Agent declarations | [`skills/`](../../skills/), [`recipes/`](../../recipes/), [`agent_policies/`](../../agent_policies/), [`doc/agent_contracts/`](../agent_contracts/) | Human playbooks, task recipes, approved decisions, machine-readable contracts | declared workflow or policy changes |
| Dense standalone variant | [`sleep2vec2/`](../../sleep2vec2/), [`configs/sleep2vec2/`](../../configs/sleep2vec2/) | Package-local dense runtime and RoFormer implementation | `sleep2vec2` behavior or parity changes |
| MoE standalone variant | [`sleep2expert/`](../../sleep2expert/), [`configs/sleep2expert/`](../../configs/sleep2expert/) | Package-local MoE runtime, routing, regularization, checkpoint expansion, compact export | expert routing or MoE parity changes |
| Sex/age baseline | [`sex_age_baseline/`](../../sex_age_baseline/), [`configs/sex_age_baseline/`](../../configs/sex_age_baseline/) | Standalone demographic baseline for Cox and multilabel tasks | baseline-only data/model/runtime changes |
| Contract tests | [`tests/`](../../tests/) | Pin config, data, model, runtime, variant, sleep2stat, and agent contracts | any documented contract changes |

## Core Data Contract

### Sample index

`data.default_dataset.SampleIndex` is the canonical window descriptor:

- `id` identifies the sample window;
- `start` and `end` are token-window bounds;
- `payload` carries storage/runtime fields, including `available_channels` in missing-channel presets;
- `metadata` carries split, source, path, demographic, task, and sidecar values.

Preset files are pickled `list[SampleIndex]` artifacts. Do not introduce a parallel preset object or silently infer missing semantic fields. NPZ and Kaldi backends must converge on the same `DefaultDataset` collation contract.

### Collated batch

| Key | Contract |
| --- | --- |
| `id` | sample ids in batch order |
| `tokens` | channel name to padded tensor |
| `mlm_mask` | channel name to padded boolean token mask |
| `length` | valid token length per sample after within-sample channel alignment |
| `token_start` | source token offset used by downstream record/event aggregation |
| `metadata` | encoded demographic/task fields and optional sidecar vectors |
| `pair` | selected two-channel pair when pair-first or pair evaluation is active |
| `w`, `h` | age/sex/source/path relationship matrices consumed by weighted contrastive loss |

Channel availability is resolved before loading a batch. A pair-scheduled batch must use one pair throughout, and channel sequences within each sample are cropped to a common valid length before batch padding.

### Task sidecars

- Survival sidecars are owned by `data.survival`: exact disease-column order, matching key sets, and `event_time`, `is_event`, `has_label` vectors.
- Multilabel sidecars are owned by `data.multilabel`: exact disease-column order, matching key sets, and `disease_label`, `has_label` vectors.
- The configured subject key remains metadata provenance. Repeated windows are aggregated by that key for subject-level metrics, while prediction exports retain path/window provenance.
- Built-in AHI uses token labels plus runtime `ahi`/`tst` metadata and a validation-fitted threshold stored in the checkpoint.

## Dependency Direction

- Entry points call config/common owners, loaders, Lightning modules, and artifact writers; they should not implement model or data semantics.
- Builders call registries and component modules; models should not parse raw YAML.
- `DefaultDataset` owns the shared collation path; storage backends override loading hooks, not batch shape.
- Preprocessing may import dataset/config helpers to validate generated artifacts; runtime modules should not depend on preprocessing CLIs.
- `sleep2stat` may load namespace-local downstream models and the shared record/data contract; training code does not depend on `sleep2stat`.
- `agent_tools` renders calls to existing runtime, preprocessing, and analysis entrypoints. Its context collection stays lightweight and avoids importing Torch or Lightning. External pipelines call package-local inference with a unique result root rather than interpreting runtime outputs through a second evaluator.

## High-Risk Seams And Tests

| Seam | Primary tests |
| --- | --- |
| YAML/task/registry contracts | [`tests/config/`](../../tests/config/) |
| Pair-first, bucket, and batch semantics | [`tests/data/`](../../tests/data/) |
| Downstream aggregation and losses | [`tests/models/`](../../tests/models/) |
| Checkpoints, outputs, AHI, survival, adaptation | [`tests/runtime/`](../../tests/runtime/) |
| Root-to-variant parity and namespace isolation | [`tests/variants/`](../../tests/variants/) |
| Preset, split, conversion, and utility behavior | [`tests/preprocess/`](../../tests/preprocess/) |
| Analysis config, analyzers, reducers, writers | [`tests/sleep2stat/`](../../tests/sleep2stat/) |
| Consultation, plans, hparam, and experiments | [`tests/agent_tools/`](../../tests/agent_tools/) |

## Variant Boundary

`sleep2vec2` and `sleep2expert` are standalone packages, not thin imports over root `sleep2vec`. Keep config, runtime, data, preprocessing, metrics, results, and visualization imports package-local. Shared contract changes require an explicit parity review; variant-only behavior stays in its namespace. `sleep2expert` additionally owns MoE configuration, router/expert execution, auxiliary loss, checkpoint expansion, routing analysis, and compact subnetwork export.
