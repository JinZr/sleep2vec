# Workflow Map

This page maps durable end-to-end flows to their owners and contracts. It is
not a command cookbook. Before generating or running experiment commands,
follow the consultation and experiment-management policy in
[`AGENTS.md`](../../AGENTS.md) and the task skill under [`skills/`](../../skills/).

## Workflow Overview

| Workflow | Entrypoint family | Primary owner |
| --- | --- | --- |
| Preprocessing and presets | [`preprocess/`](../../preprocess/) | preset pipeline and data contracts |
| Config validation | [`utils/check_configs.py`](../../utils/check_configs.py) | config/task contract |
| Pretraining | `sleep2vec.pretrain` | runtime orchestration and model integration |
| Adaptation | `sleep2vec.adapt` | runtime orchestration and model integration |
| Finetuning | `sleep2vec.finetune` | runtime, config/task, model integration |
| Inference and evaluation | `sleep2vec.infer` | runtime orchestration and artifact owners |
| Variant embedding extraction | package-local `extract_embeddings` modules | runtime and variant maintainers |
| Derived analysis | `sleep2stat` | sleep2stat config, pipeline, and bundle writer |
| Agent planning and experiments | `agent_tools` | agent tooling and managed experiment owners |
| Standalone variants | `sleep2vec2`, `sleep2expert`, `sex_age_baseline` | package-local maintainers |

## Preprocessing And Presets

Canonical flow:

1. Prepare an index CSV with explicit dataset, split, path, channel, and label
   provenance.
2. Use [`preprocess/split_index_by_dataset.py`](../../preprocess/split_index_by_dataset.py)
   for split assignment and shared mask normalization when needed.
3. Inspect missing-channel coverage with
   [`preprocess/mask_missing_stats.py`](../../preprocess/mask_missing_stats.py).
4. Build NPZ presets through
   [`preprocess/save_dataset_presets.py`](../../preprocess/save_dataset_presets.py),
   or convert to a Kaldi root through
   [`preprocess/convert_npz_to_kaldi.py`](../../preprocess/convert_npz_to_kaldi.py).
5. Validate the generated artifact before using it in a runnable recipe.

Preset payloads remain `list[SampleIndex]`; missing-channel presets preserve
`payload["available_channels"]`. Kaldi changes storage discovery and loading,
not the collated batch shape. Split policy, label selection, required channels,
and preset regeneration are high-impact decisions and must not be inferred.
When config defines `preset_build`, it exclusively owns required-channel and
minimum-channel runtime policy; otherwise the preset recipe owns those CLI fields.

Built-in arousal preset generation consumes existing canonical
`arousal_event` targets rather than building labels from source event files.
NPZ and Kaldi outputs must converge on the same logical tokens, metadata, path,
and token offsets; the shape and runtime contract is summarized under
[Task sidecars](./MODULE_MAP.md#task-sidecars).

Variant recipes use their package-local preprocessing modules. A shared
contract change requires explicit parity review rather than a root import.

## Config Validation

Runtime loaders in package-local `config.py` files are authoritative for model,
data, and task semantics. [`utils/check_configs.py`](../../utils/check_configs.py)
adds repository policy for checked-in YAML and example coverage.

Validation order is:

1. parse YAML and required semantic blocks;
2. validate task, label, channel, tokenizer, head, and backend compatibility;
3. validate repository recipe conventions and tracked examples;
4. let runtime-only options remain CLI-owned.

Do not infer task semantics from config filenames. Optimization and logging
convenience may have defaults; model shape, label sources, thresholds, stage
sources, and output meaning must be explicit.

## Pretraining

The package-local pretrain entrypoint:

1. loads the pretrain config and binds runtime arguments;
2. builds train and validation loaders through the shared dataset contract;
3. constructs the pretrain model through registries and builders;
4. attaches loss, optimizer schedule, diagnostics, checkpoints, and logging;
5. persists the resolved config and CLI arguments before training.

Pair-first training and sequential pair evaluation are sampler responsibilities.
The monitored validation metric is part of the checkpoint/runtime contract, not
a value for downstream scripts to reinterpret. New runnable work must declare
an experiment and step and pass agent consultation before launch.

## Adaptation

Adaptation reuses the pretrain model and data path while applying staged
freezing and modality-pair schedules.

- Stage 1 introduces configured new channels while preserving the pretrained
  feature path.
- Stage 2 changes the trainable groups and pair schedule according to the
  explicit adapt config.
- A checkpoint path resumes the same phase; a pretrained-backbone path
  initializes weights or crosses a phase boundary.
- Phase-specific configs, CLI snapshots, and checkpoint directories remain
  runtime artifacts owned by the adaptation entrypoint.

Do not add a second trainer or silently guess the source checkpoint, phase, new
channels, or overwrite behavior.

## Finetuning

The finetune flow is:

1. consult the task skill and resolve high-impact label, split, checkpoint, and
   selection decisions;
2. load and apply finetune config semantics;
3. build package-local train, validation, and test loaders;
4. compose the pretrained feature path, aggregation, and downstream head;
5. train, select the configured checkpoint, and evaluate only authorized data;
6. write results through the canonical artifact owners.

Task-specific labels, thresholds, and aggregation remain with the package-local
data and metric owners named in the [module map](./MODULE_MAP.md) and
[reuse guide](./REUSE_GUIDE.md). Validation owns fitted AHI and arousal
thresholds; test and inference reuse checkpoint state without refitting.
Survival and multilabel metrics aggregate by the configured subject key while
prediction exports retain path/window provenance. Ordinary scalar metrics keep
window and explicitly named episode denominators separate.

External or final test data stays locked until the recorded decision allows it.
Hyperparameter ranking uses the split and metric frozen in the recipe; test
evidence is eligible only when tuning explicitly unlocks and evaluates test.
Direct finetune cannot select checkpoints on test; a fixed test-selected
configuration uses a one-configuration hparam plan so every epoch checkpoint
is evaluated, ranked, and hash-bound.

## Inference And Evaluation

Inference reuses finetune config, model, loader, metric, and prediction owners:

1. resolve an explicit checkpoint or supported alias;
2. build one evaluation loader for the requested split;
3. restore the package-local finetune model;
4. optionally select and average compatible checkpoints;
5. run evaluation and write metrics, predictions, per-disease tables, and the
   inference manifest under one prediction run id;
6. optionally log the same artifact family to W&B.

`--results-root` redirects that complete artifact family without changing its
schema. Managed pipeline attempts use one fresh root each and accept only the
unique terminal `run_manifest.json` below it.

Checkpoint averaging is a runtime policy and must preserve the task contract.
AHI and arousal reject averaging when thresholds are checkpoint-specific.
Arousal inference emits the `arousal_sequence` record family with four-column
truth, probability, and prediction timelines plus total-union and ArI summaries;
it does not create a parallel prediction NPZ. Result paths and CSV schemas belong
in `results.py` and `sleep2vec_inference.py`, not in a new evaluation script.

## Variant Embedding Extraction

The package-local `sleep2vec.extract_embeddings`, `sleep2vec2.extract_embeddings`,
and `sleep2expert.extract_embeddings` entrypoints build and strictly load their
complete model configs while allowing an explicit model-channel subset for data
loading and export. Whole-night extraction is an NPZ-only, one-sample-per-path
RoFormer mode with a caller-supplied hard source-token cap; it rejects clipping,
filtered coverage, non-final-layer dual export, and non-empty output roots.

`embedding-kind=both` obtains final-layer CLS and token matrices from the same
per-channel encoder forward and writes the two arrays under explicit NPZ keys.
The terminal manifest distinguishes full model channels from selected channels
and records input hashes, source-token coverage, and the extraction-only RoFormer
position capacity. Checkpoints load at their training position capacity before
the deterministic sinusoidal table is extended for extraction. The extractors do
not own subject pooling, downstream model selection, or sealed-test authorization.

## sleep2stat

`sleep2stat` is a derived-analysis runtime, not a trainer:

1. `validate-config` checks the strict analysis schema;
2. record loading resolves NPZ or supported Kaldi manifests;
3. configured analyzers emit `AnalyzerResult` objects;
4. reducers consume analyzer results and shared stage-source semantics;
5. `AnalysisBundleWriter` writes per-record sidecars, cohort tables, progress,
   and terminal manifests;
6. plot commands read completed bundles without repairing them.

Run directories are single-use. Failures propagate as command failures; no
partial-success, skip-existing, overwrite, or summary-time repair protocol is
implicit. Agent-generated commands use `task=sleep2stat` without a model
variant and must pass consultation first.

## Agent Planning And Managed Experiments

The control flow is:

1. `doctor` evaluates recipe decisions and stop-and-consult gates;
2. `context` records repository, config, index, preset, skill, and ownership
   facts without authorizing execution;
3. `plan` freezes the resolved recipe, commands, hashes, experiment, step, and
   run identities;
4. explicit launch commands execute the existing runtime entrypoints;
5. monitor and summary commands observe canonical artifacts but do not launch
   pending work;
6. `experiment-note` reads one local YAML entry file and appends the
   evidence-backed research milestone without changing lifecycle state;
7. finalization requires no active runs and a non-empty report.

Runnable plans use the exact config bytes accepted by consultation and freeze
their recipe, commands, hashes, paths, and run identities before execution.
Selection policy and test access are likewise frozen; test-selected hparam
plans retain complete checkpoint evidence while workspace lifecycle remains one
row per run. Filename guesses and caller-local fallbacks are never semantic
authority.

| Concern | Canonical owner | Authoritative contract |
| --- | --- | --- |
| Consultation and plan publication | [`agent_tools/decisions.py`](../../agent_tools/decisions.py), [`agent_tools/plans.py`](../../agent_tools/plans.py) | [task recipe](../agent_contracts/task_recipe.md) |
| Workspace state, launch, and monitoring | [`agent_tools/experiment_workspace.py`](../../agent_tools/experiment_workspace.py), [`agent_tools/hparam.py`](../../agent_tools/hparam.py) | [experiment workspace](../agent_contracts/experiment_workspace.md), [run manifest](../agent_contracts/run_manifest.md) |
| Hparam ranking and test access | [`agent_tools/hparam_selection.py`](../../agent_tools/hparam_selection.py) | [task recipe](../agent_contracts/task_recipe.md), [external test locking](../agent_contracts/external_test_locking.md) |
| Direct and Slurm lifecycle | [`agent_tools/managed_scheduler.py`](../../agent_tools/managed_scheduler.py), [`agent_tools/slurm.py`](../../agent_tools/slurm.py) | [run manifest](../agent_contracts/run_manifest.md) |
| External evaluation matrix | [`agent_tools/experiment_pipeline.py`](../../agent_tools/experiment_pipeline.py) | [experiment pipeline](../agent_contracts/experiment_pipeline.md) |
| Adaptive proposals | [`agent_tools/adaptive_proposals.py`](../../agent_tools/adaptive_proposals.py), [`agent_tools/adaptive_hparam.py`](../../agent_tools/adaptive_hparam.py) | [task recipe](../agent_contracts/task_recipe.md), [`agent_tools/ARCHITECTURE.md`](../../agent_tools/ARCHITECTURE.md) |

### Managed state and launching

`run_manifest.tsv` is the only lifecycle and execution-identity owner; status
tables, events, reports, and `RESEARCH_LOG.md` are projections or narrative.
Managed direct and Slurm follow-up always uses frozen canonical identity.
Slurm terminal truth normally combines scheduler and sidecar evidence; a purged
job with explicitly disabled accounting has one narrow authenticated recovery
path. Other uncertain observations remain nonterminal and never authorize
relaunch or retry.

For direct execution, `hparam-launch` starts one capacity-limited wave; Slurm
submits every launchable leaf job. `hparam-run-queue --execute` owns queue
advancement. `hparam-monitor` continuously rereads canonical state by default,
and `--once` performs one observation round; neither mode launches pending work.
Schema-v1 external evaluation remains direct-only.

Slurm priority warnings and `doctor` capability checks are diagnostic only;
they do not mutate frozen resource requests or promise scheduler priority.

At task handoff, read the experiment metadata, research log when present, and
then current canonical manifests. Add a note only when there is a meaningful
new action, observation, interpretation, decision, conclusion, or correction;
unchanged polling does not produce a log entry.

### External evaluation

`experiment-run` owns the resumable source-ranking-to-external-evaluation flow.
It accepts only checkpoint identities frozen by the registered ranking,
preflights external recipes, and runs package-local inference in isolated
attempt roots. Resume and retry require exact canonical evidence, and
finalization requires one verified success per declared job. See the
[experiment pipeline contract](../agent_contracts/experiment_pipeline.md).

### Adaptive proposals

Adaptive tuning keeps external-agent suggestions inside authenticated proposal
snapshots and parameter envelopes; planning, preflight, launch, and lifecycle
mutation remain tool-owned. Recipes with `adaptive.enabled=true` enter through
`hparam-adaptive-init`; the generic `plan` command does not publish incomplete
adaptive workspaces. `agent_proposal` is terminal-only, while automatic
neighborhood suggestions and active replacement require explicit
`best_neighborhood`. See the [task recipe contract](../agent_contracts/task_recipe.md)
and [`agent_tools/ARCHITECTURE.md`](../../agent_tools/ARCHITECTURE.md).

## Variants And Routing

Recipe `variant` determines the package-local runtime:

- `sleep2vec` uses the root dense implementation;
- `sleep2vec2` uses its standalone dense/RoFormer namespace;
- `sleep2expert` uses its standalone MoE namespace;
- `sex_age_baseline` uses its dedicated demographic baseline where supported;
- `sleep2stat` is a task and has no model variant.

Use package-local entrypoints. See the
[variant boundary](./MODULE_MAP.md#variant-boundary) for ownership and
[standalone variant guidance](./REUSE_GUIDE.md#standalone-variants) for reuse
and parity rules.

## Verification

Start with the focused suite in
[High-Risk Seams And Tests](./MODULE_MAP.md#high-risk-seams-and-tests), then run
the owning verification gate from [`AGENTS.md`](../../AGENTS.md).
