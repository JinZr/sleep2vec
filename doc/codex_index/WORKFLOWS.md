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

External or final test data stays locked until the recorded decision allows it;
see [selection and test access](../agent_contracts/external_test_locking.md#selection-and-test-access-policy).
Direct finetune cannot select checkpoints on test; the supported route is a
[one-configuration hparam plan](../agent_contracts/external_test_locking.md#test-selected-runtime-requirements).

Managed tuning follows [search-space authorization](../agent_contracts/task_recipe.md#search-space),
[registration preflight](../agent_contracts/task_recipe.md#registration-preflight),
[launch and queue](../agent_contracts/task_recipe.md#launch-and-queue), then
[selection](../agent_contracts/task_recipe.md#selection-and-selected-candidate-consumers).
The hparam adapter delegates supported automatic profile expansion to
[`finetune_hparam_profile.py`](../../agent_tools/domain/finetune_hparam_profile.py).
The recipe contract owns candidate/config/argv validation and its evidence
limits; [workspace finalization](../agent_contracts/experiment_workspace.md#finalization)
owns report acceptance and final evidence checks.

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

For agent-managed local whole-night NPZ export, use
`task=embedding_extraction` and the
[`embedding_extraction` skill](../../skills/embedding_extraction/SKILL.md).
Planning freezes a RoFormer pretrain or finetune config with BERT-style CLS,
validates the explicit index and fresh output topology, binds checkpoint and
index CSV hashes for pre-launch verification, then routes to the selected
package-local entrypoint. Missing row sources use the authored index path, and
embedding output cannot occupy the experiment-managed `plans`, `reports`, or
`steps` namespaces.
Config-window, preset, Kaldi, source-override, remote, and alternate-runtime
workflows are outside this v1 task contract.

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

For an existing experiment, start with
[takeover and continue execution](../agent_contracts/experiment_workspace.md#takeover-and-continue-execution),
including its [identity legend](../agent_contracts/experiment_workspace.md#execution-identity-legend)
and [conditional initialization refresh](../agent_contracts/experiment_workspace.md#conditional-runtime-refresh-before-initialization).
For command choice, use the [agent contract router](../agent_contracts/README.md).

1. Resolve high-impact decisions with `doctor` through
   [consultation and diagnostics](../agent_contracts/task_recipe.md#consultation-and-diagnostics);
   stop on `NEEDS_USER_INPUT` and use the
   [decision input](../agent_contracts/user_decisions.md#generated-decision-templates).
   [`context`](../agent_contracts/context_bundle.md) is diagnostic, not execution authority.
2. Freeze ordinary plans with `plan` through
   [plan registration](../agent_contracts/experiment_workspace.md#publication-and-registration).
   Hparam [registration preflight](../agent_contracts/task_recipe.md#registration-preflight)
   owns final-config/argv checks and provenance limits.
3. Before a new launch wave, inspect or explicitly fast-forward the one runtime checkout with
   `runtime-sync`; this never clones, resets, or changes an already running process. Frozen plan commits remain
   baseline provenance while each new run records the checkout commit seen at start.
4. Execute through [launch and queue](../agent_contracts/task_recipe.md#launch-and-queue),
   with [snapshot revalidation](../agent_contracts/task_recipe.md#execution-snapshot-and-launch-revalidation).
   A rolling checkout may use a newer commit, but it must expose the current managed launch protocol before a direct
   claim or Slurm submission. Direct and Slurm starts use point-in-time identity and artifact checks under the short
   runtime lock; the self-contained Slurm bootstrap forwards signals and records checkout-local import/start failures.
   Other ordinary routes use the
   [non-hparam runtime contract](../agent_contracts/task_recipe.md#non-hparam-runtime-identity);
   `preset-launch` / `preset-stop` use
   [managed preset preparation](../agent_contracts/task_recipe.md#managed-preset-preparation);
   `infer-launch` / `infer-stop` use the shared
   [managed ordinary Slurm inference](../agent_contracts/task_recipe.md#managed-ordinary-inference) owner.
5. Inspect recorded state with
   [read-only status](../agent_contracts/experiment_workspace.md#read-only-status-and-advisory-actions),
   or explicitly refresh evidence with non-launching monitors; the
   [entrypoint side-effect table](../agent_contracts/experiment_workspace.md#lifecycle-entrypoints)
   distinguishes them. [Run-manifest evidence](../agent_contracts/run_manifest.md#evidence-ownership)
   and [Slurm evidence](../agent_contracts/run_manifest.md#slurm-scheduler-evidence)
   own lifecycle interpretation.
6. Select and consume candidates through the
   [selection and consumer workflow](../agent_contracts/task_recipe.md#selection-and-selected-candidate-consumers),
   append meaningful [research notes](../agent_contracts/experiment_workspace.md#research-log),
   then follow [finalization](../agent_contracts/experiment_workspace.md#finalization).

Adaptive recipes enter through
[`hparam-adaptive-init`](../agent_contracts/task_recipe.md#initialization-readiness)
and follow the [proposal handshake](../agent_contracts/task_recipe.md#proposal-handshake).
An exact committed agent-proposal execute is safe to retry after a lost client
receipt only when successful completion is recorded; incomplete, conflicting,
or uncommitted launch evidence remains fail closed, and monitors never gain
launch authority.
External matrices use [`experiment-run`](../agent_contracts/experiment_pipeline.md#invocation-and-frozen-state)
over [registered-ranking-selected checkpoints](../agent_contracts/experiment_pipeline.md#source-and-checkpoint-gates),
with [managed attempts](../agent_contracts/experiment_pipeline.md#managed-attempts-and-results)
and [completion gates](../agent_contracts/experiment_pipeline.md#completion-and-finalization).
Recurring follow-up starts from the workspace's
[short heartbeat guidance](../agent_contracts/experiment_workspace.md#short-heartbeat-maintenance).

For implementation changes, use the [canonical reuse table](./REUSE_GUIDE.md#canonical-implementations),
[agent tooling boundaries](./REUSE_GUIDE.md#agent-tooling), and
[`agent_tools/ARCHITECTURE.md`](../../agent_tools/ARCHITECTURE.md).

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
