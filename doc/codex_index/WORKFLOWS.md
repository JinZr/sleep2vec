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

Built-in stage tasks remap raw `stage5` labels where appropriate. AHI uses
event-aware validation/test reduction and a validation-fitted threshold stored
with the checkpoint. Survival and multilabel tasks load explicit subject-level
sidecars, aggregate repeated windows by the configured key, and retain
path/window provenance in prediction outputs.

External or final test data stays locked until the recorded decision allows it.
Hyperparameter ranking is validation evidence, not final-test evidence.

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
AHI rejects averaging when thresholds are checkpoint-specific. Result paths and
CSV schemas belong in `results.py` and `sleep2vec_inference.py`, not in a new
evaluation script.

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
6. finalization requires no active runs and a non-empty report.

Runnable plans bind the exact config bytes accepted by consultation and
materialize from that immutable snapshot. Runtime-semantic relative paths are
validated from the frozen workdir, while planning-source config locators remain
repository-relative. Structural config ownership may constrain recipe
`variant`; filename- or directory-derived guesses are diagnostic only. See
[task_recipe.md](../agent_contracts/task_recipe.md) for recipe, path, runtime
identity, and adaptive semantics.

Staged plan bytes still freeze the final semantic paths. A plan becomes
runnable only after the complete bundle is published and its step and run rows
are canonically registered. Adaptive round 000 then publishes
`adaptive/workflow.json` last; launch and queue fail closed without that marker.
An unregistered complete round may be resumed only after exact deterministic
plan-tree comparison, while incomplete or partial-canonical rounds are not
repaired.

### Managed state and launching

`run_manifest.tsv` is the only lifecycle and execution-identity owner. Status
tables, matrices, events, and reports are projections. Managed launches use a
dedicated process group with PID, group, and OS start-token evidence; uncertain
identity never authorizes relaunch or retry. Stop verifies and signals the full
group before committing `stopped`.

`hparam-launch` starts one capacity-limited wave, while
`hparam-run-queue --execute` owns continuous queue advancement. Monitor commands
remain non-launching. Workspace layout and lifecycle entrypoints are defined in
[experiment_workspace.md](../agent_contracts/experiment_workspace.md); reducer,
commit, process, and evidence rules are defined in
[run_manifest.md](../agent_contracts/run_manifest.md).

### External evaluation

`experiment-run` owns the resumable validation-to-external-test flow. It
validates the strict spec without launching in dry-run mode, waits for successful
managed sources, freezes validation-selected checkpoints, preflights every
external recipe, and runs package-local inference in isolated result roots.
Existing state resumes only with its exact frozen identity. Only explicit
retryable canonical failures receive a fresh attempt; uncertain identity or
result-manifest validation failure does not. External metrics remain report-only,
and finalization requires one verified success for every declared job.

The complete spec, retry, result-manifest, and finalization contract is in
[experiment_pipeline.md](../agent_contracts/experiment_pipeline.md).

### Adaptive proposals

Adaptive tuning defaults to terminal-only `agent_proposal`; automatic
neighborhood suggestions and active replacement require explicit
`best_neighborhood`. The proposal flow uses a tool-issued input v2 snapshot that
binds the exact source config bytes and a single named external submission.
Phase two authenticates the issuance, reconstructs current canonical evidence,
and repeats validation around candidate preflight before any lifecycle mutation.
The external agent proposes only the search space; planning, launch, and
`run_manifest.tsv` state remain tool-owned.

The detailed handshake is defined in
[task_recipe.md](../agent_contracts/task_recipe.md). Public facades remain
`decisions.py`, `plans.py`, `hparam.py`, and `experiments.py`; shared scheduling
lives in `managed_scheduler`, external-matrix policy in `experiment_pipeline`,
and task-specific behavior in adapters/domain modules. See
[`agent_tools/ARCHITECTURE.md`](../../agent_tools/ARCHITECTURE.md) for layering.

This index supplies navigation only. It does not authorize commands or replace
live repository inspection.

## Variants And Routing

Recipe `variant` determines the package-local runtime:

- `sleep2vec` uses the root dense implementation;
- `sleep2vec2` uses its standalone dense/RoFormer namespace;
- `sleep2expert` uses its standalone MoE namespace;
- `sex_age_baseline` uses its dedicated demographic baseline where supported;
- `sleep2stat` is a task and has no model variant.

Do not route variant recipes through root entrypoints. Root-to-variant changes
to config, data, checkpoints, metrics, results, callbacks, tokenizers, or model
interfaces require package-local parity validation. `sleep2expert` routing
analysis reads MoE routing outputs; compact artifacts are created through its
subnetwork export owner rather than manual checkpoint surgery.

## Verification Routing

Use the smallest focused suite first, then the owning gate from `AGENTS.md`:

| Change | Focused tests |
| --- | --- |
| config/task/builders | `tests/config/` |
| dataset/sampler/preset contracts | `tests/data/`, `tests/preprocess/` |
| model/head/loss behavior | `tests/models/` |
| checkpoints, metrics, results, entrypoints | `tests/runtime/` |
| root-to-variant parity | `tests/variants/` |
| analysis bundles | `tests/sleep2stat/` |
| consultation and experiment management | `tests/agent_tools/` |

Runtime smoke commands are warranted only when the touched contract is not
fully represented by focused tests. Variant validation is mandatory whenever a
shared change can affect `sleep2vec2` or `sleep2expert`.
