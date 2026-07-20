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

Runnable preflight validates relative runtime paths from the same frozen
workdir used by the generated command, while planning-source config locators
remain repository-relative. Frozen Python identity is one executable token;
Conda wrapping is a separate hparam execution field.

`experiment-run` adds one explicit, resumable external-evaluation flow:

1. dry-run validates the strict v1 spec and starts no process;
2. execute waits until every declared hparam source is successful and inactive;
3. the validation-ranking owner selects and freezes one checkpoint per source;
4. all external recipes pass consultation and final-test unlock preflight before
   launch;
5. managed jobs run package-local inference on assigned physical GPUs with
   package-local logical device 0 and isolated result roots;
6. only canonical runtime `failed` or `launch_failed` receives one fresh retry;
   result-manifest verification failures remain logical, non-retryable job
   failures;
7. after every job frozen from the user-selected `--spec` has one verified
   success, the pipeline writes its N/N report before the final experiment
   commit.

Existing pipeline state requires explicit `--resume --execute` and exact frozen
spec, source, checkpoint, config, and runtime identity. Uncertain process
identity is never permission to retry. External metrics are report-only and do
not feed checkpoint selection. `hparam-monitor` and `experiment-monitor` remain
non-launching throughout this flow.

A successful plan binds the exact config bytes accepted by consultation and
materializes from that immutable snapshot even if the source path changes
later. Structural config-family ownership may constrain recipe `variant`;
filename- or directory-derived variant guesses are diagnostic only.

Step and run manifests commit through their canonical merge-and-compare-swap
owners. Remote run matrices are version-checked projections of
`run_manifest.tsv`, and experiment finalization publishes the report before
the terminal experiment manifest. Managed launches create a dedicated process
group and freeze its PID, process-group id, and OS start token. Launch timeout
is unresolved evidence rather than permission to retry; a timeout-derived first
fill also matches the live leader command to the frozen launch script. Stop
verifies that identity, signals the full group, and records `stopped` only after exit.
Lifecycle-enabled scripts own their terminal success/failure commit, so their
uncommitted process-group disappearance is failure rather than inferred
success. Hparam launch scripts instead leave terminal inference to their
monitor after confirmed process exit.

Public facades are `decisions.py`, `plans.py`, `hparam.py`, and
`experiments.py`. Shared managed scheduling lives in `managed_scheduler`, while
external-matrix policy lives in `experiment_pipeline`. Task-specific behavior
extends the adapter/domain layers; managed workspace identity and
`run_manifest.tsv` remain canonical owners.
Follow [`agent_tools/ARCHITECTURE.md`](../../agent_tools/ARCHITECTURE.md) and
[`doc/agent_contracts/`](../agent_contracts/) for detailed machine contracts.

Adaptive tuning defaults to `agent_proposal` when strategy is omitted. It uses
a terminal-only two-phase handshake: `hparam-adaptive-step` writes and records
the full-file hash of a tool-issued proposal-input v2 snapshot under
`adaptive/proposal_inputs/`. Explicit `best_neighborhood` remains available
for automatic neighborhood suggestions and active-round replacement. The
default proposal workflow also binds its request id to the exact source config
bytes. An external agent may write only the
named submission under `adaptive/proposal_submissions/`; a second invocation
previews or explicitly executes the bounded proposal. Phase two requires the
matching issuance and snapshot bytes, then rebuilds the input from current
canonical workflow state before and after candidate preflight. Config or
canonical-state drift therefore fails before lifecycle mutation. Execute
freezes the verified source-config bytes inside the next round and materializes
from the validated in-memory proposal rather than re-reading the suggestion.
Existing v1 input snapshots must be regenerated. Agent proposals cannot own
replacement, planning, launch, or `run_manifest.tsv` lifecycle state.

The shared Codex index only supplies navigation paths in context bundles. It
does not authorize commands or replace live repository inspection.

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
