# YAML-driven covariate baseline

`sex_age_baseline` is a covariate-only Cox or multilabel model. It keeps the
existing package, CLI and managed variant name; it never loads a physiological
signal array or constructs a backbone. Start with the
[Cox](../configs/sex_age_baseline/cox.yaml) or
[multilabel](../configs/sex_age_baseline/multilabel.yaml) model template and the
matching [managed recipes](../recipes/templates/).

## Model and task contract

`model.features` is an ordered, nonempty subset of `age`, `sex`, `bmi` (all seven
combinations are supported). Include exactly the corresponding encoding blocks:

```yaml
model:
  name: sex_age_mlp
  features: [age, sex, bmi]
  age:
    transform: divide
    scale: 100.0
    embedding_dim: 16
    initialization: zeros
  sex:
    encoding: binary
    embedding_dim: 16
    initialization: zeros
  bmi:
    transform: divide
    scale: 1.0
    embedding_dim: 16
    initialization: default
  head:
    name: classification
    hidden_dim: 32
    dropout: 0.1
    act: elu
    kwargs:
      num_layers: 3
```

Continuous features are divided by the positive scale, then linearly projected;
no normalization statistics are fitted. Sex uses the existing binary encoding.
Each enabled block explicitly chooses `zeros` or PyTorch `default`
initialization. To use BMI only, set `features: [bmi]` and remove `age` and `sex`.
Do not retain inactive encoding blocks.

`head.act: relu` requires `initialization: default` for every active encoder:
the production head applies activation before its first Linear, so ReLU would
block all gradients to a zero-initialized encoder. This combination is rejected.

The `classification` dense head supports one, two or three layers and reuses
the production head's activation/Linear/dropout ordering. Cox outputs raw
log-risk; multilabel outputs logits. Sidecar labels, masks and task mathematics
are shared with the existing task implementations. Multilabel exposes the
existing `finetune.loss.pos_weight`; Cox exposes `finetune.loss.eps` (default
`1e-9`). No new loss function is introduced.

## Data units

The index, preset metadata or Kaldi manifest must contain every selected
feature. BMI's field is exactly `bmi`; no aliases or derived values are inferred.
Missing/non-finite selected features and illegal sex encodings fail with counts.
Unselected columns are unnecessary. There is no imputation, participant removal
or new clinical range filter.

- `data.deduplicate_by_key: true`: one consistent row per participant.
- `false`: preserve the authored sample/window sequence and repeated-participant
  training weight. Presets supply `SampleIndex.path` and `start`; ordinary
  indexes/manifests must explicitly supply `path` and `token_start`. A plain
  participant index is not silently interpreted as a window-matched dataset.

Cross-split participants and inconsistent participant features/labels are
rejected. Evaluation collects predictions across ranks, removes padding copies
by `(key, path, token_start)`, then averages raw outputs per participant before
metrics (sigmoid follows aggregation for multilabel). Cox training risk sets
remain **rank-local batches**, not a global distributed risk set.

## Training and evaluation

Model/data/task semantics belong to the model YAML. Recipe `runtime` and the
CLI own epochs, batch size, learning rate, devices, precision, accumulation,
clipping, validation cadence and checkpoint cadence. Lightning executes these
settings for both single-device and DDP runs; only rank zero writes outputs.
Distributed training drops the sampler tail before forming local batches,
then drops incomplete local batches, so no padding copies contribute to loss;
validation and test retain all samples before distributed-padding deduplication.
AdamW uses betas `(0.9, 0.95)`, epsilon `1e-8`, and production decay/no-decay
grouping. `lr_scheduler: decay|wsd|plateau` and its warmup, decay and Plateau
fields select the shared `sleep2vec` scheduler with the same defaults and
validation; step schedules use the actual trainer step budget.

`python -m sex_age_baseline.finetune` and `python -m sex_age_baseline.infer`
take the same options as their `sleep2vec` counterparts, so recipes and
`agent_tools` render one command shape for every variant. Training seeds from
the fixed `sleep2vec` finetune seed; inference keeps `--seed`. W&B routing,
`--device`, version naming, `--export-predictions` and inference checkpoint
averaging (including `best`/`last` aliases with `--avg-ckpt-dir`) follow
`sleep2vec`. Prediction CSVs are written only when the CLI requests them; the
YAML has no output switches. The baseline has no backbone or diagnostics mode:
`--pretrained-backbone-path` and `--print-diagnostics` fail at launch, and
`--diagnostics-steps` is inert without the latter, as in `sleep2vec`. The YAML
task alone sets task semantics; `--label-name` only names the result
namespace. Inference does not offer `--override-dataset-names`.

Keep the existing choice of best-checkpoint test, explicit all-saved-epoch test,
or `test_after_fit: false`. Independent inference loads the same strict model
and label contract. Checkpoints include the model configuration, feature order,
label contract, optimizer and scheduler state; this does not add automatic
resume. Historical YAML/checkpoints must be read by their original frozen code;
incompatible contracts fail rather than being migrated or reinterpreted.

Managed consultation, test locks, frozen identities, lifecycle and selection
gates remain mandatory. Multi-GPU support does not grant test access, expand a
search domain or create a tuning budget. Use the existing `agent_tools`
doctor/plan workflow before producing runnable experiment commands.
