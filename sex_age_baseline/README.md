# Covariate baseline

`sex_age_baseline` is a covariate-only Cox or multilabel model: the signal
models' covariate pathway with no backbone. It keeps the existing package, CLI
and managed variant name; it never loads a physiological signal array. Start
with the [Cox](../configs/sex_age_baseline/cox.yaml) or
[multilabel](../configs/sex_age_baseline/multilabel.yaml) model template and the
matching [managed recipes](../recipes/templates/).

## Model and task contract

The `model` block names the model and its dense head only:

```yaml
model:
  name: sex_age_mlp
  head:
    name: classification
    hidden_dim: 32
    dropout: 0.1
    act: elu
    kwargs:
      num_layers: 3
```

Covariates are declared in `finetune.survival` or `finetune.multilabel` with the
same fields, validation and encoders as `sleep2vec` and `sleep2vec2`
(`sleep2vec/modules/covariates.py`):

```yaml
finetune:
  survival:
    key_column: eid
    # ... label sidecars ...
    covariates: [age, sex, bmi, bmi_missing]
    covariate_embedding_dim: 16
    covariate_normalization:
      bmi: {mean: 26.1, std: 4.3}
```

`covariates` is a nonempty, duplicate-free subset of `age`, `sex`, `bmi` and
`bmi_missing`, always encoded in that canonical order. `age` and `bmi` use
zero-initialized `Linear(1, d)` projections of standardized values; `sex` and
`bmi_missing` use zero-initialized `Embedding(2, d)` tables. Normalization
statistics are frozen in the YAML, fitted on the training split: `bmi` requires
an entry, `age` without one falls back to `age / 100`, and entries are accepted
only for selected `age`/`bmi`.

The head applies its activation to the concatenated embeddings before its first
Linear, so `head.act` must be `elu`, `gelu` or `silu`; ReLU would block every
gradient to the zero-initialized encoders. The `classification` dense head
supports one, two or three layers and reuses the production head's
activation/Linear/dropout ordering. Cox outputs raw log-risk; multilabel
outputs logits. Label sidecars, masks and task mathematics are shared with the
existing task implementations. Multilabel exposes `finetune.loss.pos_weight`;
Cox exposes `finetune.loss.eps` (default `1e-9`).

## Data

The baseline reads the matching signal run's data source. The `data` block has
the signal finetune spellings and semantics: `backend` (`npz` or `kaldi`),
`finetune_data_index`, `finetune_preset_path`, `kaldi_data_root`,
`kaldi_manifest`, `train_dataset_names` and `test_dataset_names`. Like the
signal templates, the checked-in YAML leaves the source null for the recipe or
CLI to bind. Only metadata is read: index CSV rows, preset `metadata` mappings,
or the per-split CSVs a Kaldi `manifest.json` lists (each split keeps the rows
whose `split` column names it, as `KaldiPSGDataset` does).

- The split column is `split`. Rows collapse to one record per
  `finetune.{survival|multilabel}.key_column` value; a key whose rows encode
  different covariates fails, and so does a key retained in two loaded splits
  after the filters below.
- Dataset-name filters match the signal loader: `train_dataset_names` for
  train/val, `test_dataset_names` for test, and inference
  `--override-dataset-names` in place of either.
- Rows whose selected covariates are missing or invalid are dropped with a
  logged count, as the signal loader's required-metadata filter does.
- A preset supplies the labels embedded at preset generation; an index or Kaldi
  manifest reads the label sidecars. Label names always come from
  `disease_columns_index`, because they belong to the checkpoint label contract.
- `bmi` is raw BMI (imputed upstream) and `bmi_missing` its 0/1 imputation
  indicator; both must be columns of the index or manifest and are copied into
  regenerated presets. A preset built before these columns existed cannot serve
  a BMI recipe.

With a shared preset the baseline and the signal run see the same cohort. From
a raw index or Kaldi manifest, the signal model additionally drops windows by
duration, NPZ validity and available channels, which the baseline cannot see;
point both runs at the same preset when the cohorts must match exactly.

Evaluation writes one prediction per subject (`path` is the key and
`n_windows` is 1). Predictions are gathered across ranks and distributed
padding copies are removed before metrics. Cox training risk sets remain
**rank-local batches**, not a global distributed risk set.

## Training and evaluation

Model/data/task semantics belong to the model YAML. Recipe `runtime` and the
CLI own epochs, batch size, learning rate, devices, precision, accumulation,
clipping, validation cadence and checkpoint cadence. Lightning executes these
settings for both single-device and DDP runs; only rank zero writes outputs.
Training loads every split it uses before it creates `log-finetune/<version>`,
so a data, sidecar or cohort error leaves no run directory behind.
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
`--device`, version naming, `--export-predictions`, `--override-dataset-names`
and inference checkpoint averaging (including `best`/`last` aliases with
`--avg-ckpt-dir`) follow `sleep2vec`. Prediction CSVs are written only when the
CLI requests them; the YAML has no output switches. The baseline has no
backbone or diagnostics mode: `--pretrained-backbone-path` and
`--print-diagnostics` fail at launch, and `--diagnostics-steps` is inert
without the latter, as in `sleep2vec`. The YAML task alone sets task
semantics; `--label-name` only names the result namespace.

Keep the existing choice of best-checkpoint test, explicit all-saved-epoch test,
or `test_after_fit: false`. Independent inference loads the same strict model
and label contract. Checkpoints include the model contract (covariates,
embedding width, normalization and head), the label contract, optimizer and
scheduler state; this does not add automatic resume. Checkpoints written before
the covariate contract was shared with the signal models carry a different
contract and fail to load; historical YAML and checkpoints must be read by their
original frozen code rather than migrated or reinterpreted.

Managed consultation, test locks, frozen identities, lifecycle and selection
gates remain mandatory. Multi-GPU support does not grant test access, expand a
search domain or create a tuning budget. Use the existing `agent_tools`
doctor/plan workflow before producing runnable experiment commands.
