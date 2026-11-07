# UKB clinical covariates

The [clinical Cox example](../configs/sex_age_baseline/ukb_clinical_cox.yaml) uses `sex_age_baseline` with a three-layer MLP, hidden width 256, and 624 outputs. The [joint clinical Cox example](../configs/sleep2vec2/ukb_joint_clinical_cox.yaml) enables LayerMix and mean pooling in `sleep2vec2`. Its existing regression head receives the pooled signal representation and the clinical embeddings through `feature_concat`. Both examples emit raw log-risk for the same ordered disease columns. Their fold statistics, category orders, signal configuration, and paths are illustrative.

## Prepared inputs

Fit the imputation medians, standardization statistics, and category definitions upstream using one row per training participant. Apply that fold's definitions to validation and held-out regions. Join the prepared table to the signal index by participant before preset generation so that all nights carry the same values. Use the same prepared preset and `fold_preprocessing` in both arms. The baseline reads metadata only and collapses rows by the task key. The joint model retains the signal records and their associated metadata. When using a raw index, the joint loader additionally applies its ordinary signal-window and channel eligibility checks.

| Index or preset columns | Input representation | Model encoding |
| --- | --- | --- |
| `age`, `bmi`, `sbp`, `dbp`, `total_cholesterol`, `hdl_cholesterol`, `ldl_cholesterol`, `hba1c` | Finite values in original units, with missing values already imputed | `float32` batch column, then `(value - mean) / std` and a linear projection to 16 dimensions |
| `sex` | Existing 0 = female, 1 = male encoding | Two-row embedding |
| `smoking_status` | Integer ID for never, previous, current, or unknown | Four-row embedding |
| `diabetes_history`, `antihypertensive_use`, `lipid_lowering_use`, `parental_cvd_history` | Integer ID for no, yes, or unknown, with three distinct IDs per field | Three-row embedding per field |
| `ethnicity` | Integer ID for White, Asian, Black, Mixed or other, or unknown | Five-row embedding |
| `alcohol_frequency` | Integer ID for one of six frequency levels or unknown | Seven-row embedding |
| One `<name>_missing` for each continuous variable | Numeric 0 = observed, 1 = imputed | Two-row embedding per flag |

The batch uses `torch.long` for IDs and indicators. The position of a label in `fold_preprocessing.categorical.<name>` is its zero-based ID, including `unknown`. Supply the IDs in the prepared metadata. YAML parses unquoted `no` and `yes` as booleans, so quote them in category lists. Missing fields, nonfinite continuous values, and invalid IDs or indicators raise errors in fold mode. An imputed value paired with an indicator of 1 and a valid unknown ID are eligible inputs.

## Frozen fold configuration

Set `fold_preprocessing` under the selected `finetune.survival` or `finetune.multilabel` block. Its `continuous` mapping has one `{median, mean, std}` entry per selected continuous column, with a positive standard deviation. Its `categorical` mapping has the ordered labels for each selected multiclass column. Keep raw units in the index and preset. The model uses the mean and standard deviation once, and the saved median records the upstream imputation parameter. Sex and missing indicators use the fixed binary encoding.

The examples select 24 inputs. The feature order keeps the historical slots first, then the additional continuous columns, multiclass columns, and indicators in the example order. Fold-mode configs and checkpoints record the canonical order, category definitions, and scaling. The `sleep2vec2` fold path requires `feature_concat`. Legacy age/sex and age/sex/BMI configurations continue to use `covariate_normalization` and the historical age scaling when age statistics are absent.

## Detection

For either Detection arm, use `finetune.task.type: multilabel_classification`, `output_dim: 624`, `is_seq: false`, and the Detection monitor (for example, `val_macro_auroc` with `monitor_mod: max`). Move the task block from `finetune.survival` to `finetune.multilabel` and replace `event_time_index` and `is_event_index` with `label_index`. Point `disease_columns_index` and `has_label_index` to the matching Detection sidecars. The baseline's classification head stays in place, and its Cox-only `finetune.loss.eps` is removed. For the joint arm use the existing Detection classification head with mean pooling, retaining LayerMix and `feature_concat`. A classification head with covariate embeddings uses an activation such as ELU with nonzero derivative at zero. Detection emits raw logits.
