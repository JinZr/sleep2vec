# User Decisions

User-decision files resolve high-impact ambiguity with explicit user intent. An explicitly supplied file must contain a `decisions` mapping.

Concrete values are materialized into the effective recipe's existing `inputs`, `evaluation_policy`, `preset`, `search`, or artifact fields before config inspection and consultation are rerun. A user `task` may fill a missing recipe task but cannot replace an explicit recipe task. `train_val_test_policy` must be exactly `train`, `val`, or `test`; descriptive text is not interpreted as a split.

```yaml
decisions:
  label_name:
    value: ahi
    source: explicit_user
    rationale: Use the AHI label for this experiment.

  external_test_locked:
    value: true
    source: explicit_user
    rationale: Keep test data locked during tuning.

  overwrite_policy:
    value: false
    source: explicit_user
```

Resolution precedence is:

1. explicit user-decision file
2. explicit CLI argument
3. explicit recipe decision
4. explicit recipe field
5. explicit config field
6. ambiguous or missing

An empty or `ASK_USER` config decision remains unresolved. For other fields, empty values are not materialized and the field-specific consultation rule determines whether they block. The one intentional null semantic is `pretrained_backbone_path: null`, which explicitly selects training without a pretrained backbone. Concrete materialized values use only the canonical field and do not create an alternate semantic source.
