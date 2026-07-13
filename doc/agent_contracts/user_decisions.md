# User Decisions

User-decision files resolve high-impact ambiguity with explicit user intent.

An explicitly supplied file must contain a `decisions` mapping. Concrete values are materialized into the effective recipe's existing `inputs`, `evaluation_policy`, `preset`, `search`, or artifact owner before config inspection and consultation are rerun. A user `task` may fill a missing recipe task but cannot replace an explicit one. `train_val_test_policy` must be exactly `train`, `val`, or `test`; descriptive text is not interpreted as a split.

```yaml
decisions:
  label_name:
    value: ahi
    source: explicit_user
    rationale: "The current experiment is AHI prediction."

  pretrained_backbone_path:
    value: checkpoints/sleep2vec_pretrain/best.ckpt
    source: explicit_user
    rationale: "Use the pretrained backbone from the previous pretraining run."

  external_test_locked:
    value: true
    source: explicit_user
    rationale: "This is hyper-parameter tuning; test should remain untouched."

  test_after_fit:
    value: false
    source: explicit_user
    rationale: "Do not evaluate test after each run."

  overwrite_policy:
    value: false
    source: explicit_user
    rationale: "Avoid overwriting existing results."
```

Resolution precedence:

1. explicit user-decision file
2. explicit CLI argument
3. explicit recipe decision
4. explicit recipe field
5. explicit config field
6. approved default
7. ambiguous or missing
