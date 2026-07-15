# User Decisions

User-decision files resolve high-impact ambiguity with explicit user intent. The file is a closed mapping with exactly one top-level field, `decisions`. An explicitly supplied file must contain that mapping.

Decision names are task-aware: each name must be applicable to the current task through `agent_policies/consultation_policy.yaml` or an existing owner-local optional decision. Mapping entries accept only `value`, `source`, `meaning`, `question`, and `rationale`; scalar shorthand is also accepted. Unknown names and misspelled entry fields fail before context or plan output is written.

Concrete values with a task-owned canonical field are materialized into the effective recipe's existing `inputs`, `evaluation_policy`, `preset`, `search`, or artifact fields before config inspection and consultation are rerun. Policy-only choices remain under `decisions` rather than creating inert canonical fields. In particular, non-preset `required_channels` is checked against the selected config, `preset_regeneration` remains decision evidence while `preset.overwrite` controls the rendered overwrite flag, and hparam `ckpt_path` selects only the final-evaluation checkpoint. A user `task` may fill a missing recipe task but cannot replace an explicit recipe task. For layered hparam recipes, the user task is compared with the local overlay rather than the base finetune task. `train_val_test_policy` must be exactly `train`, `val`, or `test`; descriptive text is not interpreted as a split.

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
