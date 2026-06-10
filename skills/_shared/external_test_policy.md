# External Test Policy

Hyper-parameter search selects on validation only. External test evaluation is a separate explicit final step and must not happen during trial generation.

For hparam orchestration, `hparam-select` must rank candidates from validation metrics. `hparam-external-eval` requires `--unlock-final-test`, and copied external-test configs may replace data entry fields only.
