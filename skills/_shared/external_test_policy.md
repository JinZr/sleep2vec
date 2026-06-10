# External Test Policy

Hyper-parameter search selects on validation only. External test evaluation is a separate explicit final step and must not happen during trial generation.

For hparam orchestration, `hparam-select` must rank candidates from validation metrics. `hparam-external-eval` requires `--unlock-final-test`, and copied external-test configs may replace data entry fields only.

Adaptive hparam workflows are the explicit exception: they may optimize test/external metrics only when `adaptive.test_feedback_for_selection=true`, and every digest/ranking/report must mark the run as `external_optimized=true`.
