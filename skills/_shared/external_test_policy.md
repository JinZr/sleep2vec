# External Test Policy

Hyper-parameter search selects on validation only. Managed runs evaluate their configured test split after fit by default, but those test metrics must not affect candidate selection. Set `test_after_fit: false` together with `external_test_locked: true` for an explicit no-test run policy.

The selected-model external evaluation remains a separate explicit step. For hparam orchestration, `hparam-select` must rank candidates from validation metrics. `hparam-external-eval` requires `--unlock-final-test`, and copied external-test configs may replace data entry fields only.

Adaptive hparam workflows are the explicit exception: they may optimize test/external metrics only when `adaptive.test_feedback_for_selection=true`, and every digest/ranking/report must mark the run as `external_optimized=true`.
