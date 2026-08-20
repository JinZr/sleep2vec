# External Test Policy

Hyper-parameter search selects on validation only. Managed candidate trials must explicitly set `test_after_fit: false`, render `--no-test-after-fit`, and exclude the test split from preflight so checkpoint selection is frozen before external metrics are read.

The selected-model external evaluation remains a separate explicit step. For hparam orchestration, `hparam-select` must rank candidates from validation metrics. `hparam-external-eval` requires `--unlock-final-test`, and copied external-test configs may replace data entry fields only.

Adaptive hparam workflows are the explicit exception: they may optimize test/external metrics only when `adaptive.test_feedback_for_selection=true`, and every digest/ranking/report must mark the run as `external_optimized=true`.
