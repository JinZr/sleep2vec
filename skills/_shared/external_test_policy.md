# External Test Policy

Hyper-parameter search selects on the explicitly frozen `selection_split`. Omitted `test_after_fit` defaults to `true`; generated commands render the resolved `--test-after-fit` or `--no-test-after-fit` choice explicitly.

For test-selected tuning, set `selection_split: test`, `external_test_locked: false`, and `test_after_fit: true`. The generated trial command evaluates every saved immutable `epoch=*.ckpt`; `hparam-select` requires complete `checkpoint_test_results`, ranks all checkpoint-level test metrics globally, and freezes the selected checkpoint path and SHA-256. A separate selected-model external evaluation still requires `--unlock-final-test`, and copied external-test configs may replace data entry fields only.

Adaptive hparam workflows that optimize test/external metrics must also set `adaptive.test_feedback_for_selection=true`, and every digest/ranking/report must mark the run as `external_optimized=true`.
