# External Test Policy

Hyper-parameter search selects on the explicitly frozen validation `selection_split`; the test split is never a selection split, and recipe validation rejects `selection_split: test`. Omitted `test_after_fit` defaults to `true`; generated commands render the resolved `--test-after-fit` or `--no-test-after-fit` choice explicitly.

A selected-model external evaluation requires `--unlock-final-test`, and copied external-test configs may replace data entry fields only.

