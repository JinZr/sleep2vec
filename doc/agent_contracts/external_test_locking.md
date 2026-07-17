# External Test Locking

- Hyper-parameter search selects on validation.
- External test is locked during runs by default.
- Run test-after-fit may run only when `external_test_locked=false`, `final_test_unlocked=true`, and `test_after_fit=true` are explicit.
- Final external-test evaluation is a separate, explicit command.
- Final external-test scripts require an explicit existing checkpoint path; unlock does not authorize checkpoint guessing.
- If final script generation is skipped, a stale `final_external_test.sh` must be blocked or removed under explicit overwrite approval.
- Hparam recipes with `yaml:/...` config overrides also require an explicit selected final-test config path.
- Agents should report when a recipe would violate this.
