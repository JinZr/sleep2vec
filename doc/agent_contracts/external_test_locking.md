# External Test Locking

- Hyper-parameter search selects on validation.
- External test is locked during runs by default.
- Run test-after-fit may run only when `external_test_locked=false`, `final_test_unlocked=true`, and `test_after_fit=true` are explicit.
- Final external-test evaluation is a separate, explicit command.
- Final external-test scripts require an explicit existing checkpoint path; unlock does not authorize checkpoint guessing.
- If final script generation is skipped, stale `final_external_test.sh` and frozen final-test config artifacts must be blocked or removed under explicit overwrite approval.
- Hparam recipes with `yaml:/...` config overrides also require an explicit selected final-test config path. Any explicit final-test config is captured and semantically validated during preflight, frozen with its SHA-256 in the plan, and referenced by the generated script instead of the mutable source path.
- Agents should report when a recipe would violate this.
