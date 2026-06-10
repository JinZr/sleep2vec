# External Test Locking

- Hyper-parameter search selects on validation.
- External test is locked during trials.
- Final external-test evaluation is a separate, explicit command.
- Final external-test scripts require an explicit existing checkpoint path; unlock does not authorize checkpoint guessing.
- Hparam recipes with `yaml:/...` config overrides also require an explicit selected final-test config path.
- Agents should report when a recipe would violate this.
