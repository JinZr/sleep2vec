# External Test Locking

- Hyper-parameter search selects on its explicitly frozen `selection_split`; `test` is supported when the test set is unlocked for tuning.
- For finetune and hparam recipes, an omitted `test_after_fit` is materialized as `evaluation_policy.test_after_fit=true` with a `decisions.test_after_fit` record sourced from `policy_default` before consultation. The resolved recipe and generated command therefore make the default auditable.
- Set `test_after_fit=false` to opt out; agent-generated commands always render `--test-after-fit` or `--no-test-after-fit` explicitly. A locked policy never silently changes the resolved choice.
- Test-selected hyper-parameter tuning requires `selection_split=test`, `external_test_locked=false`, and `test_after_fit=true`. The hparam planner also renders `--test-all-checkpoints-after-fit`: after fitting, each trial evaluates every regular non-alias `epoch=*.ckpt` in its frozen checkpoint directory and commits `checkpoint_test_results` to its terminal runtime manifest.
- `hparam-select` requires complete finite checkpoint-level evidence for the frozen `test_*` metric, ranks every saved checkpoint globally in `checkpoint_test_ranking.csv`, reduces the existing workspace ranking to the best checkpoint per run, and records the exact globally selected checkpoint path and SHA-256.
- Validation-selected tuning remains supported by setting `selection_split=val`; it may opt out of post-fit testing with `test_after_fit=false`.
- Direct `infer` or `evaluate` on `eval_split=test` requires both `external_test_locked=false` and `final_test_unlocked=true`.
- Final external-test evaluation is a separate, explicit command.
- Final external-test scripts require an explicit existing checkpoint path; unlock does not authorize checkpoint guessing.
- `experiment-run` may execute a final external-test matrix only with the explicit `--unlock-final-test` gate. It derives and freezes each checkpoint from the source plan's registered ranking before launch; metrics produced by that separate matrix never rewrite selection.
- The pipeline runner is an explicit launcher. `hparam-monitor` and `experiment-monitor` remain observational and never start pending external jobs.
- If final script generation is skipped, stale `final_external_test.sh` and frozen final-test config artifacts must be blocked or removed under explicit overwrite approval.
- Hparam recipes with `yaml:/...` config overrides also require an explicit selected final-test config path. Any explicit final-test config is captured and semantically validated during preflight, frozen with its SHA-256 in the plan, and referenced by the generated script instead of the mutable source path.
- Agents should report when a recipe would violate this.

The multi-source, multi-job workflow, retry boundary, manifest validation, and finalization order belong to [experiment_pipeline.md](experiment_pipeline.md).
