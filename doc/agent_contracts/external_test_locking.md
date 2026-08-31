# External Test Locking

This contract owns test-access authorization and final-test boundaries. The
[selection contract](task_recipe.md#selection-and-selected-candidate-consumers)
owns ranking, checkpoint choice, and selected-candidate consumers; the
[pipeline contract](experiment_pipeline.md) owns external matrices and retries.

## Selection and test access policy

A template's validation default is not a prohibition on explicitly authorized
test-selected tuning. Search selects on the frozen `selection_split`, metric,
and mode. Choosing an automatic technical search profile or asking to tune does
not select a scientific split, unlock test access, or authorize final evaluation.

| Requested behavior | Selection / evaluation split | Post-fit testing | Required access policy |
| --- | --- | --- | --- |
| Validation-selected tuning without post-fit test | `selection_split: val` | `test_after_fit: false` | Test may remain locked. |
| Validation-selected tuning with post-fit test | `selection_split: val` | `test_after_fit: true` | `external_test_locked: false`; results do not change the selection split. |
| Test-selected hparam tuning | `selection_split: test` | `test_after_fit: true`, with complete all-checkpoint evidence | `external_test_locked: false` explicitly authorizes tuning access; final external evaluation remains separate. |
| Direct `infer` / `evaluate` on test | `eval_split: test` | Not a post-fit operation | Both `external_test_locked: false` and `final_test_unlocked: true`. |
| Managed final external-test matrix | Frozen pipeline evaluation policy | Separate explicit operation | `experiment-run --unlock-final-test`; source selection is not rewritten. |

For finetune and hparam recipes, omitted `test_after_fit` is materialized as
`evaluation_policy.test_after_fit=true` before consultation, with a
`decisions.test_after_fit` record sourced from `policy_default`. Set it to false
to opt out. Generated commands always render `--test-after-fit` or
`--no-test-after-fit` explicitly; a locked policy never silently changes the
resolved choice. An incompatible lock/test choice must be reported and resolved.

## Test-selected runtime requirements

Test-selected hparam tuning additionally requires effective positive-integer
`runtime.epochs` and effective `runtime.ckpt_every_n_epochs=1` for every run.
AHI/arousal runs require `runtime.check_val_every_n_epoch=1`: their epoch
checkpoints are saved after validation to retain validation-fitted thresholds.
The planner renders `--test-all-checkpoints-after-fit`; after fitting, the run
evaluates every regular non-alias `epoch=*.ckpt` in its frozen checkpoint
directory and commits complete `checkpoint_test_results` to the terminal
runtime manifest. Its [evidence shape](run_manifest.md#runtime-artifact-evidence)
and [selection algorithm](task_recipe.md#selection-and-selected-candidate-consumers)
remain separately owned.

Direct `finetune` plans cannot use `selection_split=test` because they do not
own all-checkpoint test ranking. Represent a fixed configuration as
`task=hparam_tune` with one configuration and `max_runs: 1`.
Adaptive test/external feedback also needs the explicit authorization in
[adaptive strategy](task_recipe.md#strategy-and-budget); tuning access alone
does not select an adaptive protocol.

## Final external-test boundary

Final external-test evaluation is a separate, explicitly unlocked command.
Scripts require an explicit existing checkpoint path: unlock never authorizes
checkpoint guessing. Hparam recipes with `yaml:/...` overrides also require an
explicit selected final-test config path. Any explicit final-test config is
captured and semantically validated during preflight, frozen with its SHA-256,
and referenced by the generated script rather than its mutable source path.
If final script generation is skipped, stale `final_external_test.sh` and
frozen final-test config artifacts must be blocked or removed under explicit
overwrite approval.

`experiment-run` derives and freezes checkpoints from source plans' registered
ranking before its explicitly unlocked matrix launch. Metrics from that matrix
never rewrite selection. The pipeline runner is a launcher; `hparam-monitor`
and `experiment-monitor` never start pending external jobs. Multi-source scope,
attempt isolation, retries, result validation, and finalization order belong to
[experiment_pipeline.md](experiment_pipeline.md), not an inferred retry after
missing output or an SSH disconnect.
