# External Test Locking

This contract owns test-access authorization and final-test boundaries. The
[selection contract](hparam_workflow.md#selection-and-selected-candidate-consumers)
owns source ranking, checkpoint choice, and selected-candidate consumers.

## Selection and test access policy

Search selects on the frozen validation `selection_split`, metric, and mode.
The test split is never a selection split: recipe validation rejects
`selection_split: test`. Asking to tune does not unlock test access or
authorize final evaluation.

| Requested behavior | Selection / evaluation split | Post-fit testing | Required access policy |
| --- | --- | --- | --- |
| Validation-selected tuning without post-fit test | `selection_split: val` | `test_after_fit: false` | Test may remain locked. |
| Validation-selected tuning with post-fit test | `selection_split: val` | `test_after_fit: true` | `external_test_locked: false`; results do not change the selection split. |
| Direct `infer` on test | `eval_split: test` | Not a post-fit operation | Both `external_test_locked: false` and `final_test_unlocked: true`. |

For finetune and hparam recipes, omitted `test_after_fit` is materialized as
`evaluation_policy.test_after_fit=true` before consultation, with a
`decisions.test_after_fit` record sourced from `policy_default`. Set it to false
to opt out. Generated commands always render `--test-after-fit` or
`--no-test-after-fit` explicitly; a locked policy never silently changes the
resolved choice. An incompatible lock/test choice must be reported and resolved.

## Final external-test boundary

Final external-test evaluation is a separate, explicitly unlocked command.
Scripts require an explicit existing checkpoint path: unlock never authorizes
checkpoint guessing. Hparam recipes with `yaml:/...` overrides also require an
explicit selected final-test config path. Any explicit final-test config is
captured and semantically validated during preflight, frozen with its SHA-256,
and referenced by the generated script rather than its mutable source path.
Before evaluation starts, `final_external_test.sh` re-hashes the frozen config it
evaluates (`config.final_eval.yaml`, or `config.source.yaml` when no explicit
final-test config is required) against its plan-time SHA-256 and stops on a
mismatch.
If final script generation is skipped, stale `final_external_test.sh` and
frozen final-test config artifacts must be blocked; publish into a fresh
output directory instead of reusing one that holds them.
