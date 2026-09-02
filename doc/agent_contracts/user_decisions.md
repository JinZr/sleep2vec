# User Decisions

User-decision files resolve high-impact ambiguity with explicit user intent. The file is a closed mapping with exactly one top-level field, `decisions`. An explicitly supplied file must contain that mapping.

Decision names are task-aware: each name must be applicable to the current task through `agent_policies/consultation_policy.yaml` or an existing owner-local optional decision. Mapping entries accept only `value`, `source`, `meaning`, `question`, and `rationale`; scalar shorthand is also accepted. Unknown names and misspelled entry fields fail before context or plan output is written.

## Generated decision templates

For a pure `NEEDS_USER_INPUT` result, `doctor --output-dir` and a safely
published blocked `plan` write `decisions.yaml` when at least one blocker maps
to a task-valid user decision. The template uses this same schema, preserves
already resolved `explicit_user` entries, and represents each unanswered value
as `ASK_USER` with its existing question. It does not add recommendations,
consequences, or inferred rationale.

An unedited template remains unresolved and is not authorization. Fill only
values the user has actually decided, then pass the file explicitly with
`--user-decisions`. A blocked-plan retry must use a fresh `--output-dir`.
Existing independent `decisions.yaml` files are not overwritten by doctor;
a file that appears during blocked-plan publication fails before canonical registration.
Doctor may reuse a doctor-only output directory only while its existing
`decisions.yaml` contains every currently requested decision. A newly exposed
decision requires a fresh doctor `--output-dir`; the old file is never merged or
overwritten. Any blocked or PASS plan marker also makes the directory plan-owned
and requires a fresh doctor `--output-dir`.

No template is written for PASS/WARN, FAIL or mixed FAIL/NEEDS_USER_INPUT results,
non-decision-only blockers, `plan --validate-only`, unsafe or occupied plan
outputs, or registration preflight failures. `context` remains diagnostic-only
and never emits the file. `questions.json`, `questions.md`, and
`plan.blocked.md` remain explanatory views rather than decision inputs.

## Cross-stage reuse

One authorized decision file may be passed unchanged to later `doctor` calls
and to a fresh `plan` output directory. Each consumer revalidates the entries
against the current task, recipe and consultation policy; reuse never bypasses
validation or permits an occupied plan directory. The resolved plan artifacts
then carry those decisions into launch, monitoring and recovery without a
second approval format.

If later validation exposes another decision, preserve the concrete
`explicit_user` entries and ask only for the new `ASK_USER` delta. A changed
value, incompatible task or expanded scientific scope requires renewed user
intent. A decision file records that intent. Its mere presence does not
authorize publication or launch, and an unedited template does not authorize
final-test access. A concrete user-authorized `final_eval_unlock` value retains
its task-owned final-test semantics.

Publish a newly exposed doctor delta in a fresh output directory. Reuse the
authorized values in the new input file; do not expect doctor to merge a new
field into an earlier `decisions.yaml`.

Concrete values with a task-owned canonical field are materialized into the
effective recipe's existing `inputs`, `evaluation_policy`, `preset`, `search`,
or artifact fields before config inspection and consultation are rerun.
Policy-only choices remain under `decisions` rather than creating inert
canonical fields.

Special cases are explicit:

- Non-preset `required_channels` is checked against the selected config.
- `preset_regeneration` remains decision evidence; `preset.overwrite` controls
  the rendered overwrite flag.
- Hparam `ckpt_path` selects only the final-evaluation checkpoint.
- A user `task` may fill a missing recipe task but cannot replace an explicit
  recipe task. For layered hparam recipes, it is compared with the local
  overlay rather than the base finetune task.
- `train_val_test_policy` must be exactly `val` or `test`; direct finetune
  rejects `test`, and hparam tuning requires explicit test access.
  Descriptive text is not interpreted as a split.

```yaml
decisions:
  label_name:
    value: ahi
    source: explicit_user
    rationale: Use the AHI label for this experiment.

  external_test_locked:
    value: false
    source: explicit_user
    rationale: Allow test data to be used for the explicitly chosen tuning objective.

  test_after_fit:
    value: true
    source: explicit_user
    rationale: Produce test metrics for every saved epoch checkpoint in every tuning trial.

  overwrite_policy:
    value: false
    source: explicit_user
```

Resolution precedence is:

1. explicit user-decision file
2. explicit CLI argument
3. explicit recipe decision
4. explicit recipe field
5. explicit config field
6. task policy default, when defined
7. ambiguous or missing

The current task policy default applies to finetune and hparam
`test_after_fit`: when the field remains absent after authored and user
decisions, agent tools materialize `true` with source `policy_default` in the
effective recipe before consultation and preserve it in resolved artifacts.

An empty or `ASK_USER` config decision remains unresolved. For other fields,
empty values are not materialized, and the field-specific consultation rule
determines whether they block. The one intentional null semantic is
`pretrained_backbone_path: null`, which explicitly selects training without a
pretrained backbone. Concrete materialized values use only the canonical field
and do not create an alternate semantic source.
