# Skill: finetuning

## When to use
Use for `finetune` tasks that train downstream heads.

## Required inputs
Requires config YAML, `--label-name`, data backend inputs, pretrained-backbone policy, monitor metric/mode, and result path. When `test_after_fit` is omitted, agent tools materialize `evaluation_policy.test_after_fit=true` and a `decisions.test_after_fit` record sourced from `policy_default` before consultation; set `test_after_fit: false` when the run must skip test evaluation.

## First information-gathering commands
- `python -m agent_tools config-summary --config <config> --json`
- `python -m agent_tools doctor --recipe <recipe>`
- `python utils/check_configs.py <config>`

## Decision checklist
Confirm `data.finetune_preset_path`, `data.finetune_data_index`, `data.backend`, task monitor, results CSV, and any explicit `--no-test-after-fit` opt-out.

On `sleep2vec`, `sleep2vec2` and `sleep2expert`, also confirm `pretrained_backbone_path` and the `finetune.tuning` block that decides what trains (its `preset` and any `groups` overrides, LoRA included). A `sleep2vec2` channel with `model.channels[].init_from` takes its tokenizer and `mask_embed` weights from the differently named pretrained-checkpoint channel it names; `alias` is the data-side npz key, not a weight mapping. `sex_age_baseline` has neither: it trains its own model and its loader never reads `finetune.tuning`, so asking for a tuning policy there sends the agent hunting for a block no config in that variant carries.

## Stop-and-consult gates
The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

Stop and consult the user if:

- `label_name` is missing.
- The config contains several plausible labels.
- `pretrained_backbone_path` is absent on a `sleep2vec`, `sleep2vec2` or `sleep2expert` recipe and neither scratch training nor supported full-model initialization is explicitly intended.
- Any `model.channels[].init_from` is set while `pretrained_backbone_path` is absent and neither `init_checkpoint_path` nor a resume `ckpt_path` is given; doctor and plan fail consultation on `pretrained_backbone_path` for that recipe instead of emitting a command that `validate_init_from_sources` rejects at launch. For hparam recipes `ckpt_path` does not count (it is final-evaluation-only), and search overrides that add `init_from` are checked per candidate.
- `finetune.task.monitor` is missing or inconsistent with the label.
- The data backend is unclear.
- The config points to both index and preset inputs without a clear priority.
- `external_test_locked=true` conflicts with the default test-after-fit behavior and `test_after_fit: false` was not explicitly chosen.

## Canonical commands
Use the recipe `variant` to choose the module: `python -m sleep2vec.finetune`, `python -m sleep2vec2.finetune`, `python -m sleep2expert.finetune`, or `python -m sex_age_baseline.finetune`. `sex_age_baseline.finetune` takes the same options as `sleep2vec.finetune`, including LR schedulers and W&B routing, but fails at launch when `--pretrained-backbone-path` or `--print-diagnostics` is set; `--diagnostics-steps` is inert without `--print-diagnostics`, as in `sleep2vec`.

## Expected artifacts
`log-finetune/<version>/checkpoints/`, stable `best.ckpt`, run manifest, copied config/CLI snapshots, optional results CSV.

## Validation gates
For `sleep2vec2` continuation from a selected full finetune checkpoint, use
`inputs.init_checkpoint_path` / `--init-checkpoint-path` for weights-only
initialization, not `pretrained_backbone_path` or trainer resume. Follow the
[full-model initialization contract](../../doc/agent_contracts/hparam_workflow.md#full-model-weights-only-initialization)
for semantic compatibility, local publication, hash binding and fresh trainer state.

Run the selected variant's `finetune --help`, config checks, and targeted runtime/result tests.

## Common failure modes
Missing label, missing preset/index, wrong backend, monitor mismatch, checkpoint
path errors, and attempting test-based checkpoint selection in a direct finetune
plan. Explicitly authorized test-selected work instead uses the
[hparam route, including for one fixed configuration](../../doc/agent_contracts/external_test_locking.md#test-selected-runtime-requirements);
this does not grant test access or permit direct finetune to select on test.

## Relevant owners and index pages
Owners: `runtime-orchestrator`, `config-task-contract`, `model-integration`, `agent-tooling-maintainer`. Index: finetune workflow.
