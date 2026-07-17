# Skill: finetuning

## When to use
Use for `finetune` tasks that train downstream heads.

## Required inputs
Requires config YAML, `--label-name`, data backend inputs, pretrained-backbone policy, monitor metric/mode, test-after-fit policy, and result path.

## First information-gathering commands
- `python -m agent_tools config-summary --config <config> --json`
- `python -m agent_tools doctor --recipe <recipe>`
- `python utils/check_configs.py <config>`

## Decision checklist
Confirm `data.finetune_preset_path`, `data.finetune_data_index`, `data.backend`, `pretrained_backbone_path`, LoRA options, task monitor, results CSV, and `--no-test-after-fit` during model selection.

## Stop-and-consult gates
The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

Stop and consult the user if:

- `label_name` is missing.
- The config contains several plausible labels.
- `pretrained_backbone_path` is absent and the recipe does not explicitly say scratch training is intended.
- `finetune.task.monitor` is missing or inconsistent with the label.
- The data backend is unclear.
- The config points to both index and preset inputs without a clear priority.
- The run would evaluate test data during hyper-parameter tuning.

## Canonical commands
Use the recipe `variant` to choose the module: `python -m sleep2vec.finetune`, `python -m sleep2vec2.finetune`, or `python -m sleep2expert.finetune`.

## Expected artifacts
`log-finetune/<version>/checkpoints/`, stable `best.ckpt`, run manifest, copied config/CLI snapshots, optional results CSV.

## Validation gates
Run the selected variant's `finetune --help`, config checks, and targeted runtime/result tests.

## Common failure modes
Missing label, missing preset/index, wrong backend, monitor mismatch, checkpoint path errors, and accidental test evaluation during tuning.

## Relevant owners and index pages
Owners: `runtime-orchestrator`, `config-task-contract`, `model-integration`, `agent-tooling-maintainer`. Index: finetune workflow.
