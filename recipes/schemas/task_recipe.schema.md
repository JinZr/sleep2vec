# Task Recipe Schema

Recipes are YAML mappings.

```yaml
schema_version: 1
name: ppg_ahi_finetune_example
task: finetune
variant: sleep2vec

inputs:
  config: configs/ppg_ahi_finetune_large.yaml
  label_name: ahi
  pretrained_backbone_path: null
  ckpt_path: null
  final_eval_config_path: null

runtime:
  devices: [0]
  accelerator: gpu
  precision: bf16-mixed
  epochs: 30
  batch_size: 12
  num_workers: 8
  lr: 1.0e-6
  weight_decay: 1.0e-5
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  wandb_mode: offline

artifacts:
  version_name: ppg-ahi-agent-example
  results_csv_path: results/ppg_ahi_agent_example.csv

execution:
  target: local
  host: null
  workdir: null
  conda_env: null
  gpu_pool: []
  max_concurrent: 1
  wandb_project: null
  wandb_group: null
  log_dir: logs
  pid_dir: pids
  env: {}

adaptive:
  enabled: false
  objective_metric: test_auroc
  objective_mode: max
  test_feedback_for_selection: false
  max_rounds: 3
  max_trials_total: 24
  round_size: 3
  poll_seconds: 60
  replacement:
    enabled: true
    allow_running_stop: true
    grace_epochs: 1
    grace_minutes: 10
    kill_margin: 0.05
  suggest:
    strategy: best_neighborhood

evaluation_policy:
  selection_split: val
  final_eval_split: test
  external_test_locked: true
  test_after_fit: false
  require_manual_unlock_for_final_test: true

decisions:
  task:
    value: finetune
    source: explicit_recipe
  label_name:
    value: ahi
    source: explicit_recipe
  pretrained_backbone_path:
    value: null
    source: explicit_recipe
    meaning: "train downstream model without loading a pretrained backbone"
  external_test_locked:
    value: true
    source: explicit_recipe
  overwrite_policy:
    value: false
    source: explicit_recipe
```

Use `ASK_USER` when a recipe author intentionally wants the agent to stop and ask the user before generating commands.

```yaml
decisions:
  label_name:
    value: ASK_USER
    source: unresolved
    question: "Should the task use ahi, stage5, age, sex, or src_isDep?"
```

High-impact fields must not be silently inferred from filenames, nearby configs, or previous runs.

Common top-level fields:

- `schema_version`: must be `1`.
- `name`: stable recipe name.
- `task`: one of `preset_prepare`, `pretrain`, `adapt`, `finetune`, `infer`, `evaluate`, `hparam_tune`.
- `variant`: one of `sleep2vec`, `sleep2vec2`, or `sleep2expert`; generated commands use the matching package namespace.
- `inputs`: paths and task-specific inputs.
- `inputs.eval_split`: explicit split for inference/evaluation; use `ASK_USER` only when the agent must stop.
- `inputs.final_eval_config_path`: selected config for unlocked final external-test evaluation when hparam search uses `yaml:/...` config overrides.
- `runtime`: low-impact runtime knobs and CLI hyperparameters.
- `artifacts`: generated output paths and version names.
- `execution`: optional hparam orchestration settings. Existing recipes may omit this and still generate local scripts only.
- `adaptive`: optional append-only hparam workflow. Existing recipes may omit this and remain static validation-only tuning.
- `evaluation_policy`: split, selection, and external-test locking policy.
- `search`: hyper-parameter tuning method, budget, and parameters. V1 supports `method: grid` only.
- `search.parameters`: keys must be `runtime.lr`, `runtime.weight_decay`, `runtime.batch_size`, `runtime.epochs`, `runtime.num_workers`, `runtime.precision`, `runtime.gradient_clip_val`, `runtime.accumulate_grad_batches`, `runtime.warmup_steps`, `runtime.patience`, `runtime.check_val_every_n_epoch`, `runtime.ckpt_every_n_epochs`, or `yaml:/json/pointer/path`.
- `yaml:/...`: JSON Pointer-like config overrides used for generated config copies.
- `execution.target`: `local` or `ssh`; `execution.host` is required for `ssh`.
- `execution.gpu_pool`: GPU ids used by `agent_tools hparam-launch` for `CUDA_VISIBLE_DEVICES`.
- `execution.max_concurrent`: maximum trials launched immediately by `hparam-launch --execute`.
- `execution.conda_env`, `execution.wandb_project`, `execution.wandb_group`, `execution.log_dir`, `execution.pid_dir`, and `execution.env`: runtime wrapper settings only; they do not change generated trainer configs.
- `adaptive.enabled`: when true, `agent_tools hparam-adaptive-*` commands create `adaptive/` ledgers and per-round plans without modifying old trials.
- `adaptive.objective_metric`: defaults to `test_auroc` for external-optimized tuning.
- `adaptive.test_feedback_for_selection`: must be true if `objective_metric` starts with `test_` or `external_`.
- `adaptive.replacement.allow_running_stop`: allows stopping manifest-recorded running jobs only when failure/timeout/live-metric evidence says they are bad.
- `adaptive.suggest.strategy`: v1 supports `best_neighborhood`.
- `decisions`: explicit high-impact decision sources.
