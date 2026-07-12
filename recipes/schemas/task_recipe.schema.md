# Task Recipe Schema

Recipes are YAML mappings.

```yaml
name: ppg_ahi_finetune_example
task: finetune
variant: sleep2vec

experiment:
  id: ppg-ahi-finetune
  title: PPG AHI finetuning
  objective: Establish the validation-selected AHI baseline.
  root: artifacts/experiments/ppg-ahi-finetune
  baseline:
    type: none
    rationale: First managed experiment.

step:
  id: train-baseline
  phase: train
  purpose: Train the initial validation-selected baseline.

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
  workdir: null  # when set, must be an absolute path used verbatim as the run cwd
  conda_env: null
  gpu_pool: []
  gpus_per_run: 1
  max_concurrent: 1
  wandb_project: null
  wandb_group: null
  env: {}

adaptive:
  enabled: false
  objective_metric: test_auroc
  objective_mode: max
  test_feedback_for_selection: false
  max_rounds: 3
  max_runs_total: 24
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
    question: "Should the task use ahi, stage5, age, sex, or a custom metadata label?"
```

High-impact fields must not be silently inferred from filenames, nearby configs, or previous runs.

Common top-level fields:

- `name`: stable recipe name.
- `task`: one of `preset_prepare`, `pretrain`, `adapt`, `finetune`, `infer`, `evaluate`, `hparam_tune`, `sleep2stat`.
- `experiment`: required experiment id, title, objective, root, and baseline metadata for runnable plans.
- `step`: required id, phase, and purpose for the current preparation, training, evaluation, or analysis step.
- `variant`: one of `sleep2vec`, `sleep2vec2`, `sleep2expert`, or `sex_age_baseline` for model tasks; omit it or set it to `null` for `task: sleep2stat`.
- `inputs`: paths and task-specific inputs.
- `inputs.eval_split`: explicit split for inference/evaluation; use `ASK_USER` only when the agent must stop.
- `inputs.final_eval_config_path`: selected config for unlocked final external-test evaluation when hparam search uses `yaml:/...` config overrides.
- `runtime`: low-impact runtime knobs and CLI hyperparameters.
- `artifacts`: generated output paths and version names.
- `execution`: optional hparam orchestration settings. Existing recipes may omit this and still generate local scripts only.
- `adaptive`: optional append-only hparam workflow. Existing recipes may omit this and remain static validation-only tuning.
- `evaluation_policy`: split, selection, and external-test locking policy.
- `search`: hyper-parameter tuning method, budget, and parameters. The supported method is `grid`.
- `search.max_runs`: required positive run budget for hparam tuning.
- `search.parameters`: keys must be `runtime.lr`, `runtime.weight_decay`, `runtime.batch_size`, `runtime.epochs`, `runtime.num_workers`, `runtime.precision`, `runtime.gradient_clip_val`, `runtime.accumulate_grad_batches`, `runtime.warmup_steps`, `runtime.patience`, `runtime.check_val_every_n_epoch`, `runtime.ckpt_every_n_epochs`, or `yaml:/json/pointer/path`.
- `yaml:/...`: JSON Pointer-like config overrides used for generated config copies.
- `execution.target`: `local` or `ssh`; `execution.host` is required for `ssh`.
- `execution.path_context`: optional `local` or `remote`; remote absolute paths are not checked with local `Path.exists`.
- `execution.path_validation`: optional `local`, `defer`, or `ssh`; remote defaults to `defer`, and `ssh` uses short `test -e` checks.
- `execution.gpu_pool`: GPU ids used by `agent_tools hparam-launch` for `CUDA_VISIBLE_DEVICES`.
- `execution.gpus_per_run`: number of GPU ids assigned to each managed run.
- `execution.max_concurrent`: maximum runs launched immediately by `hparam-launch --execute`.
- `execution.conda_env`, `execution.wandb_project`, `execution.wandb_group`, and `execution.env`: runtime wrapper settings only; they do not change generated trainer configs. Logs and PID files are always co-located in the managed run directory.
- `adaptive.enabled`: when true, `agent_tools hparam-adaptive-*` commands create `adaptive/` ledgers and per-round plans without modifying old runs.
- `adaptive.max_runs_total`: maximum number of registered runs across adaptive rounds.
- `adaptive.objective_metric`: defaults to `test_auroc` for external-optimized tuning.
- `adaptive.test_feedback_for_selection`: must be true if `objective_metric` starts with `test_` or `external_`.
- `adaptive.replacement.allow_running_stop`: allows stopping manifest-recorded running jobs only when failure/timeout/live-metric evidence says they are bad.
- `adaptive.suggest.strategy`: supports `best_neighborhood`.
- `decisions`: explicit high-impact decision sources.

Sleep2stat recipes use the existing `sleep2stat` CLI and do not use a model variant:

- `task`: must be `sleep2stat`.
- `variant`: must be omitted or `null`; `sleep2stat` is not a supported variant value.
- `inputs.config`: required sleep2stat YAML.
- `inputs.split`: optional CLI split override; when absent, `data.split` from the config is used.
- `runtime.device`, `runtime.num_workers`, `runtime.batch_size`, `runtime.limit_records`, and `runtime.dry_run`: optional `sleep2stat run` CLI knobs.
- `runtime.summarize_after_run`, `runtime.plot_cohort_after_run`, `runtime.plot_group_column`, and `runtime.plot_stage_source`: optional post-run command rendering controls; summarize and plot commands are skipped for `runtime.dry_run=true`; `plot_stage_source`, when present, is passed through to the CLI, so use a concrete analyzer name for a successful plot.
- `artifacts.run_dir`: optional, but if present it must exactly match config `run.output_dir`.
- `evaluation_policy.external_test_locked`: must be explicitly `true` when the effective split includes `test`.
- `decisions.sleep2stat_split_policy`, `decisions.sleep2stat_metric_use_policy`, and `decisions.overwrite_policy`: explicit high-impact decisions.

The referenced sleep2stat YAML must pass `sleep2stat.config.load_config()` directly. Agent tools do not infer or translate sleep2stat config fields; required data fields, AHI postprocess fields, SpO2 desaturation fields, and YASA bandpower output modes belong in the YAML.
