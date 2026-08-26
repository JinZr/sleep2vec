# Task Recipe Schema

Recipes are YAML mappings. This minimal skeleton shows the major sections; it is not a complete runnable recipe.

```yaml
name: unit_hparam
task: hparam_tune
variant: sleep2vec
experiment: {id: unit-experiment, title: Unit experiment, objective: Exercise tuning, root: artifacts/experiments/unit, baseline: {type: none, rationale: First run.}}
step: {id: tune, phase: train, purpose: Select a configuration on the frozen validation metric.}
inputs:
  config: configs/example.yaml
  label_name: ahi
evaluation_policy:
  selection_metric: val_ahi_pearson
  selection_split: val
  external_test_locked: true
  test_after_fit: false
  final_test_unlocked: false
search:
  profile: finetune_balanced
```

See [`recipes/examples/tiny_fixture_hparam.yaml`](../examples/tiny_fixture_hparam.yaml) for a complete runnable example.

Use `ASK_USER` when a recipe author intentionally wants the agent to stop and ask the user before generating commands.

```yaml
decisions:
  label_name:
    value: ASK_USER
    source: unresolved
    question: "Should the task use ahi, stage5, age, sex, or a custom metadata label?"
```

High-impact fields must not be silently inferred from filenames, nearby configs, or previous runs.

## Closed authored boundary

Authored recipes are closed mappings. Unknown fields, task-inapplicable fields, non-mapping sections, and authored top-level names beginning with `_` fail before config loading or experiment-workspace creation. Internal `_recipe_path`, `_base_recipe`, and `_local_recipe` values are added only by the trusted loader and may appear in frozen plan artifacts; they are not valid authored YAML fields.

For `hparam_tune`, the referenced base file is checked as a `finetune` recipe, the tuning file is checked as a hparam overlay, and the merged recipe is then checked semantically. An unknown base field cannot be hidden by replacing its section in the local overlay. Failure evidence identifies `base`, `local`, `effective`, or `user` as the source layer.

Every task accepts `name`, `task`, `variant`, `experiment`, `step`, and `decisions`. Its only additional top-level fields are:

| task | additional top-level fields |
| --- | --- |
| `preset_prepare` | `execution`, `inputs`, `preset` |
| `finetune`, `infer`, `evaluate` | `artifacts`, `evaluation_policy`, `execution`, `inputs`, `runtime` |
| `embedding_extraction` | `artifacts`, `evaluation_policy`, `extraction`, `inputs`, `runtime` |
| `hparam_tune` | `adaptive`, `artifacts`, `base_recipe`, `evaluation_policy`, `execution`, `inputs`, `runtime`, `search` |
| `sleep2stat` | `artifacts`, `evaluation_policy`, `execution`, `inputs`, `runtime` |

The closed section fields are:

- `experiment`: `id`, `title`, `objective`, `root`, `baseline`. `baseline` remains an opaque description rather than a deeply enumerated schema.
- Recipe `step`: `id`, `phase`, `purpose`. The separate experiment step-registration command also owns `inputs` and `outputs`; those are not recipe-step fields.
- `inputs`:
  - `preset_prepare`: `config`, `dataset_name`, `index`.
  - `finetune`: `ckpt_path`, `config`, `data_backend`, `label_name`, `pretrained_backbone_path`.
  - `infer` / `evaluate`: `ckpt_path`, `config`, `data_backend`, `eval_split`, `inference_preset_path`, `label_name`, `override_dataset_names`, `pretrained_backbone_path`.
  - `hparam_tune`: `ckpt_path`, `config`, `data_backend`, `final_eval_config_path`, `inference_preset_path`, `label_name`, `override_dataset_names`, `pretrained_backbone_path`. For hparam recipes, `ckpt_path` selects the final-evaluation checkpoint and is not passed to tuning runs.
  - `sleep2stat`: `config`, `split`.
  - `embedding_extraction`: `config`, `ckpt_path`, `data_index`, `eval_split`.
- `runtime`:
  - `finetune`: `accumulate_grad_batches`, `batch_size`, `check_val_every_n_epoch`, `ckpt_every_n_epochs`, `device`, `devices`, `epochs`, `gradient_clip_val`, `lr`, `num_workers`, `patience`, `precision`, `warmup_steps`, `wandb_mode`, `weight_decay`. `sex_age_baseline` omits `wandb_mode` because its renderer does not consume it.
  - `infer` / `evaluate`: `accelerator`, `avg_ckpt_dir`, `avg_ckpts`, `batch_size`, `device`, `devices`, `lr`, `num_workers`, `precision`, `results_root`, `seed`, `wandb_mode`, `weight_decay`.
  - `hparam_tune`: the explicit union of finetune and inference runtime fields, because managed runs use the former and final evaluation uses the latter.
  - `sleep2stat`: `batch_size`, `device`, `dry_run`, `limit_records`, `num_workers`, `plot_adjust_covariates`, `plot_cohort_after_run`, `plot_group_column`, `plot_stage_source`, `summarize_after_run`.
  - `embedding_extraction`: `device`, `num_workers`.
- `infer` / `evaluate` checkpoint averaging rejects AHI and `sex_age_baseline`; `avg_ckpts` is a positive integer and `ckpt_path=best/last` requires `avg_ckpt_dir`, whose explicit directory is resolved from the runtime workdir.
- Checkpoint, pretrained-backbone, index, manifest, preset, and sidecar inputs are files. Kaldi finetune/inference additionally requires `kaldi_data_root` as a directory and rejects effective NPZ preset paths. NPZ runtime presets suppress sidecar reopening, while `preset_prepare` requires survival or multilabel sidecars to build them. Runtime-semantic input paths reject leading `~`; use an absolute or workdir-relative path.
- `artifacts`:
  - `finetune`: `overwrite`, `results_csv_path`, `version_name`.
  - `infer` / `evaluate`: `overwrite`.
  - `hparam_tune`: `overwrite`, `results_csv_path`.
  - `sleep2stat`: `overwrite`, `run_dir`.
  - `embedding_extraction`: `embedding_dir`, `overwrite`.
- `evaluation_policy`:
  - `finetune`: `external_test_locked`, `selection_metric`, `selection_mode`, `selection_split`, `test_after_fit`.
  - `infer` / `evaluate`: `external_test_locked`, `final_test_unlocked`.
  - `hparam_tune`: `external_test_locked`, `final_eval_split`, `final_test_unlocked`, `require_manual_unlock_for_final_test`, `selection_metric`, `selection_mode`, `selection_split`, `test_after_fit`.
  - `sleep2stat`: `external_test_locked`.
  - `embedding_extraction`: `external_test_locked`, `final_test_unlocked`.
  - For `finetune` and `hparam_tune`, omitted `test_after_fit` is materialized as `true` with source `policy_default` before consultation and frozen in the resolved recipe. Generated commands always render `--test-after-fit` or `--no-test-after-fit`; set `test_after_fit: false` explicitly to opt out.
  - Hparam selection uses the explicitly frozen `selection_split`. Test-selected tuning requires `selection_split: test`, `external_test_locked: false`, `test_after_fit: true`, effective positive-integer `runtime.epochs`, and effective `runtime.ckpt_every_n_epochs: 1` for every trial; its `selection_metric` may be a `test_*` metric distinct from the validation checkpoint monitor in `finetune.task.monitor`. The positive epoch budget and checkpoint interval guarantee at least one immutable epoch-checkpoint opportunity even when early stopping occurs before a wider interval. The planner derives `--test-all-checkpoints-after-fit` for this policy; it is not a second recipe field. Every regular non-alias saved `epoch=*.ckpt` is evaluated, and selection requires its finite checkpoint-level test metric.
  - Concrete `external_test_locked` values must be YAML booleans; strings and numbers are not coerced.
- Non-hparam `execution`: `embedding_extraction` rejects the entire block and uses the current local checkout. Every other non-hparam task accepts `host`, `path_context`, `path_validation`, `target`, and an absolute `workdir` used for cwd/PYTHONPATH. Relative runtime-semantic paths are validated from that workdir, or `REPO_ROOT` when it is omitted; local relative `inputs.config` remains a planning-source locator under `REPO_ROOT`. Only `infer` / `evaluate` additionally accept `python` and `runtime_commit`; declaring either creates an all-or-none local runtime identity with `workdir`, where `python` is one executable name or path without whitespace, arguments, or `~` shorthand and `runtime_commit` is a lowercase 40-character Git commit SHA. Other non-hparam tasks reject these two inert identity fields. `experiment-run` supplies the complete identity for every managed attempt.
- Hparam `execution`: `conda_env`, `env`, `gpu_pool`, `gpus_per_run`, `host`, `max_concurrent`, `path_context`, `path_validation`, `python`, `runtime_commit`, `scheduler`, `target`, `wandb_group`, `wandb_project`, `workdir`. `python` is one target executable name or path without whitespace, arguments, or `~` shorthand and `runtime_commit` is a full Git hash; Conda wrapping belongs in `conda_env`. They may be omitted only for a local target at `REPO_ROOT` without `conda_env`; planning then freezes the current manager interpreter and manager repository HEAD. SSH targets, separate local workdirs, and conda-wrapped targets require both fields explicitly. `env` has dynamic environment-variable names but may not duplicate `PYTHONPATH` or the explicit W&B fields. Local relative `final_eval_config_path` remains a planning-source locator under `REPO_ROOT` before it is frozen.
- Hparam `execution.scheduler` is optional in authored recipes and is materialized as `{type: direct}` in resolved plans. `type: slurm` requires `partition`, positive `cpus_per_task`, Slurm `memory` such as `64G`, and `walltime` in `HH:MM:SS` or `D-HH:MM:SS` form; `nice`, `nodelist`, and boolean `direct_controller` are optional. `direct_controller` defaults to false, which routes follow-up commands to the bound cluster with `--clusters`; set it to true only when the submission endpoint already talks directly to that controller and federation routing is unavailable. Slurm submits one frozen single-node leaf job per run. `gpus_per_run` must be a positive YAML integer when authored, defaults to one, derives logical `runtime.devices` as `[0, ..., N-1]`, and launches exactly N `srun` tasks against N allocated GPUs without explicit task-level GPU binding. `cpus_per_task` applies to every task, so a run requests `N * cpus_per_task` CPUs; `memory` is the whole-node allocation limit. Slurm `sex_age_baseline` recipes reject `gpus_per_run > 1`. Slurm recipes also reject `gpu_pool`, `max_concurrent`, `conda_env`, hparam-overlay `runtime.devices`, distributed-launcher variables (`SLURM_*`, `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_*`), `CUDA_VISIBLE_DEVICES` entries in `execution.env`, unknown scheduler keys, and arbitrary sbatch arguments. `target` remains only the local/SSH control transport and does not imply controller topology. Schema-v1 external-evaluation pipelines remain direct-only.
- `extraction` for `embedding_extraction`: non-empty unique `channels`, `embedding_kind: both`, `layer_index: -1`, integer `max_source_tokens` in `[1, 4095]`, `output_format: npz`, and `sequence_mode: whole-night`.
- `preset`: `allow_missing_channels`, `batch_size`, `channels`, `dry_run`, `include_no_metadata`, `include_overlap_eval_splits`, `manifest_output`, `mask_rate`, `meta_data_names`, `min_channels`, `n_tokens`, `num_workers`, `output_template`, `overwrite`, `shuffle`, `split`, `stride_tokens`, `write_sidecar_manifest`.
- `search`: `profile`, `method`, `max_runs`, `parameters`, `configurations`. An explicit search uses `method: grid`, a positive `max_runs`, and exactly one of `parameters` (a per-key candidate mapping expanded by Cartesian product) or `configurations` (a list of complete joint points expanded verbatim). `profile: finetune_balanced` is instead an authored generation intent for supported `sleep2vec` or `sleep2vec2` finetuning labels `ahi`, `arousal`, and `stage4`; it is mutually exclusive with authored `parameters` and `configurations`. Before consultation, the task adapter deterministically materializes `method: grid`, exact joint `configurations`, and a default `max_runs: 12`; an authored override must be between 4 and 32 and large enough to cover every profile level. The expansion searches bounded technical levels for learning rate, weight decay, the atomic LayerMix block, synchronized supported dropout fields, atomic full/head-only/LoRA adaptation arms only when `inputs.pretrained_backbone_path` is non-empty, and positive scalar `pos_weight` when present; it freezes batch size, epochs, patience, aggregation, EMA, pretrained checkpoint, channels, and class weights. Hparam `inputs.ckpt_path` is final-evaluation-only and does not enable frozen-backbone tuning arms. The first profile version rejects adaptive tuning. Materialized configurations are executable frozen expansion, not a second authored truth source, and selection means only the best observed candidate within the frozen search domain, metric, split, and budget. Explicit parameter names are `runtime.lr`, `runtime.weight_decay`, `runtime.batch_size`, `runtime.epochs`, `runtime.num_workers`, `runtime.precision`, `runtime.gradient_clip_val`, `runtime.accumulate_grad_batches`, `runtime.warmup_steps`, `runtime.patience`, `runtime.check_val_every_n_epoch`, `runtime.ckpt_every_n_epochs`, or `yaml:/json/pointer/path`; configuration-point keys follow the same rules. Adaptive source recipes must use explicit `parameters`.
- `adaptive`: `enabled`, `max_rounds`, `max_runs_total`, `objective_metric`, `objective_mode`, `poll_seconds`, `replacement`, `round_size`, `suggest`, `test_feedback_for_selection`. `replacement` accepts `allow_running_stop`, `enabled`, `grace_epochs`, `grace_minutes`, `kill_margin`; `suggest` accepts `strategy` and `bounds`. Adaptive control flags must be YAML booleans, run/round/poll budgets must be positive YAML integers, and replacement grace/margin values must be finite non-negative numbers. `strategy` is `agent_proposal` (the default when omitted) or `best_neighborhood`; the latter must be explicit. `bounds` is valid only for `agent_proposal`; its keys are a subset of `search.parameters`, and each value is a closed two-number interval for that numeric parameter. In an enabled `agent_proposal` workflow, `objective_metric` is conditionally required as an explicit non-blank string, while `objective_mode`, `round_size`, `max_rounds`, and `max_runs_total` require explicit non-empty values. Omission, null, or blank required values stop consultation before workspace mutation; a non-string objective metric fails the recipe contract. For `selection_split: test`, every adaptive `test_*` objective uses the finite complete checkpoint-level result selected by `objective_mode`, with its exact checkpoint path and epoch, even when that objective differs from the frozen selection metric. Validation/run-level objectives such as `val_*` and `best_model_score` retain top-level evidence; incomplete checkpoint evidence cannot trigger metric-based replacement. `replacement` must be omitted or exactly `{enabled: false}` because proposals are generated only after the current round is terminal. A disabled adaptive block does not start this protocol.
- `decisions`: names must be applicable to the current task in `agent_policies/consultation_policy.yaml` or an owner-local optional decision. A mapping entry accepts only `meaning`, `question`, `rationale`, `source`, and `value`; scalar shorthand remains valid.

Removed names such as `runtime.data_backend`, `inputs.preset_path`, `preset.regenerate`, `search.max_trials`, `execution.gpus_per_trial`, and adaptive `max_trials_total` are rejected rather than translated. Keep regeneration intent in `decisions.preset_regeneration`; `preset.overwrite` is the actual `--overwrite` behavior. Use `inputs.data_backend`, `inputs.inference_preset_path`, `search.max_runs`, `execution.gpus_per_run`, and `adaptive.max_runs_total` for the other replacements.

The supported task values are `preset_prepare`, `finetune`, `infer`, `evaluate`, `embedding_extraction`, `hparam_tune`, and `sleep2stat`. Embedding extraction supports `sleep2vec`, `sleep2vec2`, and `sleep2expert`; it requires an explicit NPZ index and a RoFormer config with `model.cls.embedding_type: bert`, rejects `data_backend`, preset/Kaldi inputs, dataset source overrides, `runtime.batch_size`, and `execution`, and requires any config-owned dataset-source filter to retain every selected index row. A missing or blank row `source` uses the authored index path. Its fresh absolute `embedding_dir` must not overlap its plan or occupy experiment-managed `plans`, `reports`, or `steps` namespaces. Other model tasks use `sleep2vec`, `sleep2vec2`, `sleep2expert`, or `sex_age_baseline` as applicable; `sleep2stat` omits `variant` or sets it to `null`.

Hparam `execution.target` is `local` or `ssh`; `host` is required for SSH. `workdir`, GPU allocation, W&B settings, and optional environment/conda wrapping are consumed by the managed launcher. Adaptive test/external objectives require explicit test-feedback authorization.

`pretrain` and `adapt` are not runnable task-recipe values because agent tools do not render those commands. Use the corresponding skill and direct variant runtime CLI instead.

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
