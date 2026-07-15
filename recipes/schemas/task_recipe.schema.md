# Task Recipe Schema

Recipes are YAML mappings. This minimal skeleton shows the major sections; it is not a complete runnable recipe.

```yaml
name: unit_hparam
task: hparam_tune
variant: sleep2vec
experiment: {id: unit-experiment, title: Unit experiment, objective: Exercise tuning, root: artifacts/experiments/unit, baseline: {type: none, rationale: First run.}}
step: {id: tune, phase: train, purpose: Select a validation configuration.}
inputs:
  config: configs/example.yaml
  label_name: ahi
evaluation_policy:
  selection_split: val
  external_test_locked: true
search:
  method: grid
  max_runs: 1
  parameters: {runtime.lr: [1.0e-6]}
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
- `runtime`:
  - `finetune`: `accumulate_grad_batches`, `batch_size`, `check_val_every_n_epoch`, `ckpt_every_n_epochs`, `device`, `devices`, `epochs`, `gradient_clip_val`, `lr`, `num_workers`, `patience`, `precision`, `warmup_steps`, `wandb_mode`, `weight_decay`. `sex_age_baseline` omits `wandb_mode` because its renderer does not consume it.
  - `infer` / `evaluate`: `accelerator`, `avg_ckpt_dir`, `avg_ckpts`, `batch_size`, `device`, `devices`, `lr`, `num_workers`, `precision`, `seed`, `wandb_mode`, `weight_decay`.
  - `hparam_tune`: the explicit union of finetune and inference runtime fields, because managed runs use the former and final evaluation uses the latter.
  - `sleep2stat`: `batch_size`, `device`, `dry_run`, `limit_records`, `num_workers`, `plot_adjust_covariates`, `plot_cohort_after_run`, `plot_group_column`, `plot_stage_source`, `summarize_after_run`.
- `artifacts`:
  - `finetune`: `overwrite`, `results_csv_path`, `version_name`.
  - `infer` / `evaluate`: `overwrite`.
  - `hparam_tune`: `overwrite`, `results_csv_path`.
  - `sleep2stat`: `overwrite`, `run_dir`.
- `evaluation_policy`:
  - `finetune`: `external_test_locked`, `selection_metric`, `selection_mode`, `selection_split`, `test_after_fit`.
  - `infer` / `evaluate`: `external_test_locked`, `final_test_unlocked`.
  - `hparam_tune`: `external_test_locked`, `final_eval_split`, `final_test_unlocked`, `require_manual_unlock_for_final_test`, `selection_metric`, `selection_mode`, `selection_split`, `test_after_fit`.
  - `sleep2stat`: `external_test_locked`.
- Non-hparam `execution`: `host`, `path_context`, `path_validation`, `target`.
- Hparam `execution`: `conda_env`, `env`, `gpu_pool`, `gpus_per_run`, `host`, `max_concurrent`, `path_context`, `path_validation`, `python`, `runtime_commit`, `target`, `wandb_group`, `wandb_project`, `workdir`. `python` is a non-empty target command/path and `runtime_commit` is a full Git hash. They may be omitted only for a local target at `REPO_ROOT` without `conda_env`; planning then freezes the current manager interpreter and manager repository HEAD. SSH targets, separate local workdirs, and conda-wrapped targets require both fields explicitly. `env` has dynamic environment-variable names but may not duplicate `PYTHONPATH` or the explicit W&B fields.
- `preset`: `allow_missing_channels`, `batch_size`, `channels`, `dry_run`, `include_no_metadata`, `include_overlap_eval_splits`, `manifest_output`, `mask_rate`, `meta_data_names`, `min_channels`, `n_tokens`, `num_workers`, `output_template`, `overwrite`, `shuffle`, `split`, `stride_tokens`, `write_sidecar_manifest`.
- `search`: `method`, `max_runs`, `parameters`. `method` is `grid`; `max_runs` is a positive run budget. Parameter names are `runtime.lr`, `runtime.weight_decay`, `runtime.batch_size`, `runtime.epochs`, `runtime.num_workers`, `runtime.precision`, `runtime.gradient_clip_val`, `runtime.accumulate_grad_batches`, `runtime.warmup_steps`, `runtime.patience`, `runtime.check_val_every_n_epoch`, `runtime.ckpt_every_n_epochs`, or `yaml:/json/pointer/path`.
- `adaptive`: `enabled`, `max_rounds`, `max_runs_total`, `objective_metric`, `objective_mode`, `poll_seconds`, `replacement`, `round_size`, `suggest`, `test_feedback_for_selection`. `replacement` accepts `allow_running_stop`, `enabled`, `grace_epochs`, `grace_minutes`, `kill_margin`; `suggest` accepts only `strategy`.
- `decisions`: names must be applicable to the current task in `agent_policies/consultation_policy.yaml` or an owner-local optional decision. A mapping entry accepts only `meaning`, `question`, `rationale`, `source`, and `value`; scalar shorthand remains valid.

Removed names such as `runtime.data_backend`, `inputs.preset_path`, `preset.regenerate`, `search.max_trials`, `execution.gpus_per_trial`, and adaptive `max_trials_total` are rejected rather than translated. Keep regeneration intent in `decisions.preset_regeneration`; `preset.overwrite` is the actual `--overwrite` behavior. Use `inputs.data_backend`, `inputs.inference_preset_path`, `search.max_runs`, `execution.gpus_per_run`, and `adaptive.max_runs_total` for the other replacements.

The supported task values are `preset_prepare`, `finetune`, `infer`, `evaluate`, `hparam_tune`, and `sleep2stat`. Model tasks use `sleep2vec`, `sleep2vec2`, `sleep2expert`, or `sex_age_baseline` as applicable; `sleep2stat` omits `variant` or sets it to `null`.

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
