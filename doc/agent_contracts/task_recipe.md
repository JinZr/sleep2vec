# Task Recipe Contract

Task recipes live under `recipes/`. They describe their owning experiment and step, task inputs, runtime knobs, artifacts, evaluation policy, search space, and explicit decisions.

Every new runnable recipe must define:

```yaml
experiment:
  id: ukb-cox-router-freeze
  title: UKB Cox router-freeze search
  objective: Determine whether freezing the router improves validation C-index.
  root: /wujidata/<user>/sleep2vec/experiments/ukb-cox-router-freeze
  baseline:
    run_id: run-000
    rationale: Current production configuration.

step:
  id: tune-router-freeze
  phase: train
  purpose: Compare learning rate, router freezing, and class weighting.
```

`phase` is one of `prepare`, `train`, `evaluate`, or `analyze`. The generated plan directory must be inside `experiment.root`. Missing experiment or step metadata blocks runnable plan generation. A hparam recipe must declare its own complete experiment and step; ownership inherited from its base finetune recipe is not accepted.

For local plans, a relative recipe `experiment.root` is based at the repository root, then user-home expansion and absolute path resolution produce the value persisted in the resolved recipe, plan, and workspace manifests. Experiment CLI paths use the caller's current working directory before the same canonicalization. SSH experiment roots remain exact remote strings and are not resolved locally.

Repository-owned management locators are frozen as absolute local paths at the public entry boundary. This includes the source recipe and plan directory, managed run/report directories, frozen config/script/artifacts files, derived runtime/checkpoint directories, postprocessing config/logits outputs, replay-script plan and candidate-table arguments plus the repository cwd/PYTHONPATH bootstrap, adaptive workflow/round/recipe locations, registry copies, and matching event locators. The rule makes later commands independent of their current working directory; it does not normalize user-authored dataset, external-checkpoint, or other semantic input paths. SSH locators remain exact remote strings.

Runnable task and variant routing is a finite contract:

| task | accepted variant | generated runtime |
|---|---|---|
| `sleep2stat` | omitted or `null` | `python -m sleep2stat` |
| `preset_prepare` | `sleep2vec`, `sleep2vec2`, `sleep2expert` | `preprocess/save_dataset_presets.py` |
| `finetune`, `hparam_tune` | `sleep2vec`, `sleep2vec2`, `sleep2expert`, `sex_age_baseline` | `<variant>.finetune` |
| `infer`, `evaluate` | `sleep2vec`, `sleep2vec2`, `sleep2expert`, `sex_age_baseline` | `<variant>.infer` |

`sex_age_baseline` is a standalone runtime variant but does not own preset generation. Missing, extra, or unsupported variant values block command generation rather than falling back to another namespace.

Hyper-parameter search keys must be explicit: use `runtime.<name>` for supported CLI/runtime knobs or `yaml:/json/pointer/path` for generated config overrides. Those keys keep the same spelling in all current managed run artifacts; prefixed `param.*` columns are removed-format data. `search.max_runs` is the required positive run budget.

Hparam recipes may add an optional `execution:` block for active orchestration after `agent_tools plan` has generated run scripts. `execution.target` is `local` or `ssh`; `execution.host` is required for SSH. A non-empty `execution.workdir` must be an absolute path and is used verbatim as the generated script working directory and PYTHONPATH root; when omitted, the repository root is used. Frozen `runtime_dir` and `checkpoint_dir` are derived from that same working directory. `execution.path_context=remote` and `execution.path_validation=defer|ssh` prevent local false-negative checks for remote paths; config-copy generation still requires readable local YAML. `execution.gpu_pool`, `gpus_per_run`, `max_concurrent`, `conda_env`, `wandb_project`, `wandb_group`, and `env` are wrapper settings only and must not replace trainer config semantics. Logs and PID files are always co-located in the managed run directory.

Hparam recipes may add an optional `adaptive:` block for append-only external-optimized tuning. `adaptive.max_runs_total` bounds the workflow. When `adaptive.objective_metric` starts with `test_` or `external_`, `adaptive.test_feedback_for_selection` must be `true`. Adaptive commands write under `adaptive/` and must not rewrite earlier round plans, run configs, logs, or checkpoints.

Removed run-budget, GPU-allocation, identity, plan-list, status-file, and registry names are rejected at input boundaries. They are not aliases for the current fields.

Hparam external evaluation, logits export, thresholding, and ensembling validate the complete candidate table before filtering ownership-valid rows from other steps or earlier plans in the same step. Any removed-format field or incomplete managed identity therefore rejects the whole input. After validation, every retained `(step_id, run_id)` must belong to the current managed plan. When a shared step ranking contains earlier-plan rows, `top_k` selects current-plan rows in numeric rank order rather than treating the step-global rank number as a local cutoff. Candidate tables may contribute derived checkpoint, prediction, score, and rank fields, but a checkpoint must be a direct child of that run's frozen checkpoint directory. Frozen identity, version, config, and artifact paths come from the plan. Every deterministic postprocess output is topology-checked before any directory creation, inference, or write.

When an unlocked final external-test script is requested for a hparam recipe that uses `yaml:/...` overrides, the recipe or user-decision file must provide `final_eval_config_path` for the selected checkpoint.

High-impact decisions must be explicit in `decisions:` or supplied by a user-decision file.

When a user-decision file selects a different finetune, inference, evaluation, or hparam base config, that effective config replaces the recipe input for the resolved plan and passes the same config summary and consultation gates before any workspace or run artifact is written. Hparam plans also freeze it in the base-recipe owner used to generate every run snapshot. An explicit empty or `ASK_USER` decision remains unresolved; a missing, unreadable, non-finetune, or semantically blocked selected config cannot leave a partially initialized plan.
