# Task Recipe Contract

Task recipes live under `recipes/` and use schema version `1`. They describe task, inputs, runtime knobs, artifacts, evaluation policy, search space, and explicit decisions.

`variant` must be `sleep2vec`, `sleep2vec2`, or `sleep2expert`. Generated runtime commands use the matching package namespace.

Hyper-parameter search keys must be explicit: use `runtime.<name>` for supported CLI/runtime knobs or `yaml:/json/pointer/path` for generated config overrides.

Hparam recipes may add an optional `execution:` block for active orchestration after `agent_tools plan` has generated trial scripts. `execution.target` is `local` or `ssh`; `execution.host` is required for SSH. `execution.gpu_pool`, `max_concurrent`, `conda_env`, `wandb_project`, `wandb_group`, `log_dir`, `pid_dir`, and `env` are wrapper settings only and must not replace trainer config semantics.

Hparam recipes may add an optional `adaptive:` block for append-only external-optimized tuning. When `adaptive.objective_metric` starts with `test_` or `external_`, `adaptive.test_feedback_for_selection` must be `true`. Adaptive commands write under `adaptive/` and must not rewrite earlier round plans, trial configs, logs, or checkpoints.

When an unlocked final external-test script is requested for a hparam recipe that uses `yaml:/...` overrides, the recipe or user-decision file must provide `final_eval_config_path` for the selected checkpoint.

High-impact decisions must be explicit in `decisions:` or supplied by a user-decision file.
