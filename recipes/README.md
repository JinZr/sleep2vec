# Agent Task Recipes

Recipes are declarative task cards for agent tooling. They are safe to read and validate without importing the ML runtime.

Recipes use a closed, task-aware contract. Unknown or task-inapplicable fields fail before config loading and before a managed workspace is created; the authoritative allowlists are documented in `recipes/schemas/task_recipe.schema.md`.

`recipes/templates/` contains editable templates with placeholder or site-local paths that may require user input.
`recipes/examples/` contains tiny fixture recipes that should pass consultation gates in a clean checkout.

Set `variant` to `sleep2vec`, `sleep2vec2`, `sleep2expert`, or `sex_age_baseline`; generated commands use the matching package namespace.

For hyper-parameter tuning, use `runtime.<name>` search keys for supported CLI/runtime values and `yaml:/json/pointer/path` keys for generated config overrides.

Use `python -m agent_tools doctor --recipe <recipe.yaml>` to validate a recipe and consultation policy.
Use `python -m agent_tools plan --recipe <recipe.yaml> --output-dir <dir>` to generate a safe command plan after gates pass.

New runnable recipes must declare an `experiment` (`id`, `title`, `objective`, `root`, and `baseline`) and a named `step` (`id`, `phase`, and `purpose`). The output directory must be inside the experiment root. See `doc/agent_contracts/experiment_workspace.md`.

High-impact decisions must be explicit in `decisions:` or in a user-decision file. Do not rely on filenames or previous runs.
