# Skill: preset_preparation

## When to use
Use for `preset_prepare` tasks that build or inspect NPZ preset pickles with `preprocess/save_dataset_presets.py`.

## Required inputs
Requires a config YAML, one index CSV, dataset name, split list, token window settings, channel policy, missing-channel policy, and overwrite/regeneration decision.

## First information-gathering commands
- `python -m agent_tools config-summary --config <config> --json`
- `python -m agent_tools index-summary --index <index> --config <config> --json`
- `python -m agent_tools doctor --recipe <recipe>`

## Decision checklist
Confirm `preset_build.required_channels`, `preset_build.min_channels`, split handling, metadata labels, dry-run behavior, overwrite behavior, and sidecar preset manifest policy.

## Stop-and-consult gates
The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

Stop and consult the user if:

- The split list is missing.
- The required channels are unclear.
- Missing-channel policy is unclear.
- Existing preset files would be overwritten.
- The recipe does not say whether to reuse or regenerate presets.

## Canonical commands
Use the generated top-level plan script, which delegates to `preset-launch`
and defaults to dry-run; add `--execute` only when execution is authorized. New
direct plans detach the worker with closed input, persistent stdout/stderr logs,
and recorded process identity. Do not wrap the worker in an ad hoc background
SSH shell or retry a launch after a lost receipt. Observe with
`experiment-monitor` and stop with `preset-stop --reason` under the
[managed preset contract](../../doc/agent_contracts/task_recipe.md#managed-preset-preparation).
The workload and lifecycle commits share the planned interpreter.
New local plans at the manager checkout freeze its current
Python and baseline Git HEAD by default. For a different workdir or remote path
context, provide a complete local `execution.python`,
`execution.runtime_commit`, and absolute `execution.workdir` identity in the
recipe; use an absolute Python path to avoid launcher PATH drift. Plan on the
execution host: preset plans do not provide recipe-driven SSH execution.
Historical plans without the direct
scheduler declaration retain their original script behavior; do not patch them
to add identity or route them through the new launcher. The provenance-aware
launcher orders frozen script/config verification, HEAD capture, and child
`Popen` inside the same short runtime lock; the script uses that value when it
records planned and actual commits in the canonical manifest. The lock is
released for the child lifetime. A mismatch warns without blocking or rewriting
plan bytes, and the observation does not freeze checkout bytes for the whole preset job.
Entry points
remain variant-local: `preprocess/save_dataset_presets.py`,
`sleep2vec2/preprocess/save_dataset_presets.py`, or
`sleep2expert/preprocess/save_dataset_presets.py`.

For rolling latest-main maintenance, keep one existing checkout and run
`python -m agent_tools runtime-sync --workdir <checkout>` as a dry-run before an
authorized `--execute` fast-forward of `origin/main`. Do not clone or reset.
Sync and launch lock only their short critical sections, so an older preset
process may continue after its spawn boundary records A while a later process
records B. `runtime-sync` rejects tracked or importable-code dirt. Preset launch
itself requires the frozen Python to start and rechecks the frozen plan/run,
script/config, and emitted input hashes; it does not claim the hparam
managed-scheduler module-origin or live-argv gates. Mixed commits are provenance
rather than a scientific variable.

When the config defines `preset_build`, that block is the sole runtime owner of
`required_channels` and `min_channels`. Keep matching decisions for provenance,
but do not add `preset.channels` or `preset.min_channels`; generated commands
must not include `--channels` or `--min-channels`. When `preset_build` is absent,
the recipe decisions materialize those two preset CLI fields instead.

## Expected artifacts
Preset pickle files plus sidecar `<preset>.manifest.json` files when sidecar writing is enabled.

## Validation gates
Run `python preprocess/save_dataset_presets.py --help`, `python -m agent_tools preset-summary --preset <preset> --json`, and targeted preset tests.

## Common failure modes
Missing `path`, `split`, or `duration` columns; missing mask columns; missing configured NPZ keys; accidental overwrite; missing `preset_build` policy.

## Relevant owners and index pages
Owners: `preset-pipeline`, `data-contract-guardian`, `agent-tooling-maintainer`.
Index: [preprocessing workflow](../../doc/codex_index/WORKFLOWS.md#preprocessing-and-presets).
