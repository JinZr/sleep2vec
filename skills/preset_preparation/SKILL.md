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
Use `python preprocess/save_dataset_presets.py --config <config> --index <csv> ...`.

## Expected artifacts
Preset pickle files plus sidecar `<preset>.manifest.json` files when sidecar writing is enabled.

## Validation gates
Run `python preprocess/save_dataset_presets.py --help`, `python -m agent_tools preset-summary --preset <preset> --json`, and targeted preset tests.

## Common failure modes
Missing `path`, `split`, or `duration` columns; missing mask columns; missing configured NPZ keys; accidental overwrite; missing `preset_build` policy.

## Relevant owners and index pages
Owners: `preset-pipeline`, `data-contract-guardian`, `agent-tooling-maintainer`. Index: preprocessing workflow and preprocessing/data function catalogs.
