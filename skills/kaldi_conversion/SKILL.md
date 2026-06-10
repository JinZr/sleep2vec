# Skill: kaldi_conversion

## When to use
Use for `kaldi_convert` tasks that convert NPZ indexes to Kaldi roots.

## Required inputs
Requires index CSV, config YAML, output directory, selected channels, token window settings, and missing-channel policy.

## First information-gathering commands
- `python -m agent_tools config-summary --config <config> --json`
- `python -m agent_tools index-summary --index <index> --config <config> --json`

## Decision checklist
Confirm `--channels-from-config`, `--extra-channels`, `--max-tokens`, `--stride-tokens`, split filters, and output root.

## Stop-and-consult gates
The agent must stop and ask the user before continuing if any high-impact decision is missing, ambiguous, conflicting, or marked as `ASK_USER`.

## Canonical commands
Use `python preprocess/convert_npz_to_kaldi.py --index <csv> --config <yaml> --output-dir <root> --max-tokens <n> --channels-from-config`.

## Expected artifacts
Kaldi `manifest.json`, split manifest CSV files, sorted `.scp` files, and ark shards. Kaldi runtime uses manifest roots instead of legacy preset pickles.

## Validation gates
Run converter `--help`, config summary, and index summary before conversion.

## Common failure modes
Duplicate sample keys, missing NPZ keys, missing `stage5`/`ahi` extra channels for label tasks, and missing Kaldi optional dependencies.

## Relevant owners and index pages
Owners: `preset-pipeline`, `data-contract-guardian`, `agent-tooling-maintainer`. Index: preprocessing workflow.
