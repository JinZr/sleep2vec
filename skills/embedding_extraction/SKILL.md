# Skill: embedding_extraction

## When to use
Use for a local `embedding_extraction` plan that exports whole-night NPZ embeddings from a pretrain or finetune checkpoint.

## Required inputs
Requires a `sleep2vec`, `sleep2vec2`, or `sleep2expert` variant; model config; checkpoint; explicit NPZ index CSV; eval split; unique model channels; fresh absolute embedding directory; and the fixed whole-night contract (`both`, final layer, NPZ output, batch size 1).

## First information-gathering commands
- `python -m agent_tools config-summary --config <config> --json`
- `python -m agent_tools doctor --recipe <recipe>`

## Decision checklist
Confirm config and checkpoint identity, NPZ index, split, channels, source-token cap, variant, device, worker count, output directory, and external-test unlock state. Use `max_source_tokens` from 1 through 4095 and keep `artifacts.overwrite: false`.

## Stop-and-consult gates
Stop when any required value is missing or `ASK_USER`. Test extraction requires `external_test_locked: false` and `final_test_unlocked: true`. Do not infer a channel set, split, checkpoint, token cap, or test unlock.

## Canonical commands
Run `python -m agent_tools plan --recipe <recipe> --output-dir <plan-dir>`, then use the frozen plan. The generated command routes to `<variant>.extract_embeddings` in the current checkout with NPZ, whole-night, `--embedding-kind both`, `--layer-index -1`, and `--batch-size 1`.

## Expected artifacts
The fresh embedding directory receives NPZ files and terminal `manifest.json`. The manifest records hashes for the frozen config, checkpoint, package-local extractor, and effective index files.

## Validation gates
Planning strictly loads the selected pretrain or finetune config, validates the selected index rows and token cap, checks every referenced NPZ file, and rejects non-empty or plan-overlapping output directories. Inspect the frozen config and command before launch.

## Common failure modes
Unsupported config-window, preset, Kaldi, source-override, remote-runtime, or alternate-workdir fields; empty split; duplicate paths; non-30-second duration; missing NPZ; token-cap overflow; locked test split; or a reused output directory.

## Relevant owners and index pages
Owners: `agent-tooling-maintainer`, `runtime-orchestrator`, `variant-maintainer`, `regression-guard`. Index: `doc/codex_index/WORKFLOWS.md` and `doc/codex_index/REUSE_GUIDE.md`.
