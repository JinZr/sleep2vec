# Agent Tooling Workflow

## Purpose

Generate agent context bundles, validate task recipes, enforce stop-and-consult policy, and create safe command plans around the existing canonical runtime entrypoints.

## Entry Commands

- `python -m agent_tools skills --list`
- `python -m agent_tools skills --validate`
- `python -m agent_tools repo-summary --json`
- `python -m agent_tools config-summary --config <yaml> --json`
- `python -m agent_tools index-summary --index <csv> --config <yaml> --json`
- `python -m agent_tools preset-summary --preset <pickle> --json`
- `python -m agent_tools doctor --recipe <recipe.yaml> [--user-decisions <yaml>]`
- `python -m agent_tools context --task <task> --variant <variant> --config <yaml> --output-dir <dir>`
- `python -m agent_tools plan --recipe <recipe.yaml> --output-dir <dir>`
- `python -m agent_tools hparam-launch --plan-dir <dir> [--execute]`
- `python -m agent_tools hparam-monitor --run-dir <dir> [--once] [--health]`
- `python -m agent_tools progress --run-dir <dir> [--remote <host>] [--json]`
- `python -m agent_tools hparam-stop --run-dir <dir> --trial-id <trial_id>`
- `python -m agent_tools hparam-select --run-dir <dir> --metric <metric> --mode max|min`
- `python -m agent_tools hparam-checkpoint-scan --run-dir <dir> --metric <metric> --mode max|min`
- `python -m agent_tools hparam-external-eval --run-dir <dir> --selected <csv> --unlock-final-test [--top-k N|--all-candidates]`
- `python -m agent_tools hparam-threshold --run-dir <dir> --selected <csv>`
- `python -m agent_tools hparam-ensemble --run-dir <dir> --candidates <csv> [--search-combinations]`
- `python -m agent_tools hparam-digest --run-dir <dir>`
- `python -m agent_tools hparam-suggest --workflow-dir <dir>`
- `python -m agent_tools hparam-adaptive-init --recipe <recipe.yaml> --output-dir <dir>`
- `python -m agent_tools hparam-adaptive-step --workflow-dir <dir> [--execute]`
- `python -m agent_tools hparam-adaptive-loop --workflow-dir <dir> --execute`

## Layers

- `skills/`: human-readable task playbooks and examples.
- `recipes/`: declarative task cards for finetune, preset, inference, and tuning plans.
- `agent_policies/`: consultation policy and approved defaults.
- `agent_tools/`: lightweight Python package for summaries, decision gates, context bundles, plans, and run collection.
- `doc/agent_contracts/`: context bundle, recipe, run manifest, user decision, and external-test-locking contracts.

## Stop-And-Consult Contract

Agent tooling returns:

- `PASS`: safe to continue.
- `WARN`: safe to continue with warnings.
- `NEEDS_USER_INPUT`: exit code `2`; ask the generated questions and do not generate runnable training scripts.
- `FAIL`: exit code `1`; invalid or missing required input.

High-impact decisions must come from explicit user decisions, explicit CLI arguments, explicit recipe fields, or explicit config fields. Approved defaults are reserved for low-impact runtime convenience values.

## Context Bundles

`agent_tools doctor` is read-only unless `--output-dir` is explicitly supplied. When an output directory is supplied and blocking issues exist, it writes diagnostic `questions.json` and `questions.md` without applying experiment overwrite policy.

`agent_tools context` writes `context.json` and `context.md` for every run. If any blocking issue exists, it writes `questions.json`, `questions.md`, and `commands.blocked.sh` instead of runnable `commands.sh`; `consultation_required` is true when any issue needs user input, even if another issue makes the overall status `FAIL`.

Context bundles include skill metadata, owners, relevant index docs, expected agent artifacts, and best-effort index or preset summaries when the config points to `data.finetune_data_index` or `data.finetune_preset_path`.

## Plan Generation

`agent_tools plan` runs consultation gates before writing scripts. Blocked plans write `plan.blocked.md` and questions only. Hyper-parameter trial scripts use validation-only selection and `--no-test-after-fit`; final external-test scripts require explicit final-test unlock and an explicit existing checkpoint path.

Recipe `variant` controls the generated module namespace. Supported values are `sleep2vec`, `sleep2vec2`, and `sleep2expert`; missing or unsupported variants block command generation.

Recipe `base_recipe` paths are resolved relative to the local recipe file first, then by the repository root fallback used by checked-in examples.

Generated finetune, hparam trial, infer, and final-test commands should propagate explicit supported runtime/input fields from the recipe or user decisions. Do not drop explicit checkpoint, pretrained-backbone, eval split, test-after-fit, device, batch-size, precision, or scheduler/checkpoint cadence fields when rendering scripts.

Hyper-parameter search uses `runtime.<name>` keys for supported CLI knobs and `yaml:/json/pointer/path` keys for generated config overrides. Bare search keys are invalid, and `search.max_trials` must be a positive integer. Hparam recipes also inherit base finetune consultation gates; a base config with no usable preset/index or missing high-impact finetune decision blocks tuning. Generated `run_all.sh` scripts are written so they can find sibling `trial_*.sh` files when invoked from outside the plan directory.

Optional hparam `execution` fields enable active orchestration after a plan has been generated. `hparam-launch` reads the generated `plan.json` and trial scripts, assigns GPUs from `execution.gpu_pool`, wraps commands with optional `conda run`, W&B project/group environment variables, nohup logs, and PID files, then writes `launch_manifest.tsv` and `trial_status.tsv`. Dry-run is the default; `--execute` is required to start processes. `execution.path_context=remote` with `path_validation=defer|ssh` lets doctor/plan avoid local false negatives for remote absolute paths; hparam config generation still requires local YAML content when config copies must be written. `hparam-stop` only terminates a PID recorded in the launch manifest. `hparam-monitor` updates trial state from recorded PIDs, logs, run manifests, W&B summaries, and checkpoint directories; `--health` adds opt-in short-command GPU, `/proc/<pid>/io`, log-age, checkpoint-count, and progress-file checks. `hparam-select` ranks trials by validation metrics and writes fixed `epoch=XX.ckpt` checkpoint paths rather than moving best aliases. `hparam-checkpoint-scan` scans local W&B history and checkpoint directories to rank fixed epoch checkpoints. `hparam-external-eval` remains locked unless `--unlock-final-test` is present and only copies selected trial YAMLs while replacing data entry fields; `--top-k` and `--all-candidates` allow evaluating more than rank 1. `hparam-ensemble --search-combinations` enumerates probability-average model combinations and ranks them by the requested metric.

Optional hparam `adaptive` fields enable append-only external-optimized tuning. `hparam-adaptive-init` creates `adaptive/rounds/round_000/` from the recipe and records `adaptive/events.jsonl`, `trial_registry.tsv`, and `workflow.json`. `hparam-digest` summarizes trial status, logs, run manifests, checkpoints, and metrics into `adaptive/digests/`. `hparam-suggest` writes deterministic `best_neighborhood` next-round recipes into `adaptive/suggestions/`. `hparam-adaptive-step` performs monitor, digest, suggestion, pending-trial supersede events, optional PID-safe bad-running-trial stops, and optional next-round launch. Adaptive workflows may use test/external metrics only when `adaptive.test_feedback_for_selection=true`; all adaptive reports are marked `external_optimized=true`.

Preset generation commands should render explicit supported `preset:` fields such as stride, channels, metadata, missing-channel policy, output template, overwrite, dry-run, manifest, and sidecar-manifest flags.

## Reuse Guidance

Do not create a second trainer or preprocessing runtime. Generated commands must call the existing variant package entrypoints, `preprocess/save_dataset_presets.py`, `preprocess/convert_npz_to_kaldi.py`, `sleep2vec2/preprocess/*`, `sleep2expert/preprocess/*`, and `utils/check_configs.py`. Long-running preprocessing should keep tqdm for humans and write `status/progress.json` for agents; `agent_tools progress` reads that file locally or through a single short SSH `cat`. Current progress writers cover root and variant Kaldi conversion, preset generation, preset merging, index splitting, missing-mask statistics, WatchPAT EDF conversion, `utils/fill_index_duration.py`, `utils/cut_ukb_sleep_with_asleep.py`, `utils/collect_ukb_demographics.py`, and `utils/parse_ukb_annotations_by_person.py`.

## Edit Hotspots

- Change consultation policy: `agent_policies/consultation_policy.yaml` plus `agent_tools/decisions.py`.
- Change recipe loading or base-recipe inheritance: `agent_tools/recipes.py`.
- Change context or plan artifacts: `agent_tools/plans.py`.
- Change skill validation: `agent_tools/skills.py`.
