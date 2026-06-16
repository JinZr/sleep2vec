# Skill: sleep2stat

## When to use
Use for `sleep2stat` tasks that generate sleep-signal alignment and statistics bundles for later QC review, cohort harmonization, reporting, or error analysis.

Use this skill for model-derived sleep statistics, YASA PSG microstructure summaries, SpO2/desaturation summaries, and per-record or cohort sleep2stat plots.

Do not use this skill for model training, hyper-parameter tuning, direct sample exclusion, raw NPZ/index mutation, or automatic cleaning-rule selection.

## Required inputs
Requires a sleep2stat config YAML, explicit split policy, explicit metric-use policy, and explicit recipe-level overwrite policy.

Sleep2stat run directories are single-use. Re-run with a new `run.output_dir`, or manually clear the old directory before running again.

The config must explicitly set `data.backend`, `data.path_column`, `data.duration_column`, `data.split_column`, `data.token_sec`, and `data.max_tokens`. `sleep2stat.config.load_config()` is the schema boundary; agent tooling must report its blocking error instead of inferring or translating sleep2stat fields.

For enabled `sleep2vec_downstream` analyzers, require concrete downstream config and checkpoint paths. Placeholder values such as `/path/to/...`, `ASK_USER`, `TODO`, or `<...>` must block command generation.

For AHI `sleep2vec_downstream` analyzers, `postprocess.min_event_duration_sec`, `postprocess.merge_tolerance_sec`, `postprocess.output_second_alignment`, and `postprocess.output_event_alignment` must be explicit. For `spo2_desaturation`, `drop_thresholds` and `min_duration_sec` must be explicit. For `yasa_bandpower`, `outputs.by_epoch`, `outputs.by_stage`, `outputs.by_night`, and `outputs.relative` must be explicit. Do not duplicate these schema checks in agent tooling.

## First information-gathering commands
- `python -m agent_tools config-summary --config <config> --json`
- `python -m sleep2stat validate-config --config <config>`
- `python utils/check_configs.py <config>`
- `python -m agent_tools doctor --recipe <recipe> --output-dir artifacts/agent_context/<recipe-name>`

For NPZ configs, also inspect the index when path validation is local:

- `python -m agent_tools index-summary --index <index> --json`

Use the index path reported by `config-summary` or `context`; do not pass a sleep2stat config to `index-summary`.

For enabled `yasa_stage` configs, run record metadata preflight before the bundle run:

```bash
python -m sleep2stat validate-config --config <config> --check-records --split <split>
```

## Decision checklist
Confirm:

- `task: sleep2stat` and no `variant` field.
- `inputs.config` points to a sleep2stat config.
- `inputs.split` is intentional, or config `data.split` should be used.
- `artifacts.run_dir`, when present, exactly matches config `run.output_dir`.
- test split outputs are descriptive-only unless the user explicitly unlocks a different policy.
- model-derived respiratory metrics are treated as proxy metrics unless clinical annotation analyzers are configured.
- recipe `overwrite_policy` is explicit; config-level `run.overwrite` and output-dir reuse are unsupported.

## Stop-and-consult gates
Stop and ask the user before continuing if:

- `variant` is present with any non-null value, including `sleep2stat`.
- the config path is missing or not a sleep2stat config.
- `sleep2stat.config.load_config()` reports a config error.
- an enabled model analyzer has a missing or placeholder downstream config or checkpoint path.
- `artifacts.run_dir` differs from config `run.output_dir`.
- the split includes `test` and `external_test_locked` is not explicitly true.
- `sleep2stat_split_policy` or `sleep2stat_metric_use_policy` is missing or marked `ASK_USER`.
- YASA or SpO2 unit/scale semantics are unclear to the user.
- a requested report describes model-derived AHI, ODI, or hypoxic-burden-like fields as clinical labels without clinical annotation sources.
- remote paths are present but path context and validation policy are not explicit.

## Canonical commands
Validate config:

```bash
python -m sleep2stat validate-config --config <config>
python utils/check_configs.py <config>
```

Run a bundle:

```bash
python -m sleep2stat run \
  --config <config> \
  --split <split> \
  --device cuda \
  --num-workers 8
```

For NPZ/YASA/SpO2-only configs, `--num-workers N` runs records through an internal single-machine `splitN` plan. Configs with `sleep2vec_downstream` keep the canonical model path and pass `num_workers` to the model DataLoader.

Summarize and plot:

```bash
python -m sleep2stat summarize --run-dir <run_dir> --num-workers 8
python -m sleep2stat plot-record --run-dir <run_dir> --record-id <record_id>
python -m sleep2stat plot-cohort --run-dir <run_dir> --group-column source
```

Finalize completed runs:

```bash
python -m sleep2stat cohort-finalize --output-run-dir <out> --input-run-dir <run1> --input-run-dir <run2>
```

Generate an agent plan:

```bash
python -m agent_tools doctor --recipe <recipe> --output-dir artifacts/agent_context/<recipe-name>
python -m agent_tools plan --recipe <recipe> --output-dir artifacts/agent_plans/<recipe-name>
```

## Expected artifacts
A completed run should create:

```text
<run_dir>/
  config.yaml
  cli_args.yaml
  run_manifest.json
  record_manifest.csv
  status/
    pid.json
    progress.json
    failures.csv
  tables/
    night_stats.csv
    event_alignment.csv.gz
    model_summary.csv
    analyzer_summary.csv
  per_record/
    <record_id>/
      events.csv.gz
      night_stats.json
      result_manifest.csv
      arrays.npz
```

`events.csv` is used instead of `events.csv.gz` when compression is disabled. `arrays.npz` is optional and appears only when analyzers emit arrays. Per-record `epoch_alignment.csv.gz`, `second_alignment.csv.gz`, `event_alignment.csv.gz`, and `night_stats.csv` may exist as analysis sidecars.

## Validation gates
Minimum validation:

```bash
python -m sleep2stat validate-config --config <config>
python -m sleep2stat validate-config --config <config> --check-records  # when YASA stage is enabled
python utils/check_configs.py <config>
python -m agent_tools skills --validate
```

Agent-tool validation:

```bash
python -m agent_tools doctor --recipe <recipe> --output-dir artifacts/agent_context/<recipe-name>
python -m agent_tools plan --recipe <recipe> --output-dir artifacts/agent_plans/<recipe-name>
```

Targeted tests after changing this skill or sleep2stat agent support:

```bash
python -m pytest -q \
  tests/test_agent_tools_sleep2stat.py \
  tests/test_agent_tools_skills.py \
  tests/test_agent_tools_recipes.py \
  tests/test_agent_consultation_policy.py \
  tests/test_agent_plan_blocks_on_ambiguity.py \
  tests/test_agent_tools_config_summary.py
```

## Common failure modes
Run directory mismatch:

- Symptom: run writes one directory but summarize or plot reads another.
- Fix: make recipe `artifacts.run_dir` exactly match config `run.output_dir`.

Placeholder checkpoint:

- Symptom: doctor blocks on `sleep2stat_config`.
- Fix: provide a concrete checkpoint path or disable the model analyzer.

Wrong path context:

- Symptom: local path validation fails for remote absolute paths.
- Fix: set `execution.path_context: remote` and `execution.path_validation: defer` or `ssh`.

Metric misuse:

- Symptom: report calls `ahi_model` or predicted AHI a clinical AHI.
- Fix: label it as model-derived/proxy unless clinical annotation analyzers are configured.

## Relevant owners and index pages
Owners: `agent-tooling-maintainer`, `runtime-orchestrator`, `regression-guard`.

Relevant index:

- `doc/codex_index/branches/main/WORKFLOWS/AGENT_TOOLING.md`
- `doc/codex_index/branches/main/FUNCTIONS/AGENT_TOOLING.md`
- `doc/codex_index/branches/main/WORKFLOWS/SLEEP2STAT.md`
- `doc/codex_index/branches/main/FUNCTIONS/SLEEP2STAT.md`
