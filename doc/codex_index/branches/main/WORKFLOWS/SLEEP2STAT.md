# sleep2stat Workflow

## Purpose

Run derived sleep-statistics bundles from model outputs, NPZ reference signals, YASA analyzers, SpO2 analyzers, and reducers. This workflow writes auditable per-record sidecars and optional global cohort tables. It is not a training workflow.

## Entry Commands

- `python -m sleep2stat validate-config --config configs/sleep2stat/<name>.yaml`
- `python -m sleep2stat run --config configs/sleep2stat/<name>.yaml [--split val test] [--device cuda] [--num-workers N] [--batch-size N] [--limit-records N] [--dry-run]`
- `python -m sleep2stat summarize --run-dir <run.output_dir>`
- `python -m sleep2stat plot-record --run-dir <run.output_dir> --record-id <record_id>`
- `python -m sleep2stat plot-cohort --run-dir <run.output_dir> [--group-column source] [--stage-source <analyzer_name>] [--adjust-covariates age sex]`
- Agent path: `python -m agent_tools doctor|context|plan` with `task=sleep2stat`.

## Agent Consultation

Before generating runnable sleep2stat commands from a recipe, run the agent consultation gate through `agent_tools doctor`, `agent_tools context`, or `agent_tools plan`.

For `task=sleep2stat`:

- `variant` must be omitted or `null`; `sleep2stat` is not a model variant.
- `inputs.config` must point to a valid sleep2stat YAML.
- `artifacts.run_dir`, when present, must match `run.output_dir` in the YAML because the CLI writes to the config path.
- test/final-test split use must respect the external-test locking policy.
- placeholder downstream analyzer `config` or `ckpt_path` values are agent risk issues and should block runnable plans until resolved.

Generated commands should call only the existing CLI: `validate-config`, `run`, optional `summarize`, and optional `plot-cohort`.

## Config Validation

Use `sleep2stat.config.load_config` directly or through:

```bash
python -m sleep2stat validate-config --config configs/sleep2stat/model_first.yaml
python utils/check_configs.py configs/sleep2stat/model_first.yaml
```

Validation is strict:

- unknown top-level or nested fields fail
- `data.backend`, `data.path_column`, `data.duration_column`, `data.split_column`, `data.token_sec`, and `data.max_tokens` must be explicit
- `data.backend=kaldi` requires `kaldi_data_root` and `kaldi_manifest`
- YASA analyzers are not supported with Kaldi backend
- AHI `sleep2vec_downstream` analyzers require explicit postprocess duration and output-alignment controls
- `spo2_desaturation` requires explicit `drop_thresholds` and `min_duration_sec`
- `yasa_bandpower.outputs` requires explicit `by_epoch`, `by_stage`, `by_night`, and `relative`
- analyzer and reducer names must be unique
- reducer and stage-source references must point to enabled, earlier analyzer results
- configured global tables must be known table names

## Record Loading

Use `sleep2stat.io.records.load_records`.

NPZ records:

- require `data.index`
- preserve `data.path_column` as provided
- use configured `record_id_columns` when present, otherwise `<path.stem>__row<row_idx>`
- reject path-unsafe record ids and duplicate ids

Kaldi records:

- read `data.kaldi_manifest` under `data.kaldi_data_root` when relative
- iterate requested split manifests
- use `sample_key` as the default record id unless `record_id_columns` is configured
- preserve row metadata, including `sample_key`, for model analyzer filtering

## Analyzer Order

Analyzer order matters because later analyzers can use earlier stage sources.

Common chain:

1. `npz_stage_reference` or `yasa_stage` produces an epoch stage source.
2. `sleep2vec_downstream` produces model stage, scalar, or AHI outputs.
3. `yasa_bandpower`, YASA event analyzers, SpO2 desaturation, and event-related hypoxic burden consume prior stage or event sources when configured.
4. reducers summarize or compare analyzer outputs.

Canonical analyzer fields are intentionally narrow: `npz_stage_reference` uses `stage_key`, `yasa_bandpower` uses top-level `stage_source`, SpO2 analyzers use `input_channels[0]`, `spo2_desaturation` uses explicit thresholds and duration, and AHI model threshold uses scalar top-level `threshold` or checkpoint metadata.

Use `StageSourceResolver` for sleep-hour, REM/NREM-hour, stage-minute, and onset-stage lookup. Do not recalculate stage masks inside analyzers.

## Editing Output Metrics

1. Identify whether the field is analyzer-produced or reducer-produced.
2. Reuse `StageSourceResolver` for stage denominators instead of creating local masks.
3. Encode units and denominators in public night-stat field names.
4. Do not add duplicate aliases for existing output metrics.
5. Add or update focused tests in `tests/test_sleep2stat_analyzers.py`, `tests/test_sleep2stat_reducers.py`, `tests/test_sleep2stat_writers.py`, or `tests/test_sleep2stat_cli.py`.

## Model Analyzer

`sleep2vec_downstream` loads a namespace-local finetune model:

- `namespace: sleep2vec`, `sleep2vec2`, or `sleep2expert`
- `config`: finetune config path
- `ckpt_path`: model checkpoint path
- `label_name`: downstream label
- `input_channels`: sleep2stat signal channels to feed the model

NPZ model inference builds `_Sleep2statDataset`, which reuses `DefaultDataset` collate semantics. Kaldi model inference builds filtered `KaldiPSGDataset` instances per split. Kaldi manifests produced by embedding export are rejected because they contain backbone embeddings rather than raw token inputs.

For AHI:

- AHI threshold comes from scalar analyzer `threshold` or checkpoint `ahi_eval_threshold`
- postprocess `min_event_duration_sec`, `merge_tolerance_sec`, `output_second_alignment`, and `output_event_alignment` must be explicit in analyzer config
- second alignment covers only model-covered seconds
- event extraction merges short gaps before minimum-duration filtering
- model-hour and recording-hour event rates are QC outputs
- clinical `pred_ahi` is written only when a configured stage source supplies sleep-hour denominators

## YASA And SpO2 Boundaries

YASA and SpO2 analyzers read raw NPZ arrays through `data.utils.load_npz`; they do not support Kaldi backend.

YASA:

- `yasa_stage` emits epoch stage ids and optional probability arrays
- `yasa_bandpower` emits epoch, night, and optional stage-specific bandpower summaries using explicit output mode fields
- `yasa_spindles`, `yasa_slowwaves`, and `yasa_rem` share event-summary logic
- `yasa_hrv_stage` requires a stage source

SpO2:

- `_spo2_signal` owns scaling and validity masks
- summary metrics keep recording-denominator T90/T88 explicit
- `spo2_desaturation` requires explicit drop thresholds and minimum duration; missing values are config errors
- ODI metrics distinguish recording-hour, valid-SpO2-hour, and optional sleep-hour denominators
- event-related hypoxic burden deduplicates source events by onset/offset before integration

## Writing

`AnalysisBundleWriter` owns output side effects:

- `config.yaml`
- `cli_args.yaml`
- `record_manifest.csv`
- `status/progress.json`
- `run_manifest.json`
- per-record `events.csv.gz`, `night_stats.json`, `result_manifest.csv`, and optional `arrays.npz`
- global table shards under `tables/_shards/`
- rebuilt `tables/night_stats.csv`, `model_summary.csv`, `analyzer_summary.csv`, and optional alignment tables

Output directories are single-use. `run` and `dry-run` fail when `run.output_dir` already exists and is non-empty. Summarize can rebuild global tables from an existing complete run directory and its archived config.

Config-level overwrite and skip-existing are not supported. Reruns must use a new `run.output_dir` or a manually cleared directory.

## Plotting

Use bundle outputs as the source of truth:

- `plot-record` reads one per-record directory and uses the stable per-record `events.csv(.gz)` sidecar for event overlays.
- `plot-cohort` reads canonical columns from `tables/night_stats.csv`, selects a stage source, and writes sleep composition, sleep metrics, respiratory metrics, microstructure metrics, and optional harmonization diagnostics.

## Verification

Small config or command-rendering changes:

```bash
python -m sleep2stat validate-config --config configs/sleep2stat/model_first.yaml
python utils/check_configs.py configs/sleep2stat
python -m agent_tools skills --validate
```

Analyzer/reducer/writer changes:

```bash
PYTHONPYCACHEPREFIX=/tmp/sleep2vec_pycache python3 -m compileall sleep2stat tests
python3 -m pytest -q \
  tests/test_sleep2stat_config.py \
  tests/test_sleep2stat_analyzers.py \
  tests/test_sleep2stat_reducers.py \
  tests/test_sleep2stat_writers.py \
  tests/test_sleep2stat_cli.py
```

Agent recipe changes:

```bash
python3 -m pytest -q tests/test_agent_tools_sleep2stat.py tests/test_agent_plan_blocks_on_ambiguity.py
python -m agent_tools skills --validate
```

Use the repository `exp` environment on this machine when base Python lacks project dependencies.

## Common Edit Hotspots

- Schema or analyzer type support: `sleep2stat/config.py`, `configs/sleep2stat/`, `tests/test_sleep2stat_config.py`
- Model analyzer data/model flow: `sleep2stat/analyzers/model_downstream.py`, `data/default_dataset.py`, `data/kaldi_psg_dataset.py`, `tests/test_sleep2stat_analyzers.py`
- Stage denominators: `sleep2stat/core/stage_sources.py`, analyzer tests
- SpO2 metrics: `sleep2stat/analyzers/spo2.py`, `tests/test_sleep2stat_analyzers.py`
- YASA metrics: `sleep2stat/analyzers/yasa.py`, `tests/test_sleep2stat_analyzers.py`
- Reducer metrics: `sleep2stat/reducers/`, `tests/test_sleep2stat_reducers.py`
- Bundle outputs: `sleep2stat/io/writers.py`, `sleep2stat/core/pipeline.py`, `tests/test_sleep2stat_writers.py`, `tests/test_sleep2stat_cli.py`
- Agent recipe support: `agent_tools/configs.py`, `agent_tools/decisions.py`, `agent_tools/plans.py`, `recipes/templates/sleep2stat_*.yaml`, `skills/sleep2stat/`, `tests/test_agent_tools_sleep2stat.py`
