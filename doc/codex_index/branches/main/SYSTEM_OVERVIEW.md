# System Overview

## Repository Shape

The repository is a config-driven multimodal sleep modeling system with six operational layers:

1. Schema and task semantics: `sleep2vec/config.py`, `sleep2vec/common.py`
2. Construction and extension: `sleep2vec/registry.py`, `sleep2vec/builders.py`, `sleep2vec/backbones/`, `sleep2vec/modules/`, `sleep2vec/cls/`, `sleep2vec/downstreams/`
3. Model and trainer runtime: `sleep2vec/pretrain_model.py`, `sleep2vec/downstream_model.py`, `sleep2vec/sleep2vec_modelling.py`, `sleep2vec/sleep2vec_finetuning.py`, `sleep2vec/sleep2vec_adaptation.py`
4. Data and preprocessing: `data/`, `preprocess/`, `sleep2vec/utils.py`, including both NPZ and Kaldi manifest backends
5. Analysis bundles and derived statistics: `sleep2stat/`
6. Runtime support and tooling: `sleep2vec/checkpoints.py`, `sleep2vec/results.py`, `sleep2vec/sleep2vec_inference.py`, `sleep2vec/distributed.py`, `sleep2vec/visualization/`, `utils/check_configs.py`, `agent_tools/`, and standalone utilities under `utils/`

`sleep2vec2/` and `sleep2expert/` are tracked standalone namespaces on this branch. They mirror the root runtime surface with package-local `data/` and `preprocess/` modules instead of importing root `data` or `preprocess`. `sleep2expert/` also owns the MoE RoFormer, MoE regularization, finetune tuning policy, routing-analysis export path, and compact MoE subnetwork export path.

`sleep2stat/` is a tracked analysis-bundle package. It reads NPZ or Kaldi record manifests, runs configured analyzers and reducers, and writes per-record sidecars plus optional global tables. Its model analyzer reuses the existing finetune model/data paths; raw YASA and SpO2 analyzers operate directly on NPZ signals.

Top-level behavior is not encoded in YAML alone. YAML defines model, loss, task, head, evaluation-visualization, and adapt references; entrypoints still inject runtime-only values such as learning rate, devices, checkpoint paths, diagnostics mode, and experiment naming.

## Primary Entrypoints

- `python -m sleep2vec.pretrain`
- `python -m sleep2vec.adapt`
- `python -m sleep2vec.finetune`
- `python -m sleep2vec.infer`
- `python -m sleep2expert.routing_analysis`
- `python -m sleep2expert.export_subnetwork`
- `python -m sleep2stat validate-config`
- `python -m sleep2stat run`
- `python -m sleep2stat summarize`
- `python -m sleep2stat plot-record`
- `python -m sleep2stat plot-cohort`
- `python utils/check_configs.py`
- `python -m agent_tools doctor --recipe ...`
- `python -m agent_tools context --task ...`
- `python -m agent_tools plan --recipe ...`
- Preprocessing CLIs under `preprocess/`

The package-local variant mirrors expose equivalent pretrain/adapt/finetune/infer and preprocessing module entrypoints under `sleep2vec2.*` and `sleep2expert.*`.

### sleep2stat

`sleep2stat` is a derived-analysis runtime rather than a trainer:

1. `sleep2stat.config.load_config` validates the strict YAML blocks: `run`, `data`, `signals`, `analyzers`, `reducers`, and `outputs`.
2. `sleep2stat.io.records.load_records` builds `SleepRecord` objects from NPZ index rows or Kaldi `manifest.json` split manifests.
3. `sleep2stat.core.pipeline.run_pipeline` prepares an `AnalysisBundleWriter`, rejects non-empty output directories, prepares enabled analyzers, executes analyzers chunk by chunk, applies reducers, and writes progress, manifests, and result bundles.
4. Analyzer/reducer construction goes through `sleep2stat.registry.create_analyzer` and `create_reducer`; registration side effects live under `sleep2stat/analyzers/` and `sleep2stat/reducers/`.
5. `sleep2stat.io.writers.AnalysisBundleWriter` owns per-record `events.csv.gz`, `night_stats.json`, `arrays.npz`, `result_manifest.csv`, global table shards, rebuilt cohort tables, and run manifests.
6. `sleep2stat.plot` renders per-record traces and cohort-level sleep, respiratory, microstructure, and harmonization panels from completed bundles.

`sleep2stat` does not support config-level overwrite or skip-existing; use a new or manually cleared `run.output_dir` when rerunning.

Agent-generated `sleep2stat` commands must go through `agent_tools` consultation gates first. `task=sleep2stat` is variantless; adding a `variant` value blocks command generation.

## Runtime Stack

### Pretrain

`pretrain.py` performs the orchestration:

1. Parse CLI.
2. Load YAML with `load_pretrain_config`.
3. Copy model-derived channel metadata and data-backend settings into `args`.
4. Build one training loader plus one validation loader whose batch sampler iterates channel pairs through `get_pretrain_dataloader`. The loader chooses `PSGPretrainDataset` for `npz` and `KaldiPSGDataset` for `kaldi`.
5. Instantiate `Sleep2vecPretraining`, which owns a `Sleep2vecPretrainModel`, contrastive loss, diagnostics hooks, and optional model averager.
6. Configure Lightning callbacks:
   - `ModelCheckpoint`
   - `EarlyStopping`
   - `LearningRateMonitor`
   - `PairAccLoggerCallback`
   - `GradScaleLoggerCallback`
7. Persist `config.yaml` and `cli_args.yaml`.
8. Train with `trainer.fit(...)`.

The monitored validation metric is `val_contrastive_acc`.

### Adapt

`adapt.py` is a staged pretrain variant for adding modalities to an existing backbone:

1. Parse CLI, including `--phase stage1|stage2`.
2. Load pretrain-style YAML and require a top-level `adapt` block.
3. Derive initial pair probabilities from `adapt.new_channels` and `adapt.stage2.pair_schedule`.
4. Build missing-channel-aware train/validation loaders with `get_pretrain_dataloader`.
5. Resolve or validate the experiment directory and phase-specific checkpoint layout via `_resolve_adapt_run_artifacts`.
6. Persist root-level and phase-specific `config*.yaml` / `cli_args*.yaml`.
7. Instantiate `Sleep2vecAdaptation`, which wraps `Sleep2vecPretraining` but applies adaptation freeze policy and stage-specific optimizer grouping.
8. Train with optional `AdaptPairScheduleCallback` during stage 2.

Stage transitions are strict: `--ckpt-path` resumes within the same phase only, while `--pretrained-backbone-path` is used for weight initialization and for stage1 -> stage2 transitions.

### Finetune

`finetune.py` normalizes downstream configuration and launches supervised training:

1. Parse CLI.
2. Call `apply_finetune_config(args)`.
3. That call loads YAML via `load_finetune_config`, validates task semantics, converts configured data paths to `Path`, applies `data.backend`, and enforces `data.data_channel_names == model.channels`.
4. Build train/val/test loaders via `get_finetune_dataloaders`; NPZ runs use preset/index inputs, and Kaldi runs use `kaldi_data_root` plus `manifest.json`.
5. Instantiate `Sleep2vecFinetuning`, which owns:
   - a `Sleep2vecPretrainModel` backbone
   - a `Sleep2vecDownstreamModel` head stack
   - optional pretrained-backbone loading
   - optional LoRA/DoRA insertion with YAML-configured rank, alpha, dropout, target modules, and separate adapters
   - optional model averaging
   - optional downstream evaluation visualization hooks
   - Cox survival loss and subject-level survival risk aggregation when `finetune.task.type=survival`
6. Run `trainer.fit(...)`.
7. Test the best checkpoint or requested checkpoint with `trainer.test(...)`.
8. Append evaluation metrics to a CSV via `sleep2vec.results.save_result_csv`.

Built-in task semantics include `stage3`, `stage4`, `stage5`, `ahi`, `sex`, and `age`. Custom YAML tasks may also declare `finetune.task.type=survival`, which requires `finetune.survival` sidecars and monitors either `val_loss/min` or `val_c_index/max`.

### Inference

`infer.py` reuses the finetune configuration path:

1. Parse CLI and validate explicit checkpoint path unless the alias is `best` or `last`.
2. Call `apply_finetune_config(args)`.
3. Build a single evaluation loader with `_build_inference_loader`.
4. Instantiate `Sleep2vecFinetuning`.
5. Optionally select and average checkpoints with `select_checkpoints` and `average_checkpoints`.
6. Run `trainer.test(...)`.
7. Prepare a run-local inference output directory under `results/inference/<namespace>/<label>/<prediction_run_id>/`.
8. Write run metrics, append `results/inference/overview.csv`, write path-level predictions, and write `run_manifest.json`.
9. If W&B is enabled, log metrics plus `prediction_row_count` and upload metrics, predictions, manifest, and overview files as one inference artifact.

AHI inference deliberately rejects checkpoint averaging because the fitted `ahi_eval_threshold` is checkpoint-specific.

## Core Data Contract

The root runtime supports two data backends:

- `npz`: `PSGPretrainDataset` reads index CSVs or preset pickles and loads NPZ arrays at collate time.
- `kaldi`: `KaldiPSGDataset` reads `manifest.json` format v2, split CSVs, and sorted `.scp` channel files through `KaldiReaderPool`; legacy NPZ preset pickles are rejected for this backend.

The runtime assumes a batch dictionary with these keys:

| Key | Producer | Consumer | Notes |
| --- | --- | --- | --- |
| `id` | `DefaultDataset.dataloader` | logging only | Sample ids |
| `length` | `DefaultDataset.dataloader` | backbone, downstream heads, losses | Token count per sample before padding |
| `token_start` | `DefaultDataset.dataloader` | AHI event aggregation | Preserves window offset for later record merging |
| `tokens` | `DefaultDataset.dataloader` | backbone, finetune loss, AHI eval | Channel-name keyed padded token tensors |
| `mlm_mask` | `DefaultDataset.dataloader` | pretrain backbone | Channel-name keyed boolean mask tensors |
| `metadata` | `DefaultDataset.dataloader` | finetune loss, metrics, negative weighting | Always includes `source` and `path`; includes `age`/`sex` only when present in the index or preset; built-in AHI also backfills `ahi` and `tst` |
| `w` | `DefaultDataset.dataloader` | `WeightedInfoNCELoss` | Negative-sample weight matrix |
| `h` | `DefaultDataset.dataloader` | `WeightedInfoNCELoss` | Same-path hardness mask |
| `pair` | `DefaultDataset.dataloader` | callbacks/logging | Present for pair-first or sequential pair-eval batches |

### Important Invariants

- All configured channels must share one tokenizer output dimension.
- `model.cls.downstream='cls'` requires a real CLS strategy such as `bert`.
- Built-in sequence tasks are `stage3`, `stage4`, `stage5`, and `ahi` only.
- `stage3` and `stage4` are runtime remaps over raw `stage5` tokens; their source labels still come from `stage5`.
- Built-in `ahi` requires NPZ key `ah_event` plus scalar NPZ keys `ahi` and `tst`; evaluation also expects `stage5` tokens as an auxiliary label source.
- Stage/AHI-only indexes do not need `age` or `sex` columns, but built-in `age` and `sex` runs fail fast if the loaded preset/index lacks valid labels after split/source filtering.
- Non-sequence metadata classification remains binary-only.
- Survival finetuning uses sidecar metadata vectors named `event_time`, `is_event`, and `has_label`, plus the configured survival key column.
- Pair-first missing-channel pretraining requires `payload["available_channels"]` on every retained sample.
- Weighted InfoNCE requires both `w` and `h` in the batch.

## Construction Model

The canonical creation path is:

1. Parse YAML into dataclasses with `load_pretrain_config`, `load_finetune_config`, or `load_model_config`.
2. Bind runtime args with `apply_model_config_args` or `apply_finetune_config`.
3. Resolve extensions through registries:
   - backbone registry
   - tokenizer registry
   - projection registry
   - model averager registry
4. Build the encoder factory, tokenizer mapping, and projection head via `builders.py`.
5. Build CLS behavior via `build_cls_embedding`.
6. Build downstream temporal/channel/head composition through `sleep2vec/downstreams/`; temporal aggregation supports `mean`, `attn`, and `lstm`.

## Preprocessing And Config Validation

The preprocessing surface is split between reusable CLIs, standalone data utilities, and one notebook:

- `split_index_by_dataset.py`: assign `train/val/test/external`, normalize mask truthiness, optionally enforce global pair coverage
- `mask_missing_stats.py`: summarize `_mask` coverage
- `save_dataset_presets.py`: build preset pickles through `PSGPretrainDataset`, including YAML-driven `preset_build` validation
- `merge_dataset_presets.py`: concatenate multiple preset pickles
- `convert_npz_to_kaldi.py`: convert CSV-indexed NPZ windows into split-specific Kaldi ark/scp roots plus `manifest.json` format v2
- `configs/hypnodata/`: example configuration and notes for the external `hypnodata` raw-PSG normalization layer, which can feed NPZ, Kaldi conversion, preset generation, and sleep2stat record manifests
- `watchpat_zzp_to_edf.py`: convert WatchPAT `.zzp` archives to EDF and optional JSON summary
- `utils/cut_ukb_sleep_with_asleep.py`: cut UKB `.cwa` nights with the standalone `asleep` package
- `utils/parse_ukb_annotations_by_person.py`: parse UKB annotation bundles into dataset metadata and per-participant JSON
- `utils/collect_ukb_demographics.py`: collect age/sex fields from participant JSON trees
- `utils/fix_kaldi_index.py`: repair duplicate Kaldi sample-key prefixes by assigning unique `session_id` values
- `utils/match_case_controls.py`: build matched case-control cohorts with exact matching, calipers, propensity scores, and balance diagnostics
- `preprocess_pipeline.ipynb`: manual, dataset-specific workflow history

The canonical NPZ preset path is:

`CSV split prep -> optional mask analysis -> preset generation -> optional preset merge`

The canonical Kaldi path is:

`CSV split prep -> optional mask analysis -> convert_npz_to_kaldi -> runtime data.backend=kaldi`

`utils/check_configs.py` is the branch-local tooling path for validating:

- runtime config loader compatibility
- shared tokenizer-dimension parity
- YAML `preset_build` strictness
- repo-specific `ppg_*finetune*.yaml` contracts
- tracked example recipes under `configs/examples/**`
- `configs/sleep2stat/*.yaml` through `sleep2stat.config.load_config`

## Outputs And Side Effects

- Pretrain checkpoints: `log-pretrain/<run>/checkpoints/`
- Adapt checkpoints:
  - stage1: `log-adapt/<run>/checkpoints/`
  - stage2: `log-adapt/<run>/checkpoints.stage2/`
- Finetune checkpoints: `log-finetune/<version>/checkpoints/`
- Per-run copied configs: `config.yaml`, plus phase-specific `config.stage1.yaml` / `config.stage2.yaml` for adaptation
- Per-run CLI snapshot: `cli_args.yaml`, plus phase-specific adaptation snapshots
- Downstream results table: caller-specified CSV via `sleep2vec.results.save_result_csv`
- Inference outputs: run-local `metrics__*.csv`, `predictions__*.csv`, `run_manifest.json`, shared `results/inference/overview.csv`, and optional W&B inference artifacts
- Survival inference prediction rows include raw log-risk vectors and sidecar label vectors when prediction export is enabled.
- Visualization side effects:
  - pair-accuracy heatmaps
  - layer-mix heatmaps and tables
  - confusion matrices, ROC curves, and regression scatter plots
- Preprocessing outputs:
  - preset pickles
  - Kaldi `manifest.json`, split CSV manifests, sorted `.scp` files, and ark shards
  - split CSVs
  - mask statistics CSVs
  - EDF files and optional JSON summaries
- Agent outputs:
  - context bundles under `artifacts/agent_context/`
  - command plans under `artifacts/agent_plans/`
  - generated trial configs under `runs/generated/`
  - blocked question files when consultation gates require user input
- sleep2stat outputs:
  - `config.yaml`, `cli_args.yaml`, `run_manifest.json`, and `record_manifest.csv`
  - `status/progress.json`
  - per-record `events.csv.gz`, `night_stats.json`, optional `arrays.npz`, and `result_manifest.csv`
  - global `tables/night_stats.csv`, summary tables, and optionally epoch/second/event alignment tables
  - `plots/` outputs for per-record and cohort visualization commands

## Variant State On This Branch

`sleep2vec2/` is an active tracked standalone mirror of the root recipe with a package-local RoFormer implementation and LoRA/DoRA adapter support aligned with root downstream behavior. `sleep2expert/` is an active tracked standalone variant that adds MoE configuration, sparse RoFormer FFN routing, MoE pretrain/finetune regularization, MoE-aware LoRA optimizer grouping, MoE checkpoint initialization, runtime route filtering, routing export/visualization, and compact subnetwork export.
