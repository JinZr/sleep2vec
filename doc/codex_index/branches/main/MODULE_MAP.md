# Module Map

## Natural Edit Boundaries

These are the stable cross-file boundaries that matter before editing:

| Boundary | Primary files | Responsibility | Key dependencies | Extension points | Notes |
| --- | --- | --- | --- | --- | --- |
| Config schema and built-in task semantics | `sleep2vec/config.py`, `sleep2vec/common.py` | Parse YAML, validate required blocks, define built-in labels, bind config state into CLI namespaces, persist run artifacts, normalize NPZ/Kaldi backend settings, and carry LoRA/DoRA adapter settings | `yaml`, dataclasses | New config fields, new built-in tasks, new eval-visualization knobs, new adapt fields, new data backends | Canonical parse and task-semantics boundary |
| Registry and builders | `sleep2vec/registry.py`, `sleep2vec/builders.py`, `sleep2vec/backbones/encoder_factory.py`, `sleep2vec/modules/`, `sleep2vec/cls/` | Register and instantiate backbones, tokenizers, projections, CLS strategies, and model averagers | Transformers, Torch | New registries or builder targets | Construction should flow through these helpers |
| Backbone model and adaptation freeze policy | `sleep2vec/pretrain_model.py` | Tokenize channels, add CLS, project to hidden size, run encoder, expose hidden states, and partition parameters for staged adaptation | builders, tokenizers, projection, CLS | New backbone behavior, tokenizer usage, adaptation grouping | Config-backed construction only |
| Downstream model and head composition | `sleep2vec/downstream_model.py`, `sleep2vec/downstreams/` | Temporal aggregation, channel fusion, downstream heads, layer mix, pretrained loading, LoRA insertion | `peft`, downstream registries | New heads, fusion modes, adapter behavior | Canonical downstream feature path |
| Lightning runtime | `sleep2vec/sleep2vec_modelling.py`, `sleep2vec/sleep2vec_finetuning.py`, `sleep2vec/sleep2vec_adaptation.py`, `sleep2vec/schedulers.py` | Training-loop glue, optimizer schedule, diagnostics, model averaging, AHI metrics, eval visualization logging, staged adaptation optimizer groups | PyTorch Lightning, loss/metrics, callbacks | New trainer behavior | Keep optimizer groups in runtime modules and shared LR schedule in package-local scheduler helpers |
| Runtime entrypoints | `sleep2vec/pretrain.py`, `sleep2vec/adapt.py`, `sleep2vec/finetune.py`, `sleep2vec/infer.py`, `sleep2vec/extract_embeddings.py` | CLI parsing, experiment folder setup, trainer creation, phase validation, checkpoint/test orchestration, inference W&B artifact logging, and token-embedding export | `sleep2vec.common`, `sleep2vec.utils`, Lightning | New CLI flags, run naming, checkpoint policy, embedding export options | Thin orchestration only |
| Runtime support | `sleep2vec/checkpoints.py`, `sleep2vec/results.py`, `sleep2vec/sleep2vec_inference.py`, `sleep2vec/distributed.py`, `sleep2vec/callbacks/`, `sleep2vec/visualization/`, `sleep2vec/diagnostics.py` | Checkpoint init/averaging, distributed rank helpers, result CSV writes, inference prediction export, pair-accuracy and grad-scale monitoring, progress bars, plots, diagnostics hooks | Torch, sklearn, wandb, matplotlib | New logging/export surfaces | Mostly support code, not model semantics |
| Dataset core | `data/default_dataset.py`, `data/psg_pretrain_dataset.py` | Build `SampleIndex` windows, validate/load presets, materialize batches, backfill built-in AHI metadata, and choose samplers | `data.utils`, `data.metadata`, `data.samplers` | New dataset formats or collate behavior | Main data contract boundary |
| Data helpers and samplers | `data/utils.py`, `data/metadata.py`, `data/channel_selection.py`, `data/samplers.py`, `data/kaldi_io.py`, `data/kaldi_psg_dataset.py`, `sleep2vec/utils.py` | NPZ I/O, Kaldi `.scp` readers, token windowing, preset/manifest filtering, metadata tensorization, pair scheduling, bucketed batching, finetune/pretrain loader assembly | NumPy, Torch, optional `kaldi_native_io` | New samplers, metadata encodings, loader policies, storage backends | Missing-channel pretrain and Kaldi routing rely on these |
| Preprocessing CLIs | `preprocess/*.py` | CSV splitting, preset generation/merge, NPZ-to-Kaldi conversion, required-channel prefiltering, missing-mask stats, WatchPAT conversion | pandas, pickle, NumPy, optional `kaldi_native_io`, optional EDF deps | New data-prep utilities | CLI tools are the canonical prep surface |
| Config-policy tooling | `utils/check_configs.py` | Validate repo YAMLs against loader contracts, tokenizer parity, preset-build strictness, and repo-specific ppg policy | config loaders, preset-build helpers | New static checks | Tooling boundary, not runtime |
| Agent tooling | `agent_tools/`, `skills/`, `recipes/`, `agent_policies/`, `doc/agent_contracts/` | Summarize repo/config/index/preset/sleep2stat state, validate skills and recipes, enforce stop-and-consult gates, write context bundles, generate safe command plans, orchestrate hparam trials, monitor experiments, and write progress artifacts | yaml, pandas, standard library, optional wandb | New agent task surfaces and policy gates | Must call canonical runtime/preprocess/sleep2stat entrypoints rather than replacing them |
| sleep2stat analysis bundles | `sleep2stat/config.py`, `sleep2stat/cli.py`, `sleep2stat/core/`, `sleep2stat/analyzers/`, `sleep2stat/reducers/`, `sleep2stat/io/`, `sleep2stat/plot.py`, `configs/sleep2stat/`, `tests/test_sleep2stat_*.py` | Validate analysis YAMLs, load NPZ/Kaldi records, run model/YASA/SpO2 analyzers, reduce derived metrics, write single-use per-record and global bundles, and render record/cohort plots | root `data` package, variant finetune modules through analyzer namespace, optional YASA/MNE, pandas/NumPy/matplotlib | New analyzer types, reducers, bundle outputs, plot panels, and sleep2stat recipe behavior | Derived analysis runtime; do not add trainer behavior here |
| Standalone data utilities | `utils/cut_ukb_sleep_with_asleep.py`, `utils/parse_ukb_annotations_by_person.py`, `utils/collect_ukb_demographics.py`, `utils/fix_kaldi_index.py`, `utils/match_case_controls.py` | Cut UKB `.cwa` nights, parse UKB annotation exports, collect age/sex demographics, repair duplicate Kaldi key inputs, and build matched case-control cohorts | `asleep`, pandas, NumPy, patsy, statsmodels, scipy, tqdm | External data preparation and cohort construction | These utilities are intentionally outside the model runtime and preset schema |
| Config recipes | `configs/` | Encode model/head/task variants, preset-build policy, adapt recipes, example recipes, standalone `sleep2vec2` recipes, and `sleep2expert` dense/MoE recipes | Config parser + entrypoints | New recipe variants | Folder names are not always semantically authoritative |
| Tests | `tests/` | Pin config, registry, LoRA adapter, AHI/specificity metric, checkpoint, result CSV, inference W&B artifact, preset-build, pair-sampler, and visualization contracts | pytest | New contract coverage | Many important contracts live here now |
| Formatting wrapper | `utils/style_check.sh` | Run `isort`, `black`, `flake8` over repo | Python env toolchain | None | Lint wrapper only |
| Standalone dense variant | `sleep2vec2/`, `configs/sleep2vec2/` | Package-local mirror of root runtime/data/preprocess plus standalone RoFormer, with tests for namespace isolation, Kaldi backend, LoRA/DoRA adapter parity, and dense parity | root contracts for behavior parity, local package imports | Variant-specific dense recipe changes | LoRA/DoRA support follows the root downstream adapter contract |
| Standalone MoE variant | `sleep2expert/`, `configs/sleep2expert/` | Package-local mirror plus MoE RoFormer, routing regularization, MoE finetune tuning, LoRA/DoRA adapter support, checkpoint expansion, routing export, and routing visualizations | `sleep2expert.config`, `sleep2expert.backbones.roformer.moe`, `sleep2expert.losses.moe_regularization` | MoE architecture, tuning, routing analysis | Router LoRA is intentionally unsupported; adapter params use a separate MoE finetune LR group |

## Key Dependencies

### External libraries used across boundaries

- PyTorch and PyTorch Lightning: core model and trainer runtime
- Hugging Face Transformers: backbone implementations via encoder factory
- PEFT: LoRA insertion in downstream model
- pandas / NumPy: CSV preprocessing, preset filtering, metrics, and result export
- kaldi_native_io: optional Kaldi ark/scp conversion and runtime matrix readers
- matplotlib / seaborn / wandb / scikit-learn: evaluation visualization and experiment logging
- optional EDF stack in WatchPAT conversion: `pyedflib`, `scipy`, `tqdm`

### Internal dependency flow

- `pretrain.py` -> `config.py` + `common.py` + `sleep2vec.utils` + `sleep2vec_modelling.py`
- `adapt.py` -> `config.py` + `common.py` + `sleep2vec.utils` + `sleep2vec_adaptation.py`
- `finetune.py` / `infer.py` -> `common.py` -> `config.py` + `sleep2vec.utils` + `sleep2vec_finetuning.py`
- `infer.py` -> `sleep2vec.results.prepare_inference_result_paths` + `save_result_csv` + `save_prediction_csv` + `save_inference_manifest`
- `infer.py` -> `_log_inference_outputs_to_wandb` after inference output files exist when W&B is enabled
- `Sleep2vecFinetuning` -> `sleep2vec.sleep2vec_inference` for path-level prediction rows during test/inference
- `sleep2vec.utils` -> `PSGPretrainDataset` for `data_backend=npz`, `KaldiPSGDataset` for `data_backend=kaldi`
- `KaldiPSGDataset` -> `KaldiReaderPool` -> sorted `scp:` readers from `kaldi_native_io`
- `Sleep2vecPretrainModel` -> `builders.py` -> registries/backbone/tokenizer/projection/CLS
- `Sleep2vecDownstreamModel` -> `Sleep2vecPretrainModel` + `downstreams/` + optional `peft`
- `Sleep2vecFinetuning` -> `metrics.py` + `results.py` + `visualization/downstream_eval.py`
- `DefaultDataset.dataloader` -> `data.utils` + `data.metadata` + `data.samplers`
- `save_dataset_presets.py` -> `split_index_by_dataset.normalize_mask_frame` + `PSGPretrainDataset` -> `DefaultDataset`
- `convert_npz_to_kaldi.py` -> `PSGPretrainDataset` channel registry + `data.utils.window` + Kaldi ark/scp writers
- `utils/check_configs.py` -> config loaders + preset-build helpers from `save_dataset_presets.py`
- `agent_tools.cli` -> `agent_tools.decisions` + summary modules + `agent_tools.plans` + `agent_tools.hparam` + `agent_tools.experiments` + `agent_tools.adaptive_hparam`; generated scripts call canonical variant package and `preprocess` entrypoints
- `agent_tools.plans` -> `sleep2stat.config.load_config` through `agent_tools.configs.sleep2stat_config_summary`; generated sleep2stat plans call `python -m sleep2stat validate-config`, `run`, `summarize`, and optional `plot-cohort`
- `sleep2stat.cli` -> `sleep2stat.config.load_config` + `sleep2stat.core.pipeline.run_pipeline` + `AnalysisBundleWriter.rebuild_global_tables` + `sleep2stat.plot`
- `sleep2stat.core.pipeline` -> `load_records` + `AnalysisBundleWriter` + analyzer/reducer registries
- `Sleep2vecDownstreamAnalyzer` -> namespace-local `common.apply_finetune_config`, `Sleep2vecFinetuning`, root `DefaultDataset` for NPZ, and root `KaldiPSGDataset` for Kaldi
- YASA and SpO2 sleep2stat analyzers -> raw NPZ arrays through `data.utils.load_npz`; they do not support Kaldi backend
- `utils/cut_ukb_sleep_with_asleep.py` -> standalone `asleep.get_sleep` helpers; no sleep2vec imports
- `utils/parse_ukb_annotations_by_person.py` -> raw UKB `.tab` / `.html` / `.r` / withdrawal files; writes derived metadata and participant JSON
- `utils/collect_ukb_demographics.py` -> participant JSON trees from the parser or similar UKB-style exports
- `utils/fix_kaldi_index.py` -> index CSVs before `convert_npz_to_kaldi.py` when duplicate sample keys would be generated
- `utils/match_case_controls.py` -> flat cohort CSVs; writes matched, unmatched, excluded, count, and balance CSVs
- `sleep2expert.routing_analysis` -> `sleep2expert.infer._build_inference_loader` + `Sleep2vecFinetuning` + `backbone.last_moe_aux`

## Important Ownership Notes

- Config/task edits usually span `sleep2vec/config.py` and `sleep2vec/common.py`.
- Forward-path edits usually span `sleep2vec/pretrain_model.py`, `sleep2vec/downstream_model.py`, and the downstream fusion modules.
- AHI runtime edits usually span `sleep2vec/common.py`, `sleep2vec/utils.py`, `data/default_dataset.py`, `data/utils.py`, `sleep2vec/metrics.py`, and `sleep2vec/sleep2vec_finetuning.py`.
- LoRA/DoRA adapter edits usually span `sleep2vec/config.py`, `sleep2vec/common.py`, `sleep2vec/downstream_model.py`, `sleep2vec/sleep2vec_finetuning.py`, and `tests/test_downstream_separate_adapters.py`; package-local variant edits mirror those files under `sleep2vec2/` or `sleep2expert/`.
- Missing-channel pretraining edits usually span `data/default_dataset.py`, `data/utils.py`, `data/samplers.py`, and `sleep2vec/utils.py`.
- Kaldi backend edits usually span `sleep2vec/common.py`, `sleep2vec/utils.py`, `data/kaldi_io.py`, `data/kaldi_psg_dataset.py`, and `preprocess/convert_npz_to_kaldi.py`; duplicate-key index repairs belong in `utils/fix_kaldi_index.py`.
- Adaptation edits usually span `sleep2vec/adapt.py`, `sleep2vec/sleep2vec_adaptation.py`, `sleep2vec/pretrain_model.py`, and sampler/callback surfaces used by pair scheduling.
- Preprocessing edits should stay in `preprocess/` or standalone `utils/` scripts unless the preset schema or `SampleIndex` payload changes.
- sleep2stat config/parser edits usually span `sleep2stat/config.py`, `configs/sleep2stat/`, `agent_tools/configs.py`, `utils/check_configs.py`, and `tests/test_sleep2stat_config.py`.
- sleep2stat output-contract edits usually span `sleep2stat/io/writers.py`, `sleep2stat/core/pipeline.py`, `sleep2stat/plot.py`, `agent_tools/plans.py`, and writer/CLI tests.
- sleep2stat metric edits should stay in the owning analyzer or reducer and update the focused test file: `tests/test_sleep2stat_analyzers.py`, `tests/test_sleep2stat_reducers.py`, or `tests/test_sleep2stat_writers.py`.
- Agent-facing workflow edits should stay in `agent_tools/`, `skills/`, `recipes/`, `agent_policies/`, or `doc/agent_contracts/`; do not add new model-training runtimes.
- `sleep2vec2` and `sleep2expert` edits should preserve package-local imports and mirror tests; do not route them through root `data` or `preprocess`.
- `sleep2expert` MoE edits usually span `sleep2expert/config.py`, `sleep2expert/backbones/roformer/moe.py`, `sleep2expert/pretrain_model.py`, `sleep2expert/sleep2vec_modelling.py`, `sleep2expert/sleep2vec_finetuning.py`, and `sleep2expert/losses/moe_regularization.py`; MoE LoRA trainability and LR-scale grouping belongs in `sleep2expert/sleep2vec_finetuning.py`.

## Ambiguities Worth Remembering

- `configs/cls_emb/` is not a guaranteed semantic truth source. At least one file in that folder still uses token downstream semantics.
- Config filenames are not authoritative task semantics. Inspect `finetune.task`, `model.cls`, and `preset_build` rather than inferring from the path.
- Head registration still depends on import side effects through downstream package imports. That behavior works now but is structurally implicit.
- AHI threshold fitting happens inside validation reduction, not in entrypoints. Checkpoint-specific threshold state is therefore part of the downstream runtime contract.
