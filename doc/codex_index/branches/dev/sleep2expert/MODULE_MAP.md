# Module Map

## Natural Edit Boundaries

These are the stable cross-file boundaries that matter before editing:

| Boundary | Primary files | Responsibility | Key dependencies | Extension points | Notes |
| --- | --- | --- | --- | --- | --- |
| Config schema and built-in task semantics | `sleep2vec/config.py`, `sleep2vec/common.py` | Parse YAML, validate required blocks, define built-in labels, bind config state into CLI namespaces, persist run artifacts | `yaml`, dataclasses | New config fields, new built-in tasks, new eval-visualization knobs, new adapt fields | Canonical parse and task-semantics boundary |
| Registry and builders | `sleep2vec/registry.py`, `sleep2vec/builders.py`, `sleep2vec/backbones/encoder_factory.py`, `sleep2vec/modules/`, `sleep2vec/cls/` | Register and instantiate backbones, tokenizers, projections, CLS strategies, and model averagers | Transformers, Torch | New registries or builder targets | Construction should flow through these helpers |
| Backbone model and adaptation freeze policy | `sleep2vec/pretrain_model.py` | Tokenize channels, add CLS, project to hidden size, run encoder, expose hidden states, and partition parameters for staged adaptation | builders, tokenizers, projection, CLS | New backbone behavior, tokenizer usage, adaptation grouping | Contains both canonical config-backed path and legacy hardcoded path |
| Downstream model and head composition | `sleep2vec/downstream_model.py`, `sleep2vec/downstreams/` | Temporal aggregation, channel fusion, downstream heads, layer mix, pretrained loading, LoRA insertion | `peft`, downstream registries | New heads, fusion modes, adapter behavior | Canonical downstream feature path |
| Lightning runtime | `sleep2vec/sleep2vec_modelling.py`, `sleep2vec/sleep2vec_finetuning.py`, `sleep2vec/sleep2vec_adaptation.py` | Training-loop glue, optimizer schedule, diagnostics, model averaging, AHI metrics, eval visualization logging, staged adaptation optimizer groups | PyTorch Lightning, loss/metrics, callbacks | New trainer behavior | Keep scheduler, loss, and epoch-reduction changes here |
| Runtime entrypoints | `sleep2vec/pretrain.py`, `sleep2vec/adapt.py`, `sleep2vec/finetune.py`, `sleep2vec/infer.py`; variant-local mirrors such as `sleep2expert/routing_analysis.py` | CLI parsing, experiment folder setup, trainer creation, phase validation, checkpoint/test orchestration, and variant-local routing export | `sleep2vec.common`, `sleep2vec.utils`, Lightning; package-local variant mirrors | New CLI flags, run naming, checkpoint policy, routing export policy | Thin orchestration only |
| Runtime support | `sleep2vec/checkpoints.py`, `sleep2vec/results.py`, `sleep2vec/distributed.py`, `sleep2vec/callbacks/`, `sleep2vec/visualization/`, `sleep2vec/diagnostics.py` | Checkpoint init/averaging, distributed rank helpers, result CSV writes, pair-accuracy monitoring, progress bars, plots, diagnostics hooks | Torch, sklearn, wandb, matplotlib | New logging/export surfaces | Mostly support code, not model semantics |
| Dataset core | `data/default_dataset.py`, `data/psg_pretrain_dataset.py` | Build `SampleIndex` windows, validate/load presets, materialize batches, backfill built-in AHI metadata, and choose samplers | `data.utils`, `data.metadata`, `data.samplers` | New dataset formats or collate behavior | Main data contract boundary |
| Data helpers and samplers | `data/utils.py`, `data/metadata.py`, `data/channel_selection.py`, `data/samplers.py`, `sleep2vec/utils.py` | NPZ I/O, token windowing, preset filtering, metadata tensorization, pair scheduling, bucketed batching, finetune/pretrain loader assembly | NumPy, Torch | New samplers, metadata encodings, loader policies | Missing-channel pretrain relies on these |
| Preprocessing CLIs | `preprocess/*.py` | CSV splitting, preset generation/merge, required-channel prefiltering, missing-mask stats, WatchPAT conversion | pandas, pickle, NumPy, optional EDF deps | New data-prep utilities | CLI tools are the canonical prep surface |
| Config-policy tooling | `utils/check_configs.py` | Validate repo YAMLs against loader contracts, tokenizer parity, preset-build strictness, and repo-specific ppg policy | base or variant-local config loaders, preset-build helpers | New static checks | Tooling boundary, not runtime |
| Config recipes | `configs/` | Encode model/head/task variants, preset-build policy, adapt recipes | Config parser + entrypoints | New recipe variants | Folder names are not always semantically authoritative |
| Tests | `tests/` | Pin config, registry, AHI metric, checkpoint, result CSV, preset-build, pair-sampler, and visualization contracts | pytest | New contract coverage | Many important contracts live here now |
| Formatting wrapper | `utils/style_check.sh` | Run `isort`, `black`, `flake8` over repo | Python env toolchain | None | Lint wrapper only |
| Variant namespaces | `sleep2vec2/`, `sleep2expert/`, `sleep2vec_moe/`, `sleep2vec_hires/` | `sleep2vec2` and `sleep2expert` are standalone mirror recipes with local data/preprocess/configs and standalone RoFormer; the other roots remain placeholders | variant-local copies, Torch, optional Transformers for parity tests | Variant-specific backbones and recipe-local data/preprocess flows | Keep variant imports package-local; do not route RoFormer through base `sleep2vec` or another variant |

## Key Dependencies

### External libraries used across boundaries

- PyTorch and PyTorch Lightning: core model and trainer runtime
- Hugging Face Transformers: backbone implementations via encoder factory
- PEFT: LoRA insertion in downstream model
- pandas / NumPy: CSV preprocessing, preset filtering, metrics, and result export
- matplotlib / seaborn / wandb / scikit-learn: evaluation visualization and experiment logging
- optional EDF stack in WatchPAT conversion: `pyedflib`, `scipy`, `tqdm`

### Internal dependency flow

- `pretrain.py` -> `config.py` + `common.py` + `sleep2vec.utils` + `sleep2vec_modelling.py` + for `sleep2expert`, `sleep2expert.model_stats`
- `adapt.py` -> `config.py` + `common.py` + `sleep2vec.utils` + `sleep2vec_adaptation.py`
- `finetune.py` / `infer.py` -> `common.py` -> `config.py` + `sleep2vec.utils` + `sleep2vec_finetuning.py`
- `sleep2expert/routing_analysis.py` -> `sleep2expert.common` + `sleep2expert.infer._build_inference_loader` + `Sleep2vecFinetuning._get_eval_model` + `Sleep2vecPretrainModel.last_moe_aux`
- `Sleep2vecPretrainModel` -> `builders.py` -> registries/backbone/tokenizer/projection/CLS
- `Sleep2vecDownstreamModel` -> `Sleep2vecPretrainModel` + `downstreams/` + optional `peft`
- `Sleep2vecFinetuning` -> `metrics.py` + `results.py` + `visualization/downstream_eval.py`
- `DefaultDataset.dataloader` -> `data.utils` + `data.metadata` + `data.samplers`
- `save_dataset_presets.py` -> `split_index_by_dataset.normalize_mask_frame` + `PSGPretrainDataset` -> `DefaultDataset`
- `utils/check_configs.py` -> path-selected config loaders + preset-build helpers from base or variant-local `save_dataset_presets.py`

## Important Ownership Notes

- Config/task edits usually span `sleep2vec/config.py` and `sleep2vec/common.py`.
- Forward-path edits usually span `sleep2vec/pretrain_model.py`, `sleep2vec/downstream_model.py`, and the downstream fusion modules.
- AHI runtime edits usually span `sleep2vec/common.py`, `sleep2vec/utils.py`, `data/default_dataset.py`, `data/utils.py`, `sleep2vec/metrics.py`, and `sleep2vec/sleep2vec_finetuning.py`.
- Missing-channel pretraining edits usually span `data/default_dataset.py`, `data/utils.py`, `data/samplers.py`, and `sleep2vec/utils.py`.
- Adaptation edits usually span `sleep2vec/adapt.py`, `sleep2vec/sleep2vec_adaptation.py`, `sleep2vec/pretrain_model.py`, and sampler/callback surfaces used by pair scheduling.
- Preprocessing edits should stay in `preprocess/` unless the preset schema or `SampleIndex` payload changes.
- `sleep2vec2` and `sleep2expert` edits should stay inside the matching package, matching `configs/<variant>/`, and matching tests unless a failing isolation/parity check proves a shared-base change is necessary.

## Ambiguities Worth Remembering

- `configs/cls_emb/` is not a guaranteed semantic truth source. At least one file in that folder still uses token downstream semantics.
- Config filenames are not authoritative task semantics. Inspect `finetune.task`, `model.cls`, and `preset_build` rather than inferring from the path.
- Head registration still depends on import side effects through downstream package imports. That behavior works now but is structurally implicit.
- AHI threshold fitting happens inside validation reduction, not in entrypoints. Checkpoint-specific threshold state is therefore part of the downstream runtime contract.
- `sleep2vec_moe/` and `sleep2vec_hires/` have no tracked source and no explicit owner in current branch metadata; ownership is `unknown`.
