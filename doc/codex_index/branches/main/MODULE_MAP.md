# Module Map

## Natural Edit Boundaries

These are the stable cross-file boundaries that matter before editing:

| Boundary | Primary files | Responsibility | Key dependencies | Extension points | Notes |
| --- | --- | --- | --- | --- | --- |
| Config schema | `sleep2vec/config.py` | Parse YAML, validate required blocks, task semantics, head/layer-mix/model-averaging schema | `yaml`, dataclasses | New config fields, new task semantics | Canonical parse boundary |
| Finetune CLI binding | `sleep2vec/common.py` | Copy finetune YAML state into CLI namespace, enforce built-in task rules | `sleep2vec.config` | New built-in tasks, CLI serialization | Do not parse finetune YAML directly in entrypoints |
| Registry and builders | `sleep2vec/registry.py`, `sleep2vec/builders.py`, `sleep2vec/backbones/encoder_factory.py`, `sleep2vec/modules/`, `sleep2vec/cls/` | Register and instantiate backbones, tokenizers, projections, CLS strategies, model averagers | Transformers, Torch | New registries or builder targets | Construction should flow through these helpers |
| Backbone model | `sleep2vec/pretrain_model.py` | Tokenize channels, add CLS, project to hidden size, run encoder, emit contrastive or encoded views | builders, tokenizers, projection, CLS | New backbone behavior, tokenizer usage | Contains both canonical config-backed path and legacy hardcoded path |
| Downstream model | `sleep2vec/downstream_model.py`, `sleep2vec/downstreams/` | Temporal aggregation, channel fusion, downstream heads, layer mix, pretrained loading, LoRA insertion | `peft`, downstream registries | New heads or fusion modes | Canonical downstream feature path |
| Lightning runtime | `sleep2vec/sleep2vec_modelling.py`, `sleep2vec/sleep2vec_finetuning.py` | Training loop glue, optimizer schedule, diagnostics, model averaging, metric logging | PyTorch Lightning, loss/metrics, callbacks | New trainer behavior | Keep scheduler/loss/logging changes here |
| Runtime entrypoints | `sleep2vec/pretrain.py`, `sleep2vec/finetune.py`, `sleep2vec/infer.py` | CLI parsing, experiment folder setup, trainer creation, checkpoint/test orchestration | `sleep2vec.common`, `sleep2vec.utils`, Lightning | New CLI flags, run naming, checkpoint policy | Thin orchestration only |
| Runtime utilities | `sleep2vec/checkpoints.py`, `sleep2vec/metrics.py`, `sleep2vec/diagnostics.py`, `sleep2vec/callbacks/pair_acc_logger.py`, `sleep2vec/visualization/` | Checkpoint averaging, metrics, diagnostics hooks, pair accuracy logging, visualization artifacts | Torch, sklearn, wandb, matplotlib | New logging/export surfaces | Mostly runtime support, not model semantics |
| Dataset core | `data/default_dataset.py`, `data/psg_pretrain_dataset.py` | Build `SampleIndex` windows, validate/load presets, collate batches, materialize metadata, pick sampler | `data.utils`, `data.metadata`, `data.samplers` | New dataset formats or collate behavior | Main data contract boundary |
| Data helpers | `data/utils.py`, `data/metadata.py`, `data/channel_selection.py`, `data/samplers.py` | NPZ I/O, token windowing, metadata tensorization, pair scheduling, distributed-aware sampling | NumPy, Torch | New samplers, metadata encodings | Missing-channel pretrain relies on these |
| Preprocessing CLIs | `preprocess/*.py` | CSV splitting, preset generation/merge, missing-mask stats, WatchPAT conversion | pandas, pickle, NumPy, optional EDF deps | New data-prep utilities | CLI tools are the canonical prep surface |
| Config recipes | `configs/` | Encode model/head/task variants | Config parser + entrypoints | New recipe variants | Folder names are not always semantically authoritative |
| Tests | `tests/` | Pin config, registry, loss, checkpoint, pair-sampler, and visualization contracts | pytest | New contract coverage | Some important tests are not in CI |
| Formatting wrapper | `utils/style_check.sh` | Run `isort`, `black`, `flake8` over repo | Python env toolchain | None | Lint wrapper only |
| Variant namespaces | `sleep2vec2/`, `sleep2vec_moe/`, `sleep2vec_hires/` | Branch-state placeholders on `main` | unknown | unknown | No tracked source files on this branch |

## Key Dependencies

### External libraries used across boundaries

- PyTorch and PyTorch Lightning: core model and trainer runtime
- Hugging Face Transformers: backbone implementations via encoder factory
- PEFT: LoRA insertion in downstream model
- pandas / NumPy: CSV preprocessing and metrics
- matplotlib / seaborn / wandb: visualization and experiment logging
- optional EDF stack in WatchPAT conversion: `pyedflib`, `scipy`, `tqdm`

### Internal dependency flow

- `pretrain.py` -> `config.py` + `sleep2vec.utils` + `sleep2vec_modelling.py`
- `finetune.py` / `infer.py` -> `common.py` -> `config.py` + `sleep2vec.utils` + `sleep2vec_finetuning.py`
- `Sleep2vecPretrainModel` -> `builders.py` -> registries/backbone/tokenizer/projection/CLS
- `Sleep2vecDownstreamModel` -> `Sleep2vecPretrainModel` + `downstreams/` + optional `peft`
- `DefaultDataset.dataloader` -> `data.utils` + `data.metadata` + `data.samplers`
- `save_dataset_presets.py` -> `PSGPretrainDataset` -> `DefaultDataset`

## Important Ownership Notes

- Config/task edits usually span `sleep2vec/config.py` and `sleep2vec/common.py`.
- Forward-path edits usually span `sleep2vec/pretrain_model.py`, `sleep2vec/downstream_model.py`, and the downstream fusion modules.
- Missing-channel pretraining edits usually span `data/default_dataset.py`, `data/utils.py`, `data/samplers.py`, and `sleep2vec/utils.py`.
- Preprocessing edits should stay in `preprocess/` unless the preset schema or `SampleIndex` payload changes.
- Variant edits cannot be completed on this branch without first adding tracked source files to the variant directories.

## Ambiguities Worth Remembering

- `configs/cls_emb/` is not a guaranteed semantic truth source. At least one file in that folder still uses token downstream semantics.
- Head registration depends on import side effects through the downstream package imports. That behavior works now but is structurally implicit.
- `sleep2vec_hires/` has no tracked source and no explicit owner in current branch metadata; ownership is `unknown`.
