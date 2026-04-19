# Config And Registries

## `sleep2vec.config.load_pretrain_config`

- File: `sleep2vec/config.py`
- Signature: `load_pretrain_config(path: str | Path) -> PretrainConfigBundle`
- Purpose and contract: Parse pretrain YAML into typed dataclasses; require `model.backbone`, `model.projection`, and `model.cls`; build channels, loss config, data config, and optional model-averaging config.
- Important inputs/outputs: input is a filesystem path to YAML; output is a `PretrainConfigBundle`.
- Side effects: reads YAML from disk only.
- Key callers/callees: caller is `sleep2vec.pretrain.sleep2vec_pretrain`; callees include `_require_channels`, `_build_cls_config`, `_build_head_config`, `_build_loss`, `_build_model_averaging_config`.
- Reuse guidance: reuse for every pretrain code path that needs model semantics from YAML.
- Duplication risk notes: do not duplicate parse logic in `pretrain.py`; keep entrypoint code orchestration-only.

## `sleep2vec.config.load_finetune_config`

- File: `sleep2vec/config.py`
- Signature: `load_finetune_config(path: str | Path) -> FinetuneConfigBundle`
- Purpose and contract: Parse finetune YAML, require `finetune` block plus model `backbone/projection/cls/head`, validate task and layer-mix semantics, and return the typed bundle used by finetune and inference.
- Important inputs/outputs: YAML path in, `FinetuneConfigBundle` out.
- Side effects: reads YAML from disk only.
- Key callers/callees: caller is `sleep2vec.common.apply_finetune_config`; callees include `_require_channels`, `_build_cls_config`, `_build_head_config`, `_build_layer_mix_config`, `_build_task_config`, `_validate_layer_mix_config`, `_build_model_averaging_config`.
- Reuse guidance: this is the canonical finetune schema boundary.
- Duplication risk notes: do not reimplement task/head parsing in trainers or CLI code.

## `sleep2vec.config.validate_model_config`

- File: `sleep2vec/config.py`
- Signature: `validate_model_config(model_cfg: ModelConfig) -> int`
- Purpose and contract: Enforce cross-channel tokenizer output-dimension parity and validate CLS/head options; returns the shared tokenizer feature dimension.
- Important inputs/outputs: `ModelConfig` in, shared feature dimension out.
- Side effects: none.
- Key callers/callees: caller is `sleep2vec.builders.build_tokenizers_and_dim`.
- Reuse guidance: call this before constructing tokenizers or downstream heads from a `ModelConfig`.
- Duplication risk notes: tokenizer dimension parity must not be rechecked differently elsewhere.

## Built-in task helper family

- File: `sleep2vec/common.py`
- Functions:
  - `is_builtin_seq_task(label_name: str | None) -> bool`
  - `is_builtin_stage_task(label_name: str | None) -> bool`
  - `get_task_label_source_name(label_name: str) -> str`
  - `get_task_stage_names(label_name: str) -> list[str] | None`
  - `get_task_class_labels(label_name: str) -> list[str] | None`
  - `get_task_label_merge_map(label_name: str) -> dict[int, int] | None`
  - `get_task_is_multilabel(label_name: str) -> bool`
  - `get_task_auxiliary_label_source_names(label_name: str) -> list[str]`
  - `remap_stage_labels(labels, label_name: str)`
- Purpose and contract: expose the normalized built-in task spec and keep sleep-stage remapping separate from raw `ahi` sequence labels. Built-in `ahi` additionally declares `stage5` as an auxiliary runtime label source so final event metrics can apply sleep-stage masking without changing the primary target path.
- Important inputs/outputs: built-in label name in; canonical task attributes or remapped labels out.
- Side effects: none.
- Key callers/callees: callers are `apply_task_flags`, `sleep2vec.utils._build_finetune_loader`, and `Sleep2vecFinetuning._get_targets`.
- Reuse guidance: use these helpers instead of hardcoding task-specific label sources or merge maps.
- Duplication risk notes: `ahi` and sleep-stage semantics must stay centralized here; do not invent a second built-in-task map elsewhere or wire `stage5` for AHI only in loader code.

## `sleep2vec.common.apply_task_flags`

- File: `sleep2vec/common.py`
- Signature: `apply_task_flags(args, task_cfg: TaskConfig | None = None) -> None`
- Purpose and contract: Populate `args.output_dim`, `args.is_classification`, `args.is_seq`, `args.is_multilabel`, `args.monitor`, `args.monitor_mod`, `args.label_source_name`, `args.stage_names`, `args.class_labels`, `args.label_merge_map`, and `args.auxiliary_label_source_names` from either built-in label semantics or explicit finetune task config. Built-in `ahi` still enforces `type=classification`, `output_dim=30`, and `is_seq=true`, and now accepts only the monitor names that the runtime actually emits stably: full validation keeps `val_ahi_pearson/max`, while lightweight validation supports `val_loss/min` plus pointwise `accuracy/precision/recall/f1` with `max`.
- Important inputs/outputs: mutates `argparse.Namespace` in place.
- Side effects: namespace mutation only.
- Key callers/callees: caller is `apply_finetune_config`; callees are `_validate_builtin_task_cfg`, `_validate_metadata_label_support`, and the built-in task-spec helpers.
- Reuse guidance: use for any finetune/infer code path that depends on label semantics.
- Duplication risk notes: built-in task semantics must remain centralized here; `ahi` and sleep-staging should not grow separate task parsers, and the AHI validation-mode switch should stay derived from the existing YAML task monitor rather than a second side channel.

## `sleep2vec.common.apply_finetune_config`

- File: `sleep2vec/common.py`
- Signature: `apply_finetune_config(args) -> tuple[Any, Any]`
- Purpose and contract: Load finetune YAML, copy data/model/task/LoRA settings into the CLI namespace, enforce dataloader/model channel parity, and derive all built-in task attributes before loaders are built.
- Important inputs/outputs: mutates `args`; returns `(config_bundle, model_cfg)` for convenience.
- Side effects: converts configured data paths into `Path` objects and mutates many runtime flags.
- Key callers/callees: callers are `sleep2vec.finetune` and `sleep2vec.infer`; callees are `load_finetune_config`, `apply_model_config_args`, and `apply_task_flags`.
- Reuse guidance: this is the canonical CLI binding layer for finetune and inference.
- Duplication risk notes: do not partially mirror its behavior in entrypoints.

## `sleep2vec.common.dump_cli_args_yaml`

- File: `sleep2vec/common.py`
- Signature: `dump_cli_args_yaml(args: argparse.Namespace, dest_path: Path) -> Path`
- Purpose and contract: Persist CLI plus derived runtime arguments as YAML next to experiment artifacts.
- Important inputs/outputs: namespace in, YAML path out.
- Side effects: creates parent directories and writes a YAML file.
- Key callers/callees: callers are `sleep2vec.pretrain.sleep2vec_pretrain`, `sleep2vec.finetune.supervised`, and `persist_run_config_and_args`.
- Reuse guidance: use this for run metadata snapshots instead of creating new serializers.
- Duplication risk notes: artifact-writing flows are duplicated at the entrypoint level, but this serializer is the canonical primitive.

## `sleep2vec.builders.build_encoder_factory`, `build_tokenizers_and_dim`, `build_projection`

- File: `sleep2vec/builders.py`
- Signatures:
  - `build_encoder_factory(backbone_cfg: BackboneConfig) -> TransformerEncoderFactory`
  - `build_tokenizers_and_dim(model_cfg: ModelConfig, *, device: str = "cuda") -> tuple[dict[str, nn.Module], int]`
  - `build_projection(projection_cfg: ProjectionConfig, *, in_dim: int) -> nn.Module | None`
- Purpose and contract: Resolve config-backed builders into concrete runtime modules without leaking registry details to callers.
- Important inputs/outputs: typed config blocks in, constructed objects out.
- Side effects: none beyond module instantiation.
- Key callers/callees: main caller is `Sleep2vecPretrainModel.__init__`; callees are registry lookups, `validate_model_config`, `build_tokenizer_mapping`, and `build_projection_head`.
- Reuse guidance: use these helpers any time code starts from `ModelConfig`.
- Duplication risk notes: avoid hand-instantiating tokenizers/backbones in new code.

## Registry decorator family

- File: `sleep2vec/registry.py`
- Functions:
  - `register_backbone(name: str)`
  - `register_tokenizer(name: str)`
  - `register_projection(name: str)`
  - `register_model_averager(name: str)`
- Purpose and contract: return decorators that insert implementations into the relevant registry and reject duplicate names.
- Important inputs/outputs: registry name in, decorator out.
- Side effects: mutate global registries.
- Key callers/callees: used by `encoder_factory.py`, `modules/tokenizers.py`, `modules/projection.py`, and `averagings/*`.
- Reuse guidance: use registry decorators for new pluggable implementations rather than adding `if/elif` branches in runtime code.
- Duplication risk notes: each registry has identical semantics; do not create ad hoc side registries unless there is a real new extension dimension.

## Registry getter family

- File: `sleep2vec/registry.py`
- Functions:
  - `get_backbone_builder(name: str) -> BackboneBuilder`
  - `get_tokenizer_builder(name: str) -> TokenizerBuilder`
  - `get_projection_builder(name: str) -> ProjectionBuilder`
  - `get_model_averager_builder(name: str) -> ModelAveragerBuilder`
- Purpose and contract: look up registered implementations and fail fast with the available names listed.
- Important inputs/outputs: symbolic name in, callable builder out.
- Side effects: none.
- Key callers/callees: callers are `builders.py`, `modules/projection.py`, `modules/tokenizers.py`, and `averagings/base.py`.
- Reuse guidance: use getters instead of reading registry dictionaries directly.
- Duplication risk notes: direct dictionary reads bypass consistent error messages and make future registry refactors harder.

## `sleep2vec.backbones.encoder_factory.TransformerEncoderFactory.build`

- File: `sleep2vec/backbones/encoder_factory.py`
- Signature: `build(self) -> tuple[nn.Module, int]`
- Purpose and contract: instantiate the configured transformer encoder and return both the module and its hidden size.
- Important inputs/outputs: no runtime inputs; output is `(encoder, hidden_size)`.
- Side effects: constructs a fresh encoder instance.
- Key callers/callees: caller is `Sleep2vecPretrainModel.__init__`.
- Reuse guidance: treat the factory, not raw Hugging Face configs, as the backbone abstraction boundary.
- Duplication risk notes: hidden-size inference should not be reimplemented elsewhere.

## Backbone builder family

- File: `sleep2vec/backbones/encoder_factory.py`
- Functions:
  - `build_roformer(cfg: BackboneConfig) -> TransformerEncoderFactory`
  - `build_hf_bert(cfg: BackboneConfig) -> TransformerEncoderFactory`
- Purpose and contract: register supported backbone families and translate `BackboneConfig` into factory objects.
- Important inputs/outputs: `BackboneConfig` in, `TransformerEncoderFactory` out.
- Side effects: none beyond object instantiation.
- Key callers/callees: resolved through `build_encoder_factory`.
- Reuse guidance: add new backbones here through registry decoration.
- Duplication risk notes: this is the canonical place to map YAML names to concrete encoder families.

## `sleep2vec.modules.tokenizers.build_tokenizer_from_channel` and `build_tokenizer_mapping`

- File: `sleep2vec/modules/tokenizers.py`
- Signatures:
  - `build_tokenizer_from_channel(channel: ChannelConfig, *, device: str = "cuda") -> nn.Module`
  - `build_tokenizer_mapping(channels: list[ChannelConfig], *, device: str = "cuda") -> dict[str, nn.Module]`
- Purpose and contract: instantiate per-channel tokenizer modules from config and guarantee `tokenizer.out_dim` is present.
- Important inputs/outputs: channel config(s) in, module(s) out.
- Side effects: module instantiation only.
- Key callers/callees: caller is `build_tokenizers_and_dim`; callee is `get_tokenizer_builder`.
- Reuse guidance: use this instead of hardcoding tokenizer maps.
- Duplication risk notes: `Sleep2vecPretrainModel` still contains a legacy hardcoded mapping when `model_config` is absent; avoid extending that path.

## `sleep2vec.modules.projection.build_projection_head`

- File: `sleep2vec/modules/projection.py`
- Signature: `build_projection_head(cfg: ProjectionConfig, *, in_dim: int) -> nn.Module | None`
- Purpose and contract: create a registered projection head or return `None` when projection is disabled.
- Important inputs/outputs: projection config and input dimension in, module or `None` out.
- Side effects: module instantiation only.
- Key callers/callees: caller is `builders.build_projection`; callee is `get_projection_builder`.
- Reuse guidance: use this for all projection-head construction.
- Duplication risk notes: disabling projection should stay config-driven here.

## `sleep2vec.cls.factory.build_cls_embedding`

- File: `sleep2vec/cls/factory.py`
- Signature: `build_cls_embedding(strategy: str | None, hidden_size: int | None, **kwargs: Any) -> ClsEmbedding`
- Purpose and contract: resolve CLS behavior to `NoClsEmbedding` or `BertClsEmbedding` and enforce `hidden_size` when required.
- Important inputs/outputs: CLS strategy and hidden size in, strategy object out.
- Side effects: module instantiation only.
- Key callers/callees: caller is `Sleep2vecPretrainModel.__init__`.
- Reuse guidance: all CLS behavior should flow through this factory.
- Duplication risk notes: do not branch on CLS strategies inside model code beyond calling the strategy interface.
