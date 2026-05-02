# Config And Registries

## Config bundle families

- File: `sleep2vec/config.py`
- Dataclasses:
  - `ModelConfig`, `ChannelConfig`, `TokenizerConfig`, `BackboneConfig`, `ProjectionConfig`, `ClsConfig`
  - `HeadConfig`, `TemporalAggConfig`, `ChannelAggConfig`, `LayerMixConfig`
  - `TaskConfig`, `FinetuneConfig`, `FinetuneDataConfig`, `EvalVisualizationsConfig`
  - `AdaptConfig`, `AdaptStage1Config`, `AdaptStage2Config`, `AdaptLrScalesConfig`, `AdaptPairSchedulePoint`
  - `PretrainConfigBundle`, `FinetuneConfigBundle`
- Purpose and contract: define the typed schema used everywhere else in the runtime. These dataclasses are the canonical in-memory shape after YAML parsing.
- Reuse guidance: extend these dataclasses before adding new ad hoc dict plumbing in entrypoints or Lightning modules.

## `sleep2vec.config.load_model_config`

- File: `sleep2vec/config.py`
- Signature: `load_model_config(path: str | Path, *, require_head: bool = False) -> ModelConfig`
- Purpose and contract: load only the `model` block from YAML and return a typed `ModelConfig`.
- Important inputs/outputs: YAML path in, `ModelConfig` out.
- Side effects: reads YAML from disk only.
- Key callers/callees: callee chain is `_load_yaml_mapping` -> `_build_model_config`.
- Reuse guidance: use when a caller needs channel/backbone/head structure without full runtime config binding.
- Duplication risk notes: do not hand-parse `model.channels` in tooling.

## `sleep2vec.config.load_pretrain_config`

- File: `sleep2vec/config.py`
- Signature: `load_pretrain_config(path: str | Path) -> PretrainConfigBundle`
- Purpose and contract: parse pretrain YAML into typed dataclasses; require `model.backbone`, `model.projection`, and `model.cls`; build channels, loss config, data config, optional model-averaging config, and optional `adapt` config.
- Important inputs/outputs: filesystem path in, `PretrainConfigBundle` out.
- Side effects: reads YAML from disk only.
- Key callers/callees: callers are `sleep2vec.pretrain.sleep2vec_pretrain`, `sleep2vec.adapt.sleep2vec_adapt`, and `utils/check_configs.py` for base configs; variant-local mirrors are used by `utils/check_configs.py` for `configs/sleep2expert/**` and `configs/sleep2vec2/**`. Callees include `_require_channels`, `_build_model_config`, `_build_loss`, `_build_model_averaging_config`, `_build_adapt_config`, and `_validate_adapt_config`.
- Reuse guidance: reuse for every pretrain or adaptation code path that needs model semantics from YAML.
- Duplication risk notes: do not duplicate parse logic in `pretrain.py` or `adapt.py`; keep entrypoint code orchestration-only.

## `sleep2vec.config.load_finetune_config`

- File: `sleep2vec/config.py`
- Signature: `load_finetune_config(path: str | Path) -> FinetuneConfigBundle`
- Purpose and contract: parse finetune YAML, require `finetune` plus model `backbone/projection/cls/head`, validate task, layer-mix, and evaluation-visualization semantics, and return the typed bundle used by finetune and inference.
- Important inputs/outputs: YAML path in, `FinetuneConfigBundle` out.
- Side effects: reads YAML from disk only.
- Key callers/callees: caller is `sleep2vec.common.apply_finetune_config`; callees include `_build_layer_mix_config`, `_build_task_config`, `_build_eval_visualizations_config`, `_validate_layer_mix_config`, and `_build_model_averaging_config`.
- Reuse guidance: this is the canonical finetune schema boundary.
- Duplication risk notes: do not reimplement task/head/visualization parsing in trainers or CLI code.

## `sleep2vec.config.validate_model_config`

- File: `sleep2vec/config.py`
- Signature: `validate_model_config(model_cfg: ModelConfig) -> int`
- Purpose and contract: enforce cross-channel tokenizer output-dimension parity plus CLS/head option validity; returns the shared tokenizer feature dimension.
- Important inputs/outputs: `ModelConfig` in, shared feature dimension out.
- Side effects: none.
- Key callers/callees: callers are `sleep2vec.builders.build_tokenizers_and_dim` and `utils/check_configs.py` for base configs; variant-local mirrors are used by `utils/check_configs.py` for standalone variant configs.
- Reuse guidance: call this before constructing tokenizers or downstream heads from a `ModelConfig`.
- Duplication risk notes: tokenizer-dimension parity must not be rechecked differently elsewhere.

## `sleep2vec.common.apply_model_config_args`

- File: `sleep2vec/common.py`
- Signature: `apply_model_config_args(args, model_cfg: ModelConfig, *, set_backbone_arch: bool = False) -> None`
- Purpose and contract: copy channel names and channel input dimensions from a typed `ModelConfig` into an argparse namespace, optionally also setting `backbone_arch`.
- Important inputs/outputs: mutates `args` in place.
- Side effects: namespace mutation only.
- Key callers/callees: callers are `sleep2vec.pretrain.sleep2vec_pretrain` and `sleep2vec.adapt.sleep2vec_adapt`.
- Reuse guidance: use this for pretrain/adapt-style entrypoints instead of re-copying channel metadata manually.
- Duplication risk notes: channel-name normalization should remain centralized here.

## `sleep2vec.common.apply_task_flags`

- File: `sleep2vec/common.py`
- Signature: `apply_task_flags(args, task_cfg: TaskConfig | None = None) -> None`
- Purpose and contract: populate `args.output_dim`, `args.is_classification`, `args.is_seq`, `args.monitor`, `args.monitor_mod`, and built-in label metadata from either built-in task semantics or explicit `finetune.task`.
- Important inputs/outputs: mutates `argparse.Namespace` in place.
- Side effects: namespace mutation only.
- Key callers/callees: caller is `apply_finetune_config`; callees are `_validate_builtin_task_cfg`, `_validate_metadata_label_support`, and the built-in task helpers such as `get_task_label_source_name`, `get_task_label_merge_map`, and `get_task_auxiliary_label_source_names`.
- Reuse guidance: use for any finetune/infer code path that depends on label semantics.
- Duplication risk notes: built-in `stage3` / `stage4` / `stage5` / `ahi` behavior must remain centralized here.

## `sleep2vec.common.apply_finetune_config`

- File: `sleep2vec/common.py`
- Signature: `apply_finetune_config(args) -> tuple[Any, Any]`
- Purpose and contract: load finetune YAML, copy data/model/task/LoRA/eval-visualization settings into the CLI namespace, and enforce dataloader/model channel parity.
- Important inputs/outputs: mutates `args`; returns `(config_bundle, model_cfg)` for convenience.
- Side effects: converts configured data paths into `Path` objects and mutates many runtime flags.
- Key callers/callees: callers are `sleep2vec.finetune` and `sleep2vec.infer`; callee is `load_finetune_config`.
- Reuse guidance: this is the canonical CLI binding layer for finetune/inference.
- Duplication risk notes: do not partially mirror its behavior in entrypoints.

## `sleep2vec.common.persist_run_config_and_args`

- File: `sleep2vec/common.py`
- Signature: `persist_run_config_and_args(args: argparse.Namespace, exp_dir: Path, *, phase_name: str | None = None, write_root_files: bool = True) -> None`
- Purpose and contract: copy the config YAML plus YAML-serialized CLI args into an experiment directory, optionally also writing phase-scoped snapshots.
- Important inputs/outputs: namespace and experiment directory in; files on disk out.
- Side effects: creates directories, copies YAML, writes CLI snapshots.
- Key callers/callees: callers are `sleep2vec.adapt.sleep2vec_adapt` and `sleep2vec.finetune.supervised`; callees include `_copy_file` and `dump_cli_args_yaml`.
- Reuse guidance: use this helper for new run metadata snapshots instead of writing separate entrypoint-specific copy logic.
- Duplication risk notes: root vs phase-scoped persistence should stay centralized here.

## `sleep2vec.common.dump_cli_args_yaml`

- File: `sleep2vec/common.py`
- Signature: `dump_cli_args_yaml(args: argparse.Namespace, dest_path: Path) -> Path`
- Purpose and contract: persist CLI plus derived runtime arguments as YAML next to experiment artifacts.
- Important inputs/outputs: namespace in, YAML path out.
- Side effects: creates parent directories and writes a YAML file.
- Key callers/callees: used indirectly by `persist_run_config_and_args`; callee is `_to_yamlable`.
- Reuse guidance: use this for CLI snapshots rather than creating new serializers.
- Duplication risk notes: serialization rules for `Path`, dataclass, and argparse objects live here.

## `sleep2vec.builders.build_encoder_factory`, `build_tokenizers_and_dim`, `build_projection`

- File: `sleep2vec/builders.py`
- Signatures:
  - `build_encoder_factory(backbone_cfg: BackboneConfig) -> TransformerEncoderFactory`
  - `build_tokenizers_and_dim(model_cfg: ModelConfig, *, device: str = "cuda") -> tuple[dict[str, nn.Module], int]`
  - `build_projection(projection_cfg: ProjectionConfig, *, in_dim: int) -> nn.Module | None`
- Purpose and contract: resolve config-backed builders into concrete runtime modules without leaking registry details to callers.
- Important inputs/outputs: typed config blocks in, constructed objects out.
- Side effects: none beyond module instantiation.
- Key callers/callees: main caller is `Sleep2vecPretrainModel.__init__`; callees are registry lookups, `validate_model_config`, `build_tokenizer_mapping`, and `build_projection_head`.
- Reuse guidance: use these helpers any time code starts from `ModelConfig`.
- Duplication risk notes: avoid hand-instantiating tokenizers or backbones in new code.

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
