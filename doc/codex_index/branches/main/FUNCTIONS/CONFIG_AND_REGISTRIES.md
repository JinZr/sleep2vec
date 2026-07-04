# Config And Registries

## Config bundle families

- File: `sleep2vec/config.py`
- Dataclasses:
  - `ModelConfig`, `ChannelConfig`, `TokenizerConfig`, `BackboneConfig`, `ProjectionConfig`, `ClsConfig`
  - `HeadConfig`, `TemporalAggConfig`, `ChannelAggConfig`, `LayerMixConfig`
  - `TaskConfig`, `FinetuneConfig`, `SurvivalConfig`, `MultilabelConfig`, `LoraConfig`, `FinetuneLossConfig`, `FinetuneSamplerConfig`, `FinetuneDataConfig`, `EvalVisualizationsConfig`
  - `AdaptConfig`, `AdaptStage1Config`, `AdaptStage2Config`, `AdaptLrScalesConfig`, `AdaptPairSchedulePoint`
  - `PretrainDataConfig` with `backend`, `kaldi_data_root`, and `kaldi_manifest`
  - `PretrainConfigBundle`, `FinetuneConfigBundle`
- Purpose and contract: define the typed schema used everywhere else in the runtime. These dataclasses are the canonical in-memory shape after YAML parsing; `ChannelConfig.aliases` carries optional YAML-declared NPZ input-key fallbacks.
- Reuse guidance: extend these dataclasses before adding new ad hoc dict plumbing in entrypoints or Lightning modules.

## `sleep2vec.config.LoraConfig` and package-local variant mirrors

- File: `sleep2vec/config.py`, `sleep2vec2/config.py`, `sleep2expert/config.py`
- Signature: dataclass with `freeze_backbone_and_insert_lora`, `insert_lora`, `separate_adapters`, `r`, `alpha`, `dropout`, `target_modules`, and `use_dora`.
- Purpose and contract: carry downstream adapter policy from YAML through config parsing, including LoRA rank/alpha/dropout/target-module settings and the DoRA toggle. `sleep2expert` rejects router target modules at config load time.
- Important inputs/outputs: YAML `finetune.lora` mapping in; typed config fields consumed by `apply_finetune_config`.
- Side effects: none.
- Key callers/callees: built by `load_finetune_config`; consumed by `sleep2vec.common.apply_finetune_config` and package-local variant mirrors.
- Reuse guidance: add downstream adapter schema fields to the canonical root contract first, then mirror the same shape in standalone namespaces without adding shared helpers.
- Duplication risk notes: do not add separate CLI-only adapter hyperparameters that bypass this config object.

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
- Purpose and contract: parse pretrain YAML into typed dataclasses; require `model.backbone`, `model.projection`, and `model.cls`; build channels, loss config, data config including `npz`/`kaldi` backend fields, optional model-averaging config, and optional `adapt` config.
- Important inputs/outputs: filesystem path in, `PretrainConfigBundle` out.
- Side effects: reads YAML from disk only.
- Key callers/callees: callers are `sleep2vec.pretrain.sleep2vec_pretrain`, `sleep2vec.adapt.sleep2vec_adapt`, and `utils/check_configs.py`; callees include `_require_channels`, `_build_model_config`, `_build_loss`, `_build_model_averaging_config`, `_build_adapt_config`, and `_validate_adapt_config`.
- Reuse guidance: reuse for every pretrain or adaptation code path that needs model semantics from YAML.
- Duplication risk notes: do not duplicate parse logic in `pretrain.py` or `adapt.py`; keep entrypoint code orchestration-only.

## `sleep2vec.config.SurvivalConfig`

- File: `sleep2vec/config.py`
- Signature: dataclass with `key_column`, `disease_columns_index`, `event_time_index`, `is_event_index`, and `has_label_index`.
- Purpose and contract: carry Cox survival sidecar paths from YAML into the runtime, with one key column and one disease-column list shared across event-time, event-status, and label-mask CSVs.
- Important inputs/outputs: YAML `finetune.survival` mapping in; typed config consumed by `apply_finetune_config`, dataset constructors, preset generation, and Kaldi conversion.
- Side effects: none.
- Key callers/callees: built by `load_finetune_config`; consumed by `sleep2vec.common.apply_finetune_config`, `data.survival.load_survival_label_table`, and preprocessing helpers.
- Reuse guidance: add survival label-source fields here before changing loaders or trainers.
- Duplication risk notes: do not add parallel disease-column lists or alternate sidecar spellings.

## `sleep2vec.config.MultilabelConfig`

- File: `sleep2vec/config.py`
- Signature: dataclass with `key_column`, `disease_columns_index`, `label_index`, `has_label_index`, `covariates`, and `covariate_embedding_dim`.
- Purpose and contract: carry subject-level multilabel disease-detection sidecar paths from YAML into runtime and preset generation. The task is non-sequence only and uses one disease-column list shared by label and label-mask CSVs.
- Important inputs/outputs: YAML `finetune.multilabel` mapping in; typed config consumed by `apply_finetune_config`, dataset constructors, and preset generation.
- Side effects: none.
- Key callers/callees: built by `load_finetune_config`; consumed by `sleep2vec.common.apply_finetune_config`, `data.multilabel.load_multilabel_label_table`, and preprocessing helpers.
- Reuse guidance: add multilabel label-source fields here before changing loaders or trainers.
- Duplication risk notes: do not store disease labels as ordinary wide metadata columns or introduce alternate disease-column sidecar names.

## `sleep2vec.config.load_finetune_config`

- File: `sleep2vec/config.py`
- Signature: `load_finetune_config(path: str | Path) -> FinetuneConfigBundle`
- Purpose and contract: parse finetune YAML, require `finetune` plus model `backbone/projection/cls/head`, validate task, survival sidecars, multilabel sidecars, layer-mix, LoRA/DoRA settings, imbalance loss/sampler, data backend, and evaluation-visualization semantics, and return the typed bundle used by finetune and inference.
- Important inputs/outputs: YAML path in, `FinetuneConfigBundle` out.
- Side effects: reads YAML from disk only.
- Key callers/callees: caller is `sleep2vec.common.apply_finetune_config`; callees include `_build_layer_mix_config`, `_build_finetune_loss_config`, `_build_finetune_sampler_config`, `_build_task_config`, `_build_eval_visualizations_config`, `_validate_layer_mix_config`, `_build_model_averaging_config`, `SurvivalConfig`, and `MultilabelConfig`.
- Reuse guidance: this is the canonical finetune schema boundary.
- Duplication risk notes: do not reimplement task/head/visualization parsing in trainers or CLI code.

## `sleep2vec.config.validate_model_config`

- File: `sleep2vec/config.py`
- Signature: `validate_model_config(model_cfg: ModelConfig) -> int`
- Purpose and contract: enforce cross-channel tokenizer output-dimension parity plus CLS/head option validity, including allowed temporal aggregators `mean`, `attn`, and `lstm`; returns the shared tokenizer feature dimension.
- Important inputs/outputs: `ModelConfig` in, shared feature dimension out.
- Side effects: none.
- Key callers/callees: callers are `sleep2vec.builders.build_tokenizers_and_dim` and `utils/check_configs.py`.
- Reuse guidance: call this before constructing tokenizers or downstream heads from a `ModelConfig`.
- Duplication risk notes: tokenizer-dimension parity must not be rechecked differently elsewhere.

## `sleep2vec.common.apply_model_config_args`

- File: `sleep2vec/common.py`
- Signature: `apply_model_config_args(args, model_cfg: ModelConfig, *, set_backbone_arch: bool = False) -> None`
- Purpose and contract: copy channel names, channel input dimensions, and YAML-declared channel aliases from a typed `ModelConfig` into an argparse namespace, optionally also setting `backbone_arch`.
- Important inputs/outputs: mutates `args` in place.
- Side effects: namespace mutation only.
- Key callers/callees: callers are `sleep2vec.pretrain.sleep2vec_pretrain` and `sleep2vec.adapt.sleep2vec_adapt`.
- Reuse guidance: use this for pretrain/adapt-style entrypoints instead of re-copying channel metadata manually.
- Duplication risk notes: channel-name normalization should remain centralized here.

## `sleep2vec.common.apply_task_flags`

- File: `sleep2vec/common.py`
- Signature: `apply_task_flags(args, task_cfg: TaskConfig | None = None) -> None`
- Purpose and contract: populate `args.output_dim`, `args.is_classification`, `args.is_multilabel`, `args.is_survival`, `args.is_seq`, `args.monitor`, `args.monitor_mod`, and built-in label metadata from either built-in task semantics or explicit `finetune.task`; built-in classification recipes may monitor `val_accuracy`, `val_f1_macro`, `val_f1_weighted`, `val_cohen_kappa`, and binary `val_roc_auc`, while survival tasks may monitor `val_loss/min` or `val_c_index/max`.
- Important inputs/outputs: mutates `argparse.Namespace` in place.
- Side effects: namespace mutation only.
- Key callers/callees: caller is `apply_finetune_config`; callees are `_validate_builtin_task_cfg`, `_validate_metadata_label_support`, and the built-in task helpers such as `get_task_label_source_name`, `get_task_label_merge_map`, and `get_task_auxiliary_label_source_names`.
- Reuse guidance: use for any finetune/infer code path that depends on label semantics.
- Duplication risk notes: built-in `stage3` / `stage4` / `stage5` / `ahi` behavior must remain centralized here.

## `sleep2vec.common.apply_finetune_config`

- File: `sleep2vec/common.py`
- Signature: `apply_finetune_config(args) -> tuple[Any, Any]`
- Purpose and contract: load finetune YAML, copy data/model/task/survival/multilabel/imbalance/LoRA/DoRA/eval-visualization settings into the CLI namespace, apply data-backend settings, and enforce dataloader/model channel parity.
- Important inputs/outputs: mutates `args`, including `args.lora_r`, `args.lora_alpha`, `args.lora_dropout`, `args.lora_target_modules`, and `args.lora_use_dora`; returns `(config_bundle, model_cfg)` for convenience.
- Side effects: converts configured data paths into `Path` objects and mutates many runtime flags.
- Key callers/callees: callers are `sleep2vec.finetune` and `sleep2vec.infer`; callees are `load_finetune_config`, `apply_task_flags`, and `_validate_and_apply_imbalance_config`.
- Reuse guidance: this is the canonical CLI binding layer for finetune/inference.
- Duplication risk notes: do not partially mirror its behavior in entrypoints.

## `sleep2vec.common.apply_data_backend_args`

- File: `sleep2vec/common.py`
- Signature: `apply_data_backend_args(args, data_cfg, *, preset_attr: str | None = None) -> None`
- Purpose and contract: normalize `args.data_backend` from CLI or YAML, convert Kaldi paths to `Path`, require `kaldi_data_root` and `kaldi_manifest` for Kaldi runs, and reject legacy preset pickle paths when the backend is Kaldi.
- Important inputs/outputs: CLI namespace, typed data config, and optional preset attribute name in; mutates `args`.
- Side effects: namespace mutation only.
- Key callers/callees: callers are `sleep2vec.pretrain`, `sleep2vec.adapt`, and `apply_finetune_config`; callee is `_optional_path`.
- Reuse guidance: use this helper before constructing loaders for any runtime path that supports `data.backend`.
- Duplication risk notes: do not repeat Kaldi path and preset incompatibility checks in entrypoints.

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

## `sleep2expert.config.MoeConfig` and `_validate_moe_config`

- File: `sleep2expert/config.py`
- Signature: `_validate_moe_config(moe_cfg: MoeConfig, backbone_cfg: BackboneConfig, channel_names: Sequence[str] | None = None) -> None`
- Purpose and contract: define and validate the sleep2expert MoE backbone schema, including enabled layers, expert count, `top_k`, router type, regularization coefficients, modality group masks, expert groups, modality-to-group mapping, and route-consistency layers.
- Important inputs/outputs: typed MoE/backbone configs and optional channel names in; raises on invalid schema, otherwise returns `None`.
- Side effects: none.
- Key callers/callees: caller is `_build_backbone_config`; callees include `_validate_moe_int_list` and `_validate_moe_nonnegative_number`.
- Reuse guidance: add MoE backbone config semantics here before touching routers or trainers.
- Duplication risk notes: `model.backbone.config_overrides.moe` is intentionally rejected; MoE belongs under `model.backbone.moe`.

## `sleep2expert.config._build_finetune_moe_tuning_config`

- File: `sleep2expert/config.py`
- Signature: `_build_finetune_moe_tuning_config(raw: Any) -> FinetuneMoeTuningConfig | None`
- Purpose and contract: parse finetune-time MoE tuning policy, including `head_only`, conservative full-router modes, `top_moe_layer_expert_only`, custom freeze flags, LR scales, LoRA adapter LR scale, and downstream MoE regularization.
- Important inputs/outputs: raw YAML block in; typed `FinetuneMoeTuningConfig` or `None` out.
- Side effects: none, except `_validate_finetune_moe_tuning_config` may fill a default top MoE layer when `top_moe_layer_expert_only` omits `train_moe_layer_indices`.
- Key callers/callees: caller is `load_finetune_config`; callees include `_build_finetune_lr_scales_config`, `_build_finetune_moe_regularization_config`, and `_validate_finetune_moe_tuning_config`.
- Reuse guidance: use this path for sleep2expert MoE finetune recipes instead of adding CLI-only tuning switches; `finetune.moe_tuning.lr_scales.lora` controls adapter trainability and optimizer LR scale.
- Duplication risk notes: downstream load-balance, modality-balance, route-consistency, and entropy regularization are currently rejected for finetune; keep that unsupported subset centralized here.

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
