# Functions: Config And Registry

## `sleep2vec.config.load_pretrain_config`

- File: `sleep2vec/config.py`
- Signature: `load_pretrain_config(path: str | Path) -> PretrainConfigBundle`
- Purpose and contract: parses top-level YAML for pretraining or adaptation, requires `model` and `loss`, accepts optional `model_averaging` and `adapt`, and validates adaptation semantics against `model.channels`.
- Important inputs: YAML path.
- Important outputs: `PretrainConfigBundle` containing `model`, `loss`, `data`, optional `averaging`, optional `adapt`.
- Side effects: reads YAML from disk.
- Notable callers/callees: called by `sleep2vec/pretrain.py` and `sleep2vec/adapt.py`; delegates to `_build_model_config`, `_build_loss`, `_build_adapt_config`, `_validate_adapt_config`.
- Reuse guidance: use this for any new pretrain-like runtime instead of reading YAML directly.
- Duplication-risk notes: high; a second pretrain parser will drift from adaptation validation quickly.

## `sleep2vec.config.load_finetune_config`

- File: `sleep2vec/config.py`
- Signature: `load_finetune_config(path: str | Path) -> FinetuneConfigBundle`
- Purpose and contract: parses downstream YAML, requires `finetune` and `model.head`, validates task config and optional layer-mix config.
- Important inputs: YAML path.
- Important outputs: `FinetuneConfigBundle`.
- Side effects: reads YAML from disk.
- Notable callers/callees: called by `sleep2vec.common.apply_finetune_config`; delegates to `_build_model_config`, `_build_task_config`, `_validate_layer_mix_config`.
- Reuse guidance: canonical downstream YAML loader.
- Duplication-risk notes: high.

## `sleep2vec.config.validate_model_config`

- File: `sleep2vec/config.py`
- Signature: `validate_model_config(model_cfg: ModelConfig) -> int`
- Purpose and contract: enforces shared tokenizer output dimension, valid CLS settings, and supported downstream aggregator names; returns the common channel feature dimension.
- Important inputs: `ModelConfig`.
- Important outputs: shared tokenizer feature dimension.
- Side effects: none.
- Notable callers/callees: called by `sleep2vec.builders.build_tokenizers_and_dim`.
- Reuse guidance: always validate before constructing tokenizers from YAML configs.
- Duplication-risk notes: medium-high.

## `sleep2vec.config._validate_adapt_config`

- File: `sleep2vec/config.py`
- Signature: `_validate_adapt_config(adapt_cfg: AdaptConfig | None, model_cfg: ModelConfig) -> None`
- Purpose and contract: enforces that `adapt.new_channels` exist in `model.channels`, pair-schedule points are ordered and bounded, final `until` equals `1.0`, and LR scales are non-negative.
- Important inputs: parsed `AdaptConfig`, parsed `ModelConfig`.
- Important outputs: none; raises on invalid config.
- Side effects: none.
- Notable callers/callees: called by `load_pretrain_config`.
- Reuse guidance: branch-canonical validation for adaptation YAML.
- Duplication-risk notes: high.

## `sleep2vec.common.apply_model_config_args`

- File: `sleep2vec/common.py`
- Signature: `apply_model_config_args(args, model_cfg: ModelConfig, *, set_backbone_arch: bool = False) -> None`
- Purpose and contract: copies YAML-derived channel names and `channel_input_dims` onto the argparse namespace; can also set `args.backbone_arch`.
- Important inputs: argparse namespace and parsed `ModelConfig`.
- Important outputs: mutated namespace.
- Side effects: mutates `args`.
- Notable callers/callees: called by `sleep2vec/pretrain.py`, `sleep2vec/adapt.py`, and `sleep2vec.common.apply_finetune_config`.
- Reuse guidance: preferred bridge from YAML config to runtime CLI state.
- Duplication-risk notes: medium.

## `sleep2vec.common.apply_task_flags`

- File: `sleep2vec/common.py`
- Signature: `apply_task_flags(args, task_cfg: TaskConfig | None = None) -> None`
- Purpose and contract: resolves built-in or YAML-defined task semantics into runtime flags such as `output_dim`, `is_classification`, `is_seq`, `monitor`, `monitor_mod`, `label_source_name`, `stage_names`, and `label_merge_map`.
- Important inputs: argparse namespace with `label_name`; optional YAML `TaskConfig`.
- Important outputs: mutated namespace.
- Side effects: mutates `args`; raises for unsupported metadata task semantics.
- Notable callers/callees: called by `sleep2vec.common.apply_finetune_config`; uses `_validate_builtin_task_cfg`, `_validate_metadata_label_support`, and the built-in task helper accessors in `sleep2vec.common`.
- Reuse guidance: canonical downstream task resolver.
- Duplication-risk notes: high.

## `sleep2vec.common.remap_stage_labels`

- File: `sleep2vec/common.py`
- Signature: `remap_stage_labels(labels, label_name: str)`
- Purpose and contract: remaps raw built-in sleep-staging labels into merged label spaces for `stage3` and `stage4`; returns the original labels unchanged for tasks without a merge map.
- Important inputs: tensor-like labels and the requested downstream label name.
- Important outputs: remapped labels with ignore indices preserved.
- Side effects: clones the label tensor when a merge map exists.
- Notable callers/callees: called by `Sleep2vecFinetuning._get_targets`; uses `get_task_label_merge_map`.
- Reuse guidance: canonical stage-label merge path.
- Duplication-risk notes: high.

## `sleep2vec.common.apply_finetune_config`

- File: `sleep2vec/common.py`
- Signature: `apply_finetune_config(args) -> tuple[Any, Any]`
- Purpose and contract: loads finetune YAML, copies model/data/task fields into `args`, enforces `data.data_channel_names == model.channels`, and resolves task flags.
- Important inputs: argparse namespace with `config` and `label_name`.
- Important outputs: `(config_bundle, model_cfg)`.
- Side effects: mutates `args`; reads YAML from disk.
- Notable callers/callees: called by `sleep2vec/infer.py`; reused conceptually by `sleep2vec/finetune.py`.
- Reuse guidance: downstream runtime entrypoints should go through this function.
- Duplication-risk notes: high.

## `sleep2vec.builders.build_tokenizers_and_dim`

- File: `sleep2vec/builders.py`
- Signature: `build_tokenizers_and_dim(model_cfg: ModelConfig, *, device: str = "cuda") -> tuple[dict[str, nn.Module], int]`
- Purpose and contract: validates the model config, builds per-channel tokenizer modules, and returns the shared channel feature dimension.
- Important inputs: `ModelConfig`, target device string.
- Important outputs: tokenizer mapping and shared feature dimension.
- Side effects: constructs `nn.Module` instances.
- Notable callers/callees: called by `Sleep2vecPretrainModel.__init__`; uses `validate_model_config` and `build_tokenizer_mapping`.
- Reuse guidance: canonical tokenizer-construction entrypoint.
- Duplication-risk notes: medium-high.

## `sleep2vec.registry.register_backbone` / `get_backbone_builder`

- File: `sleep2vec/registry.py`
- Signatures:
  - `register_backbone(name: str)`
  - `get_backbone_builder(name: str) -> BackboneBuilder`
- Purpose and contract: central registry surface for backbone factory functions.
- Important inputs: registry name string.
- Important outputs: decorator or registered builder lookup.
- Side effects: mutates in-memory registry at import time.
- Notable callers/callees: used by `sleep2vec/backbones/encoder_factory.py`; consumed by `sleep2vec.builders.build_encoder_factory`.
- Reuse guidance: new backbones should register here instead of branching on names in runtime code.
- Duplication-risk notes: high if conditional logic appears elsewhere.

## `sleep2vec.modules.tokenizers.build_tokenizer_from_channel`

- File: `sleep2vec/modules/tokenizers.py`
- Signature: `build_tokenizer_from_channel(channel: ChannelConfig, *, device: str = "cuda") -> nn.Module`
- Purpose and contract: instantiates a tokenizer from a YAML `ChannelConfig`, requiring `tokenizer.out_dim`.
- Important inputs: `ChannelConfig`, device string.
- Important outputs: tokenizer module.
- Side effects: constructs `nn.Module`.
- Notable callers/callees: used by `build_tokenizer_mapping`.
- Reuse guidance: canonical per-channel tokenizer factory.
- Duplication-risk notes: medium.
