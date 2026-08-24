from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import inspect
import json
import logging
from pathlib import Path
import re
import sys
import typing as t

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.whole_night_index import build_embedding_sample_key, validate_whole_night_index
from sleep2vec2.checkpoints import get_state_dict_from_checkpoint, load_checkpoint
from sleep2vec2.common import apply_data_backend_args, apply_model_config_args
from sleep2vec2.config import load_finetune_config, load_pretrain_config
from sleep2vec2.data.kaldi_psg_dataset import KaldiPSGDataset
from sleep2vec2.data.psg_pretrain_dataset import PSGPretrainDataset
from sleep2vec2.preprocess.save_dataset_presets import (
    _load_preset_build_block,
    _resolve_effective_min_channels,
    _resolve_validation_channels,
)
from sleep2vec2.pretrain_model import Sleep2vecPretrainModel
from sleep2vec2.utils import move_to_device

PACKAGE_NAMESPACE = "sleep2vec2"

MANIFEST_COLUMNS = (
    "sample_key",
    "path",
    "source",
    "dataset",
    "split",
    "token_start",
    "token_end",
    "num_tokens",
    "matrix_rows",
    "cls_matrix_rows",
    "available_channels",
)

_KALDI_SAMPLE_KEY_RE = re.compile(r".*_\d{6}_\d{6}$")
WHOLE_NIGHT_POSITION_CAPACITY = 4096


class CheckpointLoadPlan(t.NamedTuple):
    checkpoint_kind: str
    checkpoint_prefix: str


def _import_kaldi_native_io():
    try:
        import kaldi_native_io
    except ImportError as exc:
        raise RuntimeError(
            "kaldi_native_io is required to write Kaldi ark/scp files. "
            "Install requirements.txt before running with --output-format kaldi."
        ) from exc
    return kaldi_native_io


def _load_config_data(path: Path) -> dict[str, t.Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping.")
    return data


def _config_is_finetune(config_data: t.Mapping[str, t.Any]) -> bool:
    return isinstance(config_data.get("finetune"), dict)


def _load_config_bundle(args: argparse.Namespace):
    config_data = _load_config_data(args.config)
    if _config_is_finetune(config_data):
        bundle = load_finetune_config(args.config)
        config_kind = "finetune"
    else:
        bundle = load_pretrain_config(args.config)
        config_kind = "pretrain"

    model_cfg = bundle.model
    data_cfg = bundle.data
    apply_model_config_args(args, model_cfg)
    args.model_channel_names = list(args.channel_names)
    args.model_channel_input_dims = dict(args.channel_input_dims)
    selected_channels = list(getattr(args, "selected_channels", None) or args.model_channel_names)
    if len(selected_channels) != len(set(selected_channels)):
        raise ValueError("--channels must not contain duplicates.")
    unknown_channels = sorted(set(selected_channels) - set(args.model_channel_names))
    if unknown_channels:
        raise ValueError(
            f"--channels contains channels absent from model.channels: {unknown_channels}. "
            f"Model channels: {args.model_channel_names}."
        )
    if not selected_channels:
        raise ValueError("--channels must select at least one model channel.")
    unsafe_channels = [
        name
        for name in selected_channels
        if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}
    ]
    if unsafe_channels:
        raise ValueError(f"--channels must contain single safe path components: {unsafe_channels}.")
    args.channel_names = selected_channels
    args.channel_input_dims = {name: args.model_channel_input_dims[name] for name in selected_channels}
    args.channel_aliases = {name: alias for name, alias in args.channel_aliases.items() if name in selected_channels}
    args.training_max_tokens = int(data_cfg.max_tokens)
    args.max_tokens = args.training_max_tokens
    if config_kind == "finetune":
        args.data_channel_names = data_cfg.data_channel_names or args.model_channel_names
        if set(args.data_channel_names) != set(args.model_channel_names):
            raise ValueError(
                "data.data_channel_names in YAML must match model.channels for embedding extraction. "
                f"Model channels: {args.model_channel_names}; data channels: {args.data_channel_names}."
            )

    if args.preset_path is not None and args.data_index is not None:
        raise ValueError("--preset-path and --data-index are mutually exclusive. Choose one NPZ data source.")
    if (
        config_kind == "finetune"
        and args.data_index is not None
        and args.preset_path is None
        and data_cfg.finetune_preset_path
    ):
        raise ValueError(
            "YAML data.finetune_preset_path conflicts with --data-index. "
            "Use --preset-path for preset extraction, or clear the YAML preset to use --data-index."
        )
    if config_kind == "finetune" and args.preset_path is None and data_cfg.finetune_preset_path:
        args.preset_path = Path(data_cfg.finetune_preset_path)
    if config_kind == "finetune" and args.data_index is None and data_cfg.finetune_data_index:
        args.data_index = [Path(data_cfg.finetune_data_index)]

    yaml_backend = getattr(data_cfg, "backend", "npz") or "npz"
    if args.data_backend is not None and args.data_backend != yaml_backend:
        raise ValueError(f"--data-backend={args.data_backend!r} conflicts with YAML data.backend={yaml_backend!r}.")
    if yaml_backend == "kaldi" and args.data_index is not None:
        raise ValueError("Kaldi backend uses manifest.json; --data-index is only valid for data.backend=npz.")
    if yaml_backend != "kaldi" and (args.kaldi_data_root is not None or args.kaldi_manifest is not None):
        raise ValueError("--kaldi-data-root/--kaldi-manifest require YAML data.backend=kaldi.")

    apply_data_backend_args(args, data_cfg, preset_attr="preset_path")
    sequence_mode = getattr(args, "sequence_mode", "config-windows")
    if args.embedding_kind == "both" and sequence_mode != "whole-night":
        raise ValueError("--embedding-kind both requires --sequence-mode whole-night.")
    if sequence_mode == "whole-night":
        if args.embedding_kind != "both":
            raise ValueError("--sequence-mode whole-night requires --embedding-kind both.")
        if args.layer_index != -1:
            raise ValueError("--embedding-kind both only supports --layer-index -1.")
        if args.output_format != "npz":
            raise ValueError("--embedding-kind both only supports --output-format npz.")
        if args.batch_size != 1:
            raise ValueError("--sequence-mode whole-night requires --batch-size 1.")
        if args.num_workers < 0 or args.num_workers > 8:
            raise ValueError("--sequence-mode whole-night requires --num-workers in [0, 8].")
        if args.max_source_tokens is None or args.max_source_tokens <= 0:
            raise ValueError("--sequence-mode whole-night requires a positive --max-source-tokens.")
        if args.max_source_tokens + 1 > WHOLE_NIGHT_POSITION_CAPACITY:
            raise ValueError(f"--max-source-tokens plus CLS must not exceed {WHOLE_NIGHT_POSITION_CAPACITY}.")
        if args.data_backend != "npz":
            raise ValueError("--sequence-mode whole-night requires the NPZ data backend.")
        if args.preset_path is not None or not args.data_index:
            raise ValueError("--sequence-mode whole-night requires --data-index and does not accept --preset-path.")
        if model_cfg.backbone.name != "roformer":
            raise ValueError("--sequence-mode whole-night is implemented only for RoFormer backbones.")

        position_overrides = dict(model_cfg.backbone.config_overrides or {})
        args.training_position_capacity = int(position_overrides.get("max_position_embeddings", 1536))
        args.effective_position_capacity = WHOLE_NIGHT_POSITION_CAPACITY
        args.max_tokens = int(args.max_source_tokens)
        args.dataset_channel_names = list(args.channel_names)
        args.dataset_channel_input_dims = dict(args.channel_input_dims)
        return bundle, model_cfg, config_kind

    preset_required_channels, preset_min_channels = _load_preset_build_block(config_data)
    if preset_required_channels is None:
        args.dataset_channel_names = list(args.channel_names)
        args.dataset_channel_input_dims = dict(args.channel_input_dims)
    else:
        preset_channels, preset_dims = _resolve_validation_channels(
            model_channels=list(args.model_channel_names),
            channel_input_dims=dict(args.model_channel_input_dims),
            preset_required_channels=preset_required_channels,
            selected_channels=(list(args.selected_channels) if args.selected_channels is not None else None),
        )
        _resolve_effective_min_channels(
            channel_names=preset_channels,
            cli_min_channels=len(preset_channels),
            preset_min_channels=preset_min_channels,
        )
        args.dataset_channel_names = list(dict.fromkeys([*args.channel_names, *preset_channels]))
        args.dataset_channel_input_dims = {**dict(args.channel_input_dims), **preset_dims}
    return bundle, model_cfg, config_kind


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight_output_dir(output_dir: Path) -> None:
    if ".." in output_dir.parts:
        raise ValueError(f"Embedding output path must not contain '..' path components: {output_dir}")
    if output_dir.is_symlink() or any(
        parent.is_symlink() or (parent.exists() and not parent.is_dir()) for parent in output_dir.parents
    ):
        raise ValueError(
            f"Embedding output path must not be a symlink or traverse a non-directory ancestor: {output_dir}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Embedding output directory must be empty: {output_dir}")


def _preflight_whole_night_index(args: argparse.Namespace) -> None:
    expected_tokens_by_path = validate_whole_night_index(
        args.data_index, eval_split=args.eval_split, max_source_tokens=args.max_source_tokens
    )
    args.expected_tokens_by_path = expected_tokens_by_path
    args.expected_sample_count = len(expected_tokens_by_path)
    args.expected_total_tokens = int(sum(expected_tokens_by_path.values()))
    args.observed_min_source_tokens = int(min(expected_tokens_by_path.values()))
    args.observed_max_source_tokens = int(max(expected_tokens_by_path.values()))


def _validate_whole_night_dataset(dataset: t.Any, args: argparse.Namespace) -> None:
    data = list(getattr(dataset, "data", []) or [])
    if len(data) != args.expected_sample_count:
        raise ValueError(
            "Whole-night dataset coverage changed during channel validation: "
            f"expected {args.expected_sample_count}, got {len(data)}."
        )
    seen_paths: set[str] = set()
    for sample in data:
        path = str(sample.path)
        expected_tokens = args.expected_tokens_by_path.get(path)
        if expected_tokens is None:
            raise ValueError(f"Whole-night dataset emitted an unexpected path: {path}")
        if path in seen_paths:
            raise ValueError(f"Whole-night dataset emitted a duplicate path: {path}")
        if int(sample.start) != 0 or int(sample.end) != expected_tokens:
            raise ValueError(
                f"Whole-night sample for {path} must span [0, {expected_tokens}], "
                f"got [{sample.start}, {sample.end}]."
            )
        seen_paths.add(path)


def _sources_for_extraction(args: argparse.Namespace, bundle: t.Any, config_kind: str) -> list[str]:
    if args.override_dataset_names:
        return list(args.override_dataset_names)
    if config_kind != "finetune":
        return []
    data_cfg = bundle.data
    if args.eval_split == "test":
        return list(data_cfg.test_dataset_names or [])
    return list(data_cfg.train_dataset_names or [])


def _metadata_lookup_from_dataset(dataset: t.Any) -> dict[str, dict[str, t.Any]]:
    lookup: dict[str, dict[str, t.Any]] = {}
    for item in getattr(dataset, "data", []) or []:
        sample_id = getattr(item, "id", None)
        if sample_id is None:
            continue
        lookup.setdefault(str(sample_id), dict(getattr(item, "metadata", {}) or {}))
    return lookup


def _metadata_lookup_from_npz_index(args: argparse.Namespace) -> dict[str, dict[str, t.Any]]:
    if not args.data_index or args.preset_path is not None:
        return {}

    rows: list[dict[str, t.Any]] = []
    for index_path in args.data_index:
        with Path(index_path).open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row.get("split") != args.eval_split:
                    continue
                if not row.get("source"):
                    row["source"] = str(index_path)
                rows.append(row)
    return {str(idx): row for idx, row in enumerate(rows)}


def _attach_metadata_lookup(dataloader: t.Any, dataset: t.Any, args: argparse.Namespace) -> t.Any:
    lookup = _metadata_lookup_from_dataset(dataset)
    for sample_id, metadata in _metadata_lookup_from_npz_index(args).items():
        merged = dict(lookup.get(sample_id, {}))
        merged.update(metadata)
        lookup[sample_id] = merged
    setattr(dataloader, "_embedding_metadata_by_id", lookup)
    return dataloader


def _build_extraction_loader(args: argparse.Namespace, bundle: t.Any, config_kind: str):
    sources = _sources_for_extraction(args, bundle, config_kind)
    channel_names = list(getattr(args, "dataset_channel_names", args.channel_names))
    channel_input_dims = dict(getattr(args, "dataset_channel_input_dims", None) or args.channel_input_dims)
    dataset_kwargs = dict(
        channel_names=channel_names,
        channel_input_dims=channel_input_dims,
        split=[args.eval_split],
        max_tokens=args.max_tokens,
        mask_rate=0.0,
        sources=sources,
        randomly_select_channels=False,
        allow_missing_channels=False,
        min_channels=len(channel_names),
        is_train_set=False,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        filter_max_workers=(
            min(max(args.num_workers, 1), 8)
            if getattr(args, "sequence_mode", "config-windows") == "whole-night"
            else None
        ),
    )

    if getattr(args, "data_backend", "npz") == "kaldi":
        dataset = KaldiPSGDataset(
            **dataset_kwargs,
            kaldi_data_root=args.kaldi_data_root,
            manifest=args.kaldi_manifest,
        )
        return _attach_metadata_lookup(dataset.dataloader(device=args.device), dataset, args)

    if args.preset_path is None and not args.data_index:
        raise ValueError("NPZ extraction requires --data-index or --preset-path.")

    if getattr(args, "sequence_mode", "config-windows") == "whole-night":
        dataset_kwargs["channel_length_tolerance"] = 0
    dataset = PSGPretrainDataset(
        **dataset_kwargs,
        save_preset_path=None,
        load_preset_path=args.preset_path,
        index=args.data_index,
        stride_tokens=0 if getattr(args, "sequence_mode", "config-windows") == "whole-night" else args.max_tokens,
        channel_aliases=getattr(args, "channel_aliases", {}),
    )
    if getattr(args, "sequence_mode", "config-windows") == "whole-night":
        _validate_whole_night_dataset(dataset, args)
    return _attach_metadata_lookup(dataset.dataloader(device=args.device), dataset, args)


def _infer_checkpoint_load_plan(state_dict: t.Mapping[str, torch.Tensor]) -> CheckpointLoadPlan:
    keys = tuple(state_dict.keys())
    downstream_markers = (
        "ema_model.backbone.",
        "model.backbone.",
        "backbone.",
        "ema_model.head.",
        "model.head.",
        "ema_model.temporal_agg.",
        "model.temporal_agg.",
        "ema_model.layer_mix.",
        "model.layer_mix.",
    )
    finetune_prefixes = ("ema_model.backbone.", "model.backbone.", "backbone.")
    pretrain_prefixes = ("ema_model.", "model.")
    pretrain_markers = (
        "ema_model.encoder.",
        "ema_model.tokenizer_mapping.",
        "model.encoder.",
        "model.tokenizer_mapping.",
    )

    has_downstream = any(any(key.startswith(marker) for marker in downstream_markers) for key in keys)
    has_pretrain = any(key.startswith(pretrain_markers) for key in keys)
    if has_downstream and has_pretrain:
        preview = ", ".join(keys[:8])
        raise ValueError(f"Checkpoint mixes downstream and pretrain-only key layouts. Example keys: [{preview}]")
    if has_downstream:
        for prefix in finetune_prefixes:
            if any(key.startswith(prefix) for key in keys):
                return CheckpointLoadPlan("finetune", prefix)
        preview = ", ".join(keys[:8])
        raise ValueError(
            "Checkpoint looks like a downstream checkpoint but no backbone subtree was found. "
            f"Example keys: [{preview}]"
        )

    if has_pretrain:
        for prefix in pretrain_prefixes:
            if any(key.startswith(prefix) for key in keys):
                return CheckpointLoadPlan("pretrain", prefix)

    preview = ", ".join(keys[:8])
    raise ValueError(
        "Could not infer checkpoint kind. Expected a pretrain subtree under model./ema_model. "
        "or a finetune backbone subtree under model.backbone./ema_model.backbone./backbone. "
        f"Example keys: [{preview}]"
    )


def _has_adapter_keys(state_dict: t.Mapping[str, torch.Tensor]) -> bool:
    return any("lora_" in key for key in state_dict)


def _cls_state_keys(keys: t.Iterable[str]) -> list[str]:
    return [key for key in keys if key.startswith("cls_embedding.")]


def _validate_embedding_kind_compatible(model: Sleep2vecPretrainModel, embedding_kind: str) -> None:
    cls_embedding = getattr(model, "cls_embedding", None)
    if embedding_kind in {"cls", "both"} and (cls_embedding is None or not getattr(cls_embedding, "has_cls", False)):
        raise ValueError(
            f"Requested --embedding-kind {embedding_kind}, but the loaded checkpoint/config is not CLS-enabled. "
            "Use a checkpoint and config trained with model.cls.embedding_type=bert, or export token embeddings."
        )


def _load_backbone_checkpoint(
    model: Sleep2vecPretrainModel,
    ckpt_path: Path,
    device: str,
    *,
    adapters_enabled: bool = False,
) -> CheckpointLoadPlan:
    ckpt = load_checkpoint(ckpt_path, device=torch.device("cpu"))
    state_dict = get_state_dict_from_checkpoint(ckpt)
    load_plan = _infer_checkpoint_load_plan(state_dict)
    filtered = {
        key[len(load_plan.checkpoint_prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(load_plan.checkpoint_prefix)
    }
    if _has_adapter_keys(filtered) and not adapters_enabled:
        raise ValueError(
            "Checkpoint contains adapter weights, but the YAML finetune.lora settings do not enable adapters."
        )
    load_info = model.load_state_dict(filtered, strict=False)
    unexpected_cls_keys = _cls_state_keys(load_info.unexpected_keys)
    if unexpected_cls_keys:
        raise ValueError(
            "Checkpoint contains CLS embedding weights, but the YAML config does not enable CLS embeddings: "
            f"{unexpected_cls_keys}"
        )
    if load_info.unexpected_keys:
        raise ValueError(
            "Checkpoint contains keys incompatible with the configured backbone: " f"{list(load_info.unexpected_keys)}"
        )
    missing_cls_keys = _cls_state_keys(load_info.missing_keys)
    if missing_cls_keys:
        raise ValueError(
            "YAML config enables CLS embeddings, but the checkpoint is missing CLS embedding weights: "
            f"{missing_cls_keys}"
        )
    if load_info.missing_keys:
        raise ValueError(
            "Checkpoint is missing keys required by the configured extraction backbone: "
            f"{list(load_info.missing_keys)}"
        )
    logging.info(
        "Loaded %s checkpoint using prefix=%s from %s",
        load_plan.checkpoint_kind,
        load_plan.checkpoint_prefix,
        ckpt_path,
    )
    return load_plan


def _extend_roformer_position_capacity(model: Sleep2vecPretrainModel, capacity: int) -> int:
    encoder = model.get_encoder() if hasattr(model, "get_encoder") else model.encoder
    position_embeddings = encoder.encoder.embed_positions
    training_capacity = int(position_embeddings.weight.shape[0])
    embedding_dim = int(position_embeddings.weight.shape[1])
    encoder.encoder.embed_positions = type(position_embeddings)(capacity, embedding_dim).to(
        device=position_embeddings.weight.device
    )
    encoder.config.max_position_embeddings = capacity
    return training_capacity


def _finetune_adapters_enabled(bundle: t.Any, config_kind: str) -> bool:
    if config_kind != "finetune":
        return False
    lora_cfg = getattr(getattr(bundle, "finetune", None), "lora", None)
    return bool(
        lora_cfg
        and getattr(lora_cfg, "freeze_backbone_and_insert_lora", False)
        and getattr(lora_cfg, "insert_lora", False)
    )


def _apply_finetune_adapters(
    backbone: Sleep2vecPretrainModel,
    model_cfg: t.Any,
    finetune_cfg: t.Any,
) -> Sleep2vecPretrainModel:
    from sleep2vec2.downstream_model import Sleep2vecDownstreamModel

    lora_cfg = finetune_cfg.lora
    adapter_host = Sleep2vecDownstreamModel.__new__(Sleep2vecDownstreamModel)
    torch.nn.Module.__init__(adapter_host)
    adapter_host.backbone = backbone
    adapter_host.channel_names = [c.name for c in model_cfg.channels]
    adapter_host.separate_adapters = False
    adapter_host.freeze_backbone_and_insert_lora(
        insert_lora=lora_cfg.insert_lora,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules,
        use_dora=lora_cfg.use_dora,
        separate_adapters=lora_cfg.separate_adapters,
    )
    setattr(adapter_host.backbone, "_extract_separate_adapters", bool(lora_cfg.separate_adapters))
    return adapter_host.backbone


def _build_backbone(
    model_cfg: t.Any,
    device: str,
    *,
    bundle: t.Any,
    config_kind: str,
) -> Sleep2vecPretrainModel:
    backbone = Sleep2vecPretrainModel(
        model_config=model_cfg,
        device=device,
    ).to(device)
    if _finetune_adapters_enabled(bundle, config_kind):
        backbone = _apply_finetune_adapters(backbone, model_cfg, bundle.finetune)
    return backbone


def _select_layer_state(
    hidden_states: t.Sequence[torch.Tensor],
    layer_index: int,
    num_hidden_layers: int,
) -> tuple[torch.Tensor, int]:
    if not isinstance(hidden_states, (list, tuple)) or not hidden_states:
        raise ValueError("Backbone did not return hidden states.")
    if len(hidden_states) not in {num_hidden_layers, num_hidden_layers + 1}:
        raise ValueError(
            f"Expected {num_hidden_layers} or {num_hidden_layers + 1} hidden states, got {len(hidden_states)}."
        )

    has_input_state = len(hidden_states) == num_hidden_layers + 1
    if layer_index == -1:
        return hidden_states[-1], num_hidden_layers
    if layer_index == 0:
        if not has_input_state:
            raise ValueError("layer_index=0 requested, but the backbone did not return the projected input state.")
        return hidden_states[0], 0
    if layer_index < -1:
        raise ValueError("--layer-index only accepts -1, 0, or a positive transformer layer index.")
    if layer_index < 1 or layer_index > num_hidden_layers:
        raise ValueError(f"--layer-index must be in [1, {num_hidden_layers}], 0, or -1; got {layer_index}.")

    offset = 0 if has_input_state else -1
    return hidden_states[layer_index + offset], layer_index


def _trim_hidden_to_numpy(
    model: Sleep2vecPretrainModel,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
    lengths: torch.Tensor,
    *,
    embedding_kind: str,
) -> list[np.ndarray]:
    cls_embedding = getattr(model, "cls_embedding", None)
    if embedding_kind == "cls":
        _validate_embedding_kind_compatible(model, embedding_kind)
        _, cls_hidden, _ = cls_embedding.split_hidden(hidden, attention_mask)
    elif cls_embedding is not None:
        token_hidden, _, _ = cls_embedding.split_hidden(hidden, attention_mask)
    else:
        token_hidden = hidden

    rows: list[np.ndarray] = []
    if embedding_kind == "cls":
        for idx in range(cls_hidden.size(0)):
            matrix = cls_hidden[idx : idx + 1].detach().to(torch.float32).cpu().numpy()
            rows.append(np.ascontiguousarray(matrix, dtype=np.float32))
        return rows

    for idx, raw_length in enumerate(lengths.detach().cpu().tolist()):
        num_tokens = min(int(raw_length), int(token_hidden.size(1)))
        matrix = token_hidden[idx, :num_tokens].detach().to(torch.float32).cpu().numpy()
        rows.append(np.ascontiguousarray(matrix, dtype=np.float32))
    return rows


def _encode_channel(
    model: Sleep2vecPretrainModel,
    batch: dict[str, t.Any],
    channel_name: str,
    token_embeddings: torch.Tensor,
    layer_index: int,
    num_hidden_layers: int,
    *,
    embedding_kind: str,
) -> tuple[dict[str, list[np.ndarray]], int]:
    kwargs: dict[str, t.Any] = {"return_hidden_states": embedding_kind != "both"}
    params = inspect.signature(model._token_embeddings_to_hidden).parameters
    if "modality_name" in params:
        kwargs["modality_name"] = channel_name

    if getattr(model, "_extract_separate_adapters", False):
        encoder = model.get_encoder() if hasattr(model, "get_encoder") else getattr(model, "encoder", None)
        if not hasattr(encoder, "set_adapter"):
            raise ValueError("Configured separate adapters, but the backbone encoder does not support set_adapter.")
        encoder.set_adapter(f"ch_{channel_name}")

    hidden, attention_mask, hidden_states = model._token_embeddings_to_hidden(token_embeddings, batch, **kwargs)
    if embedding_kind == "both":
        return (
            {
                "cls_embedding": _trim_hidden_to_numpy(
                    model,
                    hidden,
                    attention_mask,
                    batch["length"],
                    embedding_kind="cls",
                ),
                "token_embedding": _trim_hidden_to_numpy(
                    model,
                    hidden,
                    attention_mask,
                    batch["length"],
                    embedding_kind="token",
                ),
            },
            num_hidden_layers,
        )
    selected_state, resolved_layer_index = _select_layer_state(hidden_states, layer_index, num_hidden_layers)
    return (
        {
            "embedding": _trim_hidden_to_numpy(
                model,
                selected_state,
                attention_mask,
                batch["length"],
                embedding_kind=embedding_kind,
            )
        },
        resolved_layer_index,
    )


def _sanitize_key_part(value: t.Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "unknown"


def _record_key_from_path(path_value: t.Any) -> str:
    path = Path(str(path_value))
    return _sanitize_key_part(f"{path.parent.name}_{path.stem}")


def _metadata_value_present(value: t.Any) -> bool:
    if value is None:
        return False
    try:
        if bool(np.isnan(value)):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() != "nan" and str(value).strip() != ""


def _metadata_value(metadata: t.Mapping[str, t.Any], key: str, fallback: t.Any = None) -> t.Any:
    value = metadata.get(key)
    return value if _metadata_value_present(value) else fallback


def _record_key_from_metadata(record_key_value: t.Any, session_id_value: t.Any, path_value: t.Any) -> str:
    if _metadata_value_present(record_key_value):
        return _sanitize_key_part(record_key_value)
    if _metadata_value_present(session_id_value):
        return _sanitize_key_part(session_id_value)
    return _record_key_from_path(path_value)


def _sample_key(
    *,
    sample_id: t.Any,
    source_value: t.Any,
    path_value: t.Any,
    record_key_value: t.Any,
    session_id_value: t.Any,
    token_start: int,
    token_end: int,
) -> str:
    if isinstance(sample_id, str) and _KALDI_SAMPLE_KEY_RE.match(sample_id):
        return sample_id
    return build_embedding_sample_key(
        source_value=source_value,
        record_key=_record_key_from_metadata(record_key_value, session_id_value, path_value),
        token_start=token_start,
        token_end=token_end,
    )


def _metadata_values(batch: dict[str, t.Any], key: str, sample_count: int, default: t.Any = "nan") -> list[t.Any]:
    metadata = batch.get("metadata", {})
    values = metadata.get(key, None) if isinstance(metadata, dict) else None
    if values is None:
        return [default] * sample_count
    if torch.is_tensor(values):
        return values.detach().cpu().tolist()
    if isinstance(values, (str, bytes)):
        return [values] * sample_count
    return list(values)


def _open_kaldi_writers(output_dir: Path, split: str, channel_names: t.Sequence[str], stack: ExitStack):
    kaldi_native_io = _import_kaldi_native_io()
    writers = {}
    split_dir = output_dir / "channels" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for channel in channel_names:
        ark_path = split_dir / f"{channel}.ark"
        scp_path = split_dir / f"{channel}.scp"
        writers[channel] = stack.enter_context(kaldi_native_io.FloatMatrixWriter(f"ark,scp:{ark_path},{scp_path}"))
    return writers


def _channel_manifest_entry(output_format: str, split: str, channel: str, hidden_size: int) -> dict[str, t.Any]:
    if output_format == "kaldi":
        return {
            "input_dim": int(hidden_size),
            "scp": (Path("channels") / split / f"{channel}.scp").as_posix(),
            "ark_storage": "float_matrix",
        }
    return {
        "hidden_size": int(hidden_size),
        "npz_dir": (Path("channels") / split / channel).as_posix(),
    }


def _extract_and_write_embeddings(
    args: argparse.Namespace,
    model: Sleep2vecPretrainModel,
    dataloader: t.Iterable[dict[str, t.Any]],
    model_cfg: t.Any,
    load_plan: CheckpointLoadPlan,
    *,
    namespace: str = PACKAGE_NAMESPACE,
) -> Path:
    output_dir = Path(args.output_dir)
    split = str(args.eval_split)
    channel_names = list(args.channel_names)
    hidden_size = int(model_cfg.backbone.hidden_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv_path = manifests_dir / f"{split}.csv"

    if args.output_format == "npz":
        for channel in channel_names:
            (output_dir / "channels" / split / channel).mkdir(parents=True, exist_ok=True)

    sample_count = 0
    total_source_tokens = 0
    total_array_bytes = 0
    resolved_layer_index = None
    seen_sample_keys: set[str] = set()
    metadata_by_id = getattr(dataloader, "_embedding_metadata_by_id", {})

    with ExitStack() as stack:
        manifest_file = stack.enter_context(manifest_csv_path.open("w", newline=""))
        manifest_writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        manifest_writer.writeheader()
        kaldi_writers = (
            _open_kaldi_writers(output_dir, split, channel_names, stack) if args.output_format == "kaldi" else {}
        )

        model.eval()
        with torch.no_grad():
            for raw_batch in dataloader:
                batch = move_to_device(raw_batch, args.device)
                token_embeddings_by_channel = {
                    channel: model.tokenizer_mapping[channel](batch["tokens"][channel]) for channel in channel_names
                }
                channel_matrices: dict[str, dict[str, list[np.ndarray]]] = {}
                for channel in channel_names:
                    matrices, current_layer_index = _encode_channel(
                        model,
                        batch,
                        channel,
                        token_embeddings_by_channel[channel],
                        int(args.layer_index),
                        int(model_cfg.backbone.num_hidden_layers),
                        embedding_kind=args.embedding_kind,
                    )
                    channel_matrices[channel] = matrices
                    resolved_layer_index = current_layer_index

                batch_size = len(batch["id"])
                source_values = _metadata_values(batch, "source", batch_size)
                path_values = _metadata_values(batch, "path", batch_size)
                dataset_values = _metadata_values(batch, "dataset", batch_size, default=None)
                token_starts = batch["token_start"].detach().cpu().tolist()
                source_lengths = batch["length"].detach().cpu().tolist()
                ids = list(batch["id"])

                for sample_idx in range(batch_size):
                    source_num_tokens = int(source_lengths[sample_idx])
                    primary_key = "token_embedding" if args.embedding_kind == "both" else "embedding"
                    matrix_rows = int(channel_matrices[channel_names[0]][primary_key][sample_idx].shape[0])
                    num_tokens = source_num_tokens
                    token_start = int(token_starts[sample_idx])
                    token_end = token_start + num_tokens
                    source_value = source_values[sample_idx]
                    path_value = path_values[sample_idx]
                    sample_metadata = metadata_by_id.get(str(ids[sample_idx]), {})
                    source_value = _metadata_value(sample_metadata, "source", source_value)
                    path_value = _metadata_value(sample_metadata, "path", path_value)
                    if getattr(args, "sequence_mode", "config-windows") == "whole-night":
                        expected_tokens = args.expected_tokens_by_path.get(str(path_value))
                        if expected_tokens != source_num_tokens:
                            raise ValueError(
                                f"Whole-night token count mismatch for {path_value}: "
                                f"expected {expected_tokens}, got {source_num_tokens}."
                            )
                    dataset_value = _metadata_value(sample_metadata, "dataset", dataset_values[sample_idx])
                    if not _metadata_value_present(dataset_value):
                        dataset_value = source_value
                    sample_key = _sample_key(
                        sample_id=ids[sample_idx],
                        source_value=source_value,
                        path_value=path_value,
                        record_key_value=_metadata_value(sample_metadata, "record_key"),
                        session_id_value=_metadata_value(sample_metadata, "session_id"),
                        token_start=token_start,
                        token_end=token_end,
                    )
                    if sample_key in seen_sample_keys:
                        raise ValueError(f"Duplicate embedding sample_key generated: {sample_key}")
                    seen_sample_keys.add(sample_key)

                    for channel in channel_names:
                        matrices = {key: rows[sample_idx] for key, rows in channel_matrices[channel].items()}
                        if args.embedding_kind == "both":
                            if matrices["token_embedding"].shape != (source_num_tokens, hidden_size):
                                raise ValueError(
                                    f"Channel {channel!r} produced token shape "
                                    f"{matrices['token_embedding'].shape} for {sample_key}; "
                                    f"expected {(source_num_tokens, hidden_size)}."
                                )
                            if matrices["cls_embedding"].shape != (1, hidden_size):
                                raise ValueError(
                                    f"Channel {channel!r} produced CLS shape "
                                    f"{matrices['cls_embedding'].shape} for {sample_key}."
                                )
                        else:
                            expected_shape = (
                                (source_num_tokens, hidden_size) if args.embedding_kind == "token" else (1, hidden_size)
                            )
                            if matrices["embedding"].shape != expected_shape:
                                raise ValueError(
                                    f"Channel {channel!r} produced shape {matrices['embedding'].shape} "
                                    f"for {sample_key}; expected {expected_shape}."
                                )
                        for matrix in matrices.values():
                            if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
                                raise ValueError(f"Channel {channel!r} produced invalid values for {sample_key}.")
                            total_array_bytes += int(matrix.nbytes)
                        if args.output_format == "kaldi":
                            kaldi_writers[channel].write(sample_key, matrices["embedding"])
                        else:
                            npz_path = output_dir / "channels" / split / channel / f"{sample_key}.npz"
                            np.savez(npz_path, **matrices)

                    manifest_writer.writerow(
                        {
                            "sample_key": sample_key,
                            "path": path_value,
                            "source": source_value,
                            "dataset": dataset_value,
                            "split": split,
                            "token_start": token_start,
                            "token_end": token_end,
                            "num_tokens": num_tokens,
                            "matrix_rows": matrix_rows,
                            "cls_matrix_rows": 1 if args.embedding_kind in {"cls", "both"} else 0,
                            "available_channels": json.dumps(channel_names),
                        }
                    )
                    sample_count += 1
                    total_source_tokens += source_num_tokens

    if sample_count == 0:
        raise ValueError("No samples were exported.")
    if getattr(args, "sequence_mode", "config-windows") == "whole-night":
        if sample_count != args.expected_sample_count or total_source_tokens != args.expected_total_tokens:
            raise ValueError(
                "Whole-night export coverage mismatch: "
                f"samples={sample_count}/{args.expected_sample_count}, "
                f"tokens={total_source_tokens}/{args.expected_total_tokens}."
            )

    manifest = {
        "namespace": namespace,
        "config_path": str(args.config),
        "ckpt_path": str(args.ckpt_path),
        "checkpoint_kind": load_plan.checkpoint_kind,
        "checkpoint_prefix": load_plan.checkpoint_prefix,
        "output_format": args.output_format,
        "embedding_kind": args.embedding_kind,
        "sequence_mode": getattr(args, "sequence_mode", "config-windows"),
        "layer_index": int(args.layer_index),
        "resolved_layer_index": int(resolved_layer_index if resolved_layer_index is not None else args.layer_index),
        "hidden_size": hidden_size,
        "training_max_tokens": int(args.training_max_tokens),
        "model_channels": list(args.model_channel_names),
        "selected_channels": channel_names,
        "projection_applied": False,
        "output_keys": ["cls_embedding", "token_embedding"] if args.embedding_kind == "both" else ["embedding"],
        "output_dtype": "float32",
        "source_tokens": total_source_tokens,
        "array_bytes": total_array_bytes,
        "splits": {
            split: {
                "manifest": (Path("manifests") / f"{split}.csv").as_posix(),
                "sample_count": sample_count,
                "channels": {
                    channel: _channel_manifest_entry(args.output_format, split, channel, hidden_size)
                    for channel in channel_names
                },
            }
        },
        "channels": channel_names,
        "hashes": dict(args.input_hashes),
    }
    if getattr(args, "sequence_mode", "config-windows") == "whole-night":
        manifest["whole_night"] = {
            "declared_max_source_tokens": int(args.max_source_tokens),
            "declared_max_encoder_tokens": int(args.max_source_tokens) + 1,
            "observed_min_source_tokens": int(args.observed_min_source_tokens),
            "observed_max_source_tokens": int(args.observed_max_source_tokens),
            "observed_max_encoder_tokens": int(args.observed_max_source_tokens) + 1,
            "training_position_capacity": int(args.training_position_capacity),
            "effective_position_capacity": int(args.effective_position_capacity),
            "one_sample_per_path": True,
            "stride_tokens": 0,
        }
    manifest_json_path = output_dir / "manifest.json"
    manifest_json_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_json_path


def run_extraction(args: argparse.Namespace, *, namespace: str = PACKAGE_NAMESPACE) -> Path:
    args.config = Path(args.config).expanduser()
    args.ckpt_path = Path(args.ckpt_path).expanduser()
    args.output_dir = Path(args.output_dir).expanduser()
    args.preset_path = Path(args.preset_path).expanduser() if args.preset_path is not None else None
    args.data_index = [Path(path).expanduser() for path in args.data_index] if args.data_index is not None else None

    _preflight_output_dir(args.output_dir)

    if not args.config.exists():
        raise FileNotFoundError(f"Config YAML not found: {args.config}")
    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    input_hashes = {
        "config_sha256": _sha256_path(args.config),
        "checkpoint_sha256": _sha256_path(args.ckpt_path),
        "extractor_sha256": _sha256_path(Path(__file__)),
    }
    bundle, model_cfg, config_kind = _load_config_bundle(args)
    input_hashes["index_sha256"] = {str(path): _sha256_path(Path(path)) for path in (args.data_index or [])}
    if args.preset_path is not None:
        input_hashes["preset_sha256"] = _sha256_path(args.preset_path)
    args.input_hashes = input_hashes
    if getattr(args, "sequence_mode", "config-windows") == "whole-night":
        _preflight_whole_night_index(args)
    dataloader = _build_extraction_loader(args, bundle, config_kind)
    adapters_enabled = _finetune_adapters_enabled(bundle, config_kind)
    model = _build_backbone(model_cfg, args.device, bundle=bundle, config_kind=config_kind)
    load_plan = _load_backbone_checkpoint(
        model,
        args.ckpt_path,
        args.device,
        adapters_enabled=adapters_enabled,
    )
    if getattr(args, "sequence_mode", "config-windows") == "whole-night":
        training_capacity = _extend_roformer_position_capacity(model, args.effective_position_capacity)
        if training_capacity != args.training_position_capacity:
            raise ValueError(
                "Configured and constructed RoFormer position capacities differ: "
                f"config={args.training_position_capacity}, model={training_capacity}."
            )
    _validate_embedding_kind_compatible(model, args.embedding_kind)
    return _extract_and_write_embeddings(args, model, dataloader, model_cfg, load_plan, namespace=namespace)


def parse_args(argv: t.Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract backbone embeddings from a trained checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Pretrain or finetune YAML config.")
    parser.add_argument("--ckpt-path", type=Path, required=True, help="Checkpoint (.ckpt) path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Embedding output directory.")
    parser.add_argument("--output-format", choices=["npz", "kaldi"], required=True, help="Output storage format.")
    parser.add_argument(
        "--layer-index",
        type=int,
        default=-1,
        help="Layer to export: -1 final block, 0 projected input, or 1..N transformer block.",
    )
    parser.add_argument(
        "--embedding-kind",
        type=str,
        default="token",
        choices=("token", "cls", "both"),
        help="Save token or CLS embeddings; both is reserved for whole-night CLS-enabled extraction.",
    )
    parser.add_argument(
        "--channels",
        dest="selected_channels",
        type=str,
        nargs="+",
        default=None,
        help="Model-channel subset to read and export; the full model config is still loaded strictly.",
    )
    parser.add_argument(
        "--sequence-mode",
        choices=("config-windows", "whole-night"),
        default="config-windows",
        help="Use configured windows or one untruncated sample per indexed recording.",
    )
    parser.add_argument(
        "--max-source-tokens",
        type=int,
        default=None,
        help="Required hard source-token cap for --sequence-mode whole-night.",
    )
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="test", help="Split to export.")
    parser.add_argument("--batch-size", type=int, default=12, help="Extraction dataloader batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="Extraction dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device used for extraction.")
    parser.add_argument(
        "--data-backend",
        choices=["npz", "kaldi"],
        default=None,
        help="Data backend assertion. When set, it must match the YAML data.backend value.",
    )
    parser.add_argument("--kaldi-data-root", type=Path, default=None, help="Kaldi data root override.")
    parser.add_argument("--kaldi-manifest", type=Path, default=None, help="Kaldi manifest.json override.")
    parser.add_argument("--data-index", type=Path, nargs="+", default=None, help="Optional NPZ index CSV override.")
    parser.add_argument("--preset-path", type=Path, default=None, help="Optional NPZ preset pickle override.")
    parser.add_argument(
        "--override-dataset-names",
        type=str,
        nargs="+",
        default=None,
        help="Optional dataset/source list override.",
    )
    return parser.parse_args(argv)


def main(argv: t.Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest_path = run_extraction(parse_args(argv), namespace=PACKAGE_NAMESPACE)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
