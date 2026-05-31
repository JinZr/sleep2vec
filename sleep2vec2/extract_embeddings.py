from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
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
MANIFEST_FORMAT_VERSION = 1

MANIFEST_COLUMNS = (
    "sample_key",
    "path",
    "source",
    "dataset",
    "split",
    "token_start",
    "token_end",
    "num_tokens",
    "available_channels",
)

_KALDI_SAMPLE_KEY_RE = re.compile(r".*_\d{6}_\d{6}$")


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
    args.max_tokens = data_cfg.max_tokens
    if config_kind == "finetune":
        args.data_channel_names = data_cfg.data_channel_names or args.channel_names
        if set(args.data_channel_names) != set(args.channel_names):
            raise ValueError(
                "data.data_channel_names in YAML must match model.channels for embedding extraction. "
                f"Model channels: {args.channel_names}; data channels: {args.data_channel_names}."
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
    preset_required_channels, preset_min_channels = _load_preset_build_block(config_data)
    if preset_required_channels is None:
        args.dataset_channel_names = list(args.channel_names)
        args.dataset_channel_input_dims = dict(args.channel_input_dims)
    else:
        preset_channels, preset_dims = _resolve_validation_channels(
            model_channels=list(args.channel_names),
            channel_input_dims=dict(args.channel_input_dims),
            preset_required_channels=preset_required_channels,
            selected_channels=None,
        )
        _resolve_effective_min_channels(
            channel_names=preset_channels,
            cli_min_channels=len(preset_channels),
            preset_min_channels=preset_min_channels,
        )
        args.dataset_channel_names = list(dict.fromkeys([*args.channel_names, *preset_channels]))
        args.dataset_channel_input_dims = {**dict(args.channel_input_dims), **preset_dims}
    return bundle, model_cfg, config_kind


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

    dataset = PSGPretrainDataset(
        **dataset_kwargs,
        save_preset_path=None,
        load_preset_path=args.preset_path,
        index=args.data_index,
        stride_tokens=args.max_tokens,
    )
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
    if load_info.unexpected_keys:
        raise ValueError(
            "Checkpoint contains keys incompatible with the configured backbone: " f"{list(load_info.unexpected_keys)}"
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
        channel_feature_dim=None,
        transformer_hidden_size=model_cfg.backbone.hidden_size,
        transformer_num_hidden_layers=model_cfg.backbone.num_hidden_layers,
        transformer_num_attention_heads=model_cfg.backbone.num_attention_heads,
        channel_names=[c.name for c in model_cfg.channels],
        projection=model_cfg.projection.enabled,
        encoder_factory=None,
        model_config=model_cfg,
        projection_config=model_cfg.projection,
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
) -> list[np.ndarray]:
    cls_embedding = getattr(model, "cls_embedding", None)
    if cls_embedding is not None:
        token_hidden, _, _ = cls_embedding.split_hidden(hidden, attention_mask)
    else:
        token_hidden = hidden

    rows: list[np.ndarray] = []
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
) -> tuple[list[np.ndarray], int]:
    kwargs: dict[str, t.Any] = {"return_hidden_states": True}
    params = inspect.signature(model._token_embeddings_to_hidden).parameters
    if "modality_name" in params:
        kwargs["modality_name"] = channel_name

    if getattr(model, "_extract_separate_adapters", False):
        encoder = model.get_encoder() if hasattr(model, "get_encoder") else getattr(model, "encoder", None)
        if not hasattr(encoder, "set_adapter"):
            raise ValueError("Configured separate adapters, but the backbone encoder does not support set_adapter.")
        encoder.set_adapter(f"ch_{channel_name}")

    _, attention_mask, hidden_states = model._token_embeddings_to_hidden(token_embeddings, batch, **kwargs)
    selected_state, resolved_layer_index = _select_layer_state(hidden_states, layer_index, num_hidden_layers)
    return _trim_hidden_to_numpy(model, selected_state, attention_mask, batch["length"]), resolved_layer_index


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
    return (
        f"{_sanitize_key_part(source_value)}_"
        f"{_record_key_from_metadata(record_key_value, session_id_value, path_value)}_"
        f"{token_start:06d}_{token_end:06d}"
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
            "hidden_size": int(hidden_size),
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
                token_embeddings_by_channel = model._tokenize_all(batch["tokens"])
                channel_matrices: dict[str, list[np.ndarray]] = {}
                for channel in channel_names:
                    matrices, current_layer_index = _encode_channel(
                        model,
                        batch,
                        channel,
                        token_embeddings_by_channel[channel],
                        int(args.layer_index),
                        int(model_cfg.backbone.num_hidden_layers),
                    )
                    channel_matrices[channel] = matrices
                    resolved_layer_index = current_layer_index

                batch_size = len(batch["id"])
                source_values = _metadata_values(batch, "source", batch_size)
                path_values = _metadata_values(batch, "path", batch_size)
                dataset_values = _metadata_values(batch, "dataset", batch_size, default=None)
                token_starts = batch["token_start"].detach().cpu().tolist()
                ids = list(batch["id"])

                for sample_idx in range(batch_size):
                    num_tokens = int(channel_matrices[channel_names[0]][sample_idx].shape[0])
                    token_start = int(token_starts[sample_idx])
                    token_end = token_start + num_tokens
                    source_value = source_values[sample_idx]
                    path_value = path_values[sample_idx]
                    sample_metadata = metadata_by_id.get(str(ids[sample_idx]), {})
                    source_value = _metadata_value(sample_metadata, "source", source_value)
                    path_value = _metadata_value(sample_metadata, "path", path_value)
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
                        matrix = channel_matrices[channel][sample_idx]
                        if matrix.shape[0] != num_tokens:
                            raise ValueError(f"Channel {channel!r} produced a mismatched token count for {sample_key}.")
                        if args.output_format == "kaldi":
                            kaldi_writers[channel].write(sample_key, matrix)
                        else:
                            npz_path = output_dir / "channels" / split / channel / f"{sample_key}.npz"
                            np.savez(npz_path, embedding=matrix)

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
                            "available_channels": json.dumps(channel_names),
                        }
                    )
                    sample_count += 1

    if sample_count == 0:
        raise ValueError("No samples were exported.")

    manifest = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "namespace": namespace,
        "config_path": str(args.config),
        "ckpt_path": str(args.ckpt_path),
        "checkpoint_kind": load_plan.checkpoint_kind,
        "checkpoint_prefix": load_plan.checkpoint_prefix,
        "output_format": args.output_format,
        "layer_index": int(args.layer_index),
        "resolved_layer_index": int(resolved_layer_index if resolved_layer_index is not None else args.layer_index),
        "hidden_size": hidden_size,
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

    if not args.config.exists():
        raise FileNotFoundError(f"Config YAML not found: {args.config}")
    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    bundle, model_cfg, config_kind = _load_config_bundle(args)
    dataloader = _build_extraction_loader(args, bundle, config_kind)
    adapters_enabled = _finetune_adapters_enabled(bundle, config_kind)
    model = _build_backbone(model_cfg, args.device, bundle=bundle, config_kind=config_kind)
    load_plan = _load_backbone_checkpoint(
        model,
        args.ckpt_path,
        args.device,
        adapters_enabled=adapters_enabled,
    )
    return _extract_and_write_embeddings(args, model, dataloader, model_cfg, load_plan, namespace=namespace)


def parse_args(argv: t.Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract token-level backbone embeddings from a trained checkpoint.")
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
