from __future__ import annotations

import typing as t

import torch


def count_total_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def estimate_active_parameters_per_token(model_config) -> int:
    backbone = model_config.backbone
    moe_cfg = getattr(backbone, "moe", None)
    hidden_size = int(backbone.hidden_size)
    dense_hidden_size = _dense_intermediate_size(backbone)
    moe_layers = set(getattr(moe_cfg, "layer_indices", None) or []) if moe_cfg and moe_cfg.enabled else set()

    active_params = 0
    for layer_idx in range(1, int(backbone.num_hidden_layers) + 1):
        if layer_idx in moe_layers:
            expert_hidden_size = int(moe_cfg.expert_hidden_size or dense_hidden_size)
            active_params += int(moe_cfg.top_k) * _ffn_parameter_count(hidden_size, expert_hidden_size)
            active_params += hidden_size * int(moe_cfg.num_experts) + int(moe_cfg.num_experts)
        else:
            active_params += _ffn_parameter_count(hidden_size, dense_hidden_size)
    return active_params


def estimate_moe_ffn_active_flops(model_config, seq_len: int) -> int:
    backbone = model_config.backbone
    moe_cfg = getattr(backbone, "moe", None)
    hidden_size = int(backbone.hidden_size)
    dense_hidden_size = _dense_intermediate_size(backbone)
    moe_layers = set(getattr(moe_cfg, "layer_indices", None) or []) if moe_cfg and moe_cfg.enabled else set()

    flops_per_token = 0
    for layer_idx in range(1, int(backbone.num_hidden_layers) + 1):
        if layer_idx in moe_layers:
            expert_hidden_size = int(moe_cfg.expert_hidden_size or dense_hidden_size)
            flops_per_token += int(moe_cfg.top_k) * _ffn_flops_per_token(hidden_size, expert_hidden_size)
        else:
            flops_per_token += _ffn_flops_per_token(hidden_size, dense_hidden_size)
    return int(seq_len) * flops_per_token


def estimate_dense_equivalent_ffn_flops(model_config, seq_len: int) -> int:
    backbone = model_config.backbone
    hidden_size = int(backbone.hidden_size)
    intermediate_size = _dense_intermediate_size(backbone)
    return int(seq_len) * int(backbone.num_hidden_layers) * _ffn_flops_per_token(hidden_size, intermediate_size)


def summarize_expert_usage(moe_aux) -> dict[int, int]:
    usage: dict[int, int] = {}
    for aux in _iter_aux(moe_aux):
        topk_indices = getattr(aux, "topk_indices", None)
        if topk_indices is None:
            continue
        values, counts = torch.unique(topk_indices.detach().cpu(), return_counts=True)
        for expert_id, count in zip(values.tolist(), counts.tolist()):
            usage[int(expert_id)] = usage.get(int(expert_id), 0) + int(count)
    return usage


def _dense_intermediate_size(backbone) -> int:
    overrides = getattr(backbone, "config_overrides", None) or {}
    return int(overrides.get("intermediate_size", int(backbone.hidden_size) * 4))


def _ffn_parameter_count(hidden_size: int, intermediate_size: int) -> int:
    return hidden_size * intermediate_size + intermediate_size + intermediate_size * hidden_size + hidden_size


def _ffn_flops_per_token(hidden_size: int, intermediate_size: int) -> int:
    return 2 * hidden_size * intermediate_size + 2 * intermediate_size * hidden_size


def _iter_aux(moe_aux) -> t.Iterator[t.Any]:
    if moe_aux is None:
        return
    records = [moe_aux] if isinstance(moe_aux, dict) else moe_aux
    for record in records:
        if isinstance(record, dict):
            aux_values = record.get("aux")
            if aux_values is None:
                continue
            if not isinstance(aux_values, (list, tuple)):
                aux_values = (aux_values,)
            yield from aux_values
        else:
            yield record
