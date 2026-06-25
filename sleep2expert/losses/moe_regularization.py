from __future__ import annotations

import typing as t

import torch

from .base import LossOutput


def compute_moe_regularization(moe_aux, moe_cfg, batch, *, prefix: str | None = None) -> LossOutput:
    device, dtype = _context_device_dtype(moe_aux, batch)
    zero = torch.zeros((), device=device, dtype=dtype)
    if moe_cfg is None or not getattr(moe_cfg, "enabled", False):
        return LossOutput(loss=zero, metrics={}, extras={})
    if not moe_aux:
        raise ValueError("MoE regularization requires model.last_moe_aux when backbone.moe.enabled is true.")
    if getattr(moe_cfg, "expert_diversity_coef", 0.0) > 0:
        raise ValueError("backbone.moe.expert_diversity_coef is not supported yet and must be 0.0.")

    records = _normalize_records(moe_aux)
    if not records:
        raise ValueError("MoE regularization received no valid routing aux records.")
    routing_stats = [
        {"record": record, "aux": aux, **_routing_stats(record, aux, batch)}
        for record in records
        for aux in record["aux"]
    ]
    if not routing_stats:
        raise ValueError("MoE regularization received no MoE layer aux outputs.")

    load_balance_loss = _load_balance_loss(routing_stats, zero)
    modality_balance_loss = _modality_balance_loss(routing_stats, moe_cfg, zero)
    router_z_loss = _mean_scalars([stat["z_loss"] for stat in routing_stats], zero)
    entropy = _mean_scalars([stat["entropy"] for stat in routing_stats], zero)
    route_consistency_loss = _route_consistency_loss(records, moe_cfg, batch, zero)
    expert_diversity_loss = zero
    expert_usage_entropy = _expert_usage_entropy(routing_stats, zero)
    active_experts_per_token = _active_experts_per_token(routing_stats, zero)

    entropy_loss = -entropy
    total = zero
    total = total + float(moe_cfg.load_balance_coef) * load_balance_loss
    total = total + float(moe_cfg.modality_balance_coef) * modality_balance_loss
    total = total + float(moe_cfg.router_z_loss_coef) * router_z_loss
    total = total + float(moe_cfg.router_entropy_coef) * entropy_loss
    total = total + float(moe_cfg.route_consistency_coef) * route_consistency_loss

    metrics = {
        "moe_load_balance_loss": load_balance_loss.detach(),
        "moe_modality_balance_loss": modality_balance_loss.detach(),
        "moe_route_consistency_loss": route_consistency_loss.detach(),
        "moe_router_z_loss": router_z_loss.detach(),
        "moe_entropy": entropy.detach(),
        "moe_expert_diversity_loss": expert_diversity_loss.detach(),
        "moe_expert_usage_entropy": expert_usage_entropy.detach(),
        "moe_active_experts_per_token": active_experts_per_token.detach(),
    }
    if prefix:
        metrics = {f"{prefix}_{name}": value for name, value in metrics.items()}

    extras = {
        "load_balance_loss": load_balance_loss,
        "modality_balance_loss": modality_balance_loss,
        "route_consistency_loss": route_consistency_loss,
        "router_z_loss": router_z_loss,
        "router_entropy_loss": entropy_loss,
        "expert_diversity_loss": expert_diversity_loss,
    }
    return LossOutput(loss=total, metrics=metrics, extras=extras)


def compute_downstream_moe_regularization(moe_aux, reg_cfg, batch, *, prefix: str | None = None) -> LossOutput:
    device, dtype = _context_device_dtype(moe_aux, batch)
    zero = torch.zeros((), device=device, dtype=dtype)
    if reg_cfg is None or not getattr(reg_cfg, "enabled", False):
        return LossOutput(loss=zero, metrics={}, extras={})
    if not moe_aux:
        raise ValueError("Downstream MoE regularization requires model.backbone.last_moe_aux.")

    unsupported = {
        "route_consistency_coef": "downstream route consistency is not supported yet",
        "load_balance_coef": "downstream load balancing is not supported yet",
        "modality_balance_coef": "downstream modality balancing is not supported yet",
        "entropy_coef": "downstream entropy regularization is not supported yet",
    }
    for field_name, message in unsupported.items():
        if float(getattr(reg_cfg, field_name, 0.0)) > 0:
            raise ValueError(message)

    values = _downstream_moe_metric_values(
        moe_aux,
        batch,
        zero,
        missing_ok=False,
        empty_records_message="Downstream MoE regularization received no valid routing aux records.",
        empty_layers_message="Downstream MoE regularization received no MoE layer aux outputs.",
    )
    router_z_loss = values["router_z_loss"]
    entropy = values["entropy"]
    expert_usage_entropy = values["expert_usage_entropy"]
    active_experts_per_token = values["active_experts_per_token"]
    total = float(getattr(reg_cfg, "router_z_loss_coef", 0.0)) * router_z_loss

    metrics = {
        "downstream_moe_router_z_loss": router_z_loss.detach(),
        "downstream_moe_entropy": entropy.detach(),
        "downstream_moe_expert_usage_entropy": expert_usage_entropy.detach(),
        "downstream_moe_active_experts_per_token": active_experts_per_token.detach(),
    }
    if prefix:
        metrics = {f"{prefix}_{name}": value for name, value in metrics.items()}

    extras = {
        "router_z_loss": router_z_loss,
        "entropy": entropy,
        "expert_usage_entropy": expert_usage_entropy,
        "active_experts_per_token": active_experts_per_token,
    }
    return LossOutput(loss=total, metrics=metrics, extras=extras)


def compute_downstream_moe_metrics(moe_aux, batch, *, prefix: str | None = None) -> dict[str, torch.Tensor]:
    device, dtype = _context_device_dtype(moe_aux, batch)
    zero = torch.zeros((), device=device, dtype=dtype)
    values = _downstream_moe_metric_values(moe_aux, batch, zero, missing_ok=True)
    if not values:
        return {}

    metrics = {
        "downstream_moe_router_z_loss": values["router_z_loss"].detach(),
        "downstream_moe_entropy": values["entropy"].detach(),
        "downstream_moe_expert_usage_entropy": values["expert_usage_entropy"].detach(),
        "downstream_moe_active_experts_per_token": values["active_experts_per_token"].detach(),
    }
    if prefix:
        metrics = {f"{prefix}_{name}": value for name, value in metrics.items()}
    return metrics


def _downstream_moe_metric_values(
    moe_aux,
    batch,
    zero: torch.Tensor,
    *,
    missing_ok: bool,
    empty_records_message: str = "",
    empty_layers_message: str = "",
) -> dict[str, torch.Tensor]:
    if not moe_aux:
        if missing_ok:
            return {}
        raise ValueError("Downstream MoE regularization requires model.backbone.last_moe_aux.")

    records = _normalize_records(moe_aux)
    if not records:
        if missing_ok:
            return {}
        raise ValueError(empty_records_message)
    routing_stats = [
        {"record": record, "aux": aux, **_routing_stats(record, aux, batch)}
        for record in records
        for aux in record["aux"]
    ]
    if not routing_stats:
        if missing_ok:
            return {}
        raise ValueError(empty_layers_message)

    return {
        "router_z_loss": _mean_scalars([stat["z_loss"] for stat in routing_stats], zero),
        "entropy": _mean_scalars([stat["entropy"] for stat in routing_stats], zero),
        "expert_usage_entropy": _expert_usage_entropy(routing_stats, zero),
        "active_experts_per_token": _active_experts_per_token(routing_stats, zero),
    }


def _normalize_records(moe_aux) -> list[dict[str, t.Any]]:
    raw_records = [moe_aux] if isinstance(moe_aux, dict) else list(moe_aux)
    records: list[dict[str, t.Any]] = []
    for item in raw_records:
        if isinstance(item, dict):
            aux_values = item.get("aux")
            if aux_values is None:
                continue
            if not isinstance(aux_values, (list, tuple)):
                aux_values = (aux_values,)
            records.append(
                {
                    "modality": item.get("modality"),
                    "attention_mask": item.get("attention_mask"),
                    "aux": tuple(aux for aux in aux_values if aux is not None),
                }
            )
        else:
            records.append(
                {
                    "modality": getattr(item, "modality_name", None),
                    "attention_mask": None,
                    "aux": (item,),
                }
            )
    return records


def _load_balance_loss(routing_stats: list[dict[str, t.Any]], zero: torch.Tensor) -> torch.Tensor:
    losses = []
    for stat in routing_stats:
        load, importance = _balance_vectors(stat)
        load = _normalize_vector(load)
        importance = _normalize_vector(importance)
        losses.append(load.numel() * torch.sum(load * importance))
    return _mean_scalars(losses, zero)


def _modality_balance_loss(
    routing_stats: list[dict[str, t.Any]],
    moe_cfg,
    zero: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for stat in routing_stats:
        record = stat["record"]
        allowed_experts = None
        if getattr(moe_cfg, "use_modality_group_mask", False):
            modality_name = record.get("modality")
            if modality_name is None or modality_name not in getattr(moe_cfg, "modality_to_groups", {}):
                raise ValueError("MoE modality balance requires modality_name when modality group masks are enabled.")
            allowed_experts = sorted(
                {
                    expert_id
                    for group_name in moe_cfg.modality_to_groups[modality_name]
                    for expert_id in moe_cfg.expert_groups[group_name]
                }
            )
        load, importance = _balance_vectors(stat, allowed_experts=allowed_experts)
        load = _normalize_vector(load)
        importance = _normalize_vector(importance)
        losses.append(load.numel() * torch.sum(load * importance))
    return _mean_scalars(losses, zero)


def _route_consistency_loss(
    records: list[dict[str, t.Any]],
    moe_cfg,
    batch,
    zero: torch.Tensor,
) -> torch.Tensor:
    if float(getattr(moe_cfg, "route_consistency_coef", 0.0)) <= 0:
        return zero
    layers = getattr(moe_cfg, "route_consistency_layers", None) or []
    if not layers:
        raise ValueError("MoE route consistency requires route_consistency_layers when its coefficient is positive.")
    if len(records) < 2:
        raise ValueError("MoE route consistency requires two routing aux records.")

    first_by_layer = {int(aux.layer_idx): aux for aux in records[0]["aux"]}
    second_by_layer = {int(aux.layer_idx): aux for aux in records[1]["aux"]}
    losses = []
    skipped_incomparable = False
    for layer_idx in layers:
        first_aux = first_by_layer.get(int(layer_idx))
        second_aux = second_by_layer.get(int(layer_idx))
        if first_aux is None or second_aux is None:
            raise ValueError(f"MoE route consistency layer {layer_idx} is missing from routing aux records.")
        first_probs, first_mask = _router_probs_and_mask(records[0], first_aux, batch)
        second_probs, second_mask = _router_probs_and_mask(records[1], second_aux, batch)
        if first_probs.shape != second_probs.shape:
            raise ValueError(
                f"MoE route consistency requires matching router shapes; got "
                f"{tuple(first_probs.shape)} and {tuple(second_probs.shape)}."
            )
        common_experts = _common_expert_index(
            records[0],
            records[1],
            moe_cfg,
            num_experts=first_probs.size(-1),
            device=first_probs.device,
        )
        if common_experts.numel() == 0:
            skipped_incomparable = True
            continue
        first_probs = first_probs.index_select(-1, common_experts)
        second_probs = second_probs.index_select(-1, common_experts.to(device=second_probs.device))
        first_mass = first_probs.sum(dim=-1, keepdim=True)
        second_mass = second_probs.sum(dim=-1, keepdim=True)
        valid_mask = first_mask & second_mask
        valid_mask = valid_mask & first_mass.squeeze(-1).gt(0) & second_mass.squeeze(-1).gt(0)
        if not bool(valid_mask.any()):
            skipped_incomparable = True
            continue
        eps = torch.finfo(first_probs.dtype).eps
        first_probs = first_probs / first_mass.clamp_min(eps)
        second_probs = second_probs / second_mass.clamp_min(eps)
        losses.append(_js_divergence(first_probs[valid_mask], second_probs[valid_mask]).mean())
    if not losses:
        if skipped_incomparable:
            return zero
        raise ValueError("MoE route consistency found no valid tokens for configured layers.")
    return _mean_scalars(losses, zero)


def _router_probs_and_mask(record: dict[str, t.Any], aux, batch) -> tuple[torch.Tensor, torch.Tensor]:
    probs = _strip_cls_if_present(aux.router_probs, batch)
    mask = _strip_cls_if_present(_valid_attention_mask(record.get("attention_mask"), aux.router_probs), batch)
    if mask.shape != probs.shape[:2]:
        raise ValueError(
            f"MoE aux attention mask shape {tuple(mask.shape)} does not match router shape {tuple(probs.shape)}."
        )
    return probs, mask


def _routing_stats(record: dict[str, t.Any], aux, batch) -> dict[str, t.Any]:
    probs, mask = _router_probs_and_mask(record, aux, batch)
    logits = _strip_cls_if_present(aux.router_logits, batch)
    expert_mask = _strip_cls_if_present(aux.expert_mask, batch)
    if logits.shape[:2] != probs.shape[:2] or expert_mask.shape != probs.shape:
        raise ValueError("MoE routing aux tensors must share batch, sequence, and expert dimensions.")

    token_weight = mask.to(device=probs.device, dtype=probs.dtype)
    token_weight_expanded = token_weight.unsqueeze(-1)
    expert_load_mask = expert_mask.to(dtype=probs.dtype)
    load = (expert_load_mask * token_weight_expanded).sum(dim=(0, 1))
    importance = (probs * token_weight_expanded).sum(dim=(0, 1))
    valid_tokens = token_weight.sum().clamp_min(torch.finfo(probs.dtype).eps)
    required_expert_ids = tuple(int(expert_id) for expert_id in (getattr(aux, "required_expert_ids", ()) or ()))
    if required_expert_ids:
        required_index = torch.tensor(required_expert_ids, device=probs.device, dtype=torch.long)
        balance_expert_mask = expert_load_mask.clone()
        balance_logits = logits.clone()
        balance_expert_mask.index_fill_(-1, required_index, 0.0)
        balance_logits.index_fill_(-1, required_index, torch.finfo(balance_logits.dtype).min)
        balance_probs = torch.softmax(balance_logits, dim=-1)
    else:
        balance_expert_mask = expert_load_mask
        balance_probs = probs
    balance_load = (balance_expert_mask * token_weight_expanded).sum(dim=(0, 1))
    balance_importance = (balance_probs * token_weight_expanded).sum(dim=(0, 1))
    z_loss_per_token = torch.logsumexp(logits, dim=-1).pow(2)
    entropy_per_token = -(probs * probs.clamp_min(torch.finfo(probs.dtype).eps).log()).sum(dim=-1)
    return {
        "load": load,
        "importance": importance,
        "balance_load": balance_load,
        "balance_importance": balance_importance,
        "valid_tokens": valid_tokens,
        "required_expert_ids": required_expert_ids,
        "z_loss": (z_loss_per_token * token_weight).sum() / valid_tokens,
        "entropy": (entropy_per_token * token_weight).sum() / valid_tokens,
    }


def _valid_attention_mask(attention_mask, probs: torch.Tensor) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones(probs.shape[:2], device=probs.device, dtype=torch.bool)

    mask = attention_mask.to(device=probs.device)
    if mask.dim() == 4:
        mask = mask[:, 0, 0, :]
    elif mask.dim() == 3:
        mask = mask[:, 0, :]
    elif mask.dim() != 2:
        raise ValueError(f"attention_mask should have 2, 3, or 4 dimensions; got {tuple(mask.shape)}")

    if mask.dtype == torch.bool:
        return mask
    if mask.is_floating_point() and bool((mask < 0).any()):
        return mask.eq(0)
    return mask.gt(0)


def _strip_cls_if_present(values: torch.Tensor, batch) -> torch.Tensor:
    raw_token_len = _raw_token_length(batch)
    if raw_token_len is not None and values.dim() >= 2 and values.size(1) == raw_token_len + 1:
        return values[:, 1:]
    return values


def _js_divergence(first_probs: torch.Tensor, second_probs: torch.Tensor) -> torch.Tensor:
    midpoint = 0.5 * (first_probs + second_probs)
    eps = torch.finfo(first_probs.dtype).eps
    first_log = first_probs.clamp_min(eps).log()
    second_log = second_probs.clamp_min(eps).log()
    midpoint_log = midpoint.clamp_min(eps).log()
    first_kl = (first_probs * (first_log - midpoint_log)).sum(dim=-1)
    second_kl = (second_probs * (second_log - midpoint_log)).sum(dim=-1)
    return 0.5 * (first_kl + second_kl)


def _common_expert_index(
    first_record: dict[str, t.Any],
    second_record: dict[str, t.Any],
    moe_cfg,
    *,
    num_experts: int,
    device: torch.device,
) -> torch.Tensor:
    first_allowed = _allowed_expert_set(first_record, moe_cfg, num_experts)
    second_allowed = _allowed_expert_set(second_record, moe_cfg, num_experts)
    common = sorted(first_allowed & second_allowed)
    return torch.tensor(common, device=device, dtype=torch.long)


def _allowed_expert_set(record: dict[str, t.Any], moe_cfg, num_experts: int) -> set[int]:
    if not getattr(moe_cfg, "use_modality_group_mask", False):
        return set(range(num_experts))
    modality_name = record.get("modality")
    if modality_name is None or modality_name not in getattr(moe_cfg, "modality_to_groups", {}):
        raise ValueError("MoE route consistency requires modality_name when modality group masks are enabled.")
    return {
        expert_id
        for group_name in moe_cfg.modality_to_groups[modality_name]
        for expert_id in moe_cfg.expert_groups[group_name]
    }


def _expert_usage_entropy(routing_stats: list[dict[str, t.Any]], zero: torch.Tensor) -> torch.Tensor:
    entropies = []
    for stat in routing_stats:
        load, _ = _balance_vectors(stat)
        usage = _normalize_vector(load)
        entropies.append(-(usage * usage.clamp_min(torch.finfo(usage.dtype).eps).log()).sum())
    return _mean_scalars(entropies, zero)


def _active_experts_per_token(routing_stats: list[dict[str, t.Any]], zero: torch.Tensor) -> torch.Tensor:
    values = []
    for stat in routing_stats:
        load = stat["balance_load"].to(dtype=zero.dtype)
        valid_tokens = stat["valid_tokens"].to(dtype=zero.dtype)
        values.append(load.sum() / valid_tokens.clamp_min(torch.finfo(valid_tokens.dtype).eps))
    return _mean_scalars(values, zero)


def _balance_vectors(
    stat: dict[str, t.Any],
    *,
    allowed_experts: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    load = stat["balance_load"]
    importance = stat["balance_importance"]
    required_expert_ids = set(stat["required_expert_ids"])
    if allowed_experts is None and not required_expert_ids:
        return load, importance

    if allowed_experts is None:
        expert_ids = [expert_id for expert_id in range(load.numel()) if expert_id not in required_expert_ids]
    else:
        expert_ids = [int(expert_id) for expert_id in allowed_experts if int(expert_id) not in required_expert_ids]
    index = torch.tensor(expert_ids, device=load.device, dtype=torch.long)
    return load.index_select(0, index), importance.index_select(0, index.to(device=importance.device))


def _normalize_vector(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return values / values.sum().clamp_min(torch.finfo(values.dtype).eps)


def _mean_scalars(values: list[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    if not values:
        return zero
    return torch.stack([value.to(device=zero.device, dtype=zero.dtype) for value in values]).mean()


def _raw_token_length(batch) -> int | None:
    if isinstance(batch, t.Mapping):
        tokens = batch.get("tokens")
        if isinstance(tokens, t.Mapping):
            for value in tokens.values():
                if isinstance(value, torch.Tensor) and value.dim() >= 2:
                    return int(value.shape[1])
        lengths = batch.get("length")
        if isinstance(lengths, torch.Tensor) and lengths.numel() > 0:
            return int(lengths.max().item())
    return None


def _context_device_dtype(moe_aux, batch) -> tuple[torch.device, torch.dtype]:
    tensor = _find_tensor(moe_aux)
    if tensor is None:
        tensor = _find_tensor(batch)
    if tensor is None:
        return torch.device("cpu"), torch.float32
    dtype = tensor.dtype if tensor.is_floating_point() else torch.float32
    return tensor.device, dtype


def _find_tensor(value) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, t.Mapping):
        for item in value.values():
            found = _find_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_tensor(item)
            if found is not None:
                return found
    if hasattr(value, "router_probs"):
        return value.router_probs
    return None
