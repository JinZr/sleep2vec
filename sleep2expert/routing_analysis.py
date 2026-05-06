from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import sys
import typing as t

import torch
from tqdm import tqdm

# Make sure the repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleep2expert.checkpoints import average_checkpoints, load_checkpoint, select_checkpoints
from sleep2expert.common import apply_finetune_config, remap_stage_labels
from sleep2expert.infer import _build_inference_loader
from sleep2expert.sleep2vec_finetuning import Sleep2vecFinetuning
from sleep2expert.utils import move_to_device

ROUTING_CSV_COLUMNS = [
    "sample_id",
    "source",
    "path",
    "token_start",
    "modality",
    "layer_idx",
    "expert_id",
    "expert_group",
    "usage_count",
    "mean_router_prob",
    "router_entropy",
    "label_name",
    "label_value_if_available",
    "analysis_tag",
    "split",
]


def run_routing_analysis(args: argparse.Namespace) -> list[dict[str, t.Any]]:
    pretrained_only = bool(getattr(args, "pretrained_only", False))
    if pretrained_only:
        if getattr(args, "ckpt_path", None):
            raise ValueError("--pretrained-only cannot be combined with --ckpt-path.")
        pretrained_backbone_path = getattr(args, "pretrained_backbone_path", None)
        if not pretrained_backbone_path:
            raise ValueError("--pretrained-only requires --pretrained-backbone-path.")
        if not Path(pretrained_backbone_path).exists():
            raise FileNotFoundError(f"Pretrained backbone checkpoint not found: {pretrained_backbone_path}")
    elif not getattr(args, "ckpt_path", None):
        raise ValueError("Routing analysis requires --ckpt-path unless --pretrained-only is set.")

    config_bundle, model_cfg = apply_finetune_config(args)
    moe_cfg = getattr(model_cfg.backbone, "moe", None)
    if moe_cfg is None or not getattr(moe_cfg, "enabled", False):
        raise ValueError("Routing analysis requires model.backbone.moe.enabled=true.")

    dataloader = _build_inference_loader(args)
    module = Sleep2vecFinetuning(
        args,
        model_cfg,
        finetune_config=config_bundle.finetune,
        averaging_config=config_bundle.averaging,
    )
    if not pretrained_only:
        _load_analysis_weights(module, args)
    module = module.to(torch.device(args.device))
    module.eval()

    expert_groups = _build_expert_group_lookup(moe_cfg)
    rows: list[dict[str, t.Any]] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Exporting routing ({args.eval_split})", unit="batch"):
            batch = move_to_device(batch, args.device)
            eval_model = module._get_eval_model()
            eval_model(batch)
            moe_aux = getattr(eval_model.backbone, "last_moe_aux", None)
            if not moe_aux:
                raise ValueError("MoE routing analysis expected backbone.last_moe_aux after downstream eval forward.")
            rows.extend(build_routing_rows(moe_aux, batch, args, expert_groups))

    _write_rows(rows, Path(args.output))
    logging.info("Wrote %d routing rows to %s", len(rows), args.output)
    return rows


def build_routing_rows(
    moe_aux: t.Sequence[dict[str, t.Any]],
    batch: dict[str, t.Any],
    args: argparse.Namespace,
    expert_groups: dict[int, str],
) -> list[dict[str, t.Any]]:
    rows: list[dict[str, t.Any]] = []
    label_name = str(getattr(args, "label_name", ""))
    is_seq = bool(getattr(args, "is_seq", False))

    for record in moe_aux:
        modality = record.get("modality")
        aux_values = record.get("aux")
        if aux_values is None:
            continue
        if not isinstance(aux_values, (list, tuple)):
            aux_values = (aux_values,)

        for aux in aux_values:
            if aux is None:
                continue
            rows.extend(
                _build_aux_rows(
                    record,
                    aux,
                    batch,
                    label_name=label_name,
                    is_seq=is_seq,
                    args=args,
                    expert_groups=expert_groups,
                    modality=modality,
                )
            )
    return rows


def _build_aux_rows(
    record: dict[str, t.Any],
    aux,
    batch: dict[str, t.Any],
    *,
    label_name: str,
    is_seq: bool,
    args: argparse.Namespace,
    expert_groups: dict[int, str],
    modality: str | None,
) -> list[dict[str, t.Any]]:
    router_probs = aux.router_probs.detach()
    topk_indices = aux.topk_indices.detach()
    token_mask = _valid_attention_mask(record.get("attention_mask"), router_probs, batch)
    labels = _labels_for_batch(batch, args, label_name=label_name, is_seq=is_seq)

    router_probs, topk_indices, token_mask, labels = _align_token_axis(
        router_probs,
        topk_indices,
        token_mask,
        labels,
        batch,
    )
    entropy = _token_entropy(router_probs)

    rows: list[dict[str, t.Any]] = []
    batch_size = int(router_probs.size(0))
    for sample_idx in range(batch_size):
        sample_context = _sample_context(batch, sample_idx)
        if is_seq and labels is not None and labels.dim() >= 2:
            label_values = labels[sample_idx]
            valid_labels = label_values != -1
            sample_mask = token_mask[sample_idx] & valid_labels.to(device=token_mask.device)
            grouped_labels = sorted(
                {_scalar_value(value) for value in label_values[sample_mask.to(device=label_values.device)]},
                key=str,
            )
        else:
            sample_mask = token_mask[sample_idx]
            grouped_labels = [_scalar_value(labels[sample_idx]) if labels is not None and labels.dim() == 1 else ""]

        for label_value in grouped_labels:
            if is_seq and labels is not None and labels.dim() >= 2:
                label_tensor = labels[sample_idx]
                label_mask = torch.tensor(
                    [_scalar_value(value) == label_value for value in label_tensor],
                    device=sample_mask.device,
                    dtype=torch.bool,
                )
                group_mask = sample_mask & label_mask
            else:
                group_mask = sample_mask
            if not bool(group_mask.any()):
                continue

            selected_indices = topk_indices[sample_idx][group_mask]
            selected_router_probs = router_probs[sample_idx][group_mask]
            selected_entropy = entropy[sample_idx][group_mask]
            for expert_id in sorted(int(value) for value in selected_indices.unique().detach().cpu().tolist()):
                expert_mask = selected_indices == expert_id
                usage_count = int(expert_mask.sum().item())
                if usage_count == 0:
                    continue
                token_expert_mask = expert_mask.any(dim=-1)
                rows.append(
                    {
                        **sample_context,
                        "modality": "" if modality is None else str(modality),
                        "layer_idx": int(aux.layer_idx),
                        "expert_id": expert_id,
                        "expert_group": expert_groups.get(expert_id, ""),
                        "usage_count": usage_count,
                        "mean_router_prob": float(selected_router_probs[token_expert_mask, expert_id].mean().item()),
                        "router_entropy": float(selected_entropy[token_expert_mask].mean().item()),
                        "label_name": label_name,
                        "label_value_if_available": label_value,
                        "analysis_tag": getattr(args, "analysis_tag", ""),
                        "split": getattr(args, "eval_split", ""),
                    }
                )
    return rows


def _load_analysis_weights(module: Sleep2vecFinetuning, args: argparse.Namespace) -> None:
    ckpt_path = str(args.ckpt_path)
    if args.avg_ckpts > 1:
        ckpt_dir = args.avg_ckpt_dir
        end_ckpt = None if ckpt_path in {"best", "last"} else Path(ckpt_path)
        if ckpt_dir is None:
            if end_ckpt is None:
                raise ValueError("Use --avg-ckpt-dir when averaging with ckpt-path=best/last.")
            ckpt_dir = end_ckpt.parent
        ckpt_paths = select_checkpoints(Path(ckpt_dir), end_ckpt=end_ckpt, num_ckpts=int(args.avg_ckpts))
        logging.info("Averaging checkpoints: %s", ", ".join(str(path) for path in ckpt_paths))
        state_dict = average_checkpoints(ckpt_paths, device=torch.device("cpu"))
    else:
        if ckpt_path in {"best", "last"}:
            raise ValueError(
                "Routing analysis requires a concrete .ckpt path; best/last aliases need Lightning Trainer."
            )
        checkpoint = load_checkpoint(ckpt_path, device=torch.device("cpu"))
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError("Checkpoint payload must be a Lightning .ckpt with top-level 'state_dict'.")
        state_dict = checkpoint["state_dict"]

    missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logging.warning("Missing keys when loading routing-analysis checkpoint: %s", missing_keys)
    if unexpected_keys:
        logging.warning("Unexpected keys when loading routing-analysis checkpoint: %s", unexpected_keys)


def _build_expert_group_lookup(moe_cfg) -> dict[int, str]:
    names_by_expert: dict[int, list[str]] = {}
    for group_name, expert_ids in getattr(moe_cfg, "expert_groups", {}).items():
        for expert_id in expert_ids:
            names_by_expert.setdefault(int(expert_id), []).append(str(group_name))
    return {expert_id: "|".join(sorted(group_names)) for expert_id, group_names in names_by_expert.items()}


def _valid_attention_mask(attention_mask, router_probs: torch.Tensor, batch: dict[str, t.Any]) -> torch.Tensor:
    batch_size, seq_len = router_probs.shape[:2]
    if attention_mask is None:
        lengths = batch.get("length")
        if torch.is_tensor(lengths):
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=router_probs.device)
            for sample_idx in range(batch_size):
                valid_len = int(lengths[sample_idx].item())
                if seq_len == valid_len + 1:
                    mask[sample_idx, : valid_len + 1] = True
                else:
                    mask[sample_idx, : min(valid_len, seq_len)] = True
            return mask
        return torch.ones(batch_size, seq_len, dtype=torch.bool, device=router_probs.device)

    mask = attention_mask.detach() if torch.is_tensor(attention_mask) else torch.as_tensor(attention_mask)
    mask = mask.to(device=router_probs.device)
    if mask.dim() == 4:
        mask = mask[:, 0, 0, :].eq(0)
    elif mask.dim() == 3:
        mask = mask[:, 0, :]
        if mask.dtype.is_floating_point:
            mask = mask.eq(0) if mask.min() < 0 else mask.ne(0)
        else:
            mask = mask.to(torch.bool)
    elif mask.dim() == 2:
        mask = mask.to(torch.bool)
    else:
        raise ValueError(f"attention_mask should have 2, 3, or 4 dimensions; got {tuple(mask.shape)}")
    if mask.shape != router_probs.shape[:2]:
        raise ValueError(
            f"MoE aux attention mask shape {tuple(mask.shape)} does not match router shape "
            f"{tuple(router_probs.shape)}."
        )
    return mask


def _labels_for_batch(
    batch: dict[str, t.Any],
    args: argparse.Namespace,
    *,
    label_name: str,
    is_seq: bool,
) -> torch.Tensor | None:
    if is_seq:
        source_name = getattr(args, "label_source_name", label_name)
        tokens = batch.get("tokens", {})
        labels = tokens.get(source_name)
        if not torch.is_tensor(labels):
            return None
        labels = labels.detach()
        if labels.dim() == 3 and labels.size(-1) == 1:
            labels = labels.squeeze(-1)
        elif labels.dim() == 3:
            valid = labels != -1
            labels = torch.where(valid.any(dim=-1), (labels > 0).any(dim=-1).to(torch.long), -1)
        if labels.dim() != 2:
            return None
        if not getattr(args, "is_multilabel", False):
            labels = remap_stage_labels(labels, label_name)
        return labels

    metadata = batch.get("metadata", {})
    value = metadata.get(label_name) if isinstance(metadata, dict) else None
    if torch.is_tensor(value):
        return value.detach().view(-1)
    if isinstance(value, (list, tuple)):
        return torch.as_tensor(value)
    return None


def _align_token_axis(
    router_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    token_mask: torch.Tensor,
    labels: torch.Tensor | None,
    batch: dict[str, t.Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    seq_len = int(router_probs.size(1))
    reference_len = _reference_token_length(labels, batch)
    if reference_len is not None and seq_len == reference_len + 1:
        router_probs = router_probs[:, 1:]
        topk_indices = topk_indices[:, 1:]
        token_mask = token_mask[:, 1:]
        seq_len -= 1
    if labels is not None and labels.dim() >= 2 and labels.size(1) != seq_len:
        raise ValueError(f"Sequence label length {labels.size(1)} does not match routing length {seq_len}.")
    if token_mask.shape != router_probs.shape[:2]:
        raise ValueError(
            f"Token mask shape {tuple(token_mask.shape)} does not match routing shape "
            f"{tuple(router_probs.shape[:2])}."
        )
    return router_probs, topk_indices, token_mask, labels


def _reference_token_length(labels: torch.Tensor | None, batch: dict[str, t.Any]) -> int | None:
    if labels is not None and labels.dim() >= 2:
        return int(labels.size(1))
    tokens = batch.get("tokens", {})
    if isinstance(tokens, dict):
        for value in tokens.values():
            if torch.is_tensor(value) and value.dim() >= 2:
                return int(value.size(1))
    return None


def _token_entropy(router_probs: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(router_probs.dtype).eps
    return -(router_probs * router_probs.clamp_min(eps).log()).sum(dim=-1)


def _sample_context(batch: dict[str, t.Any], sample_idx: int) -> dict[str, t.Any]:
    metadata = batch.get("metadata", {})
    return {
        "sample_id": _sample_value(batch.get("id"), sample_idx, default=sample_idx),
        "source": _sample_value(metadata.get("source") if isinstance(metadata, dict) else None, sample_idx, default=""),
        "path": _sample_value(metadata.get("path") if isinstance(metadata, dict) else None, sample_idx, default=""),
        "token_start": _sample_value(batch.get("token_start"), sample_idx, default=""),
    }


def _sample_value(values, sample_idx: int, *, default):
    if values is None:
        return default
    if torch.is_tensor(values):
        if values.dim() == 0:
            return _scalar_value(values)
        return _scalar_value(values[sample_idx])
    if isinstance(values, (list, tuple)):
        return _scalar_value(values[sample_idx])
    return _scalar_value(values)


def _scalar_value(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return str(value.detach().cpu().tolist())
        value = value.detach().cpu().item()
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return "" if value is None else str(value)


def _write_rows(rows: list[dict[str, t.Any]], output: Path) -> None:
    if output.parent:
        output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=ROUTING_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in ROUTING_CSV_COLUMNS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sleep2expert MoE routing summaries to CSV.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config used for downstream finetuning.")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Concrete Lightning .ckpt path. Required unless --pretrained-only is set.",
    )
    parser.add_argument(
        "--label-name",
        type=str,
        required=True,
        help="Downstream label name used by the finetune config.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination routing summary CSV.")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size for routing export.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device used for model evaluation.")
    parser.add_argument(
        "--eval-split",
        "--split",
        dest="eval_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--override-dataset-names", type=str, nargs="+", default=None)
    parser.add_argument("--analysis-tag", type=str, default="")
    parser.add_argument(
        "--pretrained-only",
        action="store_true",
        help="Export routing from the pretrained backbone without loading a downstream checkpoint.",
    )
    parser.add_argument("--avg-ckpts", type=int, default=1, help="Average this many checkpoints before export.")
    parser.add_argument("--avg-ckpt-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=4523)
    parser.add_argument("--pretrained-backbone-path", type=str, default=None)
    args = parser.parse_args()

    # Sleep2vecFinetuning expects optimizer fields on args, but routing export never trains or builds an optimizer.
    args.lr = 1e-6
    args.weight_decay = 1e-5
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not args.pretrained_only and not args.ckpt_path:
        raise ValueError("Routing analysis requires --ckpt-path unless --pretrained-only is set.")
    if not args.pretrained_only and args.ckpt_path not in {"best", "last"}:
        ckpt_path = Path(args.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        args.ckpt_path = str(ckpt_path)
    run_routing_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
