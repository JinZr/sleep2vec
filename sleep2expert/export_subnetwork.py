from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import typing as t

import torch
import yaml

from sleep2expert.backbones.roformer.moe import resolve_route_expert_ids
from sleep2expert.checkpoints import get_state_dict_from_checkpoint, load_checkpoint
from sleep2expert.config import load_finetune_config, load_pretrain_config

_EXPERT_KEY_RE = re.compile(r"^(?P<prefix>.*moe_ffn\.experts\.)(?P<expert_id>\d+)(?P<suffix>\..*)$")
_ROUTER_KEY_RE = re.compile(r"^.*moe_ffn\.router\.router\.(?:weight|bias)$")
_RESUME_STATE_KEYS = ("optimizer_states", "lr_schedulers")


def export_subnetwork(args: argparse.Namespace) -> dict[str, t.Any]:
    config_path = Path(args.config)
    ckpt_path = Path(args.ckpt_path)
    output_dir = Path(args.output_dir)

    raw_config, config_type, model_cfg = _load_config(config_path)
    moe_cfg = model_cfg.backbone.moe
    group_names = [str(group_name) for group_name in args.route_expert_groups]
    selected_expert_ids = resolve_route_expert_ids(moe_cfg, group_names)
    if selected_expert_ids is None:
        raise ValueError("--route-expert-groups must select at least one MoE expert group.")

    old_to_new = {old_id: new_id for new_id, old_id in enumerate(selected_expert_ids)}
    exported_config = _rewrite_config(raw_config, moe_cfg, group_names, selected_expert_ids, old_to_new)
    _validate_output_dir(output_dir)
    checkpoint = load_checkpoint(ckpt_path, device=torch.device("cpu"))
    exported_checkpoint, rewrite_stats = _rewrite_checkpoint(
        checkpoint,
        selected_expert_ids,
        old_to_new,
        exported_config,
    )

    _prepare_output_dir(output_dir)
    config_out = output_dir / "config.yaml"
    ckpt_out = output_dir / "model.ckpt"
    manifest_out = output_dir / "manifest.json"
    expert_map_out = output_dir / "expert_id_map.csv"

    config_out.write_text(yaml.safe_dump(exported_config, sort_keys=False))
    torch.save(exported_checkpoint, ckpt_out)
    _write_expert_id_map(expert_map_out, old_to_new)
    manifest = _build_manifest(
        config_path=config_path,
        ckpt_path=ckpt_path,
        output_dir=output_dir,
        config_type=config_type,
        group_names=group_names,
        selected_expert_ids=selected_expert_ids,
        old_to_new=old_to_new,
        rewrite_stats=rewrite_stats,
    )
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _load_config(path: Path):
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML must be a mapping.")

    # Keep raw YAML for faithful output, but use the typed loaders for existing schema validation.
    if isinstance(raw.get("finetune"), dict):
        bundle = load_finetune_config(path)
        config_type = "finetune"
    else:
        bundle = load_pretrain_config(path)
        config_type = "pretrain"

    moe_cfg = bundle.model.backbone.moe
    if moe_cfg is None or not getattr(moe_cfg, "enabled", False):
        raise ValueError("Subnetwork export requires model.backbone.moe.enabled=true.")
    return raw, config_type, bundle.model


def _rewrite_config(
    raw_config: dict[str, t.Any],
    moe_cfg,
    group_names: t.Sequence[str],
    selected_expert_ids: t.Sequence[int],
    old_to_new: dict[int, int],
) -> dict[str, t.Any]:
    exported = copy.deepcopy(raw_config)
    moe_block = exported["model"]["backbone"]["moe"]
    selected_groups = set(group_names)
    if len(selected_expert_ids) < int(moe_cfg.top_k):
        raise ValueError(
            f"Route expert groups {list(group_names)} select {len(selected_expert_ids)} experts, "
            f"but top_k={moe_cfg.top_k}. v2 export does not lower top_k."
        )

    expert_groups = {}
    for group_name in group_names:
        remapped_ids = [
            old_to_new[int(expert_id)]
            for expert_id in moe_cfg.expert_groups[group_name]
            if int(expert_id) in old_to_new
        ]
        if remapped_ids:
            expert_groups[group_name] = remapped_ids

    modality_to_groups = {}
    for modality_name, modality_groups in moe_cfg.modality_to_groups.items():
        # v2 preserves model.channels, so every configured modality must still be routable.
        kept_groups = [group_name for group_name in modality_groups if group_name in selected_groups]
        allowed_experts = {
            expert_id
            for group_name in kept_groups
            for expert_id in moe_cfg.expert_groups[group_name]
            if int(expert_id) in old_to_new
        }
        if len(allowed_experts) < int(moe_cfg.top_k):
            raise ValueError(
                f"Route expert groups {list(group_names)} leave modality/channel '{modality_name}' with "
                f"{len(allowed_experts)} experts, but top_k={moe_cfg.top_k}. v2 export does not crop channels."
            )
        modality_to_groups[modality_name] = kept_groups

    moe_block["num_experts"] = len(selected_expert_ids)
    moe_block["expert_groups"] = expert_groups
    moe_block["modality_to_groups"] = modality_to_groups
    return exported


def _rewrite_checkpoint(
    checkpoint: t.Any,
    selected_expert_ids: t.Sequence[int],
    old_to_new: dict[int, int],
    exported_config: dict[str, t.Any],
) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    state_dict = get_state_dict_from_checkpoint(checkpoint)
    new_state: dict[str, t.Any] = {}
    selected_index = torch.tensor(list(selected_expert_ids), dtype=torch.long)
    expert_keys_seen = 0
    expert_keys_dropped = 0
    router_keys_sliced = 0

    for key, value in state_dict.items():
        expert_match = _EXPERT_KEY_RE.match(key)
        if expert_match is not None:
            expert_keys_seen += 1
            old_id = int(expert_match.group("expert_id"))
            if old_id not in old_to_new:
                expert_keys_dropped += 1
                continue
            # Compact checkpoints need old expert IDs remapped into the small model's 0..N-1 ModuleList.
            new_key = f"{expert_match.group('prefix')}{old_to_new[old_id]}{expert_match.group('suffix')}"
            new_state[new_key] = value
            continue

        if _ROUTER_KEY_RE.match(key) and isinstance(value, torch.Tensor):
            if value.dim() == 0 or max(selected_expert_ids) >= value.shape[0]:
                raise ValueError(f"Router tensor '{key}' cannot be sliced for selected expert IDs.")
            # Learned router outputs are indexed by old expert ID; keep the same order as the compact remap.
            new_state[key] = value.index_select(0, selected_index.to(device=value.device))
            router_keys_sliced += 1
            continue

        new_state[key] = value

    if expert_keys_seen == 0:
        raise ValueError("Checkpoint state_dict does not contain sleep2expert MoE expert weights.")

    exported_checkpoint = dict(checkpoint)
    removed_resume_keys = [key for key in _RESUME_STATE_KEYS if key in exported_checkpoint]
    for key in removed_resume_keys:
        # Parameter shapes changed, so Lightning optimizer/scheduler resume state is no longer valid.
        exported_checkpoint.pop(key, None)
    exported_checkpoint["state_dict"] = new_state

    if "model_config" in exported_checkpoint or "model_config_yaml" in exported_checkpoint:
        # Keep exported checkpoints self-describing for downstream loading and inspection.
        exported_checkpoint["model_config"] = exported_config["model"]
        exported_checkpoint["model_config_yaml"] = yaml.safe_dump(exported_config["model"], sort_keys=True)

    return exported_checkpoint, {
        "state_keys_in": len(state_dict),
        "state_keys_out": len(new_state),
        "expert_keys_seen": expert_keys_seen,
        "expert_keys_dropped": expert_keys_dropped,
        "router_keys_sliced": router_keys_sliced,
        "removed_resume_keys": removed_resume_keys,
    }


def _prepare_output_dir(path: Path) -> None:
    _validate_output_dir(path)
    path.mkdir(parents=True, exist_ok=True)


def _validate_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"Output directory must be empty or absent: {path}")


def _write_expert_id_map(path: Path, old_to_new: dict[int, int]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["old_expert_id", "new_expert_id"], lineterminator="\n")
        writer.writeheader()
        for old_id, new_id in old_to_new.items():
            writer.writerow({"old_expert_id": old_id, "new_expert_id": new_id})


def _build_manifest(
    *,
    config_path: Path,
    ckpt_path: Path,
    output_dir: Path,
    config_type: str,
    group_names: t.Sequence[str],
    selected_expert_ids: t.Sequence[int],
    old_to_new: dict[int, int],
    rewrite_stats: dict[str, t.Any],
) -> dict[str, t.Any]:
    return {
        "exported_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "namespace": "sleep2expert",
        "config_type": config_type,
        "source_config": str(config_path),
        "source_checkpoint": str(ckpt_path),
        "output_dir": str(output_dir),
        "route_expert_groups": list(group_names),
        "selected_expert_ids": list(selected_expert_ids),
        "old_to_new": {str(old_id): new_id for old_id, new_id in old_to_new.items()},
        "dropped": {
            "expert_state_keys": rewrite_stats["expert_keys_dropped"],
            "removed_resume_keys": rewrite_stats["removed_resume_keys"],
        },
        "state_dict": {
            "keys_in": rewrite_stats["state_keys_in"],
            "keys_out": rewrite_stats["state_keys_out"],
            "expert_keys_seen": rewrite_stats["expert_keys_seen"],
            "router_keys_sliced": rewrite_stats["router_keys_sliced"],
        },
    }


def parse_args(argv: t.Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a compact sleep2expert MoE subnetwork.")
    parser.add_argument("--config", type=Path, required=True, help="sleep2expert pretrain or finetune YAML config.")
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        required=True,
        help="Concrete sleep2expert Lightning checkpoint path.",
    )
    parser.add_argument(
        "--route-expert-groups",
        type=str,
        nargs="+",
        required=True,
        help="MoE expert group names to keep in the exported route expert subnetwork.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Empty output directory for exported artifacts.")
    args = parser.parse_args(argv)
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    return args


def main() -> None:
    manifest = export_subnetwork(parse_args())
    print(f"Exported sleep2expert MoE subnetwork to {manifest['output_dir']}")


if __name__ == "__main__":
    main()
