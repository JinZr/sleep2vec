import argparse
import logging
from pathlib import Path
import random
import sys

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torch
import wandb

# Make sure the repository root is importable when running this file directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleep2expert.backbones.roformer.moe import apply_route_expert_filter
from sleep2expert.checkpoints import average_checkpoints, select_checkpoints
from sleep2expert.common import apply_finetune_config
from sleep2expert.distributed import is_rank_zero_process
from sleep2expert.results import (
    DEFAULT_INFERENCE_RESULTS_ROOT,
    _route_filter_flat_metadata,
    _route_filter_payload,
    prepare_inference_result_paths,
    save_inference_manifest,
    save_multilabel_per_disease_metrics_csv,
    save_prediction_csv,
    save_result_csv,
    save_survival_per_disease_metrics_csv,
    set_route_filter_metadata,
)
from sleep2expert.sleep2vec_finetuning import Sleep2vecFinetuning
from sleep2expert.utils import _build_finetune_loader


def _build_inference_loader(args):
    """Create a single split dataloader for evaluation-only runs."""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.eval_split == "test":
        sources = args.override_dataset_names or args.test_dataset_names
    else:
        sources = args.override_dataset_names or args.train_dataset_names

    return _build_finetune_loader(
        args,
        split=[args.eval_split],
        sources=sources,
        shuffle=False,
        is_train_set=False,
    )


def _init_wandb(args):
    if not args.wandb:
        return None
    if not is_rank_zero_process():
        return None

    inference_preset_path = getattr(args, "inference_preset_path", None) or getattr(args, "finetune_preset_path", None)
    route_filter = _route_filter_flat_metadata(args)
    init_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_name,
        "entity": args.wandb_entity,
        "group": args.wandb_group,
        "id": args.wandb_id,
        "resume": "allow" if args.wandb_id else None,
        "mode": args.wandb_mode,
        "config": {
            "config": str(args.config),
            "ckpt_path": args.ckpt_path,
            "label_name": args.label_name,
            "eval_split": args.eval_split,
            "batch_size": args.batch_size,
            "devices": args.devices,
            "accelerator": args.accelerator,
            "precision": args.precision,
            "avg_ckpts": args.avg_ckpts,
            "inference_preset_path": str(inference_preset_path) if inference_preset_path is not None else None,
            **route_filter,
        },
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    return wandb.init(**init_kwargs)


def _log_inference_outputs_to_wandb(
    args,
    metrics,
    prediction_row_count,
    survival_per_disease_metric_count=0,
    multilabel_per_disease_metric_count=0,
):
    wandb.log(
        {
            **metrics,
            "prediction_row_count": prediction_row_count,
            "survival_per_disease_metric_count": survival_per_disease_metric_count,
            "multilabel_per_disease_metric_count": multilabel_per_disease_metric_count,
        }
    )
    if not getattr(args, "wandb_artifact", True):
        return

    # W&B caps artifact names at 128 chars; CSVs and the manifest keep the full prediction_run_id.
    run_id_hash = args.prediction_run_id.rsplit("__", 1)[-1]
    artifact = wandb.Artifact(
        f"inference-{args.timestamp_utc}__{args.inference_namespace}__{run_id_hash}",
        type="inference",
        metadata={"route_filter": _route_filter_payload(args)},
    )
    artifact.add_file(str(args.inference_metrics_csv_path), name="metrics.csv")
    artifact.add_file(str(args.inference_prediction_csv_path), name="predictions.csv")
    if survival_per_disease_metric_count:
        artifact.add_file(
            str(args.inference_survival_per_disease_metrics_csv_path),
            name="survival_per_disease_metrics.csv",
        )
    if multilabel_per_disease_metric_count:
        artifact.add_file(
            str(args.inference_multilabel_per_disease_metrics_csv_path),
            name="multilabel_per_disease_metrics.csv",
        )
    artifact.add_file(str(args.manifest_path), name="run_manifest.json")
    artifact.add_file(str(args.inference_overview_csv_path), name="overview.csv")
    wandb.log_artifact(artifact)


def run_inference(args):
    config_bundle, model_cfg = apply_finetune_config(args)
    inference_preset_path = getattr(args, "inference_preset_path", None)
    if inference_preset_path is not None:
        if getattr(args, "data_backend", "npz") == "kaldi":
            raise ValueError("Kaldi backend uses manifest.json; legacy NPZ preset pickles are unsupported.")
        args.finetune_preset_path = Path(inference_preset_path)
    if args.label_name == "ahi" and args.avg_ckpts > 1:
        raise ValueError(
            "AHI inference does not support average checkpoints because `ahi_eval_threshold` is checkpoint-specific."
        )

    trainer_precision = args.precision
    if args.accelerator == "cpu" and isinstance(trainer_precision, str) and "bf16" in trainer_precision:
        trainer_precision = 32

    strategy = (
        DDPStrategy(find_unused_parameters=True) if args.accelerator != "cpu" and len(args.devices) > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices if args.accelerator != "cpu" else None,
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        precision=trainer_precision,
    )

    dataloader = _build_inference_loader(args)
    model = Sleep2vecFinetuning(
        args,
        model_cfg,
        finetune_config=config_bundle.finetune,
        averaging_config=config_bundle.averaging,
    )

    ckpt_path = args.ckpt_path
    selected_ckpt_paths = None
    if args.avg_ckpts > 1:
        ckpt_dir = args.avg_ckpt_dir
        end_ckpt = None if args.ckpt_path in {"best", "last"} else Path(args.ckpt_path)
        if ckpt_dir is None:
            if end_ckpt is None:
                raise ValueError("Use --avg-ckpt-dir when averaging with ckpt-path=best/last.")
            ckpt_dir = end_ckpt.parent
        ckpt_paths = select_checkpoints(Path(ckpt_dir), end_ckpt=end_ckpt, num_ckpts=args.avg_ckpts)
        selected_ckpt_paths = ckpt_paths
        logging.info("Averaging checkpoints: %s", ", ".join(str(p) for p in ckpt_paths))
        avg_state = average_checkpoints(ckpt_paths, device=torch.device("cpu"))
        missing_keys, unexpected_keys = model.load_state_dict(avg_state, strict=False)
        if missing_keys:
            logging.warning("Missing keys when loading averaged checkpoint: %s", missing_keys)
        if unexpected_keys:
            logging.warning("Unexpected keys when loading averaged checkpoint: %s", unexpected_keys)
        ckpt_path = None

    route_expert_groups = getattr(args, "route_expert_groups", None)
    active_expert_ids = None
    if route_expert_groups:
        active_expert_ids = apply_route_expert_filter(model, model_cfg.backbone.moe, route_expert_groups)
    set_route_filter_metadata(args, route_expert_groups, active_expert_ids)

    prepare_inference_result_paths(
        args,
        namespace="sleep2expert",
        root=getattr(args, "results_root", DEFAULT_INFERENCE_RESULTS_ROOT),
        checkpoint_paths=selected_ckpt_paths,
    )
    logging.info("Running inference on split=%s with %s samples/batch", args.eval_split, args.batch_size)
    wandb_run = _init_wandb(args)
    try:
        test_results = trainer.test(model=model, ckpt_path=ckpt_path, dataloaders=dataloader)
        metrics = test_results[0] if test_results else {}
        resolved_ckpt_path = getattr(trainer, "ckpt_path", None)
        if args.ckpt_path in {"best", "last"} and resolved_ckpt_path not in (None, "", args.ckpt_path):
            args.ckpt_resolved_path = str(resolved_ckpt_path)
            prepare_inference_result_paths(
                args,
                namespace="sleep2expert",
                root=getattr(args, "results_root", DEFAULT_INFERENCE_RESULTS_ROOT),
                checkpoint_paths=selected_ckpt_paths,
                timestamp=args.timestamp_utc,
            )
        logging.info("Inference metrics: %s", metrics)

        prediction_rows = getattr(model, "prediction_rows", [])
        prediction_row_count = len(prediction_rows)
        survival_per_disease_metric_rows = getattr(model, "survival_per_disease_metric_rows", [])
        survival_per_disease_metric_count = len(survival_per_disease_metric_rows)
        multilabel_per_disease_metric_rows = getattr(model, "multilabel_per_disease_metric_rows", [])
        multilabel_per_disease_metric_count = len(multilabel_per_disease_metric_rows)
        save_result_csv(metrics, str(args.inference_metrics_csv_path), args)
        save_result_csv(metrics, str(args.inference_overview_csv_path), args)
        save_prediction_csv(prediction_rows, str(args.inference_prediction_csv_path), args)
        save_survival_per_disease_metrics_csv(
            survival_per_disease_metric_rows,
            str(args.inference_survival_per_disease_metrics_csv_path),
            args,
        )
        save_multilabel_per_disease_metrics_csv(
            multilabel_per_disease_metric_rows,
            str(args.inference_multilabel_per_disease_metrics_csv_path),
            args,
        )
        save_inference_manifest(args, metrics, prediction_row_count=prediction_row_count)
        if wandb_run is not None:
            _log_inference_outputs_to_wandb(
                args,
                metrics,
                prediction_row_count,
                survival_per_disease_metric_count,
                multilabel_per_disease_metric_count,
            )
    finally:
        if wandb_run is not None:
            primary_exc_active = sys.exc_info()[0] is not None
            try:
                wandb.finish()
            except BaseException as exc:
                if primary_exc_active:
                    logging.warning("wandb.finish() failed during inference cleanup: %s", exc)
                else:
                    raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run downstream sleep2vec inference from a fine-tuned checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config used for downstream finetuning.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Checkpoint (.ckpt) path or Lightning alias such as 'best'/'last'.",
    )
    parser.add_argument(
        "--label-name",
        type=str,
        required=True,
        help=(
            "downstream label to predict (built-ins: age, sex, stage3, stage4, stage5, ahi; "
            "custom labels require finetune.task in the YAML config)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size for inference dataloader.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[0],
        help="Device ids passed to Lightning Trainer.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "auto"],
        help="Device accelerator used by Lightning.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device string passed into models.")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate placeholder used by optimizer init.")
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay placeholder used by optimizer init.",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--override-dataset-names",
        type=str,
        nargs="+",
        default=None,
        help="Optional dataset name list to override YAML train/test lists.",
    )
    parser.add_argument(
        "--inference-preset-path",
        type=Path,
        default=None,
        help="Optional preset pickle path for this inference run; overrides data.finetune_preset_path from YAML.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_INFERENCE_RESULTS_ROOT,
        help="Root directory for inference result artifacts.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        help="Precision flag forwarded to Lightning Trainer.",
    )
    parser.add_argument(
        "--avg-ckpts",
        type=int,
        default=1,
        help="Average this many checkpoints before inference (1 disables averaging).",
    )
    parser.add_argument(
        "--avg-ckpt-dir",
        type=Path,
        default=None,
        help="Optional checkpoint directory for averaging (defaults to ckpt_path parent).",
    )
    parser.add_argument("--seed", type=int, default=4523, help="Random seed for dataloader shuffling.")
    parser.add_argument(
        "--pretrained-backbone-path",
        type=str,
        default=None,
        help="Optional pretrain-model init checkpoint to load before downstream weights.",
    )
    parser.add_argument(
        "--route-expert-groups",
        type=str,
        nargs="+",
        default=None,
        help="Optional MoE expert group names to keep active during inference.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (needed for confusion matrix logging).",
    )
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name.")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team.")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group name.")
    parser.add_argument("--wandb-id", type=str, default=None, help="W&B run id (for resume).")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="W&B mode override (online/offline/disabled).",
    )
    parser.add_argument(
        "--no-wandb-artifact",
        dest="wandb_artifact",
        action="store_false",
        default=True,
        help="Log inference metrics to W&B without uploading CSV artifacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.ckpt_path not in {"best", "last"}:
        ckpt_path = Path(args.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        args.ckpt_path = str(ckpt_path)
    run_inference(args)
