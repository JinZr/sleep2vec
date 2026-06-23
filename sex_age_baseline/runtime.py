from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sleep2vec.common import persist_run_config_and_args
from sleep2vec.losses.cox import CoxPHLossVectorized
from sleep2vec.metrics import (
    compute_multilabel_classification_metrics,
    compute_multilabel_metrics_by_disease,
    compute_survival_c_index_by_disease,
)
from sleep2vec.results import (
    prepare_inference_result_paths,
    save_inference_manifest,
    save_multilabel_per_disease_metrics_csv,
    save_prediction_csv,
    save_result_csv,
    save_survival_per_disease_metrics_csv,
    save_training_run_manifest,
)

from .config import BaselineConfig
from .data import SexAgeDataset, load_split_dataset, make_dataloader
from .model import SexAgeMLP


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    prediction_rows: list[dict[str, object]]
    survival_per_disease_rows: list[dict[str, object]]
    multilabel_per_disease_rows: list[dict[str, object]]


def configure_result_args(args: Namespace, cfg: BaselineConfig) -> None:
    args.monitor = cfg.finetune.task.monitor
    args.monitor_mod = cfg.finetune.task.monitor_mod
    args.output_dim = cfg.finetune.task.output_dim
    args.is_seq = False
    args.is_survival = cfg.finetune.task.type == "survival"
    args.is_multilabel = cfg.finetune.task.type == "multilabel_classification"
    args.is_classification = False
    args.channel_names = []
    args.finetune_preset_path = cfg.data.finetune_preset_path
    if not hasattr(args, "inference_preset_path"):
        args.inference_preset_path = None
    args.survival = cfg.finetune.survival
    args.multilabel = cfg.finetune.multilabel
    args.task_family = cfg.finetune.task.type


def build_version_name(args: Namespace, cfg: BaselineConfig) -> str:
    if getattr(args, "version_name", None):
        return str(args.version_name)
    task_name = "cox" if cfg.finetune.task.type == "survival" else "multilabel"
    return f"sex-age-baseline-{task_name}-{args.label_name}"


def train_and_save(args: Namespace, cfg: BaselineConfig) -> None:
    configure_result_args(args, cfg)
    args.version = build_version_name(args, cfg)
    epochs = int(args.epochs)
    if epochs == 0 and not args.ckpt_path:
        raise ValueError("--epochs 0 requires --ckpt-path for sex_age_baseline evaluation.")
    _seed_everything(getattr(args, "seed", 4523))

    device = torch.device(args.device)
    run_dir = Path("log-finetune") / args.version
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"sex_age_baseline run directory already exists and is not empty: {run_dir}. "
            "Use a new --version-name or manually clear the existing directory."
        )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    persist_run_config_and_args(args, run_dir)

    model = SexAgeMLP(cfg).to(device)
    if args.ckpt_path:
        load_checkpoint(model, args.ckpt_path, device=device, cfg=cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loaded_splits = ["train", "val"] if epochs > 0 else []
    if args.test_after_fit:
        loaded_splits.append("test")
    if epochs > 0:
        train_set = _required_dataset(cfg, "train", loaded_splits=loaded_splits)
        val_set = _required_dataset(cfg, "val", loaded_splits=loaded_splits)
        train_loader = make_dataloader(
            train_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        val_loader = make_dataloader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    best_path = checkpoint_dir / "best.ckpt"
    last_path = checkpoint_dir / "last.ckpt"
    best_score: float | None = None
    best_metrics: dict[str, float] = {}
    patience = int(getattr(args, "patience", 100))
    stale_epochs = 0
    global_step = 0

    for epoch in range(epochs):
        train_loss, steps = _train_one_epoch(model, train_loader, cfg, optimizer, device, args)
        global_step += steps
        val_result = evaluate_model(model, val_loader, cfg, device=device, stage="val")
        epoch_metrics = {"train_loss": train_loss, **val_result.metrics}
        save_checkpoint(last_path, model, cfg, epoch=epoch, global_step=global_step, metrics=epoch_metrics)
        if (epoch + 1) % int(getattr(args, "ckpt_every_n_epochs", 1)) == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch={epoch:02d}.ckpt",
                model,
                cfg,
                epoch=epoch,
                global_step=global_step,
                metrics=epoch_metrics,
            )

        monitor = cfg.finetune.task.monitor
        if monitor not in epoch_metrics:
            available_metrics = ", ".join(sorted(epoch_metrics))
            raise ValueError(f"Configured monitor {monitor!r} was not emitted. Available metrics: {available_metrics}")
        monitor_value = epoch_metrics[monitor]
        if _is_better(monitor_value, best_score, cfg.finetune.task.monitor_mod):
            best_score = float(monitor_value)
            best_metrics = dict(epoch_metrics)
            stale_epochs = 0
            save_checkpoint(best_path, model, cfg, epoch=epoch, global_step=global_step, metrics=epoch_metrics)
        else:
            stale_epochs += 1
        if patience > 0 and stale_epochs >= patience:
            break

    if epochs == 0 and args.ckpt_path:
        best_path = Path(args.ckpt_path)
    elif epochs > 0 and not best_path.exists():
        raise ValueError(
            f"No finite best checkpoint was selected for monitor {cfg.finetune.task.monitor!r}. "
            "Check validation labels and monitor configuration."
        )

    manifest_path = run_dir / "run_manifest.json"
    if not args.test_after_fit:
        save_training_run_manifest(
            args,
            manifest_path=manifest_path,
            status="skipped_test",
            monitor=cfg.finetune.task.monitor,
            monitor_mode=cfg.finetune.task.monitor_mod,
            best_model_path=best_path if best_path.exists() else None,
            best_model_score=best_score,
            last_checkpoint_path=last_path if last_path.exists() else None,
            metrics=best_metrics,
        )
        return

    if best_path.exists():
        load_checkpoint(model, best_path, device=device, cfg=cfg)
        args.ckpt_path = str(best_path)
        args.ckpt_resolved_path = str(best_path)

    test_set = _required_dataset(cfg, "test", loaded_splits=loaded_splits)
    test_loader = make_dataloader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    args.eval_split = "test"
    test_result = evaluate_model(
        model,
        test_loader,
        cfg,
        device=device,
        stage="test",
        export_predictions=cfg.outputs.prediction_csv,
    )

    save_result_csv(test_result.metrics, str(args.results_csv_path), args)
    prediction_csv_path = run_dir / "predictions.csv"
    if cfg.outputs.prediction_csv:
        save_prediction_csv(test_result.prediction_rows, str(prediction_csv_path), args)
    survival_csv_path = None
    multilabel_csv_path = None
    if cfg.outputs.per_disease_metrics_csv and test_result.survival_per_disease_rows:
        survival_csv_path = run_dir / "survival_per_disease_metrics.csv"
        save_survival_per_disease_metrics_csv(test_result.survival_per_disease_rows, str(survival_csv_path), args)
    if cfg.outputs.per_disease_metrics_csv and test_result.multilabel_per_disease_rows:
        multilabel_csv_path = run_dir / "multilabel_per_disease_metrics.csv"
        save_multilabel_per_disease_metrics_csv(test_result.multilabel_per_disease_rows, str(multilabel_csv_path), args)
    save_training_run_manifest(
        args,
        manifest_path=manifest_path,
        status="completed",
        monitor=cfg.finetune.task.monitor,
        monitor_mode=cfg.finetune.task.monitor_mod,
        best_model_path=best_path if best_path.exists() else None,
        best_model_score=best_score,
        last_checkpoint_path=last_path if last_path.exists() else None,
        results_csv_path=args.results_csv_path,
        survival_per_disease_metrics_csv_path=survival_csv_path,
        multilabel_per_disease_metrics_csv_path=multilabel_csv_path,
        metrics=test_result.metrics,
    )


def run_inference_and_save(args: Namespace, cfg: BaselineConfig) -> None:
    cfg = _config_with_inference_preset(args, cfg)
    configure_result_args(args, cfg)
    _seed_everything(getattr(args, "seed", 4523))
    device = torch.device(args.device)
    model = SexAgeMLP(cfg).to(device)
    load_checkpoint(model, args.ckpt_path, device=device, cfg=cfg)
    args.ckpt_resolved_path = str(args.ckpt_path)
    prepare_inference_result_paths(args, namespace="sex_age_baseline")
    args.task_family = cfg.finetune.task.type

    dataset = _required_dataset(cfg, args.eval_split, loaded_splits=[args.eval_split])
    loader = make_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    result = evaluate_model(
        model,
        loader,
        cfg,
        device=device,
        stage=args.eval_split,
        export_predictions=cfg.outputs.prediction_csv,
    )
    save_result_csv(result.metrics, str(args.inference_metrics_csv_path), args)
    save_result_csv(result.metrics, str(args.inference_overview_csv_path), args)
    if cfg.outputs.prediction_csv:
        save_prediction_csv(result.prediction_rows, str(args.inference_prediction_csv_path), args)
    if cfg.outputs.per_disease_metrics_csv:
        save_survival_per_disease_metrics_csv(
            result.survival_per_disease_rows,
            str(args.inference_survival_per_disease_metrics_csv_path),
            args,
        )
        save_multilabel_per_disease_metrics_csv(
            result.multilabel_per_disease_rows,
            str(args.inference_multilabel_per_disease_metrics_csv_path),
            args,
        )
    save_inference_manifest(args, result.metrics, prediction_row_count=len(result.prediction_rows))


def _config_with_inference_preset(args: Namespace, cfg: BaselineConfig) -> BaselineConfig:
    preset_path = getattr(args, "inference_preset_path", None)
    if preset_path in (None, ""):
        return cfg
    if cfg.data.backend != "npz":
        raise ValueError("--inference-preset-path is only supported for data.backend=npz.")
    data = replace(cfg.data, finetune_data_index=None, finetune_preset_path=str(preset_path))
    return replace(cfg, data=data)


def evaluate_model(
    model: SexAgeMLP,
    loader,
    cfg: BaselineConfig,
    *,
    device: torch.device,
    stage: str,
    export_predictions: bool = False,
) -> EvaluationResult:
    model.eval()
    keys: list[str] = []
    logits_list: list[torch.Tensor] = []
    label_tensors: dict[str, list[torch.Tensor]] = {"has_label": []}
    task_type = cfg.finetune.task.type

    with torch.no_grad():
        for batch in loader:
            age = batch["age"].to(device)
            sex = batch["sex"].to(device)
            logits = model(age, sex).detach().cpu()
            logits_list.append(logits)
            keys.extend(str(key) for key in batch["key"])
            label_tensors["has_label"].append(batch["has_label"].detach().cpu())
            if task_type == "survival":
                label_tensors.setdefault("event_time", []).append(batch["event_time"].detach().cpu())
                label_tensors.setdefault("is_event", []).append(batch["is_event"].detach().cpu())
            else:
                label_tensors.setdefault("disease_label", []).append(batch["disease_label"].detach().cpu())

    if not logits_list:
        raise ValueError(f"Sex/age baseline split {stage!r} has no rows.")
    logits = torch.cat(logits_list, dim=0)
    has_label = torch.cat(label_tensors["has_label"], dim=0)
    if task_type == "survival":
        event_time = torch.cat(label_tensors["event_time"], dim=0)
        is_event = torch.cat(label_tensors["is_event"], dim=0)
        return _evaluate_survival(cfg, stage, keys, logits, event_time, is_event, has_label, export_predictions)
    labels = torch.cat(label_tensors["disease_label"], dim=0)
    return _evaluate_multilabel(cfg, stage, keys, logits, labels, has_label, export_predictions)


def masked_multilabel_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    has_label: torch.Tensor,
    *,
    pos_weight: Any | None = None,
) -> torch.Tensor:
    valid = has_label > 0.5
    if not valid.any():
        return logits.sum() * 0.0
    safe_labels = torch.where(valid, labels.float(), torch.zeros_like(labels.float()))
    weight = _pos_weight_tensor(pos_weight, logits) if pos_weight is not None else None
    losses = F.binary_cross_entropy_with_logits(logits, safe_labels, pos_weight=weight, reduction="none")
    return losses[valid].mean()


def save_checkpoint(
    path: str | Path,
    model: SexAgeMLP,
    cfg: BaselineConfig,
    *,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "label_contract": _label_contract(cfg),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "metrics": dict(metrics),
        },
        path,
    )


def load_checkpoint(
    model: SexAgeMLP,
    path: str | Path,
    *,
    device: torch.device,
    cfg: BaselineConfig | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if cfg is not None:
        _validate_checkpoint_label_contract(checkpoint, cfg, path)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    model.load_state_dict(state_dict, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {}


def _validate_checkpoint_label_contract(checkpoint: Any, cfg: BaselineConfig, path: str | Path) -> None:
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint does not contain a saved sex_age_baseline label contract: {path}")
    saved_contract = checkpoint.get("label_contract")
    if not isinstance(saved_contract, Mapping):
        saved_config = checkpoint.get("config")
        if not isinstance(saved_config, Mapping):
            raise ValueError(f"Checkpoint does not contain a saved sex_age_baseline config: {path}")
        saved_contract = _label_contract(saved_config)
    current_contract = _label_contract(cfg)
    if dict(saved_contract) != current_contract:
        raise ValueError(
            "Checkpoint label contract does not match current sex_age_baseline config: "
            f"checkpoint={dict(saved_contract)}, current={current_contract}."
        )


def _label_contract(cfg: BaselineConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, BaselineConfig):
        task_type = cfg.finetune.task.type
        output_dim = cfg.finetune.task.output_dim
        if task_type == "survival":
            disease_columns_index = cfg.finetune.survival.disease_columns_index
        else:
            disease_columns_index = cfg.finetune.multilabel.disease_columns_index
    else:
        finetune = cfg.get("finetune") if isinstance(cfg.get("finetune"), Mapping) else {}
        task = finetune.get("task") if isinstance(finetune.get("task"), Mapping) else {}
        task_type = task.get("type")
        output_dim = task.get("output_dim")
        if task_type == "survival":
            label_cfg = finetune.get("survival")
        elif task_type == "multilabel_classification":
            label_cfg = finetune.get("multilabel")
        else:
            label_cfg = None
        if not isinstance(label_cfg, Mapping):
            raise ValueError("Checkpoint config is missing task label configuration.")
        disease_columns_index = label_cfg.get("disease_columns_index")

    return {
        "task_type": str(task_type),
        "output_dim": int(output_dim),
        "label_names": _label_names(task_type, disease_columns_index),
    }


def _label_names(task_type: Any, disease_columns_index: Any) -> list[str]:
    if task_type == "survival":
        from data.survival import load_survival_disease_columns

        return load_survival_disease_columns(disease_columns_index)
    if task_type == "multilabel_classification":
        from data.multilabel import load_multilabel_disease_columns

        return load_multilabel_disease_columns(disease_columns_index)
    raise ValueError(f"Unsupported sex_age_baseline checkpoint task type: {task_type}")


def _train_one_epoch(
    model: SexAgeMLP,
    loader,
    cfg: BaselineConfig,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: Namespace,
) -> tuple[float, int]:
    model.train()
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    accum = max(1, int(getattr(args, "accumulate_grad_batches", 1)))
    steps = 0
    for batch_idx, batch in enumerate(loader, start=1):
        logits = model(batch["age"].to(device), batch["sex"].to(device))
        if cfg.finetune.task.type == "survival":
            loss = CoxPHLossVectorized()(
                logits,
                batch["has_label"].to(device),
                batch["event_time"].to(device),
                batch["is_event"].to(device),
            )
        else:
            loss = masked_multilabel_bce(
                logits,
                batch["disease_label"].to(device),
                batch["has_label"].to(device),
                pos_weight=cfg.finetune.loss.pos_weight if cfg.finetune.loss else None,
            )
        (loss / accum).backward()
        if batch_idx % accum == 0 or batch_idx == len(loader):
            clip_val = getattr(args, "gradient_clip_val", None)
            if clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_val))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan"), steps


def _evaluate_survival(
    cfg: BaselineConfig,
    stage: str,
    keys: list[str],
    logits: torch.Tensor,
    event_time: torch.Tensor,
    is_event: torch.Tensor,
    has_label: torch.Tensor,
    export_predictions: bool,
) -> EvaluationResult:
    loss = CoxPHLossVectorized()(logits, has_label, event_time, is_event)
    disease_names = _survival_disease_names(cfg)
    metric_rows = compute_survival_c_index_by_disease(logits, event_time, is_event, has_label, disease_names)
    for row in metric_rows:
        row["stage"] = stage
    c_indices = [row["c_index"] for row in metric_rows if np.isfinite(row["c_index"])]
    metrics = {
        f"{stage}_loss": float(loss.detach().cpu()),
        f"{stage}_c_index": float(np.mean(c_indices)) if c_indices else float("nan"),
    }
    rows = (
        _survival_prediction_rows(keys, logits, event_time, is_event, has_label, disease_names)
        if export_predictions
        else []
    )
    return EvaluationResult(metrics, rows, metric_rows, [])


def _evaluate_multilabel(
    cfg: BaselineConfig,
    stage: str,
    keys: list[str],
    logits: torch.Tensor,
    labels: torch.Tensor,
    has_label: torch.Tensor,
    export_predictions: bool,
) -> EvaluationResult:
    loss = masked_multilabel_bce(logits, labels, has_label, pos_weight=cfg.finetune.loss.pos_weight)
    disease_names = _multilabel_disease_names(cfg)
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    has_label_np = has_label.numpy()
    metric_rows = compute_multilabel_metrics_by_disease(labels_np, probs, has_label_np, disease_names)
    for row in metric_rows:
        row["stage"] = stage
    metrics = {f"{stage}_loss": float(loss.detach().cpu())}
    metrics.update(
        {
            f"{stage}_{key}": float(value)
            for key, value in compute_multilabel_classification_metrics(labels_np, probs, has_label_np).items()
        }
    )
    rows = (
        _multilabel_prediction_rows(keys, logits.numpy(), labels_np, has_label_np, disease_names)
        if export_predictions
        else []
    )
    return (
        EvaluationResult(metrics, [], [], metric_rows)
        if not export_predictions
        else EvaluationResult(metrics, rows, [], metric_rows)
    )


def _survival_prediction_rows(
    keys: list[str],
    logits: torch.Tensor,
    event_time: torch.Tensor,
    is_event: torch.Tensor,
    has_label: torch.Tensor,
    disease_names: list[str],
) -> list[dict[str, object]]:
    pred = logits.numpy()
    times = event_time.numpy()
    events = (is_event.numpy() > 0.5).astype(np.int64)
    masks = (has_label.numpy() > 0.5).astype(np.int64)
    return [
        {
            "path": key,
            "survival_key": key,
            "kind": "survival",
            "disease_names": list(disease_names),
            "groundtruth": {
                "event_time": times[idx].tolist(),
                "is_event": events[idx].tolist(),
                "has_label": masks[idx].tolist(),
            },
            "prediction": pred[idx].tolist(),
            "log_risk": pred[idx].tolist(),
            "event_time": times[idx].tolist(),
            "is_event": events[idx].tolist(),
            "has_label": masks[idx].tolist(),
            "n_predictions": int(pred.shape[1]),
            "n_windows": 1,
            "token_starts": [0],
        }
        for idx, key in enumerate(keys)
    ]


def _multilabel_prediction_rows(
    keys: list[str],
    logits: np.ndarray,
    labels: np.ndarray,
    has_label: np.ndarray,
    disease_names: list[str],
) -> list[dict[str, object]]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    masks = (has_label > 0.5).astype(np.int64)
    return [
        {
            "path": key,
            "paths": [key],
            "multilabel_key": key,
            "kind": "multilabel_classification",
            "disease_names": list(disease_names),
            "groundtruth": labels[idx].tolist(),
            "prediction": (probs[idx] >= 0.5).astype(np.int64).tolist(),
            "probability": probs[idx].tolist(),
            "logit": logits[idx].tolist(),
            "has_label": masks[idx].tolist(),
            "n_predictions": int(logits.shape[1]),
            "n_windows": 1,
            "token_starts": [0],
        }
        for idx, key in enumerate(keys)
    ]


def _survival_disease_names(cfg: BaselineConfig) -> list[str]:
    from data.survival import load_survival_disease_columns

    return load_survival_disease_columns(cfg.finetune.survival.disease_columns_index)


def _multilabel_disease_names(cfg: BaselineConfig) -> list[str]:
    from data.multilabel import load_multilabel_disease_columns

    return load_multilabel_disease_columns(cfg.finetune.multilabel.disease_columns_index)


def _required_dataset(cfg: BaselineConfig, split: str, *, loaded_splits: list[str] | None = None) -> SexAgeDataset:
    dataset = load_split_dataset(cfg, split, loaded_splits=loaded_splits)
    if len(dataset) == 0:
        raise ValueError(f"Sex/age baseline split {split!r} has no rows.")
    return dataset


def _is_better(value: Any, best: float | None, mode: str) -> bool:
    if value is None:
        return False
    value = float(value)
    if not np.isfinite(value):
        return False
    if best is None:
        return True
    return value < best if mode == "min" else value > best


def _pos_weight_tensor(pos_weight: Any, logits: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    if tensor.ndim == 0:
        return tensor
    if tensor.numel() != logits.shape[1]:
        raise ValueError("finetune.loss.pos_weight must be scalar or match output_dim.")
    return tensor.view(-1)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    return timestamp
