from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
import copy
from dataclasses import asdict, dataclass, replace
import logging
import os
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler
import wandb

from sleep2vec.checkpoints import average_checkpoints, select_checkpoints
from sleep2vec.common import persist_run_config_and_args
from sleep2vec.distributed import is_rank_zero_process
from sleep2vec.infer import _init_wandb, _log_inference_outputs_to_wandb
from sleep2vec.losses.cox import CoxPHLossVectorized
from sleep2vec.metrics.core import (
    compute_multilabel_classification_metrics,
    compute_multilabel_metrics_by_disease,
    compute_survival_c_index_by_disease,
)
from sleep2vec.modules.covariates import ordered_covariates
from sleep2vec.results import (
    DEFAULT_INFERENCE_RESULTS_ROOT,
    prepare_inference_result_paths,
    save_inference_manifest,
    save_multilabel_per_disease_metrics_csv,
    save_prediction_csv,
    save_result_csv,
    save_result_rows_csv,
    save_survival_per_disease_metrics_csv,
    save_training_run_manifest,
)
from sleep2vec.schedulers import build_warmup_cosine_scheduler, validate_finetune_scheduler_args
from sleep2vec.sleep2vec_inference import prediction_export_enabled

from .common import apply_task_flags
from .config import BaselineConfig, covariate_task_config
from .data import SexAgeDataset, load_split_dataset, make_dataloader
from .model import SexAgeMLP

# Like sleep2vec finetuning, training seeds from a fixed value; only inference exposes --seed.
FINETUNE_SEED = 4523


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    prediction_rows: list[dict[str, object]]
    survival_per_disease_rows: list[dict[str, object]]
    multilabel_per_disease_rows: list[dict[str, object]]


class LastCheckpoint(Callback):
    def __init__(self, path: Path):
        self.path = path

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.save_checkpoint(self.path)


class BaselineModule(pl.LightningModule):
    """Covariate training with the same step-based optimization and subject evaluation contract."""

    def __init__(self, cfg: BaselineConfig, args: Namespace):
        super().__init__()
        self.cfg = cfg
        self.args = args
        self.model = SexAgeMLP(cfg)
        self.records = []
        self.evaluation_result = None
        self.evaluation_stage = "test"
        self._validation_epoch = None

    def forward(self, metadata):
        return self.model(metadata)

    def train_dataloader(self):
        # Build after rank setup; drop the global tail instead of padding training identities.
        sampler = None
        if self.trainer.world_size > 1:
            sampler = DistributedSampler(
                self.train_set,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True,
                seed=FINETUNE_SEED,
                drop_last=True,
            )
        return make_dataloader(
            self.train_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=sampler is None,
            drop_last=sampler is not None,
            sampler=sampler,
        )

    def training_step(self, batch, batch_idx):
        logits = self(batch["metadata"])
        loss = _batch_loss(logits, batch, self.cfg)
        if self.cfg.finetune.task.type == "multilabel_classification" and self.trainer.world_size > 1:
            valid_count = (batch["has_label"] > 0.5).sum()
            global_count = valid_count.clone()
            torch.distributed.all_reduce(global_count)
            # DDP averages gradients; scale the local mean to the global valid-entry mean.
            loss = loss * (valid_count * self.trainer.world_size / global_count.clamp_min(1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=logits.shape[0], sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.records = []

    def validation_step(self, batch, batch_idx):
        self.records.append(_evaluation_record(batch, self(batch["metadata"])))

    def on_validation_epoch_end(self):
        result = _evaluate_records(self.records, self.cfg, "val", False)
        monitor = self.cfg.finetune.task.monitor
        if monitor not in result.metrics:
            raise ValueError(
                f"Configured monitor {monitor!r} was not emitted. Available metrics: {sorted(result.metrics)}"
            )
        if not np.isfinite(result.metrics[monitor]):
            raise ValueError(f"No finite best checkpoint can be selected for monitor {monitor!r}.")
        for name, value in result.metrics.items():
            self.log(name, value, sync_dist=False)
        self.evaluation_result = result
        self._validation_epoch = self.current_epoch
        if (
            getattr(self.args, "lr_scheduler", "decay") == "plateau"
            and self.trainer.state.fn == "fit"
            and not self.trainer.sanity_checking
        ):
            # Consume this validation result before ModelCheckpoint saves optimizer and scheduler state.
            self.lr_schedulers().step(result.metrics[monitor])

    def lr_scheduler_step(self, scheduler, metric):
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            super().lr_scheduler_step(scheduler, metric)

    def on_test_epoch_start(self):
        self.records = []

    def test_step(self, batch, batch_idx):
        self.records.append(_evaluation_record(batch, self(batch["metadata"])))

    def on_test_epoch_end(self):
        self.evaluation_result = _evaluate_records(
            self.records, self.cfg, self.evaluation_stage, prediction_export_enabled(self.args)
        )
        for name, value in self.evaluation_result.metrics.items():
            self.log(name, value, sync_dist=False)

    def configure_optimizers(self):
        groups = {"decay": [], "no_decay": []}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                decay = parameter.ndim >= 2 and "norm" not in name.lower() and "bias" not in name.lower()
                groups["decay" if decay else "no_decay"].append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": values, "weight_decay": self.args.weight_decay if name == "decay" else 0.0}
                for name, values in groups.items()
                if values
            ],
            lr=self.args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        if getattr(self.args, "lr_scheduler", "decay") == "plateau":
            floor = float(getattr(self.args, "lr_decay_floor", 0.1))
            plateau_factor = getattr(self.args, "lr_plateau_factor", None)
            plateau_patience = getattr(self.args, "lr_plateau_patience", None)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.cfg.finetune.task.monitor_mod,
                factor=0.1 if plateau_factor is None else plateau_factor,
                patience=10 if plateau_patience is None else plateau_patience,
                min_lr=[group["lr"] * floor for group in optimizer.param_groups],
                eps=0.0,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.cfg.finetune.task.monitor,
                    "interval": "epoch",
                    "frequency": getattr(self.args, "check_val_every_n_epoch", 1),
                    "strict": True,
                },
            }
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            warmup_steps=getattr(self.args, "warmup_steps", None),
            decay_shape=getattr(self.args, "lr_decay_shape", "cosine"),
            decay_floor=getattr(self.args, "lr_decay_floor", 0.1),
            decay_ratio=getattr(self.args, "lr_decay_ratio", None),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["config"] = asdict(self.cfg)
        checkpoint["label_contract"] = _label_contract(self.cfg)
        checkpoint["model_contract"] = _model_contract(self.cfg)
        checkpoint["metrics"] = {
            name: float(value)
            for name, value in self.trainer.callback_metrics.items()
            if not name.startswith("val_") or self._validation_epoch == self.current_epoch
        }


def _trainer(args, *, callbacks=(), training=False, logger=False):
    torch.set_float32_matmul_precision("high")
    accelerator = "cpu" if getattr(args, "device", "cuda") == "cpu" else getattr(args, "accelerator", "gpu")
    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = getattr(args, "devices", [0])
    if accelerator == "cpu":
        devices = len(devices)
    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=getattr(args, "precision", "32-true"),
        strategy="ddp" if (devices if isinstance(devices, int) else len(devices)) > 1 else "auto",
        max_epochs=args.epochs if training else 1,
        accumulate_grad_batches=getattr(args, "accumulate_grad_batches", 1),
        gradient_clip_val=getattr(args, "gradient_clip_val", 0.0),
        check_val_every_n_epoch=getattr(args, "check_val_every_n_epoch", 1),
        callbacks=list(callbacks),
        enable_checkpointing=training,
        logger=logger,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
    )


def _test(trainer, module, loader, cfg, checkpoint_path, stage="test"):
    load_checkpoint(module.model, checkpoint_path, device=torch.device("cpu"), cfg=cfg)
    module.evaluation_stage = stage
    trainer.test(module, dataloaders=loader, verbose=False)
    return module.evaluation_result


def configure_result_args(args: Namespace, cfg: BaselineConfig) -> None:
    apply_task_flags(args, cfg.finetune.task)
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
    pieces = [args.version_prefix, "sex-age-baseline", task_name, args.label_name]
    if args.version_tag:
        pieces.append(args.version_tag)
    return "-".join(pieces)


def train_and_save(args: Namespace, cfg: BaselineConfig) -> None:
    if not hasattr(args, "test_all_checkpoints_after_fit"):
        args.test_all_checkpoints_after_fit = False
    if args.test_all_checkpoints_after_fit and not args.test_after_fit:
        raise ValueError("--test-all-checkpoints-after-fit requires --test-after-fit.")
    if args.test_all_checkpoints_after_fit and args.epochs <= 0:
        raise ValueError("--test-all-checkpoints-after-fit requires --epochs greater than 0.")
    ckpt_every_n_epochs = int(getattr(args, "ckpt_every_n_epochs", 1))
    if ckpt_every_n_epochs <= 0:
        raise ValueError("--ckpt-every-n-epochs must be positive for sex_age_baseline.")
    if args.test_all_checkpoints_after_fit and ckpt_every_n_epochs != 1:
        raise ValueError("--test-all-checkpoints-after-fit requires --ckpt-every-n-epochs 1.")
    configure_result_args(args, cfg)
    validate_finetune_scheduler_args(args)
    args.version = build_version_name(args, cfg)
    epochs = int(args.epochs)
    if epochs < 0:
        raise ValueError("--epochs must be non-negative for sex_age_baseline.")
    if epochs == 0 and not args.ckpt_path:
        raise ValueError("--epochs 0 requires --ckpt-path for sex_age_baseline evaluation.")
    _seed_everything(FINETUNE_SEED)

    run_dir = Path("log-finetune") / args.version
    checkpoint_dir = run_dir / "checkpoints"
    # DDP subprocesses re-enter the CLI; only the original rank creates the single-use root.
    if is_rank_zero_process():
        if run_dir.is_symlink():
            raise FileExistsError(f"sex_age_baseline run directory must not be a symlink: {run_dir}.")
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"sex_age_baseline run directory already exists and is not empty: {run_dir}. Use a new --version-name."
            )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        persist_run_config_and_args(args, run_dir)

    module = BaselineModule(cfg, args)
    model = module.model
    if args.ckpt_path:
        load_checkpoint(model, args.ckpt_path, device=torch.device("cpu"), cfg=cfg)
    loaded_splits = ["train", "val"] if epochs > 0 else []
    if args.test_after_fit:
        loaded_splits.append("test")
    if epochs > 0:
        sources = cfg.data.train_dataset_names
        module.train_set = _required_dataset(cfg, "train", sources=sources, loaded_splits=loaded_splits)
        val_set = _required_dataset(cfg, "val", sources=sources, loaded_splits=loaded_splits)
        val_loader = make_dataloader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    best = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor=cfg.finetune.task.monitor,
        mode=cfg.finetune.task.monitor_mod,
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=False,
        enable_version_counter=False,
    )
    periodic = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch={epoch:02d}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        save_last=False,
        every_n_epochs=ckpt_every_n_epochs,
        save_on_train_epoch_end=True,
    )
    callbacks = [best, periodic, LastCheckpoint(checkpoint_dir / "last.ckpt")]
    patience = int(getattr(args, "patience", 100))
    if patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.finetune.task.monitor,
                mode=cfg.finetune.task.monitor_mod,
                patience=patience,
                check_on_train_epoch_end=False,
            )
        )
    preexisting_wandb_run = wandb.run
    logger = WandbLogger(
        project=getattr(args, "wandb_project", None) or "sex-age-baseline",
        name=args.version,
        group=getattr(args, "wandb_group", None),
        mode=getattr(args, "wandb_mode", None),
        save_dir="./wandb_logs",
        log_model=False,
    )
    try:
        trainer = _trainer(args, callbacks=callbacks if epochs else (), training=epochs > 0, logger=logger)
        if epochs:
            trainer.fit(module, val_dataloaders=val_loader)
        best_path = (
            Path(best.best_model_path)
            if epochs and best.best_model_path
            else Path(args.ckpt_path or checkpoint_dir / "best.ckpt")
        )
        last_path = checkpoint_dir / "last.ckpt"
        best_score = float(best.best_model_score) if best.best_model_score is not None else None
        best_metrics = {}
        if not best_path.is_file() or (epochs and (best_score is None or not np.isfinite(best_score))):
            raise ValueError(
                f"No finite best checkpoint was selected for monitor {cfg.finetune.task.monitor!r}. "
                "Check validation labels and monitor configuration."
            )
        best_metrics = dict(torch.load(best_path, map_location="cpu", weights_only=False)["metrics"])

        manifest_path = run_dir / "run_manifest.json"
        if not args.test_after_fit:
            if not trainer.is_global_zero:
                return
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

        test_set = _required_dataset(cfg, "test", sources=cfg.data.test_dataset_names, loaded_splits=loaded_splits)
        test_loader = make_dataloader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        args.eval_split = "test"
        checkpoint_test_results = []
        original_ckpt_path = args.ckpt_path
        checkpoint_result_rows = []
        evaluated = []
        if getattr(args, "export_predictions", False):
            args.inference_prediction_csv_path = run_dir / "predictions.csv"
        prediction_csv_path = getattr(args, "inference_prediction_csv_path", None)
        if args.test_all_checkpoints_after_fit:
            best_checkpoint = load_checkpoint(model, best_path, device=torch.device("cpu"), cfg=cfg)
            best_epoch = int(best_checkpoint["epoch"])
            resolved_checkpoint_dir = checkpoint_dir.resolve()
            frozen_checkpoint_dir = os.environ.get("_SLEEP2VEC_FROZEN_CHECKPOINT_DIR")
            recorded_checkpoint_dir = (
                Path(frozen_checkpoint_dir)
                if frozen_checkpoint_dir
                else checkpoint_dir if checkpoint_dir.is_absolute() else Path.cwd() / checkpoint_dir
            )
            periodic_checkpoints = []
            seen_epochs = set()
            for path in checkpoint_dir.glob("epoch=*.ckpt"):
                if path.is_symlink():
                    continue
                if not path.is_file() or path.resolve().parent != resolved_checkpoint_dir:
                    raise ValueError(f"Invalid periodic checkpoint: {path}")
                match = re.fullmatch(r"epoch=(\d+)(?:-step=\d+)?\.ckpt", path.name)
                if match is None:
                    raise ValueError(f"Malformed periodic checkpoint name: {path.name}")
                epoch = int(match.group(1))
                if epoch in seen_epochs:
                    raise ValueError(f"Duplicate periodic checkpoint epoch: {epoch}")
                seen_epochs.add(epoch)
                recorded_path = recorded_checkpoint_dir / path.name
                if recorded_path.resolve() != path.resolve():
                    raise ValueError(f"Frozen checkpoint path does not identify the saved checkpoint: {recorded_path}")
                # Physical ownership is checked above; manifests preserve the plan's frozen path spelling.
                periodic_checkpoints.append((epoch, recorded_path))
            if not periodic_checkpoints:
                raise ValueError("No regular epoch=*.ckpt checkpoints were saved for test evaluation.")
            if best_epoch not in seen_epochs:
                raise ValueError(
                    f"Validation-best epoch checkpoint is missing from periodic test evidence: {best_epoch}"
                )
            periodic_checkpoints.sort(key=lambda item: (item[0] == best_epoch, item[0], str(item[1])))

            test_result = None
            for epoch, checkpoint_path in periodic_checkpoints:
                args.ckpt_path = str(checkpoint_path)
                args.ckpt_resolved_path = str(checkpoint_path)
                result = _test(trainer, module, test_loader, cfg, checkpoint_path)
                evaluated.append((copy.copy(args), result))
                checkpoint_result_rows.append((result.metrics, str(checkpoint_path)))
                checkpoint_test_results.append(
                    {"checkpoint_path": str(checkpoint_path), "epoch": epoch, "metrics": result.metrics}
                )
                if epoch == best_epoch:
                    test_result = result

        else:
            if best_path.exists():
                args.ckpt_path = str(best_path)
                args.ckpt_resolved_path = str(best_path)
            test_result = _test(trainer, module, test_loader, cfg, best_path)
            evaluated.append((copy.copy(args), test_result))

        if not trainer.is_global_zero:
            return
        # Like sleep2vec, every evaluated checkpoint contributes prediction and per-disease rows.
        if prediction_csv_path:
            for row_args, result in evaluated:
                save_prediction_csv(result.prediction_rows, str(prediction_csv_path), row_args)
        survival_csv_path = None
        multilabel_csv_path = None
        for row_args, result in evaluated:
            if result.survival_per_disease_rows:
                survival_csv_path = run_dir / "survival_per_disease_metrics.csv"
                save_survival_per_disease_metrics_csv(
                    result.survival_per_disease_rows, str(survival_csv_path), row_args
                )
            if result.multilabel_per_disease_rows:
                multilabel_csv_path = run_dir / "multilabel_per_disease_metrics.csv"
                save_multilabel_per_disease_metrics_csv(
                    result.multilabel_per_disease_rows, str(multilabel_csv_path), row_args
                )
        if checkpoint_result_rows:
            # Publish the checkpoint matrix only after every required run artifact succeeds.
            save_result_rows_csv(checkpoint_result_rows, str(args.results_csv_path), args)
        else:
            save_result_csv(test_result.metrics, str(args.results_csv_path), args)
        args.ckpt_path = original_ckpt_path
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
            checkpoint_test_results=checkpoint_test_results,
            prediction_csv_path=prediction_csv_path,
        )
    finally:
        # Finish only the run this call created, and do not let teardown hide a primary error.
        _finish_wandb_run(preexisting_wandb_run, "finetune")


def run_inference_and_save(args: Namespace, cfg: BaselineConfig) -> None:
    cfg = _config_with_inference_preset(args, cfg)
    configure_result_args(args, cfg)
    _seed_everything(getattr(args, "seed", 4523))
    module = BaselineModule(cfg, args)
    selected_ckpt_paths = _load_inference_checkpoint(module.model, args, cfg)

    # sleep2vec inference's source selection: an override, else the YAML list for the evaluated split.
    yaml_sources = cfg.data.test_dataset_names if args.eval_split == "test" else cfg.data.train_dataset_names
    sources = args.override_dataset_names or yaml_sources
    dataset = _required_dataset(cfg, args.eval_split, sources=sources, loaded_splits=[args.eval_split])
    loader = make_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    trainer = _trainer(args)
    prepare_inference_result_paths(
        args,
        namespace="sex_age_baseline",
        root=getattr(args, "results_root", DEFAULT_INFERENCE_RESULTS_ROOT),
        checkpoint_paths=selected_ckpt_paths,
    )
    preexisting_wandb_run = wandb.run
    wandb_run = _init_wandb(args)
    try:
        module.evaluation_stage = args.eval_split
        trainer.test(module, dataloaders=loader, verbose=False)
        result = module.evaluation_result
        if not trainer.is_global_zero:
            return
        save_result_csv(result.metrics, str(args.inference_metrics_csv_path), args)
        save_result_csv(result.metrics, str(args.inference_overview_csv_path), args)
        save_prediction_csv(result.prediction_rows, str(args.inference_prediction_csv_path), args)
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
        if wandb_run is not None:
            _log_inference_outputs_to_wandb(
                args,
                result.metrics,
                len(result.prediction_rows),
                len(result.survival_per_disease_rows),
                len(result.multilabel_per_disease_rows),
            )
    finally:
        if wandb_run is not None:
            _finish_wandb_run(preexisting_wandb_run, "inference")


def _load_inference_checkpoint(model: SexAgeMLP, args: Namespace, cfg: BaselineConfig) -> list[Path] | None:
    """Load --ckpt-path, or average the checkpoints ending there, with sleep2vec inference selection."""
    if args.avg_ckpts <= 1:
        if args.ckpt_path in {"best", "last"}:
            raise ValueError("sex_age_baseline inference resolves best/last only with --avg-ckpts > 1.")
        load_checkpoint(model, args.ckpt_path, device=torch.device("cpu"), cfg=cfg)
        return None
    ckpt_dir = args.avg_ckpt_dir
    end_ckpt = None if args.ckpt_path in {"best", "last"} else Path(args.ckpt_path)
    if ckpt_dir is None:
        if end_ckpt is None:
            raise ValueError("Use --avg-ckpt-dir when averaging with ckpt-path=best/last.")
        ckpt_dir = end_ckpt.parent
    ckpt_paths = select_checkpoints(Path(ckpt_dir), end_ckpt=end_ckpt, num_ckpts=args.avg_ckpts)
    logging.info("Averaging checkpoints: %s", ", ".join(str(path) for path in ckpt_paths))
    for path in ckpt_paths:
        _validate_checkpoint_contracts(torch.load(path, map_location="cpu", weights_only=False), cfg, path)
    _load_model_state(model, average_checkpoints(ckpt_paths, device=torch.device("cpu")), ckpt_dir)
    return ckpt_paths


def _finish_wandb_run(preexisting_run, stage: str) -> None:
    active_run = wandb.run
    if active_run is None or active_run is preexisting_run:
        return
    primary_exc_active = sys.exc_info()[0] is not None
    try:
        wandb.finish()
    except BaseException as exc:
        if not primary_exc_active:
            raise
        logging.warning("wandb.finish() failed during %s cleanup: %s", stage, exc)


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
    records = []
    with torch.no_grad():
        for batch in loader:
            records.append(_evaluation_record(batch, model(batch["metadata"])))
    return _evaluate_records(records, cfg, stage, export_predictions)


def _evaluation_record(batch, logits):
    return {
        "key": list(batch["key"]),
        "sample_index": [int(value) for value in batch["sample_index"]],
        "logits": logits.detach().float().cpu(),
        **{
            name: batch[name].detach().cpu()
            for name in ("has_label", "event_time", "is_event", "disease_label")
            if name in batch
        },
    }


def _evaluate_records(records, cfg, stage, export_predictions):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, records)
        records = [record for rank_records in gathered for record in rank_records]
    labels = (
        ["has_label", "event_time", "is_event"]
        if cfg.finetune.task.type == "survival"
        else ["has_label", "disease_label"]
    )
    # The dataset holds one record per key; distributed evaluation samplers pad by repeating dataset rows.
    rows = {}
    for record in records:
        for i, sample_index in enumerate(record["sample_index"]):
            rows.setdefault(sample_index, (record, i))
    if not rows:
        raise ValueError(f"Sex/age baseline split {stage!r} has no rows.")
    ordered = [rows[sample_index] for sample_index in sorted(rows)]
    keys = [record["key"][i] for record, i in ordered]
    logits = torch.stack([record["logits"][i] for record, i in ordered])
    tensors = {name: torch.stack([record[name][i] for record, i in ordered]) for name in labels}
    if cfg.finetune.task.type == "survival":
        return _evaluate_survival(
            cfg,
            stage,
            keys,
            logits,
            tensors["event_time"],
            tensors["is_event"],
            tensors["has_label"],
            export_predictions,
        )
    return _evaluate_multilabel(
        cfg, stage, keys, logits, tensors["disease_label"], tensors["has_label"], export_predictions
    )


def _batch_loss(logits, batch, cfg):
    if cfg.finetune.task.type == "survival":
        return CoxPHLossVectorized(eps=cfg.finetune.loss.eps)(
            logits, batch["has_label"], batch["event_time"], batch["is_event"]
        )
    return masked_multilabel_bce(
        logits, batch["disease_label"], batch["has_label"], pos_weight=cfg.finetune.loss.pos_weight
    )


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
            "state_dict": {f"model.{name}": value for name, value in model.state_dict().items()},
            "config": asdict(cfg),
            "label_contract": _label_contract(cfg),
            "model_contract": _model_contract(cfg),
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
        _validate_checkpoint_contracts(checkpoint, cfg, path)
    _load_model_state(model, checkpoint["state_dict"], path)
    return checkpoint


def _load_model_state(model: SexAgeMLP, state_dict: Mapping[str, Any], path: str | Path) -> None:
    if not all(name.startswith("model.") for name in state_dict):
        raise ValueError(f"Checkpoint is not a covariate Lightning model: {path}")
    model.load_state_dict({name.removeprefix("model."): value for name, value in state_dict.items()}, strict=True)


def _validate_checkpoint_contracts(checkpoint: Any, cfg: BaselineConfig, path: str | Path) -> None:
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint does not contain a saved sex_age_baseline label contract: {path}")
    saved_contract = checkpoint.get("label_contract")
    if not isinstance(saved_contract, Mapping):
        raise ValueError(f"Checkpoint does not contain a saved sex_age_baseline label contract: {path}")
    current_contract = _label_contract(cfg)
    if dict(saved_contract) != current_contract:
        raise ValueError(
            "Checkpoint label contract does not match current sex_age_baseline config: "
            f"checkpoint={dict(saved_contract)}, current={current_contract}."
        )
    saved_model_contract = checkpoint.get("model_contract")
    if not isinstance(saved_model_contract, Mapping):
        raise ValueError(f"Checkpoint does not contain a saved sex_age_baseline model contract: {path}")
    current_model_contract = _model_contract(cfg)
    if dict(saved_model_contract) != current_model_contract:
        raise ValueError(
            "Checkpoint model contract does not match current sex_age_baseline config: "
            f"checkpoint={dict(saved_model_contract)}, current={current_model_contract}."
        )


def _model_contract(cfg: BaselineConfig) -> dict[str, Any]:
    task_cfg = covariate_task_config(cfg)
    return {
        "covariates": list(ordered_covariates(task_cfg.covariates)),
        "covariate_embedding_dim": task_cfg.covariate_embedding_dim,
        "covariate_normalization": dict(task_cfg.covariate_normalization),
        "head": asdict(cfg.model.head),
    }


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
    loss = CoxPHLossVectorized(eps=cfg.finetune.loss.eps)(logits, has_label, event_time, is_event)
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


def _required_dataset(
    cfg: BaselineConfig, split: str, *, sources: list[str] | None, loaded_splits: list[str] | None = None
) -> SexAgeDataset:
    dataset = load_split_dataset(cfg, split, sources=sources, loaded_splits=loaded_splits)
    if len(dataset) == 0:
        raise ValueError(f"Sex/age baseline split {split!r} has no rows.")
    return dataset


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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
