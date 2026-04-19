from dataclasses import asdict
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import wandb
import yaml

from sleep2vec import diagnostics
from sleep2vec.averagings.base import BaseModelAverager, build_model_averager
from sleep2vec.common import remap_stage_labels
from sleep2vec.metrics import (
    AHI_COARSE_THRESHOLD_GRID,
    _aggregate_prepared_ahi_records,
    _compute_ahi_event_metrics_from_prepared,
    _prepare_ahi_records,
    compute_ahi_pointwise_metrics,
    compute_downstream_metrics,
    extract_ahi_summary_scatter_arrays,
)
from sleep2vec.visualization.downstream_eval import DownstreamEvalVisualizer
from sleep2vec.visualization.layer_mix import build_layer_mix_rows, render_layer_mix_heatmap

from .downstream_model import Sleep2vecDownstreamModel
from .pretrain_model import Sleep2vecPretrainModel


class Sleep2vecFinetuning(pl.LightningModule):
    def __init__(self, args, model_config, finetune_config=None, averaging_config=None):
        super().__init__()
        self.args = args
        self.model_config = model_config
        self.finetune_config = finetune_config
        self.averaging_config = averaging_config

        self.backbone = Sleep2vecPretrainModel(
            channel_feature_dim=None,
            transformer_hidden_size=model_config.backbone.hidden_size,
            transformer_num_hidden_layers=model_config.backbone.num_hidden_layers,
            transformer_num_attention_heads=model_config.backbone.num_attention_heads,
            channel_names=[c.name for c in model_config.channels],
            projection=model_config.projection.enabled,
            encoder_factory=None,
            model_config=model_config,
            projection_config=model_config.projection,
            device=args.device,
        ).to(args.device)

        head_kwargs = getattr(args, "head_kwargs", None)
        self.model = Sleep2vecDownstreamModel(
            args.label_name,
            self.backbone,
            channel_names=[c.name for c in model_config.channels],
            output_dim=args.output_dim,
            is_classification=args.is_classification,
            is_seq=args.is_seq,
            head_name=getattr(args, "head_name", None),
            head_kwargs=head_kwargs,
            model_config=model_config,
            layer_mix_cfg=getattr(finetune_config, "layer_mix", None) if finetune_config else None,
            head_config=model_config.head,
        ).to(args.device)

        if args.pretrained_backbone_path:
            logging.info(f"loading pretrain model from {args.pretrained_backbone_path}")
            self.model.load_pretrained_backbone(args.pretrained_backbone_path)
            logging.info(f"loaded pretrain model from {args.pretrained_backbone_path}")

        if args.freeze_backbone_and_insert_lora:
            self.model.freeze_backbone_and_insert_lora(
                insert_lora=args.insert_lora,
                separate_adapters=args.separate_adapters,
            )

        if getattr(args, "freeze_tokenizer", True):
            self.backbone.set_tokenizers_trainable(False)
        else:
            self.backbone.set_tokenizers_trainable(True)

        self._stage_outputs = {"train": [], "val": [], "test": []}
        self._classification_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self._multilabel_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self._regression_loss = torch.nn.MSELoss()
        self._ahi_eval_threshold: float | None = None
        self._ahi_train_pointwise_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        self._eval_loss_sums = {"val": 0.0, "test": 0.0}
        self._eval_loss_counts = {"val": 0, "test": 0}

        # Optional tensor diagnostics (borrowed from icefall)
        self._diagnostic = None
        self._diag_steps = getattr(args, "diagnostics_steps", 5)
        if getattr(args, "print_diagnostics", False):
            opts = diagnostics.TensorDiagnosticOptions(max_eig_dim=512)
            self._diagnostic = diagnostics.attach_diagnostics(self.model, opts)

        self.model_averager: BaseModelAverager | None = build_model_averager(averaging_config, self.model)
        if self.model_averager is not None:
            self.model_averager.attach_to_module(self)
        self._eval_visualizer = DownstreamEvalVisualizer(
            getattr(finetune_config, "eval_visualizations", None) if finetune_config is not None else None
        )

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["model_config"] = asdict(self.model_config)
        checkpoint["model_config_yaml"] = yaml.safe_dump(checkpoint["model_config"], sort_keys=True)
        if self.finetune_config is not None:
            checkpoint["finetune_config"] = asdict(self.finetune_config)
            checkpoint["finetune_config_yaml"] = yaml.safe_dump(checkpoint["finetune_config"], sort_keys=True)

        student_layer_mix = self._layer_mix_snapshot(self.model)
        if student_layer_mix is not None:
            checkpoint["layer_mix_weights_student"] = student_layer_mix

        eval_model = self._get_eval_model()
        if eval_model is not self.model:
            eval_layer_mix = self._layer_mix_snapshot(eval_model)
            if eval_layer_mix is not None:
                checkpoint["layer_mix_weights_eval"] = eval_layer_mix
        if self._is_ahi_task() and self._ahi_eval_threshold is not None:
            checkpoint["ahi_eval_threshold"] = float(self._ahi_eval_threshold)

    # ---------- Lightning hooks ----------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        eval_model = self._get_eval_model()
        self._shared_step(batch, stage="val", model=eval_model)

    def test_step(self, batch, batch_idx):
        eval_model = self._get_eval_model()
        self._shared_step(batch, stage="test", model=eval_model)

    def on_train_epoch_end(self):
        self._log_layer_mix_weights(stage="train", model=self.model)
        self._finalize_epoch(stage="train")

    def on_validation_epoch_end(self):
        self._log_layer_mix_weights(stage="val", model=self._get_eval_model())
        self._finalize_epoch(stage="val")

    def on_test_epoch_end(self):
        self._finalize_epoch(stage="test")

    def on_fit_start(self):
        super().on_fit_start()
        if self.model_averager is not None:
            self.model_averager.on_fit_start(self.trainer)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if self.model_averager is not None:
            self.model_averager.on_load_checkpoint(checkpoint)
        if self._is_ahi_task():
            threshold = checkpoint.get("ahi_eval_threshold")
            self._ahi_eval_threshold = None if threshold is None else float(threshold)

    def on_test_start(self):
        super().on_test_start()
        if (
            self._is_ahi_task()
            and self._ahi_eval_threshold is None
            and self._ahi_search_thresholds_for_stage("test") is None
        ):
            raise ValueError(
                "AHI test/inference requires a validation-fitted threshold stored in the checkpoint. "
                "This checkpoint does not contain `ahi_eval_threshold`."
            )

    def load_state_dict(self, state_dict, strict: bool = True):
        # Allow missing/extra layer-mix weights when loading older checkpoints.
        result = super().load_state_dict(state_dict, strict=False)
        if not strict:
            return result

        allowed_prefixes = ("model.layer_mix.",)
        missing = [k for k in result.missing_keys if not k.startswith(allowed_prefixes)]
        unexpected = [k for k in result.unexpected_keys if not k.startswith(allowed_prefixes)]

        if missing or unexpected:
            raise RuntimeError(
                "Error(s) in loading state_dict: " f"missing keys={missing}, unexpected keys={unexpected}"
            )

        if result.missing_keys:
            logging.warning("Missing layer-mix keys while loading checkpoint: %s", result.missing_keys)
        if result.unexpected_keys:
            logging.warning("Unexpected layer-mix keys while loading checkpoint: %s", result.unexpected_keys)

        return result

    # ---------- Internal helpers ----------
    def _shared_step(self, batch, stage: str, model=None):
        model = model or self.model
        logits = model(batch)
        loss_info = self._compute_loss(logits, batch)
        if loss_info is None:
            if stage == "train":
                raise ValueError("No valid labels found in the current training batch.")
            valid_count = 0
            loss = None
        else:
            loss, valid_count = loss_info
            eval_loss_sums = getattr(self, "_eval_loss_sums", {})
            eval_loss_counts = getattr(self, "_eval_loss_counts", {})
            if stage == "train":
                self.log(
                    f"{stage}_loss",
                    loss,
                    prog_bar=True,
                    sync_dist=True,
                    on_step=True,
                    on_epoch=True,
                    batch_size=max(valid_count, 1),
                )
            elif stage in eval_loss_sums and stage in eval_loss_counts:
                eval_loss_sums[stage] += float(loss.detach().item()) * valid_count
                eval_loss_counts[stage] += int(valid_count)

        if self._is_ahi_task() and stage == "train":
            self._accumulate_ahi_train_pointwise_counts(batch, logits)
        elif self._is_ahi_task() and stage in {"val", "test"} and (stage != "val" or self._uses_full_ahi_validation()):
            records = self._extract_ahi_event_records(batch, logits)
            if records:
                self._stage_outputs[stage].extend(records)
        else:
            preds = self._extract_valid_predictions(batch, logits)
            if preds is not None:
                self._stage_outputs[stage].append(preds)

        return loss if stage == "train" else None

    def _compute_loss(self, logits, batch):
        targets = self._get_targets(batch)

        if getattr(self.args, "is_multilabel", False):
            valid_mask = targets != -1.0
            if not valid_mask.any():
                return None
            loss = self._multilabel_loss(logits, targets.float())[valid_mask].mean()
            return loss, int(valid_mask.sum().item())

        if self.args.is_classification:
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1).long()
            valid_mask = targets_flat != -1
            if not valid_mask.any():
                return None
            loss = self._classification_loss(logits_flat, targets_flat)
            return loss, int(valid_mask.sum().item())

        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1).float()
        valid_mask = targets_flat != -1.0
        if not valid_mask.any():
            return None
        preds = logits_flat[valid_mask]
        valid_targets = targets_flat[valid_mask]
        loss = self._regression_loss(preds, valid_targets)
        return loss, int(valid_targets.numel())

    def _extract_valid_predictions(self, batch, logits):
        labels = self._get_targets(batch)

        if getattr(self.args, "is_multilabel", False):
            mask = labels != -1.0
            if not mask.any():
                return None

            probs = torch.sigmoid(logits[mask]).to(torch.float32).detach().cpu().numpy()
            labels_np = labels[mask].to(torch.int64).detach().cpu().numpy()
            return probs, labels_np

        if self.args.is_classification:
            if logits.dim() == 3:
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
            else:
                labels = labels.view(-1)

            mask = labels != -1
            if not mask.any():
                return None

            probs = torch.softmax(logits[mask], dim=-1).detach().cpu().numpy()
            labels_np = labels[mask].detach().cpu().numpy()
            return probs, labels_np

        logits = logits.view(-1)
        labels = labels.view(-1).float()
        mask = labels != -1.0
        if not mask.any():
            return None

        preds = logits[mask].to(torch.float32).detach().cpu().numpy()
        labels_np = labels[mask].to(torch.float32).detach().cpu().numpy()
        return preds, labels_np

    def _extract_ahi_event_records(self, batch, logits) -> list[dict[str, np.ndarray]]:
        labels = batch["tokens"]["ahi"].detach().cpu()
        probs = torch.sigmoid(logits).to(torch.float32).detach().cpu()
        true_ahi = batch["metadata"]["ahi"].to(torch.float32).detach().cpu()
        tst_hours = batch["metadata"]["tst"].to(torch.float32).detach().cpu()
        token_start = batch["token_start"].to(torch.long).detach().cpu()
        paths = list(batch["metadata"]["path"])
        stage5 = batch["tokens"]["stage5"].detach().cpu()

        records: list[dict[str, np.ndarray]] = []
        for idx in range(labels.size(0)):
            second_valid_mask = labels[idx].reshape(-1) != -1.0
            if not second_valid_mask.any():
                continue
            stage5_tokens = stage5[idx].to(torch.int64).reshape(-1).numpy()
            truth = labels[idx].reshape(-1)[second_valid_mask].to(torch.int64).numpy()
            score = probs[idx].reshape(-1)[second_valid_mask].numpy()
            record = {
                "path": str(paths[idx]),
                "token_start": int(token_start[idx].item()),
                "truth": truth,
                "score": score,
                "true_ahi": float(true_ahi[idx].item()),
                "tst_hours": float(tst_hours[idx].item()),
                "stage5": stage5_tokens,
                "second_valid_mask": second_valid_mask.numpy(),
            }
            records.append(record)
        return records

    def _get_targets(self, batch):
        if not self.args.is_seq:
            return batch["metadata"][self.args.label_name].to(self.args.device)

        label_source_name = getattr(self.args, "label_source_name", self.args.label_name)
        labels = batch["tokens"][label_source_name].to(self.args.device)
        if getattr(self.args, "is_multilabel", False):
            return labels
        return remap_stage_labels(labels, self.args.label_name)

    def _is_ahi_task(self) -> bool:
        return getattr(self.args, "label_name", None) == "ahi"

    def _accumulate_ahi_train_pointwise_counts(self, batch, logits) -> None:
        labels = self._get_targets(batch)
        valid_mask = labels != -1.0
        if not valid_mask.any():
            return

        probs = torch.sigmoid(logits[valid_mask])
        targets = labels[valid_mask].to(torch.int64)
        preds = (probs >= 0.5).to(torch.int64)

        self._ahi_train_pointwise_counts["tp"] += int(((preds == 1) & (targets == 1)).sum().item())
        self._ahi_train_pointwise_counts["fp"] += int(((preds == 1) & (targets == 0)).sum().item())
        self._ahi_train_pointwise_counts["tn"] += int(((preds == 0) & (targets == 0)).sum().item())
        self._ahi_train_pointwise_counts["fn"] += int(((preds == 0) & (targets == 1)).sum().item())

    def _compute_reduced_ahi_train_pointwise_metrics(self) -> dict[str, float]:
        counts = self._ahi_train_pointwise_counts
        stats = torch.tensor(
            [counts["tp"], counts["fp"], counts["tn"], counts["fn"]],
            dtype=torch.float64,
            device=torch.device(getattr(self.args, "device", "cpu")),
        )
        trainer = getattr(self, "trainer", None)
        if dist.is_available() and dist.is_initialized():
            if trainer is not None and hasattr(trainer, "strategy"):
                stats = trainer.strategy.reduce(stats, reduce_op="sum")
            else:  # pragma: no cover - trainer-less distributed fallback
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        tp, fp, tn, fn = [int(value) for value in stats.tolist()]
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        self._ahi_train_pointwise_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        return {
            "ahi_pointwise_accuracy": float(accuracy),
            "ahi_pointwise_precision": float(precision),
            "ahi_pointwise_recall": float(recall),
            "ahi_pointwise_f1": float(f1),
        }

    def _uses_full_ahi_validation(self) -> bool:
        return self._is_ahi_task() and self.args.monitor == "val_ahi_pearson" and self.args.monitor_mod == "max"

    def _ahi_search_thresholds_for_stage(self, stage: str) -> tuple[float, ...] | None:
        if stage == "val":
            thresholds = getattr(self.args, "ahi_val_search_thresholds", AHI_COARSE_THRESHOLD_GRID)
        elif stage == "test":
            thresholds = getattr(self.args, "ahi_test_search_thresholds", None)
        else:
            return None

        if thresholds is None:
            return None
        return tuple(float(value) for value in thresholds)

    @staticmethod
    def _can_broadcast_ahi_metrics() -> bool:
        return dist.is_available() and dist.is_initialized() and hasattr(dist, "broadcast_object_list")

    def _compute_ahi_metrics_for_stage(
        self,
        stage: str,
        records: list[dict[str, np.ndarray]],
    ) -> tuple[dict[str, float], float, tuple[np.ndarray, np.ndarray] | None]:
        prepared_records = _prepare_ahi_records(records)

        if stage == "val":
            metrics, eval_threshold = _compute_ahi_event_metrics_from_prepared(
                prepared_records,
                threshold=None,
                search_thresholds=self._ahi_search_thresholds_for_stage("val"),
            )
            self._ahi_eval_threshold = float(eval_threshold)
            aggregate = _aggregate_prepared_ahi_records(prepared_records, threshold=float(eval_threshold))
            return metrics, float(eval_threshold), (aggregate["true_ahi"], aggregate["pred_ahi"])

        test_search_thresholds = self._ahi_search_thresholds_for_stage("test")
        if test_search_thresholds is not None:
            try:
                metrics, eval_threshold = _compute_ahi_event_metrics_from_prepared(
                    prepared_records,
                    threshold=None,
                    search_thresholds=test_search_thresholds,
                )
                aggregate = _aggregate_prepared_ahi_records(prepared_records, threshold=float(eval_threshold))
                return metrics, float(eval_threshold), (aggregate["true_ahi"], aggregate["pred_ahi"])
            except ValueError as exc:
                if (
                    self._ahi_eval_threshold is not None
                    and "Need at least 1 non-skipped sample with TST >= 2h." in str(exc)
                ):
                    eval_threshold = float(self._ahi_eval_threshold)
                    logging.info(
                        "AHI threshold search fallback: reusing saved threshold=%.2f because no eligible summary samples were found",
                        eval_threshold,
                    )
                    metrics, _ = _compute_ahi_event_metrics_from_prepared(prepared_records, threshold=eval_threshold)
                    aggregate = _aggregate_prepared_ahi_records(prepared_records, threshold=eval_threshold)
                    return metrics, eval_threshold, (aggregate["true_ahi"], aggregate["pred_ahi"])
                raise

        if self._ahi_eval_threshold is None:
            raise ValueError(
                "AHI evaluation requires a validation-fitted threshold. "
                "No `ahi_eval_threshold` is available for test/inference."
            )

        eval_threshold = float(self._ahi_eval_threshold)
        metrics, _ = _compute_ahi_event_metrics_from_prepared(prepared_records, threshold=eval_threshold)
        aggregate = _aggregate_prepared_ahi_records(prepared_records, threshold=eval_threshold)
        return metrics, eval_threshold, (aggregate["true_ahi"], aggregate["pred_ahi"])

    def _compute_or_broadcast_ahi_metrics(
        self,
        stage: str,
        records: list[dict[str, np.ndarray]],
    ) -> tuple[dict[str, float], float, tuple[np.ndarray, np.ndarray] | None]:
        trainer = getattr(self, "trainer", None)
        if trainer is None or not self._can_broadcast_ahi_metrics():
            return self._compute_ahi_metrics_for_stage(stage, records)

        payload: list[dict[str, object] | None] = [None]
        scatter_arrays: tuple[np.ndarray, np.ndarray] | None = None
        if trainer.is_global_zero:
            try:
                metrics, eval_threshold, scatter_arrays = self._compute_ahi_metrics_for_stage(stage, records)
                payload[0] = {
                    "metrics": metrics,
                    "eval_threshold": float(eval_threshold),
                    "error_type": None,
                    "error_message": None,
                }
            except Exception as exc:  # pragma: no cover - distributed error fan-out
                payload[0] = {
                    "metrics": None,
                    "eval_threshold": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }

        dist.broadcast_object_list(payload, src=0)
        result = payload[0] or {}
        error_message = result.get("error_message")
        if error_message is not None:
            if result.get("error_type") == "ValueError":
                raise ValueError(str(error_message))
            raise RuntimeError(str(error_message))

        metrics = result["metrics"]
        eval_threshold = float(result["eval_threshold"])
        if stage == "val":
            self._ahi_eval_threshold = eval_threshold
        return metrics, eval_threshold, scatter_arrays

    @staticmethod
    def _layer_mix_snapshot(model: torch.nn.Module):
        getter = getattr(model, "layer_mix_snapshot", None)
        if not callable(getter):
            return None
        return getter()

    def _empty_epoch_outputs(self):
        if getattr(self.args, "is_multilabel", False):
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        if self.args.is_classification:
            output_dim = int(getattr(self.args, "output_dim", 0) or 0)
            return np.empty((0, output_dim), dtype=np.float32), np.empty((0,), dtype=np.int64)
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    def _concat_epoch_outputs(self, outputs):
        if not outputs:
            return self._empty_epoch_outputs()

        preds, gts = zip(*outputs)
        return np.concatenate(preds, axis=0), np.concatenate(gts, axis=0)

    def _gather_eval_outputs(self, preds: np.ndarray, gts: np.ndarray):
        if not dist.is_available() or not dist.is_initialized() or not hasattr(dist, "all_gather_object"):
            return preds, gts

        world_size = dist.get_world_size()
        gathered_preds: list[np.ndarray | None] = [None] * world_size
        gathered_gts: list[np.ndarray | None] = [None] * world_size
        dist.all_gather_object(gathered_preds, preds)
        dist.all_gather_object(gathered_gts, gts)

        gathered_preds = [item for item in gathered_preds if isinstance(item, np.ndarray) and item.size > 0]
        gathered_gts = [item for item in gathered_gts if isinstance(item, np.ndarray) and item.size > 0]
        if not gathered_preds or not gathered_gts:
            return self._empty_epoch_outputs()

        return np.concatenate(gathered_preds, axis=0), np.concatenate(gathered_gts, axis=0)

    def _gather_ahi_event_records(self, records: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
        if not dist.is_available() or not dist.is_initialized() or not hasattr(dist, "all_gather_object"):
            return records

        world_size = dist.get_world_size()
        gathered: list[list[dict[str, np.ndarray]] | None] = [None] * world_size
        dist.all_gather_object(gathered, records)

        merged: list[dict[str, np.ndarray]] = []
        for item in gathered:
            if isinstance(item, list):
                merged.extend(item)
        return merged

    def _log_eval_loss(self, stage: str) -> None:
        eval_loss_sums = getattr(self, "_eval_loss_sums", {})
        eval_loss_counts = getattr(self, "_eval_loss_counts", {})
        if stage not in eval_loss_sums or stage not in eval_loss_counts:
            return

        loss_sum = float(eval_loss_sums[stage])
        loss_count = int(eval_loss_counts[stage])
        eval_loss_sums[stage] = 0.0
        eval_loss_counts[stage] = 0

        stats = torch.tensor(
            [loss_sum, float(loss_count)],
            dtype=torch.float64,
            device=torch.device(getattr(self.args, "device", "cpu")),
        )
        trainer = getattr(self, "trainer", None)
        if dist.is_available() and dist.is_initialized():
            if trainer is not None and hasattr(trainer, "strategy"):
                stats = trainer.strategy.reduce(stats, reduce_op="sum")
            else:  # pragma: no cover - trainer-less distributed fallback
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        global_loss_count = int(stats[1].item())
        if global_loss_count == 0:
            return

        self.log(
            f"{stage}_loss",
            float(stats[0].item()) / global_loss_count,
            prog_bar=True,
            logger=True,
            sync_dist=False,
            on_step=False,
            on_epoch=True,
        )

    def _log_layer_mix_weights(self, stage: str, model: torch.nn.Module) -> None:
        snapshot = self._layer_mix_snapshot(model)
        if snapshot is None:
            return

        trainer = getattr(self, "trainer", None)
        if trainer is not None and not trainer.is_global_zero:
            return
        if getattr(wandb, "run", None) is None:
            return

        layer_ids = [int(v) for v in snapshot.get("layer_indices", [])]
        effective = snapshot.get("effective_by_modality", {})
        shared = bool(snapshot.get("shared_across_modalities", False))
        if not layer_ids or not isinstance(effective, dict) or not effective:
            return

        modality_names = list(effective.keys())
        matrix_rows: list[list[float]] = []
        for modality in modality_names:
            mod_info = effective.get(modality, {})
            weights = mod_info.get("layer_weights", []) if isinstance(mod_info, dict) else []
            if len(weights) < len(layer_ids):
                logging.warning(
                    "Skipping layer-mix visualization for stage=%s due to malformed weights for modality=%s.",
                    stage,
                    modality,
                )
                return
            matrix_rows.append([float(weights[idx]) for idx in range(len(layer_ids))])

        matrix = np.array(matrix_rows, dtype=np.float32)
        title = (
            f"{stage.title()} Layer-Mix Weights (epoch {self.current_epoch}, "
            f"{'shared' if shared else 'per-modality'})"
        )
        fig = render_layer_mix_heatmap(matrix, modality_names, layer_ids, title=title)
        rows = build_layer_mix_rows(
            stage=stage,
            epoch=int(self.current_epoch),
            shared=shared,
            layer_ids=layer_ids,
            effective_by_modality=effective,
        )
        columns = [
            "stage",
            "epoch",
            "modality",
            "layer_id",
            "weight",
            "shared_across_modalities",
            "row_name",
            "row_index",
        ]
        table = wandb.Table(columns=columns, data=[[row[col] for col in columns] for row in rows])

        wandb.log(
            {
                f"{stage}_layer_mix/heatmap": wandb.Image(fig),
                f"{stage}_layer_mix/table": table,
            },
            commit=False,
        )
        plt.close(fig)

    def _finalize_epoch(self, stage: str):
        outputs = self._stage_outputs[stage]
        if stage in getattr(self, "_eval_loss_sums", {}):
            self._log_eval_loss(stage)

        if self._is_ahi_task() and stage == "train":
            metrics = self._compute_reduced_ahi_train_pointwise_metrics()
            for k, v in metrics.items():
                self.log(
                    f"{stage}_{k}",
                    v,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                    on_epoch=True,
                )
            return None

        if self._is_ahi_task() and stage == "val" and not self._uses_full_ahi_validation():
            preds, gts = self._concat_epoch_outputs(outputs)
            outputs.clear()
            preds, gts = self._gather_eval_outputs(preds, gts)
            if preds.size == 0 or gts.size == 0:
                return None

            metrics = compute_ahi_pointwise_metrics(gts, preds)
            for k, v in metrics.items():
                self.log(
                    f"{stage}_{k}",
                    v,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                    on_epoch=True,
                )
            return preds, gts

        if self._is_ahi_task() and stage in {"val", "test"}:
            records = list(outputs)
            outputs.clear()
            records = self._gather_ahi_event_records(records)
            if not records:
                return None

            metrics, eval_threshold, scatter_arrays = self._compute_or_broadcast_ahi_metrics(stage, records)

            for k, v in metrics.items():
                self.log(
                    f"{stage}_{k}",
                    v,
                    prog_bar=(stage != "train"),
                    logger=True,
                    sync_dist=False,
                    on_epoch=True,
                )
            trainer = getattr(self, "trainer", None)
            if trainer is not None and trainer.is_global_zero:
                if scatter_arrays is None:
                    true_ahi, pred_ahi = extract_ahi_summary_scatter_arrays(records, threshold=eval_threshold)
                else:
                    true_ahi, pred_ahi = scatter_arrays
                self._eval_visualizer.log_ahi_summary_scatter(
                    stage=stage,
                    preds=pred_ahi,
                    targets=true_ahi,
                    label_name=self.args.label_name,
                    current_epoch=int(self.current_epoch),
                )
            if trainer is not None and dist.is_available() and dist.is_initialized() and hasattr(trainer, "strategy"):
                trainer.strategy.barrier(f"ahi_{stage}_epoch_end")
            return records

        preds, gts = self._concat_epoch_outputs(outputs)
        outputs.clear()

        if stage in {"val", "test"}:
            preds, gts = self._gather_eval_outputs(preds, gts)
        if preds.size == 0 or gts.size == 0:
            return None

        metrics = compute_downstream_metrics(
            gts,
            preds,
            is_classification=self.args.is_classification,
            is_multilabel=getattr(self.args, "is_multilabel", False),
            output_dim=getattr(self.args, "output_dim", None),
            stage_names=getattr(self.args, "stage_names", None),
        )
        for k, v in metrics.items():
            self.log(
                f"{stage}_{k}",
                v,
                prog_bar=(stage != "train"),
                logger=True,
                sync_dist=(stage == "train"),
                on_epoch=True,
            )

        trainer = getattr(self, "trainer", None)
        if (
            stage in {"val", "test"}
            and trainer is not None
            and trainer.is_global_zero
            and not getattr(self.args, "is_multilabel", False)
        ):
            self._eval_visualizer.log(
                stage=stage,
                preds=preds,
                targets=gts,
                is_classification=self.args.is_classification,
                output_dim=getattr(self.args, "output_dim", None),
                label_name=self.args.label_name,
                current_epoch=int(self.current_epoch),
                class_labels=getattr(self.args, "class_labels", None),
            )

        return preds, gts

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        if self.model_averager is not None:
            self.model_averager.on_train_batch_end(trainer=self.trainer, global_step=self.global_step)
        if self._diagnostic is not None and self.global_step >= self._diag_steps:
            if self.trainer is not None:
                self.trainer.should_stop = True

    def on_train_end(self):
        super().on_train_end()
        if self._diagnostic is not None:
            self._diagnostic.print_diagnostics()

    def _get_eval_model(self):
        if self.model_averager is not None:
            return self.model_averager.eval_model()
        return self.model

    def configure_optimizers(self):
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and ("norm" not in n.lower()) and ("bias" not in n.lower()):
                decay.append(p)
            else:
                no_decay.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.args.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = getattr(self.args, "warmup_steps", None)
        if warmup_steps is None:
            warmup = int(0.03 * total_steps)
        else:
            warmup = int(warmup_steps)
        warmup = max(0, min(warmup, total_steps))

        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, total_steps - warmup))
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
