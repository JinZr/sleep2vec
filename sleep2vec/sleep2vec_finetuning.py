from dataclasses import asdict
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import wandb
import yaml

from sleep2vec import diagnostics
from sleep2vec.averagings.base import BaseModelAverager, build_model_averager
from sleep2vec.metrics import compute_downstream_metrics
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
        self._regression_loss = torch.nn.MSELoss()

        # Optional tensor diagnostics (borrowed from icefall)
        self._diagnostic = None
        self._diag_steps = getattr(args, "diagnostics_steps", 5)
        if getattr(args, "print_diagnostics", False):
            opts = diagnostics.TensorDiagnosticOptions(max_eig_dim=512)
            self._diagnostic = diagnostics.attach_diagnostics(self.model, opts)

        self.model_averager: BaseModelAverager | None = build_model_averager(averaging_config, self.model)
        if self.model_averager is not None:
            self.model_averager.attach_to_module(self)

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
        result = self._finalize_epoch(stage="test")
        if result is None:
            return
        preds, gts = result
        trainer = getattr(self, "trainer", None)
        if self.args.is_classification and trainer is not None and trainer.is_global_zero:
            self._log_confusion_matrix(preds, gts)

    def on_fit_start(self):
        super().on_fit_start()
        if self.model_averager is not None:
            self.model_averager.on_fit_start(self.trainer)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if self.model_averager is not None:
            self.model_averager.on_load_checkpoint(checkpoint)

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
            self.log(
                f"{stage}_loss",
                loss,
                prog_bar=True,
                sync_dist=True,
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=max(valid_count, 1),
            )

        preds = self._extract_valid_predictions(batch, logits)
        if preds is not None:
            self._stage_outputs[stage].append(preds)

        return loss if stage == "train" else None

    def _compute_loss(self, logits, batch):
        if self.args.is_seq:
            targets = batch["tokens"][self.args.label_name].to(self.args.device)
        else:
            targets = batch["metadata"][self.args.label_name].to(self.args.device)

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
        if self.args.is_seq:
            labels = batch["tokens"][self.args.label_name].to(self.args.device)
        else:
            labels = batch["metadata"][self.args.label_name].to(self.args.device)

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

    @staticmethod
    def _layer_mix_snapshot(model: torch.nn.Module):
        getter = getattr(model, "layer_mix_snapshot", None)
        if not callable(getter):
            return None
        return getter()

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
        if not outputs:
            return None

        preds, gts = zip(*outputs)
        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)

        metrics = compute_downstream_metrics(
            gts,
            preds,
            is_classification=self.args.is_classification,
            output_dim=getattr(self.args, "output_dim", None),
            stage_names=getattr(self.args, "stage_names", None),
        )
        for k, v in metrics.items():
            self.log(
                f"{stage}_{k}",
                v,
                prog_bar=(stage != "train"),
                logger=True,
                sync_dist=True,
                on_epoch=True,
            )

        outputs.clear()
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

    def _log_confusion_matrix(self, preds: np.ndarray, gts: np.ndarray):
        if getattr(wandb, "run", None) is None:
            return
        pred_labels = preds.argmax(axis=1)
        cm = confusion_matrix(gts, pred_labels)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Test Confusion Matrix (epoch {self.current_epoch})")
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

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
