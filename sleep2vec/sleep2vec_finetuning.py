from dataclasses import asdict
import logging

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import wandb
import yaml

from sleep2vec import diagnostics
from sleep2vec.metrics import compute_downstream_metrics

from .downstream_model import Sleep2vecDownstreamModel
from .pretrain_model import Sleep2vecPretrainModel


class Sleep2vecFinetuning(pl.LightningModule):
    def __init__(self, args, model_config):
        super().__init__()
        self.args = args
        self.model_config = model_config

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

        self._stage_outputs = {"train": [], "val": [], "test": []}
        self._classification_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self._regression_loss = torch.nn.MSELoss()

        # Optional tensor diagnostics (borrowed from icefall)
        self._diagnostic = None
        self._diag_steps = getattr(args, "diagnostics_steps", 5)
        if getattr(args, "print_diagnostics", False):
            opts = diagnostics.TensorDiagnosticOptions(max_eig_dim=512)
            self._diagnostic = diagnostics.attach_diagnostics(self.model, opts)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["model_config"] = asdict(self.model_config)
        checkpoint["model_config_yaml"] = yaml.safe_dump(checkpoint["model_config"], sort_keys=True)

    # ---------- Lightning hooks ----------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, stage="test")

    def on_train_epoch_end(self):
        self._finalize_epoch(stage="train")

    def on_validation_epoch_end(self):
        self._finalize_epoch(stage="val")

    def on_test_epoch_end(self):
        result = self._finalize_epoch(stage="test")
        if result is None:
            return
        preds, gts = result
        trainer = getattr(self, "trainer", None)
        if self.args.is_classification and trainer is not None and trainer.is_global_zero:
            self._log_confusion_matrix(preds, gts)

    # ---------- Internal helpers ----------
    def _shared_step(self, batch, stage: str):
        logits = self.model(batch)
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
        if self._diagnostic is not None and self.global_step >= self._diag_steps:
            if self.trainer is not None:
                self.trainer.should_stop = True

    def on_train_end(self):
        super().on_train_end()
        if self._diagnostic is not None:
            self._diagnostic.print_diagnostics()

    def _log_confusion_matrix(self, preds: np.ndarray, gts: np.ndarray):
        if wandb.run is None:
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
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
