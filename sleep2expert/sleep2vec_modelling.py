from dataclasses import asdict

import pytorch_lightning as pl
import torch
import yaml

from sleep2expert import diagnostics
from sleep2expert.averagings.base import BaseModelAverager, build_model_averager
from sleep2expert.losses import create_loss
from sleep2expert.losses.moe_regularization import compute_moe_regularization
from sleep2expert.pretrain_model import Sleep2vecPretrainModel
from sleep2expert.schedulers import build_warmup_cosine_scheduler


class Sleep2vecPretraining(pl.LightningModule):
    def __init__(self, args, model_config, loss_config, averaging_config=None):
        super().__init__()
        self.args = args
        self.model_config = model_config
        self.loss_config = loss_config
        self.averaging_config = averaging_config
        self.loss_fn = self._build_loss()
        self.model = Sleep2vecPretrainModel(
            model_config=model_config,
        )

        # Optional tensor diagnostics (borrowed from icefall)
        self._diagnostic = None
        self._diag_steps = getattr(args, "diagnostics_steps", 5)
        if getattr(args, "print_diagnostics", False):
            opts = diagnostics.TensorDiagnosticOptions(max_eig_dim=512)
            self._diagnostic = diagnostics.attach_diagnostics(self.model, opts)

        # Cache per-step validation metrics that are summarized at epoch end.
        self.val_contrastive_loss = []
        self.val_contrastive_sample = []

        self.model_averager: BaseModelAverager | None = build_model_averager(averaging_config, self.model)
        if self.model_averager is not None:
            self.model_averager.attach_to_module(self)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        # Stash the full model config for later inspection/reproduction.
        checkpoint["model_config"] = asdict(self.model_config)
        checkpoint["model_config_yaml"] = yaml.safe_dump(checkpoint["model_config"], sort_keys=True)

    # ---------- Train ----------
    def training_step(self, batch, batch_idx):
        loss, acc = self._contrastive_step(batch, log_prefix="train", model=self.model)
        return loss

    # # ---------- Validation ----------

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        eval_model = self._get_eval_backbone()
        loss, acc = self._contrastive_step(batch, log_prefix="val", model=eval_model)
        batch_size = 1
        length = batch.get("length") if isinstance(batch, dict) else None
        if length is not None:
            batch_size = int(length.shape[0])
        return {"loss": loss, "acc": acc, "batch_size": batch_size}

    def on_validation_epoch_end(self):

        # 原有总 val_loss/acc（如果要保留，可继续）
        if self.val_contrastive_loss:
            # sample-wise
            _val_contrastive_loss = torch.stack(self.val_contrastive_loss).mean()
            self.log(
                "val_contrastive_loss",
                _val_contrastive_loss,
                prog_bar=True,
                sync_dist=True,
            )
            if self.val_contrastive_sample:
                _val_contrastive_sample = torch.stack(self.val_contrastive_sample).mean()
                self.log(
                    "val_contrastive_acc",
                    _val_contrastive_sample,
                    prog_bar=True,
                    sync_dist=True,
                )
            self.val_contrastive_loss.clear()
            self.val_contrastive_sample.clear()

    def on_fit_start(self):
        super().on_fit_start()
        if self.model_averager is not None:
            self.model_averager.on_fit_start(self.trainer)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if self.model_averager is not None:
            self.model_averager.on_load_checkpoint(checkpoint)

    def _get_eval_backbone(self):
        if self.model_averager is not None:
            return self.model_averager.eval_model()
        return self.model

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        if self.model_averager is not None:
            self.model_averager.on_train_batch_end(trainer=self.trainer, global_step=self.global_step)
        if self._diagnostic is not None and self.global_step >= self._diag_steps:
            # stop as soon as enough batches have been observed
            if self.trainer is not None:
                self.trainer.should_stop = True

    def on_train_end(self):
        super().on_train_end()
        if self._diagnostic is not None:
            self._diagnostic.print_diagnostics()

    # ---------- 公共计算逻辑 ----------
    def _contrastive_step(self, batch, log_prefix=None, model=None):
        model = model or self.model
        first_hidden, second_hidden = model(batch, apply_mask=True)

        loss_out = self.loss_fn(first_hidden, second_hidden, batch)
        moe_out = compute_moe_regularization(
            getattr(model, "last_moe_aux", None),
            self.model_config.backbone.moe,
            batch,
        )
        total_loss = loss_out.loss + moe_out.loss
        metrics = loss_out.metrics or {}
        moe_metrics = moe_out.metrics or {}
        contrastive_loss = metrics.get("contrastive_loss", loss_out.loss.detach())
        acc_contrastive = metrics.get("contrastive_acc")

        # # ---- logging ----
        # if log_prefix is not None:
        #     # step 级
        #     self.log(f"{log_prefix}_loss", total_loss, prog_bar=True, sync_dist=True)
        #     self.log(f"{log_prefix}_contrastive_loss", loss_contrastive_sample, prog_bar=False, sync_dist=True)
        #     self.log(f"{log_prefix}_contrastive_acc", acc_contrastive, prog_bar=True, sync_dist=True)

        # ---- logging ----
        if log_prefix is not None:
            B = first_hidden.size(0)  # 用于正确做加权平均
            self.log(
                f"{log_prefix}_loss",
                total_loss,
                prog_bar=True,
                sync_dist=True,
                on_step=True,  # 仍然保留每 step
                on_epoch=True,  # ✅ 新增：按 epoch 聚合
                batch_size=B,
            )  # ✅ 新增：正确做加权平均

            self.log(
                f"{log_prefix}_contrastive_loss",
                contrastive_loss,
                prog_bar=False,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                batch_size=B,
            )

            for metric_name, metric_value in moe_metrics.items():
                self.log(
                    f"{log_prefix}_{metric_name}",
                    metric_value,
                    prog_bar=False,
                    sync_dist=True,
                    on_step=True,
                    on_epoch=True,
                    batch_size=B,
                )

            if acc_contrastive is not None:
                self.log(
                    f"{log_prefix}_contrastive_acc",
                    acc_contrastive,
                    prog_bar=True,
                    sync_dist=True,
                    on_step=True,
                    on_epoch=True,
                    batch_size=B,
                )

        # 验证集：缓存到 epoch 末求均值
        if log_prefix == "val":
            self.val_contrastive_loss.append(contrastive_loss.detach())
            if acc_contrastive is not None:
                self.val_contrastive_sample.append(acc_contrastive.detach())

        return total_loss, acc_contrastive  # 返回一个主 acc（sample-wise）

    def configure_optimizers(self):
        # 参数分组：LN/BN权重与bias不做WD
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
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

        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            warmup_steps=getattr(self.args, "warmup_steps", None),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _build_loss(self):
        loss_kwargs = dict(self.loss_config.params or {})
        loss_kwargs.setdefault("temperature", self.loss_config.temperature)
        return create_loss(self.loss_config.name, **loss_kwargs)
