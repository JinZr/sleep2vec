import math

import pytorch_lightning as pl
import torch

from sleep2vec.losses import create_loss
from sleep2vec.pretrain.ema import clone_ema_model, cosine_ema_momentum, ema_update
from sleep2vec.pretrain_model import Sleep2vecPretrainModel


class Sleep2vecPretraining(pl.LightningModule):
    def __init__(self, args, model_config, loss_config, ema_config=None):
        super().__init__()
        self.args = args
        self.model_config = model_config
        self.loss_config = loss_config
        self.ema_config = ema_config
        self.loss_fn = self._build_loss()
        self.model = Sleep2vecPretrainModel(
            channel_feature_dim=None,
            transformer_hidden_size=model_config.backbone.hidden_size,
            transformer_num_hidden_layers=model_config.backbone.num_hidden_layers,
            transformer_num_attention_heads=model_config.backbone.num_attention_heads,
            channel_names=[c.name for c in model_config.channels],
            projection=True,
            encoder_factory=None,
            model_config=model_config,
            projection_config=model_config.projection,
        )

        # 缓存 val 损失（每 step append，epoch 末取均值）
        self.val_losses = []
        self.val_contrastive_laccs = []
        self.val_contrastive_loss = []
        self.val_contrastive_sample = []

        self.ema_model = None
        self._ema_total_steps = None
        if self.ema_config is not None and getattr(self.ema_config, "enabled", False):
            self.ema_model = clone_ema_model(self.model)

    # ---------- Train ----------
    def training_step(self, batch, batch_idx):
        loss, acc = self._contrastive_step(batch, log_prefix="train", model=self.model)
        return loss

    # # ---------- Validation ----------

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        eval_model = self._get_eval_backbone()
        loss, acc = self._contrastive_step(batch, log_prefix="val", model=eval_model)

        if dataloader_idx == 0:
            self.val_losses.append(loss.detach())
        elif dataloader_idx == 1:
            self.log("extra_val_loss", loss, prog_bar=True, sync_dist=True)

        if acc is not None:
            self.val_contrastive_laccs.append(acc.detach())
        return loss

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
        if self.ema_model is not None:
            if hasattr(self.trainer, "estimated_stepping_batches"):
                self._ema_total_steps = int(self.trainer.estimated_stepping_batches)
            else:
                self._ema_total_steps = None

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if not self._should_use_ema():
            return
        state_dict = checkpoint.get("state_dict", {})
        has_ema = any(k.startswith("ema_model.") for k in state_dict)
        if has_ema:
            return
        # Seed EMA weights from student when resuming old checkpoints without EMA.
        if self.ema_model is None:
            self.ema_model = clone_ema_model(self.model)
        ema_state = {f"ema_model.{k[len('model.'):]}": v for k, v in state_dict.items() if k.startswith("model.")}
        if ema_state:
            state_dict.update(ema_state)
            checkpoint["state_dict"] = state_dict

    def _get_eval_backbone(self):
        if (
            self.ema_model is not None
            and self.ema_config is not None
            and getattr(self.ema_config, "use_for_eval", False)
        ):
            return self.ema_model
        return self.model

    def _should_use_ema(self) -> bool:
        return self.ema_model is not None and self.ema_config is not None and getattr(self.ema_config, "enabled", False)

    def _ema_momentum_for_step(self) -> float:
        if not self._should_use_ema():
            return 0.0
        total_steps = self._ema_total_steps
        if total_steps is None:
            total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
        base_m = self.ema_config.base_momentum
        final_m = getattr(self.ema_config, "final_momentum", 1.0)
        return cosine_ema_momentum(
            step=self.global_step,
            total_steps=total_steps,
            base_momentum=base_m,
            final_momentum=final_m,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)
        if self._should_use_ema():
            momentum = self._ema_momentum_for_step()
            ema_update(self.model, self.ema_model, momentum=momentum)

    # ---------- 公共计算逻辑 ----------
    def _contrastive_step(self, batch, log_prefix=None, model=None):
        model = model or self.model
        first_hidden, second_hidden = model(batch, apply_mask=True)

        loss_out = self.loss_fn(first_hidden, second_hidden, batch)
        total_loss = loss_out.loss
        metrics = loss_out.metrics or {}
        contrastive_loss = metrics.get("contrastive_loss", total_loss.detach())
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

        # 线性 warmup + 余弦退火
        total_steps = self.trainer.estimated_stepping_batches
        warmup = int(0.03 * total_steps)  # 3% 亦可 2%~5%

        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            # cosine from 1→0.1
            progress = (step - warmup) / float(max(1, total_steps - warmup))
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _build_loss(self):
        loss_kwargs = dict(self.loss_config.params or {})
        loss_kwargs.setdefault("temperature", self.loss_config.temperature)
        return create_loss(self.loss_config.name, **loss_kwargs)
