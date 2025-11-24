import math

import pytorch_lightning as pl
import torch
from transformers.models.switch_transformers.modeling_switch_transformers import (
    load_balancing_loss_func,
)

from .losses import create_loss
from .pretrain_model import Sleep2vecPretrainModel


class Sleep2vecPretraining(pl.LightningModule):
    def __init__(self, args, model_config, loss_config):
        super().__init__()
        self.args = args
        self.model_config = model_config
        self.loss_config = loss_config
        self.loss_fn = self._build_loss()
        self.router_loss_weight = getattr(self.loss_config, "params", {}).get("router_lb_coef", 0.0)
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

    # ---------- Train ----------
    def training_step(self, batch, batch_idx):
        loss, acc = self._contrastive_step(batch, log_prefix="train")
        return loss

    # # ---------- Validation ----------

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, acc = self._contrastive_step(batch, log_prefix="val")

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

    # ---------- 公共计算逻辑 ----------
    def _contrastive_step(self, batch, log_prefix=None):
        model_out = self.model(batch, apply_mask=True, return_router=self.router_loss_weight > 0)
        if isinstance(model_out, (list, tuple)) and len(model_out) == 3:
            first_hidden, second_hidden, router_outputs = model_out
        else:
            first_hidden, second_hidden = model_out
            router_outputs = None

        loss_out = self.loss_fn(first_hidden, second_hidden, batch)
        total_loss = loss_out.loss
        metrics = loss_out.metrics or {}
        contrastive_loss = metrics.get("contrastive_loss", total_loss.detach())
        acc_contrastive = metrics.get("contrastive_acc")

        router_lb_loss = None
        if self.router_loss_weight > 0 and router_outputs is not None:
            router_lb_loss = self._compute_router_aux_loss(router_outputs)
            if router_lb_loss is not None:
                total_loss = total_loss + self.router_loss_weight * router_lb_loss

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

            if router_lb_loss is not None:
                self.log(
                    f"{log_prefix}_router_lb_loss",
                    router_lb_loss,
                    prog_bar=False,
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

    def _compute_router_aux_loss(self, router_outputs):
        """Compute mean load-balancing loss across all available router outputs."""
        if router_outputs is None:
            return None

        per_view_losses: list[torch.Tensor] = []

        for single_router in router_outputs:
            if single_router is None:
                continue

            # router outputs may be a tuple/list over layers; normalize to a flat list
            layers = list(single_router) if isinstance(single_router, (list, tuple)) else [single_router]
            for layer_out in layers:
                if isinstance(layer_out, (list, tuple)) and layer_out and isinstance(layer_out[0], torch.Tensor):
                    router_logits = layer_out[0]
                elif isinstance(layer_out, torch.Tensor):
                    router_logits = layer_out
                else:
                    continue

                if router_logits.dim() < 2:
                    continue

                router_probs = torch.softmax(router_logits, dim=-1)
                expert_indices = torch.argmax(router_logits, dim=-1)
                per_view_losses.append(load_balancing_loss_func(router_probs, expert_indices))

        if not per_view_losses:
            return None

        return torch.stack(per_view_losses).mean()

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
