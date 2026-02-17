from __future__ import annotations

import pytorch_lightning as pl
import torch
import wandb


class MoEUtilizationLoggerCallback(pl.Callback):
    """Logs MoE expert utilization summaries during training."""

    def __init__(self, every_n_steps: int = 200) -> None:
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if trainer.global_step == 0 or trainer.global_step % self.every_n_steps != 0:
            return
        if not trainer.is_global_zero:
            return

        metrics = getattr(pl_module, "_last_moe_metrics", None)
        if not isinstance(metrics, dict) or not metrics:
            return

        expert_load = metrics.get("moe/avg/expert_load")
        if expert_load is None:
            expert_load = metrics.get("moe/avg/mean/expert_load")
        if expert_load is None or not torch.is_tensor(expert_load):
            return

        expert_load = expert_load.detach().to(dtype=torch.float32).cpu()
        if expert_load.numel() == 0:
            return

        top_expert = int(torch.argmax(expert_load).item())
        top_share = float(expert_load[top_expert].item())
        pl_module.log(
            "train/moe_top_expert_share",
            top_share,
            prog_bar=False,
            logger=True,
            sync_dist=False,
            on_step=True,
            on_epoch=False,
        )
        pl_module.log(
            "train/moe_top_expert_id",
            float(top_expert),
            prog_bar=False,
            logger=True,
            sync_dist=False,
            on_step=True,
            on_epoch=False,
        )

        for expert_idx, value in enumerate(expert_load.tolist()):
            pl_module.log(
                f"train/moe_expert_load/{expert_idx}",
                float(value),
                prog_bar=False,
                logger=True,
                sync_dist=False,
                on_step=True,
                on_epoch=False,
            )

        if getattr(wandb, "run", None) is None:
            return

        table = wandb.Table(columns=["expert_id", "load_fraction"])
        for expert_idx, value in enumerate(expert_load.tolist()):
            table.add_data(expert_idx, float(value))

        wandb.log(
            {
                "train/moe_expert_load_hist": wandb.Histogram(expert_load.numpy()),
                "train/moe_expert_load_table": table,
            },
            step=trainer.global_step,
            commit=False,
        )
