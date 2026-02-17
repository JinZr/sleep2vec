from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import torch
import wandb


class MoEStatsCallback(pl.Callback):
    """Logs MoE utilization stats at a fixed step interval."""

    def __init__(self, every_n_steps: int = 50) -> None:
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))

    @staticmethod
    def _entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        probs = probs.to(dtype=torch.float32)
        probs = probs / probs.sum().clamp_min(eps)
        return -torch.sum(probs * torch.log(probs.clamp_min(eps)))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if trainer.global_step == 0 or trainer.global_step % self.every_n_steps != 0:
            return
        if not trainer.is_global_zero:
            return

        model = getattr(pl_module, "model", None)
        stats = getattr(model, "last_moe_stats", None)
        if not isinstance(stats, dict):
            return
        merged = stats.get("merged")
        if not isinstance(merged, dict):
            return

        load = merged.get("mean/expert_load")
        importance = merged.get("mean/expert_importance")
        if not torch.is_tensor(load) or load.numel() == 0:
            return
        load = load.detach().to(dtype=torch.float32).cpu()

        ent = self._entropy(load)
        load_mean = load.mean().clamp_min(1e-8)
        load_std = load.std(unbiased=False)
        load_cv = load_std / load_mean
        load_max = torch.max(load)
        load_min = torch.min(load).clamp_min(1e-8)
        max_min_ratio = load_max / load_min
        top_idx = int(torch.argmax(load).item())
        top_share = float(load[top_idx].item())

        pl_module.log("train/moe_util_entropy", ent, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        pl_module.log("train/moe_util_cv", load_cv, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        pl_module.log("train/moe_util_max_min_ratio", max_min_ratio, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        pl_module.log("train/moe_top1_expert_id", float(top_idx), on_step=True, on_epoch=False, logger=True, sync_dist=False)
        pl_module.log(
            "train/moe_top1_expert_share",
            top_share,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )

        dropped = merged.get("mean/dropped_fraction")
        if torch.is_tensor(dropped) and dropped.numel() == 1:
            pl_module.log("train/moe_dropped_rate", dropped.detach(), on_step=True, on_epoch=False, logger=True, sync_dist=False)

        if torch.is_tensor(importance) and importance.numel() == load.numel():
            importance = importance.detach().to(dtype=torch.float32).cpu()
            imp_ent = self._entropy(importance)
            pl_module.log("train/moe_importance_entropy", imp_ent, on_step=True, on_epoch=False, logger=True, sync_dist=False)

        if getattr(wandb, "run", None) is None:
            return

        log_payload = {
            "train/moe_expert_load_hist": wandb.Histogram(load.numpy()),
            "train/moe_expert_load_table": wandb.Table(
                columns=["expert_id", "load_fraction"],
                data=[[idx, float(value)] for idx, value in enumerate(load.tolist())],
            ),
        }

        by_modality = stats.get("by_modality")
        if isinstance(by_modality, dict) and by_modality:
            rows = []
            modality_names = sorted(by_modality.keys())
            for modality in modality_names:
                modality_stats = by_modality.get(modality, {})
                mod_load = modality_stats.get("mean/expert_load")
                if not torch.is_tensor(mod_load):
                    continue
                mod_vec = mod_load.detach().to(dtype=torch.float32).cpu().numpy()
                rows.append(mod_vec)
            if rows:
                matrix = np.stack(rows, axis=0)
                columns = ["modality"] + [f"expert_{idx}" for idx in range(matrix.shape[1])]
                table_data = []
                for modality, vec in zip(modality_names, matrix.tolist()):
                    table_data.append([modality] + [float(value) for value in vec])
                log_payload["train/moe_modality_expert_load"] = wandb.Table(columns=columns, data=table_data)

        wandb.log(log_payload, step=trainer.global_step, commit=False)
