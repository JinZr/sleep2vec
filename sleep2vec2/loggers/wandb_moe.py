from __future__ import annotations

import typing as t

from pytorch_lightning.loggers import WandbLogger
import torch
import wandb


class WandbMoELogger:
    def __init__(self, *, hist_every_n_steps: int = 200) -> None:
        self.hist_every_n_steps = int(hist_every_n_steps)

    def log_step(
        self,
        module,
        *,
        log_prefix: str,
        batch_size: int,
        loss: torch.Tensor,
        contrastive_loss: torch.Tensor,
        contrastive_acc: torch.Tensor | None,
        moe_aux: torch.Tensor | None,
        moe_z: torch.Tensor | None,
        route_align: torch.Tensor | None,
        extras: dict | None,
    ) -> None:
        self._log_scalars(
            module,
            log_prefix=log_prefix,
            batch_size=batch_size,
            loss=loss,
            contrastive_loss=contrastive_loss,
            contrastive_acc=contrastive_acc,
            moe_aux=moe_aux,
            moe_z=moe_z,
            route_align=route_align,
        )

        if not extras:
            return

        expert_load, expert_importance, route_entropy = self._extract_expert_stats(extras)
        self._log_expert_scalars(
            module,
            log_prefix=log_prefix,
            batch_size=batch_size,
            expert_load=expert_load,
            expert_importance=expert_importance,
            route_entropy=route_entropy,
        )
        self._log_expert_histograms(
            module,
            log_prefix=log_prefix,
            expert_load=expert_load,
            expert_importance=expert_importance,
            route_entropy=route_entropy,
        )

    def _log_scalars(
        self,
        module,
        *,
        log_prefix: str,
        batch_size: int,
        loss: torch.Tensor,
        contrastive_loss: torch.Tensor,
        contrastive_acc: torch.Tensor | None,
        moe_aux: torch.Tensor | None,
        moe_z: torch.Tensor | None,
        route_align: torch.Tensor | None,
    ) -> None:
        module.log(
            f"{log_prefix}_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        module.log(
            f"{log_prefix}_contrastive_loss",
            contrastive_loss,
            prog_bar=False,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        if contrastive_acc is not None:
            module.log(
                f"{log_prefix}_contrastive_acc",
                contrastive_acc,
                prog_bar=True,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
        if moe_aux is not None:
            module.log(
                f"{log_prefix}_moe_aux",
                moe_aux,
                prog_bar=False,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
        if moe_z is not None:
            module.log(
                f"{log_prefix}_moe_z",
                moe_z,
                prog_bar=False,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
        if route_align is not None:
            module.log(
                f"{log_prefix}_route_align",
                route_align,
                prog_bar=False,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )

    def _extract_expert_stats(self, extras: dict) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        def _coalesce(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                if not value:
                    return None
                return torch.stack(list(value), dim=0).mean(dim=0)
            return value

        load_first = _coalesce(extras.get("moe_load_first"))
        load_second = _coalesce(extras.get("moe_load_second"))
        if load_first is not None and load_second is not None:
            expert_load = 0.5 * (load_first + load_second)
        else:
            expert_load = load_first if load_first is not None else load_second

        imp_first = _coalesce(extras.get("moe_importance_first"))
        imp_second = _coalesce(extras.get("moe_importance_second"))
        if imp_first is not None and imp_second is not None:
            expert_importance = 0.5 * (imp_first + imp_second)
        else:
            expert_importance = imp_first if imp_first is not None else imp_second

        ent_first = _coalesce(extras.get("moe_entropy_first"))
        ent_second = _coalesce(extras.get("moe_entropy_second"))
        if ent_first is not None and ent_second is not None:
            route_entropy = 0.5 * (ent_first + ent_second)
        else:
            route_entropy = ent_first if ent_first is not None else ent_second

        return expert_load, expert_importance, route_entropy

    def _log_expert_scalars(
        self,
        module,
        *,
        log_prefix: str,
        batch_size: int,
        expert_load: torch.Tensor | None,
        expert_importance: torch.Tensor | None,
        route_entropy: torch.Tensor | None,
    ) -> None:
        if route_entropy is not None:
            module.log(
                f"{log_prefix}_moe_route_entropy",
                route_entropy,
                prog_bar=False,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )

        if expert_load is not None:
            expert_load = expert_load.detach()
            load_mean = expert_load.mean().clamp_min(1e-6)
            module.log(
                f"{log_prefix}_moe_load_cv",
                expert_load.std() / load_mean,
                prog_bar=False,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
            module.log(
                f"{log_prefix}_moe_load_max",
                expert_load.max(),
                prog_bar=False,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
            module.log(
                f"{log_prefix}_moe_load_min",
                expert_load.min(),
                prog_bar=False,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
            for idx, val in enumerate(expert_load):
                module.log(
                    f"{log_prefix}_moe_load_e{idx}",
                    val,
                    prog_bar=False,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                )

        if expert_importance is not None:
            expert_importance = expert_importance.detach()
            for idx, val in enumerate(expert_importance):
                module.log(
                    f"{log_prefix}_moe_importance_e{idx}",
                    val,
                    prog_bar=False,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                )

    def _log_expert_histograms(
        self,
        module,
        *,
        log_prefix: str,
        expert_load: torch.Tensor | None,
        expert_importance: torch.Tensor | None,
        route_entropy: torch.Tensor | None,
    ) -> None:
        logger = self._get_wandb_logger(module)
        if logger is None:
            return

        step = getattr(module, "global_step", None)
        if step is None or step < 0:
            return
        if self.hist_every_n_steps > 0 and (step % self.hist_every_n_steps) != 0:
            return

        payload: dict[str, t.Any] = {}
        if expert_load is not None:
            payload[f"{log_prefix}/moe_load_hist"] = wandb.Histogram(
                expert_load.detach().cpu().numpy()
            )
        if expert_importance is not None:
            payload[f"{log_prefix}/moe_importance_hist"] = wandb.Histogram(
                expert_importance.detach().cpu().numpy()
            )
        if route_entropy is not None:
            payload[f"{log_prefix}/moe_route_entropy_hist"] = wandb.Histogram(
                route_entropy.detach().cpu().numpy()
            )

        if payload:
            logger.experiment.log(payload, step=step)

    @staticmethod
    def _get_wandb_logger(module) -> WandbLogger | None:
        logger = getattr(module, "logger", None)
        if isinstance(logger, WandbLogger):
            return logger
        if hasattr(module, "loggers"):
            for candidate in module.loggers:
                if isinstance(candidate, WandbLogger):
                    return candidate
        return None

