from __future__ import annotations

import logging
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from data.channel_selection import build_all_pairs
from sleep2vec.visualization.pair_acc import render_pair_acc_heatmap


class PairAccLoggerCallback(pl.Callback):
    def __init__(
        self,
        modality_names: t.Sequence[str],
        *,
        log_prefix: str = "val_pair_acc",
        matrix_key: str = "val_pair_acc_matrix",
        train_pair_monitor_enabled: bool = True,
        train_pair_log_prefix: str = "train_pair_sampling",
        train_pair_skew_warn_threshold: float = 0.05,
        train_pair_min_unique_coverage_warn_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self._modality_names = list(modality_names)
        self._log_prefix = log_prefix
        self._matrix_key = matrix_key
        self._train_pair_monitor_enabled = bool(train_pair_monitor_enabled)
        self._train_pair_log_prefix = train_pair_log_prefix
        self._train_pair_skew_warn_threshold = float(train_pair_skew_warn_threshold)
        self._train_pair_min_unique_coverage_warn_threshold = float(train_pair_min_unique_coverage_warn_threshold)
        self._val_pairs: list[tuple[str, str] | None] = []
        self._pair_sums: torch.Tensor | None = None
        self._pair_counts: torch.Tensor | None = None

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        loaders = trainer.val_dataloaders or []
        if not loaders:
            self._val_pairs = []
            self._pair_sums = None
            self._pair_counts = None
            return
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]

        val_pairs: list[tuple[str, str] | None] = []
        missing_pair = False
        for loader in loaders:
            dataset = getattr(loader, "dataset", None)
            if dataset is not None and hasattr(dataset, "reset_pair_selector"):
                dataset.reset_pair_selector()
            pair = getattr(dataset, "pair", None) if dataset is not None else None
            if pair is None:
                missing_pair = True
            val_pairs.append(pair)

        if missing_pair:
            fallback = build_all_pairs(self._modality_names)
            if len(fallback) == len(val_pairs):
                val_pairs = list(fallback)

        self._val_pairs = val_pairs
        device = pl_module.device
        self._pair_sums = torch.zeros(len(val_pairs), device=device)
        self._pair_counts = torch.zeros(len(val_pairs), device=device)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is None or self._pair_sums is None or self._pair_counts is None:
            return
        if not isinstance(outputs, dict):
            return
        acc = outputs.get("acc")
        if acc is None:
            return
        batch_size = outputs.get("batch_size")
        if batch_size is None:
            length = batch.get("length") if isinstance(batch, dict) else None
            batch_size = int(length.shape[0]) if length is not None else 1
        if dataloader_idx >= self._pair_sums.numel():
            return
        self._pair_sums[dataloader_idx] += acc.detach() * batch_size
        self._pair_counts[dataloader_idx] += batch_size

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self._pair_sums is None or self._pair_counts is None or not self._val_pairs:
            return
        sums_g = pl_module.all_gather(self._pair_sums)
        counts_g = pl_module.all_gather(self._pair_counts)
        if not isinstance(sums_g, torch.Tensor):
            return

        if sums_g.dim() == 1:
            sum_total = sums_g
            count_total = counts_g
        else:
            sum_total = sums_g.sum(dim=0)
            count_total = counts_g.sum(dim=0)
        count_total = count_total.clamp_min(1)
        mean_acc = sum_total / count_total

        if trainer.is_global_zero:
            mean_acc_cpu = mean_acc.detach().cpu().numpy()
            for pair, val in zip(self._val_pairs, mean_acc_cpu):
                if pair is None:
                    continue
                pair_name = f"{pair[0]}__{pair[1]}"
                pl_module.log(
                    f"{self._log_prefix}/{pair_name}",
                    float(val),
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                    on_step=False,
                    on_epoch=True,
                )
            matrix = self._build_matrix(mean_acc_cpu)
            if getattr(wandb, "run", None) is not None:
                fig = render_pair_acc_heatmap(matrix, self._modality_names)
                wandb.log({self._matrix_key: wandb.Image(fig)}, commit=False)
                plt.close(fig)

        self._pair_sums = None
        self._pair_counts = None

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not self._train_pair_monitor_enabled:
            return

        sampler = self._resolve_train_pair_sampler(trainer)
        if sampler is None:
            return

        pair_list = getattr(sampler, "pairs", None)
        if not pair_list:
            return
        pair_list = list(pair_list)

        local_count_map = sampler.get_last_epoch_counts() if hasattr(sampler, "get_last_epoch_counts") else {}
        local_unique_map = (
            sampler.get_last_epoch_unique_sample_counts()
            if hasattr(sampler, "get_last_epoch_unique_sample_counts")
            else {}
        )
        pool_size_map = sampler.get_pair_pool_sizes() if hasattr(sampler, "get_pair_pool_sizes") else {}
        target_dist = sampler.get_target_distribution() if hasattr(sampler, "get_target_distribution") else {}

        device = pl_module.device
        counts_local = torch.tensor([float(local_count_map.get(pair, 0)) for pair in pair_list], device=device)
        unique_local = torch.tensor([float(local_unique_map.get(pair, 0)) for pair in pair_list], device=device)
        pool_sizes = torch.tensor([float(pool_size_map.get(pair, 0)) for pair in pair_list], device=device)
        targets = torch.tensor(
            [float(target_dist.get(pair, target_dist.get((pair[1], pair[0]), 0.0))) for pair in pair_list],
            device=device,
        )

        if float(targets.sum().item()) <= 0.0:
            targets = torch.full_like(targets, 1.0 / float(max(1, len(pair_list))))
        else:
            targets = targets / targets.sum()

        counts_g = pl_module.all_gather(counts_local)
        if not isinstance(counts_g, torch.Tensor):
            return
        if counts_g.dim() == 1:
            counts_total = counts_g
        else:
            counts_total = counts_g.sum(dim=0)

        unique_g = pl_module.all_gather(unique_local)
        if isinstance(unique_g, torch.Tensor):
            if unique_g.dim() == 1:
                unique_total = unique_g
            else:
                unique_total = unique_g.sum(dim=0)
        else:
            unique_total = unique_local
        unique_total = torch.minimum(unique_total, pool_sizes.clamp_min(0.0))

        total_batches = counts_total.sum().clamp_min(1.0)
        ratios = counts_total / total_batches
        abs_dev = (ratios - targets).abs()
        unique_coverage = unique_total / pool_sizes.clamp_min(1.0)
        skew_alert = bool(torch.any(abs_dev > self._train_pair_skew_warn_threshold).item())
        low_coverage_alert = bool(
            torch.any(unique_coverage < self._train_pair_min_unique_coverage_warn_threshold).item()
        )

        if trainer.is_global_zero:
            counts_cpu = counts_total.detach().cpu().numpy()
            ratios_cpu = ratios.detach().cpu().numpy()
            targets_cpu = targets.detach().cpu().numpy()
            dev_cpu = abs_dev.detach().cpu().numpy()
            unique_cpu = unique_total.detach().cpu().numpy()
            pool_cpu = pool_sizes.detach().cpu().numpy()
            cov_cpu = unique_coverage.detach().cpu().numpy()

            for pair, cnt, ratio, target, dev, uniq, pool, cov in zip(
                pair_list, counts_cpu, ratios_cpu, targets_cpu, dev_cpu, unique_cpu, pool_cpu, cov_cpu
            ):
                pair_name = f"{pair[0]}__{pair[1]}"
                pl_module.log(
                    f"{self._train_pair_log_prefix}/count/{pair_name}",
                    float(cnt),
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                    on_step=False,
                    on_epoch=True,
                )
                pl_module.log(
                    f"{self._train_pair_log_prefix}/ratio/{pair_name}",
                    float(ratio),
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                    on_step=False,
                    on_epoch=True,
                )
                pl_module.log(
                    f"{self._train_pair_log_prefix}/target_ratio/{pair_name}",
                    float(target),
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                    on_step=False,
                    on_epoch=True,
                )
                pl_module.log(
                    f"{self._train_pair_log_prefix}/abs_dev/{pair_name}",
                    float(dev),
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                    on_step=False,
                    on_epoch=True,
                )
                pl_module.log(
                    f"{self._train_pair_log_prefix}/pool_size/{pair_name}",
                    float(pool),
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                    on_step=False,
                    on_epoch=True,
                )
                pl_module.log(
                    f"{self._train_pair_log_prefix}/unique_samples/{pair_name}",
                    float(uniq),
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                    on_step=False,
                    on_epoch=True,
                )
                pl_module.log(
                    f"{self._train_pair_log_prefix}/unique_coverage/{pair_name}",
                    float(cov),
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                    on_step=False,
                    on_epoch=True,
                )

                if float(dev) > self._train_pair_skew_warn_threshold:
                    logging.warning(
                        "Train pair sampling skew exceeds threshold: epoch=%s pair=%s actual=%.6f target=%.6f dev=%.6f "
                        "threshold=%.6f",
                        trainer.current_epoch,
                        pair_name,
                        float(ratio),
                        float(target),
                        float(dev),
                        self._train_pair_skew_warn_threshold,
                    )
                if float(cov) < self._train_pair_min_unique_coverage_warn_threshold:
                    logging.warning(
                        "Train pair sampling unique coverage is low: epoch=%s pair=%s unique_coverage=%.6f "
                        "threshold=%.6f unique=%d pool=%d",
                        trainer.current_epoch,
                        pair_name,
                        float(cov),
                        self._train_pair_min_unique_coverage_warn_threshold,
                        int(uniq),
                        int(pool),
                    )

            pl_module.log(
                f"{self._train_pair_log_prefix}/num_pairs",
                float(len(pair_list)),
                prog_bar=False,
                logger=True,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
            pl_module.log(
                f"{self._train_pair_log_prefix}/total_batches",
                float(total_batches.item()),
                prog_bar=False,
                logger=True,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
            pl_module.log(
                f"{self._train_pair_log_prefix}/max_abs_dev",
                float(abs_dev.max().item()),
                prog_bar=False,
                logger=True,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
            pl_module.log(
                f"{self._train_pair_log_prefix}/mean_abs_dev",
                float(abs_dev.mean().item()),
                prog_bar=False,
                logger=True,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
            pl_module.log(
                f"{self._train_pair_log_prefix}/min_unique_coverage",
                float(unique_coverage.min().item()),
                prog_bar=False,
                logger=True,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
            pl_module.log(
                f"{self._train_pair_log_prefix}/skew_alert",
                float(skew_alert),
                prog_bar=False,
                logger=True,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
            pl_module.log(
                f"{self._train_pair_log_prefix}/low_unique_coverage_alert",
                float(low_coverage_alert),
                prog_bar=False,
                logger=True,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )

    def _build_matrix(self, pair_acc: np.ndarray) -> np.ndarray:
        size = len(self._modality_names)
        idx_map = {name: i for i, name in enumerate(self._modality_names)}
        mat = np.zeros((size, size), dtype=np.float32)
        for pair, val in zip(self._val_pairs, pair_acc):
            if pair is None:
                continue
            left, right = pair
            if left not in idx_map or right not in idx_map:
                continue
            i = idx_map[left]
            j = idx_map[right]
            mat[i, j] = val
            mat[j, i] = val
        for i in range(size):
            mat[i, i] = 1.0
        return mat

    def _resolve_train_pair_sampler(self, trainer):
        train_loader = getattr(trainer, "train_dataloader", None)
        if isinstance(train_loader, dict):
            if not train_loader:
                return None
            train_loader = next(iter(train_loader.values()))
        if isinstance(train_loader, (list, tuple)):
            if not train_loader:
                return None
            train_loader = train_loader[0]
        if train_loader is None:
            return None
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if batch_sampler is None:
            return None
        if hasattr(batch_sampler, "get_last_epoch_counts") and hasattr(batch_sampler, "get_target_distribution"):
            return batch_sampler
        return None
