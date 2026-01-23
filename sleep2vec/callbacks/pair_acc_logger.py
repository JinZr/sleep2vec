from __future__ import annotations

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
    ) -> None:
        super().__init__()
        self._modality_names = list(modality_names)
        self._log_prefix = log_prefix
        self._matrix_key = matrix_key
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
            fig = render_pair_acc_heatmap(matrix, self._modality_names)
            wandb.log({self._matrix_key: wandb.Image(fig)}, commit=False)
            plt.close(fig)

        self._pair_sums = None
        self._pair_counts = None

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
