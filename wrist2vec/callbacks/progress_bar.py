from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import RichProgressBar, TQDMProgressBar
from pytorch_lightning.utilities.imports import _RICH_AVAILABLE


class DistributedAHITQDMProgressBar(TQDMProgressBar):
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Skip rank-zero-only epoch-end UI work; batch-level progress stays intact.
        return None


class DistributedAHIRichProgressBar(RichProgressBar):
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Skip rank-zero-only epoch-end UI work; batch-level progress stays intact.
        return None


def build_distributed_ahi_progress_bar():
    return DistributedAHIRichProgressBar() if _RICH_AVAILABLE else DistributedAHITQDMProgressBar()
