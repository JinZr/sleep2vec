import pytorch_lightning as pl


class GradScaleLoggerCallback(pl.Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        if not trainer.is_global_zero:
            return

        scaler = getattr(trainer.precision_plugin, "scaler", None)
        get_scale = getattr(scaler, "get_scale", None)
        if get_scale is None:
            return

        pl_module.log(
            "train/grad_scale",
            float(get_scale()),
            prog_bar=False,
            logger=True,
            sync_dist=False,
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )
