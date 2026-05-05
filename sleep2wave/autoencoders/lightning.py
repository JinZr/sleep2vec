from __future__ import annotations

from dataclasses import asdict

import pytorch_lightning as pl
import torch
import yaml

from sleep2wave.autoencoders.losses import Sleep2WaveAutoencoderLoss
from sleep2wave.autoencoders.model import Sleep2WaveAutoencoder
from sleep2wave.generative.config import Sleep2WaveConfig


class Sleep2WaveAutoencoderLightning(pl.LightningModule):
    def __init__(self, config: Sleep2WaveConfig) -> None:
        super().__init__()
        if config.autoencoder is None:
            raise ValueError("sleep2wave autoencoder training requires an autoencoder config.")
        if config.training is None:
            raise ValueError("sleep2wave autoencoder training requires a training config.")
        self.config_bundle = config
        self.model = Sleep2WaveAutoencoder(
            latent_dim=config.autoencoder.latent_dim,
            encoder_type=config.autoencoder.encoder_type,
            decoder_type=config.autoencoder.decoder_type,
            modalities=config.modalities.all,
        )
        self.loss_fn = Sleep2WaveAutoencoderLoss(config.autoencoder.losses)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["sleep2wave_config"] = asdict(self.config_bundle)
        checkpoint["sleep2wave_config_yaml"] = yaml.safe_dump(checkpoint["sleep2wave_config"], sort_keys=True)

    def training_step(self, batch, batch_idx):
        output = self.model(batch["clean_signals"])
        losses = self.loss_fn(
            output.reconstructions,
            batch["clean_signals"],
            availability_mask=batch.get("availability_mask"),
            quality_mask=batch.get("quality_mask"),
            channel_mask=batch.get("channel_mask"),
        )
        batch_size = next(iter(batch["clean_signals"].values())).shape[0]
        for name, value in losses.items():
            self.log(
                f"train_{name}",
                value,
                prog_bar=(name == "loss"),
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
        return losses["loss"]

    def configure_optimizers(self):
        training_cfg = self.config_bundle.training
        if training_cfg is None:
            raise ValueError("sleep2wave autoencoder training requires a training config.")

        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and "bias" not in name.lower() and "norm" not in name.lower():
                decay.append(param)
            else:
                no_decay.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": training_cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=training_cfg.lr,
        )


__all__ = ["Sleep2WaveAutoencoderLightning"]
