from __future__ import annotations

from dataclasses import asdict

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
import yaml

from sleep2wave.autoencoders.losses import Sleep2WaveAutoencoderLoss
from sleep2wave.autoencoders.model import Sleep2WaveAutoencoder
from sleep2wave.generative.config import Sleep2WaveConfig
from sleep2wave.visualization.downstream_eval_plots import render_waveform_example_plot


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
            latent_frames_per_epoch=config.autoencoder.latent_frames_per_epoch,
            channel_specific=config.autoencoder.channel_specific,
            modalities=config.modalities.all,
        )
        self.loss_fn = Sleep2WaveAutoencoderLoss(config.autoencoder.losses)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["sleep2wave_config"] = asdict(self.config_bundle)
        checkpoint["sleep2wave_config_yaml"] = yaml.safe_dump(checkpoint["sleep2wave_config"], sort_keys=True)

    def _compute_losses(self, batch):
        output = self.model(batch["clean_signals"])
        losses = self.loss_fn(
            output.reconstructions,
            batch["clean_signals"],
            availability_mask=batch.get("availability_mask"),
            quality_mask=batch.get("quality_mask"),
            channel_mask=batch.get("channel_mask"),
        )
        return output, losses

    def _log_losses(self, losses, batch_size: int, *, stage: str, on_step: bool) -> None:
        for name, value in losses.items():
            self.log(
                f"{stage}_{name}",
                value,
                prog_bar=(name == "loss"),
                on_step=on_step,
                on_epoch=True,
                batch_size=batch_size,
            )

    def training_step(self, batch, batch_idx):
        _output, losses = self._compute_losses(batch)
        batch_size = next(iter(batch["clean_signals"].values())).shape[0]
        self._log_losses(losses, batch_size, stage="train", on_step=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        output, losses = self._compute_losses(batch)
        batch_size = next(iter(batch["clean_signals"].values())).shape[0]
        self._log_losses(losses, batch_size, stage="val", on_step=False)
        self._log_validation_examples(batch, output.reconstructions, batch_idx)
        return losses

    def _select_example_channel(self, tensor: torch.Tensor, channel_mask, sample_idx: int):
        if tensor.dim() == 3:
            return tensor[sample_idx]
        channel_idx = 0
        if channel_mask is not None:
            sample_mask = channel_mask[sample_idx]
            valid_channels = sample_mask.any(dim=0)
            valid_indices = valid_channels.nonzero(as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                return None
            channel_idx = int(valid_indices[0].item())
        return tensor[sample_idx, :, channel_idx, :]

    def _log_validation_examples(self, batch, reconstructions, batch_idx: int) -> None:
        if batch_idx != 0:
            return
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and not trainer.is_global_zero:
            return
        if getattr(wandb, "run", None) is None:
            return
        if self.config_bundle.training is None:
            return

        example_cfg = self.config_bundle.training.validation.examples
        batch_size = next(iter(batch["clean_signals"].values())).shape[0]
        example_count = min(example_cfg.num_examples, batch_size)
        metadata = batch.get("metadata", {})
        sample_ids = metadata.get("id", [])
        payload = {}
        for modality in example_cfg.modalities:
            clean = batch["clean_signals"][modality]
            reconstruction = reconstructions[modality]
            availability_mask = batch.get("availability_mask", {}).get(modality)
            channel_mask = batch.get("channel_mask", {}).get(modality)
            sample_rate_hz = self.config_bundle.modalities.sample_rates[modality]
            for sample_idx in range(example_count):
                if availability_mask is not None and not bool(availability_mask[sample_idx].any().item()):
                    continue
                clean_example = self._select_example_channel(clean, channel_mask, sample_idx)
                reconstruction_example = self._select_example_channel(reconstruction, channel_mask, sample_idx)
                if clean_example is None or reconstruction_example is None:
                    continue
                sample_id = sample_ids[sample_idx] if sample_idx < len(sample_ids) else sample_idx
                fig = render_waveform_example_plot(
                    clean_example.detach().cpu().numpy(),
                    reconstruction_example.detach().cpu().numpy(),
                    sample_rate_hz=sample_rate_hz,
                    title=f"Val Autoencoder {modality} sample {sample_id} (epoch {self.current_epoch})",
                    generated_label="Reconstruction",
                )
                payload[f"val_autoencoder_examples/{modality}_sample_{sample_idx}"] = wandb.Image(fig)
                plt.close(fig)
        if payload:
            wandb.log(payload, commit=False)

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
