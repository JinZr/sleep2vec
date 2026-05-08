from __future__ import annotations

from dataclasses import asdict

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
import yaml

from sleep2wave.autoencoders.checkpoints import load_sleep2wave_autoencoder_checkpoint
from sleep2wave.data.corruptions import apply_corruption
from sleep2wave.diffusion.losses import compute_diffusion_loss
from sleep2wave.diffusion.model import Sleep2WaveDiffusionTransformer
from sleep2wave.diffusion.samplers import build_sampler
from sleep2wave.diffusion.schedule import DiffusionSchedule, build_diffusion_schedule
from sleep2wave.diffusion.task_masks import build_patch_condition_availability
from sleep2wave.diffusion.tasks import GenerationTask, build_generation_task, is_restoration_task
from sleep2wave.generative.config import Sleep2WaveConfig
from sleep2wave.training.task_sampler import Sleep2WaveTaskSampler
from sleep2wave.visualization.downstream_eval_plots import render_waveform_example_plot


class Sleep2WaveDiffusionLightning(pl.LightningModule):
    def __init__(self, config: Sleep2WaveConfig, *, seed: int = 0) -> None:
        super().__init__()
        if config.diffusion is None:
            raise ValueError("sleep2wave diffusion training requires a diffusion config.")
        if config.training is None:
            raise ValueError("sleep2wave diffusion training requires a training config.")
        self.config_bundle = config
        self.model = Sleep2WaveDiffusionTransformer.from_config(config.diffusion, modalities=config.modalities.all)
        self.schedule = build_diffusion_schedule(config.diffusion.diffusion_steps, config.diffusion.beta_schedule)
        self.autoencoder = _load_autoencoder_for_diffusion(config)
        self.task_sampler = Sleep2WaveTaskSampler(
            modalities=config.modalities.all,
            phase=config.training.phase,
            task_mix=config.training.task_mix,
            condition_counts=config.training.condition_counts,
            restoration_condition_counts=config.training.restoration_condition_counts,
            auxiliary_restoration_token=config.diffusion.auxiliary_restoration_token,
            replay_enabled=config.training.replay.enabled,
            seed=seed,
        )
        self.validation_task_sampler = Sleep2WaveTaskSampler(
            modalities=config.modalities.all,
            phase=config.training.phase,
            task_mix=config.training.task_mix,
            condition_counts=config.training.condition_counts,
            restoration_condition_counts=config.training.restoration_condition_counts,
            auxiliary_restoration_token=config.diffusion.auxiliary_restoration_token,
            replay_enabled=config.training.replay.enabled,
            seed=seed + 1,
        )

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["sleep2wave_config"] = asdict(self.config_bundle)
        checkpoint["sleep2wave_config_yaml"] = yaml.safe_dump(checkpoint["sleep2wave_config"], sort_keys=True)

    def _schedule_to_device(self, device: torch.device) -> DiffusionSchedule:
        return DiffusionSchedule(
            betas=self.schedule.betas.to(device),
            alphas=self.schedule.alphas.to(device),
            alpha_bars=self.schedule.alpha_bars.to(device),
            sqrt_alpha_bars=self.schedule.sqrt_alpha_bars.to(device),
            sqrt_one_minus_alpha_bars=self.schedule.sqrt_one_minus_alpha_bars.to(device),
        )

    def _encode_signals(self, signals: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.autoencoder is None:
            raise ValueError("Signal encoding requires diffusion.autoencoder_checkpoint.")
        with torch.no_grad():
            return self.autoencoder(signals).latents

    def _sample_noisy_targets(
        self,
        clean_latents: dict[str, torch.Tensor],
        task: GenerationTask,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        first = clean_latents[task.target_modalities[0]]
        batch_size = first.shape[0]
        device = first.device
        schedule = self._schedule_to_device(device)
        timesteps = torch.randint(0, schedule.betas.numel(), (batch_size,), dtype=torch.long, device=device)
        noisy: dict[str, torch.Tensor] = {}
        target_noise: dict[str, torch.Tensor] = {}
        for modality in task.target_modalities:
            clean = clean_latents[modality]
            noise = torch.randn_like(clean)
            broadcast_shape = (batch_size,) + (1,) * (clean.dim() - 1)
            sqrt_alpha = schedule.sqrt_alpha_bars[timesteps].view(broadcast_shape)
            sqrt_one_minus = schedule.sqrt_one_minus_alpha_bars[timesteps].view(broadcast_shape)
            noisy[modality] = sqrt_alpha * clean + sqrt_one_minus * noise
            target_noise[modality] = noise
        return noisy, target_noise, timesteps

    def _apply_condition_dropout(self, task: GenerationTask) -> GenerationTask:
        diffusion_cfg = self.config_bundle.diffusion
        if diffusion_cfg is None or diffusion_cfg.condition_dropout <= 0.0:
            return task
        if is_restoration_task(task) or len(task.condition_modalities) <= 1:
            return task

        kept = [
            modality
            for modality in task.condition_modalities
            if torch.rand((), device=self.device).item() >= diffusion_cfg.condition_dropout
        ]
        if not kept:
            kept = [
                task.condition_modalities[
                    torch.randint(len(task.condition_modalities), (1,), device=self.device).item()
                ]
            ]
        target_modalities = task.target_modalities
        if task.task_type == "partial_full":
            original_modalities = [*task.condition_modalities, *task.target_modalities]
            target_modalities = [modality for modality in original_modalities if modality not in kept]
        return build_generation_task(
            task.task_type,
            condition_modalities=kept,
            target_modalities=target_modalities,
            auxiliary_restoration_token=task.use_auxiliary_token,
            allow_target_target_attention=task.allow_target_target_attention,
        )

    def _corruption_kwargs(self, name: str, signal: torch.Tensor, kwargs: dict) -> dict:
        parsed = dict(kwargs)
        if "window_frames" not in parsed and name in {
            "contiguous_window_mask",
            "flatline_dropout",
            "spo2_plateau_dropout",
            "rpeak_drop_or_jitter_for_ibi",
            "belt_failure",
        }:
            parsed["window_frames"] = max(1, signal.shape[-1] // 10)
        return parsed

    def _apply_task_corruption(
        self,
        observed_signals: dict[str, torch.Tensor],
        task: GenerationTask,
        *,
        return_masks: bool = False,
    ):
        training_cfg = self.config_bundle.training
        if training_cfg is None:
            return (observed_signals, {}) if return_masks else observed_signals
        policy = training_cfg.corruptions.for_task(task.task_type)
        if policy is None:
            return (observed_signals, {}) if return_masks else observed_signals

        updated = dict(observed_signals)
        masks: dict[str, torch.Tensor] = {}
        corruption_modalities = task.target_modalities if is_restoration_task(task) else task.condition_modalities
        for modality in corruption_modalities:
            spec = policy.for_modality(modality)
            if spec is None:
                continue
            signal = observed_signals[modality]
            modality_offset = self.config_bundle.modalities.all.index(modality)
            seed = int(self.global_step) * len(self.config_bundle.modalities.all) + modality_offset
            choice = spec.select(seed=seed)
            corrupted, _mask = apply_corruption(
                choice.name,
                signal,
                seed=seed,
                **self._corruption_kwargs(choice.name, signal, choice.kwargs),
            )
            updated[modality] = corrupted
            masks[modality] = _mask
        return (updated, masks) if return_masks else updated

    def _patch_condition_availability(
        self,
        availability_mask: dict[str, torch.Tensor],
        corruption_mask: dict[str, torch.Tensor],
        task: GenerationTask,
    ) -> dict[str, torch.Tensor]:
        diffusion_cfg = self.config_bundle.diffusion
        if diffusion_cfg is None or not corruption_mask:
            return availability_mask
        return build_patch_condition_availability(
            availability_mask,
            corruption_mask,
            task,
            patches_per_epoch=diffusion_cfg.patches_per_epoch,
        )

    def _validation_task_for_batch(self, batch, batch_idx: int):
        validation_cfg = self.config_bundle.training.validation
        modalities = tuple(validation_cfg.examples.modalities)
        families = self._active_task_families()
        if not modalities or not families:
            return None

        modality_span = len(modalities) * validation_cfg.max_batches_per_modality
        family = families[(batch_idx // modality_span) % len(families)]
        modality_index = (batch_idx % modality_span) // validation_cfg.max_batches_per_modality
        for offset in range(len(modalities)):
            modality = modalities[(modality_index + offset) % len(modalities)]
            try:
                task = self.validation_task_sampler.sample_family(
                    family,
                    batch.get("availability_mask"),
                    target_modalities=[modality],
                )
            except ValueError:
                continue
            return family, task
        return None

    def _compute_step_losses(self, batch, *, apply_condition_dropout: bool, batch_idx: int | None = None):
        sampler = self.task_sampler if apply_condition_dropout else self.validation_task_sampler
        if apply_condition_dropout:
            sampled = sampler.sample_with_family(availability_mask=batch.get("availability_mask"))
            task_family = sampled.task_family
            task = sampled.task
        else:
            selected = self._validation_task_for_batch(batch, 0 if batch_idx is None else batch_idx)
            if selected is None:
                return None
            task_family, task = selected
        if apply_condition_dropout:
            task = self._apply_condition_dropout(task)
        if self.autoencoder is None:
            clean_latents = batch["clean_latents"]
            observed_latents = batch.get("observed_latents", clean_latents)
            condition_availability = None
        else:
            clean_latents = self._encode_signals(batch["clean_signals"])
            observed_signals, corruption_mask = self._apply_task_corruption(
                batch["observed_signals"],
                task,
                return_masks=True,
            )
            observed_latents = self._encode_signals(observed_signals)
            condition_availability = self._patch_condition_availability(
                batch["availability_mask"],
                corruption_mask,
                task,
            )
        condition_latents = {modality: observed_latents[modality] for modality in task.condition_modalities}
        noisy_targets, target_noise, timesteps = self._sample_noisy_targets(clean_latents, task)
        output = self.model(
            noisy_target_latents=noisy_targets,
            timesteps=timesteps,
            task=task,
            condition_latents=condition_latents,
            availability_mask=batch["availability_mask"],
            condition_availability_mask=condition_availability if self.autoencoder is not None else None,
            channel_mask=batch.get("channel_mask"),
            quality_mask=batch["quality_mask"],
            night_position=batch["night_position"],
        )
        losses = compute_diffusion_loss(
            output.predicted_noise,
            target_noise,
            task,
            target_mask=batch.get("availability_mask"),
            channel_mask=batch.get("channel_mask"),
            quality_mask=batch.get("quality_mask"),
        )
        return losses, task_family, task

    def training_step(self, batch, batch_idx):
        losses, task_family, _task = self._compute_step_losses(batch, apply_condition_dropout=True)
        batch_size = next(iter(batch["clean_signals"].values())).shape[0]
        self.log("train_loss", losses["loss"], prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log(f"train_task_{task_family}", torch.tensor(1.0, device=losses["loss"].device), on_step=True)
        for name, value in losses.items():
            if name != "loss":
                self.log(f"train_{name}", value, on_step=True, on_epoch=False, batch_size=batch_size)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        step_losses = self._compute_step_losses(batch, apply_condition_dropout=False, batch_idx=batch_idx)
        if step_losses is None:
            return None
        losses, task_family, _task = step_losses
        batch_size = next(iter(batch["clean_signals"].values())).shape[0]
        self.log("val_loss", losses["loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(
            f"val_task_{task_family}",
            torch.tensor(1.0, device=losses["loss"].device),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        for name, value in losses.items():
            if name != "loss":
                self.log(f"val_{name}", value, on_step=False, on_epoch=True, batch_size=batch_size)
        self._log_validation_examples(batch, batch_idx)
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

    def _active_task_families(self) -> list[str]:
        return [family for family, weight in self.validation_task_sampler.schedule.task_mix.items() if weight > 0]

    def _decode_first_generated_sample(self, generated_latents: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        first_sample = {modality: values[0] for modality, values in generated_latents.items()}
        if self.autoencoder is None:
            raise ValueError("Signal decoding requires diffusion.autoencoder_checkpoint.")
        return self.autoencoder.decode_latents(first_sample)

    def _log_validation_examples(self, batch, batch_idx: int) -> None:
        if batch_idx != 0:
            return
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and not trainer.is_global_zero:
            return
        if getattr(wandb, "run", None) is None:
            return
        if self.autoencoder is None:
            return
        diffusion_cfg = self.config_bundle.diffusion
        sampler_cfg = self.config_bundle.sampler
        training_cfg = self.config_bundle.training
        if diffusion_cfg is None or sampler_cfg is None or training_cfg is None:
            return

        example_cfg = training_cfg.validation.examples
        batch_size = next(iter(batch["clean_signals"].values())).shape[0]
        example_count = min(example_cfg.num_examples, batch_size)
        metadata = batch.get("metadata", {})
        sample_ids = metadata.get("id", [])
        sampler = build_sampler(
            sampler_cfg,
            diffusion_steps=diffusion_cfg.diffusion_steps,
            beta_schedule=diffusion_cfg.beta_schedule,
        )
        payload = {}
        with torch.no_grad():
            for task_family in self._active_task_families():
                try:
                    task = self.validation_task_sampler.sample_family(
                        task_family,
                        batch.get("availability_mask"),
                        target_modalities=example_cfg.modalities,
                    )
                except ValueError:
                    continue
                observed_signals, corruption_mask = self._apply_task_corruption(
                    batch["observed_signals"],
                    task,
                    return_masks=True,
                )
                observed_latents = self._encode_signals(observed_signals)
                condition_latents = {modality: observed_latents[modality] for modality in task.condition_modalities}
                condition_availability = self._patch_condition_availability(
                    batch["availability_mask"],
                    corruption_mask,
                    task,
                )
                output = sampler.sample(
                    self.model,
                    condition_latents=condition_latents,
                    task=task,
                    availability_mask=batch["availability_mask"],
                    quality_mask=batch["quality_mask"],
                    night_position=batch["night_position"],
                    condition_availability_mask=condition_availability,
                    channel_mask=batch.get("channel_mask"),
                )
                decoded = self._decode_first_generated_sample(output.generated_latents)
                target_modality = next(
                    modality for modality in example_cfg.modalities if modality in task.target_modalities
                )
                clean_signal = batch["clean_signals"][target_modality]
                generated_signal = decoded[target_modality]
                observed_signal = (
                    observed_signals[target_modality] if target_modality in task.condition_modalities else None
                )
                channel_mask = batch.get("channel_mask", {}).get(target_modality)
                sample_rate_hz = self.config_bundle.modalities.sample_rates[target_modality]
                for sample_idx in range(example_count):
                    clean_example = self._select_example_channel(clean_signal, channel_mask, sample_idx)
                    generated_example = self._select_example_channel(generated_signal, channel_mask, sample_idx)
                    if clean_example is None or generated_example is None:
                        continue
                    observed_example = None
                    if observed_signal is not None:
                        observed_example = self._select_example_channel(observed_signal, channel_mask, sample_idx)
                    sample_id = sample_ids[sample_idx] if sample_idx < len(sample_ids) else sample_idx
                    fig = render_waveform_example_plot(
                        clean_example.detach().cpu().numpy(),
                        generated_example.detach().cpu().numpy(),
                        observed=(observed_example.detach().cpu().numpy() if observed_example is not None else None),
                        sample_rate_hz=sample_rate_hz,
                        title=(
                            f"Val Diffusion {task_family} {target_modality} "
                            f"sample {sample_id} (epoch {self.current_epoch})"
                        ),
                        generated_label="Generated",
                    )
                    payload[f"val_diffusion_examples/{task_family}/{target_modality}_sample_{sample_idx}"] = (
                        wandb.Image(fig)
                    )
                    plt.close(fig)
        if payload:
            wandb.log(payload, commit=False)

    def configure_optimizers(self):
        training_cfg = self.config_bundle.training
        if training_cfg is None:
            raise ValueError("sleep2wave diffusion training requires a training config.")
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


def _load_autoencoder_for_diffusion(config: Sleep2WaveConfig):
    if config.diffusion is None:
        raise ValueError("diffusion config is required.")
    checkpoint_path = config.diffusion.autoencoder_checkpoint
    if checkpoint_path is None:
        return None
    return load_sleep2wave_autoencoder_checkpoint(
        checkpoint_path,
        latent_dim=config.diffusion.latent_dim,
        latent_frames_per_epoch=config.diffusion.latent_frames_per_epoch,
        modalities=config.modalities.all,
        device="cpu",
    )


__all__ = ["Sleep2WaveDiffusionLightning"]
