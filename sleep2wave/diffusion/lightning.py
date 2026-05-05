from __future__ import annotations

from dataclasses import asdict

import pytorch_lightning as pl
import torch
import yaml

from sleep2wave.autoencoders.checkpoints import load_sleep2wave_autoencoder_checkpoint
from sleep2wave.data.corruptions import apply_corruption
from sleep2wave.diffusion.losses import compute_diffusion_loss
from sleep2wave.diffusion.model import Sleep2WaveDiffusionTransformer
from sleep2wave.diffusion.schedule import DiffusionSchedule, build_diffusion_schedule
from sleep2wave.diffusion.tasks import GenerationTask, build_generation_task, is_restoration_task
from sleep2wave.generative.config import Sleep2WaveConfig
from sleep2wave.training.task_sampler import Sleep2WaveTaskSampler


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
            auxiliary_restoration_token=config.diffusion.auxiliary_restoration_token,
            replay_enabled=config.training.replay.enabled,
            seed=seed,
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
            sqrt_alpha = schedule.sqrt_alpha_bars[timesteps].view(batch_size, 1, 1)
            sqrt_one_minus = schedule.sqrt_one_minus_alpha_bars[timesteps].view(batch_size, 1, 1)
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
        return build_generation_task(
            task.task_type,
            condition_modalities=kept,
            target_modalities=task.target_modalities,
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
    ) -> dict[str, torch.Tensor]:
        training_cfg = self.config_bundle.training
        if training_cfg is None:
            return observed_signals
        policy = training_cfg.corruptions.for_task(task.task_type)
        if policy is None:
            return observed_signals

        updated = dict(observed_signals)
        for modality in task.condition_modalities:
            spec = policy.for_modality(modality)
            if spec is None:
                continue
            signal = observed_signals[modality]
            modality_offset = self.config_bundle.modalities.all.index(modality)
            corrupted, _mask = apply_corruption(
                spec.name,
                signal,
                seed=int(self.global_step) * len(self.config_bundle.modalities.all) + modality_offset,
                **self._corruption_kwargs(spec.name, signal, spec.kwargs),
            )
            updated[modality] = corrupted
        return updated

    def training_step(self, batch, batch_idx):
        task = self.task_sampler.sample(availability_mask=batch.get("availability_mask"))
        task = self._apply_condition_dropout(task)
        if self.autoencoder is None:
            clean_latents = batch["clean_latents"]
            observed_latents = batch.get("observed_latents", clean_latents)
        else:
            clean_latents = self._encode_signals(batch["clean_signals"])
            observed_signals = self._apply_task_corruption(batch["observed_signals"], task)
            observed_latents = self._encode_signals(observed_signals)
        condition_latents = {modality: observed_latents[modality] for modality in task.condition_modalities}
        noisy_targets, target_noise, timesteps = self._sample_noisy_targets(clean_latents, task)
        output = self.model(
            noisy_target_latents=noisy_targets,
            timesteps=timesteps,
            task=task,
            condition_latents=condition_latents,
            availability_mask=batch["availability_mask"],
            quality_mask=batch["quality_mask"],
            night_position=batch["night_position"],
        )
        losses = compute_diffusion_loss(
            output.predicted_noise,
            target_noise,
            task,
            target_mask=batch.get("availability_mask"),
            quality_mask=batch.get("quality_mask"),
        )
        batch_size = next(iter(batch["clean_signals"].values())).shape[0]
        self.log("train_loss", losses["loss"], prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log(f"train_task_{task.task_type}", torch.tensor(1.0, device=losses["loss"].device), on_step=True)
        for name, value in losses.items():
            if name != "loss":
                self.log(f"train_{name}", value, on_step=True, on_epoch=False, batch_size=batch_size)
        return losses["loss"]

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
        modalities=config.modalities.all,
        device="cpu",
    )


__all__ = ["Sleep2WaveDiffusionLightning"]
