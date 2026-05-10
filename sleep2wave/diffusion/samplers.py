from __future__ import annotations

from dataclasses import dataclass
import typing as t

import torch

from sleep2wave.diffusion.model import Sleep2WaveDiffusionTransformer
from sleep2wave.diffusion.schedule import DiffusionSchedule, build_diffusion_schedule
from sleep2wave.diffusion.tasks import GenerationTask, validate_generation_task
from sleep2wave.generative.config import SamplerConfig


@dataclass(frozen=True)
class DiffusionSamplerOutput:
    generated_latents: dict[str, torch.Tensor]


class BaseDiffusionSampler:
    name = "base"

    def __init__(self, schedule: DiffusionSchedule, *, steps: int, num_samples: int = 1, eta: float = 0.0) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if steps > schedule.betas.numel():
            raise ValueError("steps must be <= diffusion schedule length.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if eta < 0:
            raise ValueError("eta must be non-negative.")
        self.schedule = schedule
        self.steps = int(steps)
        self.num_samples = int(num_samples)
        self.eta = float(eta)

    def _timesteps(self, device: torch.device) -> torch.Tensor:
        total_steps = self.schedule.betas.numel()
        values = torch.linspace(total_steps - 1, 0, self.steps, device=device)
        return values.round().to(dtype=torch.long)

    def _infer_shape(
        self,
        model: Sleep2WaveDiffusionTransformer,
        condition_latents: dict[str, torch.Tensor],
    ) -> tuple[int, torch.device]:
        if not condition_latents:
            raise ValueError("condition_latents must be non-empty.")
        batch_size: int | None = None
        device: torch.device | None = None
        first = next(iter(condition_latents.values()))
        device = first.device
        for modality, latent in condition_latents.items():
            current_batch_size, _context_epochs, _channels = model._validate_latent(latent, modality)
            if batch_size is None:
                batch_size = current_batch_size
            elif current_batch_size != batch_size:
                raise ValueError(f"Condition latent for '{modality}' has batch size {current_batch_size}.")
        if batch_size is None:
            raise ValueError("condition_latents must be non-empty.")
        return batch_size, device

    def _target_channel_count(
        self,
        model: Sleep2WaveDiffusionTransformer,
        channel_mask: dict[str, torch.Tensor] | None,
        modality: str,
        *,
        batch_size: int,
        device: torch.device,
    ) -> int:
        if channel_mask is None or modality not in channel_mask:
            return 1
        mask = torch.as_tensor(channel_mask[modality], dtype=torch.bool, device=device)
        expected_prefix = (batch_size, model.layout.context_epochs)
        if mask.dim() != 3 or mask.shape[:2] != expected_prefix:
            raise ValueError(f"channel_mask['{modality}'] must have shape {expected_prefix + ('C',)}.")
        if mask.shape[2] <= 0:
            raise ValueError(f"channel_mask['{modality}'] must include at least one channel.")
        return int(mask.shape[2])

    def _initial_targets(
        self,
        model: Sleep2WaveDiffusionTransformer,
        task: GenerationTask,
        *,
        batch_size: int,
        device: torch.device,
        channel_mask: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            modality: torch.randn(
                batch_size,
                model.layout.context_epochs,
                self._target_channel_count(model, channel_mask, modality, batch_size=batch_size, device=device),
                model._latent_frames_for_modality(modality),
                model.latent_dim,
                device=device,
            )
            for modality in task.target_modalities
        }

    def sample(
        self,
        model: Sleep2WaveDiffusionTransformer,
        *,
        condition_latents: dict[str, torch.Tensor],
        task: GenerationTask,
        availability_mask: dict[str, torch.Tensor],
        quality_mask: dict[str, torch.Tensor],
        night_position: torch.Tensor,
        condition_availability_mask: dict[str, torch.Tensor] | None = None,
        channel_mask: dict[str, torch.Tensor] | None = None,
    ) -> DiffusionSamplerOutput:
        raise NotImplementedError


class DDPMSampler(BaseDiffusionSampler):
    name = "ddpm"

    def __init__(self, schedule: DiffusionSchedule, *, steps: int, num_samples: int = 1, eta: float = 0.0) -> None:
        super().__init__(schedule, steps=steps, num_samples=num_samples, eta=eta)
        if self.steps != schedule.betas.numel():
            raise ValueError("DDPM sampling requires steps to equal the diffusion schedule length.")

    @torch.no_grad()
    def sample(
        self,
        model: Sleep2WaveDiffusionTransformer,
        *,
        condition_latents: dict[str, torch.Tensor],
        task: GenerationTask,
        availability_mask: dict[str, torch.Tensor],
        quality_mask: dict[str, torch.Tensor],
        night_position: torch.Tensor,
        condition_availability_mask: dict[str, torch.Tensor] | None = None,
        channel_mask: dict[str, torch.Tensor] | None = None,
    ) -> DiffusionSamplerOutput:
        task = validate_generation_task(task)
        batch_size, device = self._infer_shape(model, condition_latents)
        schedule = _schedule_to(self.schedule, device)
        collected = {modality: [] for modality in task.target_modalities}
        timesteps = self._timesteps(device)
        for _sample_idx in range(self.num_samples):
            current = self._initial_targets(
                model,
                task,
                batch_size=batch_size,
                device=device,
                channel_mask=channel_mask,
            )
            for timestep in timesteps:
                t_batch = torch.full((batch_size,), int(timestep.item()), dtype=torch.long, device=device)
                predicted = model(
                    noisy_target_latents=current,
                    timesteps=t_batch,
                    task=task,
                    condition_latents=condition_latents,
                    availability_mask=availability_mask,
                    condition_availability_mask=condition_availability_mask,
                    channel_mask=channel_mask,
                    quality_mask=quality_mask,
                    night_position=night_position,
                ).predicted_noise
                for modality in task.target_modalities:
                    beta_t = schedule.betas[timestep]
                    alpha_t = schedule.alphas[timestep]
                    alpha_bar_t = schedule.alpha_bars[timestep]
                    scaled_noise = beta_t / torch.sqrt(1.0 - alpha_bar_t) * predicted[modality]
                    mean = (current[modality] - scaled_noise) / torch.sqrt(alpha_t)
                    if timestep.item() > 0:
                        current[modality] = mean + torch.sqrt(beta_t) * torch.randn_like(current[modality])
                    else:
                        current[modality] = mean
            for modality in task.target_modalities:
                collected[modality].append(current[modality])
        return DiffusionSamplerOutput(
            generated_latents={modality: torch.stack(values, dim=0) for modality, values in collected.items()}
        )


class DDIMSampler(BaseDiffusionSampler):
    name = "ddim"

    @torch.no_grad()
    def sample(
        self,
        model: Sleep2WaveDiffusionTransformer,
        *,
        condition_latents: dict[str, torch.Tensor],
        task: GenerationTask,
        availability_mask: dict[str, torch.Tensor],
        quality_mask: dict[str, torch.Tensor],
        night_position: torch.Tensor,
        condition_availability_mask: dict[str, torch.Tensor] | None = None,
        channel_mask: dict[str, torch.Tensor] | None = None,
    ) -> DiffusionSamplerOutput:
        task = validate_generation_task(task)
        batch_size, device = self._infer_shape(model, condition_latents)
        schedule = _schedule_to(self.schedule, device)
        collected = {modality: [] for modality in task.target_modalities}
        timesteps = self._timesteps(device)
        prev_timesteps = torch.cat([timesteps[1:], torch.tensor([-1], device=device, dtype=torch.long)])
        for _sample_idx in range(self.num_samples):
            current = self._initial_targets(
                model,
                task,
                batch_size=batch_size,
                device=device,
                channel_mask=channel_mask,
            )
            for timestep, prev_timestep in zip(timesteps, prev_timesteps):
                t_batch = torch.full((batch_size,), int(timestep.item()), dtype=torch.long, device=device)
                predicted = model(
                    noisy_target_latents=current,
                    timesteps=t_batch,
                    task=task,
                    condition_latents=condition_latents,
                    availability_mask=availability_mask,
                    condition_availability_mask=condition_availability_mask,
                    channel_mask=channel_mask,
                    quality_mask=quality_mask,
                    night_position=night_position,
                ).predicted_noise
                alpha_bar_t = schedule.alpha_bars[timestep]
                if prev_timestep.item() >= 0:
                    alpha_bar_prev = schedule.alpha_bars[prev_timestep]
                else:
                    alpha_bar_prev = torch.tensor(1.0, dtype=alpha_bar_t.dtype, device=device)
                for modality in task.target_modalities:
                    pred_x0 = (current[modality] - torch.sqrt(1.0 - alpha_bar_t) * predicted[modality]) / torch.sqrt(
                        alpha_bar_t
                    )
                    sigma = self.eta * torch.sqrt(
                        torch.clamp(
                            (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_prev),
                            min=0.0,
                        )
                    )
                    direction_scale = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0))
                    current[modality] = torch.sqrt(alpha_bar_prev) * pred_x0 + direction_scale * predicted[modality]
                    if prev_timestep.item() >= 0 and self.eta > 0:
                        current[modality] = current[modality] + sigma * torch.randn_like(current[modality])
            for modality in task.target_modalities:
                collected[modality].append(current[modality])
        return DiffusionSamplerOutput(
            generated_latents={modality: torch.stack(values, dim=0) for modality, values in collected.items()}
        )


def _schedule_to(schedule: DiffusionSchedule, device: torch.device) -> DiffusionSchedule:
    return DiffusionSchedule(
        betas=schedule.betas.to(device),
        alphas=schedule.alphas.to(device),
        alpha_bars=schedule.alpha_bars.to(device),
        sqrt_alpha_bars=schedule.sqrt_alpha_bars.to(device),
        sqrt_one_minus_alpha_bars=schedule.sqrt_one_minus_alpha_bars.to(device),
    )


def build_sampler(config: SamplerConfig, *, diffusion_steps: int, beta_schedule: str) -> BaseDiffusionSampler:
    schedule = build_diffusion_schedule(diffusion_steps, beta_schedule)
    kwargs: dict[str, t.Any] = {"steps": config.steps, "num_samples": config.num_samples, "eta": config.eta}
    if config.name == "ddpm":
        return DDPMSampler(schedule, **kwargs)
    if config.name == "ddim":
        return DDIMSampler(schedule, **kwargs)
    raise ValueError("sampler.name must be 'ddim' or 'ddpm'.")


__all__ = [
    "BaseDiffusionSampler",
    "DDIMSampler",
    "DDPMSampler",
    "DiffusionSamplerOutput",
    "build_sampler",
]
