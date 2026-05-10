import importlib

_TASK_EXPORTS = {
    "AUX_MODALITY",
    "TASK_TYPES",
    "GenerationTask",
    "build_generation_task",
    "validate_generation_task",
}
_MASK_EXPORTS = {
    "TaskAttentionMask",
    "TokenLayout",
    "build_directional_task_attention_mask",
}
_SCHEDULE_EXPORTS = {
    "DiffusionSchedule",
    "build_diffusion_schedule",
    "cosine_beta_schedule",
}
_MODEL_EXPORTS = {"Sleep2WaveDiffusionOutput", "Sleep2WaveDiffusionTransformer"}
_SAMPLER_EXPORTS = {"DDIMSampler", "DDPMSampler", "DiffusionSamplerOutput", "build_sampler"}
_LIGHTNING_EXPORTS = {"Sleep2WaveDiffusionLightning"}


def __getattr__(name):
    if name in _TASK_EXPORTS:
        module = importlib.import_module("sleep2wave.diffusion.tasks")
        return getattr(module, name)
    if name in _MASK_EXPORTS:
        module = importlib.import_module("sleep2wave.diffusion.task_masks")
        return getattr(module, name)
    if name in _SCHEDULE_EXPORTS:
        module = importlib.import_module("sleep2wave.diffusion.schedule")
        return getattr(module, name)
    if name in _MODEL_EXPORTS:
        module = importlib.import_module("sleep2wave.diffusion.model")
        return getattr(module, name)
    if name in _SAMPLER_EXPORTS:
        module = importlib.import_module("sleep2wave.diffusion.samplers")
        return getattr(module, name)
    if name in _LIGHTNING_EXPORTS:
        module = importlib.import_module("sleep2wave.diffusion.lightning")
        return getattr(module, name)
    raise AttributeError(name)


__all__ = [
    "AUX_MODALITY",
    "TASK_TYPES",
    "DiffusionSchedule",
    "GenerationTask",
    "DDIMSampler",
    "DDPMSampler",
    "DiffusionSamplerOutput",
    "Sleep2WaveDiffusionLightning",
    "Sleep2WaveDiffusionOutput",
    "Sleep2WaveDiffusionTransformer",
    "TaskAttentionMask",
    "TokenLayout",
    "build_sampler",
    "build_diffusion_schedule",
    "build_directional_task_attention_mask",
    "build_generation_task",
    "cosine_beta_schedule",
    "validate_generation_task",
]
