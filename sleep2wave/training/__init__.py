import importlib

_PHASE_EXPORTS = {"DEFAULT_PHASE_TASK_MIX", "PhaseSchedule", "build_phase_schedule"}
_TASK_EXPORTS = {"Sleep2WaveTaskSampler"}


def __getattr__(name):
    if name in _PHASE_EXPORTS:
        module = importlib.import_module("sleep2wave.training.phase_schedule")
        return getattr(module, name)
    if name in _TASK_EXPORTS:
        module = importlib.import_module("sleep2wave.training.task_sampler")
        return getattr(module, name)
    raise AttributeError(name)


__all__ = [
    "DEFAULT_PHASE_TASK_MIX",
    "PhaseSchedule",
    "Sleep2WaveTaskSampler",
    "build_phase_schedule",
]
