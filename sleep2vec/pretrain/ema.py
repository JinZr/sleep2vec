import copy
import math

import torch
import torch.nn as nn


def clone_ema_model(student: nn.Module) -> nn.Module:
    """Return a frozen copy of the student model for EMA tracking."""
    ema = copy.deepcopy(student)
    for p in ema.parameters():
        p.requires_grad = False
    return ema


def ema_update(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    """In-place EMA update: teacher = m * teacher + (1 - m) * student."""
    if teacher is None or student is None:
        return
    with torch.no_grad():
        student_params = dict(student.named_parameters())
        for name, p_teacher in teacher.named_parameters():
            p_student = student_params.get(name)
            if p_student is None:
                continue
            p_teacher.data.mul_(momentum).add_(p_student.data, alpha=1.0 - momentum)


def cosine_ema_momentum(step: int, total_steps: int, base_momentum: float, final_momentum: float = 1.0) -> float:
    """
    Cosine schedule for EMA momentum (DINOv2 style).
    When step=0 -> base_momentum; when step>=total_steps -> final_momentum.
    """
    if total_steps <= 0 or step >= total_steps:
        return final_momentum
    t = step / float(total_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return final_momentum - (final_momentum - base_momentum) * cosine
