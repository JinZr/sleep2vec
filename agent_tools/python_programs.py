"""Canonical loader for the embedded kernel programs run through ``python -c``.

Layer 0 leaf, used by ``transport`` and the launch, probe and plan-rendering modules.
``python_program_sources/*.py.src`` are fragments whose names resolve only once
concatenated, so they do not lint standalone and flake8's directory walk never
sees them. ``utils/complexity_check.py`` assembles every registered program
through ``source()`` and checks it at the same complexity ceiling with ``noqa``
disabled -- registering a program here is what puts it under that gate.
"""

from __future__ import annotations

from pathlib import Path

_SOURCE_ROOT = Path(__file__).with_name("python_program_sources")

_FRAGMENTS = {
    "process_start_token": "run_evidence/process_start_token.py.src",
    "process_group_running": "run_evidence/process_group_running.py.src",
    "runtime_guard": "runtime_lock.py.src",
}

_PROGRAMS = {
    "run_evidence.process_probe": (
        "run_evidence/_process_probe_header.py.src",
        _FRAGMENTS["process_start_token"],
        _FRAGMENTS["process_group_running"],
        "run_evidence/_process_probe_body.py.src",
    ),
    "managed_scheduler.runtime_identity": ("managed_scheduler/runtime_identity.py.src",),
    "managed_scheduler.process_launch": (
        "managed_scheduler/_process_launch_header.py.src",
        _FRAGMENTS["runtime_guard"],
        _FRAGMENTS["process_start_token"],
        "managed_scheduler/_process_launch_body.py.src",
    ),
    "managed_scheduler.cli_preflight": ("managed_scheduler/cli_preflight.py.src",),
    "plan_rendering.commit_status": ("plan_rendering/commit_status.py.src",),
    "plan_rendering.slurm_allocation_guard": ("plan_rendering/slurm_allocation_guard.py.src",),
    "plan_rendering.verify_input_snapshots": ("plan_rendering/verify_input_snapshots.py.src",),
    "slurm.worker_bootstrap": ("slurm/worker_bootstrap.py.src",),
}


def registered_programs() -> tuple[str, ...]:
    """Every program name ``source()`` accepts. The complexity check assembles
    each one, since flake8's directory walk cannot see ``.py.src`` fragments."""
    return tuple(_PROGRAMS)


def source(name: str) -> str:
    parts = _PROGRAMS[name]
    return "\n\n".join((_SOURCE_ROOT / part).read_bytes().decode("utf-8") for part in parts)
