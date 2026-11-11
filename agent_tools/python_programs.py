"""Canonical loader for the embedded programs run through ``python -c``.

Used by the managed launch, execution-snapshot, plan and experiment modules.
``python_program_sources/*.py.src`` are fragments whose names resolve only once
concatenated, so they do not lint standalone and flake8's directory walk never
sees them. The launch wrapper also embeds ``process_identity.py`` verbatim,
an ordinary module that ``run_evidence`` imports. ``utils/complexity_check.py`` assembles every registered program
through ``source()`` and checks it at the same complexity ceiling with ``noqa``
disabled -- registering a program here is what puts it under that gate.
"""

from __future__ import annotations

from pathlib import Path

_SOURCE_ROOT = Path(__file__).with_name("python_program_sources")

_FRAGMENTS = {
    "runtime_guard": "runtime_lock.py.src",
}

_PROGRAMS = {
    "managed_scheduler.runtime_identity": ("managed_scheduler/runtime_identity.py.src",),
    "managed_scheduler.process_launch": (
        "managed_scheduler/_process_launch_header.py.src",
        _FRAGMENTS["runtime_guard"],
        # The importable module itself, so the launch token and run_evidence's comparison share one source.
        "../process_identity.py",
        "managed_scheduler/_process_launch_body.py.src",
    ),
    "managed_scheduler.cli_preflight": ("managed_scheduler/cli_preflight.py.src",),
    "plan_rendering.commit_status": ("plan_rendering/commit_status.py.src",),
    "plan_rendering.verify_input_snapshots": ("plan_rendering/verify_input_snapshots.py.src",),
}


def registered_programs() -> tuple[str, ...]:
    """Every program name ``source()`` accepts. The complexity check assembles
    each one, since flake8's directory walk cannot see ``.py.src`` fragments."""
    return tuple(_PROGRAMS)


def source(name: str) -> str:
    parts = _PROGRAMS[name]
    return "\n\n".join((_SOURCE_ROOT / part).read_bytes().decode("utf-8") for part in parts)
