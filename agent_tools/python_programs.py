from __future__ import annotations

from pathlib import Path

_SOURCE_ROOT = Path(__file__).with_name("python_program_sources")

_FRAGMENTS = {
    "managed_descriptors": "experiment_io/_managed_descriptors.py.src",
    "process_start_token": "run_evidence/process_start_token.py.src",
    "process_group_running": "run_evidence/process_group_running.py.src",
}

_PROGRAMS = {
    "experiment_io.remote_dir_nonempty": ("experiment_io/remote_dir_nonempty.py.src",),
    "experiment_io.path_exists": ("experiment_io/path_exists.py.src",),
    "experiment_io.list_managed_subdirectories": ("experiment_io/list_managed_subdirectories.py.src",),
    "experiment_io.read_managed_files": ("experiment_io/read_managed_files.py.src",),
    "experiment_io.read_managed_output_texts": (
        _FRAGMENTS["managed_descriptors"],
        "experiment_io/read_managed_output_texts.py.src",
    ),
    "experiment_io.validate_managed_output_paths": ("experiment_io/validate_managed_output_paths.py.src",),
    "experiment_io.read_text": ("experiment_io/read_text.py.src",),
    "experiment_io.conditional_atomic_replace_text": (
        _FRAGMENTS["managed_descriptors"],
        "experiment_io/conditional_atomic_replace_text.py.src",
    ),
    "experiment_workspace.write_run_matrix_if_current": ("experiment_workspace/write_run_matrix_if_current.py.src",),
    "run_evidence.runtime_artifacts": ("run_evidence/runtime_artifacts.py.src",),
    "run_evidence.checkpoint_file_sha256": ("run_evidence/checkpoint_file_sha256.py.src",),
    "run_evidence.read_pid_text": ("run_evidence/read_pid_text.py.src",),
    "run_evidence.process_probe": (
        "run_evidence/_process_probe_header.py.src",
        _FRAGMENTS["process_start_token"],
        _FRAGMENTS["process_group_running"],
        "run_evidence/_process_probe_body.py.src",
    ),
    "run_evidence.process_stop": (
        "run_evidence/_process_stop_header.py.src",
        _FRAGMENTS["process_start_token"],
        _FRAGMENTS["process_group_running"],
        "run_evidence/_process_stop_body.py.src",
    ),
    "run_evidence.log_tail": ("run_evidence/log_tail.py.src",),
    "run_evidence.log_tail_and_age": ("run_evidence/log_tail_and_age.py.src",),
    "managed_scheduler.runtime_identity": ("managed_scheduler/runtime_identity.py.src",),
    "managed_scheduler.process_launch": (
        "managed_scheduler/_process_launch_header.py.src",
        _FRAGMENTS["process_start_token"],
        "managed_scheduler/_process_launch_body.py.src",
    ),
    "managed_scheduler.cli_preflight": ("managed_scheduler/cli_preflight.py.src",),
    "plan_rendering.commit_status": ("plan_rendering/commit_status.py.src",),
    "plan_rendering.runtime_commit_guard": ("plan_rendering/runtime_commit_guard.py.src",),
    "plan_rendering.verify_input_snapshots": ("plan_rendering/verify_input_snapshots.py.src",),
    "hparam_postprocess.verify_checkpoint_sha256": ("hparam_postprocess/verify_checkpoint_sha256.py.src",),
}


def source(name: str) -> str:
    parts = _PROGRAMS[name]
    return "\n\n".join((_SOURCE_ROOT / part).read_bytes().decode("utf-8") for part in parts)
