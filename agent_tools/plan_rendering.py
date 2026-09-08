"""Rendering of recipe fields into CLI argv and launch script text.

Mixed bridge: ``preset_cli_args`` spells sleep preset fields, and one
``sex_age_baseline`` branch survives here. Owns the shared runtime, scheduler,
and input field lists together with the common option-appending helpers; task
adapters retain ownership of task-specific flags before calling these helpers.
"""

from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from . import python_programs
from .experiment_workspace import MONITOR_EXIT_CODE_PREFIX
from .models import REPO_ROOT, coerce_list, module_for_variant

DEFAULT_FINETUNE_LR = 1e-6
DEFAULT_FINETUNE_WEIGHT_DECAY = 1e-5

_FINETUNE_RUNTIME_DEFAULTS = (
    ("precision", "--precision", "bf16-mixed"),
    ("epochs", "--epochs", 30),
    ("batch_size", "--batch-size", 12),
    ("num_workers", "--num-workers", 8),
    ("lr", "--lr", DEFAULT_FINETUNE_LR),
    ("weight_decay", "--weight-decay", DEFAULT_FINETUNE_WEIGHT_DECAY),
)
FINETUNE_SCHEDULER_FIELDS = frozenset(
    {"lr_scheduler", "lr_decay_shape", "lr_decay_floor", "lr_decay_ratio", "lr_plateau_factor", "lr_plateau_patience"}
)
_FINETUNE_RUNTIME_OPTIONS = (
    ("lr_scheduler", "--lr-scheduler"),
    ("lr_decay_shape", "--lr-decay-shape"),
    ("lr_decay_floor", "--lr-decay-floor"),
    ("lr_decay_ratio", "--lr-decay-ratio"),
    ("lr_plateau_factor", "--lr-plateau-factor"),
    ("lr_plateau_patience", "--lr-plateau-patience"),
    ("device", "--device"),
    ("warmup_steps", "--warmup-steps"),
    ("gradient_clip_val", "--gradient-clip-val"),
    ("accumulate_grad_batches", "--accumulate-grad-batches"),
    ("patience", "--patience"),
    ("check_val_every_n_epoch", "--check-val-every-n-epoch"),
    ("ckpt_every_n_epochs", "--ckpt-every-n-epochs"),
)
FINETUNE_RUNTIME_FIELDS = frozenset(
    {"devices", "wandb_mode", *(key for key, _flag, _default in _FINETUNE_RUNTIME_DEFAULTS)}
    | {key for key, _flag in _FINETUNE_RUNTIME_OPTIONS}
)

_INFER_RUNTIME_DEFAULTS = (
    ("precision", "--precision", "bf16-mixed"),
    ("batch_size", "--batch-size", 12),
    ("num_workers", "--num-workers", 8),
    ("lr", "--lr", 1e-6),
    ("weight_decay", "--weight-decay", 1e-5),
)
_INFER_RUNTIME_OPTIONS = (
    ("accelerator", "--accelerator"),
    ("device", "--device"),
    ("avg_ckpts", "--avg-ckpts"),
    ("avg_ckpt_dir", "--avg-ckpt-dir"),
    ("results_root", "--results-root"),
    ("seed", "--seed"),
    ("wandb_mode", "--wandb-mode"),
)
INFER_RUNTIME_FIELDS = frozenset(
    {"devices", *(key for key, _flag, _default in _INFER_RUNTIME_DEFAULTS)}
    | {key for key, _flag in _INFER_RUNTIME_OPTIONS}
)

PRESET_FIELDS = frozenset(
    {
        "allow_missing_channels",
        "batch_size",
        "channels",
        "dry_run",
        "include_no_metadata",
        "include_overlap_eval_splits",
        "manifest_output",
        "mask_rate",
        "meta_data_names",
        "min_channels",
        "n_tokens",
        "num_workers",
        "output_template",
        "overwrite",
        "shuffle",
        "split",
        "stride_tokens",
        "write_sidecar_manifest",
    }
)


if TYPE_CHECKING:
    from .plan_contract import FrozenInputSnapshot


def variant_module(recipe: dict, entrypoint: str) -> str:
    return module_for_variant(str(recipe.get("variant")), entrypoint)


def render_command(parts: list[Any]) -> str:
    missing = [idx for idx, part in enumerate(parts) if part is None]
    if missing:
        raise ValueError(f"Cannot render command with missing token(s) at positions: {missing}")
    return " ".join(shlex.quote(str(part)) for part in parts)


def append_option(args: list[Any], flag: str, value: Any) -> None:
    if value in (None, "", "ASK_USER"):
        return
    args.extend([flag, value])


def append_list_option(args: list[Any], flag: str, values: Any) -> None:
    if values in (None, "", "ASK_USER"):
        return
    if isinstance(values, (list, tuple)):
        if not values:
            return
        args.extend([flag, *values])
    else:
        args.extend([flag, values])


def append_bool_option(args: list[Any], value: Any, true_flag: str, false_flag: str | None = None) -> None:
    if value is True:
        args.append(true_flag)
    elif value is False and false_flag:
        args.append(false_flag)


def loads_train_val(epochs: Any) -> bool:
    try:
        return int(epochs) > 0
    except (TypeError, ValueError):
        return True


def finetune_loaded_split_values(recipe: dict, *, load_test: bool | None = None) -> list[str]:
    raw_runtime = recipe.get("runtime")
    runtime = raw_runtime if isinstance(raw_runtime, dict) else {}
    raw_evaluation = recipe.get("evaluation_policy")
    evaluation = raw_evaluation if isinstance(raw_evaluation, dict) else {}

    splits: list[str] = []
    if loads_train_val(runtime.get("epochs", 30)):
        splits.extend(["train", "val"])

    if load_test is None:
        load_test = evaluation.get("test_after_fit")
    if load_test is True:
        splits.append("test")
    return splits


def apply_finetune_task_flags(args: SimpleNamespace, recipe: dict[str, Any], task: dict[str, Any]) -> None:
    args.label_name = recipe.get("inputs", {}).get("label_name")
    config_module = import_module(variant_module(recipe, "config"))
    common_module = import_module(variant_module(recipe, "common"))
    task_config = config_module.TaskConfig(**task) if task.get("type") else None
    common_module.apply_task_flags(args, task_config)


def validate_finetune_runtime(recipe: dict[str, Any], runtime: dict[str, Any], task: dict[str, Any]) -> None:
    if not FINETUNE_SCHEDULER_FIELDS.intersection(runtime):
        return
    if recipe.get("variant") == "sex_age_baseline" and (
        FINETUNE_SCHEDULER_FIELDS - {"lr_decay_shape", "lr_decay_floor"}
    ).intersection(runtime):
        raise ValueError("Scheduler selection, WSD and Plateau fields are not supported by sex_age_baseline.")
    args = SimpleNamespace(**{key: value for key, value in runtime.items() if value is not None})
    if getattr(args, "lr_scheduler", "decay") == "plateau":
        apply_finetune_task_flags(args, recipe, task)
    scheduler_module = import_module(
        "sleep2vec.schedulers" if recipe.get("variant") == "sex_age_baseline" else variant_module(recipe, "schedulers")
    )
    scheduler_module.validate_finetune_scheduler_args(args)


def runtime_cli_args(runtime: dict[str, Any], *, variant: str | None = None) -> list[Any]:
    args: list[Any] = [
        "--devices",
        *[str(item) for item in coerce_list(runtime.get("devices", [0])) or [0]],
    ]
    for key, flag, default in _FINETUNE_RUNTIME_DEFAULTS:
        args.extend([flag, runtime.get(key, default)])
    for key, flag in _FINETUNE_RUNTIME_OPTIONS:
        append_option(args, flag, runtime.get(key))
    append_option(args, "--wandb-mode", runtime.get("wandb_mode"))
    return args


def infer_runtime_cli_args(runtime: dict[str, Any]) -> list[Any]:
    args: list[Any] = [
        "--devices",
        *[str(item) for item in coerce_list(runtime.get("devices", [0])) or [0]],
    ]
    for key, flag, default in _INFER_RUNTIME_DEFAULTS:
        args.extend([flag, runtime.get(key, default)])
    for key, flag in _INFER_RUNTIME_OPTIONS:
        append_option(args, flag, runtime.get(key))
    return args


def finetune_input_cli_args(
    inputs: dict[str, Any],
    *,
    variant: str | None = None,
) -> list[Any]:
    args: list[Any] = []
    if variant != "sex_age_baseline":
        append_option(args, "--pretrained-backbone-path", inputs.get("pretrained_backbone_path"))
    append_option(args, "--ckpt-path", inputs.get("ckpt_path"))
    return args


def infer_input_cli_args(inputs: dict[str, Any], *, variant: str | None = None) -> list[Any]:
    args: list[Any] = []
    if variant != "sex_age_baseline":
        append_option(args, "--pretrained-backbone-path", inputs.get("pretrained_backbone_path"))
    append_option(args, "--inference-preset-path", inputs.get("inference_preset_path"))
    if variant != "sex_age_baseline":
        append_list_option(args, "--override-dataset-names", inputs.get("override_dataset_names"))
    return args


def preset_cli_args(preset: dict[str, Any]) -> list[Any]:
    args: list[Any] = []
    append_option(args, "--output-template", preset.get("output_template"))
    append_option(args, "--stride-tokens", preset.get("stride_tokens"))
    append_bool_option(args, preset.get("include_overlap_eval_splits"), "--include-overlap-eval-splits")
    append_list_option(args, "--meta-data-names", preset.get("meta_data_names"))
    append_bool_option(args, preset.get("include_no_metadata"), "--include-no-metadata")
    append_list_option(args, "--channels", preset.get("channels"))
    append_option(args, "--batch-size", preset.get("batch_size"))
    append_bool_option(args, preset.get("shuffle"), "--shuffle", "--no-shuffle")
    append_option(args, "--mask-rate", preset.get("mask_rate"))
    append_bool_option(
        args,
        preset.get("allow_missing_channels"),
        "--allow-missing-channels",
        "--no-allow-missing-channels",
    )
    append_option(args, "--min-channels", preset.get("min_channels"))
    append_bool_option(args, preset.get("overwrite"), "--overwrite")
    append_option(args, "--num-workers", preset.get("num_workers"))
    append_bool_option(args, preset.get("dry_run"), "--dry-run")
    append_option(args, "--manifest-output", preset.get("manifest_output"))
    append_bool_option(
        args,
        preset.get("write_sidecar_manifest"),
        "--write-sidecar-manifest",
        "--no-write-sidecar-manifest",
    )
    return args


def script_lines(
    commands: list[str],
    *,
    run_cwd: str | Path | None = None,
    experiment_root: str | Path | None = None,
    step_id: str | None = None,
    run_id: str | None = None,
    lifecycle_python: str | Path | None = None,
    expected_runtime_commit: str | None = None,
    input_snapshots: list[FrozenInputSnapshot] | list[dict[str, str]] | None = None,
    slurm_allocation_guard: str | None = None,
) -> list[str]:
    cwd_lines = []
    if run_cwd is not None:
        root = shlex.quote(str(run_cwd))
        cwd_lines = [f"cd {root}", f"export PYTHONPATH={root}${{PYTHONPATH:+:$PYTHONPATH}}", ""]
    lifecycle_lines = []
    if experiment_root is not None:
        if lifecycle_python is None:
            raise ValueError("Lifecycle scripts require an explicit Python interpreter.")
        commit_command = render_command(
            [
                lifecycle_python,
                "-c",
                python_programs.source("plan_rendering.commit_status"),
                experiment_root,
                step_id,
                run_id,
                "__STATUS__",
            ]
        ) + (
            f" record-runtime-commit {shlex.quote(expected_runtime_commit)}"
            if expected_runtime_commit is not None
            else ""
        )
        prelaunch_verification_lines = []
        if input_snapshots:
            prelaunch_verification_lines.extend(
                [
                    render_command(
                        [
                            lifecycle_python,
                            "-c",
                            python_programs.source("plan_rendering.verify_input_snapshots"),
                            json.dumps(input_snapshots, sort_keys=True, separators=(",", ":")),
                        ]
                    ),
                    "",
                ]
            )
        lifecycle_lines = [
            "_agent_commit_status() {",
            f'  {commit_command.replace("__STATUS__", "$1")}',
            "}",
            "_agent_finish_run() {",
            "  _agent_runtime_status=$?",
            "  trap - EXIT",
            "  set +e",
            '  if [ "$_agent_runtime_status" -eq 0 ]; then',
            "    _agent_final_status=completed",
            "  else",
            "    _agent_final_status=failed",
            "  fi",
            '  _agent_commit_status "$_agent_final_status"',
            "  _agent_commit_status_code=$?",
            '  if [ "$_agent_runtime_status" -ne 0 ]; then',
            '    exit "$_agent_runtime_status"',
            "  fi",
            '  exit "$_agent_commit_status_code"',
            "}",
            "",
            *prelaunch_verification_lines,
            "_agent_commit_status running",
            # The trap is installed only after the owner accepts running, so terminal runs never execute again.
            "trap _agent_finish_run EXIT",
            "",
        ]
        if slurm_allocation_guard is not None:
            # The outer Slurm worker owns terminal evidence; never write canonical status here.
            lifecycle_lines = [
                slurm_allocation_guard,
                *prelaunch_verification_lines,
            ]
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        *cwd_lines,
        "# Agent policy status: PASS",
        "# This script was generated only after consultation gates passed.",
        "# High-impact decisions were resolved by explicit recipe/config/user inputs.",
        "",
        *lifecycle_lines,
        *commands,
    ]


def blocked_script() -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            'echo "This command plan is blocked because user input is required."',
            'echo "See questions.md."',
            "exit 2",
            "",
        ]
    )


def hparam_script_lines(
    commands: list[str],
    *,
    test_after_fit: bool = False,
    selection_split: str = "val",
    final_external_test: bool = False,
    record_exit_code: bool = False,
    run_cwd: str | Path = REPO_ROOT,
) -> list[str]:
    external_test_policy = "# - This script evaluates the configured final test split."
    final_test_policy = "# - Final test evaluation was explicitly unlocked."
    if not final_external_test:
        if test_after_fit and selection_split == "test":
            external_test_policy = (
                "# - Run commands evaluate every saved epoch checkpoint on the configured test split after fit."
            )
        elif test_after_fit:
            external_test_policy = "# - Run commands evaluate the configured test split after fit."
        else:
            external_test_policy = "# - Run commands do not evaluate the configured test split."
        final_test_policy = "# - Final test evaluation requires explicit unlock."
    root = shlex.quote(str(run_cwd))
    exit_code_lines = []
    if record_exit_code:
        exit_code_lines = [
            "_agent_tools_record_exit() {",
            '    local exit_code="$?"',
            "    trap - EXIT",
            '    if [[ -z "${SLURM_PROCID:-}" || "${SLURM_PROCID:-}" == "0" ]]; then',
            f"        printf '\\n{MONITOR_EXIT_CODE_PREFIX}%s\\n' \"$exit_code\" || :",
            "    fi",
            '    exit "$exit_code"',
            "}",
            "trap _agent_tools_record_exit EXIT",
            "",
        ]
    slurm_task_lines = [
        'if [[ -n "${SLURM_PROCID:-}" ]]; then',
        (
            "    printf 'AGENT_TOOLS_SLURM_TASK_START job_id=%s step_id=%s procid=%s localid=%s "
            "nodeid=%s ntasks=%s node=%s pid=%s cuda_visible_devices=%s\\n' "
            '"${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" "${SLURM_PROCID:-}" "${SLURM_LOCALID:-}" '
            '"${SLURM_NODEID:-}" "${SLURM_NTASKS:-}" "${SLURMD_NODENAME:-}" "$$" '
            '"${CUDA_VISIBLE_DEVICES:-}" || :'
        ),
        "fi",
        "",
    ]
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        *exit_code_lines,
        f"cd {root}",
        f"export PYTHONPATH={root}",
        "",
        *slurm_task_lines,
        "# Agent policy status: PASS",
        "# This script was generated only after consultation gates passed.",
        "# High-impact decisions were resolved by explicit recipe/config/user inputs.",
        "# External test policy:",
        external_test_policy,
        f"# - Candidate selection uses the frozen {selection_split} split metric.",
        final_test_policy,
        "",
        *commands,
    ]
