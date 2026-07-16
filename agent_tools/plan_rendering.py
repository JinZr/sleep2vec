from __future__ import annotations

from pathlib import Path
import shlex
import sys
from typing import Any

from .models import REPO_ROOT, coerce_list, module_for_variant

_FINETUNE_RUNTIME_DEFAULTS = (
    ("precision", "--precision", "bf16-mixed"),
    ("epochs", "--epochs", 30),
    ("batch_size", "--batch-size", 12),
    ("num_workers", "--num-workers", 8),
    ("lr", "--lr", 1e-6),
    ("weight_decay", "--weight-decay", 1e-5),
)
_FINETUNE_RUNTIME_OPTIONS = (
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


def finetune_loaded_split_values(recipe: dict, *, test_split_opt_in: bool = False) -> list[str]:
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}

    splits: list[str] = []
    if loads_train_val(runtime.get("epochs", 30)):
        splits.extend(["train", "val"])

    test_after_fit = evaluation.get("test_after_fit")
    if test_split_opt_in:
        if test_after_fit is True:
            splits.append("test")
    elif test_after_fit is not False:
        splits.append("test")
    return splits


def runtime_cli_args(runtime: dict[str, Any], *, variant: str | None = None) -> list[Any]:
    args: list[Any] = [
        "--devices",
        *[str(item) for item in coerce_list(runtime.get("devices", [0])) or [0]],
    ]
    for key, flag, default in _FINETUNE_RUNTIME_DEFAULTS:
        args.extend([flag, runtime.get(key, default)])
    for key, flag in _FINETUNE_RUNTIME_OPTIONS:
        append_option(args, flag, runtime.get(key))
    if variant != "sex_age_baseline":
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
) -> list[str]:
    cwd_lines = []
    if run_cwd is not None:
        root = shlex.quote(str(run_cwd))
        cwd_lines = [f"cd {root}", f"export PYTHONPATH={root}${{PYTHONPATH:+:$PYTHONPATH}}", ""]
    lifecycle_lines = []
    if experiment_root is not None:
        commit_code = (
            "import sys; "
            "from agent_tools.experiment_workspace import merge_run_manifest; "
            "rows = merge_run_manifest(sys.argv[1], "
            "[{'step_id': sys.argv[2], 'run_id': sys.argv[3], 'status': sys.argv[4]}]); "
            "row = next((row for row in rows "
            "if row.get('step_id') == sys.argv[2] and row.get('run_id') == sys.argv[3]), None); "
            "(row is not None and row.get('status') == sys.argv[4]) or "
            "sys.exit('Canonical run status did not commit as ' + sys.argv[4])"
        )
        commit_command = render_command([sys.executable, "-c", commit_code, experiment_root, step_id, run_id]) + ' "$1"'
        lifecycle_lines = [
            "_agent_commit_status() {",
            f"  {commit_command}",
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
            "_agent_commit_status running",
            # The trap is installed only after the owner accepts running, so terminal runs never execute again.
            "trap _agent_finish_run EXIT",
            "",
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
    final_external_test: bool = False,
    run_cwd: str | Path = REPO_ROOT,
) -> list[str]:
    external_test_policy = "# - This script evaluates the configured final test split."
    final_test_policy = "# - Final test evaluation was explicitly unlocked."
    if not final_external_test:
        external_test_policy = (
            "# - Run commands evaluate the configured test split after fit."
            if test_after_fit
            else "# - Run commands do not evaluate the external test split."
        )
        final_test_policy = "# - Final test evaluation requires explicit unlock."
    root = shlex.quote(str(run_cwd))
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {root}",
        f"export PYTHONPATH={root}",
        "",
        "# Agent policy status: PASS",
        "# This script was generated only after consultation gates passed.",
        "# High-impact decisions were resolved by explicit recipe/config/user inputs.",
        "# External test policy:",
        external_test_policy,
        "# - Model selection is based on validation metrics only.",
        final_test_policy,
        "",
        *commands,
    ]
