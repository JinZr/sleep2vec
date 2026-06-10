from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
import uuid

import pandas as pd
import torch

from sleep2vec2.distributed import is_rank_zero_process

PACKAGE_NAMESPACE = "sleep2vec2"
DEFAULT_INFERENCE_RESULTS_ROOT = Path("results/inference")
_EPOCH_RE = re.compile(r"(?:epoch[=\-]?)(\d+)")
_STEP_RE = re.compile(r"(?:step[=\-]?)(\d+)")

RESULT_METADATA_COLUMNS = (
    "experiment_version",
    "result_source",
    "prediction_run_id",
    "namespace",
    "config_path",
    "label_name",
    "eval_split",
    "ckpt_path",
    "run_dir",
    "metrics_csv_path",
    "prediction_csv_path",
    "ckpt_input",
    "ckpt_resolved_path",
    "ckpt_epoch",
    "ckpt_step",
    "ckpt_tag",
    "task_family",
    "timestamp_utc",
    "lr",
    "batch_size",
    "n_few_shot",
    "channel_names",
    "preset_path",
)

PREDICTION_METADATA_COLUMNS = (
    "prediction_run_id",
    "namespace",
    "experiment_version",
    "result_source",
    "config_path",
    "label_name",
    "eval_split",
    "ckpt_path",
    "run_dir",
    "metrics_csv_path",
    "prediction_csv_path",
    "ckpt_input",
    "ckpt_resolved_path",
    "ckpt_epoch",
    "ckpt_step",
    "ckpt_tag",
    "task_family",
    "timestamp_utc",
    "path",
    "groundtruth",
    "prediction",
    "n_predictions",
    "n_windows",
    "token_starts",
)


def save_result_csv(pretrain_result: Mapping[str, float], csv_path: str, args: Any | None = None):
    """Append one experiment result row to `csv_path`.

    The rank-zero gate here intentionally follows the repository's current
    single-node assumption by delegating to `is_rank_zero_process()`. Multi-node
    distributed jobs are not considered in this write path today.
    """
    if not csv_path or not is_rank_zero_process():
        return

    new_row: dict[str, Any] = dict(copy.deepcopy(pretrain_result))
    new_row["experiment_version"] = _resolve_experiment_version(args)
    new_row["result_source"] = _resolve_result_source(args)
    prediction_run_id = getattr(args, "prediction_run_id", None) if args is not None else None
    if prediction_run_id not in (None, ""):
        new_row["prediction_run_id"] = prediction_run_id
        _add_inference_run_metadata(new_row, args)
    new_row["config_path"] = _stringify_optional_path(getattr(args, "config", None)) if args is not None else ""
    new_row["eval_split"] = getattr(args, "eval_split", None) if args is not None else None

    if args is not None:
        new_row["ckpt_path"] = getattr(args, "ckpt_path", None)
        new_row["lr"] = getattr(args, "lr", None)
        new_row["batch_size"] = getattr(args, "batch_size", None)
        new_row["n_few_shot"] = getattr(args, "n_few_shot", None)
        new_row["label_name"] = getattr(args, "label_name", None)
        channel_names = getattr(args, "channel_names", None)
        if isinstance(channel_names, (list, tuple)):
            new_row["channel_names"] = ",".join(str(name) for name in channel_names)
        elif isinstance(channel_names, str):
            new_row["channel_names"] = channel_names
        else:
            new_row["channel_names"] = ""
        preset_path = getattr(args, "inference_preset_path", None) or getattr(args, "finetune_preset_path", None)
        new_row["preset_path"] = _stringify_optional_path(preset_path)

    df_new = pd.DataFrame([new_row])
    csv_file = Path(csv_path)

    with _result_csv_lock(csv_file):
        if not csv_file.exists() or csv_file.stat().st_size == 0:
            ordered_columns = _ordered_result_columns(df_new)
            _write_result_csv(df_new.reindex(columns=ordered_columns), csv_file)
            print(f"Results written to {csv_path} [experiment_version={new_row['experiment_version']}]")
            return

        try:
            df_old = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            ordered_columns = _ordered_result_columns(df_new)
            _write_result_csv(df_new.reindex(columns=ordered_columns), csv_file)
            print(f"Results written to {csv_path} [experiment_version={new_row['experiment_version']}]")
            return

        existing_columns = list(df_old.columns)
        if all(column in existing_columns for column in df_new.columns):
            _write_result_csv(df_new.reindex(columns=existing_columns), csv_file, mode="a", header=False)
        else:
            ordered_columns = _ordered_result_columns(df_old, df_new)
            df_merged = pd.concat(
                [df_old.reindex(columns=ordered_columns), df_new.reindex(columns=ordered_columns)],
                axis=0,
                ignore_index=True,
            )
            _write_result_csv(df_merged, csv_file)

    print(f"Results written to {csv_path} [experiment_version={new_row['experiment_version']}]")


def prepare_inference_result_paths(
    args: Any,
    *,
    namespace: str = PACKAGE_NAMESPACE,
    root: str | Path = DEFAULT_INFERENCE_RESULTS_ROOT,
    checkpoint_paths: Sequence[str | Path] | None = None,
    timestamp: str | None = None,
) -> None:
    timestamp = timestamp or getattr(args, "timestamp_utc", None)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    ckpt_info = _resolve_checkpoint_info(args, checkpoint_paths=checkpoint_paths)
    label_slug = _slug_piece(getattr(args, "label_name", None) or "label") or "label"
    split_slug = _slug_piece(getattr(args, "eval_split", None) or "split") or "split"
    ckpt_tag = ckpt_info["ckpt_tag"]
    run_id = make_prediction_run_id(args, timestamp=timestamp, namespace=namespace, ckpt_info=ckpt_info)

    root_path = Path(root)
    run_dir = root_path / (_slug_piece(namespace) or "namespace") / label_slug / run_id
    metrics_csv_path = run_dir / f"metrics__{label_slug}__{split_slug}__{ckpt_tag}.csv"
    prediction_csv_path = run_dir / f"predictions__{label_slug}__{split_slug}__{ckpt_tag}.csv"
    manifest_path = run_dir / "run_manifest.json"

    args.prediction_run_id = run_id
    args.inference_namespace = namespace
    args.inference_results_root = root_path
    args.inference_overview_csv_path = root_path / "overview.csv"
    args.run_dir = run_dir
    args.inference_metrics_csv_path = metrics_csv_path
    args.inference_prediction_csv_path = prediction_csv_path
    args.manifest_path = manifest_path
    args.timestamp_utc = timestamp
    args.ckpt_input = ckpt_info["ckpt_input"]
    args.ckpt_resolved_path = ckpt_info["ckpt_resolved_path"]
    args.ckpt_epoch = ckpt_info["ckpt_epoch"]
    args.ckpt_step = ckpt_info["ckpt_step"]
    args.ckpt_tag = ckpt_tag
    args.task_family = _resolve_task_family(args)
    args.inference_checkpoint_paths = ckpt_info["checkpoint_paths"]


def make_prediction_run_id(
    args: Any,
    *,
    timestamp: str | None = None,
    namespace: str = PACKAGE_NAMESPACE,
    ckpt_info: Mapping[str, Any] | None = None,
) -> str:
    launch_nonce = uuid.uuid4().hex[:8]
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if ckpt_info is None:
        ckpt_info = _resolve_checkpoint_info(args)
    experiment_version = _resolve_experiment_version(args)
    label_name = getattr(args, "label_name", None)
    eval_split = getattr(args, "eval_split", None)
    ckpt_tag = ckpt_info["ckpt_tag"]
    hash_payload = {
        "config_path": _stringify_optional_path(getattr(args, "config", None)),
        "ckpt_input": ckpt_info["ckpt_input"],
        "ckpt_resolved_path": ckpt_info["ckpt_resolved_path"],
        "checkpoint_paths": ckpt_info["checkpoint_paths"],
        "label_name": label_name,
        "eval_split": eval_split,
        "timestamp_utc": timestamp,
        "launch_nonce": launch_nonce,
    }
    short_hash = hashlib.sha1(json.dumps(hash_payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    pieces = [timestamp, namespace, experiment_version, label_name, eval_split, ckpt_tag, short_hash]
    return "__".join(_slug_piece(piece) or "unknown" for piece in pieces if piece not in (None, ""))


def save_prediction_csv(rows: Sequence[Mapping[str, Any]], csv_path: str, args: Any | None = None) -> None:
    if not csv_path or not is_rank_zero_process():
        return

    prediction_rows = []
    for row in rows:
        new_row: dict[str, Any] = {key: _serialize_prediction_value(value) for key, value in copy.deepcopy(row).items()}
        new_row["prediction_run_id"] = getattr(args, "prediction_run_id", None) if args is not None else None
        new_row["experiment_version"] = _resolve_experiment_version(args)
        new_row["result_source"] = _resolve_result_source(args)
        new_row["config_path"] = _stringify_optional_path(getattr(args, "config", None)) if args is not None else ""
        new_row["label_name"] = getattr(args, "label_name", None) if args is not None else None
        new_row["eval_split"] = getattr(args, "eval_split", None) if args is not None else None
        new_row["ckpt_path"] = getattr(args, "ckpt_path", None) if args is not None else None
        _add_inference_run_metadata(new_row, args)
        prediction_rows.append(new_row)

    df_new = pd.DataFrame(prediction_rows) if prediction_rows else pd.DataFrame(columns=PREDICTION_METADATA_COLUMNS)
    csv_file = Path(csv_path)

    with _result_csv_lock(csv_file):
        if not csv_file.exists() or csv_file.stat().st_size == 0:
            ordered_columns = _ordered_prediction_columns(df_new)
            _write_result_csv(df_new.reindex(columns=ordered_columns), csv_file)
            print(f"Predictions written to {csv_path} [rows={len(df_new)}]")
            return

        try:
            df_old = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            ordered_columns = _ordered_prediction_columns(df_new)
            _write_result_csv(df_new.reindex(columns=ordered_columns), csv_file)
            print(f"Predictions written to {csv_path} [rows={len(df_new)}]")
            return

        existing_columns = list(df_old.columns)
        if all(column in existing_columns for column in df_new.columns):
            _write_result_csv(df_new.reindex(columns=existing_columns), csv_file, mode="a", header=False)
        else:
            ordered_columns = _ordered_prediction_columns(df_old, df_new)
            df_merged = pd.concat(
                [df_old.reindex(columns=ordered_columns), df_new.reindex(columns=ordered_columns)],
                axis=0,
                ignore_index=True,
            )
            _write_result_csv(df_merged, csv_file)

    print(f"Predictions written to {csv_path} [rows={len(df_new)}]")


def save_inference_manifest(
    args: Any,
    metrics: Mapping[str, Any] | None = None,
    *,
    prediction_row_count: int = 0,
) -> None:
    if not is_rank_zero_process():
        return
    manifest_path = getattr(args, "manifest_path", None)
    if manifest_path in (None, ""):
        return

    manifest = {
        "prediction_run_id": getattr(args, "prediction_run_id", None),
        "namespace": getattr(args, "inference_namespace", None),
        "experiment_version": _resolve_experiment_version(args),
        "result_source": _resolve_result_source(args),
        "timestamp_utc": getattr(args, "timestamp_utc", None),
        "config_path": _stringify_optional_path(getattr(args, "config", None)),
        "label_name": getattr(args, "label_name", None),
        "eval_split": getattr(args, "eval_split", None),
        "task_family": getattr(args, "task_family", None),
        "paths": {
            "run_dir": _stringify_optional_path(getattr(args, "run_dir", None)),
            "overview_csv_path": _stringify_optional_path(getattr(args, "inference_overview_csv_path", None)),
            "metrics_csv_path": _stringify_optional_path(getattr(args, "inference_metrics_csv_path", None)),
            "prediction_csv_path": _stringify_optional_path(getattr(args, "inference_prediction_csv_path", None)),
            "manifest_path": _stringify_optional_path(manifest_path),
        },
        "checkpoint": {
            "input": getattr(args, "ckpt_input", getattr(args, "ckpt_path", None)),
            "resolved_path": getattr(args, "ckpt_resolved_path", None),
            "paths": getattr(args, "inference_checkpoint_paths", None),
            "epoch": getattr(args, "ckpt_epoch", None),
            "step": getattr(args, "ckpt_step", None),
            "tag": getattr(args, "ckpt_tag", None),
            "avg_ckpts": getattr(args, "avg_ckpts", None),
        },
        "runtime": {
            "batch_size": getattr(args, "batch_size", None),
            "devices": getattr(args, "devices", None),
            "accelerator": getattr(args, "accelerator", None),
            "precision": getattr(args, "precision", None),
            "inference_preset_path": _stringify_optional_path(getattr(args, "inference_preset_path", None)),
            "finetune_preset_path": _stringify_optional_path(getattr(args, "finetune_preset_path", None)),
        },
        "metrics": dict(metrics or {}),
        "prediction_row_count": prediction_row_count,
    }
    _write_json_atomic(_json_safe(manifest), Path(manifest_path))


def save_training_run_manifest(
    args: Any,
    *,
    manifest_path: str | Path,
    status: str,
    monitor: str | None = None,
    monitor_mode: str | None = None,
    best_model_path: str | Path | None = None,
    best_model_score: Any = None,
    last_checkpoint_path: str | Path | None = None,
    results_csv_path: str | Path | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> None:
    if not is_rank_zero_process():
        return
    path = Path(manifest_path)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "schema_version": 1,
        "kind": "sleep2vec_finetune_run",
        "status": status,
        "created_at_utc": created_at,
        "finished_at_utc": created_at,
        "version": getattr(args, "version", None),
        "config_path": _stringify_optional_path(getattr(args, "config", None)),
        "config_copy_path": _stringify_optional_path(Path(f"log-finetune/{getattr(args, 'version', '')}/config.yaml")),
        "cli_args_path": _stringify_optional_path(Path(f"log-finetune/{getattr(args, 'version', '')}/cli_args.yaml")),
        "label_name": getattr(args, "label_name", None),
        "monitor": monitor,
        "monitor_mode": monitor_mode,
        "best_model_path": _stringify_optional_path(best_model_path),
        "best_model_score": _json_safe(best_model_score),
        "last_checkpoint_path": _stringify_optional_path(last_checkpoint_path),
        "test_after_fit": getattr(args, "test_after_fit", None),
        "results_csv_path": _stringify_optional_path(results_csv_path),
        "metrics": dict(metrics or {}),
        "git": _git_manifest(),
    }
    _write_json_atomic(_json_safe(manifest), path)


def _git_manifest() -> dict[str, Any]:
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"], check=False, capture_output=True, text=True
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--short"], check=False, capture_output=True, text=True
            ).stdout.strip()
        )
    except OSError:
        return {"available": False}
    return {"available": True, "branch": branch, "commit": commit, "dirty": dirty}


def _add_inference_run_metadata(row: dict[str, Any], args: Any | None) -> None:
    if args is None:
        return
    row["namespace"] = getattr(args, "inference_namespace", None)
    row["run_dir"] = _stringify_optional_path(getattr(args, "run_dir", None))
    row["metrics_csv_path"] = _stringify_optional_path(getattr(args, "inference_metrics_csv_path", None))
    row["prediction_csv_path"] = _stringify_optional_path(getattr(args, "inference_prediction_csv_path", None))
    row["ckpt_input"] = getattr(args, "ckpt_input", getattr(args, "ckpt_path", None))
    row["ckpt_resolved_path"] = getattr(args, "ckpt_resolved_path", None)
    row["ckpt_epoch"] = getattr(args, "ckpt_epoch", None)
    row["ckpt_step"] = getattr(args, "ckpt_step", None)
    row["ckpt_tag"] = getattr(args, "ckpt_tag", None)
    row["task_family"] = getattr(args, "task_family", None)
    row["timestamp_utc"] = getattr(args, "timestamp_utc", None)


def _stringify_optional_path(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def _resolve_result_source(args: Any | None) -> str:
    if args is None:
        return ""
    return "infer" if getattr(args, "eval_split", None) not in (None, "") else "finetune"


def _resolve_experiment_version(args: Any | None) -> str:
    if args is None:
        return "unversioned"

    for attr_name in ("version", "version_name", "wandb_name", "wandb_id"):
        value = getattr(args, attr_name, None)
        if value not in (None, ""):
            return str(value)

    pieces: list[str] = []
    config_path = getattr(args, "config", None)
    if config_path not in (None, ""):
        pieces.append(Path(str(config_path)).stem)
    label_name = getattr(args, "label_name", None)
    if label_name not in (None, ""):
        pieces.append(str(label_name))
    eval_split = getattr(args, "eval_split", None)
    if eval_split not in (None, ""):
        pieces.append(str(eval_split))
    ckpt_path = getattr(args, "ckpt_path", None)
    if ckpt_path not in (None, ""):
        ckpt_text = str(ckpt_path)
        pieces.append(Path(ckpt_text).stem if ckpt_text not in {"best", "last"} else ckpt_text)
    return "-".join(pieces) if pieces else "unversioned"


def _resolve_checkpoint_info(
    args: Any,
    *,
    checkpoint_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    ckpt_input = getattr(args, "ckpt_input", None) or getattr(args, "ckpt_path", None)
    if checkpoint_paths is None:
        checkpoint_paths = getattr(args, "inference_checkpoint_paths", None)

    if checkpoint_paths:
        paths = [Path(path) for path in checkpoint_paths]
        infos = [{"epoch": epoch, "step": step} for epoch, step in (_parse_checkpoint_name(path) for path in paths)]
        epochs = [info["epoch"] for info in infos if info["epoch"] is not None]
        steps = [info["step"] for info in infos if info["step"] is not None]
        if epochs:
            ckpt_tag = f"avg{len(paths)}_epoch{min(epochs):02d}-{max(epochs):02d}"
        else:
            ckpt_tag = f"avg{len(paths)}_{_slug_piece(paths[-1].stem) or 'ckpt'}"
        return {
            "ckpt_input": _stringify_optional_path(ckpt_input),
            "ckpt_resolved_path": ",".join(str(path) for path in paths),
            "checkpoint_paths": [str(path) for path in paths],
            "ckpt_epoch": max(epochs) if epochs else None,
            "ckpt_step": steps[-1] if steps else None,
            "ckpt_tag": ckpt_tag,
        }

    resolved_path = getattr(args, "ckpt_resolved_path", None)
    ckpt_path = resolved_path or getattr(args, "ckpt_path", None)
    if ckpt_path in (None, ""):
        return {
            "ckpt_input": _stringify_optional_path(ckpt_input),
            "ckpt_resolved_path": _stringify_optional_path(resolved_path),
            "checkpoint_paths": [],
            "ckpt_epoch": None,
            "ckpt_step": None,
            "ckpt_tag": "ckpt",
        }
    if str(ckpt_path) in {"best", "last"}:
        return {
            "ckpt_input": _stringify_optional_path(ckpt_input),
            "ckpt_resolved_path": _stringify_optional_path(resolved_path),
            "checkpoint_paths": [],
            "ckpt_epoch": None,
            "ckpt_step": None,
            "ckpt_tag": _slug_piece(ckpt_path) or "ckpt",
        }

    path = Path(str(ckpt_path))
    info = _read_single_checkpoint_info(path)
    ckpt_tag = _format_checkpoint_tag(info["epoch"], info["step"], fallback=path.stem)
    return {
        "ckpt_input": _stringify_optional_path(ckpt_input),
        "ckpt_resolved_path": str(path),
        "checkpoint_paths": [str(path)],
        "ckpt_epoch": info["epoch"],
        "ckpt_step": info["step"],
        "ckpt_tag": ckpt_tag,
    }


def _read_single_checkpoint_info(path: Path) -> dict[str, int | None]:
    epoch, step = _parse_checkpoint_name(path)
    if path.exists():
        ckpt = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        if isinstance(ckpt, Mapping):
            epoch = _as_int(ckpt.get("epoch"), epoch)
            step = _as_int(ckpt.get("global_step"), step)
    return {"epoch": epoch, "step": step}


def _parse_checkpoint_name(path: Path) -> tuple[int | None, int | None]:
    epoch_match = _EPOCH_RE.search(path.name)
    step_match = _STEP_RE.search(path.name)
    epoch = int(epoch_match.group(1)) if epoch_match else None
    if epoch is None and path.stem.isdigit():
        epoch = int(path.stem)
    step = int(step_match.group(1)) if step_match else None
    return epoch, step


def _format_checkpoint_tag(epoch: int | None, step: int | None, *, fallback: str) -> str:
    if epoch is not None and step is not None:
        return f"epoch{epoch:02d}_step{step}"
    if epoch is not None:
        return f"epoch{epoch:02d}"
    return _slug_piece(fallback) or "ckpt"


def _as_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_task_family(args: Any) -> str:
    if getattr(args, "label_name", None) == "ahi":
        return "ahi_sequence"
    if getattr(args, "is_multilabel", False):
        return "multilabel_sequence"
    if getattr(args, "is_classification", False):
        return "sequence_classification" if getattr(args, "is_seq", False) else "scalar_classification"
    return "sequence_regression" if getattr(args, "is_seq", False) else "scalar_regression"


def _slug_piece(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text).strip("_")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _write_json_atomic(payload: Mapping[str, Any], json_path: Path) -> None:
    if json_path.parent:
        json_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = json_path.with_name(f".{json_path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temp_path.open("w") as file_obj:
            json.dump(payload, file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")
        os.replace(temp_path, json_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _serialize_prediction_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def _ordered_result_columns(*frames: pd.DataFrame) -> list[str]:
    ordered_columns: list[str] = []
    for column in RESULT_METADATA_COLUMNS:
        if any(column in frame.columns for frame in frames):
            ordered_columns.append(column)
    for frame in frames:
        for column in frame.columns:
            if column not in ordered_columns:
                ordered_columns.append(column)
    return ordered_columns


def _ordered_prediction_columns(*frames: pd.DataFrame) -> list[str]:
    ordered_columns: list[str] = []
    for column in PREDICTION_METADATA_COLUMNS:
        if any(column in frame.columns for frame in frames):
            ordered_columns.append(column)
    prob_columns = sorted(
        {
            column
            for frame in frames
            for column in frame.columns
            if isinstance(column, str) and column.startswith("prob_") and column[5:].isdigit()
        },
        key=lambda value: int(value[5:]),
    )
    for column in prob_columns:
        if column not in ordered_columns:
            ordered_columns.append(column)
    for frame in frames:
        for column in frame.columns:
            if column not in ordered_columns:
                ordered_columns.append(column)
    return ordered_columns


@contextmanager
def _result_csv_lock(csv_path: Path):
    lock_path = csv_path.with_name(f".{csv_path.name}.lock")
    if lock_path.parent:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_result_csv(df: pd.DataFrame, csv_path: Path, *, mode: str = "w", header: bool = True) -> None:
    if csv_path.parent:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "a":
        df.to_csv(csv_path, mode=mode, header=header, index=False)
        return

    temp_path = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, csv_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
