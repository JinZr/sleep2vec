from contextlib import contextmanager
import copy
import fcntl
import os
from pathlib import Path
import time
from typing import Any, Mapping

import pandas as pd

from wrist2vec.distributed import is_rank_zero_process

RESULT_METADATA_COLUMNS = (
    "experiment_version",
    "result_source",
    "config_path",
    "label_name",
    "eval_split",
    "ckpt_path",
    "lr",
    "batch_size",
    "n_few_shot",
    "channel_names",
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
