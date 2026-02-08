import argparse

import pytest

from sleep2vec.common import apply_task_flags
from sleep2vec.config import TaskConfig


def _args(label_name: str) -> argparse.Namespace:
    return argparse.Namespace(label_name=label_name)


def test_apply_task_flags_rejects_multiclass_metadata_label():
    args = _args("custom_label")
    task_cfg = TaskConfig(
        type="classification",
        output_dim=3,
        is_seq=False,
        monitor="val_accuracy",
        monitor_mod="max",
    )

    with pytest.raises(ValueError, match="Metadata classification currently supports only binary labels"):
        apply_task_flags(args, task_cfg)


def test_apply_task_flags_allows_stage5_multiclass():
    args = _args("stage5")
    task_cfg = TaskConfig(
        type="classification",
        output_dim=5,
        is_seq=True,
        monitor="val_accuracy",
        monitor_mod="max",
    )

    apply_task_flags(args, task_cfg)

    assert args.is_classification is True
    assert args.output_dim == 5
    assert args.is_seq is True


def test_apply_task_flags_allows_binary_metadata_classification():
    args = _args("custom_label")
    task_cfg = TaskConfig(
        type="classification",
        output_dim=2,
        is_seq=False,
        monitor="val_accuracy",
        monitor_mod="max",
    )

    apply_task_flags(args, task_cfg)

    assert args.is_classification is True
    assert args.output_dim == 2
    assert args.is_seq is False


def test_apply_task_flags_allows_metadata_regression():
    args = _args("custom_label")
    task_cfg = TaskConfig(
        type="regression",
        output_dim=1,
        is_seq=False,
        monitor="val_mae",
        monitor_mod="min",
    )

    apply_task_flags(args, task_cfg)

    assert args.is_classification is False
    assert args.output_dim == 1
    assert args.is_seq is False
