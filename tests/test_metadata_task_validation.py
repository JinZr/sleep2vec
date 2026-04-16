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


@pytest.mark.parametrize(
    ("label_name", "output_dim", "stage_names", "class_labels"),
    [
        ("stage3", 3, ["W", "NREM", "REM"], ["W", "NREM", "REM"]),
        ("stage4", 4, ["W", "N1N2", "N3", "REM"], ["W", "N1N2", "N3", "REM"]),
        ("stage5", 5, ["W", "N1", "N2", "N3", "REM"], ["W", "N1", "N2", "N3", "REM"]),
    ],
)
def test_apply_task_flags_allows_builtin_stage_multiclass(
    label_name: str,
    output_dim: int,
    stage_names: list[str],
    class_labels: list[str],
):
    args = _args(label_name)
    task_cfg = TaskConfig(
        type="classification",
        output_dim=output_dim,
        is_seq=True,
        monitor="val_accuracy",
        monitor_mod="max",
    )

    apply_task_flags(args, task_cfg)

    assert args.is_classification is True
    assert args.output_dim == output_dim
    assert args.is_seq is True
    assert args.label_source_name == "stage5"
    assert args.stage_names == stage_names
    assert args.class_labels == class_labels


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
    assert args.class_labels is None


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
    assert args.class_labels is None
