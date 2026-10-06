"""Task-flag helpers for the shared variant contract's ``sex_age_baseline.common`` module."""

from sleep2vec.common import validate_and_apply_imbalance_config

__all__ = ["apply_task_flags", "validate_and_apply_imbalance_config"]


def apply_task_flags(args, task_cfg) -> None:
    """Apply the YAML task; ``label_name`` only names the result namespace, so built-in labels never apply."""
    args.monitor = task_cfg.monitor
    args.monitor_mod = task_cfg.monitor_mod
    args.output_dim = task_cfg.output_dim
    args.is_seq = False
    args.is_survival = task_cfg.type == "survival"
    args.is_multilabel = task_cfg.type == "multilabel_classification"
    args.is_classification = False
