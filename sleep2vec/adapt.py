import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import typing as t

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
import wandb
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.samplers import handles_distributed_sharding
from sleep2vec.callbacks.pair_acc_logger import PairAccLoggerCallback
from sleep2vec.common import apply_data_backend_args, apply_model_config_args, persist_run_config_and_args
from sleep2vec.config import load_pretrain_config
from sleep2vec.sleep2vec_adaptation import AdaptPairScheduleCallback, Sleep2vecAdaptation, initial_pair_probs_for_phase
from sleep2vec.utils import get_pretrain_dataloader


def _optional_path(value):
    if value.lower() in {"null", "none"}:
        return None
    return Path(value)


@dataclass(frozen=True)
class AdaptRunArtifacts:
    save_path: Path
    run_name: str
    wandb_id: t.Optional[str]
    trainer_ckpt_path: t.Optional[Path]
    write_root_files: bool


def _require_checkpoint_file(ckpt_path: Path, *, arg_name: str) -> Path:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{arg_name} not found: {ckpt_path}")
    if not ckpt_path.is_file():
        raise ValueError(f"{arg_name} must be a file: {ckpt_path}")
    return ckpt_path


def _checkpoint_dir_name_for_phase(phase: str) -> str:
    if phase == "stage1":
        return "checkpoints"
    if phase == "stage2":
        return "checkpoints.stage2"
    raise ValueError(f"Unsupported adaptation phase '{phase}'.")


def _checkpoint_dir_for_phase(exp_dir: Path, phase: str) -> Path:
    return exp_dir / _checkpoint_dir_name_for_phase(phase)


def _validate_checkpoint_dir_for_phase(*, ckpt_path: Path, expected_phase: str, context: str) -> Path:
    exp_dir = ckpt_path.parent.parent
    expected_dir = _checkpoint_dir_for_phase(exp_dir, expected_phase)
    if ckpt_path.parent != expected_dir:
        raise ValueError(f"{context}. Expected checkpoint under {expected_dir}, got {ckpt_path.parent}.")
    return exp_dir


def _load_saved_cli_args(exp_dir: Path, *, phase: str | None = None) -> tuple[Path, dict[str, t.Any]]:
    candidates: list[Path] = []
    if phase:
        candidates.append(exp_dir / f"cli_args.{phase}.yaml")
    candidates.append(exp_dir / "cli_args.yaml")

    for candidate in candidates:
        if not candidate.exists():
            continue
        data = yaml.safe_load(candidate.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Saved CLI args must be a mapping: {candidate}")
        return candidate, data
    expected = ", ".join(str(path) for path in candidates)
    raise ValueError(f"Expected saved CLI args in the checkpoint run directory: {expected}.")


def _validate_saved_phase(*, exp_dir: Path, expected_phase: str, context: str) -> Path:
    cli_args_path, cli_args = _load_saved_cli_args(exp_dir, phase=expected_phase)
    phase = cli_args.get("phase")
    if phase != expected_phase:
        raise ValueError(f"{context}. Found phase={phase!r} in {cli_args_path}.")
    return cli_args_path


def _resolve_stage1_transition_checkpoint(pretrained_backbone_path: Path) -> Path:
    ckpt_path = _require_checkpoint_file(pretrained_backbone_path, arg_name="pretrained_backbone_path")
    exp_dir = _validate_checkpoint_dir_for_phase(
        ckpt_path=ckpt_path,
        expected_phase="stage1",
        context=("--phase stage2 requires --pretrained-backbone-path to point to a prior adapt stage1 checkpoint"),
    )
    _validate_saved_phase(
        exp_dir=exp_dir,
        expected_phase="stage1",
        context="--phase stage2 requires --pretrained-backbone-path to point to a prior adapt stage1 checkpoint",
    )
    return ckpt_path


def _resolve_adapt_run_artifacts(
    *,
    ckpt_path: t.Optional[Path],
    pretrained_backbone_path: t.Optional[Path],
    version_name: str,
    backbone_arch: str,
    phase: str,
    exp_info: str = "",
) -> AdaptRunArtifacts:
    if ckpt_path is not None:
        ckpt_path = _require_checkpoint_file(ckpt_path, arg_name="ckpt_path")
        exp_dir = _validate_checkpoint_dir_for_phase(
            ckpt_path=ckpt_path,
            expected_phase=phase,
            context=(
                f"--ckpt-path resumes an exact adapt {phase} run only; "
                "use --pretrained-backbone-path for stage transitions"
            ),
        )
        _validate_saved_phase(
            exp_dir=exp_dir,
            expected_phase=phase,
            context=(
                f"--ckpt-path resumes an exact adapt {phase} run only; "
                "use --pretrained-backbone-path for stage transitions"
            ),
        )
        save_path = ckpt_path.parent
        run_name = save_path.parent.name
        return AdaptRunArtifacts(
            save_path=save_path,
            run_name=run_name,
            wandb_id=run_name,
            trainer_ckpt_path=ckpt_path,
            write_root_files=False,
        )

    if phase == "stage2" and pretrained_backbone_path is not None:
        stage1_ckpt_path = _resolve_stage1_transition_checkpoint(pretrained_backbone_path)
        exp_dir = stage1_ckpt_path.parent.parent
        save_path = _checkpoint_dir_for_phase(exp_dir, phase)
        existing_ckpts = sorted(save_path.glob("*.ckpt")) if save_path.exists() else []
        if existing_ckpts:
            raise ValueError(
                "Fresh stage2 transition refuses to reuse a non-empty checkpoints.stage2 directory. "
                "Use --ckpt-path to resume the existing stage2 run, or clear/move the old checkpoints.stage2 directory first."
            )
        run_name = save_path.parent.name
        return AdaptRunArtifacts(
            save_path=save_path,
            run_name=run_name,
            wandb_id=run_name,
            trainer_ckpt_path=None,
            write_root_files=False,
        )

    exp_bits = [version_name, backbone_arch, "adapt", phase]
    extra_tag = exp_info.strip().replace(" ", "_")
    if extra_tag:
        exp_bits.append(extra_tag)
    run_name = "-".join(filter(None, exp_bits))
    return AdaptRunArtifacts(
        save_path=Path("log-adapt") / run_name / "checkpoints",
        run_name=run_name,
        wandb_id=None,
        trainer_ckpt_path=None,
        write_root_files=True,
    )


def sleep2vec_adapt(args):
    config_bundle = load_pretrain_config(args.config)
    if config_bundle.adapt is None:
        raise ValueError("Adaptation config requires a top-level 'adapt' block.")

    model_config = config_bundle.model
    loss_config = config_bundle.loss
    averaging_config = config_bundle.averaging
    adapt_config = config_bundle.adapt

    args.mask_rate = config_bundle.data.mask_rate
    args.max_tokens = config_bundle.data.max_tokens
    apply_model_config_args(args, model_config, set_backbone_arch=True)
    apply_data_backend_args(args, config_bundle.data, preset_attr="pretrain_preset_path")
    args.train_pair_probs = initial_pair_probs_for_phase(
        args.phase,
        channel_names=args.channel_names,
        adapt_config=adapt_config,
    )

    train_loader, val_loader = get_pretrain_dataloader(args)
    train_batch_sampler = getattr(train_loader, "batch_sampler", None)
    val_batch_sampler = getattr(val_loader, "batch_sampler", None)
    use_distributed_sampler = not handles_distributed_sharding(
        train_batch_sampler
    ) and not handles_distributed_sharding(val_batch_sampler)

    artifacts = _resolve_adapt_run_artifacts(
        ckpt_path=args.ckpt_path,
        pretrained_backbone_path=args.pretrained_backbone_path,
        version_name=args.version_name,
        backbone_arch=args.backbone_arch,
        phase=args.phase,
        exp_info=getattr(args, "exp_info", "") or "",
    )
    save_path = artifacts.save_path
    run_name = artifacts.run_name
    wandb_id = artifacts.wandb_id
    trainer_ckpt_path = artifacts.trainer_ckpt_path
    write_root_files = artifacts.write_root_files
    save_path.mkdir(parents=True, exist_ok=True)
    if wandb_id is not None:
        logging.info("Reusing adapt run directory %s (wandb_id=%s)", save_path.parent, wandb_id)

    exp_dir = save_path.parent
    persist_run_config_and_args(
        args,
        exp_dir,
        phase_name=str(args.phase),
        write_root_files=write_root_files,
    )

    model = Sleep2vecAdaptation(
        args,
        model_config,
        loss_config,
        adapt_config=adapt_config,
        averaging_config=averaging_config,
    )

    logger = WandbLogger(
        project="sleep2vec-adapt",
        name=f"s2v-adapt-{run_name}",
        save_dir=str(save_path.parent),
        id=wandb_id,
        resume="allow" if wandb_id else None,
    )

    monitor = "val_contrastive_acc"
    mode = "max"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(save_path),
        monitor=monitor,
        mode=mode,
        filename="epoch={epoch}-step={step}",
        save_on_train_epoch_end=True,
        every_n_epochs=1,
        save_top_k=50,
    )
    early_stop_cb = EarlyStopping(
        monitor=monitor,
        patience=args.patience,
        mode=mode,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    if args.strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=True)
    elif args.strategy == "deepspeed":
        if args.deepspeed_config is None:
            raise ValueError("deepspeed_config must be provided when using DeepSpeed strategy.")
        strategy = DeepSpeedStrategy(config=args.deepspeed_config)
    else:
        strategy = "auto"

    pair_acc_cb = PairAccLoggerCallback(
        args.channel_names,
        train_pair_monitor_enabled=args.train_pair_monitor_enable,
        train_pair_log_prefix=args.train_pair_monitor_log_prefix,
        train_pair_skew_warn_threshold=args.train_pair_skew_warn_threshold,
        train_pair_min_unique_coverage_warn_threshold=args.train_pair_min_unique_coverage_warn_threshold,
    )
    callbacks = [checkpoint_cb, early_stop_cb, lr_monitor, pair_acc_cb]
    if args.phase == "stage2":
        callbacks.append(
            AdaptPairScheduleCallback(
                new_channels=adapt_config.new_channels,
                pair_schedule=adapt_config.stage2.pair_schedule,
            )
        )

    enable_checkpointing = True
    trainer_kwargs = dict(
        devices=args.devices,
        accelerator="gpu",
        strategy=strategy,
        benchmark=True,
        use_distributed_sampler=use_distributed_sampler,
        logger=logger,
        max_epochs=args.epochs,
        log_every_n_steps=5,
        num_sanity_val_steps=0,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    if args.print_diagnostics:
        callbacks = []
        enable_checkpointing = False
        trainer_kwargs.update(
            dict(
                enable_progress_bar=False,
                max_steps=args.diagnostics_steps,
                limit_val_batches=0,
            )
        )

    trainer = pl.Trainer(
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing,
        **trainer_kwargs,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader if not args.print_diagnostics else None,
        ckpt_path=trainer_ckpt_path,
    )


if __name__ == "__main__":
    wandb.login()

    parser = argparse.ArgumentParser(description="Run staged modality adaptation for sleep2vec.")
    parser.add_argument("--config", type=Path, required=True, help="Pretrain-style YAML containing an adapt block.")
    parser.add_argument("--phase", type=str, choices=["stage1", "stage2"], required=True, help="Adaptation phase.")
    parser.add_argument("--epochs", type=int, default=120, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Base learning rate.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override warmup steps for LR schedule (default: 3%% of total steps).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for AdamW.")
    parser.add_argument("--batch-size", type=int, default=320, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=16, help="Training dataloader workers.")
    parser.add_argument(
        "--val-num-workers",
        type=int,
        default=4,
        help="Validation dataloader workers.",
    )
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1], help="GPU device ids.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience in epochs.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device used by dataloader.")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Number of batches to accumulate before each optimizer step.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        help="Mixed precision setting passed to Lightning Trainer.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="Optional Lightning checkpoint to resume exactly within the same adapt phase/run.",
    )
    parser.add_argument(
        "--pretrained-backbone-path",
        type=Path,
        default=None,
        help=(
            "Weight-init checkpoint. Use this for fresh stage1 runs and for stage1->stage2 transitions; "
            "stage2 expects it to point to a prior adapt stage1 checkpoint. "
            "Loads ema_model. first and falls back to model."
        ),
    )
    parser.add_argument("--version-name", type=str, required=True, help="Version name used for logging.")
    parser.add_argument("--exp-info", type=str, default="", help="Optional extra tag appended to the run name.")
    parser.add_argument(
        "--pretrain-data-index",
        type=Path,
        default="index/hsp_psg_pretrain.csv",
        help="CSV index file for adaptation data.",
    )
    parser.add_argument(
        "--pretrain-preset-path",
        type=_optional_path,
        default=None,
        help="Path to precomputed preset pickle for adaptation data; use null/none to disable.",
    )
    parser.add_argument(
        "--allow-missing-channels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable pair-first sampling over heterogeneous modality availability.",
    )
    parser.add_argument("--min-channels", type=int, default=2, help="Minimum available channels for adaptation.")
    parser.add_argument(
        "--bucket-by-available-channels",
        dest="bucket_by_available_channels",
        action="store_true",
        default=True,
        help="Bucket batches by available-channel signature when pair-first is not active.",
    )
    parser.add_argument(
        "--no-bucket-by-available-channels",
        dest="bucket_by_available_channels",
        action="store_false",
        help="Disable available-channel bucketing fallback.",
    )
    parser.add_argument(
        "--train-pair-track-unique-samples",
        action="store_true",
        help="Track per-pair unique sampled indices during training monitoring.",
    )
    parser.add_argument(
        "--train-pair-skew-warn-threshold",
        type=float,
        default=0.05,
        help="Warn when |actual_pair_ratio - target_pair_ratio| exceeds this threshold.",
    )
    parser.add_argument(
        "--train-pair-monitor-enable",
        dest="train_pair_monitor_enable",
        action="store_true",
        default=True,
        help="Enable epoch-level train pair sampling distribution monitoring.",
    )
    parser.add_argument(
        "--no-train-pair-monitor-enable",
        dest="train_pair_monitor_enable",
        action="store_false",
        help="Disable train pair sampling distribution monitoring.",
    )
    parser.add_argument(
        "--train-pair-monitor-log-prefix",
        type=str,
        default="train_pair_sampling",
        help="Metric prefix for train pair sampling logs.",
    )
    parser.add_argument(
        "--train-pair-min-unique-coverage-warn-threshold",
        type=float,
        default=0.1,
        help="Warn when unique sampled indices / pair pool size falls below this threshold.",
    )
    parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "deepspeed", "auto"])
    parser.add_argument("--deepspeed-config", type=Path, default=None, help="DeepSpeed config path when used.")
    parser.add_argument(
        "--print-diagnostics",
        action="store_true",
        help="Run a short diagnostic pass, print tensor stats, and exit.",
    )
    parser.add_argument(
        "--diagnostics-steps",
        type=int,
        default=5,
        help="Number of training steps to gather diagnostics before stopping.",
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sleep2vec_adapt(parser.parse_args())
