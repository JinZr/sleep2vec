from __future__ import annotations

import logging
import math
import typing as t

import pytorch_lightning as pl
import torch

from wrist2vec.checkpoints import load_pretrain_init_weights
from wrist2vec.config import AdaptConfig
from wrist2vec.data.channel_selection import build_all_pairs
from wrist2vec.wrist2vec_modelling import Wrist2vecPretraining

Pair = tuple[str, str]


def build_new_modality_pair_probs(
    pairs: t.Sequence[Pair],
    *,
    new_channels: t.Sequence[str],
    new_pair_ratio: float,
) -> dict[Pair, float]:
    if not pairs:
        raise ValueError("Pair schedule requires at least one available pair.")
    if not (0.0 <= float(new_pair_ratio) <= 1.0):
        raise ValueError(f"new_pair_ratio must be in [0, 1], got {new_pair_ratio}")

    new_channel_set = {str(name) for name in new_channels}
    new_pairs = [pair for pair in pairs if pair[0] in new_channel_set or pair[1] in new_channel_set]
    legacy_pairs = [pair for pair in pairs if pair not in new_pairs]
    if not new_pairs:
        raise ValueError("Pair schedule found no pairs involving adapt.new_channels.")

    if not legacy_pairs:
        uniform = 1.0 / float(len(new_pairs))
        return {pair: uniform for pair in new_pairs}

    probs: dict[Pair, float] = {}
    new_share = float(new_pair_ratio)
    legacy_share = 1.0 - new_share
    for pair in new_pairs:
        probs[pair] = new_share / float(len(new_pairs))
    for pair in legacy_pairs:
        probs[pair] = legacy_share / float(len(legacy_pairs))
    return probs


class AdaptPairScheduleCallback(pl.Callback):
    def __init__(self, *, new_channels: t.Sequence[str], pair_schedule: t.Sequence[t.Any]) -> None:
        super().__init__()
        self._new_channels = [str(name) for name in new_channels]
        self._pair_schedule = list(pair_schedule)

    def _resolve_ratio(self, progress: float) -> float:
        for point in self._pair_schedule:
            if progress <= float(point.until):
                return float(point.new_pair_ratio)
        return float(self._pair_schedule[-1].new_pair_ratio)

    @staticmethod
    def _resolve_train_pair_sampler(trainer) -> t.Any:
        train_loader = getattr(trainer, "train_dataloader", None)
        if isinstance(train_loader, dict):
            train_loader = next(iter(train_loader.values()), None)
        if isinstance(train_loader, (list, tuple)):
            train_loader = train_loader[0] if train_loader else None
        if train_loader is None:
            return None
        return getattr(train_loader, "batch_sampler", None)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        sampler = self._resolve_train_pair_sampler(trainer)
        if sampler is None or not hasattr(sampler, "set_pair_probs") or not hasattr(sampler, "pairs"):
            return

        max_epochs = max(1, int(getattr(trainer, "max_epochs", 1)))
        progress = float(trainer.current_epoch + 1) / float(max_epochs)
        ratio = self._resolve_ratio(progress)
        pair_probs = build_new_modality_pair_probs(sampler.pairs, new_channels=self._new_channels, new_pair_ratio=ratio)
        sampler.set_pair_probs(pair_probs)
        logging.info(
            "Updated train pair schedule for epoch=%s progress=%.4f new_pair_ratio=%.4f",
            trainer.current_epoch,
            progress,
            ratio,
        )


class Wrist2vecAdaptation(Wrist2vecPretraining):
    def __init__(self, args, model_config, loss_config, adapt_config: AdaptConfig, averaging_config=None):
        super().__init__(args, model_config, loss_config, averaging_config=averaging_config)
        self.adapt_config = adapt_config
        self.phase = str(args.phase)

        if not args.pretrained_backbone_path and args.ckpt_path is None:
            raise ValueError("Adaptation requires --pretrained-backbone-path.")

        if args.pretrained_backbone_path and args.ckpt_path is None:
            load_info = load_pretrain_init_weights(
                self.model, args.pretrained_backbone_path, device="cpu", strict=False
            )
            logging.info(
                "Loaded adaptation init from %s using prefix=%s (%d keys).",
                args.pretrained_backbone_path,
                load_info.used_prefix,
                load_info.loaded_keys,
            )
            if load_info.missing_keys:
                logging.warning("Missing adaptation init keys: %s", load_info.missing_keys)
            if load_info.unexpected_keys:
                logging.warning("Unexpected adaptation init keys: %s", load_info.unexpected_keys)

        if self.model_averager is not None:
            self.model_averager.sync_from_student()

        train_shared_projection = (
            bool(self.adapt_config.stage1.train_shared_projection) if self.phase == "stage1" else False
        )
        self.model.apply_adaptation_freeze_policy(
            phase=self.phase,
            new_channels=self.adapt_config.new_channels,
            train_shared_projection=train_shared_projection,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.model.apply_forced_module_modes()
            if self.model_averager is not None and self.model_averager.averaged_model is not None:
                self.model_averager.averaged_model.eval()
        return self

    def configure_optimizers(self):
        grouped = self.model.get_adaptation_param_groups(self.adapt_config.new_channels)

        if self.phase == "stage1":
            group_defs: list[tuple[str, float]] = [("new_modalities", 1.0)]
            if self.adapt_config.stage1.train_shared_projection:
                group_defs.append(("shared_projection", 1.0))
        elif self.phase == "stage2":
            scales = self.adapt_config.stage2.lr_scales
            grouped["shared_legacy"] = list(grouped["shared_projection"]) + list(grouped["legacy_modalities"])
            group_defs = [
                ("encoder_cls", float(scales.encoder)),
                ("shared_legacy", float(scales.shared_legacy)),
                ("new_modalities", float(scales.new_modalities)),
            ]
        else:
            raise ValueError(f"Unsupported adaptation phase '{self.phase}'.")

        optimizer_groups = []
        for group_name, lr_scale in group_defs:
            params = grouped.get(group_name, [])
            if not params:
                continue
            decay = []
            no_decay = []
            for name, param in params:
                if not param.requires_grad:
                    continue
                if param.ndim >= 2 and ("norm" not in name.lower()) and ("bias" not in name.lower()):
                    decay.append(param)
                else:
                    no_decay.append(param)

            lr = float(self.args.lr) * float(lr_scale)
            if decay:
                optimizer_groups.append(
                    {
                        "params": decay,
                        "weight_decay": self.args.weight_decay,
                        "lr": lr,
                        "group_name": f"{group_name}_decay",
                    }
                )
            if no_decay:
                optimizer_groups.append(
                    {
                        "params": no_decay,
                        "weight_decay": 0.0,
                        "lr": lr,
                        "group_name": f"{group_name}_no_decay",
                    }
                )

        if not optimizer_groups:
            raise ValueError("Adaptation optimizer found no trainable parameters.")

        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = getattr(self.args, "warmup_steps", None)
        if warmup_steps is None:
            warmup = int(0.03 * total_steps)
        else:
            warmup = int(warmup_steps)
        warmup = max(0, min(warmup, total_steps))

        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, total_steps - warmup))
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def initial_pair_probs_for_phase(
    phase: str,
    *,
    channel_names: t.Sequence[str],
    adapt_config: AdaptConfig,
) -> dict[Pair, float] | None:
    pairs = build_all_pairs(channel_names)
    if phase == "stage1":
        return build_new_modality_pair_probs(pairs, new_channels=adapt_config.new_channels, new_pair_ratio=1.0)
    if phase == "stage2":
        first_ratio = float(adapt_config.stage2.pair_schedule[0].new_pair_ratio)
        return build_new_modality_pair_probs(pairs, new_channels=adapt_config.new_channels, new_pair_ratio=first_ratio)
    raise ValueError(f"Unsupported adaptation phase '{phase}'.")
