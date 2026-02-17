from dataclasses import asdict
import logging
import math
import typing as t
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import wandb
import yaml

from sleep2vec import diagnostics
from sleep2vec.averagings.base import BaseModelAverager, build_model_averager
from sleep2vec.metrics import compute_downstream_metrics
from sleep2vec.visualization.layer_mix import build_layer_mix_rows, render_layer_mix_heatmap

from .downstream_model import Sleep2vecDownstreamModel
from .pretrain_model import Sleep2vecPretrainModel


class Sleep2vecFinetuning(pl.LightningModule):
    def __init__(self, args, model_config, finetune_config=None, averaging_config=None):
        super().__init__()
        self.args = args
        self.model_config = model_config
        self.finetune_config = finetune_config
        self.averaging_config = averaging_config
        self.moe_cfg = getattr(finetune_config, "moe", None) if finetune_config is not None else None

        self.backbone = Sleep2vecPretrainModel(
            channel_feature_dim=None,
            transformer_hidden_size=model_config.backbone.hidden_size,
            transformer_num_hidden_layers=model_config.backbone.num_hidden_layers,
            transformer_num_attention_heads=model_config.backbone.num_attention_heads,
            channel_names=[c.name for c in model_config.channels],
            projection=model_config.projection.enabled,
            encoder_factory=None,
            model_config=model_config,
            projection_config=model_config.projection,
            device=args.device,
        ).to(args.device)

        head_kwargs = getattr(args, "head_kwargs", None)
        self.model = Sleep2vecDownstreamModel(
            args.label_name,
            self.backbone,
            channel_names=[c.name for c in model_config.channels],
            output_dim=args.output_dim,
            is_classification=args.is_classification,
            is_seq=args.is_seq,
            head_name=getattr(args, "head_name", None),
            head_kwargs=head_kwargs,
            model_config=model_config,
            layer_mix_cfg=getattr(finetune_config, "layer_mix", None) if finetune_config else None,
            head_config=model_config.head,
            moe_finetune_cfg=self.moe_cfg,
        ).to(args.device)

        if args.pretrained_backbone_path:
            logging.info(f"loading pretrain model from {args.pretrained_backbone_path}")
            self.model.load_pretrained_backbone(args.pretrained_backbone_path)
            logging.info(f"loaded pretrain model from {args.pretrained_backbone_path}")

        if args.freeze_backbone_and_insert_lora:
            self.model.freeze_backbone_and_insert_lora(
                insert_lora=args.insert_lora,
                separate_adapters=args.separate_adapters,
            )

        if getattr(args, "freeze_tokenizer", True):
            self.backbone.set_tokenizers_trainable(False)
        else:
            self.backbone.set_tokenizers_trainable(True)
        self._apply_moe_runtime_overrides()

        self._stage_outputs = {"train": [], "val": [], "test": []}
        self._stage_group_ids = {"train": [], "val": [], "test": []}
        self._classification_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self._regression_loss = torch.nn.MSELoss()

        # Optional tensor diagnostics (borrowed from icefall)
        self._diagnostic = None
        self._diag_steps = getattr(args, "diagnostics_steps", 5)
        if getattr(args, "print_diagnostics", False):
            opts = diagnostics.TensorDiagnosticOptions(max_eig_dim=512)
            self._diagnostic = diagnostics.attach_diagnostics(self.model, opts)

        self.model_averager: BaseModelAverager | None = build_model_averager(averaging_config, self.model)
        if self.model_averager is not None:
            self.model_averager.attach_to_module(self)

    def _apply_moe_runtime_overrides(self) -> None:
        if self.moe_cfg is None:
            return
        cap_train = getattr(self.moe_cfg, "capacity_factor_train", None)
        cap_eval = getattr(self.moe_cfg, "capacity_factor_eval", None)
        if cap_train is None and cap_eval is None:
            return

        encoder_cfg = getattr(getattr(self.backbone, "encoder", None), "config", None)
        if cap_train is not None:
            if encoder_cfg is not None:
                setattr(encoder_cfg, "moe_capacity_factor_train", float(cap_train))
        if cap_eval is not None:
            if encoder_cfg is not None:
                setattr(encoder_cfg, "moe_capacity_factor_eval", float(cap_eval))

        updated_layers = 0
        for module in self.backbone.modules():
            if hasattr(module, "capacity_factor_train") and cap_train is not None:
                module.capacity_factor_train = float(cap_train)
                updated_layers += 1
            if hasattr(module, "capacity_factor_eval") and cap_eval is not None:
                module.capacity_factor_eval = float(cap_eval)
                updated_layers += 1
        logging.info(
            "Applied finetune MoE capacity overrides: train=%s eval=%s (updated modules=%s)",
            cap_train,
            cap_eval,
            updated_layers,
        )

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["model_config"] = asdict(self.model_config)
        checkpoint["model_config_yaml"] = yaml.safe_dump(checkpoint["model_config"], sort_keys=True)
        if self.finetune_config is not None:
            checkpoint["finetune_config"] = asdict(self.finetune_config)
            checkpoint["finetune_config_yaml"] = yaml.safe_dump(checkpoint["finetune_config"], sort_keys=True)

        student_layer_mix = self._layer_mix_snapshot(self.model)
        if student_layer_mix is not None:
            checkpoint["layer_mix_weights_student"] = student_layer_mix

        eval_model = self._get_eval_model()
        if eval_model is not self.model:
            eval_layer_mix = self._layer_mix_snapshot(eval_model)
            if eval_layer_mix is not None:
                checkpoint["layer_mix_weights_eval"] = eval_layer_mix

    # ---------- Lightning hooks ----------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        eval_model = self._get_eval_model()
        self._shared_step(batch, stage="val", model=eval_model)

    def test_step(self, batch, batch_idx):
        eval_model = self._get_eval_model()
        self._shared_step(batch, stage="test", model=eval_model)

    def on_train_epoch_end(self):
        self._log_layer_mix_weights(stage="train", model=self.model)
        self._finalize_epoch(stage="train")

    def on_validation_epoch_end(self):
        self._log_layer_mix_weights(stage="val", model=self._get_eval_model())
        self._finalize_epoch(stage="val")

    def on_test_epoch_end(self):
        result = self._finalize_epoch(stage="test")
        if result is None:
            return
        preds, gts = result
        trainer = getattr(self, "trainer", None)
        if self.args.is_classification and trainer is not None and trainer.is_global_zero:
            self._log_confusion_matrix(preds, gts)

    def on_fit_start(self):
        super().on_fit_start()
        if self.model_averager is not None:
            self.model_averager.on_fit_start(self.trainer)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if self.model_averager is not None:
            self.model_averager.on_load_checkpoint(checkpoint)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Allow missing/extra layer-mix weights when loading older checkpoints.
        result = super().load_state_dict(state_dict, strict=False)
        if not strict:
            return result

        allowed_prefixes = ("model.layer_mix.",)
        missing = [k for k in result.missing_keys if not k.startswith(allowed_prefixes)]
        unexpected = [k for k in result.unexpected_keys if not k.startswith(allowed_prefixes)]

        if missing or unexpected:
            raise RuntimeError(
                "Error(s) in loading state_dict: " f"missing keys={missing}, unexpected keys={unexpected}"
            )

        if result.missing_keys:
            logging.warning("Missing layer-mix keys while loading checkpoint: %s", result.missing_keys)
        if result.unexpected_keys:
            logging.warning("Unexpected layer-mix keys while loading checkpoint: %s", result.unexpected_keys)

        return result

    # ---------- Internal helpers ----------
    def _shared_step(self, batch, stage: str, model=None):
        model = model or self.model
        logits = model(batch)
        loss_info = self._compute_loss(logits, batch)
        moe_aux_loss, moe_log_values = self._compute_moe_regularization(batch=batch, model=model)
        if loss_info is None:
            if stage == "train":
                raise ValueError("No valid labels found in the current training batch.")
            valid_count = 0
            task_loss = None
            total_loss = None
        else:
            task_loss, valid_count = loss_info
            total_loss = task_loss + moe_aux_loss
            self.log(
                f"{stage}_loss",
                total_loss,
                prog_bar=True,
                sync_dist=True,
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=max(valid_count, 1),
            )
            if self.moe_cfg is not None and bool(self.moe_cfg.enable_aux_losses):
                self.log(
                    f"{stage}_task_loss",
                    task_loss,
                    prog_bar=False,
                    sync_dist=True,
                    on_step=(stage == "train"),
                    on_epoch=True,
                    batch_size=max(valid_count, 1),
                )
                self.log(
                    f"{stage}_moe_loss",
                    moe_aux_loss.detach(),
                    prog_bar=False,
                    sync_dist=True,
                    on_step=(stage == "train"),
                    on_epoch=True,
                    batch_size=max(valid_count, 1),
                )
                for key, value in moe_log_values.items():
                    if value is None:
                        continue
                    self.log(
                        f"{stage}_{key}",
                        value,
                        prog_bar=False,
                        sync_dist=True,
                        on_step=(stage == "train"),
                        on_epoch=True,
                        batch_size=max(valid_count, 1),
                    )

        preds = self._extract_valid_predictions(batch, logits)
        if preds is not None:
            self._stage_outputs[stage].append(preds)
            group_ids = self._extract_valid_group_ids(batch, logits)
            if group_ids is not None:
                self._stage_group_ids[stage].append(group_ids)

        if getattr(model, "last_moe_stats", None) is not None:
            model.last_moe_stats = self._detach_nested(model.last_moe_stats)

        return total_loss if stage == "train" else None

    def _compute_loss(self, logits, batch):
        if self.args.is_seq:
            targets = batch["tokens"][self.args.label_name].to(self.args.device)
        else:
            targets = batch["metadata"][self.args.label_name].to(self.args.device)

        if self.args.is_classification:
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1).long()
            valid_mask = targets_flat != -1
            if not valid_mask.any():
                return None
            loss = self._classification_loss(logits_flat, targets_flat)
            return loss, int(valid_mask.sum().item())

        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1).float()
        valid_mask = targets_flat != -1.0
        if not valid_mask.any():
            return None
        preds = logits_flat[valid_mask]
        valid_targets = targets_flat[valid_mask]
        loss = self._regression_loss(preds, valid_targets)
        return loss, int(valid_targets.numel())

    def _compute_moe_regularization(self, batch, model) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = logits_device = next(model.parameters()).device
        zero = torch.zeros((), dtype=torch.float32, device=logits_device)
        cfg = self.moe_cfg
        if cfg is None or not bool(cfg.enable_aux_losses):
            return zero, {}

        stats = getattr(model, "last_moe_stats", None)
        if not isinstance(stats, dict):
            return zero, {}

        router_probs_list = stats.get("router_probs", [])
        expert_indices_list = stats.get("expert_indices", [])
        dispatch_masks_list = stats.get("dispatch_mask", [])
        router_logits_list = stats.get("router_logits", [])
        dropped_masks_list = stats.get("dropped_mask", [])
        if not isinstance(router_probs_list, list) or not isinstance(expert_indices_list, list):
            return zero, {}
        if not router_probs_list or not expert_indices_list:
            return zero, {}

        num_experts = None
        merged_stats = stats.get("merged")
        if isinstance(merged_stats, dict):
            load_vec = merged_stats.get("mean/expert_load")
            if torch.is_tensor(load_vec) and load_vec.numel() > 0:
                num_experts = int(load_vec.numel())
        if num_experts is None:
            num_experts = int(router_probs_list[0].shape[-1])

        switch_losses: list[torch.Tensor] = []
        dropped_rates: list[torch.Tensor] = []
        for layer_idx, (router_probs, expert_idx) in enumerate(zip(router_probs_list, expert_indices_list)):
            if not (torch.is_tensor(router_probs) and torch.is_tensor(expert_idx)):
                continue
            dropped_mask = None
            if layer_idx < len(dropped_masks_list) and torch.is_tensor(dropped_masks_list[layer_idx]):
                dropped_mask = dropped_masks_list[layer_idx]
            dispatch_mask = None
            if layer_idx < len(dispatch_masks_list) and torch.is_tensor(dispatch_masks_list[layer_idx]):
                dispatch_mask = dispatch_masks_list[layer_idx]
            switch_losses.append(
                self._switch_aux_layer_loss(
                    router_probs=router_probs,
                    expert_indices=expert_idx,
                    dropped_mask=dropped_mask,
                    dispatch_mask=dispatch_mask,
                    num_experts=num_experts,
                )
            )
            if dropped_mask is not None:
                dropped_rates.append(dropped_mask.to(dtype=torch.float32).mean())

        if switch_losses:
            switch_loss = torch.stack([loss.to(dtype=torch.float32) for loss in switch_losses], dim=0).mean()
        else:
            switch_loss = zero

        z_losses: list[torch.Tensor] = []
        for router_logits in router_logits_list:
            if not torch.is_tensor(router_logits):
                continue
            z = torch.logsumexp(router_logits.to(dtype=torch.float32), dim=-1)
            z_losses.append(torch.mean(z.square()))
        if z_losses:
            z_loss = torch.stack(z_losses, dim=0).mean()
        else:
            z_loss = zero

        group_loss = zero
        group_cfg = cfg.group_balance
        if group_cfg is not None and bool(group_cfg.enabled) and group_cfg.weight > 0:
            group_losses: list[torch.Tensor] = []
            if group_cfg.group_key == "modality":
                by_modality = stats.get("by_modality")
                if isinstance(by_modality, dict) and by_modality:
                    modality_names = sorted(by_modality.keys())
                    max_layers = max(
                        len(mod_stats.get("router_probs", [])) for mod_stats in by_modality.values() if isinstance(mod_stats, dict)
                    )
                    for layer_idx in range(max_layers):
                        probs_parts: list[torch.Tensor] = []
                        idx_parts: list[torch.Tensor] = []
                        drop_parts: list[torch.Tensor] = []
                        disp_parts: list[torch.Tensor] = []
                        gid_parts: list[torch.Tensor] = []
                        for gid, modality in enumerate(modality_names):
                            mod_stats = by_modality.get(modality, {})
                            probs_list = mod_stats.get("router_probs", [])
                            idx_list = mod_stats.get("expert_indices", [])
                            if layer_idx >= len(probs_list) or layer_idx >= len(idx_list):
                                continue
                            probs = probs_list[layer_idx]
                            idx = idx_list[layer_idx]
                            if not (torch.is_tensor(probs) and torch.is_tensor(idx)):
                                continue
                            drop_list = mod_stats.get("dropped_mask", [])
                            if layer_idx < len(drop_list) and torch.is_tensor(drop_list[layer_idx]):
                                dropped = drop_list[layer_idx]
                            else:
                                dropped = torch.zeros(
                                    probs.shape[0],
                                    probs.shape[1],
                                    dtype=torch.bool,
                                    device=probs.device,
                                )
                            disp_list = mod_stats.get("dispatch_mask", [])
                            if layer_idx < len(disp_list) and torch.is_tensor(disp_list[layer_idx]):
                                dispatched = disp_list[layer_idx]
                            else:
                                dispatched = torch.ones(
                                    probs.shape[0],
                                    probs.shape[1],
                                    idx.shape[-1] if idx.dim() == 3 else 1,
                                    dtype=torch.bool,
                                    device=probs.device,
                                )
                            probs_parts.append(probs)
                            idx_parts.append(idx)
                            drop_parts.append(dropped)
                            disp_parts.append(dispatched)
                            gid_parts.append(torch.full((probs.shape[0],), gid, dtype=torch.long, device=probs.device))
                        if probs_parts:
                            group_losses.append(
                                self._group_balance_layer_loss(
                                    router_probs=torch.cat(probs_parts, dim=0),
                                    expert_indices=torch.cat(idx_parts, dim=0),
                                    dropped_mask=torch.cat(drop_parts, dim=0),
                                    dispatch_mask=torch.cat(disp_parts, dim=0),
                                    group_ids=torch.cat(gid_parts, dim=0),
                                    num_experts=num_experts,
                                    loss_type=group_cfg.loss_type,
                                    min_group_size=int(group_cfg.min_group_size),
                                )
                            )
            else:
                group_ids = self._resolve_group_ids(batch, group_cfg.group_key, device=device)
                if group_ids is not None:
                    for layer_idx, (router_probs, expert_idx) in enumerate(zip(router_probs_list, expert_indices_list)):
                        if not (torch.is_tensor(router_probs) and torch.is_tensor(expert_idx)):
                            continue
                        dropped_mask = None
                        if layer_idx < len(dropped_masks_list) and torch.is_tensor(dropped_masks_list[layer_idx]):
                            dropped_mask = dropped_masks_list[layer_idx]
                        dispatch_mask = None
                        if layer_idx < len(dispatch_masks_list) and torch.is_tensor(dispatch_masks_list[layer_idx]):
                            dispatch_mask = dispatch_masks_list[layer_idx]
                        group_losses.append(
                            self._group_balance_layer_loss(
                                router_probs=router_probs,
                                expert_indices=expert_idx,
                                dropped_mask=dropped_mask,
                                dispatch_mask=dispatch_mask,
                                group_ids=group_ids.to(device=router_probs.device),
                                num_experts=num_experts,
                                loss_type=group_cfg.loss_type,
                                min_group_size=int(group_cfg.min_group_size),
                            )
                        )
            if group_losses:
                group_loss = torch.stack([loss.to(dtype=torch.float32) for loss in group_losses], dim=0).mean()

        total_loss = cfg.aux_loss_weight * switch_loss + cfg.router_z_loss_weight * z_loss
        if group_cfg is not None and bool(group_cfg.enabled):
            total_loss = total_loss + group_cfg.weight * group_loss

        log_values: dict[str, torch.Tensor] = {
            "moe_switch_aux_loss": switch_loss.detach(),
            "moe_router_z_loss": z_loss.detach(),
            "moe_group_loss": group_loss.detach(),
        }
        if dropped_rates:
            log_values["moe_dropped_rate"] = torch.stack(dropped_rates, dim=0).mean().detach()
        if isinstance(merged_stats, dict):
            for src_key, dst_key in (
                ("mean/load_cv2", "moe_load_cv2"),
                ("mean/importance_cv2", "moe_importance_cv2"),
                ("mean/lb_loss_raw", "moe_lb_loss_raw"),
            ):
                value = merged_stats.get(src_key)
                if torch.is_tensor(value) and value.numel() == 1:
                    log_values[dst_key] = value.detach()
        model.last_moe_stats = self._detach_nested(stats)
        return total_loss, log_values

    @staticmethod
    def _detach_nested(obj):
        if torch.is_tensor(obj):
            return obj.detach()
        if isinstance(obj, dict):
            return {key: Sleep2vecFinetuning._detach_nested(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [Sleep2vecFinetuning._detach_nested(value) for value in obj]
        if isinstance(obj, tuple):
            return tuple(Sleep2vecFinetuning._detach_nested(value) for value in obj)
        return obj

    @staticmethod
    def _switch_aux_layer_loss(
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        dropped_mask: torch.Tensor | None,
        dispatch_mask: torch.Tensor | None,
        num_experts: int,
    ) -> torch.Tensor:
        if expert_indices.dim() == 2:
            expert_ids = expert_indices.unsqueeze(-1)
        elif expert_indices.dim() == 3:
            expert_ids = expert_indices
        else:
            raise ValueError(f"expert_indices must be [B,T] or [B,T,K], got {tuple(expert_indices.shape)}")
        bsz, seq_len, top_k = expert_ids.shape
        expert_ids = expert_ids.to(dtype=torch.long).clamp(min=0, max=num_experts - 1)
        one_hot = F.one_hot(expert_ids, num_classes=num_experts).to(dtype=router_probs.dtype)

        if dispatch_mask is not None:
            if dispatch_mask.dim() == 2:
                valid_assign = dispatch_mask.to(dtype=torch.bool).unsqueeze(-1).expand(bsz, seq_len, top_k)
            elif dispatch_mask.dim() == 3:
                valid_assign = dispatch_mask.to(dtype=torch.bool)
                if valid_assign.shape[-1] != top_k:
                    raise ValueError(
                        f"dispatch_mask top_k mismatch: mask={tuple(valid_assign.shape)} idx={tuple(expert_ids.shape)}"
                    )
            else:
                raise ValueError(f"dispatch_mask must be [B,T] or [B,T,K], got {tuple(dispatch_mask.shape)}")
        elif dropped_mask is not None:
            if dropped_mask.dim() == 2:
                valid_assign = (~dropped_mask.to(dtype=torch.bool)).unsqueeze(-1).expand(bsz, seq_len, top_k)
            elif dropped_mask.dim() == 3:
                valid_assign = ~dropped_mask.to(dtype=torch.bool)
                if valid_assign.shape[-1] != top_k:
                    raise ValueError(
                        f"dropped_mask top_k mismatch: mask={tuple(valid_assign.shape)} idx={tuple(expert_ids.shape)}"
                    )
            else:
                raise ValueError(f"dropped_mask must be [B,T] or [B,T,K], got {tuple(dropped_mask.shape)}")
        else:
            valid_assign = torch.ones((bsz, seq_len, top_k), dtype=torch.bool, device=expert_ids.device)

        one_hot_flat = one_hot.reshape(-1, num_experts)
        valid_flat = valid_assign.reshape(-1)
        if valid_flat.any():
            f = one_hot_flat[valid_flat].mean(dim=0)
        else:
            f = one_hot_flat.new_zeros((num_experts,))
        g = router_probs.to(dtype=torch.float32).mean(dim=(0, 1))
        return float(num_experts) * torch.sum(f.to(dtype=torch.float32) * g)

    @staticmethod
    def _group_balance_layer_loss(
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        dropped_mask: torch.Tensor | None,
        dispatch_mask: torch.Tensor | None,
        group_ids: torch.Tensor,
        num_experts: int,
        loss_type: str,
        min_group_size: int,
    ) -> torch.Tensor:
        device = router_probs.device
        dtype = router_probs.dtype
        if group_ids.dim() != 1 or group_ids.shape[0] != router_probs.shape[0]:
            return router_probs.new_zeros((), dtype=torch.float32)

        if expert_indices.dim() == 2:
            expert_ids = expert_indices.unsqueeze(-1)
        elif expert_indices.dim() == 3:
            expert_ids = expert_indices
        else:
            raise ValueError(f"expert_indices must be [B,T] or [B,T,K], got {tuple(expert_indices.shape)}")
        bsz, seq_len, top_k = expert_ids.shape
        expert_ids = expert_ids.to(dtype=torch.long).clamp(min=0, max=num_experts - 1)
        one_hot = F.one_hot(expert_ids, num_classes=num_experts).to(dtype=dtype)
        valid_group = group_ids >= 0
        if not valid_group.any():
            return router_probs.new_zeros((), dtype=torch.float32)

        if dispatch_mask is not None:
            if dispatch_mask.dim() == 2:
                valid_assign = dispatch_mask.to(dtype=torch.bool).unsqueeze(-1).expand(bsz, seq_len, top_k)
            elif dispatch_mask.dim() == 3:
                valid_assign = dispatch_mask.to(dtype=torch.bool)
                if valid_assign.shape[-1] != top_k:
                    raise ValueError(
                        f"dispatch_mask top_k mismatch: mask={tuple(valid_assign.shape)} idx={tuple(expert_ids.shape)}"
                    )
            else:
                raise ValueError(f"dispatch_mask must be [B,T] or [B,T,K], got {tuple(dispatch_mask.shape)}")
        elif dropped_mask is None:
            valid_assign = torch.ones((bsz, seq_len, top_k), dtype=torch.bool, device=device)
        else:
            if dropped_mask.dim() == 2:
                valid_assign = (~dropped_mask.to(dtype=torch.bool, device=device)).unsqueeze(-1).expand(bsz, seq_len, top_k)
            elif dropped_mask.dim() == 3:
                valid_assign = ~dropped_mask.to(dtype=torch.bool, device=device)
                if valid_assign.shape[-1] != top_k:
                    raise ValueError(
                        f"dropped_mask top_k mismatch: mask={tuple(valid_assign.shape)} idx={tuple(expert_ids.shape)}"
                    )
            else:
                raise ValueError(f"dropped_mask must be [B,T] or [B,T,K], got {tuple(dropped_mask.shape)}")

        valid_assign = valid_assign & valid_group.unsqueeze(1).unsqueeze(2)
        if not valid_assign.any():
            return router_probs.new_zeros((), dtype=torch.float32)

        unique_groups, sample_counts = torch.unique(group_ids[valid_group], return_counts=True)
        candidate_groups = unique_groups[sample_counts >= int(min_group_size)]
        if candidate_groups.numel() == 0:
            return router_probs.new_zeros((), dtype=torch.float32)

        global_f = one_hot[valid_assign].mean(dim=0).to(dtype=torch.float32)
        eps = 1e-8
        weighted_losses: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []

        for gid in candidate_groups.tolist():
            assign_mask = valid_assign & (group_ids.unsqueeze(1).unsqueeze(2) == int(gid))
            if not assign_mask.any():
                continue
            f_g = one_hot[assign_mask].mean(dim=0).to(dtype=torch.float32)
            token_mask = assign_mask.any(dim=-1)
            g_g = router_probs[token_mask].mean(dim=0).to(dtype=torch.float32)
            if loss_type == "group_to_global_l2":
                group_loss = torch.sum((f_g - global_f) ** 2)
            elif loss_type == "group_to_global_kl":
                group_loss = torch.sum((f_g + eps) * torch.log((f_g + eps) / (global_f + eps)))
            else:
                group_loss = float(num_experts) * torch.sum(f_g * g_g)
            assign_count = assign_mask.to(dtype=torch.float32).sum()
            weighted_losses.append(group_loss * assign_count)
            weights.append(assign_count)

        if not weighted_losses:
            return router_probs.new_zeros((), dtype=torch.float32)
        return torch.stack(weighted_losses).sum() / torch.stack(weights).sum().clamp_min(1.0)

    def _resolve_group_ids(self, batch, group_key: str, device: torch.device) -> torch.Tensor | None:
        metadata = batch.get("metadata") if isinstance(batch, dict) else None
        if not isinstance(metadata, dict):
            return None
        if group_key == "source":
            source = metadata.get("source")
            if source is None:
                return None
            ctx_encoder = getattr(self.backbone, "router_ctx_encoder", None)
            if ctx_encoder is not None and hasattr(ctx_encoder, "encode_source_ids"):
                batch_size = int(batch["length"].shape[0])
                return ctx_encoder.encode_source_ids(source, device=device, batch_size=batch_size)
            if isinstance(source, (list, tuple)):
                encoded: list[int] = []
                for value in source:
                    if value is None:
                        encoded.append(0)
                        continue
                    src = str(value).strip().lower()
                    if src in {"", "none", "nan"}:
                        encoded.append(0)
                        continue
                    try:
                        numeric = int(float(value))
                    except (TypeError, ValueError):
                        encoded.append(int(zlib.crc32(src.encode("utf-8"))))
                        continue
                    encoded.append(max(numeric, 0))
                return torch.tensor(encoded, dtype=torch.long, device=device)
            if torch.is_tensor(source):
                return source.to(device=device, dtype=torch.long)
            return None
        if group_key == "sex":
            sex = metadata.get("sex")
            if sex is None:
                return None
            return torch.as_tensor(sex, dtype=torch.long, device=device)
        if group_key == "age_bin":
            age = metadata.get("age")
            if age is None:
                return None
            age_tensor = torch.as_tensor(age, dtype=torch.float32, device=device)
            encoder_cfg = getattr(getattr(self.backbone, "encoder", None), "config", None)
            age_bins = int(getattr(encoder_cfg, "moe_age_bins", 10))
            valid = age_tensor >= 0
            bins = torch.full(age_tensor.shape, -1, dtype=torch.long, device=device)
            if valid.any():
                bins[valid] = torch.clamp((age_tensor[valid] / 100.0 * age_bins).long(), min=0, max=age_bins - 1)
            return bins
        return None

    def _extract_valid_group_ids(self, batch, logits) -> np.ndarray | None:
        if self.args.is_seq:
            labels = batch["tokens"][self.args.label_name].to(self.args.device).view(-1)
            valid_mask = labels != -1
        elif self.args.is_classification:
            labels = batch["metadata"][self.args.label_name].to(self.args.device).view(-1)
            valid_mask = labels != -1
        else:
            labels = batch["metadata"][self.args.label_name].to(self.args.device).view(-1).float()
            valid_mask = labels != -1.0
        if not valid_mask.any():
            return None

        group_key = "source"
        if self.moe_cfg is not None and self.moe_cfg.group_balance is not None:
            group_key = self.moe_cfg.group_balance.group_key
        if group_key == "modality":
            return None

        group_ids = self._resolve_group_ids(batch, group_key, device=labels.device)
        if group_ids is None:
            return None
        if self.args.is_seq:
            if group_ids.dim() == 1:
                seq_len = int(logits.shape[1]) if logits.dim() >= 2 else 1
                group_ids = group_ids.unsqueeze(1).expand(-1, seq_len).reshape(-1)
        else:
            group_ids = group_ids.view(-1)
        if group_ids.numel() != valid_mask.numel():
            return None
        return group_ids[valid_mask].detach().cpu().numpy()

    def _extract_valid_predictions(self, batch, logits):
        if self.args.is_seq:
            labels = batch["tokens"][self.args.label_name].to(self.args.device)
        else:
            labels = batch["metadata"][self.args.label_name].to(self.args.device)

        if self.args.is_classification:
            if logits.dim() == 3:
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
            else:
                labels = labels.view(-1)

            mask = labels != -1
            if not mask.any():
                return None

            probs = torch.softmax(logits[mask], dim=-1).detach().cpu().numpy()
            labels_np = labels[mask].detach().cpu().numpy()
            return probs, labels_np

        logits = logits.view(-1)
        labels = labels.view(-1).float()
        mask = labels != -1.0
        if not mask.any():
            return None

        preds = logits[mask].to(torch.float32).detach().cpu().numpy()
        labels_np = labels[mask].to(torch.float32).detach().cpu().numpy()
        return preds, labels_np

    @staticmethod
    def _layer_mix_snapshot(model: torch.nn.Module):
        getter = getattr(model, "layer_mix_snapshot", None)
        if not callable(getter):
            return None
        return getter()

    def _log_layer_mix_weights(self, stage: str, model: torch.nn.Module) -> None:
        snapshot = self._layer_mix_snapshot(model)
        if snapshot is None:
            return

        trainer = getattr(self, "trainer", None)
        if trainer is not None and not trainer.is_global_zero:
            return
        if getattr(wandb, "run", None) is None:
            return

        layer_ids = [int(v) for v in snapshot.get("layer_indices", [])]
        effective = snapshot.get("effective_by_modality", {})
        shared = bool(snapshot.get("shared_across_modalities", False))
        if not layer_ids or not isinstance(effective, dict) or not effective:
            return

        modality_names = list(effective.keys())
        matrix_rows: list[list[float]] = []
        for modality in modality_names:
            mod_info = effective.get(modality, {})
            weights = mod_info.get("layer_weights", []) if isinstance(mod_info, dict) else []
            if len(weights) < len(layer_ids):
                logging.warning(
                    "Skipping layer-mix visualization for stage=%s due to malformed weights for modality=%s.",
                    stage,
                    modality,
                )
                return
            matrix_rows.append([float(weights[idx]) for idx in range(len(layer_ids))])

        matrix = np.array(matrix_rows, dtype=np.float32)
        title = (
            f"{stage.title()} Layer-Mix Weights (epoch {self.current_epoch}, "
            f"{'shared' if shared else 'per-modality'})"
        )
        fig = render_layer_mix_heatmap(matrix, modality_names, layer_ids, title=title)
        rows = build_layer_mix_rows(
            stage=stage,
            epoch=int(self.current_epoch),
            shared=shared,
            layer_ids=layer_ids,
            effective_by_modality=effective,
        )
        columns = [
            "stage",
            "epoch",
            "modality",
            "layer_id",
            "weight",
            "shared_across_modalities",
            "row_name",
            "row_index",
        ]
        table = wandb.Table(columns=columns, data=[[row[col] for col in columns] for row in rows])

        wandb.log(
            {
                f"{stage}_layer_mix/heatmap": wandb.Image(fig),
                f"{stage}_layer_mix/table": table,
            },
            commit=False,
        )
        plt.close(fig)

    def _finalize_epoch(self, stage: str):
        outputs = self._stage_outputs[stage]
        if not outputs:
            return None

        preds, gts = zip(*outputs)
        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)

        metrics = compute_downstream_metrics(
            gts,
            preds,
            is_classification=self.args.is_classification,
            output_dim=getattr(self.args, "output_dim", None),
            stage_names=getattr(self.args, "stage_names", None),
        )
        for k, v in metrics.items():
            self.log(
                f"{stage}_{k}",
                v,
                prog_bar=(stage != "train"),
                logger=True,
                sync_dist=True,
                on_epoch=True,
            )

        stage_group_ids = self._stage_group_ids.get(stage, [])
        if stage_group_ids:
            groups = np.concatenate(stage_group_ids, axis=0)
            if groups.shape[0] == gts.shape[0]:
                group_metric_name = str(getattr(self.args, "monitor", ""))
                monitor_prefix = f"{stage}_"
                if group_metric_name.startswith(monitor_prefix):
                    group_metric_name = group_metric_name[len(monitor_prefix) :]
                if group_metric_name not in metrics:
                    fallback = ["accuracy", "macro_f1", "mae", "rmse", "mse", "r2"]
                    group_metric_name = next((name for name in fallback if name in metrics), None)

                if group_metric_name is not None:
                    unique_groups = np.unique(groups)
                    group_values: list[float] = []
                    for group_id in unique_groups.tolist():
                        mask = groups == group_id
                        if not np.any(mask):
                            continue
                        group_metrics = compute_downstream_metrics(
                            gts[mask],
                            preds[mask],
                            is_classification=self.args.is_classification,
                            output_dim=getattr(self.args, "output_dim", None),
                            stage_names=getattr(self.args, "stage_names", None),
                        )
                        if group_metric_name not in group_metrics:
                            continue
                        group_value = float(group_metrics[group_metric_name])
                        group_values.append(group_value)
                        self.log(
                            f"{stage}_group_{group_metric_name}_{group_id}",
                            group_value,
                            prog_bar=False,
                            logger=True,
                            sync_dist=True,
                            on_epoch=True,
                        )

                    if group_values:
                        values = np.array(group_values, dtype=np.float32)
                        maximize = str(getattr(self.args, "monitor_mod", "max")).lower() == "max"
                        avg_value = float(values.mean())
                        worst_value = float(values.min() if maximize else values.max())
                        best_value = float(values.max() if maximize else values.min())
                        self.log(
                            f"{stage}_avg_group_{group_metric_name}",
                            avg_value,
                            prog_bar=False,
                            logger=True,
                            sync_dist=True,
                            on_epoch=True,
                        )
                        self.log(
                            f"{stage}_worst_group_{group_metric_name}",
                            worst_value,
                            prog_bar=False,
                            logger=True,
                            sync_dist=True,
                            on_epoch=True,
                        )
                        self.log(
                            f"{stage}_best_group_{group_metric_name}",
                            best_value,
                            prog_bar=False,
                            logger=True,
                            sync_dist=True,
                            on_epoch=True,
                        )

        outputs.clear()
        stage_group_ids.clear()
        return preds, gts

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        if self.model_averager is not None:
            self.model_averager.on_train_batch_end(trainer=self.trainer, global_step=self.global_step)
        if self._diagnostic is not None and self.global_step >= self._diag_steps:
            if self.trainer is not None:
                self.trainer.should_stop = True

    def on_train_end(self):
        super().on_train_end()
        if self._diagnostic is not None:
            self._diagnostic.print_diagnostics()

    def _get_eval_model(self):
        if self.model_averager is not None:
            return self.model_averager.eval_model()
        return self.model

    def _log_confusion_matrix(self, preds: np.ndarray, gts: np.ndarray):
        if getattr(wandb, "run", None) is None:
            return
        pred_labels = preds.argmax(axis=1)
        cm = confusion_matrix(gts, pred_labels)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Test Confusion Matrix (epoch {self.current_epoch})")
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

    def configure_optimizers(self):
        moe_cfg = self.moe_cfg
        router_lr_mult = float(getattr(moe_cfg, "router_lr_mult", 1.0)) if moe_cfg is not None else 1.0
        experts_lr_mult = float(getattr(moe_cfg, "experts_lr_mult", 1.0)) if moe_cfg is not None else 1.0

        def _decay_flag(name: str, param: torch.Tensor) -> bool:
            return param.ndim >= 2 and ("norm" not in name.lower()) and ("bias" not in name.lower())

        def _group_name(param_name: str) -> str:
            if param_name.startswith("head."):
                return "head"
            if "lora_" in param_name:
                return "lora"
            if param_name.startswith("backbone."):
                inner_name = param_name[len("backbone.") :]
                if self.model.is_router_parameter_name(inner_name):
                    return "router"
                if self.model.is_expert_parameter_name(inner_name):
                    return "experts"
            return "other"

        lr_by_group = {
            "head": float(self.args.lr),
            "lora": float(self.args.lr),
            "router": float(self.args.lr) * router_lr_mult,
            "experts": float(self.args.lr) * experts_lr_mult,
            "other": float(self.args.lr),
        }
        bucketed: dict[tuple[str, str], list[torch.Tensor]] = {}
        bucket_counts: dict[tuple[str, str], int] = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            group = _group_name(name)
            decay = "decay" if _decay_flag(name, param) else "no_decay"
            key = (group, decay)
            bucketed.setdefault(key, []).append(param)
            bucket_counts[key] = bucket_counts.get(key, 0) + int(param.numel())

        param_groups = []
        for (group, decay), params in bucketed.items():
            if not params:
                continue
            weight_decay = float(self.args.weight_decay) if decay == "decay" else 0.0
            param_groups.append(
                {
                    "params": params,
                    "weight_decay": weight_decay,
                    "lr": lr_by_group[group],
                    "group_name": group,
                    "decay_name": decay,
                }
            )
            logging.info(
                "[optimizer] group=%s decay=%s params=%s lr=%.6e wd=%.6e",
                group,
                decay,
                bucket_counts[(group, decay)],
                lr_by_group[group],
                weight_decay,
            )

        if not param_groups:
            raise ValueError("No trainable parameters found for optimizer construction.")

        optimizer = torch.optim.AdamW(
            param_groups,
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
