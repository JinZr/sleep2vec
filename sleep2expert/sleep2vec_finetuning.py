from dataclasses import asdict
import logging

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import wandb
import yaml

from sleep2expert import diagnostics
from sleep2expert.averagings.base import BaseModelAverager, build_model_averager
from sleep2expert.common import remap_stage_labels
from sleep2expert.distributed import get_rank_world_size, is_torch_distributed_ready
from sleep2expert.losses.cox import CoxPHLossVectorized
from sleep2expert.losses.moe_regularization import compute_downstream_moe_metrics, compute_downstream_moe_regularization
from sleep2expert.metrics import (
    AHI_FINE_THRESHOLD_GRID,
    _aggregate_prepared_ahi_records,
    _compute_ahi_event_metrics_from_prepared,
    _prepare_ahi_records,
    compute_downstream_metrics,
    extract_ahi_summary_scatter_arrays,
)
from sleep2expert.schedulers import build_warmup_cosine_scheduler
from sleep2expert.sleep2vec_inference import (
    build_ahi_prediction_rows,
    build_prediction_rows,
    extract_prediction_records,
    prediction_export_enabled,
)
from sleep2expert.visualization.downstream_eval import DownstreamEvalVisualizer
from sleep2expert.visualization.layer_mix import build_layer_mix_rows, render_layer_mix_heatmap

from .downstream_model import Sleep2vecDownstreamModel
from .pretrain_model import Sleep2vecPretrainModel


class Sleep2vecFinetuning(pl.LightningModule):
    def __init__(self, args, model_config, finetune_config=None, averaging_config=None):
        super().__init__()
        self.args = args
        self.model_config = model_config
        self.finetune_config = finetune_config
        self.averaging_config = averaging_config

        self.backbone = Sleep2vecPretrainModel(
            model_config=model_config,
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
        ).to(args.device)

        if args.pretrained_backbone_path:
            logging.info(f"loading pretrain model from {args.pretrained_backbone_path}")
            self.model.load_pretrained_backbone(args.pretrained_backbone_path)
            logging.info(f"loaded pretrain model from {args.pretrained_backbone_path}")

        if args.freeze_backbone_and_insert_lora:
            self.model.freeze_backbone_and_insert_lora(
                insert_lora=args.insert_lora,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                use_dora=args.lora_use_dora,
                separate_adapters=args.separate_adapters,
            )

        if getattr(args, "freeze_tokenizer", True):
            self.backbone.set_tokenizers_trainable(False)
        else:
            self.backbone.set_tokenizers_trainable(True)

        self._finetune_param_to_group: dict[str, str] = {}
        self._finetune_group_summary: dict[str, dict[str, int]] = {}
        self._finetune_lr_scales: dict[str, float] = {}
        moe_tuning = getattr(finetune_config, "moe_tuning", None) if finetune_config is not None else None
        if moe_tuning is not None:
            self._apply_moe_tuning_policy()
            moe_reg = moe_tuning.moe_regularization
            self.model.collect_train_moe_aux = bool(
                getattr(moe_reg, "enabled", False) and getattr(moe_reg, "collect_train_moe_aux", False)
            )
        self.moe_finetune_status = self._build_moe_finetune_status()

        self._stage_outputs = {"train": [], "val": [], "test": []}
        self._prediction_records = {"test": []}
        self.prediction_rows = []
        class_weights = getattr(args, "class_weights", None)
        class_weight_tensor = None
        if class_weights is not None:
            class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=args.device)
        pos_weight = getattr(args, "pos_weight", None)
        pos_weight_tensor = None
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=args.device)
        self._classification_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=class_weight_tensor)
        self._multilabel_loss = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight_tensor)
        self._regression_loss = torch.nn.MSELoss()
        self._survival_loss = CoxPHLossVectorized()
        self._ahi_eval_threshold: float | None = None
        self._ahi_train_pointwise_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        self._eval_loss_sums = {"val": 0.0, "test": 0.0}
        self._eval_loss_counts = {"val": 0, "test": 0}

        # Optional tensor diagnostics (borrowed from icefall)
        self._diagnostic = None
        self._diag_steps = getattr(args, "diagnostics_steps", 5)
        if getattr(args, "print_diagnostics", False):
            opts = diagnostics.TensorDiagnosticOptions(max_eig_dim=512)
            self._diagnostic = diagnostics.attach_diagnostics(self.model, opts)

        self.model_averager: BaseModelAverager | None = build_model_averager(averaging_config, self.model)
        if self.model_averager is not None:
            self.model_averager.attach_to_module(self)
        self._eval_visualizer = DownstreamEvalVisualizer(
            getattr(finetune_config, "eval_visualizations", None) if finetune_config is not None else None
        )

    def _apply_moe_tuning_policy(self):
        cfg = self.finetune_config.moe_tuning
        lr_scales = cfg.lr_scales
        self._finetune_lr_scales = {
            "head": float(lr_scales.head),
            "backbone": float(lr_scales.backbone),
            "experts": float(lr_scales.experts),
            "routers": float(lr_scales.routers),
            "tokenizers": float(lr_scales.tokenizers),
            "projection": float(lr_scales.projection),
            "lora": float(lr_scales.lora),
        }
        self._set_param_trainability_from_policy(cfg)
        self._log_finetune_param_group_summary()

    def _semantic_group_for_param(self, name: str) -> str:
        if "lora_" in name:
            return "lora"
        if name.startswith(("head.", "temporal_agg.", "layer_mix.")):
            return "head"
        if name.startswith("backbone.tokenizer_mapping."):
            return "tokenizers"
        if ".moe_ffn.router." in name:
            return "routers"
        if ".moe_ffn.experts." in name:
            return "experts"
        if name.startswith(("backbone.proj_head.", "backbone.mask_embed.")):
            return "projection"
        if name.startswith("backbone."):
            return "backbone"
        return "head"

    def _set_param_trainability_from_policy(self, cfg):
        groups = ("head", "backbone", "experts", "routers", "tokenizers", "projection", "lora")
        self._finetune_param_to_group = {}
        self._finetune_group_summary = {
            group: {"total_params": 0, "trainable_params": 0, "total_tensors": 0, "trainable_tensors": 0}
            for group in groups
        }
        selected_layers = set(cfg.train_moe_layer_indices or [])
        selected_expert_param_count = 0

        def expert_layer_idx(name: str) -> int | None:
            marker = ".encoder.layer."
            marker_pos = name.find(marker)
            if marker_pos < 0:
                return None
            idx_text, _, suffix = name[marker_pos + len(marker) :].partition(".")
            if not suffix.startswith("moe_ffn.experts."):
                return None
            try:
                return int(idx_text) + 1
            except ValueError:
                return None

        def is_selected_expert(name: str) -> bool:
            layer_idx = expert_layer_idx(name)
            return layer_idx is not None and layer_idx in selected_layers

        for name, param in self.model.named_parameters():
            group = self._semantic_group_for_param(name)
            if group not in self._finetune_lr_scales:
                raise ValueError(f"Unknown finetune parameter group '{group}' for parameter '{name}'.")
            self._finetune_param_to_group[name] = group

            trainable = False
            if group == "lora":
                trainable = self._finetune_lr_scales[group] > 0.0
            elif cfg.mode == "head_only":
                trainable = group == "head"
            elif cfg.mode == "conservative_full_router_frozen":
                trainable = group in {"head", "backbone", "experts"}
            elif cfg.mode == "conservative_full_router_trainable":
                trainable = group in {"head", "backbone", "experts", "routers"}
            elif cfg.mode == "top_moe_layer_expert_only":
                trainable = group == "head" or (group == "experts" and is_selected_expert(name))
            elif cfg.mode == "custom":
                trainable = self._finetune_lr_scales[group] > 0.0
                if group == "routers" and cfg.freeze_router:
                    trainable = False
                if group == "experts" and cfg.freeze_experts:
                    trainable = False
            else:
                raise ValueError(f"Unsupported finetune MoE tuning mode: {cfg.mode}")

            if group == "tokenizers" and getattr(self.args, "freeze_tokenizer", True):
                trainable = False
            if self._finetune_lr_scales[group] == 0.0:
                trainable = False

            param.requires_grad = trainable
            param_count = int(param.numel())
            summary = self._finetune_group_summary[group]
            summary["total_params"] += param_count
            summary["total_tensors"] += 1
            if trainable:
                summary["trainable_params"] += param_count
                summary["trainable_tensors"] += 1
            if cfg.mode == "top_moe_layer_expert_only" and group == "experts" and is_selected_expert(name):
                selected_expert_param_count += param_count

        if cfg.mode == "head_only":
            offenders = [
                name
                for name, param in self.model.named_parameters()
                if name.startswith("backbone.")
                and param.requires_grad
                and self._semantic_group_for_param(name) != "lora"
            ]
            if offenders:
                raise ValueError(f"head_only MoE tuning left backbone parameters trainable: {offenders[:5]}")
            if self._finetune_group_summary["head"]["trainable_params"] == 0:
                raise ValueError("head_only MoE tuning found no trainable head parameters.")

        if cfg.mode == "conservative_full_router_frozen":
            offenders = [
                name
                for name, param in self.model.named_parameters()
                if ".moe_ffn.router." in name and param.requires_grad
            ]
            if offenders:
                raise ValueError(f"conservative_full_router_frozen left router parameters trainable: {offenders[:5]}")

        if cfg.mode == "top_moe_layer_expert_only":
            if not selected_layers:
                raise ValueError("top_moe_layer_expert_only requires train_moe_layer_indices.")
            if selected_expert_param_count == 0:
                raise ValueError(
                    "top_moe_layer_expert_only found no expert parameters for "
                    f"train_moe_layer_indices={sorted(selected_layers)}."
                )
            offenders = [
                name
                for name, param in self.model.named_parameters()
                if name.startswith("backbone.")
                and param.requires_grad
                and self._semantic_group_for_param(name) != "lora"
                and not is_selected_expert(name)
            ]
            if offenders:
                raise ValueError(
                    "top_moe_layer_expert_only left non-selected backbone parameters " f"trainable: {offenders[:5]}"
                )

    def _log_finetune_param_group_summary(self):
        cfg = self.finetune_config.moe_tuning
        total_params = sum(summary["total_params"] for summary in self._finetune_group_summary.values())
        trainable_params = sum(summary["trainable_params"] for summary in self._finetune_group_summary.values())
        logging.info("[finetune_moe_tuning] mode=%s", cfg.mode)
        for group in ("head", "backbone", "experts", "routers", "tokenizers", "projection", "lora"):
            summary = self._finetune_group_summary[group]
            logging.info(
                "[finetune_moe_tuning] group=%s trainable_params=%s total_params=%s lr_scale=%s",
                group,
                summary["trainable_params"],
                summary["total_params"],
                self._finetune_lr_scales[group],
            )
        logging.info(
            "[finetune_moe_tuning] trainable_params=%s total_params=%s",
            trainable_params,
            total_params,
        )

    def _build_moe_finetune_status(self) -> dict:
        moe_cfg = getattr(self.model_config.backbone, "moe", None)
        moe_tuning = getattr(self.finetune_config, "moe_tuning", None) if self.finetune_config is not None else None
        if moe_tuning is None:
            param_groups = {"legacy": self._legacy_param_group_summary()}
            lr_scales = {}
            moe_regularization = {}
            mode = None
        else:
            param_groups = {
                group: {
                    "total_params": int(summary["total_params"]),
                    "trainable_params": int(summary["trainable_params"]),
                    "total_tensors": int(summary["total_tensors"]),
                    "trainable_tensors": int(summary["trainable_tensors"]),
                    "lr_scale": float(self._finetune_lr_scales[group]),
                }
                for group, summary in self._finetune_group_summary.items()
            }
            lr_scales = dict(self._finetune_lr_scales)
            moe_regularization = asdict(moe_tuning.moe_regularization)
            mode = moe_tuning.mode

        return {
            "moe_enabled": bool(moe_cfg is not None and getattr(moe_cfg, "enabled", False)),
            "moe_layer_indices": list(getattr(moe_cfg, "layer_indices", None) or []),
            "num_experts": getattr(moe_cfg, "num_experts", None),
            "top_k": getattr(moe_cfg, "top_k", None),
            "router_type": getattr(moe_cfg, "router_type", None),
            "moe_tuning_present": moe_tuning is not None,
            "moe_tuning_mode": mode,
            "lr_scales": lr_scales,
            "collect_train_moe_aux": bool(getattr(self.model, "collect_train_moe_aux", False)),
            "moe_regularization": moe_regularization,
            "param_groups": param_groups,
            "total_params": int(sum(group["total_params"] for group in param_groups.values())),
            "trainable_params": int(sum(group["trainable_params"] for group in param_groups.values())),
        }

    def _legacy_param_group_summary(self) -> dict[str, int | float]:
        summary = {
            "total_params": 0,
            "trainable_params": 0,
            "total_tensors": 0,
            "trainable_tensors": 0,
            "lr_scale": 1.0,
        }
        for _, param in self.model.named_parameters():
            param_count = int(param.numel())
            summary["total_params"] += param_count
            summary["total_tensors"] += 1
            if param.requires_grad:
                summary["trainable_params"] += param_count
                summary["trainable_tensors"] += 1
        return summary

    def moe_finetune_hparams(self) -> dict[str, bool | int | float | str]:
        flat: dict[str, bool | int | float | str] = {}
        self._flatten_moe_status("moe_finetune", self.moe_finetune_status, flat)
        return flat

    def moe_finetune_param_group_rows(self) -> list[list[bool | int | float | str | None]]:
        rows = []
        for group, summary in self.moe_finetune_status.get("param_groups", {}).items():
            rows.append(
                [
                    group,
                    summary["total_params"],
                    summary["trainable_params"],
                    summary["total_tensors"],
                    summary["trainable_tensors"],
                    summary.get("lr_scale"),
                ]
            )
        return rows

    def _flatten_moe_status(self, prefix: str, value, flat: dict[str, bool | int | float | str]) -> None:
        if prefix.endswith("param_groups"):
            return
        if isinstance(value, dict):
            for key, item in value.items():
                self._flatten_moe_status(f"{prefix}/{key}", item, flat)
            return
        if isinstance(value, (list, tuple)):
            flat[prefix] = ",".join(str(item) for item in value)
            return
        flat[prefix] = "" if value is None else value

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
        if self._is_ahi_task() and self._ahi_eval_threshold is not None:
            checkpoint["ahi_eval_threshold"] = float(self._ahi_eval_threshold)

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
        self._finalize_epoch(stage="test")

    def on_fit_start(self):
        super().on_fit_start()
        if self.model_averager is not None:
            self.model_averager.on_fit_start(self.trainer)

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if self.model_averager is not None:
            self.model_averager.on_load_checkpoint(checkpoint)
        if self._is_ahi_task():
            threshold = checkpoint.get("ahi_eval_threshold")
            self._ahi_eval_threshold = None if threshold is None else float(threshold)

    def on_test_start(self):
        super().on_test_start()
        if self._is_ahi_task() and self._ahi_eval_threshold is None:
            raise ValueError(
                "AHI test/inference requires a validation-fitted threshold stored in the checkpoint. "
                "This checkpoint does not contain `ahi_eval_threshold`."
            )

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
        if self._is_survival_task() and stage in {"val", "test"}:
            # Cox eval risk sets span the whole epoch, so batch losses are not meaningful.
            self._collect_survival_eval_batch(stage, logits, batch)
            return None

        loss_info = self._compute_loss(logits, batch)
        if loss_info is None:
            if stage == "train":
                raise ValueError("No valid labels found in the current training batch.")
            valid_count = 0
            loss = None
        else:
            loss, valid_count = loss_info
            moe_tuning = getattr(self.finetune_config, "moe_tuning", None) if self.finetune_config is not None else None
            moe_reg = getattr(moe_tuning, "moe_regularization", None) if moe_tuning is not None else None
            if stage == "train" and getattr(moe_reg, "enabled", False):
                supervised_loss = loss
                moe_out = compute_downstream_moe_regularization(
                    getattr(model.backbone, "last_moe_aux", None),
                    moe_reg,
                    batch,
                    prefix="train",
                )
                loss = supervised_loss + moe_out.loss
                self.log(
                    "train_supervised_loss",
                    supervised_loss,
                    prog_bar=False,
                    sync_dist=True,
                    on_step=True,
                    on_epoch=True,
                    batch_size=max(valid_count, 1),
                )
                for name, value in moe_out.metrics.items():
                    self.log(
                        name,
                        value,
                        prog_bar=False,
                        sync_dist=True,
                        on_step=True,
                        on_epoch=True,
                        batch_size=max(valid_count, 1),
                    )
            if stage in {"val", "test"}:
                for name, value in compute_downstream_moe_metrics(
                    getattr(model.backbone, "last_moe_aux", None),
                    batch,
                    prefix=stage,
                ).items():
                    self.log(
                        name,
                        value,
                        prog_bar=False,
                        sync_dist=True,
                        on_step=False,
                        on_epoch=True,
                        batch_size=max(valid_count, 1),
                    )
            eval_loss_sums = getattr(self, "_eval_loss_sums", {})
            eval_loss_counts = getattr(self, "_eval_loss_counts", {})
            if stage == "train":
                self.log(
                    f"{stage}_loss",
                    loss,
                    prog_bar=True,
                    sync_dist=True,
                    on_step=True,
                    on_epoch=True,
                    batch_size=max(valid_count, 1),
                )
            elif stage in eval_loss_sums and stage in eval_loss_counts:
                eval_loss_sums[stage] += float(loss.detach().item()) * valid_count
                eval_loss_counts[stage] += int(valid_count)

        if self._is_survival_task():
            return loss if stage == "train" else None

        if self._is_ahi_task() and stage == "train":
            self._accumulate_ahi_train_pointwise_counts(batch, logits)
        elif self._is_ahi_task() and stage in {"val", "test"}:
            records = self._extract_ahi_event_records(batch, logits)
            if records:
                self._stage_outputs[stage].extend(records)
        else:
            preds = self._extract_valid_predictions(batch, logits)
            if preds is not None:
                self._stage_outputs[stage].append(preds)
            if stage == "test" and prediction_export_enabled(self.args):
                targets = self._get_targets(batch)
                self._prediction_records["test"].extend(extract_prediction_records(self.args, batch, logits, targets))

        return loss if stage == "train" else None

    def _compute_loss(self, logits, batch):
        if self._is_survival_task():
            return self._compute_survival_loss(logits, batch)

        targets = self._get_targets(batch)

        if getattr(self.args, "is_multilabel", False):
            valid_mask = targets != -1.0
            if not valid_mask.any():
                return None
            loss = self._multilabel_loss(logits, targets.float())[valid_mask].mean()
            return loss, int(valid_mask.sum().item())

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

    def _compute_survival_loss(self, logits, batch):
        metadata = batch["metadata"]
        event_time = metadata["event_time"].to(self.args.device)
        is_event = metadata["is_event"].to(self.args.device)
        has_label = metadata["has_label"].to(self.args.device)

        if logits.shape != event_time.shape:
            raise ValueError(
                f"Survival head output shape {tuple(logits.shape)} does not match labels {tuple(event_time.shape)}."
            )

        valid_mask = has_label > 0.5
        if not valid_mask.any():
            return None
        token_start = batch["token_start"].to(torch.long).detach().cpu()
        key_column = self._survival_key_column()
        if key_column not in metadata:
            raise ValueError(
                f"Survival batch metadata is missing key column {key_column!r}; "
                "regenerate presets with survival sidecars."
            )
        # Cox labels are subject-level, so repeated nights/windows in a train batch should share one risk-set row.
        tensors = self._aggregate_survival_records(
            [
                {
                    "pred": logits,
                    "event_time": event_time,
                    "is_event": is_event,
                    "has_label": has_label,
                    "survival_key": [str(key) for key in metadata[key_column]],
                    "path": [str(path) for path in metadata["path"]],
                    "token_start": [int(value) for value in token_start.tolist()],
                }
            ]
        )
        logits = tensors["pred"]
        event_time = tensors["event_time"]
        is_event = tensors["is_event"]
        has_label = tensors["has_label"]
        valid_mask = has_label > 0.5
        loss = self._survival_loss(logits, has_label, event_time, is_event)
        event_count = int(((is_event > 0.5) & valid_mask).sum().item())
        return loss, event_count

    def _collect_survival_eval_batch(self, stage: str, logits, batch) -> None:
        metadata = batch["metadata"]
        event_time = metadata["event_time"]
        is_event = metadata["is_event"]
        has_label = metadata["has_label"]
        key_column = self._survival_key_column()

        if logits.shape != event_time.shape:
            raise ValueError(
                f"Survival head output shape {tuple(logits.shape)} does not match labels {tuple(event_time.shape)}."
            )
        # Prediction export is independent of Cox likelihood terms; keep raw risk scores for unlabeled batches.
        export_predictions = stage == "test" and prediction_export_enabled(self.args)
        if not (has_label > 0.5).any() and not export_predictions:
            return

        token_start = batch["token_start"].to(torch.long).detach().cpu()
        if key_column not in metadata:
            raise ValueError(
                f"Survival batch metadata is missing key column {key_column!r}; "
                "regenerate presets with survival sidecars."
            )
        self._stage_outputs[stage].append(
            {
                "pred": logits.detach().to(device="cpu", dtype=torch.float32),
                "event_time": event_time.detach().to(device="cpu", dtype=torch.float32),
                "is_event": is_event.detach().to(device="cpu", dtype=torch.float32),
                "has_label": has_label.detach().to(device="cpu", dtype=torch.float32),
                "survival_key": [str(key) for key in metadata[key_column]],
                "path": [str(path) for path in batch["metadata"]["path"]],
                "token_start": [int(value) for value in token_start.tolist()],
            }
        )

    def _survival_key_column(self) -> str:
        survival = getattr(self.args, "survival", None)
        key_column = getattr(survival, "key_column", None)
        if key_column in (None, ""):
            raise ValueError("Survival task requires finetune.survival.key_column.")
        return str(key_column)

    def _gather_survival_eval_records(self, records):
        if not is_torch_distributed_ready():
            return records

        _, world_size = get_rank_world_size()
        gathered: list[list[dict[str, torch.Tensor]] | None] = [None] * world_size
        dist.all_gather_object(gathered, records)

        merged: list[dict[str, torch.Tensor]] = []
        for item in gathered:
            if isinstance(item, list):
                merged.extend(item)
        return merged

    def _finalize_survival_epoch(self, stage: str, outputs) -> None:
        records = list(outputs)
        outputs.clear()
        if stage == "train":
            return

        # Build the monitored Cox loss from the full eval risk set, including all DDP ranks.
        records = self._gather_survival_eval_records(records)
        if not records:
            return

        # Export before the event-count check so all-censored inference still writes risk scores.
        if stage == "test" and prediction_export_enabled(self.args):
            self.prediction_rows = self._build_survival_prediction_rows(records)

        # Loss aggregation is subject-level; prediction export above intentionally remains per path.
        tensors = self._aggregate_survival_records(records)
        device = torch.device(getattr(self.args, "device", "cpu"))
        pred = tensors["pred"].to(device)
        event_time = tensors["event_time"].to(device)
        is_event = tensors["is_event"].to(device)
        has_label = tensors["has_label"].to(device)
        event_count = int(((is_event > 0.5) & (has_label > 0.5)).sum().item())
        if event_count == 0:
            return

        loss = self._survival_loss(pred, has_label, event_time, is_event)
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            logger=True,
            sync_dist=False,
            on_step=False,
            on_epoch=True,
        )

    def _aggregate_survival_records(self, records) -> dict[str, torch.Tensor]:
        # Keep one Cox row per survival key. Multiple nights/windows contribute a mean raw log-risk,
        # while labels must remain identical because sidecars are subject-level metadata.
        grouped: dict[str, dict[str, object]] = {}
        seen: set[tuple[str, str, int]] = set()
        for record in records:
            pred = record["pred"]
            event_time = record["event_time"]
            is_event = record["is_event"]
            has_label = record["has_label"]
            for idx, key in enumerate(record["survival_key"]):
                path = str(record["path"][idx])
                token_start = int(record["token_start"][idx])
                dedupe_key = (str(key), path, token_start)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                item = grouped.get(str(key))
                current_labels = {
                    "event_time": event_time[idx],
                    "is_event": is_event[idx],
                    "has_label": has_label[idx],
                }
                if item is None:
                    grouped[str(key)] = {**current_labels, "preds": [pred[idx]]}
                    continue

                for label_name, label_value in current_labels.items():
                    if not torch.allclose(item[label_name], label_value, equal_nan=True):
                        raise ValueError(f"Survival labels differ across records for key {key!r}.")
                item["preds"].append(pred[idx])

        rows = []
        event_times = []
        is_events = []
        has_labels = []
        for item in grouped.values():
            rows.append(torch.stack(item["preds"], dim=0).mean(dim=0))
            event_times.append(item["event_time"])
            is_events.append(item["is_event"])
            has_labels.append(item["has_label"])
        return {
            "pred": torch.stack(rows, dim=0),
            "event_time": torch.stack(event_times, dim=0),
            "is_event": torch.stack(is_events, dim=0),
            "has_label": torch.stack(has_labels, dim=0),
        }

    def _build_survival_prediction_rows(self, records) -> list[dict[str, object]]:
        grouped: dict[str, list[dict[str, object]]] = {}
        seen: set[tuple[str, int]] = set()
        for record in records:
            pred = record["pred"].numpy()
            event_time = record["event_time"].numpy()
            is_event = record["is_event"].numpy()
            has_label = record["has_label"].numpy()
            for idx, path in enumerate(record["path"]):
                token_start = int(record["token_start"][idx])
                key = (str(path), token_start)
                if key in seen:
                    continue
                seen.add(key)
                grouped.setdefault(str(path), []).append(
                    {
                        "token_start": token_start,
                        "pred": pred[idx],
                        "event_time": event_time[idx],
                        "is_event": is_event[idx],
                        "has_label": has_label[idx],
                    }
                )

        rows: list[dict[str, object]] = []
        for path, items in grouped.items():
            items.sort(key=lambda item: int(item["token_start"]))
            token_starts = [int(item["token_start"]) for item in items]
            log_risk = np.stack([np.asarray(item["pred"], dtype=np.float32) for item in items], axis=0).mean(axis=0)
            # Survival labels are sidecar metadata, so all windows for one path share the same vectors.
            event_time = np.asarray(items[0]["event_time"], dtype=np.float32)
            is_event = (np.asarray(items[0]["is_event"], dtype=np.float32) > 0.5).astype(np.int64)
            has_label = (np.asarray(items[0]["has_label"], dtype=np.float32) > 0.5).astype(np.int64)
            groundtruth = {
                "event_time": event_time.tolist(),
                "is_event": is_event.tolist(),
                "has_label": has_label.tolist(),
            }
            rows.append(
                {
                    "path": path,
                    "kind": "survival",
                    "groundtruth": groundtruth,
                    "prediction": log_risk.tolist(),
                    "log_risk": log_risk.tolist(),
                    "event_time": event_time.tolist(),
                    "is_event": is_event.tolist(),
                    "has_label": has_label.tolist(),
                    "n_predictions": int(log_risk.size),
                    "n_windows": len(items),
                    "token_starts": token_starts,
                }
            )
        return rows

    def _extract_valid_predictions(self, batch, logits):
        labels = self._get_targets(batch)

        if getattr(self.args, "is_multilabel", False):
            mask = labels != -1.0
            if not mask.any():
                return None

            probs = torch.sigmoid(logits[mask]).to(torch.float32).detach().cpu().numpy()
            labels_np = labels[mask].to(torch.int64).detach().cpu().numpy()
            return probs, labels_np

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

    def _extract_ahi_event_records(self, batch, logits) -> list[dict[str, np.ndarray]]:
        """Build per-sample AHI eval records.

        Built-in ``ahi`` currently runs on whole-night inputs by default, so validation/test/infer
        usually emit one logical record per path. ``token_start`` is still preserved because the
        downstream AHI metric path can merge records when a caller explicitly evaluates windowed
        samples or distributed gathering surfaces duplicate windows.
        """
        labels = batch["tokens"]["ahi"].detach().cpu()
        probs = torch.sigmoid(logits).to(torch.float32).detach().cpu()
        true_ahi = batch["metadata"]["ahi"].to(torch.float32).detach().cpu()
        tst_hours = batch["metadata"]["tst"].to(torch.float32).detach().cpu()
        token_start = batch["token_start"].to(torch.long).detach().cpu()
        paths = list(batch["metadata"]["path"])
        stage5 = batch["tokens"]["stage5"].detach().cpu()

        records: list[dict[str, np.ndarray]] = []
        for idx in range(labels.size(0)):
            second_valid_mask = labels[idx].reshape(-1) != -1.0
            if not second_valid_mask.any():
                continue
            stage5_tokens = stage5[idx].to(torch.int64).reshape(-1).numpy()
            truth = labels[idx].reshape(-1)[second_valid_mask].to(torch.int64).numpy()
            score = probs[idx].reshape(-1)[second_valid_mask].numpy()
            record = {
                "path": str(paths[idx]),
                "token_start": int(token_start[idx].item()),
                "truth": truth,
                "score": score,
                "true_ahi": float(true_ahi[idx].item()),
                "tst_hours": float(tst_hours[idx].item()),
                "stage5": stage5_tokens,
                "second_valid_mask": second_valid_mask.numpy(),
            }
            records.append(record)
        return records

    def _get_targets(self, batch):
        if self._is_survival_task():
            raise ValueError("Survival tasks use event_time/is_event/has_label metadata, not label_name targets.")
        if not self.args.is_seq:
            return batch["metadata"][self.args.label_name].to(self.args.device)

        label_source_name = getattr(self.args, "label_source_name", self.args.label_name)
        labels = batch["tokens"][label_source_name].to(self.args.device)
        if getattr(self.args, "is_multilabel", False):
            return labels
        return remap_stage_labels(labels, self.args.label_name)

    def _is_ahi_task(self) -> bool:
        return getattr(self.args, "label_name", None) == "ahi"

    def _is_survival_task(self) -> bool:
        return bool(getattr(self.args, "is_survival", False))

    def _accumulate_ahi_train_pointwise_counts(self, batch, logits) -> None:
        labels = self._get_targets(batch)
        valid_mask = labels != -1.0
        if not valid_mask.any():
            return

        probs = torch.sigmoid(logits[valid_mask])
        targets = labels[valid_mask].to(torch.int64)
        preds = (probs >= 0.5).to(torch.int64)

        self._ahi_train_pointwise_counts["tp"] += int(((preds == 1) & (targets == 1)).sum().item())
        self._ahi_train_pointwise_counts["fp"] += int(((preds == 1) & (targets == 0)).sum().item())
        self._ahi_train_pointwise_counts["tn"] += int(((preds == 0) & (targets == 0)).sum().item())
        self._ahi_train_pointwise_counts["fn"] += int(((preds == 0) & (targets == 1)).sum().item())

    def _compute_reduced_ahi_train_pointwise_metrics(self) -> dict[str, float]:
        counts = self._ahi_train_pointwise_counts
        stats = torch.tensor(
            [counts["tp"], counts["fp"], counts["tn"], counts["fn"]],
            dtype=torch.float64,
            device=torch.device(getattr(self.args, "device", "cpu")),
        )
        trainer = getattr(self, "trainer", None)
        if is_torch_distributed_ready():
            if trainer is not None and hasattr(trainer, "strategy"):
                stats = trainer.strategy.reduce(stats, reduce_op="sum")
            else:  # pragma: no cover - trainer-less distributed fallback
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        tp, fp, tn, fn = [int(value) for value in stats.tolist()]
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        self._ahi_train_pointwise_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        return {
            "ahi_pointwise_accuracy": float(accuracy),
            "ahi_pointwise_precision": float(precision),
            "ahi_pointwise_recall": float(recall),
            "ahi_pointwise_f1": float(f1),
        }

    def _ahi_search_thresholds_for_stage(self, stage: str) -> tuple[float, ...] | None:
        if stage != "val":
            return None
        thresholds = getattr(self.args, "ahi_val_search_thresholds", AHI_FINE_THRESHOLD_GRID)

        if thresholds is None:
            return None
        return tuple(float(value) for value in thresholds)

    @staticmethod
    def _can_broadcast_ahi_metrics() -> bool:
        return is_torch_distributed_ready() and hasattr(dist, "broadcast_object_list")

    def _compute_ahi_metrics_for_stage(
        self,
        stage: str,
        records: list[dict[str, np.ndarray]],
    ) -> tuple[dict[str, float], float, tuple[np.ndarray, np.ndarray] | None]:
        prepared_records = _prepare_ahi_records(records)

        if stage == "val":
            metrics, eval_threshold = _compute_ahi_event_metrics_from_prepared(
                prepared_records,
                threshold=None,
                search_thresholds=self._ahi_search_thresholds_for_stage("val"),
            )
            self._ahi_eval_threshold = float(eval_threshold)
            aggregate = _aggregate_prepared_ahi_records(prepared_records, threshold=float(eval_threshold))
            return metrics, float(eval_threshold), (aggregate["true_ahi"], aggregate["pred_ahi"])

        if self._ahi_eval_threshold is None:
            raise ValueError(
                "AHI evaluation requires a validation-fitted threshold. "
                "No `ahi_eval_threshold` is available for test/inference."
            )

        eval_threshold = float(self._ahi_eval_threshold)
        metrics, _ = _compute_ahi_event_metrics_from_prepared(prepared_records, threshold=eval_threshold)
        aggregate = _aggregate_prepared_ahi_records(prepared_records, threshold=eval_threshold)
        return metrics, eval_threshold, (aggregate["true_ahi"], aggregate["pred_ahi"])

    def _compute_or_broadcast_ahi_metrics(
        self,
        stage: str,
        records: list[dict[str, np.ndarray]],
    ) -> tuple[dict[str, float], float, tuple[np.ndarray, np.ndarray] | None]:
        trainer = getattr(self, "trainer", None)
        if trainer is None or not self._can_broadcast_ahi_metrics():
            return self._compute_ahi_metrics_for_stage(stage, records)

        payload: list[dict[str, object] | None] = [None]
        scatter_arrays: tuple[np.ndarray, np.ndarray] | None = None
        if trainer.is_global_zero:
            try:
                metrics, eval_threshold, scatter_arrays = self._compute_ahi_metrics_for_stage(stage, records)
                payload[0] = {
                    "metrics": metrics,
                    "eval_threshold": float(eval_threshold),
                    "error_type": None,
                    "error_message": None,
                }
            except Exception as exc:  # pragma: no cover - distributed error fan-out
                payload[0] = {
                    "metrics": None,
                    "eval_threshold": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }

        dist.broadcast_object_list(payload, src=0)
        result = payload[0] or {}
        error_message = result.get("error_message")
        if error_message is not None:
            if result.get("error_type") == "ValueError":
                raise ValueError(str(error_message))
            raise RuntimeError(str(error_message))

        metrics = result["metrics"]
        eval_threshold = float(result["eval_threshold"])
        if stage == "val":
            self._ahi_eval_threshold = eval_threshold
        return metrics, eval_threshold, scatter_arrays

    @staticmethod
    def _layer_mix_snapshot(model: torch.nn.Module):
        getter = getattr(model, "layer_mix_snapshot", None)
        if not callable(getter):
            return None
        return getter()

    def _empty_epoch_outputs(self):
        if getattr(self.args, "is_multilabel", False):
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
        if self.args.is_classification:
            output_dim = int(getattr(self.args, "output_dim", 0) or 0)
            return np.empty((0, output_dim), dtype=np.float32), np.empty((0,), dtype=np.int64)
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    def _concat_epoch_outputs(self, outputs):
        if not outputs:
            return self._empty_epoch_outputs()

        preds, gts = zip(*outputs)
        return np.concatenate(preds, axis=0), np.concatenate(gts, axis=0)

    def _gather_eval_outputs(self, preds: np.ndarray, gts: np.ndarray):
        if not is_torch_distributed_ready() or not hasattr(dist, "all_gather_object"):
            return preds, gts

        _, world_size = get_rank_world_size()
        gathered_preds: list[np.ndarray | None] = [None] * world_size
        gathered_gts: list[np.ndarray | None] = [None] * world_size
        dist.all_gather_object(gathered_preds, preds)
        dist.all_gather_object(gathered_gts, gts)

        gathered_preds = [item for item in gathered_preds if isinstance(item, np.ndarray) and item.size > 0]
        gathered_gts = [item for item in gathered_gts if isinstance(item, np.ndarray) and item.size > 0]
        if not gathered_preds or not gathered_gts:
            return self._empty_epoch_outputs()

        return np.concatenate(gathered_preds, axis=0), np.concatenate(gathered_gts, axis=0)

    def _gather_ahi_event_records(self, records: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
        if not is_torch_distributed_ready() or not hasattr(dist, "all_gather_object"):
            return records

        _, world_size = get_rank_world_size()
        gathered: list[list[dict[str, np.ndarray]] | None] = [None] * world_size
        dist.all_gather_object(gathered, records)

        merged: list[dict[str, np.ndarray]] = []
        for item in gathered:
            if isinstance(item, list):
                merged.extend(item)
        return merged

    def _gather_prediction_records(self, records: list[dict[str, object]]) -> list[dict[str, object]]:
        if not is_torch_distributed_ready() or not hasattr(dist, "all_gather_object"):
            return records

        _, world_size = get_rank_world_size()
        gathered: list[list[dict[str, object]] | None] = [None] * world_size
        dist.all_gather_object(gathered, records)

        merged: list[dict[str, object]] = []
        for item in gathered:
            if isinstance(item, list):
                merged.extend(item)
        return merged

    def _log_eval_loss(self, stage: str) -> None:
        eval_loss_sums = getattr(self, "_eval_loss_sums", {})
        eval_loss_counts = getattr(self, "_eval_loss_counts", {})
        if stage not in eval_loss_sums or stage not in eval_loss_counts:
            return

        loss_sum = float(eval_loss_sums[stage])
        loss_count = int(eval_loss_counts[stage])
        eval_loss_sums[stage] = 0.0
        eval_loss_counts[stage] = 0

        stats = torch.tensor(
            [loss_sum, float(loss_count)],
            dtype=torch.float64,
            device=torch.device(getattr(self.args, "device", "cpu")),
        )
        trainer = getattr(self, "trainer", None)
        if is_torch_distributed_ready():
            if trainer is not None and hasattr(trainer, "strategy"):
                stats = trainer.strategy.reduce(stats, reduce_op="sum")
            else:  # pragma: no cover - trainer-less distributed fallback
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        global_loss_count = int(stats[1].item())
        if global_loss_count == 0:
            return

        self.log(
            f"{stage}_loss",
            float(stats[0].item()) / global_loss_count,
            prog_bar=True,
            logger=True,
            sync_dist=False,
            on_step=False,
            on_epoch=True,
        )

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
        if stage == "test":
            self.prediction_rows = []
        if self._is_survival_task():
            self._finalize_survival_epoch(stage, outputs)
            return None

        if stage in getattr(self, "_eval_loss_sums", {}):
            self._log_eval_loss(stage)

        if self._is_ahi_task() and stage == "train":
            metrics = self._compute_reduced_ahi_train_pointwise_metrics()
            for k, v in metrics.items():
                self.log(
                    f"{stage}_{k}",
                    v,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                    on_epoch=True,
                )
            return None

        if self._is_ahi_task() and stage in {"val", "test"}:
            records = list(outputs)
            outputs.clear()
            records = self._gather_ahi_event_records(records)
            if not records:
                return None

            metrics, eval_threshold, scatter_arrays = self._compute_or_broadcast_ahi_metrics(stage, records)

            for k, v in metrics.items():
                self.log(
                    f"{stage}_{k}",
                    v,
                    prog_bar=(stage != "train"),
                    logger=True,
                    sync_dist=False,
                    on_epoch=True,
                )
            trainer = getattr(self, "trainer", None)
            if trainer is not None and trainer.is_global_zero:
                if scatter_arrays is None:
                    true_ahi, pred_ahi = extract_ahi_summary_scatter_arrays(records, threshold=eval_threshold)
                else:
                    true_ahi, pred_ahi = scatter_arrays
                self._eval_visualizer.log_ahi_summary_scatter(
                    stage=stage,
                    preds=pred_ahi,
                    targets=true_ahi,
                    label_name=self.args.label_name,
                    current_epoch=int(self.current_epoch),
                )
            if stage == "test" and prediction_export_enabled(self.args):
                self.prediction_rows = build_ahi_prediction_rows(records, eval_threshold)
            if trainer is not None and is_torch_distributed_ready() and hasattr(trainer, "strategy"):
                trainer.strategy.barrier(f"ahi_{stage}_epoch_end")
            return records

        preds, gts = self._concat_epoch_outputs(outputs)
        outputs.clear()

        if stage in {"val", "test"}:
            preds, gts = self._gather_eval_outputs(preds, gts)
        if preds.size == 0 or gts.size == 0:
            if stage == "test" and prediction_export_enabled(self.args):
                self._prediction_records["test"].clear()
            return None

        metrics = compute_downstream_metrics(
            gts,
            preds,
            is_classification=self.args.is_classification,
            is_multilabel=getattr(self.args, "is_multilabel", False),
            output_dim=getattr(self.args, "output_dim", None),
            stage_names=getattr(self.args, "stage_names", None),
        )
        for k, v in metrics.items():
            self.log(
                f"{stage}_{k}",
                v,
                prog_bar=(stage != "train"),
                logger=True,
                sync_dist=(stage == "train"),
                on_epoch=True,
            )

        trainer = getattr(self, "trainer", None)
        if (
            stage in {"val", "test"}
            and trainer is not None
            and trainer.is_global_zero
            and not getattr(self.args, "is_multilabel", False)
        ):
            self._eval_visualizer.log(
                stage=stage,
                preds=preds,
                targets=gts,
                is_classification=self.args.is_classification,
                output_dim=getattr(self.args, "output_dim", None),
                label_name=self.args.label_name,
                current_epoch=int(self.current_epoch),
                class_labels=getattr(self.args, "class_labels", None),
            )

        if stage == "test" and prediction_export_enabled(self.args):
            records = self._gather_prediction_records(self._prediction_records["test"])
            self._prediction_records["test"].clear()
            self.prediction_rows = build_prediction_rows(records)

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

    def configure_optimizers(self):
        moe_tuning = getattr(self.finetune_config, "moe_tuning", None) if self.finetune_config is not None else None
        if moe_tuning is not None:
            grouped_params = {
                (group, decay_type): []
                for group in ("head", "backbone", "experts", "routers", "tokenizers", "projection", "lora")
                for decay_type in ("decay", "no_decay")
            }
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                group = self._finetune_param_to_group.get(n)
                if group is None:
                    group = self._semantic_group_for_param(n)
                    self._finetune_param_to_group[n] = group
                lr_scale = self._finetune_lr_scales[group]
                if lr_scale == 0.0:
                    raise ValueError(f"Parameter '{n}' is trainable but its finetune LR scale is zero.")
                if p.ndim >= 2 and ("norm" not in n.lower()) and ("bias" not in n.lower()):
                    decay_type = "decay"
                else:
                    decay_type = "no_decay"
                grouped_params[(group, decay_type)].append(p)

            optimizer_groups = []
            for group in ("head", "backbone", "experts", "routers", "tokenizers", "projection", "lora"):
                for decay_type in ("decay", "no_decay"):
                    params = grouped_params[(group, decay_type)]
                    if not params:
                        continue
                    optimizer_groups.append(
                        {
                            "params": params,
                            "weight_decay": self.args.weight_decay if decay_type == "decay" else 0.0,
                            "lr": self.args.lr * self._finetune_lr_scales[group],
                            "name": f"{group}/{decay_type}",
                        }
                    )

            if moe_tuning.mode == "head_only" and not any(
                group["name"].startswith("head/") for group in optimizer_groups
            ):
                raise ValueError("head_only MoE tuning found no trainable head optimizer group.")
            if moe_tuning.mode == "top_moe_layer_expert_only" and not any(
                group["name"].startswith("experts/") for group in optimizer_groups
            ):
                raise ValueError("top_moe_layer_expert_only found no trainable expert optimizer group.")

            optimizer = torch.optim.AdamW(
                optimizer_groups,
                lr=self.args.lr,
                betas=(0.9, 0.95),
                eps=1e-8,
            )
        else:
            decay, no_decay = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim >= 2 and ("norm" not in n.lower()) and ("bias" not in n.lower()):
                    decay.append(p)
                else:
                    no_decay.append(p)

            optimizer = torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": self.args.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.args.lr,
                betas=(0.9, 0.95),
                eps=1e-8,
            )

        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            warmup_steps=getattr(self.args, "warmup_steps", None),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
