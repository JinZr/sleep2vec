from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.default_dataset import DefaultDataset, SampleIndex
from data.utils import default_extractor, default_mlm_mask_generator, default_tokenizer
from sleep2stat.analyzers.base import BaseAnalyzer
from sleep2stat.config import AnalyzerConfig, ChannelSpec
from sleep2stat.core.artifacts import AnalyzerResult, FailureRecord
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.io.records import SleepRecord
from sleep2stat.registry import register_analyzer

STAGE5_NAMES = ["W", "N1", "N2", "N3", "REM"]


class _Sleep2statDataset(DefaultDataset):
    def __init__(
        self,
        *,
        records: list[SleepRecord],
        channel_specs: dict[str, ChannelSpec],
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.channel_names = list(channel_specs)
        self.randomly_select_channels = False
        self.min_channels = len(self.channel_names)
        self.allow_missing_channels = False
        self.bucket_by_available_channels = False
        self.train_pair_probs = None
        self.train_pair_track_unique_samples = False
        self.token_sec = records[0].token_sec if records else 30
        self.generative = False
        self.is_train_set = False

        data = []
        for idx, record in enumerate(records):
            n_tokens = max(0, int(record.duration_sec // record.token_sec))
            if n_tokens <= 0:
                continue
            metadata = dict(record.metadata)
            metadata.update(
                {
                    "record_id": record.record_id,
                    "path": str(record.path),
                    "source": record.source or "",
                    "split": record.split,
                    "duration_sec": record.duration_sec,
                }
            )
            data.append(
                SampleIndex(
                    id=idx,
                    path=str(record.path),
                    start=0,
                    end=min(n_tokens, record.max_tokens),
                    metadata=metadata,
                )
            )

        extractors = {}
        tokenizers = {}
        mask_generators = {}
        for name, spec in channel_specs.items():
            extractors[name] = default_extractor(name, spec.input_dim, source_name=spec.source)
            tokenizers[name] = default_tokenizer(spec.input_dim)
            mask_generators[name] = default_mlm_mask_generator(0.0)

        super().__init__(
            save_preset_path=None,
            load_preset_path=None,
            data=data,
            split=sorted({record.split for record in records}) or [""],
            extractors=extractors,
            tokenizers=tokenizers,
            mask_generators=mask_generators,
            dataloader_config={"batch_size": batch_size, "shuffle": False, "num_workers": num_workers},
        )


@register_analyzer("sleep2vec_downstream")
class Sleep2vecDownstreamAnalyzer(BaseAnalyzer):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.args: argparse.Namespace | None = None
        self.model = None
        self.threshold: float | None = None
        self.move_to_device = None
        self.include_probabilities = True
        self.include_raw_logits = False

    def prepare(self, context: Sleep2statContext) -> None:
        namespace = self.config.namespace or "sleep2vec"
        common_mod = importlib.import_module(f"{namespace}.common")
        finetune_mod = importlib.import_module(f"{namespace}.sleep2vec_finetuning")
        utils_mod = importlib.import_module(f"{namespace}.utils")

        args = argparse.Namespace(
            config=Path(self.config.config),
            label_name=self.config.label_name,
            device=context.device,
            batch_size=self.config.batch_size or context.batch_size or 12,
            num_workers=context.num_workers,
            data_backend=None,
            kaldi_data_root=None,
            kaldi_manifest=None,
            pretrained_backbone_path=None,
            print_diagnostics=False,
            diagnostics_steps=5,
        )
        config_bundle, model_cfg = common_mod.apply_finetune_config(args)
        module = finetune_mod.Sleep2vecFinetuning(
            args,
            model_cfg,
            finetune_config=config_bundle.finetune,
            averaging_config=config_bundle.averaging,
        )
        checkpoint = torch.load(self.config.ckpt_path, map_location=torch.device(context.device), weights_only=False)
        if isinstance(checkpoint, dict):
            module.on_load_checkpoint(checkpoint)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        module.load_state_dict(state_dict, strict=True)
        module.eval()
        module.to(context.device)

        self.args = args
        self.model = module._get_eval_model()
        self.move_to_device = utils_mod.move_to_device
        self.threshold = self._resolve_threshold(checkpoint)
        self.include_probabilities = bool(context.config.outputs.include_probabilities)
        self.include_raw_logits = bool(context.config.outputs.include_raw_logits)

    def run(
        self,
        records: list[SleepRecord],
        context: Sleep2statContext,
    ) -> tuple[list[AnalyzerResult], list[FailureRecord]]:
        if self.args is None or self.model is None or self.move_to_device is None:
            raise RuntimeError("Analyzer was not prepared.")
        if not records:
            return [], []
        results: list[AnalyzerResult] = []
        failures: list[FailureRecord] = []
        channel_specs = {name: context.config.signals.channels[name] for name in self.config.input_channels}
        dataset = _Sleep2statDataset(
            records=records,
            channel_specs=channel_specs,
            batch_size=self.config.batch_size or context.batch_size or self.args.batch_size,
            num_workers=context.num_workers,
        )
        retained_record_ids = {
            str(sample.metadata.get("record_id")) for sample in getattr(dataset, "data", []) if sample.metadata
        }
        for record in records:
            if record.record_id not in retained_record_ids:
                failures.append(
                    FailureRecord(
                        record_id=record.record_id,
                        analyzer=self.config.name,
                        error_type="RecordUnavailable",
                        message="Record was dropped before model inference; check duration and required input channels.",
                    )
                )
        if not retained_record_ids:
            return results, failures
        loader = dataset.dataloader(device=self.args.device)
        record_by_path = {str(record.path): record for record in records}
        with torch.no_grad():
            for batch in loader:
                try:
                    batch = self.move_to_device(batch, self.args.device)
                    logits = self.model(batch)
                    results.extend(self._decode_batch(batch, logits, record_by_path))
                except Exception as exc:
                    for path in batch.get("metadata", {}).get("path", []):
                        record = record_by_path.get(str(path))
                        failures.append(
                            FailureRecord(
                                record_id=record.record_id if record else str(path),
                                analyzer=self.config.name,
                                error_type=type(exc).__name__,
                                message=str(exc),
                            )
                        )
        return results, failures

    def _decode_batch(
        self,
        batch: dict[str, Any],
        logits: torch.Tensor,
        record_by_path: dict[str, SleepRecord],
    ) -> list[AnalyzerResult]:
        label_name = self.config.label_name
        if label_name == "ahi":
            if self.threshold is None:
                raise ValueError(
                    f"Analyzer {self.config.name!r} requires ahi_eval_threshold in checkpoint or explicit threshold."
                )
            return decode_ahi_logits(
                self.config.name,
                logits,
                batch,
                record_by_path,
                threshold=self.threshold,
                include_probabilities=self.include_probabilities,
            )
        if self.args and getattr(self.args, "is_classification", False):
            return decode_classification_logits(
                self.config.name,
                logits,
                batch,
                record_by_path,
                include_probabilities=self.include_probabilities,
                include_logits=self.include_raw_logits,
            )
        return decode_regression_logits(self.config.name, logits, batch, record_by_path)

    def _resolve_threshold(self, checkpoint: Any) -> float | None:
        explicit = self.config.threshold
        if isinstance(explicit, dict):
            explicit = explicit.get("value")
        if explicit not in (None, ""):
            return float(explicit)
        if isinstance(checkpoint, dict) and checkpoint.get("ahi_eval_threshold") is not None:
            return float(checkpoint["ahi_eval_threshold"])
        return None


def decode_classification_logits(
    analyzer_name: str,
    logits: torch.Tensor,
    batch: dict[str, Any],
    record_by_path: dict[str, SleepRecord],
    *,
    include_probabilities: bool,
    include_logits: bool,
) -> list[AnalyzerResult]:
    logits_cpu = logits.detach().to(torch.float32).cpu()
    paths = list(batch["metadata"]["path"])
    token_starts = _tensor_list(batch.get("token_start"), default=0, count=len(paths))
    lengths = _tensor_list(batch.get("length"), default=1, count=len(paths))
    results = []

    if logits_cpu.dim() == 3:
        probs = torch.softmax(logits_cpu, dim=-1).numpy()
        preds = probs.argmax(axis=-1).astype(np.int64)
        for idx, path in enumerate(paths):
            record = record_by_path[str(path)]
            n_tokens = min(int(lengths[idx]), preds.shape[1])
            token_start = int(token_starts[idx])
            frame = _epoch_base_frame(record, token_start, n_tokens)
            frame[f"{analyzer_name}_pred"] = preds[idx, :n_tokens]
            frame[f"{analyzer_name}_confidence"] = probs[idx, :n_tokens].max(axis=-1)
            frame[f"{analyzer_name}_entropy"] = _entropy(probs[idx, :n_tokens])
            if include_probabilities:
                labels = STAGE5_NAMES if probs.shape[-1] == 5 else [str(i) for i in range(probs.shape[-1])]
                for class_idx, label in enumerate(labels):
                    frame[f"{analyzer_name}_prob_{label}"] = probs[idx, :n_tokens, class_idx]
            if include_logits:
                for class_idx in range(logits_cpu.shape[-1]):
                    frame[f"{analyzer_name}_logit_{class_idx}"] = logits_cpu[idx, :n_tokens, class_idx].numpy()
            results.append(AnalyzerResult(analyzer_name, record.record_id, epoch=frame))
        return results

    probs = torch.softmax(logits_cpu, dim=-1).numpy()
    preds = probs.argmax(axis=-1).astype(np.int64)
    for idx, path in enumerate(paths):
        record = record_by_path[str(path)]
        night = {
            f"{analyzer_name}_pred": int(preds[idx]),
            f"{analyzer_name}_confidence": float(probs[idx].max()),
        }
        if include_probabilities and probs.shape[-1] == 2:
            night[f"{analyzer_name}_prob_male"] = float(probs[idx, 1])
        elif include_probabilities:
            for class_idx in range(probs.shape[-1]):
                night[f"{analyzer_name}_prob_{class_idx}"] = float(probs[idx, class_idx])
        if include_logits:
            for class_idx in range(logits_cpu.shape[-1]):
                night[f"{analyzer_name}_logit_{class_idx}"] = float(logits_cpu[idx, class_idx])
        results.append(AnalyzerResult(analyzer_name, record.record_id, night=night))
    return results


def decode_regression_logits(
    analyzer_name: str,
    logits: torch.Tensor,
    batch: dict[str, Any],
    record_by_path: dict[str, SleepRecord],
) -> list[AnalyzerResult]:
    preds = logits.detach().to(torch.float32).cpu().reshape(logits.shape[0], -1).mean(dim=1).numpy()
    results = []
    for idx, path in enumerate(batch["metadata"]["path"]):
        record = record_by_path[str(path)]
        metadata_value = record.metadata.get("age")
        night = {f"{analyzer_name}_pred": float(preds[idx])}
        if metadata_value is not None:
            try:
                age_value = float(metadata_value)
                if math.isfinite(age_value):
                    night["age_metadata"] = age_value
                    night[f"{analyzer_name}_abs_error_vs_metadata"] = abs(float(preds[idx]) - age_value)
            except (TypeError, ValueError):
                pass
        results.append(AnalyzerResult(analyzer_name, record.record_id, night=night))
    return results


def decode_ahi_logits(
    analyzer_name: str,
    logits: torch.Tensor,
    batch: dict[str, Any],
    record_by_path: dict[str, SleepRecord],
    *,
    threshold: float,
    include_probabilities: bool = True,
    min_event_duration_sec: int = 10,
    merge_tolerance_sec: int = 3,
) -> list[AnalyzerResult]:
    probs = torch.sigmoid(logits.detach().to(torch.float32).cpu()).numpy()
    paths = list(batch["metadata"]["path"])
    token_starts = _tensor_list(batch.get("token_start"), default=0, count=len(paths))
    lengths = _tensor_list(batch.get("length"), default=1, count=len(paths))
    results = []
    for idx, path in enumerate(paths):
        record = record_by_path[str(path)]
        flat_prob = probs[idx].reshape(-1)
        valid_seconds = min(int(lengths[idx]) * record.token_sec, flat_prob.shape[0])
        flat_prob = flat_prob[:valid_seconds]
        start_offset = int(token_starts[idx]) * record.token_sec
        second_data = {
            "record_id": record.record_id,
            "path": str(record.path),
            "second_idx": np.arange(valid_seconds, dtype=np.int64) + start_offset,
            "start_sec": np.arange(valid_seconds, dtype=np.float32) + float(start_offset),
            "end_sec": np.arange(1, valid_seconds + 1, dtype=np.float32) + float(start_offset),
            f"{analyzer_name}_pred": (flat_prob > float(threshold)).astype(np.int64),
        }
        if include_probabilities:
            second_data[f"{analyzer_name}_prob"] = flat_prob
        second = pd.DataFrame(second_data)
        events = _events_from_prob(
            record,
            analyzer_name,
            flat_prob,
            threshold=float(threshold),
            start_offset=start_offset,
            min_duration=min_event_duration_sec,
            merge_tolerance=merge_tolerance_sec,
        )
        denominator_hours = valid_seconds / 3600.0 if valid_seconds > 0 else 0.0
        pred_ahi = float(len(events) / denominator_hours) if denominator_hours > 0 else np.nan
        night = {
            f"{analyzer_name}_pred_ahi": pred_ahi,
            f"{analyzer_name}_pred_event_count": int(len(events)),
            f"{analyzer_name}_recording_denominator_hours": float(denominator_hours),
            f"{analyzer_name}_threshold": float(threshold),
        }
        results.append(AnalyzerResult(analyzer_name, record.record_id, second=second, events=events, night=night))
    return results


def _events_from_prob(
    record: SleepRecord,
    analyzer_name: str,
    prob: np.ndarray,
    *,
    threshold: float,
    start_offset: int,
    min_duration: int,
    merge_tolerance: int,
) -> pd.DataFrame:
    mask = prob > threshold
    raw_segments = []
    start = None
    for idx, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            raw_segments.append([start, idx])
            start = None
    merged = []
    for segment in raw_segments:
        if not merged or segment[0] - merged[-1][1] > merge_tolerance:
            merged.append(segment)
        else:
            merged[-1][1] = segment[1]
    rows = []
    for event_idx, (left, right) in enumerate(merged):
        duration = right - left
        if duration < min_duration:
            continue
        event_prob = prob[left:right]
        rows.append(
            {
                "record_id": record.record_id,
                "path": str(record.path),
                "event_id": f"{record.record_id}__{analyzer_name}__{event_idx}",
                "analyzer": analyzer_name,
                "event_type": "predicted_respiratory_event",
                "onset_sec": float(start_offset + left),
                "offset_sec": float(start_offset + right),
                "duration_sec": float(duration),
                "confidence": float(event_prob.mean()),
                "score": float(event_prob.max()),
            }
        )
    return pd.DataFrame(rows)


def _epoch_base_frame(record: SleepRecord, token_start: int, n_tokens: int) -> pd.DataFrame:
    token_idx = np.arange(n_tokens, dtype=np.int64) + int(token_start)
    start_sec = token_idx * record.token_sec
    return pd.DataFrame(
        {
            "record_id": record.record_id,
            "path": str(record.path),
            "token_idx": token_idx,
            "start_sec": start_sec.astype(np.float32),
            "end_sec": (start_sec + record.token_sec).astype(np.float32),
            "is_padding": False,
        }
    )


def _tensor_list(value: torch.Tensor | None, *, default: int, count: int) -> list[int]:
    if value is None:
        return [default] * count
    return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _entropy(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=-1)
