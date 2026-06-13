from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import torch

from data.default_dataset import DefaultDataset, SampleIndex
from data.kaldi_psg_dataset import KaldiPSGDataset
from data.utils import default_extractor, default_mlm_mask_generator, default_tokenizer
from sleep2stat.analyzers.base import BaseAnalyzer
from sleep2stat.config import AnalyzerConfig, ChannelSpec
from sleep2stat.core.artifacts import AnalyzerResult, FailureRecord
from sleep2stat.core.context import Sleep2statContext
from sleep2stat.core.stage_sources import StageSourceResolver
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
        for record in records:
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
                    id=record.record_id,
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
            extractor = default_extractor(name, spec.input_dim, source_name=spec.source)
            if spec.scale != 1.0:
                base_extractor = extractor
                scale = float(spec.scale)

                def scaled_extractor(npz, start, end, base_extractor=base_extractor, scale=scale):
                    return base_extractor(npz, start, end) * scale

                extractor = scaled_extractor
            extractors[name] = extractor
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


def _build_datasets(
    *,
    records: list[SleepRecord],
    channel_specs: dict[str, ChannelSpec],
    batch_size: int,
    num_workers: int,
    context: Sleep2statContext,
) -> list[DefaultDataset]:
    data_cfg = getattr(context.config, "data", None)
    if getattr(data_cfg, "backend", "npz") == "kaldi":
        return _build_kaldi_datasets(
            records=records,
            channel_specs=channel_specs,
            batch_size=batch_size,
            num_workers=num_workers,
            context=context,
        )
    return [
        _Sleep2statDataset(
            records=records,
            channel_specs=channel_specs,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    ]


def _build_kaldi_datasets(
    *,
    records: list[SleepRecord],
    channel_specs: dict[str, ChannelSpec],
    batch_size: int,
    num_workers: int,
    context: Sleep2statContext,
) -> list[KaldiPSGDataset]:
    data_cfg = context.config.data
    if data_cfg.kaldi_data_root is None or data_cfg.kaldi_manifest is None:
        raise ValueError("data.backend=kaldi requires data.kaldi_data_root and data.kaldi_manifest.")
    manifest_path = data_cfg.kaldi_manifest
    if not manifest_path.is_absolute():
        manifest_path = data_cfg.kaldi_data_root / manifest_path
    channel_names = list(channel_specs)
    channel_input_dims = {name: spec.input_dim for name, spec in channel_specs.items()}
    records_by_split: dict[str, list[SleepRecord]] = {}
    for record in records:
        records_by_split.setdefault(record.split, []).append(record)

    datasets = []
    for split, split_records in records_by_split.items():
        wanted_keys = {str(record.metadata.get("sample_key", record.record_id)) for record in split_records}
        filtered_manifest = _write_filtered_kaldi_manifest(
            data_cfg.kaldi_data_root,
            manifest_path,
            split,
            wanted_keys,
        )
        dataset = KaldiPSGDataset(
            channel_names=channel_names,
            channel_input_dims=channel_input_dims,
            kaldi_data_root=data_cfg.kaldi_data_root,
            manifest=filtered_manifest,
            split=[split],
            max_tokens=data_cfg.max_tokens,
            mask_rate=0.0,
            randomly_select_channels=False,
            allow_missing_channels=False,
            min_channels=len(channel_names),
            bucket_by_available_channels=False,
            generative=False,
            is_train_set=False,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if dataset.data:
            datasets.append(dataset)
    return datasets


def _write_filtered_kaldi_manifest(root: Path, manifest_path: Path, split: str, wanted_keys: set[str]) -> Path:
    manifest = json.loads(manifest_path.read_text())
    split_spec = dict(manifest["splits"][str(split)])
    split_manifest = Path(split_spec["manifest"])
    if not split_manifest.is_absolute():
        split_manifest = root / split_manifest
    frame = pd.read_csv(split_manifest, low_memory=False)
    frame = frame[frame["sample_key"].astype(str).isin(wanted_keys)].reset_index(drop=True)

    output_dir = Path(tempfile.mkdtemp(prefix="sleep2stat_kaldi_"))
    filtered_csv = output_dir / f"{split}.csv"
    frame.to_csv(filtered_csv, index=False)
    split_spec["manifest"] = str(filtered_csv)
    manifest["splits"] = {str(split): split_spec}
    filtered_manifest = output_dir / "manifest.json"
    filtered_manifest.write_text(json.dumps(manifest))
    return filtered_manifest


@register_analyzer("sleep2vec_downstream")
class Sleep2vecDownstreamAnalyzer(BaseAnalyzer):
    def __init__(self, config: AnalyzerConfig):
        super().__init__(config)
        self.args: argparse.Namespace | None = None
        self.model = None
        self.threshold: float | None = None
        self.threshold_source: str | None = None
        self.move_to_device = None
        self.include_probabilities = True
        self.include_raw_logits = False

    def prepare(self, context: Sleep2statContext) -> None:
        namespace = self.config.namespace or "sleep2vec"
        common_mod = importlib.import_module(f"{namespace}.common")
        finetune_mod = importlib.import_module(f"{namespace}.sleep2vec_finetuning")
        utils_mod = importlib.import_module(f"{namespace}.utils")

        data_cfg = getattr(context.config, "data", None)
        args = argparse.Namespace(
            config=Path(self.config.config),
            label_name=self.config.label_name,
            device=context.device,
            batch_size=self.config.batch_size or context.batch_size or 12,
            num_workers=context.num_workers,
            data_backend=getattr(data_cfg, "backend", "npz"),
            kaldi_data_root=getattr(data_cfg, "kaldi_data_root", None),
            kaldi_manifest=getattr(data_cfg, "kaldi_manifest", None),
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
        self.threshold, self.threshold_source = self._resolve_threshold(checkpoint)
        self.include_probabilities = bool(context.config.outputs.include_probabilities) and bool(
            self.config.outputs.get("epoch_proba", True)
        )
        self.include_raw_logits = bool(context.config.outputs.include_raw_logits)

    def run(
        self,
        records: list[SleepRecord],
        context: Sleep2statContext,
        prior_results: list[AnalyzerResult] | None = None,
    ) -> tuple[list[AnalyzerResult], list[FailureRecord]]:
        if self.args is None or self.model is None or self.move_to_device is None:
            raise RuntimeError("Analyzer was not prepared.")
        if not records:
            return [], []
        results: list[AnalyzerResult] = []
        failures: list[FailureRecord] = []
        channel_specs = {name: context.config.signals.channels[name] for name in self.config.input_channels}
        datasets = _build_datasets(
            records=records,
            channel_specs=channel_specs,
            batch_size=self.config.batch_size or context.batch_size or self.args.batch_size,
            num_workers=context.num_workers,
            context=context,
        )
        stage_resolver = StageSourceResolver(records, prior_results or [])
        record_by_path = {str(record.path): record for record in records}
        record_by_id = {}
        for record in records:
            record_by_id[record.record_id] = record
            sample_key = record.metadata.get("sample_key")
            if sample_key is not None:
                record_by_id[str(sample_key)] = record
        retained_record_ids = {
            record.record_id
            for dataset in datasets
            for sample in getattr(dataset, "data", [])
            for record in [_record_for_sample(sample, record_by_path, record_by_id)]
            if record is not None
        }
        for record in records:
            if record.record_id not in retained_record_ids:
                failures.append(
                    FailureRecord(
                        record_id=record.record_id,
                        analyzer=self.config.name,
                        error_type="RecordUnavailable",
                        message=(
                            "Record was dropped before model inference; " "check duration and required input channels."
                        ),
                    )
                )
        if not retained_record_ids:
            return results, failures
        with torch.no_grad():
            for dataset in datasets:
                loader = dataset.dataloader(device=self.args.device)
                for batch in loader:
                    try:
                        results.extend(self._run_batch(batch, record_by_path, record_by_id, stage_resolver))
                    except Exception as exc:
                        batch_records = _records_for_batch(batch, record_by_path, record_by_id)
                        if len(batch_records) > 1:
                            for record in batch_records:
                                try:
                                    retry_results = self._run_single_record(
                                        record,
                                        context,
                                        channel_specs,
                                        record_by_path,
                                        record_by_id,
                                        stage_resolver,
                                    )
                                    if retry_results:
                                        results.extend(retry_results)
                                    else:
                                        failures.append(
                                            FailureRecord(
                                                record_id=record.record_id,
                                                analyzer=self.config.name,
                                                error_type="RecordUnavailable",
                                                message=(
                                                    "Record was dropped during single-record retry; "
                                                    "check duration and required input channels."
                                                ),
                                            )
                                        )
                                except Exception as retry_exc:
                                    failures.append(
                                        FailureRecord(
                                            record_id=record.record_id,
                                            analyzer=self.config.name,
                                            error_type=type(retry_exc).__name__,
                                            message=str(retry_exc),
                                        )
                                    )
                        else:
                            for record in batch_records:
                                failures.append(
                                    FailureRecord(
                                        record_id=record.record_id,
                                        analyzer=self.config.name,
                                        error_type=type(exc).__name__,
                                        message=str(exc),
                                    )
                                )
        return results, failures

    def _run_batch(
        self,
        batch: dict[str, Any],
        record_by_path: dict[str, SleepRecord],
        record_by_id: dict[str, SleepRecord],
        stage_resolver: StageSourceResolver | None,
    ) -> list[AnalyzerResult]:
        batch = self.move_to_device(batch, self.args.device)
        logits = self.model(batch)
        return self._decode_batch(batch, logits, record_by_path, record_by_id, stage_resolver)

    def _run_single_record(
        self,
        record: SleepRecord,
        context: Sleep2statContext,
        channel_specs: dict[str, ChannelSpec],
        record_by_path: dict[str, SleepRecord],
        record_by_id: dict[str, SleepRecord],
        stage_resolver: StageSourceResolver | None,
    ) -> list[AnalyzerResult]:
        datasets = _build_datasets(
            records=[record],
            channel_specs=channel_specs,
            batch_size=1,
            num_workers=context.num_workers,
            context=context,
        )
        output = []
        for dataset in datasets:
            for batch in dataset.dataloader(device=self.args.device):
                output.extend(self._run_batch(batch, record_by_path, record_by_id, stage_resolver))
        return output

    def _decode_batch(
        self,
        batch: dict[str, Any],
        logits: torch.Tensor,
        record_by_path: dict[str, SleepRecord],
        record_by_id: dict[str, SleepRecord] | None = None,
        stage_resolver: StageSourceResolver | None = None,
    ) -> list[AnalyzerResult]:
        label_name = self.config.label_name
        if label_name == "ahi":
            if self.threshold is None:
                raise ValueError(
                    f"Analyzer {self.config.name!r} requires ahi_eval_threshold in checkpoint or explicit threshold."
                )
            postprocess = self._ahi_postprocess()
            return decode_ahi_logits(
                self.config.name,
                logits,
                batch,
                record_by_path,
                record_by_id=record_by_id,
                threshold=self.threshold,
                threshold_source=self.threshold_source or "unknown",
                include_probabilities=self.include_probabilities,
                min_event_duration_sec=postprocess["min_event_duration_sec"],
                merge_tolerance_sec=postprocess["merge_tolerance_sec"],
                denominator_stage_source=postprocess["denominator_stage_source"],
                output_second_alignment=postprocess["output_second_alignment"],
                output_event_alignment=postprocess["output_event_alignment"],
                stage_resolver=stage_resolver,
            )
        if self.args and getattr(self.args, "is_classification", False):
            return decode_classification_logits(
                self.config.name,
                logits,
                batch,
                record_by_path,
                record_by_id=record_by_id,
                include_probabilities=self.include_probabilities,
                include_logits=self.include_raw_logits,
            )
        return decode_regression_logits(self.config.name, logits, batch, record_by_path, record_by_id=record_by_id)

    def _resolve_threshold(self, checkpoint: Any) -> tuple[float | None, str | None]:
        post_threshold = self.config.postprocess.get("threshold")
        if isinstance(post_threshold, dict):
            explicit = post_threshold.get("value")
            source = str(post_threshold.get("source", "postprocess"))
        else:
            explicit = post_threshold
            source = "postprocess"
        if explicit not in (None, ""):
            return float(explicit), source
        explicit = self.config.threshold
        source = "config"
        if isinstance(explicit, dict):
            source = str(explicit.get("source", source))
            explicit = explicit.get("value")
        if explicit not in (None, ""):
            return float(explicit), source
        if isinstance(checkpoint, dict) and checkpoint.get("ahi_eval_threshold") is not None:
            return float(checkpoint["ahi_eval_threshold"]), "checkpoint"
        return None, None

    def _ahi_postprocess(self) -> dict[str, Any]:
        postprocess = dict(self.config.postprocess or {})
        return {
            "min_event_duration_sec": int(postprocess.get("min_event_duration_sec", 10)),
            "merge_tolerance_sec": int(postprocess.get("merge_tolerance_sec", 3)),
            "denominator_stage_source": postprocess.get("denominator_stage_source"),
            "output_second_alignment": bool(postprocess.get("output_second_alignment", True)),
            "output_event_alignment": bool(postprocess.get("output_event_alignment", True)),
        }


def _record_for_sample(
    sample: SampleIndex,
    record_by_path: dict[str, SleepRecord],
    record_by_id: dict[str, SleepRecord],
) -> SleepRecord | None:
    record = record_by_id.get(str(sample.id))
    if record is not None:
        return record
    if sample.metadata:
        record_id = sample.metadata.get("record_id")
        if record_id is not None and str(record_id) in record_by_id:
            return record_by_id[str(record_id)]
        path = sample.metadata.get("path")
        if path is not None:
            return record_by_path.get(str(path))
    return None


def _record_for_batch_item(
    batch: dict[str, Any],
    idx: int,
    path: Any,
    record_by_path: dict[str, SleepRecord],
    record_by_id: dict[str, SleepRecord] | None,
) -> SleepRecord:
    if record_by_id:
        batch_id = _batch_value(batch.get("id"), idx)
        if batch_id is not None and str(batch_id) in record_by_id:
            return record_by_id[str(batch_id)]
        record_id = _batch_value(batch.get("metadata", {}).get("record_id"), idx)
        if record_id is not None and str(record_id) in record_by_id:
            return record_by_id[str(record_id)]
    return record_by_path[str(path)]


def _records_for_batch(
    batch: dict[str, Any],
    record_by_path: dict[str, SleepRecord],
    record_by_id: dict[str, SleepRecord],
) -> list[SleepRecord]:
    paths = list(batch.get("metadata", {}).get("path", []))
    ids = batch.get("id", [])
    count = max(len(paths), len(ids))
    records = []
    for idx in range(count):
        path = paths[idx] if idx < len(paths) else None
        record = None
        if path is not None:
            record = _record_for_batch_item(batch, idx, path, record_by_path, record_by_id)
        else:
            batch_id = _batch_value(batch.get("id"), idx)
            if batch_id is not None:
                record = record_by_id.get(str(batch_id))
        if record is not None and record.record_id not in {item.record_id for item in records}:
            records.append(record)
    return records


def _batch_value(value: Any, idx: int) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().reshape(-1)[idx].item()
    try:
        return value[idx]
    except (TypeError, IndexError, KeyError):
        return None


def decode_classification_logits(
    analyzer_name: str,
    logits: torch.Tensor,
    batch: dict[str, Any],
    record_by_path: dict[str, SleepRecord],
    *,
    record_by_id: dict[str, SleepRecord] | None = None,
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
            record = _record_for_batch_item(batch, idx, path, record_by_path, record_by_id)
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
        record = _record_for_batch_item(batch, idx, path, record_by_path, record_by_id)
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
    *,
    record_by_id: dict[str, SleepRecord] | None = None,
) -> list[AnalyzerResult]:
    preds = logits.detach().to(torch.float32).cpu().reshape(logits.shape[0], -1).mean(dim=1).numpy()
    results = []
    for idx, path in enumerate(batch["metadata"]["path"]):
        record = _record_for_batch_item(batch, idx, path, record_by_path, record_by_id)
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
    record_by_id: dict[str, SleepRecord] | None = None,
    threshold: float,
    threshold_source: str = "config",
    include_probabilities: bool = True,
    min_event_duration_sec: int = 10,
    merge_tolerance_sec: int = 3,
    denominator_stage_source: str | None = None,
    output_second_alignment: bool = True,
    output_event_alignment: bool = True,
    stage_resolver: StageSourceResolver | None = None,
) -> list[AnalyzerResult]:
    probs = torch.sigmoid(logits.detach().to(torch.float32).cpu()).numpy()
    paths = list(batch["metadata"]["path"])
    token_starts = _tensor_list(batch.get("token_start"), default=0, count=len(paths))
    lengths = _tensor_list(batch.get("length"), default=1, count=len(paths))
    results = []
    for idx, path in enumerate(paths):
        record = _record_for_batch_item(batch, idx, path, record_by_path, record_by_id)
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
        model_hours = valid_seconds / 3600.0 if valid_seconds > 0 else 0.0
        recording_hours = record.duration_sec / 3600.0 if record.duration_sec > 0 else 0.0
        event_rate_model = _event_rate(len(events), model_hours)
        event_rate_recording = _event_rate(len(events), recording_hours)
        coverage_ratio = float(valid_seconds / record.duration_sec) if record.duration_sec > 0 else np.nan
        truncated_by_max_tokens = bool(
            record.max_tokens > 0
            and int(lengths[idx]) >= int(record.max_tokens)
            and valid_seconds < record.duration_sec
        )
        warnings = []
        # Model/recording-hour rates are QC outputs, not clinical AHI unless a sleep-stage denominator is available.
        night = {
            f"{analyzer_name}_pred_event_rate_per_model_hour": event_rate_model,
            f"{analyzer_name}_pred_event_rate_per_recording_hour": event_rate_recording,
            f"{analyzer_name}_pred_event_count": int(len(events)),
            f"{analyzer_name}_model_denominator_hours": float(model_hours),
            f"{analyzer_name}_recording_denominator_hours": float(recording_hours),
            f"{analyzer_name}_covered_duration_sec": int(valid_seconds),
            f"{analyzer_name}_coverage_ratio_recording": coverage_ratio,
            f"{analyzer_name}_truncated_by_max_tokens": truncated_by_max_tokens,
            f"{analyzer_name}_threshold": float(threshold),
            f"{analyzer_name}_threshold_source": threshold_source,
            f"{analyzer_name}_min_event_duration_sec": int(min_event_duration_sec),
            f"{analyzer_name}_merge_tolerance_sec": int(merge_tolerance_sec),
            f"{analyzer_name}_denominator_stage_source": denominator_stage_source,
        }
        if denominator_stage_source:
            denominators = (
                stage_resolver.get_denominator_hours(record.record_id, denominator_stage_source)
                if stage_resolver
                else None
            )
            if denominators is None:
                warnings.append(
                    f"denominator stage source {denominator_stage_source!r} not found; "
                    "only recording denominator was emitted"
                )
            else:
                stage_at_onset = (
                    stage_resolver.stage_at_seconds(
                        record.record_id,
                        denominator_stage_source,
                        events["onset_sec"].to_numpy() if not events.empty else np.asarray([], dtype=float),
                    )
                    if stage_resolver
                    else None
                )
                stage_at_onset = np.asarray([], dtype=np.int64) if stage_at_onset is None else stage_at_onset
                sleep_count = int(np.sum(np.isin(stage_at_onset, [1, 2, 3, 4])))
                rem_count = int(np.sum(stage_at_onset == 4))
                nrem_count = int(np.sum(np.isin(stage_at_onset, [1, 2, 3])))
                if not events.empty:
                    events = events.copy()
                    events[f"{analyzer_name}_stage_at_onset"] = stage_at_onset
                sleep_ahi = _event_rate(sleep_count, denominators["sleep"])
                # REM/NREM denominators count events by onset stage; overlap-based assignment is out of scope here.
                night[f"{analyzer_name}_sleep_denominator_hours"] = denominators["sleep"]
                night[f"{analyzer_name}_rem_denominator_hours"] = denominators["rem"]
                night[f"{analyzer_name}_nrem_denominator_hours"] = denominators["nrem"]
                night[f"{analyzer_name}_stage_assignment"] = "onset"
                night[f"{analyzer_name}_pred_AHI_sleep_denominator"] = sleep_ahi
                night[f"{analyzer_name}_pred_REM_AHI_onset_stage"] = _event_rate(rem_count, denominators["rem"])
                night[f"{analyzer_name}_pred_NREM_AHI_onset_stage"] = _event_rate(nrem_count, denominators["nrem"])
                night[f"{analyzer_name}_pred_ahi"] = sleep_ahi
        results.append(
            AnalyzerResult(
                analyzer_name,
                record.record_id,
                second=second if output_second_alignment else None,
                events=events if output_event_alignment else None,
                night=night,
                warnings=warnings,
            )
        )
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


def _event_rate(event_count: int, denominator_hours: float) -> float:
    return float(event_count / denominator_hours) if denominator_hours > 0 else np.nan


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
