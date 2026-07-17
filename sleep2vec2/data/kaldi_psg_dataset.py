from __future__ import annotations

import json
from pathlib import Path
import typing as t

import pandas as pd
import torch

from sleep2vec2.data.default_dataset import DefaultDataset, SampleIndex
from sleep2vec2.data.kaldi_io import KaldiChannelSpec, KaldiReaderPool
from sleep2vec2.data.multilabel import attach_multilabel_metadata, load_multilabel_label_table
from sleep2vec2.data.psg_pretrain_dataset import _build_channel_registry, _normalize_channel_input_dims
from sleep2vec2.data.survival import attach_survival_metadata, load_survival_label_table


def _is_missing(value: t.Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _first_present(row: pd.Series, names: t.Sequence[str], default: t.Any = None) -> t.Any:
    for name in names:
        if name in row.index and not _is_missing(row[name]):
            return row[name]
    return default


class KaldiPSGDataset(DefaultDataset):
    def __init__(
        self,
        channel_names: list[str],
        channel_input_dims: t.Mapping[str, int],
        kaldi_data_root: str | Path,
        manifest: str | Path,
        split: list[str],
        max_tokens: int,
        mask_rate: float,
        few_shot: int | float | None = None,
        meta_data_names: list[str] | None = None,
        meta_data_regression_names: list[str] | None = None,
        required_metadata_names: t.Sequence[str] | None = None,
        sources: list[str] | None = None,
        pair_selector: t.Any | None = None,
        randomly_select_channels: bool = True,
        allow_missing_channels: bool = False,
        min_channels: int = 2,
        bucket_by_available_channels: bool = False,
        train_pair_probs: dict[tuple[str, str], float] | None = None,
        train_pair_track_unique_samples: bool = False,
        weighted_random_sampler: bool = False,
        weighted_random_sampler_target: str | None = None,
        survival_label_config: t.Any | None = None,
        survival_output_dim: int | None = None,
        multilabel_label_config: t.Any | None = None,
        multilabel_output_dim: int | None = None,
        generative: bool = False,
        is_train_set: bool = True,
        **dataloader_kwargs: t.Any,
    ) -> None:
        self.channel_names = [str(name) for name in channel_names]
        self.kaldi_data_root = Path(kaldi_data_root).expanduser()
        self.manifest = Path(manifest).expanduser()
        self.randomly_select_channels = randomly_select_channels
        self.min_channels = min_channels
        self.allow_missing_channels = allow_missing_channels
        self.bucket_by_available_channels = bucket_by_available_channels
        self.train_pair_probs = None if train_pair_probs is None else dict(train_pair_probs)
        self.train_pair_track_unique_samples = bool(train_pair_track_unique_samples)
        self.generative = generative
        self.is_train_set = is_train_set

        split_list = [split] if isinstance(split, str) else list(split or [])
        meta_data_names = meta_data_names or []
        sources = sources or []
        if len(split_list) != 1:
            raise ValueError(f"KaldiPSGDataset expects exactly one split, got {split_list}.")

        manifest_data = self._load_manifest_json()
        survival_labels = load_survival_label_table(survival_label_config, survival_output_dim)
        multilabel_labels = load_multilabel_label_table(multilabel_label_config, multilabel_output_dim)
        split_name = str(split_list[0])
        raw_splits = manifest_data.get("splits")
        if not isinstance(raw_splits, dict) or split_name not in raw_splits:
            raise ValueError(f"Kaldi manifest.json is missing requested split {split_name!r}.")
        split_spec = raw_splits[split_name]
        if not isinstance(split_spec, dict):
            raise ValueError(f"Kaldi manifest split spec for {split_name!r} must be a mapping.")
        self.manifest = self.kaldi_data_root / split_spec["manifest"]

        channel_specs = self._load_channel_specs(channel_input_dims, split_spec)
        self.reader_pool = KaldiReaderPool(self.kaldi_data_root, channel_specs)

        data = self._load_manifest_samples(split_list, max_tokens, survival_labels, multilabel_labels)
        if not data:
            raise ValueError("No Kaldi manifest rows remain after split/channel filtering.")

        registry = _build_channel_registry(
            channel_names=self.channel_names,
            channel_input_dims=self.channel_input_dims,
            mask_rate=mask_rate,
        )
        mask_generators = {name: registry[name][2] for name in self.channel_names}

        super().__init__(
            save_preset_path=None,
            load_preset_path=None,
            data=data,
            split=split_list,
            extractors={},
            tokenizers={},
            mask_generators=mask_generators,
            few_shot=few_shot,
            meta_data_names=meta_data_names,
            meta_data_regression_names=meta_data_regression_names,
            required_metadata_names=required_metadata_names,
            sources=sources,
            pair_selector=pair_selector,
            weighted_random_sampler=weighted_random_sampler,
            weighted_random_sampler_target=weighted_random_sampler_target,
            survival_output_dim=None if survival_labels is None else len(survival_labels.label_names),
            survival_key_column=None if survival_labels is None else survival_labels.key_column,
            multilabel_output_dim=None if multilabel_labels is None else len(multilabel_labels.label_names),
            multilabel_key_column=None if multilabel_labels is None else multilabel_labels.key_column,
            dataloader_config=dataloader_kwargs,
        )

    def _load_manifest_json(self) -> dict[str, t.Any]:
        if not self.manifest.exists():
            raise FileNotFoundError(f"Kaldi manifest.json not found: {self.manifest}")
        manifest_data = json.loads(self.manifest.read_text())
        return manifest_data

    def _load_channel_specs(
        self,
        channel_input_dims: t.Mapping[str, int],
        split_spec: t.Mapping[str, t.Any],
    ) -> dict[str, KaldiChannelSpec]:
        raw_channels = split_spec.get("channels")
        if not isinstance(raw_channels, dict):
            raise ValueError("Kaldi manifest split spec must contain a 'channels' mapping.")

        provided_dims = _normalize_channel_input_dims(channel_input_dims)
        specs: dict[str, KaldiChannelSpec] = {}
        resolved_dims: dict[str, int] = {}
        missing = [name for name in self.channel_names if name not in raw_channels]
        if missing:
            raise ValueError(f"Kaldi manifest is missing requested channel(s): {sorted(missing)}.")

        for channel in self.channel_names:
            raw_spec = raw_channels[channel]
            if not isinstance(raw_spec, dict):
                raise ValueError(f"Kaldi manifest channel spec for {channel!r} must be a mapping.")
            if "input_dim" in raw_spec:
                input_dim = int(raw_spec["input_dim"])
            elif "hidden_size" in raw_spec:
                input_dim = int(raw_spec["hidden_size"])
            else:
                raise ValueError(f"Kaldi manifest channel spec for {channel!r} must contain input_dim or hidden_size.")
            if channel in provided_dims and provided_dims[channel] != input_dim:
                raise ValueError(
                    f"channel_input_dims[{channel!r}]={provided_dims[channel]} does not match "
                    f"Kaldi manifest input_dim={input_dim}."
                )
            resolved_dims[channel] = input_dim
            specs[channel] = KaldiChannelSpec(
                name=channel,
                input_dim=input_dim,
                scp_path=Path(raw_spec["scp"]),
            )

        self.channel_input_dims = {**provided_dims, **resolved_dims}
        return specs

    def _load_manifest_samples(
        self, split: list[str], max_tokens: int, survival_labels: t.Any | None, multilabel_labels: t.Any | None
    ) -> list[SampleIndex]:
        if not self.manifest.exists():
            raise FileNotFoundError(f"Kaldi split manifest CSV not found: {self.manifest}")

        read_csv_kwargs: dict[str, t.Any] = {"low_memory": False}
        key_converters = {}
        if survival_labels is not None:
            key_converters[str(survival_labels.key_column)] = str
        if multilabel_labels is not None:
            key_converters[str(multilabel_labels.key_column)] = str
        if key_converters:
            read_csv_kwargs["converters"] = key_converters
        df = pd.read_csv(self.manifest, **read_csv_kwargs)
        required = {"sample_key", "path", "split", "token_start", "token_end", "available_channels"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Kaldi split manifest CSV is missing required column(s): {missing}.")
        if split:
            df = df[df["split"].isin(split)].reset_index(drop=True)

        data: list[SampleIndex] = []
        requested = set(self.channel_names)
        for _, row in df.iterrows():
            available = self._parse_available_channels(row)
            selected = [channel for channel in self.channel_names if channel in available]
            if self.allow_missing_channels:
                if len(selected) < self.min_channels:
                    continue
            elif set(selected) != requested:
                continue

            start = int(row["token_start"])
            end = int(row["token_end"])
            num_tokens = end - start
            if num_tokens < 1 or num_tokens > max_tokens:
                raise ValueError(
                    f"Kaldi sample {row['sample_key']!r} has invalid token span "
                    f"token_start={start}, token_end={end}, max_tokens={max_tokens}."
                )
            has_num_tokens = "num_tokens" in row.index and not _is_missing(row["num_tokens"])
            if has_num_tokens and int(row["num_tokens"]) != num_tokens:
                raise ValueError(
                    f"Kaldi sample {row['sample_key']!r} has num_tokens={row['num_tokens']} "
                    f"but token span length is {num_tokens}."
                )
            matrix_rows = num_tokens
            has_matrix_rows = "matrix_rows" in row.index and not _is_missing(row["matrix_rows"])
            if has_matrix_rows:
                matrix_rows = int(row["matrix_rows"])
                if matrix_rows < 1 or matrix_rows > max_tokens:
                    raise ValueError(
                        f"Kaldi sample {row['sample_key']!r} has invalid matrix_rows={matrix_rows}, "
                        f"max_tokens={max_tokens}."
                    )

            metadata = row.to_dict()
            source = _first_present(row, ("source", "dataset", "sample_source"), "nan")
            dataset = _first_present(row, ("dataset", "sample_source", "source"), source)
            metadata.update(
                {
                    "source": source,
                    "dataset": dataset,
                    "path": row["path"],
                    "split": row["split"],
                }
            )
            if survival_labels is not None:
                if survival_labels.key_column not in row.index:
                    raise ValueError(
                        f"Required survival key column '{survival_labels.key_column}' is missing from Kaldi manifest."
                    )
                attach_survival_metadata(metadata, row[survival_labels.key_column], survival_labels)
            if multilabel_labels is not None:
                if multilabel_labels.key_column not in row.index:
                    raise ValueError(
                        f"Required multilabel key column '{multilabel_labels.key_column}' "
                        "is missing from Kaldi manifest."
                    )
                attach_multilabel_metadata(metadata, row[multilabel_labels.key_column], multilabel_labels)
            if "ahi" in requested and (
                "ahi" not in selected
                or "tst" not in row.index
                or _is_missing(row.get("ahi"))
                or _is_missing(row.get("tst"))
            ):
                continue

            data.append(
                SampleIndex(
                    id=str(row["sample_key"]),
                    path=str(row["path"]),
                    start=start,
                    end=end,
                    payload={"available_channels": selected, "matrix_rows": matrix_rows},
                    metadata=metadata,
                )
            )
        return data

    def _parse_available_channels(self, row: pd.Series) -> set[str]:
        raw = row["available_channels"]
        if isinstance(raw, str):
            channels = json.loads(raw)
        else:
            channels = raw
        if not isinstance(channels, list):
            raise ValueError(f"Kaldi sample {row['sample_key']!r} has invalid available_channels: {raw!r}.")
        return {str(channel) for channel in channels}

    def _filter_valid_sample_indices(
        self,
        data: t.Sequence[SampleIndex],
        *,
        filter_max_workers: int | None,
    ) -> list[SampleIndex]:
        return list(data)

    def filter_with_metadata(self) -> list[SampleIndex]:
        selected = super().filter_with_metadata()
        if not selected:
            raise ValueError("No Kaldi manifest rows remain after split/channel/metadata filtering.")
        return selected

    def _get_available_channels_for_src(self, src: SampleIndex) -> set[str]:
        return set(src.payload["available_channels"]) & set(self.channel_names)

    def _load_tokens_for_src(
        self,
        src: SampleIndex,
        chosen_channels: list[str],
    ) -> tuple[dict, dict, dict, dict]:
        tokens = {}
        expected_len = int(src.payload.get("matrix_rows", int(src.end) - int(src.start)))
        for channel in chosen_channels:
            arr = self.reader_pool.read_matrix(channel, str(src.id))
            if arr.shape[0] != expected_len:
                raise ValueError(
                    f"Kaldi matrix for channel {channel!r}, key {src.id!r} has {arr.shape[0]} rows, "
                    f"expected {expected_len} from manifest matrix_rows/token span "
                    f"(token_start={src.start}, token_end={src.end})."
                )
            tokens[channel] = torch.from_numpy(arr).to(torch.float32)
        masks = {channel: self.mask_generators[channel](tokens[channel]) for channel in chosen_channels}
        return {}, tokens, masks, dict(src.metadata)
