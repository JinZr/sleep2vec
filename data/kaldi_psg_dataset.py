from __future__ import annotations

import json
from pathlib import Path
import typing as t

import pandas as pd
import torch

from data.default_dataset import DefaultDataset, SampleIndex
from data.kaldi_io import KaldiChannelSpec, KaldiReaderPool
from data.psg_pretrain_dataset import _build_channel_registry, _normalize_channel_input_dims


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
        sources: list[str] | None = None,
        pair_selector: t.Any | None = None,
        randomly_select_channels: bool = True,
        allow_missing_channels: bool = False,
        min_channels: int = 2,
        bucket_by_available_channels: bool = False,
        train_pair_probs: dict[tuple[str, str], float] | None = None,
        train_pair_track_unique_samples: bool = False,
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

        channel_specs = self._load_channel_specs(channel_input_dims)
        self.reader_pool = KaldiReaderPool(self.kaldi_data_root, channel_specs)

        data = self._load_manifest_samples(split_list, max_tokens)
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
            sources=sources,
            pair_selector=pair_selector,
            dataloader_config=dataloader_kwargs,
        )

    def _load_channel_specs(
        self,
        channel_input_dims: t.Mapping[str, int],
    ) -> dict[str, KaldiChannelSpec]:
        manifest_json_path = self.kaldi_data_root / "manifest.json"
        if not manifest_json_path.exists():
            raise FileNotFoundError(f"Kaldi manifest.json not found: {manifest_json_path}")

        manifest_data = json.loads(manifest_json_path.read_text())
        raw_channels = manifest_data.get("channels")
        if not isinstance(raw_channels, dict):
            raise ValueError("Kaldi manifest.json must contain a 'channels' mapping.")

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
            input_dim = int(raw_spec["input_dim"])
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

    def _load_manifest_samples(self, split: list[str], max_tokens: int) -> list[SampleIndex]:
        if not self.manifest.exists():
            raise FileNotFoundError(f"Kaldi manifest.csv not found: {self.manifest}")

        df = pd.read_csv(self.manifest, low_memory=False)
        required = {"sample_key", "path", "split", "token_start", "token_end", "available_channels"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Kaldi manifest.csv is missing required column(s): {missing}.")
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
                    payload={"available_channels": selected},
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
        expected_len = int(src.end) - int(src.start)
        for channel in chosen_channels:
            arr = self.reader_pool.read_matrix(channel, str(src.id))
            if arr.shape[0] != expected_len:
                raise ValueError(
                    f"Kaldi matrix for channel {channel!r}, key {src.id!r} has {arr.shape[0]} rows, "
                    f"expected {expected_len} from manifest token_start={src.start}, token_end={src.end}."
                )
            tokens[channel] = torch.from_numpy(arr).to(torch.float32)
        masks = {channel: self.mask_generators[channel](tokens[channel]) for channel in chosen_channels}
        return {}, tokens, masks, dict(src.metadata)
