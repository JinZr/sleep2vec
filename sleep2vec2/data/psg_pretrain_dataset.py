import os
from pathlib import Path
import typing as t

import pandas as pd

from sleep2vec2.data.default_dataset import DefaultDataset, SampleIndex
from sleep2vec2.data.survival import attach_survival_metadata, load_survival_label_table
from sleep2vec2.data.utils import default_extractor, default_mlm_mask_generator, default_tokenizer, window

PAD_STAGE = -1


def _normalize_channel_input_dims(channel_input_dims: t.Mapping[str, int]) -> dict[str, int]:
    normalized = {str(name): int(dim) for name, dim in channel_input_dims.items()}
    invalid = sorted(name for name, dim in normalized.items() if dim <= 0)
    if invalid:
        raise ValueError(f"channel_input_dims must be positive for all channels. Invalid: {invalid}")
    return normalized


def _build_channel_registry(
    *,
    channel_names: t.Sequence[str],
    channel_input_dims: t.Mapping[str, int],
    mask_rate: float,
) -> dict[str, tuple[t.Callable, t.Callable, t.Callable]]:
    registry: dict[str, tuple[t.Callable, t.Callable, t.Callable]] = {
        "stage5": (
            default_extractor("stage5", 1),
            default_tokenizer(1),
            default_mlm_mask_generator(0.0),
        ),
        "ahi": (
            default_extractor("ahi", 30, source_name="ah_event"),
            default_tokenizer(30),
            default_mlm_mask_generator(0.0),
        ),
    }
    missing: list[str] = []
    for name in channel_names:
        if name in registry:
            continue
        frames_per_token = channel_input_dims.get(name)
        if frames_per_token is None:
            missing.append(name)
            continue
        registry[name] = (
            default_extractor(name, frames_per_token),
            default_tokenizer(frames_per_token),
            default_mlm_mask_generator(mask_rate),
        )

    if missing:
        raise ValueError(
            "Missing channel_input_dims for requested channels: "
            f"{sorted(missing)}. Provide explicit YAML-driven widths for all non-built-in channels."
        )
    return registry


class PSGPretrainDataset(DefaultDataset):
    def __init__(
        self,
        channel_names: t.List[str],
        save_preset_path: str,
        load_preset_path: str,
        index: str,
        split: t.List[str],
        max_tokens: int,
        *,
        channel_input_dims: t.Mapping[str, int],
        token_sec: int = 30,
        stride_tokens: int = 0,  # 0 for truncation
        mask_rate: float = 0.15,
        few_shot: int | float | None = None,  # ← 新增参数
        meta_data_names: t.Optional[t.List[str]] = None,  # ← 新增参数
        meta_data_regression_names: t.Optional[t.List[str]] = None,
        required_metadata_names: t.Optional[t.Sequence[str]] = None,
        sources: t.Optional[t.List[str]] = None,  # ← 新增参数
        pair_selector: t.Any | None = None,
        randomly_select_channels: bool = True,
        min_channels: int = 2,
        allow_missing_channels: bool = False,
        bucket_by_available_channels: bool = False,
        train_pair_probs: t.Optional[t.Mapping[tuple[str, str], float]] = None,
        train_pair_track_unique_samples: bool = False,
        weighted_random_sampler: bool = False,
        weighted_random_sampler_target: str | None = None,
        survival_label_config: t.Any | None = None,
        survival_output_dim: int | None = None,
        generative: bool = False,
        is_train_set: bool = True,
        filter_max_workers: int | None = None,
        **kwargs: t.Any,
    ) -> None:

        self.channel_names = channel_names
        self.channel_input_dims = _normalize_channel_input_dims(channel_input_dims)
        self.randomly_select_channels = randomly_select_channels
        self.min_channels = min_channels
        self.allow_missing_channels = allow_missing_channels
        self.bucket_by_available_channels = bucket_by_available_channels
        self.train_pair_probs = None if train_pair_probs is None else dict(train_pair_probs)
        self.train_pair_track_unique_samples = bool(train_pair_track_unique_samples)
        self.token_sec = token_sec
        self.generative = generative
        self.is_train_set = is_train_set
        meta_data_names = meta_data_names or []
        built_in_ahi_runtime_metadata = "ahi" in channel_names
        sources = sources or []
        survival_labels = None

        split_list = [split] if isinstance(split, str) else list(split or [])
        survival_key_column = getattr(survival_label_config, "key_column", None)

        if not load_preset_path:
            survival_labels = load_survival_label_table(survival_label_config, survival_output_dim)
            if survival_labels is not None:
                survival_key_column = survival_labels.key_column
            read_csv_kwargs: dict[str, t.Any] = {"low_memory": False}
            if survival_key_column is not None:
                read_csv_kwargs["converters"] = {str(survival_key_column): str}

            # --- 关键改动：读取一个或多个 CSV 并合并 ---
            def _load_index_df(
                idx: t.Union[str, os.PathLike, t.List[t.Union[str, os.PathLike]]],
            ) -> pd.DataFrame:
                # 单个路径
                if isinstance(idx, (str, os.PathLike, Path)):
                    df = pd.read_csv(idx, **read_csv_kwargs)
                    if "source" not in df.columns:
                        df["source"] = str(idx)
                    else:
                        df["source"] = df["source"].where(df["source"].notna(), str(idx))
                    return df

                # 多个路径
                if isinstance(idx, (list, tuple)):
                    dfs = []
                    for p in idx:
                        dfi = pd.read_csv(p, **read_csv_kwargs)
                        if "source" not in dfi.columns:
                            dfi["source"] = str(p)
                        else:
                            dfi["source"] = dfi["source"].where(dfi["source"].notna(), str(p))
                        dfs.append(dfi)
                    if not dfs:
                        raise ValueError("index 列表为空。")
                    return pd.concat(dfs, ignore_index=True)

            csv = _load_index_df(index)
            if split_list:
                if "split" not in csv.columns:
                    raise KeyError("Expected 'split' column in index CSV for split filtering.")
                csv = csv[csv["split"].isin(split_list)].reset_index(drop=True)

            data: t.List[SampleIndex] = []

            # 遍历所有 rows
            for i, (_, row) in enumerate(csv.iterrows()):
                # ✅ 从指定列中提取 metadata
                metadata = {
                    "source": row["source"],
                    "path": row["path"],
                    "split": row["split"],
                }
                for optional_meta_name in ("age", "sex"):
                    if optional_meta_name in row.index:
                        metadata[optional_meta_name] = row[optional_meta_name]
                if survival_labels is not None:
                    if survival_labels.key_column not in row.index:
                        raise ValueError(
                            f"Required survival key column '{survival_labels.key_column}' is missing from index CSV."
                        )
                    attach_survival_metadata(metadata, row[survival_labels.key_column], survival_labels)

                for meta_data_name in meta_data_names:
                    # Built-in AHI summary scalars come from NPZ backfill, not CSV columns.
                    if built_in_ahi_runtime_metadata and meta_data_name in {"ahi", "tst"}:
                        continue
                    if meta_data_name not in row.index:
                        raise ValueError(f"Required metadata column '{meta_data_name}' is missing from index CSV.")
                    metadata[meta_data_name] = row[meta_data_name]

                # 需要划分为 n 个 token
                n = int(row["duration"] // self.token_sec)
                if n <= 0:
                    continue

                # stride_tokens = 0 代表只取前面的1535个token
                # 否则按滑窗 ceil((n - 1535) / stride_tokens) 取多个滑窗
                for left, right in window(n, max_tokens, stride_tokens):
                    data.append(
                        SampleIndex(
                            id=i,
                            path=row["path"],
                            start=left,
                            end=right,
                            metadata=metadata,  # ✅ 添加 metadata
                        )
                    )
        else:
            data = None

        effective_survival_output_dim = survival_output_dim if load_preset_path else None
        if survival_labels is not None:
            effective_survival_output_dim = len(survival_labels.label_names)

        registry = _build_channel_registry(
            channel_names=channel_names,
            channel_input_dims=self.channel_input_dims,
            mask_rate=mask_rate,
        )

        extractors = {name: registry[name][0] for name in channel_names}
        tokenizers = {name: registry[name][1] for name in channel_names}
        mask_generators = {name: registry[name][2] for name in channel_names}

        super().__init__(
            save_preset_path,
            load_preset_path,
            data,
            split_list,
            extractors=extractors,
            tokenizers=tokenizers,
            mask_generators=mask_generators,
            few_shot=few_shot,
            meta_data_names=meta_data_names,
            meta_data_regression_names=meta_data_regression_names,
            required_metadata_names=required_metadata_names,
            sources=sources,
            pair_selector=pair_selector,
            weighted_random_sampler=weighted_random_sampler,
            weighted_random_sampler_target=weighted_random_sampler_target,
            survival_output_dim=effective_survival_output_dim,
            survival_key_column=survival_key_column,
            dataloader_config=kwargs,
            filter_max_workers=filter_max_workers,
        )
