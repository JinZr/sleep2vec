import os
from pathlib import Path
import typing as t

import pandas as pd

from data.default_dataset import DefaultDataset, SampleIndex
from data.utils import default_extractor, default_mlm_mask_generator, default_tokenizer, window

PAD_STAGE = -1


class PSGPretrainDataset(DefaultDataset):
    def __init__(
        self,
        channel_names: t.List[str],
        save_preset_path: str,
        load_preset_path: str,
        index: str,
        split: t.List[str],
        max_tokens: int,
        token_sec: int = 30,
        stride_tokens: int = 0,  # 0 for truncation
        mask_rate: float = 0.15,
        use_legacy_body_movement: bool = False,
        few_shot: int | float | None = None,  # ← 新增参数
        meta_data_names: t.Optional[t.List[str]] = None,  # ← 新增参数
        sources: t.Optional[t.List[str]] = None,  # ← 新增参数
        randomly_select_channels: bool = True,
        min_channels: int = 2,
        allow_missing_channels: bool = False,
        generative: bool = False,
        is_train_set: bool = True,
        **kwargs: t.Any,
    ) -> None:

        self.channel_names = channel_names
        self.randomly_select_channels = randomly_select_channels
        self.min_channels = min_channels
        self.allow_missing_channels = allow_missing_channels
        self.token_sec = token_sec
        self.generative = generative
        self.is_train_set = is_train_set
        meta_data_names = meta_data_names or []
        sources = sources or []

        if not load_preset_path:
            # --- 关键改动：读取一个或多个 CSV 并合并 ---
            def _load_index_df(
                idx: t.Union[str, os.PathLike, t.List[t.Union[str, os.PathLike]]],
            ) -> pd.DataFrame:
                # 单个路径
                if isinstance(idx, (str, os.PathLike, Path)):
                    df = pd.read_csv(idx, low_memory=False)
                    df["source"] = str(idx)  # 可选：标注来源文件
                    return df

                # 多个路径
                if isinstance(idx, (list, tuple)):
                    dfs = []
                    for p in idx:
                        dfi = pd.read_csv(p, low_memory=False)
                        dfi["source"] = str(p)  # 可选：标注来源文件
                        dfs.append(dfi)
                    if not dfs:
                        raise ValueError("index 列表为空。")
                    return pd.concat(dfs, ignore_index=True)

            csv = _load_index_df(index)

            data: t.List[SampleIndex] = []

            # 遍历所有 rows
            for i, (_, row) in enumerate(csv.iterrows()):
                # ✅ 从指定列中提取 metadata
                metadata = {
                    "age": row["age"],
                    "sex": row["sex"],
                    "source": row["source"],
                    "path": row["path"],
                    "split": row["split"],
                }

                for meta_data_name in meta_data_names:
                    metadata[meta_data_name] = row[meta_data_name]

                # 需要划分为 n 个 token
                n = int(row["duration"] // self.token_sec)

                # stride_tokens = 0 代表只取前面的1535个token，否则取滑窗 ceil((n - 1535) / stride_tokens) 个滑窗
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

        registry = {
            "heartbeat": (
                default_extractor("heartbeat", self.token_sec * 4),
                default_tokenizer(4 * self.token_sec),
                default_mlm_mask_generator(mask_rate),
            ),
            "breath": (
                default_extractor("breath", self.token_sec * 4),
                default_tokenizer(4 * self.token_sec),
                default_mlm_mask_generator(mask_rate),
            ),
            "eeg_original": (
                default_extractor("eeg_original", self.token_sec * 128),
                default_tokenizer(128 * self.token_sec),
                default_mlm_mask_generator(mask_rate),
            ),
            "ecg_original": (
                default_extractor("ecg_original", self.token_sec * 128),
                default_tokenizer(128 * self.token_sec),
                default_mlm_mask_generator(mask_rate),
            ),
            "eog_original": (
                default_extractor("eog_original", self.token_sec * 128),
                default_tokenizer(128 * self.token_sec),
                default_mlm_mask_generator(mask_rate),
            ),
            "emg_original": (
                default_extractor("emg_original", self.token_sec * 128),
                default_tokenizer(128 * self.token_sec),
                default_mlm_mask_generator(mask_rate),
            ),
            "spo2": (
                default_extractor("spo2", self.token_sec * 4),
                default_tokenizer(4 * self.token_sec),
                default_mlm_mask_generator(mask_rate),
            ),
            "resp_original": (
                default_extractor("resp_original", self.token_sec * 4),
                default_tokenizer(4 * self.token_sec),
                default_mlm_mask_generator(mask_rate),
            ),
            "resp_nasal_original": (
                default_extractor("resp_nasal_original", self.token_sec * 4),
                default_tokenizer(4 * self.token_sec),
                default_mlm_mask_generator(mask_rate),
            ),
            "stage5": (
                default_extractor("stage5", 1),
                default_tokenizer(1),
                default_mlm_mask_generator(0.0),
            ),
        }

        unknown = [name for name in channel_names if name not in registry]
        if unknown:
            raise ValueError(f"Unknown channels requested: {unknown}")

        extractors = {name: registry[name][0] for name in channel_names}
        tokenizers = {name: registry[name][1] for name in channel_names}
        mask_generators = {name: registry[name][2] for name in channel_names}

        super().__init__(
            save_preset_path,
            load_preset_path,
            data,
            split,
            extractors=extractors,
            tokenizers=tokenizers,
            mask_generators=mask_generators,
            few_shot=few_shot,
            meta_data_names=meta_data_names,
            sources=sources,
            dataloader_config=kwargs,
        )
