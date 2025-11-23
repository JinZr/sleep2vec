import os
from pathlib import Path
import typing as t

import numpy as np
import pandas as pd

from data.default_dataset import (
    DefaultDataset,
    SampleIndex,
    default_extractor,
    default_mlm_mask_generator,
    default_tokenizer,
)

PAD_STAGE = -1


def window(tot_len: int, max_len: int, stride: int) -> np.ndarray:
    left = np.arange(0, tot_len, stride) if stride > 0 else np.array([0])
    right = np.clip(left + max_len, 0, tot_len)
    return np.stack([left, right], axis=1)


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
        meta_data_names: t.List[str] = [],  # ← 新增参数
        sources: t.List[str] = [],  # ← 新增参数
        randomly_select_channels: bool = True,
        generative: bool = False,
        is_train_set: bool = True,
        **kwargs: t.Any,
    ) -> None:

        self.channel_names = channel_names
        self.randomly_select_channels = randomly_select_channels
        self.token_sec = token_sec
        self.generative = generative
        self.is_train_set = is_train_set

        if not load_preset_path:
            # --- 关键改动：读取一个或多个 CSV 并合并 ---
            def _load_index_df(
                idx: t.Union[str, os.PathLike, t.List[t.Union[str, os.PathLike]]],
            ) -> pd.DataFrame:
                # 单个路径
                if isinstance(idx, (str, os.PathLike, Path)):
                    df = pd.read_csv(idx)
                    df["source"] = str(idx)  # 可选：标注来源文件
                    return df

                # 多个路径
                if isinstance(idx, (list, tuple)):
                    dfs = []
                    for p in idx:
                        dfi = pd.read_csv(p)
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

        super().__init__(
            save_preset_path,
            load_preset_path,
            data,
            split,
            extractors={
                "heartbeat": default_extractor("heartbeat", self.token_sec * 4),
                "breath": default_extractor("breath", self.token_sec * 4),
                "eeg_original": default_extractor("eeg_original", self.token_sec * 128),
                "ecg_original": default_extractor("ecg_original", self.token_sec * 128),
                "eog_original": default_extractor("eog_original", self.token_sec * 128),
                "emg_original": default_extractor("emg_original", self.token_sec * 128),
                "spo2": default_extractor("spo2", self.token_sec * 4),
                "resp_original": default_extractor("resp_original", self.token_sec * 4),
                "resp_nasal_original": default_extractor("resp_nasal_original", self.token_sec * 4),
                # "ah_event": default_extractor("ah_event", self.token_sec * 1),
                "stage5": default_extractor("stage5", 1),
            },
            tokenizers={
                "heartbeat": default_tokenizer(4 * self.token_sec),
                "breath": default_tokenizer(4 * self.token_sec),
                "eeg_original": default_tokenizer(128 * self.token_sec),
                "ecg_original": default_tokenizer(128 * self.token_sec),
                "eog_original": default_tokenizer(128 * self.token_sec),
                "emg_original": default_tokenizer(128 * self.token_sec),
                "spo2": default_tokenizer(4 * self.token_sec),
                "resp_original": default_tokenizer(4 * self.token_sec),
                "resp_nasal_original": default_tokenizer(4 * self.token_sec),
                # "ah_event": default_tokenizer(1 * self.token_sec),
                "stage5": default_tokenizer(1),
            },
            # TODO: 添加各种 mask
            mask_generators={
                "heartbeat": default_mlm_mask_generator(mask_rate),
                "breath": default_mlm_mask_generator(mask_rate),
                "eeg_original": default_mlm_mask_generator(mask_rate),
                "ecg_original": default_mlm_mask_generator(mask_rate),
                "eog_original": default_mlm_mask_generator(mask_rate),
                "emg_original": default_mlm_mask_generator(mask_rate),
                "spo2": default_mlm_mask_generator(mask_rate),
                "resp_original": default_mlm_mask_generator(mask_rate),
                "resp_nasal_original": default_mlm_mask_generator(mask_rate),
                # "ah_event": default_mlm_mask_generator(0),
                "stage5": default_mlm_mask_generator(0.0),
            },
            few_shot=few_shot,
            meta_data_names=meta_data_names,
            sources=sources,
            dataloader_config=kwargs,
        )
