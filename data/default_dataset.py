from dataclasses import dataclass, field
import itertools
import logging
import math
import pickle
import random
import typing as t

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from data.metadata import (
    build_w_h_age_sex_center,
    extract_binary_labels,
    make_weighted_sampler_from_labels,
    process_metadata,
)
from data.samplers import PairBatchSampler
from data.utils import filter_valid_sample_indices, load_npz


@dataclass
class SampleIndex:
    id: int | str
    path: str
    start: int
    end: int
    payload: dict = field(default_factory=lambda: {})
    metadata: dict = field(default_factory=lambda: {})


@dataclass
class Sample:
    id: int | str
    length: int
    payload: dict
    tokens: dict
    masks: dict
    metadata: dict = field(default_factory=lambda: {})


class BaseDataset(Dataset):
    def dataloader(self) -> DataLoader:
        raise NotImplementedError


class DefaultDataset(BaseDataset):
    """
    General dataset behavior comes here
    """

    def __init__(
        self,
        save_preset_path: t.Optional[str],
        load_preset_path: t.Optional[str],
        data: t.Sequence[SampleIndex],
        split: t.List[str],
        extractors: t.Mapping[str, t.Callable],
        tokenizers: t.Mapping[str, t.Callable],
        mask_generators: t.Mapping[str, t.Callable],
        dataloader_config: t.Mapping[str, t.Any],
        few_shot: int | float | None = None,  # ← 新增参数
        meta_data_names=None,  # ← 新增参数
        sources=None,  # ← 新增参数
        seed: int = 42,
    ) -> None:
        """
        Args:
            data: list of SampleIndex
            extractors: mapping, key : ((NpzFile, start, end) -> fetched value)
            tokenizers: mapping, key : ((NpzFile, start, end) -> fetched value)
            collators: mapping, key : (is_input, (t.List[Sample]) -> batched value)
            dataloader_config: DataLoader kwargs except dataset and collate_fn
        """
        self.split = split
        self.data = data
        self.seed = seed
        self.meta_data_names = meta_data_names or []
        self.sources = sources or []
        self.few_shot = few_shot
        self.extractors = extractors
        self.tokenizers = tokenizers
        self.mask_generators = mask_generators
        # self.collators = collators
        self.dataloader_config = dataloader_config

        if load_preset_path:
            logging.info(f"Start loading data preset from {load_preset_path}")
            # 从文件中读取对象
            with open(load_preset_path, "rb") as f:
                self.data = pickle.load(f)
        elif data is not None:
            # ✅ 初始化时检查并过滤掉 token 长度不一致的样本
            allow_missing_channels = bool(getattr(self, "allow_missing_channels", False))
            channel_names = getattr(self, "channel_names", None)
            if allow_missing_channels and channel_names is None:
                raise ValueError("DefaultDataset requires channel_names when allow_missing_channels is enabled.")
            min_channels = getattr(self, "min_channels", 2)
            self.data = filter_valid_sample_indices(
                data,
                extractors,
                tokenizers,
                allow_missing_channels=allow_missing_channels,
                channel_names=channel_names,
                min_channels=min_channels,
                tolerance=1,
            )
            if save_preset_path:
                with open(save_preset_path, "wb") as f:
                    pickle.dump(self.data, f)
        else:
            raise ValueError("Either load_preset_path or data must be provided.")

        # 根据需要的 metadata 筛选数据
        self.filter_with_metadata()

        # few-shot 筛选
        if few_shot is not None:
            self.select_few_shot()

    def filter_with_metadata(
        self,
    ) -> t.List["SampleIndex"]:
        # if not self.meta_data_names:
        #     return

        random.seed(self.seed)
        selected = []

        for d in self.data:
            keep = True
            for meta_data_name in self.meta_data_names:
                value = d.metadata.get(meta_data_name, None)

                # 如果字段缺失 或 值为 NaN，则丢弃
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    keep = False
                    break

            if self.sources:
                source_path = d.metadata.get("source", None)
                if source_path is None:
                    keep = False
                else:
                    keep = any(source in str(source_path) for source in self.sources) and keep

            if d.metadata.get("split") not in self.split:
                keep = False

            if keep:
                selected.append(d)

        logging.info(f"Filtered metadata: kept {len(selected)} / {len(self.data)} samples")
        self.data = selected
        return selected

    def select_few_shot(
        self,
    ) -> t.List["SampleIndex"]:
        """
        从数据中随机选择 few_shot 个样本。
        如果 0 < few_shot < 1，则按比例采样；
        并确保较小比例采样是较大比例采样的子集。
        """
        if not self.few_shot:
            return

        total = len(self.data)
        if self.few_shot >= total:
            logging.info(f"self.few_shot={self.few_shot} >= total samples={total}, skipping sampling")
            return list(self.data)

        # 设定随机种子，保证顺序一致
        random.seed(self.seed)
        shuffled = list(self.data)
        random.shuffle(shuffled)

        # 计算采样数量
        if 0 < self.few_shot < 1:
            num_samples = max(1, int(total * self.few_shot))
        else:
            num_samples = int(self.few_shot)

        selected = shuffled[:num_samples]
        logging.info(f"Selected {len(selected)} samples out of {total} for few-shot setting")
        self.data = selected
        return selected

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Sample:

        src = self.data[idx]
        return src  # 不读取 npz，不做 tokenize
        # TODO: tokenize here!!!
        # src = self.data[idx]
        # with np.load(src.path) as npz:
        #     payload = {
        #         key: fn(npz, src.start, src.end) for key, fn in self.extractors.items()
        #     }
        #     tokens = {
        #         key: fn(payload[key]) for key, fn in self.tokenizers.items()
        #     }

        #     masks = {
        #         key: fn(tokens[key]) for key, fn in self.mask_generators.items()
        #     }
        # payload.update(src.payload)

        # # print(f"in __getitem__ metadata: {src.metadata}")
        # return Sample(
        #     id=src.id,
        #     length=src.end - src.start,
        #     payload=payload,
        #     tokens=tokens,
        #     masks=masks,
        #     metadata=src.metadata  # ✅ 传入 metadata
        # )

    def dataloader(self, device: str = "cpu") -> DataLoader:
        channel_names = self.channel_names
        randomly_select_channels = self.randomly_select_channels
        generative = self.generative
        disease_names = self.meta_data_names
        allow_missing_channels = bool(getattr(self, "allow_missing_channels", False))
        min_channels = getattr(self, "min_channels", 2)
        channel_name_set = set(channel_names)

        def collate_fn(indices, tolerance=1):

            if allow_missing_channels:

                def _available_for_src(src):
                    payload = getattr(src, "payload", None)
                    if isinstance(payload, dict) and payload.get("available_channels"):
                        avail = payload["available_channels"]
                    else:
                        with load_npz(src.path) as npz:
                            avail = [k for k in channel_names if k in npz]
                    return set([k for k in avail if k in channel_name_set])

                avail_map = []
                for src in indices:
                    avail = _available_for_src(src)
                    if len(avail) >= min_channels:
                        avail_map.append((src, avail))

                if not avail_map:
                    raise ValueError("No samples have enough available channels in this batch.")

                batch_available = set.intersection(*(avail for _, avail in avail_map))

                if len(batch_available) >= 2:
                    if randomly_select_channels:
                        chosen = random.sample(sorted(batch_available), k=2)
                        if generative and "eeg_original" in batch_available:
                            chosen[0] = "eeg_original"
                    else:
                        chosen = sorted(batch_available)
                    selected_sources = [src for src, _ in avail_map]
                else:
                    pair_counts: dict[tuple[str, str], int] = {}
                    for _, avail in avail_map:
                        for pair in itertools.combinations(sorted(avail), 2):
                            pair_counts[pair] = pair_counts.get(pair, 0) + 1
                    if not pair_counts:
                        raise ValueError("No valid channel pairs found for this batch.")
                    best_pair = max(pair_counts, key=pair_counts.get)
                    chosen = list(best_pair)
                    selected_sources = [
                        src for src, avail in avail_map if best_pair[0] in avail and best_pair[1] in avail
                    ]
            else:
                if randomly_select_channels:
                    chosen = random.sample(channel_names, k=2)
                    if generative:
                        chosen[0] = "eeg_original"
                else:
                    chosen = channel_names
                selected_sources = indices

            samples = []
            for src in selected_sources:

                with load_npz(src.path) as npz:
                    payload = {k: self.extractors[k](npz, src.start, src.end) for k in chosen}
                    tokens = {k: self.tokenizers[k](payload[k]) for k in chosen}
                    masks = {k: self.mask_generators[k](tokens[k]) for k in chosen}
                payload.update(src.payload)
                sample = Sample(
                    id=src.id,
                    length=src.end - src.start,
                    payload=payload,
                    tokens=tokens,
                    masks=masks,
                    metadata=src.metadata,  # ✅ 传入 metadata
                )
                samples.append(sample)

            # 1️⃣ 逐样本检查每个通道的 token 长度，并裁剪到样本内的最小长度
            for s_idx, sample in enumerate(samples):
                lengths = [v.shape[0] for v in sample.tokens.values()]
                max_len, min_len = max(lengths), min(lengths)

                if max_len - min_len > tolerance:
                    raise ValueError(f"Token length mismatch > tolerance in sample {sample.id}: {lengths}")

                # ✅ 将每个通道裁剪到该样本的最小长度
                for k in sample.tokens:
                    sample.tokens[k] = sample.tokens[k][:min_len]
                for k in sample.masks:
                    sample.masks[k] = sample.masks[k][:min_len]

            # 2️⃣ 获取整个 batch 中的最大 token 长度（裁剪后），用于 pad
            max_len = max(next(iter(s.tokens.values())).shape[0] for s in samples)

            batch = {
                "id": [s.id for s in samples],
                "length": torch.tensor(
                    [next(iter(s.tokens.values())).shape[0] for s in samples],
                    # device=device
                ),
                "metadata": process_metadata(samples, disease_names),
            }

            # === 在这里生成 weights 矩阵（CPU）===
            # print(batch['metadata']) # 输出 {'age': tensor([62., 40., ...]), 'sex': tensor([1, 1, ...])}
            w, h = build_w_h_age_sex_center(
                batch["metadata"]["age"],
                batch["metadata"]["sex"],
                batch["metadata"]["source"],
                batch["metadata"]["path"],
            )
            batch["w"] = w  # [N,N]，用于隐式负样本
            batch["h"] = h  # [N,N]，可选用于 margin

            # 3️⃣ 合并 tokens，pad 到 batch 内最大长度
            tokens = {}
            for key in samples[0].tokens.keys():
                token_seqs = [s.tokens[key] for s in samples]
                pad_value = -1.0 if key == "stage5" else 0.0
                padded = pad_sequence(token_seqs, batch_first=True, padding_value=pad_value)
                # padded = pad_sequence(token_seqs, batch_first=True, padding_value=0.0).to(device)
                tokens[key] = padded
            batch["tokens"] = tokens

            # 4️⃣ 合并 masks
            masks = {}
            for key in samples[0].masks.keys():
                mask_seqs = [s.masks[key] for s in samples]
                padded = pad_sequence(mask_seqs, batch_first=True, padding_value=0)
                # padded = pad_sequence(mask_seqs, batch_first=True, padding_value=0).to(device)
                masks[key] = padded.bool()
            batch["mlm_mask"] = masks

            # print(f"in collate_fn batch: batch created!")
            return batch

        sampler = None
        if (
            self.meta_data_names
            and self.meta_data_names[0]
            in [
                "allergiesorsinusproblems",
                "asthma",
                "bronchitis",
                "cerebrovasculardisease",
                "chronicobstructivepulmonarydiseasecopd",
                "coronaryheartdisease",
                "diabetes",
                "heartfailure",
                "hypertension",
                "restlesslegsyndromerls",
            ]
            and self.is_train_set
        ):
            target_name = self.meta_data_names[0]  # 或者显式传入
            labels = extract_binary_labels(self, target_name)
            sampler = make_weighted_sampler_from_labels(labels)

        dl_kwargs = dict(self.dataloader_config)
        if sampler is not None:
            dl_kwargs.pop("shuffle", None)  # sampler 与 shuffle 互斥

        batch_sampler = None
        if allow_missing_channels:
            batch_size = dl_kwargs.pop("batch_size", None)
            shuffle = dl_kwargs.pop("shuffle", False)
            if batch_size is None:
                raise ValueError("batch_size must be set when allow_missing_channels is enabled.")
            batch_sampler = PairBatchSampler(
                self,
                batch_size=batch_size,
                channel_names=channel_names,
                min_channels=min_channels,
                shuffle=shuffle,
                drop_last=self.is_train_set,
                seed=self.seed,
            )

        if batch_sampler is not None:
            return DataLoader(
                self,
                **dl_kwargs,
                collate_fn=collate_fn,
                batch_sampler=batch_sampler,
            )

        return DataLoader(
            self,
            **dl_kwargs,
            collate_fn=collate_fn,
            sampler=sampler,
            drop_last=self.is_train_set,  # only drop last batch for training
        )
