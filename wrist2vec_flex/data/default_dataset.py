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

from wrist2vec_flex.data.metadata import (
    build_w_h_age_sex_center,
    extract_binary_labels,
    make_weighted_sampler_from_labels,
    process_metadata,
)
from wrist2vec_flex.data.utils import (
    choose_source_name,
    compute_channel_matches,
    filter_valid_sample_indices,
    load_builtin_ahi_metadata,
    load_channel_source_tokens,
    load_channel_tokens,
    load_npz,
    make_missing_channel_tokens,
)
from wrist2vec_flex.source_routing import normalize_channel_source_names, uses_explicit_channel_sources


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
    source_masks: dict = field(default_factory=lambda: {})
    source_ids: dict = field(default_factory=lambda: {})


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
        meta_data_regression_names: t.Optional[t.List[str]] = None,
        sources=None,  # ← 新增参数
        pair_selector: t.Any | None = None,
        seed: int = 42,
        filter_max_workers: int | None = None,
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
        self.meta_data_regression_names = meta_data_regression_names or []
        self.sources = sources or []
        self.few_shot = few_shot
        self.extractors = extractors
        self.tokenizers = tokenizers
        self.mask_generators = mask_generators
        self.pair_selector = pair_selector
        # self.collators = collators
        self.dataloader_config = dataloader_config
        self._uses_explicit_channel_sources = uses_explicit_channel_sources(getattr(self, "channel_source_names", {}))

        if load_preset_path:
            logging.info(f"Start loading data preset from {load_preset_path}")
            # 从文件中读取对象
            with open(load_preset_path, "rb") as f:
                self.data = pickle.load(f)
            if self.allow_missing_channels:
                missing_payload = [
                    getattr(sample, "id", "?")
                    for sample in self.data
                    if not isinstance(getattr(sample, "payload", None), dict)
                    or "available_channels" not in getattr(sample, "payload", {})
                ]
                if missing_payload:
                    raise ValueError(
                        "Loaded variable-channel preset does not contain payload['available_channels']. "
                        "Rebuild the wrist2vec_flex preset with wrist2vec_flex.preprocess.save_dataset_presets."
                    )
            if self._uses_explicit_channel_sources:
                missing_payload = [
                    getattr(sample, "id", "?")
                    for sample in self.data
                    if not isinstance(getattr(sample, "payload", None), dict)
                    or "channel_sources" not in getattr(sample, "payload", {})
                ]
                if missing_payload:
                    raise ValueError(
                        "Loaded preset does not contain payload['channel_sources'] required by source_names. "
                        "Rebuild the wrist2vec_flex preset with wrist2vec_flex.preprocess.save_dataset_presets."
                    )
        elif data is not None:
            # ✅ 初始化时检查并过滤掉 token 长度不一致的样本
            allow_missing_channels = bool(self.allow_missing_channels)
            channel_names = self.channel_names
            min_channels = self.min_channels
            self.data = self._filter_valid_sample_indices(data, filter_max_workers=filter_max_workers)
            if "ahi" in channel_names and not self.data:
                raise ValueError(
                    "No valid samples remain for the built-in AHI contract. "
                    "Expected NPZ keys 'ah_event', scalar 'ahi', and scalar 'tst' for every retained sample."
                )
            if save_preset_path:
                with open(save_preset_path, "wb") as f:
                    pickle.dump(self.data, f)
        else:
            raise ValueError("Either load_preset_path or data must be provided.")

        # 根据需要的 metadata 筛选数据
        self.filter_with_metadata()
        if "ahi" in getattr(self, "channel_names", []) and not self.data:
            raise ValueError(
                "No valid samples remain for the built-in AHI contract. "
                "Expected NPZ keys 'ah_event', scalar 'ahi', and scalar 'tst' for every retained sample."
            )

        # few-shot 筛选
        if few_shot is not None:
            self.select_few_shot()

    def _filter_valid_sample_indices(
        self,
        data: t.Sequence[SampleIndex],
        *,
        filter_max_workers: int | None,
    ) -> list[SampleIndex]:
        return filter_valid_sample_indices(
            data,
            self.extractors,
            self.tokenizers,
            allow_missing_channels=bool(self.allow_missing_channels),
            channel_names=self.channel_names,
            feature_channel_names=getattr(self, "feature_channel_names", None),
            required_channel_names=getattr(self, "required_channel_names", None),
            channel_input_dims=getattr(self, "channel_input_dims", None),
            channel_source_names=getattr(self, "channel_source_names", None),
            expand_source_branches=bool(getattr(self, "expand_source_branches", False)),
            min_channels=self.min_channels,
            tolerance=1,
            max_workers=filter_max_workers,
        )

    def _get_available_channels_for_src(self, src: SampleIndex) -> set[str]:
        channel_names = self.channel_names
        channel_name_set = set(getattr(self, "feature_channel_names", channel_names))
        channel_source_names = normalize_channel_source_names(
            channel_names, getattr(self, "channel_source_names", None)
        )
        expand_source_branches = bool(getattr(self, "expand_source_branches", False))
        payload = getattr(src, "payload", None)
        if isinstance(payload, dict) and payload.get("available_channels"):
            avail = payload["available_channels"]
        else:
            with load_npz(src.path) as npz:
                avail, _ = compute_channel_matches(
                    npz,
                    channel_names,
                    channel_source_names=channel_source_names,
                    expand_source_branches=expand_source_branches,
                )
        return {str(k) for k in avail if str(k) in channel_name_set}

    def _load_tokens_for_src(
        self,
        src: SampleIndex,
        chosen_output_names: list[str],
        *,
        channel_source_names: t.Mapping[str, t.Sequence[str]],
        expand_source_branches: bool,
        effective_channel_to_logical: t.Mapping[str, str],
        effective_channel_to_source: t.Mapping[str, str],
        channel_input_dims: t.Mapping[str, int] | None,
        tolerance: int,
        missing_feature_names: set[str] | None = None,
    ) -> tuple[dict, dict, dict, dict, dict, dict]:
        with load_npz(src.path) as npz:
            missing_feature_names = set(missing_feature_names or set())
            payload_channel_sources = {}
            src_payload = getattr(src, "payload", None)
            if isinstance(src_payload, dict) and isinstance(src_payload.get("channel_sources"), dict):
                payload_channel_sources = {key: list(value) for key, value in src_payload["channel_sources"].items()}
            else:
                _, payload_channel_sources = compute_channel_matches(
                    npz,
                    self.channel_names,
                    channel_source_names=channel_source_names,
                    expand_source_branches=expand_source_branches,
                )

            payload = {}
            tokens = {}
            masks = {}
            source_masks = {}
            source_ids = {}
            for output_name in chosen_output_names:
                logical_name = effective_channel_to_logical.get(output_name, output_name)
                source_name = effective_channel_to_source.get(output_name)
                if logical_name in missing_feature_names:
                    missing_tokens, source_mask, source_id = make_missing_channel_tokens(
                        logical_name,
                        int(src.end) - int(src.start),
                        channel_input_dims,
                        channel_source_names[logical_name],
                    )
                    payload[output_name] = {}
                    tokens[output_name] = missing_tokens
                    source_masks[output_name] = source_mask
                    source_ids[output_name] = source_id
                    masks[output_name] = torch.zeros(missing_tokens.shape[0], dtype=torch.bool)
                    continue
                if not expand_source_branches and channel_source_names[logical_name] != [logical_name]:
                    source_payload, source_tokens, source_mask, source_id = load_channel_source_tokens(
                        npz,
                        channel_name=logical_name,
                        start=src.start,
                        end=src.end,
                        channel_input_dims=channel_input_dims,
                        source_names=channel_source_names[logical_name],
                        available_source_names=payload_channel_sources.get(logical_name, []),
                        tolerance=tolerance,
                    )
                    payload[output_name] = source_payload
                    tokens[output_name] = source_tokens
                    source_masks[output_name] = source_mask
                    source_ids[output_name] = source_id
                else:
                    if not expand_source_branches:
                        source_name = choose_source_name(
                            logical_name,
                            channel_sources=payload_channel_sources,
                        )
                    payload[output_name], tokens[output_name] = load_channel_tokens(
                        npz,
                        channel_name=logical_name,
                        start=src.start,
                        end=src.end,
                        channel_input_dims=channel_input_dims,
                        source_name=source_name,
                    )
                    source_masks[output_name] = torch.ones(1, dtype=torch.bool)
                    source_ids[output_name] = torch.arange(1, dtype=torch.long)
                if not source_masks[output_name].any():
                    raise ValueError(f"Channel '{logical_name}' is marked available with no source.")
                masks[output_name] = self.mask_generators[logical_name](tokens[output_name])
            metadata = dict(src.metadata)
            if "ahi" in chosen_output_names:
                ahi_value, tst_value = load_builtin_ahi_metadata(npz)
                metadata["ahi"] = ahi_value
                metadata["tst"] = tst_value
        return payload, tokens, masks, metadata, source_masks, source_ids

    def filter_with_metadata(
        self,
    ) -> t.List["SampleIndex"]:
        # if not self.meta_data_names:
        #     return

        random.seed(self.seed)
        selected = []
        built_in_ahi_runtime_metadata = "ahi" in getattr(self, "channel_names", [])

        for d in self.data:
            keep = True
            for meta_data_name in self.meta_data_names:
                if built_in_ahi_runtime_metadata and meta_data_name in {"ahi", "tst"}:
                    continue
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

    def reset_pair_selector(self) -> None:
        selector = self.pair_selector
        if selector is not None and hasattr(selector, "reset"):
            selector.reset()

    def __getitem__(self, idx: int) -> Sample:
        forced_pair: tuple[str, str] | None = None
        if isinstance(idx, tuple) and len(idx) == 2 and isinstance(idx[0], int):
            idx, raw_pair = idx
            if not isinstance(raw_pair, tuple) or len(raw_pair) != 2:
                raise ValueError(f"Invalid pair payload from sampler: {raw_pair!r}")
            forced_pair = (str(raw_pair[0]), str(raw_pair[1]))

        src = self.data[idx]
        if forced_pair is None:
            return src  # 不读取 npz，不做 tokenize
        return src, forced_pair
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
        feature_channel_names = list(getattr(self, "feature_channel_names", channel_names))
        required_channel_names = list(getattr(self, "required_channel_names", []))
        collate_all_feature_channels = bool(getattr(self, "collate_all_feature_channels", False))
        randomly_select_channels = self.randomly_select_channels
        generative = self.generative
        disease_names = self.meta_data_names
        allow_missing_channels = bool(self.allow_missing_channels)
        min_channels = self.min_channels
        bucket_by_available_channels = bool(self.bucket_by_available_channels)
        train_pair_probs = self.train_pair_probs
        train_pair_track_unique_samples = bool(self.train_pair_track_unique_samples)
        pair_selector = self.pair_selector
        channel_input_dims = getattr(self, "channel_input_dims", None)
        channel_source_names = normalize_channel_source_names(
            channel_names, getattr(self, "channel_source_names", None)
        )
        expand_source_branches = bool(getattr(self, "expand_source_branches", False))
        output_channel_names = list(getattr(self, "output_channel_names", channel_names))
        effective_channel_to_logical = dict(getattr(self, "effective_channel_to_logical", {}))
        effective_channel_to_source = dict(getattr(self, "effective_channel_to_source", {}))

        def collate_fn(indices, tolerance=1):
            selected_pair: tuple[str, str] | None = None
            resolved_indices: list[SampleIndex] = []
            for item in indices:
                src = item
                pair = None
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], SampleIndex):
                    src, pair = item
                if not isinstance(src, SampleIndex):
                    raise ValueError(f"Unexpected batch element type: {type(src).__name__}")
                if pair is not None:
                    pair = (str(pair[0]), str(pair[1]))
                    if selected_pair is None:
                        selected_pair = pair
                    elif selected_pair != pair:
                        raise ValueError(f"Mixed scheduled pairs within one batch: {selected_pair} vs {pair}")
                resolved_indices.append(src)

            supervised_feature_availability: dict[int, set[str]] = {}
            if allow_missing_channels:

                def _available_for_src(src):
                    return self._get_available_channels_for_src(src)

                if collate_all_feature_channels:
                    chosen = list(feature_channel_names)
                    for channel_name in required_channel_names:
                        if channel_name not in chosen:
                            chosen.append(channel_name)
                    selected_sources = list(resolved_indices)
                    for src in selected_sources:
                        avail = _available_for_src(src)
                        if len(avail) < min_channels:
                            raise ValueError(
                                "Supervised variable-channel sample has fewer feature channels than required: "
                                f"id={src.id}, path={src.path}, available={len(avail)}, min_channels={min_channels}."
                            )
                        supervised_feature_availability[id(src)] = set(avail)
                        if isinstance(src.payload, dict):
                            src.payload["available_channels"] = [
                                channel_name for channel_name in feature_channel_names if channel_name in avail
                            ]
                else:
                    avail_map = []
                    for src in resolved_indices:
                        avail = _available_for_src(src)
                        if len(avail) >= min_channels:
                            avail_map.append((src, avail))
                        elif selected_pair is not None:
                            raise ValueError(
                                "Pair-first sampler emitted sample without enough channels: "
                                f"id={src.id}, path={src.path}, available={len(avail)}, min_channels={min_channels}."
                            )

                    if not avail_map:
                        raise ValueError("No samples have enough available channels in this batch.")

                    if selected_pair is not None:
                        left, right = selected_pair
                        chosen = [left, right]
                        selected_sources = []
                        for src, avail in avail_map:
                            if left in avail and right in avail:
                                selected_sources.append(src)
                            else:
                                raise ValueError(
                                    f"Pair-first sampler emitted sample {src.id} without required pair {selected_pair}."
                                )
                        if not selected_sources:
                            raise ValueError(f"No samples in batch support scheduled pair {selected_pair}.")
                        if len(selected_sources) != len(resolved_indices):
                            raise ValueError(
                                "Pair-first collate batch was unexpectedly shrunk. "
                                f"scheduled_pair={selected_pair}, input={len(resolved_indices)}, "
                                f"kept={len(selected_sources)}."
                            )
                    else:
                        batch_available = set.intersection(*(avail for _, avail in avail_map))

                        if len(batch_available) >= 2:
                            if pair_selector is not None:
                                chosen = pair_selector.select(sorted(batch_available))
                            elif randomly_select_channels:
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
                if selected_pair is not None:
                    chosen = [selected_pair[0], selected_pair[1]]
                elif pair_selector is not None:
                    chosen = pair_selector.select(channel_names)
                elif randomly_select_channels:
                    chosen = random.sample(channel_names, k=2)
                    if generative:
                        chosen[0] = "eeg_original"
                else:
                    chosen = channel_names
                selected_sources = resolved_indices

            samples = []
            token_starts: list[int] = []
            for src in selected_sources:
                chosen_output_names = output_channel_names if expand_source_branches else chosen
                missing_feature_names = None
                if allow_missing_channels and collate_all_feature_channels:
                    available_features = supervised_feature_availability.get(id(src), set())
                    missing_feature_names = set(feature_channel_names) - set(available_features)
                payload, tokens, masks, metadata, source_masks, source_ids = self._load_tokens_for_src(
                    src,
                    chosen_output_names,
                    channel_source_names=channel_source_names,
                    expand_source_branches=expand_source_branches,
                    effective_channel_to_logical=effective_channel_to_logical,
                    effective_channel_to_source=effective_channel_to_source,
                    channel_input_dims=channel_input_dims,
                    tolerance=tolerance,
                    missing_feature_names=missing_feature_names,
                )
                payload.update(src.payload)
                sample = Sample(
                    id=src.id,
                    length=src.end - src.start,
                    payload=payload,
                    tokens=tokens,
                    masks=masks,
                    metadata=metadata,
                    source_masks=source_masks,
                    source_ids=source_ids,
                )
                samples.append(sample)
                token_starts.append(int(src.start))

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
                "token_start": torch.tensor(token_starts, dtype=torch.long),
                "metadata": process_metadata(samples, disease_names, self.meta_data_regression_names),
            }
            if (
                not expand_source_branches
                and len(chosen) == 2
                and (selected_pair is not None or randomly_select_channels or pair_selector is not None)
            ):
                batch["pair"] = (str(chosen[0]), str(chosen[1]))

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
                pad_value = -1.0 if key in {"stage5", "ahi"} else 0.0
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

            source_masks = {}
            source_ids = {}
            for key in samples[0].source_masks.keys():
                source_masks[key] = torch.stack([s.source_masks[key] for s in samples]).bool()
                source_ids[key] = samples[0].source_ids[key]
            batch["source_mask"] = source_masks
            batch["source_ids"] = source_ids

            channel_mask_rows = []
            for sample in samples:
                if allow_missing_channels and collate_all_feature_channels:
                    loaded_logical = set(sample.payload.get("available_channels", []))
                elif expand_source_branches:
                    loaded_logical = {
                        effective_channel_to_logical.get(output_name, output_name) for output_name in sample.tokens
                    }
                else:
                    loaded_logical = set(sample.tokens.keys())
                channel_mask_rows.append([channel_name in loaded_logical for channel_name in feature_channel_names])
            batch["channel_mask"] = torch.tensor(channel_mask_rows, dtype=torch.bool)

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
        explicit_batch_sampler = dl_kwargs.pop("batch_sampler", None)

        if explicit_batch_sampler is not None:
            dl_kwargs.pop("batch_size", None)
            dl_kwargs.pop("shuffle", None)
            return DataLoader(
                self,
                batch_sampler=explicit_batch_sampler,
                collate_fn=collate_fn,
                **dl_kwargs,
            )

        if allow_missing_channels and self.is_train_set and not collate_all_feature_channels:
            if sampler is not None:
                raise ValueError("Pair-first sampling is incompatible with metadata weighted sampler.")

            from wrist2vec_flex.data.samplers import PairFirstBatchSampler

            batch_size = int(dl_kwargs.pop("batch_size"))
            shuffle = bool(dl_kwargs.pop("shuffle", False))
            batch_sampler = PairFirstBatchSampler(
                self.data,
                channel_names=channel_names,
                batch_size=batch_size,
                min_channels=min_channels,
                shuffle=shuffle,
                drop_last=self.is_train_set,
                seed=self.seed,
                pair_sampling="uniform",
                pair_probs=train_pair_probs,
                track_unique_sample_counts=train_pair_track_unique_samples,
            )

            return DataLoader(
                self,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                **dl_kwargs,
            )

        # When pretraining with missing channels, random shuffling can mix different
        # channel-availability signatures within one batch. That makes the
        # intersection of available channels tiny and triggers the legacy fallback
        # to a single "best_pair", collapsing training.
        # When pretraining with missing channels, random shuffling can mix different
        # channel-availability signatures within one batch. That makes the
        # intersection of available channels tiny and triggers the legacy fallback
        # to a single "best_pair", collapsing training.
        #
        # If enabled, bucket batches by available-channel signature to keep each
        # batch homogeneous.

        if (
            allow_missing_channels
            and bucket_by_available_channels
            and sampler is None
            and not collate_all_feature_channels
        ):
            from wrist2vec_flex.data.samplers import AvailableChannelsBucketBatchSampler

            batch_size = int(dl_kwargs.pop("batch_size"))
            shuffle = bool(dl_kwargs.pop("shuffle", False))
            batch_sampler = AvailableChannelsBucketBatchSampler(
                self.data,
                batch_size=batch_size,
                min_channels=min_channels,
                shuffle=shuffle,
                drop_last=self.is_train_set,
                shard_across_ranks=self.is_train_set,
                seed=self.seed,
            )

            return DataLoader(
                self,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                **dl_kwargs,
            )

        return DataLoader(
            self,
            **dl_kwargs,
            collate_fn=collate_fn,
            sampler=sampler,
            drop_last=self.is_train_set,
        )
