import logging
import math
import pickle
import random
import typing as t
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


@dataclass
class SampleIndex:
    id: int | str
    path: str
    start: int
    end: int
    payload: dict = field(default_factory=lambda: {})
    metadata: dict = field(default_factory=lambda: {})  # ✅ 新增字段，用于非序列数据


@dataclass
class Sample:
    id: int | str
    length: int
    payload: dict
    tokens: dict
    masks: dict
    metadata: dict = field(default_factory=lambda: {})  # ✅ 新增字段


def default_extractor(name: str, frames_per_token: int):
    def extract(npz, start: int, end: int):
        s = start * frames_per_token
        e = end * frames_per_token
        x = np.asarray(npz[name][s:e])

        # 二维恒为 (L, 1) —— 压成一维
        if x.ndim == 2:
            # 如果你 100% 确认第二维必为 1，也可以直接：x = x.squeeze(-1)
            if x.shape[1] == 1:
                x = x[:, 0]

        # 统一转为 float32
        return torch.as_tensor(x, dtype=torch.float32)

    return extract


# 支持 1D 或 2D 信号的通用 tokenizer
def default_tokenizer(frames_per_token: int):
    def tokenize(data: torch.Tensor):
        """
        参数:
            data: 1D 或 2D torch.Tensor
                - 1D: shape = [T]
                - 2D: shape = [C, T]
        返回:
            tokens: Tensor
                - 如果输入是 1D: shape = [num_tokens, frames_per_token]
                - 如果输入是 2D: shape = [num_tokens, C, frames_per_token]
        """
        if data.dim() == 1:
            # 1D 情况
            total_length = data.shape[0]
            num_tokens = total_length // frames_per_token
            trimmed = data[: num_tokens * frames_per_token]
            tokens = trimmed.view(num_tokens, frames_per_token)
        elif data.dim() == 2:
            # 2D 情况：shape = [C, T]
            C, T = data.shape
            num_tokens = T // frames_per_token
            trimmed = data[:, : num_tokens * frames_per_token]  # [C, T']
            tokens = trimmed.view(C, num_tokens, frames_per_token)  # [C, N, L]
            tokens = tokens.permute(1, 0, 2)  # → [N, C, L]
        else:
            raise ValueError(f"Unsupported input dimension: {data.shape}")

        return tokens

    return tokenize


def default_mlm_mask_generator(mask_ratio: float = 0.15):
    """
    生成 token mask，用于 MLM 或 span-masking 等任务。
    参数:
        mask_ratio: float, 要被 mask 的 token 比例（0~1）

    返回:
        masker: 函数，接受 shape = [N, ...] 的 token 序列，返回 shape = [N] 的 bool mask（1 表示被 mask）
    """

    def generate_mask(tokens: torch.Tensor):
        """
        参数:
            tokens: Tensor, shape = [N, ...]，其中 N 是 token 数
        返回:
            mask: Tensor, shape = [N]，值为 0（不mask）或 1（mask）
        """
        num_tokens = tokens.shape[0]
        num_mask = int(num_tokens * mask_ratio)

        # 随机选中若干 index 作为 mask
        mask = torch.zeros(num_tokens, dtype=torch.bool)
        if num_mask > 0:
            mask_indices = torch.randperm(num_tokens)[:num_mask]
            mask[mask_indices] = True
        return mask

    return generate_mask


def pad(
    x, max_len: int, pad_value: torch.types.Number = 0, dim: int = 0
) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if x.shape[dim] == max_len:
        return x
    if x.shape[dim] > max_len:
        return x.narrow(dim, 0, max_len)
    pad_shape = list(x.shape)
    pad_shape[dim] = max_len - x.shape[dim]
    padding = torch.full(pad_shape, pad_value, device=x.device, dtype=x.dtype)
    return torch.concat([x, padding], dim)


def pad_batch(
    x: t.List[torch.Tensor],
    max_len: t.Union[int, None] = None,
    pad_value: torch.types.Number = 0,
    dim: int = 0,
) -> torch.Tensor:
    if max_len is None:
        max_len = max(y.shape[dim] for y in x)
    return torch.stack([pad(y, max_len, pad_value, dim) for y in x])


def filter_valid_sample_indices(
    data: t.Sequence["SampleIndex"],
    extractors: t.Mapping[str, t.Callable],
    tokenizers: t.Mapping[str, t.Callable],
    tolerance: int = 1,
    max_workers: int = 128,  # 线程数
) -> t.List["SampleIndex"]:
    """
    多线程版本：过滤掉 token 长度差距过大的样本
    """

    def process_sample(sample_index):
        try:
            with np.load(sample_index.path) as npz:
                payload = {
                    key: fn(npz, sample_index.start, sample_index.end)
                    for key, fn in extractors.items()
                }
                tokens = {key: fn(payload[key]) for key, fn in tokenizers.items()}
                lengths = [v.shape[0] for v in tokens.values()]
                max_len, min_len = max(lengths), min(lengths)

                if max_len - min_len <= tolerance:
                    return sample_index
                else:
                    logging.info(
                        f"[Skip] Token length mismatch at {sample_index.id}: {lengths}. Meta: {sample_index.metadata}"
                    )
                    return None
        except Exception as e:
            logging.info(f"[Skip] Error loading sample {sample_index.id}: {e}")
            return None

    filtered_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, s) for s in data]
        for f in tqdm(as_completed(futures), total=len(futures)):
            result = f.result()
            if result is not None:
                filtered_data.append(result)

    logging.info(f"Loaded {len(filtered_data)} valid samples (from {len(data)} total)")
    return filtered_data


# class BaseDataset(Dataset, Factory):
class BaseDataset(Dataset):
    def dataloader() -> DataLoader:
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
        self.meta_data_names = meta_data_names
        self.sources = sources
        self.few_shot = few_shot
        self.extractors = extractors
        self.tokenizers = tokenizers
        self.mask_generators = mask_generators
        # self.collators = collators
        self.dataloader_config = dataloader_config

        if load_preset_path:
            print(f"Start loading data preset from {load_preset_path}")
            # 从文件中读取对象
            with open(load_preset_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            # ✅ 初始化时检查并过滤掉 token 长度不一致的样本
            self.data = filter_valid_sample_indices(
                data, extractors, tokenizers, tolerance=1
            )
            with open(save_preset_path, "wb") as f:
                pickle.dump(data, f)

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
                # print(source_path)
                keep = any([source in source_path for source in self.sources]) and keep
                # print(keep)

            if d.metadata["split"] not in self.split:
                keep = False

            if keep:
                selected.append(d)

        logging.info(
            f"Filtered metadata: kept {len(selected)} / {len(self.data)} samples"
        )
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
            logging.info(
                f"self.few_shot={self.few_shot} >= total samples={total}, skipping sampling"
            )
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
        logging.info(
            f"Selected {len(selected)} samples out of {total} for few-shot setting"
        )
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

        def collate_fn(indices, tolerance=1):

            # ① 先为本 batch 统一抽两个通道
            if randomly_select_channels:
                chosen = random.sample(channel_names, k=2)
                if generative:
                    chosen[0] = "eeg_original"
            else:
                chosen = channel_names

            samples = []
            for src in indices:

                with np.load(src.path) as npz:
                    payload = {
                        k: self.extractors[k](npz, src.start, src.end) for k in chosen
                    }
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
                    raise ValueError(
                        f"Token length mismatch > tolerance in sample {sample.id}: {lengths}"
                    )

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
                padded = pad_sequence(token_seqs, batch_first=True, padding_value=0.0)
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

        return DataLoader(
            self,
            **dl_kwargs,
            collate_fn=collate_fn,
            sampler=sampler,
            drop_last=True,
        )


def _equal_matrix_from_ids(vals: t.Sequence[str]) -> torch.Tensor:
    """
    将字符串/可哈希对象列表映射为整数 ID，并返回 pairwise 相等矩阵。
    返回形状 [N, N]，元素∈{0,1}，dtype=float32。
    """
    mapping = {}
    ids = []
    for v in vals:
        v = str(v)
        if v not in mapping:
            mapping[v] = len(mapping)
        ids.append(mapping[v])
    x = torch.tensor(ids, dtype=torch.long)
    return (x[:, None] == x[None, :]).to(torch.float32)


def build_w_h_age_sex_center(
    age: torch.Tensor,  # [N], float，缺失=-1
    sex: torch.Tensor,  # [N], long，{0,1,-1}
    center: t.Sequence[str],  # [N]，采集中心/医院 ID
    path: t.Sequence[str],  # [N]，同受试者/同一晚的唯一路径或标识
    *,
    sigma_age: float = 20.0,  # Laplace 核的 σ（越大越平滑）
    alpha_sex: float = 0.8,  # 异性别衰减系数（0~1）   设为1关闭
    # alpha_sex: float = 1.0,     # 异性别衰减系数（0~1）   设为1关闭
    gamma_same: float = 1.3,  # w 的中心门控（同中心放大）  设为1关闭
    gamma_diff: float = 0.8,  # w 的中心门控（跨中心缩小）  设为1关闭
    # gamma_same: float = 1.0,    # w 的中心门控（同中心放大）  设为1关闭
    # gamma_diff: float = 1.0,    # w 的中心门控（跨中心缩小）  设为1关闭
    eps: float = 1e-6,  # 数值稳定用
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    构造两个 [N,N] 矩阵：
      - w：负样本权重（仅对非同 path 的 off-diag 归一化到均值=1；同 path 的负样本被中和为 eps）
      - h：同 path 的 off-diag 掩码（=1），用于在 loss 中 down-weight（logits -= λ*h/T）

    设计要点：
      - w = k(age,sex) * center_gate；同 path 的位置直接设为 eps，并从行归一化里排除
      - h = same_path_off（常数 1），避免因元数据缺失导致 “同 path 不被 down-weight”
    """
    # --- 准备张量 ---
    N = age.shape[0]
    age = age.detach().cpu().float()  # [N]
    sex = sex.detach().cpu().long()  # [N]
    # print(f"sex: {sex}")
    # print(f"age: {age}")
    # print(f"center: {center}")
    # print(f"path: {path}")

    # --- 1) age+sex 相似核 k_ij（Laplace，更平可调大 sigma_age）---
    ai, aj = age[:, None], age[None, :]
    valid_age = (ai >= 0) & (aj >= 0)
    d = (ai - aj).abs()
    age_sim = torch.zeros(N, N, dtype=torch.float32)
    if valid_age.any():
        age_sim = torch.exp(-d / float(sigma_age)) * valid_age.float()  # [N,N]
        # age_sim = torch.ones(N, N, dtype=torch.float32)

    si, sj = sex[:, None], sex[None, :]
    valid_sex = (si >= 0) & (sj >= 0)
    same_sex = (si == sj) & valid_sex
    diff_sex = (si != sj) & valid_sex
    sex_coef = same_sex.float() + float(alpha_sex) * diff_sex.float()  # [N,N]

    k = (age_sim * sex_coef).clamp(0.0, 1.0)  # [N,N]
    # print(f"age_sim: {age_sim}")
    # print(f"sex_coef: {sex_coef}")
    # print(f"k: {k}")

    # --- 2) center 门控（仅用于 w）---
    same_center = _equal_matrix_from_ids(center)  # [N,N] in {0,1}
    w_gate = same_center * float(gamma_same) + (1.0 - same_center) * float(gamma_diff)

    # --- 3) 同 path 掩码 ---
    same_path = _equal_matrix_from_ids(path)  # [N,N]，对角=1
    offdiag = torch.ones(N, N, dtype=torch.float32) - torch.eye(N, dtype=torch.float32)
    same_path_off = same_path * offdiag  # 仅 off-diag 的同 path 为 1

    # --- 4) w：原设计 + 显式中和同 path 的负样本 ---
    w_raw = k * w_gate + eps  # [N,N]
    w_raw.fill_diagonal_(1.0)  # 对角不使用，但设为 1 稳定

    # 同 path 负样本位置：设极小常数 eps，避免被放大
    w = w_raw.clone()
    w[same_path_off.bool()] = eps

    # 仅在 “有效负样本”（非同 path 的 off-diag）上做行归一化，使其平均=1
    valid_neg_mask = (offdiag - same_path_off).clamp_min(0.0)  # [N,N]，1=有效负样本
    with torch.no_grad():
        num_valid = valid_neg_mask.sum(1, keepdim=True)  # [N,1]
        # 兜底：若某行全被同 path 遮掉，没有有效负样本，则回退为“允许所有 off-diag”
        need_fallback = num_valid.squeeze(1) == 0
        if need_fallback.any():
            rows = need_fallback.nonzero(as_tuple=True)[0]
            w[rows] = w_raw[rows]
            valid_neg_mask[rows] = offdiag[rows]
            num_valid[rows] = offdiag[rows].sum(1, keepdim=True)

        denom = (w * valid_neg_mask).sum(1, keepdim=True) / (num_valid + eps)
        w = w / denom.clamp_min(eps)

    # --- 5) h：仅同 path 的 off-diag 生效（常数 1），用于在 logits 中做 down-weight ---
    h = same_path_off.clone().to(torch.float32)
    h.fill_diagonal_(0.0)

    return w, h


def safe_cast(v, default=-1):
    try:
        if isinstance(v, str) and v.lower() == "nan":
            return default
        if v is None:
            return default
        if isinstance(v, float) and math.isnan(v):
            return default
        return int(float(v))  # 支持 '42.0' 这样的字符串
    except:
        return default


def process_metadata(samples, disease_names):
    batch_metadata = {
        "age": [],
        "sex": [],
        "source": [],
        "path": [],
    }
    for disease_name in disease_names:
        batch_metadata[disease_name] = []

    for s in samples:
        meta = s.metadata
        batch_metadata["age"].append(meta.get("age", "nan"))
        batch_metadata["sex"].append(meta.get("sex", "nan"))
        batch_metadata["source"].append(meta.get("source", "nan"))
        batch_metadata["path"].append(meta.get("path", "nan"))
        for disease_name in disease_names:
            batch_metadata[disease_name].append(meta.get(disease_name, "nan"))

    processed = {}
    # return processed

    # 处理 age（转为 float tensor）
    processed["age"] = torch.tensor(
        [safe_cast(v, -1) for v in batch_metadata["age"]], dtype=torch.float
    )

    # 处理性别（0: male, 1: female, -1: 其他或缺失）

    def encode_binary_label(v):
        # 字符串：保留你原有规则
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"male", "1", "1.0", "x"}:
                return 1
            elif s in {"female", "0", "0.0"}:
                return 0
            else:
                raise NotImplementedError(f"Unknown sex: {v!r}")

        # int（排除 bool）/ numpy integer：必须为 0 或 1
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            if v in (0, 1):
                return int(v)
            raise NotImplementedError(f"Invalid int sex: {v}, must be 0 or 1")

        # float / numpy floating：必须严格等于 0.0 或 1.0；NaN 视为缺失
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return -1
            if v in (0.0, 1.0):
                return int(v)
            raise NotImplementedError(
                f"Invalid float binary label: {v}, must be 0.0 or 1.0"
            )

        # 其它类型：按缺失处理
        return -1

    processed["sex"] = torch.tensor(
        [encode_binary_label(v) for v in batch_metadata["sex"]], dtype=torch.long
    )
    for disease_name in disease_names:
        processed[disease_name] = torch.tensor(
            [encode_binary_label(v) for v in batch_metadata[disease_name]],
            dtype=torch.long,
        )

    processed["source"] = [v for v in batch_metadata["source"]]
    processed["path"] = [v for v in batch_metadata["path"]]
    return processed


def extract_binary_labels(dataset, target_name: str):
    # dataset.data 里是你的 Sample，对每个 Sample 的 metadata 读一次
    labels = np.fromiter(
        (
            (
                int(float(s.metadata[target_name]))
                if (hasattr(s, "metadata") and (target_name in s.metadata))
                else -1
            )
            for s in dataset.data
        ),
        dtype=np.int64,
    )
    return labels  # 值域 {-1, 0, 1}


def make_weighted_sampler_from_labels(
    labels: np.ndarray, epoch_size: int | None = None
):
    valid = labels != -1
    if not valid.any():
        return None  # 没有效标签就不用加权

    # 类频统计（仅 0/1）
    uniq, counts = np.unique(labels[valid], return_counts=True)
    class_weight = {
        int(c): 1.0 / float(n) for c, n in zip(uniq.tolist(), counts.tolist())
    }

    # 为每个样本分配权重；无效标签 -> 0
    w = np.zeros_like(labels, dtype=np.float32)
    for y in (0, 1):
        if y in class_weight:
            w[labels == y] = class_weight[y]

    num_samples = int(valid.sum()) if epoch_size is None else int(epoch_size)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.float32),
        num_samples=num_samples,
        replacement=True,
    )
    return sampler

