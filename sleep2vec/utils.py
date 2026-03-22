import logging
import random

import numpy as np
import torch

from data.channel_selection import RoundRobinPairSelector, build_all_pairs
from data.psg_pretrain_dataset import PSGPretrainDataset
from data.utils import load_npz


def move_to_device(data, device="cuda"):
    """递归地将输入数据中的所有 torch.Tensor 转移到指定 device 上。"""
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_device(x, device) for x in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(x, device) for x in data)
    return data


def _resolve_available_channels(sample, channel_names: list[str]) -> set[str]:
    payload = getattr(sample, "payload", None)
    if isinstance(payload, dict):
        avail = payload.get("available_channels")
        if avail:
            return {str(ch) for ch in avail}

    sample_path = getattr(sample, "path", None)
    if not sample_path:
        return set()

    try:
        with load_npz(sample_path) as npz:
            return {str(ch) for ch in channel_names if ch in npz}
    except Exception:
        return set()


def _filter_dataset_for_pair_support(dataset, pair: tuple[str, str], channel_names: list[str]) -> None:
    left, right = str(pair[0]), str(pair[1])
    filtered = []
    for sample in dataset.data:
        avail = _resolve_available_channels(sample, channel_names)
        if left in avail and right in avail:
            filtered.append(sample)

    before = len(dataset.data)
    after = len(filtered)
    if after == 0:
        raise ValueError(
            "No validation samples support scheduled pair "
            f"{left}__{right}. Check allow_missing_channels/min_channels/index/preset consistency."
        )
    dataset.data = filtered
    if after < before:
        logging.info("Filtered val dataset for pair %s__%s: kept %d / %d samples", left, right, after, before)


def get_pretrain_dataloader(args):
    # set random seed
    seed = 12345
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    logging.info(f"args: {args}")

    def _seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    allow_missing_channels = bool(getattr(args, "allow_missing_channels", False))
    min_channels = int(getattr(args, "min_channels", 6))
    bucket_by_available_channels = bool(getattr(args, "bucket_by_available_channels", True))
    train_pair_probs = getattr(args, "train_pair_probs", None)
    train_pair_track_unique_samples = bool(getattr(args, "train_pair_track_unique_samples", False))
    val_num_workers = getattr(args, "val_num_workers", None)
    if val_num_workers is None:
        val_num_workers = 0 if allow_missing_channels else int(args.num_workers)
    else:
        val_num_workers = int(val_num_workers)
    if val_num_workers < 0:
        raise ValueError(f"val_num_workers must be >= 0, got {val_num_workers}")

    if allow_missing_channels:
        logging.warning(
            "allow_missing_channels enabled: accepting samples with missing channels "
            "(min_channels=%d, bucket_by_available_channels=%s, pair_sampling=uniform, "
            "train_pair_track_unique_samples=%s).",
            min_channels,
            bucket_by_available_channels,
            train_pair_track_unique_samples,
        )
        if min_channels < 2:
            logging.warning("min_channels is < 2; contrastive pretraining may be unstable.")
        if not bucket_by_available_channels:
            logging.warning(
                "bucket_by_available_channels is disabled; this only affects validation or other non-pair-first paths."
            )
    else:
        logging.info("allow_missing_channels disabled: requiring all configured channels.")
        train_pair_track_unique_samples = False

    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "worker_init_fn": _seed_worker,
    }
    channel_input_dims = dict(getattr(args, "channel_input_dims", {}) or {})
    base_dataset_kwargs = dict(
        channel_names=args.channel_names,
        channel_input_dims=channel_input_dims,
        save_preset_path=None,
        load_preset_path=args.pretrain_preset_path,
        index=args.pretrain_data_index,
        max_tokens=args.max_tokens,
        stride_tokens=args.max_tokens,  # 0 for truncation
        mask_rate=args.mask_rate,
        generative=False,
        allow_missing_channels=allow_missing_channels,
        min_channels=min_channels,
        bucket_by_available_channels=bucket_by_available_channels,
    )

    def build_pretrain_dataset(
        *,
        split,
        dataloader_kwargs,
        pair_selector=None,
        train_pair_probs_override=None,
        train_pair_track_unique_samples_override=False,
        is_train_set,
    ):
        return PSGPretrainDataset(
            **base_dataset_kwargs,
            split=split,
            pair_selector=pair_selector,
            train_pair_probs=train_pair_probs_override,
            train_pair_track_unique_samples=train_pair_track_unique_samples_override,
            is_train_set=is_train_set,
            **dataloader_kwargs,
        )

    train_loader = build_pretrain_dataset(
        split=["train"],
        dataloader_kwargs=kwargs,
        train_pair_probs_override=train_pair_probs,
        train_pair_track_unique_samples_override=train_pair_track_unique_samples,
        is_train_set=True,
    ).dataloader(device=args.device)
    logging.info("Train DataLoader created successfully!")

    val_kwargs = dict(kwargs)
    val_kwargs["shuffle"] = False
    val_kwargs["num_workers"] = val_num_workers
    logging.info("Validation DataLoader workers: %d", val_num_workers)
    val_pairs = build_all_pairs(args.channel_names)
    val_loaders = []
    for pair in val_pairs:
        pair_selector = RoundRobinPairSelector([pair])
        val_dataset = build_pretrain_dataset(
            split=["val"],
            dataloader_kwargs=val_kwargs,
            pair_selector=pair_selector,
            train_pair_probs_override=None,
            train_pair_track_unique_samples_override=False,
            is_train_set=False,
        )
        if allow_missing_channels:
            _filter_dataset_for_pair_support(val_dataset, pair, list(args.channel_names))
        val_dataset.pair = pair
        val_loaders.append(val_dataset.dataloader(device=args.device))
    logging.info("Valid DataLoaders created successfully! (pairs=%d)", len(val_loaders))

    return train_loader, val_loaders


def _build_finetune_loader(
    args,
    *,
    split,
    sources,
    shuffle,
    is_train_set,
    few_shot=None,
):
    meta_data_names = [] if args.label_name in {"age", "sex", "stage5"} else [args.label_name]
    meta_data_regression_names = [] if args.is_classification else list(meta_data_names)
    if meta_data_names and args.is_classification and args.output_dim > 2:
        raise ValueError(
            "Metadata classification currently supports only binary labels (output_dim=2) for non-stage5 tasks. "
            f"Got --label-name '{args.label_name}' with finetune.task.output_dim={args.output_dim}. "
            "Extend metadata label encoding before using multiclass metadata targets."
        )
    dataset_channel_names = list(args.data_channel_names)
    dataset_channel_input_dims = dict(getattr(args, "channel_input_dims", {}) or {})
    if args.label_name == "stage5" and "stage5" not in dataset_channel_names:
        # stage5 is a per-token label; include it in the batch tokens so downstream loss can
        # read batch["tokens"]["stage5"] without treating it as an input modality.
        dataset_channel_names.append("stage5")

    dataset_kwargs = dict(
        channel_names=dataset_channel_names,
        channel_input_dims=dataset_channel_input_dims,
        save_preset_path=None,
        load_preset_path=args.finetune_preset_path,
        index=args.finetune_data_index,
        split=split,
        max_tokens=args.max_tokens,
        stride_tokens=args.max_tokens,
        mask_rate=0.0,
        meta_data_names=meta_data_names,
        meta_data_regression_names=meta_data_regression_names,
        sources=sources,
        randomly_select_channels=False,
        allow_missing_channels=False,
        min_channels=len(dataset_channel_names),
        is_train_set=is_train_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )

    if few_shot is not None:
        dataset_kwargs["few_shot"] = few_shot

    return PSGPretrainDataset(**dataset_kwargs).dataloader(device=args.device)


def get_finetune_dataloaders(args):
    """Construct PSG finetune train/val/test loaders for downstream tasks."""
    seed = 4523
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset_names = args.train_dataset_names
    test_dataset_names = args.test_dataset_names

    train_loader = _build_finetune_loader(
        args,
        split=["train"],
        sources=train_dataset_names,
        shuffle=True,
        is_train_set=True,
        few_shot=args.n_few_shot,
    )

    val_loader = _build_finetune_loader(
        args,
        split=["val"],
        sources=train_dataset_names,
        shuffle=False,
        is_train_set=False,
    )

    test_loader = _build_finetune_loader(
        args,
        split=["test"],
        sources=test_dataset_names,
        shuffle=False,
        is_train_set=False,
    )

    return train_loader, val_loader, test_loader
