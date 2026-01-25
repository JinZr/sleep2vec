from contextlib import contextmanager
import logging
import random
import warnings

import numpy as np
from packaging import version
import torch

TORCH_VERSION = version.parse(torch.__version__)

from data.channel_selection import RoundRobinPairSelector, build_all_pairs
from data.psg_pretrain_dataset import PSGPretrainDataset


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

    if allow_missing_channels:
        logging.warning(
            "allow_missing_channels enabled: accepting samples with missing channels "
            "(min_channels=%d, bucket_by_available_channels=%s).",
            min_channels,
            bucket_by_available_channels,
        )
        if min_channels < 2:
            logging.warning("min_channels is < 2; contrastive pretraining may be unstable.")
        if not bucket_by_available_channels:
            logging.warning(
                "bucket_by_available_channels is disabled; mixed montages may collapse to a single best_pair."
            )
    else:
        logging.info("allow_missing_channels disabled: requiring all configured channels.")

    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "worker_init_fn": _seed_worker,
    }
    train_loader = PSGPretrainDataset(
        channel_names=args.channel_names,
        save_preset_path=None,
        load_preset_path=args.pretrain_preset_path,
        index=args.pretrain_data_index,
        split=["train"],
        max_tokens=args.max_tokens,
        stride_tokens=args.max_tokens,  # 0 for truncation
        mask_rate=args.mask_rate,
        use_legacy_body_movement=False,
        generative=False,
        allow_missing_channels=allow_missing_channels,
        min_channels=min_channels,
        bucket_by_available_channels=bucket_by_available_channels,
        is_train_set=True,
        **kwargs,
    ).dataloader(device=args.device)
    logging.info("Train DataLoader created successfully!")

    kwargs["shuffle"] = False

    val_pairs = build_all_pairs(args.channel_names)
    val_loaders = []
    for pair in val_pairs:
        pair_selector = RoundRobinPairSelector([pair])
        val_dataset = PSGPretrainDataset(
            channel_names=args.channel_names,
            save_preset_path=None,
            load_preset_path=args.pretrain_preset_path,
            index=args.pretrain_data_index,
            split=["val"],
            max_tokens=args.max_tokens,
            stride_tokens=args.max_tokens,  # 0 for truncation
            mask_rate=args.mask_rate,
            use_legacy_body_movement=False,
            generative=False,
            allow_missing_channels=allow_missing_channels,
            min_channels=min_channels,
            bucket_by_available_channels=bucket_by_available_channels,
            is_train_set=False,
            pair_selector=pair_selector,
            **kwargs,
        )
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
    dataset_channel_names = list(args.data_channel_names)
    if args.label_name == "stage5" and "stage5" not in dataset_channel_names:
        # stage5 is a per-token label; include it in the batch tokens so downstream loss can
        # read batch["tokens"]["stage5"] without treating it as an input modality.
        dataset_channel_names.append("stage5")

    dataset_kwargs = dict(
        channel_names=dataset_channel_names,
        save_preset_path=None,
        load_preset_path=args.finetune_preset_path,
        index=args.finetune_data_index,
        split=split,
        max_tokens=args.max_tokens,
        stride_tokens=args.max_tokens,
        mask_rate=0.0,
        use_legacy_body_movement=False,
        meta_data_names=meta_data_names,
        sources=sources,
        randomly_select_channels=False,
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


@contextmanager
def torch_autocast(device_type="cuda", **kwargs):
    """
    To fix the following warnings:
    /icefall/egs/librispeech/ASR/zipformer/model.py:323:
    FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated.
    Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(enabled=False):
    """
    if TORCH_VERSION >= version.parse("2.3.0"):
        # Use new unified API
        with torch.amp.autocast(device_type=device_type, **kwargs):
            yield
    else:
        # Suppress deprecation warning and use old CUDA-specific autocast
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            with torch.cuda.amp.autocast(**kwargs):
                yield
