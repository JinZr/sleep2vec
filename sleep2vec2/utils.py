import logging
import random

import numpy as np
import torch

from sleep2vec2.common import is_builtin_seq_task
from sleep2vec2.data.kaldi_psg_dataset import KaldiPSGDataset
from sleep2vec2.data.metadata import _encode_binary_label, safe_cast
from sleep2vec2.data.psg_pretrain_dataset import PSGPretrainDataset
from sleep2vec2.data.samplers import SequentialPairEvalBatchSampler


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


def _dataset_class_for_args(args):
    data_backend = getattr(args, "data_backend", "npz")
    if data_backend == "npz":
        return PSGPretrainDataset
    if data_backend == "kaldi":
        return KaldiPSGDataset
    raise ValueError(f"Unknown data backend: {data_backend!r}.")


def get_pretrain_dataloader(args):
    dataset_cls = _dataset_class_for_args(args)
    data_backend = getattr(args, "data_backend", "npz")
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
    if data_backend == "npz":
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
    else:
        base_dataset_kwargs = dict(
            channel_names=args.channel_names,
            channel_input_dims=channel_input_dims,
            kaldi_data_root=args.kaldi_data_root,
            manifest=args.kaldi_manifest,
            max_tokens=args.max_tokens,
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
        return dataset_cls(
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
    val_dataset = build_pretrain_dataset(
        split=["val"],
        dataloader_kwargs=val_kwargs,
        pair_selector=None,
        train_pair_probs_override=None,
        train_pair_track_unique_samples_override=False,
        is_train_set=False,
    )
    val_batch_sampler = SequentialPairEvalBatchSampler(
        val_dataset.data,
        channel_names=args.channel_names,
        batch_size=args.batch_size,
        min_channels=min_channels,
    )
    val_dataset.dataloader_config["batch_sampler"] = val_batch_sampler
    val_loader = val_dataset.dataloader(device=args.device)
    logging.info(
        "Validation DataLoader created successfully! (pairs=%d, workers=%d)",
        len(val_batch_sampler.pairs),
        val_num_workers,
    )

    return train_loader, val_loader


def _build_finetune_loader(
    args,
    *,
    split,
    sources,
    shuffle,
    is_train_set,
    few_shot=None,
):
    dataset_cls = _dataset_class_for_args(args)
    data_backend = getattr(args, "data_backend", "npz")

    is_seq_task = is_builtin_seq_task(args.label_name)
    is_survival_task = bool(getattr(args, "is_survival", False))
    if is_survival_task:
        meta_data_names = []
        meta_data_regression_names = []
    elif args.label_name == "ahi":
        meta_data_names = ["ahi", "tst"]
        meta_data_regression_names = ["ahi", "tst"]
    else:
        meta_data_names = [] if args.label_name in {"age", "sex"} or is_seq_task else [args.label_name]
        meta_data_regression_names = [] if args.is_classification else list(meta_data_names)
    if meta_data_names and args.is_classification and args.output_dim > 2 and not is_seq_task:
        raise ValueError(
            "Metadata classification currently supports only binary labels (output_dim=2) "
            "for non-built-in sequence tasks. "
            f"Got --label-name '{args.label_name}' with finetune.task.output_dim={args.output_dim}. "
            "Extend metadata label encoding before using multiclass metadata targets."
        )
    dataset_channel_names = list(args.data_channel_names)
    dataset_channel_input_dims = dict(getattr(args, "channel_input_dims", {}) or {})
    label_source_name = getattr(args, "label_source_name", args.label_name)
    if is_seq_task and label_source_name not in dataset_channel_names:
        # Built-in sequence tasks consume runtime label channels from the NPZ batch.
        dataset_channel_names.append(label_source_name)
    for auxiliary_name in getattr(args, "auxiliary_label_source_names", []) or []:
        if auxiliary_name not in dataset_channel_names:
            dataset_channel_names.append(auxiliary_name)

    use_weighted_random_sampler = getattr(args, "weighted_random_sampler", False) and is_train_set
    dataset_kwargs = dict(
        channel_names=dataset_channel_names,
        channel_input_dims=dataset_channel_input_dims,
        split=split,
        max_tokens=args.max_tokens,
        mask_rate=0.0,
        meta_data_names=meta_data_names,
        meta_data_regression_names=meta_data_regression_names,
        sources=sources,
        randomly_select_channels=False,
        allow_missing_channels=False,
        min_channels=len(dataset_channel_names),
        weighted_random_sampler=use_weighted_random_sampler,
        weighted_random_sampler_target=args.label_name if use_weighted_random_sampler else None,
        survival_label_config=getattr(args, "survival", None) if is_survival_task else None,
        survival_output_dim=args.output_dim if is_survival_task else None,
        is_train_set=is_train_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )
    if data_backend == "npz":
        dataset_kwargs.update(
            save_preset_path=None,
            load_preset_path=args.finetune_preset_path,
            index=args.finetune_data_index,
            stride_tokens=args.max_tokens,
        )
    else:
        dataset_kwargs.update(
            kaldi_data_root=args.kaldi_data_root,
            manifest=args.kaldi_manifest,
        )

    if few_shot is not None:
        dataset_kwargs["few_shot"] = few_shot

    dataset = dataset_cls(**dataset_kwargs)
    if args.label_name in {"age", "sex"}:
        invalid = 0
        for sample in getattr(dataset, "data", []):
            metadata = getattr(sample, "metadata", {}) or {}
            value = metadata.get(args.label_name, None)
            if args.label_name == "age":
                valid = safe_cast(value, -1) != -1
            else:
                valid = _encode_binary_label(value) != -1
            if not valid:
                invalid += 1
        if invalid:
            total = len(getattr(dataset, "data", []))
            raise ValueError(
                f"Loaded preset/index has invalid or missing '{args.label_name}' labels for "
                f"{invalid}/{total} samples after split/source filtering. Regenerate the preset from "
                f"an index CSV with real '{args.label_name}' values before running --label-name {args.label_name}."
            )
    return dataset.dataloader(device=args.device)


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
