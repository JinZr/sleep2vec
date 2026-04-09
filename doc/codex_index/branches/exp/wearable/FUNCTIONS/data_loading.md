# Functions: Data Loading

## `data.utils.filter_valid_sample_indices`

- File: `data/utils.py`
- Signature: `filter_valid_sample_indices(data, extractors, tokenizers, *, allow_missing_channels, channel_names=None, min_channels=2, tolerance=1, max_workers=None) -> list[Any]`
- Purpose and contract: validates preset candidates by loading NPZ slices, tokenizing requested channels, checking token-length consistency, and optionally recording `payload["available_channels"]`.
- Important inputs: `SampleIndex` list, extractor/tokenizer mappings, missing-channel policy.
- Important outputs: filtered `SampleIndex` list.
- Side effects: loads NPZ files, uses a thread pool, mutates `sample_index.payload` when missing-channel mode is enabled.
- Notable callers/callees: used by `DefaultDataset.__init__`.
- Reuse guidance: canonical preset-validation path.
- Duplication-risk notes: high because samplers depend on its `available_channels` payload.

## `data.psg_pretrain_dataset._build_channel_registry`

- File: `data/psg_pretrain_dataset.py`
- Signature: `_build_channel_registry(*, channel_names, channel_input_dims, mask_rate) -> dict[str, tuple[Callable, Callable, Callable]]`
- Purpose and contract: maps requested channels to extractor, tokenizer, and mask-generator triples; requires explicit `channel_input_dims` for all non-`stage5` channels.
- Important inputs: channel name list, per-channel input widths, mask rate.
- Important outputs: registry keyed by channel name.
- Side effects: none.
- Notable callers/callees: used by `PSGPretrainDataset.__init__`; delegates to `data.utils.default_extractor`, `default_tokenizer`, and `default_mlm_mask_generator`.
- Reuse guidance: canonical bridge from YAML channel specs to dataset mechanics.
- Duplication-risk notes: high.

## `data.psg_pretrain_dataset.PSGPretrainDataset.__init__`

- File: `data/psg_pretrain_dataset.py`
- Signature: `PSGPretrainDataset(..., channel_input_dims, token_sec=30, stride_tokens=0, mask_rate=0.15, ..., allow_missing_channels=False, bucket_by_available_channels=False, train_pair_probs=None, train_pair_track_unique_samples=False, generative=False, is_train_set=True, **kwargs)`
- Purpose and contract: builds dataset windows either from preset pickle or from one-or-more CSV files, attaches metadata, prepares extractor/tokenizer/mask mappings, and delegates filtering and dataloader behavior to `DefaultDataset`.
- Important inputs: `channel_names`, `channel_input_dims`, preset or CSV index, split selection, token/window parameters, missing-channel controls.
- Important outputs: dataset object.
- Side effects: reads CSVs or preset pickles; may save preset pickles via the base class.
- Notable callers/callees: called by `sleep2vec.utils.get_pretrain_dataloader` and `_build_finetune_loader`.
- Reuse guidance: canonical dataset class for both pretrain and finetune flows.
- Duplication-risk notes: high.

## `data.default_dataset.DefaultDataset.filter_with_metadata`

- File: `data/default_dataset.py`
- Signature: `filter_with_metadata(self) -> list[SampleIndex]`
- Purpose and contract: filters loaded samples by requested metadata presence, source restrictions, and split membership.
- Important inputs: dataset state (`meta_data_names`, `sources`, `split`).
- Important outputs: filtered `SampleIndex` list.
- Side effects: mutates `self.data`.
- Notable callers/callees: called during `DefaultDataset.__init__`.
- Reuse guidance: use this path instead of custom metadata filtering in loaders.
- Duplication-risk notes: medium.

## `data.default_dataset.DefaultDataset.dataloader`

- File: `data/default_dataset.py`
- Signature: `dataloader(self, device: str = "cpu") -> DataLoader`
- Purpose and contract: materializes a `DataLoader` with a collate function that loads NPZ windows, selects channels or scheduled pairs, pads tokens/masks, builds metadata tensors, and chooses between weighted sampling, pair-first sampling, bucketed sampling, or default batching.
- Important inputs: dataset attributes such as `allow_missing_channels`, `bucket_by_available_channels`, `train_pair_probs`, `meta_data_names`, `is_train_set`.
- Important outputs: configured `torch.utils.data.DataLoader`.
- Side effects: loads NPZ data in collate, mutates batch shape based on missing-channel policy, may raise if scheduled pairs are invalid.
- Notable callers/callees: called by `sleep2vec.utils.get_pretrain_dataloader` and `_build_finetune_loader`; uses `PairFirstBatchSampler`, `AvailableChannelsBucketBatchSampler`, `process_metadata`, and `build_w_h_age_sex_center`.
- Reuse guidance: canonical collate path. New loader features should usually land here rather than in separate dataset classes.
- Duplication-risk notes: very high.

## `sleep2vec.utils.get_pretrain_dataloader`

- File: `sleep2vec/utils.py`
- Signature: `get_pretrain_dataloader(args)`
- Purpose and contract: seeds runtime randomness, derives missing-channel and worker settings, builds train and per-pair validation datasets, and filters validation samples to scheduled pairs when required.
- Important inputs: argparse namespace containing channel names, preset/index paths, device, batch size, missing-channel flags, and optional `train_pair_probs`.
- Important outputs: `(train_loader, val_loaders)`.
- Side effects: logging; dataset construction; validation filtering for pair support.
- Notable callers/callees: called by `sleep2vec/pretrain.py` and `sleep2vec/adapt.py`; uses `PSGPretrainDataset`, `build_all_pairs`, and `_filter_dataset_for_pair_support`.
- Reuse guidance: canonical pretrain/adapt loader factory.
- Duplication-risk notes: high.

## `sleep2vec.utils._build_finetune_loader`

- File: `sleep2vec/utils.py`
- Signature: `_build_finetune_loader(args, *, split, sources, shuffle, is_train_set, few_shot=None)`
- Purpose and contract: builds downstream loaders on top of `PSGPretrainDataset`, converts custom labels into metadata requirements, injects `stage5` as a token label when needed, and disables missing-channel behavior for downstream tasks.
- Important inputs: argparse namespace, split, source list, few-shot setting.
- Important outputs: downstream `DataLoader`.
- Side effects: dataset construction.
- Notable callers/callees: used by `get_finetune_dataloaders` and `sleep2vec.infer._build_inference_loader`.
- Reuse guidance: canonical downstream loader builder.
- Duplication-risk notes: high.

## `data.samplers.PairFirstBatchSampler`

- File: `data/samplers.py`
- Signature: `PairFirstBatchSampler(data, *, channel_names, batch_size, min_channels=2, shuffle=True, drop_last=True, seed=0, pair_sampling="uniform", pair_probs=None, track_unique_sample_counts=False)`
- Purpose and contract: first chooses a channel pair, then samples batch indices from that pair’s pool; requires every sample to expose `payload["available_channels"]`.
- Important inputs: filtered sample list, channel names, batch size, pair probabilities.
- Important outputs: batches of `(index, pair)` tuples.
- Side effects: tracks last-epoch pair counts and optional unique-sample coverage.
- Notable callers/callees: constructed by `DefaultDataset.dataloader`; reconfigured by `AdaptPairScheduleCallback`.
- Reuse guidance: canonical train sampler for missing-channel pretraining.
- Duplication-risk notes: very high.

## `data.samplers.AvailableChannelsBucketBatchSampler`

- File: `data/samplers.py`
- Signature: `AvailableChannelsBucketBatchSampler(data, *, batch_size, min_channels=2, shuffle=True, drop_last=True, shard_across_ranks=True, seed=0)`
- Purpose and contract: groups samples by exact available-channel signature and emits homogeneous batches, optionally already sharded by rank.
- Important inputs: filtered sample list with `payload["available_channels"]`, batching options.
- Important outputs: batches of integer indices.
- Side effects: internal epoch counter for deterministic shuffling.
- Notable callers/callees: constructed by `DefaultDataset.dataloader`; detected by `handles_distributed_sharding`.
- Reuse guidance: canonical fallback when pair-first sampling is not used but availability bucketing is still needed.
- Duplication-risk notes: medium-high.
