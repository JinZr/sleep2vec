# Datasets And Samplers

## `data.psg_pretrain_dataset._build_channel_registry`

- File: `data/psg_pretrain_dataset.py`
- Signature: `_build_channel_registry(*, channel_names: Sequence[str], channel_input_dims: Mapping[str, int], mask_rate: float) -> dict[str, tuple[Callable, Callable, Callable]]`
- Purpose and contract: build the extractor/tokenizer/mask-generator registry for requested channels, including built-in `stage5` and `ahi` behavior.
- Important inputs/outputs: requested channels and YAML-driven input dims in, per-channel registry out.
- Side effects: none.
- Key callers/callees: caller is `PSGPretrainDataset.__init__`; callees include `default_extractor`, `default_tokenizer`, and `default_mlm_mask_generator`.
- Reuse guidance: use this registry path instead of constructing per-channel extractors ad hoc.
- Duplication risk notes: built-in `ahi` and `stage5` registration rules belong here.

## `PSGPretrainDataset.__init__`

- File: `data/psg_pretrain_dataset.py`
- Signature: `PSGPretrainDataset.__init__(channel_names, save_preset_path, load_preset_path, index, split, max_tokens, *, channel_input_dims, token_sec=30, stride_tokens=0, mask_rate=0.15, few_shot=None, meta_data_names=None, meta_data_regression_names=None, sources=None, pair_selector=None, randomly_select_channels=True, min_channels=2, allow_missing_channels=False, bucket_by_available_channels=False, train_pair_probs=None, train_pair_track_unique_samples=False, generative=False, is_train_set=True, filter_max_workers=None, **kwargs)`
- Purpose and contract: canonical PSG dataset constructor. It either loads a preset or reads CSV indexes, windows each row into `SampleIndex` records, copies optional `age`/`sex` metadata only when present, builds the channel registry, and delegates the rest to `DefaultDataset`.
- Important inputs/outputs: channel list, YAML-driven channel widths, preset/index source, split, token windowing, and batching flags in; dataset instance out.
- Side effects: when building from CSV, stamps `metadata["source"]` from the CSV path, preserves optional metadata fields, and expands each row into one or more `SampleIndex` windows.
- Key callers/callees: callers are `sleep2vec.utils` and `preprocess/save_dataset_presets.py`; callees are `window`, `_build_channel_registry`, and `DefaultDataset.__init__`.
- Reuse guidance: use this class for PSG-style NPZ/preset loading instead of building `SampleIndex` lists manually.
- Duplication risk notes: the built-in label-channel registry lives here and should not be replicated in caller code.

## `data.kaldi_io.KaldiReaderPool`

- File: `data/kaldi_io.py`
- Signature: `KaldiReaderPool(root: str | Path, channel_specs: Mapping[str, KaldiChannelSpec])`
- Purpose and contract: lazily open process-local sorted `scp:` readers for Kaldi matrix channels and return NumPy matrices by `(channel, sample_key)`.
- Important inputs/outputs: Kaldi root and per-channel `KaldiChannelSpec` mappings in; channel matrices out.
- Side effects: imports optional `kaldi_native_io`, opens readers, closes reader handles, and reopens handles after DataLoader worker process changes.
- Key callers/callees: caller is `KaldiPSGDataset._load_tokens_for_src`; callee is `kaldi_native_io.RandomAccessFloatMatrixReader`.
- Reuse guidance: use this pool for runtime Kaldi reads instead of opening raw readers in datasets or collate functions.
- Duplication risk notes: sorted `s,scp:` reader URI construction and missing-key errors belong here.

## `KaldiPSGDataset.__init__`

- File: `data/kaldi_psg_dataset.py`
- Signature: `KaldiPSGDataset(channel_names, kaldi_data_root, manifest, split, max_tokens, *, channel_input_dims, mask_rate=0.15, few_shot=None, meta_data_names=None, meta_data_regression_names=None, sources=None, pair_selector=None, randomly_select_channels=True, min_channels=2, allow_missing_channels=False, bucket_by_available_channels=False, train_pair_probs=None, train_pair_track_unique_samples=False, generative=False, is_train_set=True, **kwargs)`
- Purpose and contract: build the same runtime batch contract as `PSGPretrainDataset` from a Kaldi `manifest.json` format v2 root rather than NPZ/preset inputs.
- Important inputs/outputs: configured channels, Kaldi root, split manifest, and split name in; dataset instance out.
- Side effects: reads `manifest.json`, selects the requested split, builds channel specs, reads the split CSV, creates `SampleIndex` records with `payload["available_channels"]`, and opens Kaldi readers lazily.
- Key callers/callees: callers are `sleep2vec.utils.get_pretrain_dataloader` and `_build_finetune_loader` when `data_backend="kaldi"`; callees include `_load_manifest_json`, `_load_channel_specs`, `_load_manifest_samples`, `DefaultDataset.__init__`, and `KaldiReaderPool`.
- Reuse guidance: use this class for Kaldi-backed pretrain, finetune, and inference. It intentionally reuses `DefaultDataset.dataloader` instead of introducing a separate collate path.
- Duplication risk notes: split CSV column requirements, `available_channels` parsing, manifest input-dim checks, and token-length validation should stay here.

## `data.utils.load_builtin_ahi_metadata`

- File: `data/utils.py`
- Signature: `load_builtin_ahi_metadata(npz) -> tuple[float, float]`
- Purpose and contract: validate the built-in AHI NPZ contract and return scalar `(ahi, tst)` values after requiring `ah_event`, scalar `ahi`, and scalar `tst`.
- Important inputs/outputs: open NPZ handle in, `(ahi, tst)` floats out.
- Side effects: none.
- Key callers/callees: callers are `filter_valid_sample_indices` and `DefaultDataset.dataloader`; callee is `_load_scalar_npz_value`.
- Reuse guidance: use this helper whenever built-in AHI metadata needs to be read from NPZ.
- Duplication risk notes: scalar-key validation must stay centralized here.

## `data.utils.filter_valid_sample_indices`

- File: `data/utils.py`
- Signature: `filter_valid_sample_indices(data, extractors, tokenizers, *, allow_missing_channels, channel_names=None, min_channels=2, tolerance=1, max_workers=None) -> list`
- Purpose and contract: validate each sample by opening the NPZ, extracting/tokenizing relevant channels, rejecting unreadable or length-mismatched samples, validating built-in AHI metadata when needed, and recording `payload["available_channels"]` in missing-channel mode.
- Important inputs/outputs: raw `SampleIndex` list in; filtered `SampleIndex` list out.
- Side effects: mutates `sample_index.payload["available_channels"]` for retained samples in missing-channel mode; may backfill `sample_index.metadata["ahi"]` and `["tst"]`.
- Key callers/callees: caller is `DefaultDataset.__init__`; callees are `load_npz`, `load_builtin_ahi_metadata`, extractors, and tokenizers.
- Reuse guidance: this is the canonical preset-validation step.
- Duplication risk notes: pair-first samplers rely on its `available_channels` side effect.

## `DefaultDataset.__init__`

- File: `data/default_dataset.py`
- Signature: `DefaultDataset.__init__(save_preset_path, load_preset_path, data, split, extractors, tokenizers, mask_generators, dataloader_config, few_shot=None, meta_data_names=None, meta_data_regression_names=None, sources=None, pair_selector=None, seed=42, filter_max_workers=None) -> None`
- Purpose and contract: own the lifecycle for either loading a preset or validating raw `SampleIndex` entries, then applying metadata/split/source filtering and optional few-shot selection.
- Important inputs/outputs: preset paths or raw `SampleIndex` records in; dataset state on `self`.
- Side effects: loads or writes pickle presets, mutates `self.data`.
- Key callers/callees: caller is `PSGPretrainDataset.__init__`; callees are `filter_valid_sample_indices`, `filter_with_metadata`, and `select_few_shot`.
- Reuse guidance: use this base class when new datasets share the same collate semantics.
- Duplication risk notes: preset loading and sample validation should stay centralized here.

## `DefaultDataset._get_available_channels_for_src` and `_load_tokens_for_src`

- File: `data/default_dataset.py`
- Signatures:
  - `_get_available_channels_for_src(self, src: SampleIndex) -> set[str]`
  - `_load_tokens_for_src(self, src: SampleIndex, chosen_channels: list[str]) -> tuple[dict, dict, dict, dict]`
- Purpose and contract: storage hooks used by `DefaultDataset.dataloader` to discover available channels and materialize payload/tokens/masks/metadata for one sample.
- Important inputs/outputs: `SampleIndex` and selected channels in; available channel set or collate-ready payload maps out.
- Side effects: default implementation reads NPZ files and may backfill `ahi`/`tst` metadata.
- Key callers/callees: caller is the nested collate function inside `DefaultDataset.dataloader`; override implementer is `KaldiPSGDataset`.
- Reuse guidance: override these hooks when adding a storage backend while preserving the canonical batch contract.
- Duplication risk notes: do not fork `DefaultDataset.dataloader` just to change where tokens are read from.

## `DefaultDataset.filter_with_metadata`

- File: `data/default_dataset.py`
- Signature: `DefaultDataset.filter_with_metadata(self) -> list[SampleIndex]`
- Purpose and contract: drop samples that lack requested metadata, do not match configured sources, or do not belong to requested splits, while allowing built-in AHI summary scalars to come from NPZ backfill instead of CSV columns.
- Important inputs/outputs: operates on `self.data`; returns the filtered list.
- Side effects: mutates `self.data`.
- Key callers/callees: called during dataset initialization.
- Reuse guidance: use this path for metadata/source/split filtering rather than re-filtering in callers.
- Duplication risk notes: filtering semantics belong at dataset-construction time, not in loaders or trainers.

## `DefaultDataset.select_few_shot`

- File: `data/default_dataset.py`
- Signature: `DefaultDataset.select_few_shot(self) -> list[SampleIndex]`
- Purpose and contract: deterministically subsample `self.data` by count or proportion using the dataset seed.
- Important inputs/outputs: uses `self.few_shot`; returns the selected sample list.
- Side effects: mutates `self.data`.
- Key callers/callees: called during dataset initialization after metadata filtering.
- Reuse guidance: use for standard few-shot contraction.
- Duplication risk notes: selection order must remain seed-stable.

## `DefaultDataset.dataloader`

- File: `data/default_dataset.py`
- Signature: `DefaultDataset.dataloader(self, device: str = "cpu") -> DataLoader`
- Purpose and contract: canonical runtime collate path. It decides channel choice, reads NPZ slices, tokenizes, builds masks, pads sequences, constructs `token_start`, metadata tensors, `w/h`, and selects the correct sampler.
- Important inputs/outputs: dataset state in; fully configured `DataLoader` out.
- Side effects: nested `collate_fn` performs NPZ I/O on every batch, may select channels dynamically, and backfills built-in AHI metadata into sample-level metadata.
- Key callers/callees: callers are `sleep2vec.utils` and preprocessing preset generation; callees include `load_npz`, `load_builtin_ahi_metadata`, `process_metadata`, `build_w_h_age_sex_center`, `PairFirstBatchSampler`, and `AvailableChannelsBucketBatchSampler`.
- Reuse guidance: this is the canonical place to change batch structure.
- Duplication risk notes: avoid adding parallel collate implementations elsewhere in the repo.

## `data.metadata.build_w_h_age_sex_center`

- File: `data/metadata.py`
- Signature: `build_w_h_age_sex_center(age, sex, center, path, *, sigma_age=20.0, alpha_sex=0.8, gamma_same=1.3, gamma_diff=0.8, eps=1e-6) -> tuple[torch.Tensor, torch.Tensor]`
- Purpose and contract: build the negative-weight matrix `w` and same-path mask `h` used by weighted InfoNCE.
- Important inputs/outputs: age/sex/center/path metadata in; two `[N, N]` matrices out.
- Side effects: none.
- Key callers/callees: caller is `DefaultDataset.dataloader`.
- Reuse guidance: use whenever weighted contrastive sampling depends on demographic/path-aware weights.
- Duplication risk notes: keep `w/h` semantics aligned with `WeightedInfoNCELoss`.

## `data.metadata.process_metadata`

- File: `data/metadata.py`
- Signature: `process_metadata(samples, disease_names, regression_names: Sequence[str] | None = None)`
- Purpose and contract: convert sample metadata dictionaries into tensorized batch metadata, including binary-label normalization and regression handling.
- Important inputs/outputs: sample list in; dict of tensors plus source/path lists out.
- Side effects: none.
- Key callers/callees: caller is `DefaultDataset.dataloader`; callees include `safe_cast`, `safe_cast_float`, and `_encode_binary_label`.
- Reuse guidance: this is the canonical metadata tensorization path.
- Duplication risk notes: do not create independent metadata encoders in trainers.

## Pair selection helpers

- File: `data/channel_selection.py`
- Functions and methods:
  - `build_all_pairs(channel_names: Sequence[str]) -> list[tuple[str, str]]`
  - `RoundRobinPairSelector.select(available: Sequence[str]) -> list[str]`
- Purpose and contract: enumerate channel pairs and choose an available scheduled pair in round-robin order.
- Important inputs/outputs: channel names in; pair list or chosen pair out.
- Side effects: `RoundRobinPairSelector` mutates internal cursor state.
- Key callers/callees: callers are `sleep2vec.utils.get_pretrain_dataloader`, `PairAccLoggerCallback`, and `DefaultDataset.dataloader`.
- Reuse guidance: use these helpers instead of open-coding pair enumeration or scheduler state.
- Duplication risk notes: pair enumeration is also used in sampler initialization; avoid spreading more variants.

## `data.samplers.handles_distributed_sharding`

- File: `data/samplers.py`
- Signature: `handles_distributed_sharding(batch_sampler: Any) -> bool`
- Purpose and contract: report whether a batch sampler already shards batches across distributed ranks.
- Important inputs/outputs: batch sampler in, boolean out.
- Side effects: none.
- Key callers/callees: callers are `sleep2vec.utils.get_pretrain_dataloader` and `sleep2vec.adapt.sleep2vec_adapt`.
- Reuse guidance: use this helper when configuring Lightning `use_distributed_sampler`.
- Duplication risk notes: sampler sharding detection should not be reimplemented in entrypoints.

## `AvailableChannelsBucketBatchSampler.__iter__`

- File: `data/samplers.py`
- Signature: `AvailableChannelsBucketBatchSampler.__iter__(self)`
- Purpose and contract: yield homogeneous batches drawn from buckets defined by exact `available_channels` signatures, with distributed-aware sharding by global batch index.
- Important inputs/outputs: sampler state in; batches of dataset indices out.
- Side effects: updates internal epoch counter unless `set_epoch` is driving it manually.
- Key callers/callees: caller is `DefaultDataset.dataloader` when missing-channel bucketing is enabled.
- Reuse guidance: use when batch homogeneity matters but pair-first sampling is not required.
- Duplication risk notes: Lightning distributed sampler injection must stay disabled when using this sampler.

## `PairFirstBatchSampler.__init__` and `__iter__`

- File: `data/samplers.py`
- Signatures:
  - `PairFirstBatchSampler.__init__(..., channel_names, batch_size, min_channels=2, shuffle=True, drop_last=True, seed=0, pair_sampling="uniform", pair_probs=None, track_unique_sample_counts=False)`
  - `PairFirstBatchSampler.__iter__(self)`
- Purpose and contract: precompute per-pair sample pools from `payload["available_channels"]`, reject empty configured pairs, and emit batches tagged with one chosen channel pair.
- Important inputs/outputs: dataset records plus channel list in; `[(index, pair), ...]` batches out.
- Side effects: caches last-epoch pair counts, optional unique-sample counts, supports target-distribution updates, and increments epoch counter unless manual epoch control is used.
- Key callers/callees: caller is `DefaultDataset.dataloader`; observers are `PairAccLoggerCallback`, `AdaptPairScheduleCallback`, and tests.
- Reuse guidance: canonical sampler for missing-channel pretraining and adaptation with explicit pair scheduling.
- Duplication risk notes: relies on `filter_valid_sample_indices` having already populated `available_channels`.

## `SequentialPairEvalBatchSampler`

- File: `data/samplers.py`
- Signature: `SequentialPairEvalBatchSampler(data: Sequence[Any], *, channel_names: Sequence[str], batch_size: int, min_channels: int = 2)`
- Purpose and contract: build validation/test batches that iterate feasible modality pairs deterministically while still carrying the scheduled pair into `DefaultDataset.__getitem__`.
- Important inputs/outputs: dataset records and channel list in; batches of `(index, pair)` tuples out.
- Side effects: none beyond iterator state.
- Key callers/callees: caller is `sleep2vec.utils.get_pretrain_dataloader`.
- Reuse guidance: use this sampler for pair-aware evaluation rather than building one loader object per pair.
- Duplication risk notes: pretrain validation now depends on this sampler instead of separate per-pair loaders.

## `sleep2vec.utils.get_pretrain_dataloader`

- File: `sleep2vec/utils.py`
- Signature: `get_pretrain_dataloader(args)`
- Purpose and contract: build the standard pretrain/adapt train loader plus one sequential pair-eval validation loader from the current CLI namespace, dispatching to `PSGPretrainDataset` or `KaldiPSGDataset` according to `args.data_backend`.
- Important inputs/outputs: pretrain-style `args` in; `(train_loader, val_loader)` out.
- Side effects: seeds RNGs and configures sampler behavior based on missing-channel flags.
- Key callers/callees: callers are `pretrain.sleep2vec_pretrain` and `adapt.sleep2vec_adapt`; callees include `_dataset_class_for_args`, `PSGPretrainDataset`, `KaldiPSGDataset`, `build_all_pairs`, `RoundRobinPairSelector`, `PairFirstBatchSampler`, `AvailableChannelsBucketBatchSampler`, and `SequentialPairEvalBatchSampler`.
- Reuse guidance: use for any standard pretrain or adaptation runtime path.
- Duplication risk notes: keep missing-channel argument normalization here rather than in the entrypoint.

## `sleep2vec.utils._build_finetune_loader` and `get_finetune_dataloaders`

- File: `sleep2vec/utils.py`
- Signatures:
  - `_build_finetune_loader(args, *, split, sources, shuffle, is_train_set, few_shot=None)`
  - `get_finetune_dataloaders(args)`
- Purpose and contract: build finetune train/val/test loaders with correct metadata label wiring, built-in sequence label-channel insertion, AHI auxiliary `stage5` injection, backend-specific dataset kwargs, and fail-fast validation that built-in `age`/`sex` runs have valid labels after split/source filtering.
- Important inputs/outputs: normalized finetune `args` in; one loader or three loaders out.
- Side effects: seed initialization in `get_finetune_dataloaders`.
- Key callers/callees: callers are `prepare_dataloader` and `_build_inference_loader`; callees are `_dataset_class_for_args`, `PSGPretrainDataset.dataloader`, and `KaldiPSGDataset.dataloader`.
- Reuse guidance: use these helpers for any finetune or inference data-loading path.
- Duplication risk notes: built-in label-channel insertion and metadata-label selection should not be duplicated in trainer code.
