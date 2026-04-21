# Datasets And Samplers

## `PSGPretrainDataset.__init__`

- File: `data/psg_pretrain_dataset.py`
- Signature: `PSGPretrainDataset.__init__(channel_names, save_preset_path, load_preset_path, index, split, max_tokens, token_sec=30, stride_tokens=0, mask_rate=0.15, ..., allow_missing_channels=False, bucket_by_available_channels=False, train_pair_sampling=None, train_pair_track_unique_samples=False, generative=False, is_train_set=True, filter_max_workers=None, **kwargs)`
- Purpose and contract: canonical PSG dataset constructor. It either loads a preset or reads one or more CSV indexes, windows each row, builds the fixed channel registry, and delegates the rest to `DefaultDataset`. When built-in `ahi` requests summary metadata (`ahi`, `tst`), the CSV path does not require same-named index columns because those scalars come from NPZ backfill.
- Important inputs/outputs: channel list, preset/index source, split, token windowing, and batching flags in; dataset instance out.
- Side effects: when building from CSV, stamps `metadata["source"]` from the CSV path and expands each row into one or more `SampleIndex` windows.
- Key callers/callees: callers are `sleep2vec.utils` and `preprocess/save_dataset_presets.py`; callees are `window`, `default_extractor`, `default_tokenizer`, `default_mlm_mask_generator`, and `DefaultDataset.__init__`.
- Reuse guidance: use this class for PSG-style NPZ/preset loading instead of building `SampleIndex` lists manually.
- Duplication risk notes: the built-in runtime label registry (`stage5`, `ahi`) plus the special `ah_event` backing key for `ahi` live here and should not be replicated in caller code.

## `data.utils.filter_valid_sample_indices`

- File: `data/utils.py`
- Signature: `filter_valid_sample_indices(data, extractors, tokenizers, *, allow_missing_channels, channel_names=None, min_channels=2, tolerance=1, max_workers=None) -> list`
- Purpose and contract: validate each sample by opening NPZ sources path-by-path, extracting/tokenizing relevant channels for every window under that path, rejecting unreadable or length-mismatched samples, recording `payload["available_channels"]` in missing-channel mode, persisting built-in AHI scalar summaries (`ahi`, `tst`) into sample metadata when that contract is requested, and dropping built-in AHI samples whose tokenized `ah_event` labels are entirely ignore-valued (`-1`). When `max_workers=None`, the function uses its internal automatic thread-count policy.
- Important inputs/outputs: raw `SampleIndex` list in; filtered `SampleIndex` list out. `max_workers=None` means automatic worker selection; explicit integers pin the validation-thread budget.
- Side effects: mutates `sample_index.payload["available_channels"]` for retained samples in missing-channel mode.
- Key callers/callees: caller is `DefaultDataset.__init__`; callees are `load_npz`, extractors, tokenizers.
- Reuse guidance: this is the canonical preset-validation step.
- Duplication risk notes: pair-first samplers rely on its `available_channels` side effect.

## `DefaultDataset.__init__`

- File: `data/default_dataset.py`
- Signature: `DefaultDataset.__init__(save_preset_path, load_preset_path, data, split, extractors, tokenizers, mask_generators, dataloader_config, few_shot=None, meta_data_names=None, meta_data_regression_names=None, sources=None, pair_selector=None, seed=42, filter_max_workers=None) -> None`
- Purpose and contract: own the lifecycle for either loading a preset or validating raw `SampleIndex` entries, then applying metadata/split/source filtering and optional few-shot selection. `filter_max_workers` is a narrow pass-through knob for the sample-validation stage.
- Important inputs/outputs: preset paths or raw `SampleIndex` records in; dataset state on `self`.
- Side effects: loads or writes pickle presets, mutates `self.data`.
- Key callers/callees: caller is `PSGPretrainDataset.__init__`; callees are `filter_valid_sample_indices`, `filter_with_metadata`, and `select_few_shot`.
- Reuse guidance: use this base class when new datasets share the same collate semantics.
- Duplication risk notes: preset loading and sample validation should stay centralized here.

## `DefaultDataset.filter_with_metadata`

- File: `data/default_dataset.py`
- Signature: `DefaultDataset.filter_with_metadata(self) -> list[SampleIndex]`
- Purpose and contract: drop samples that lack requested metadata, do not match configured sources, or do not belong to requested splits. For built-in `ahi` runtime loading, missing serialized or CSV-indexed `ahi` / `tst` metadata is tolerated because those scalars are backfilled from NPZ during collate.
- Important inputs/outputs: operates on `self.data`; returns the filtered list.
- Side effects: mutates `self.data`.
- Key callers/callees: called during dataset initialization.
- Reuse guidance: use this path for metadata/source/split filtering rather than re-filtering in callers.
- Duplication risk notes: filtering semantics belong at dataset-construction time, not in loaders or trainers. Do not add a second AHI metadata-compatibility filter in finetune helpers.

## `DefaultDataset.select_few_shot`

- File: `data/default_dataset.py`
- Signature: `DefaultDataset.select_few_shot(self) -> list[SampleIndex]`
- Purpose and contract: deterministically subsample `self.data` by count or proportion using the dataset seed.
- Important inputs/outputs: uses `self.few_shot`; returns selected sample list.
- Side effects: mutates `self.data`.
- Key callers/callees: called during dataset initialization after metadata filtering.
- Reuse guidance: use for standard few-shot contraction.
- Duplication risk notes: selection order must remain seed-stable.

## `DefaultDataset.dataloader`

- File: `data/default_dataset.py`
- Signature: `DefaultDataset.dataloader(self, device: str = "cpu") -> DataLoader`
- Purpose and contract: canonical runtime collate path. It decides channel choice, reads NPZ slices, tokenizes, builds masks, pads sequences, constructs metadata tensors and `w/h`, and selects the correct sampler.
- Important inputs/outputs: dataset state in; fully configured `DataLoader` out.
- Side effects: nested `collate_fn` performs NPZ I/O on every batch and may select channels randomly.
- Key callers/callees: callers are `sleep2vec.utils` and preprocessing preset generation; callees include `load_npz`, `process_metadata`, `build_w_h_age_sex_center`, `PairFirstBatchSampler`, and `AvailableChannelsBucketBatchSampler`.
- Reuse guidance: this is the canonical place to change batch structure.
- Duplication risk notes: avoid adding parallel collate implementations elsewhere in the repo; ignore-value padding for runtime label channels belongs here. When missing-channel batches must recompute channel availability at collate time, built-in `ahi` must be resolved with the same `ah_event` / scalar-summary contract used by `filter_valid_sample_indices`, including legacy presets that do not serialize `payload["available_channels"]`.

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
- Key callers/callees: caller is `DefaultDataset.dataloader`; callees include `safe_cast`, `safe_cast_float`, `_encode_binary_label`.
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
- Duplication risk notes: pair enumeration is also open-coded inside `PairFirstBatchSampler`; avoid spreading that further.

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
- Side effects: caches last-epoch pair counts, optional unique-sample counts, and increments epoch counter unless manual epoch control is used.
- Key callers/callees: caller is `DefaultDataset.dataloader`; observers are `PairAccLoggerCallback` and tests.
- Reuse guidance: canonical sampler for missing-channel pretraining with explicit pair scheduling.
- Duplication risk notes: relies on `filter_valid_sample_indices` having already populated `available_channels`.

## `sleep2vec.utils.get_pretrain_dataloader`

- File: `sleep2vec/utils.py`
- Signature: `get_pretrain_dataloader(args)`
- Purpose and contract: build the standard pretrain train loader plus per-pair validation loaders from the current CLI namespace.
- Important inputs/outputs: pretrain `args` in; `(train_loader, val_loaders)` out.
- Side effects: seeds RNGs; may filter validation datasets for pair support in missing-channel mode.
- Key callers/callees: caller is `pretrain.sleep2vec_pretrain`; callees include `PSGPretrainDataset`, `build_all_pairs`, `RoundRobinPairSelector`, and `_filter_dataset_for_pair_support`.
- Reuse guidance: use for any standard pretrain runtime path.
- Duplication risk notes: keep missing-channel argument normalization here rather than in the entrypoint.

## `sleep2vec.utils._filter_dataset_for_pair_support`

- File: `sleep2vec/utils.py`
- Signature: `_filter_dataset_for_pair_support(dataset, pair: tuple[str, str], channel_names: list[str]) -> None`
- Purpose and contract: mutate a validation dataset so it contains only samples supporting a scheduled pair; fail fast when none remain.
- Important inputs/outputs: dataset and required pair in; no return value.
- Side effects: mutates `dataset.data`.
- Key callers/callees: caller is `get_pretrain_dataloader`; helper `_resolve_available_channels` is its nearest dependency.
- Reuse guidance: use for validation-only pair support filtering.
- Duplication risk notes: available-channel probing partially overlaps with logic inside `DefaultDataset.dataloader`.

## `sleep2vec.utils._build_finetune_loader` and `get_finetune_dataloaders`

- File: `sleep2vec/utils.py`
- Signatures:
  - `_build_finetune_loader(args, *, split, sources, shuffle, is_train_set, few_shot=None)`
  - `get_finetune_dataloaders(args)`
- Purpose and contract: build finetune train/val/test loaders with correct metadata label wiring and built-in sequence pseudo-channel handling.
- Important inputs/outputs: normalized finetune `args` in; one loader or three loaders out.
- Side effects: seed initialization in `get_finetune_dataloaders`.
- Key callers/callees: callers are `prepare_dataloader` and `_build_inference_loader`; callee is `PSGPretrainDataset.dataloader`.
- Reuse guidance: use these helpers for any finetune or inference data-loading path.
- Duplication risk notes: built-in seq label-channel insertion (`stage5`, `ahi`) and metadata label selection should not be duplicated in trainer code. `ahi` additionally requires scalar summary metadata (`ahi`, `tst`) to survive batch tensorization as regression-style metadata for event-based evaluation, and its preset-validation channel set now auto-expands to include `stage5`; CSV-backed finetune indexes and legacy presets may rely on collate-time NPZ backfill for those scalars.

## `preprocess.save_dataset_presets._resolve_effective_min_channels`

- File: `preprocess/save_dataset_presets.py`
- Signature: `_resolve_effective_min_channels(*, channel_names, cli_min_channels, preset_min_channels) -> int`
- Purpose and contract: compute the effective missing-channel admission threshold for preset generation, while forcing built-in `ahi` presets to require every requested validation channel.
- Important inputs/outputs: requested validation channel names plus CLI/YAML minimums in; final `min_channels` out.
- Side effects: none.
- Key callers/callees: caller is `save_dataset_presets.main`.
- Reuse guidance: use this helper instead of open-coding special-case AHI admission policy in wrapper scripts.
- Duplication risk notes: AHI preset generation must stay strict because downstream preset loading does not revalidate missing `ah_event` / `ahi` / `tst` contracts.
