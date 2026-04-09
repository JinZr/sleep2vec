# Functions: Preprocess And Metrics

## `preprocess.save_dataset_presets._resolve_channels_and_dims`

- File: `preprocess/save_dataset_presets.py`
- Signature: `_resolve_channels_and_dims(config_path: Path, selected_channels: list[str] | None) -> tuple[list[str], dict[str, int]]`
- Purpose and contract: reads YAML `model.channels`, requires `input_dim` on every declared channel, and optionally validates `--channels` as an ordered subset.
- Important inputs: config path, optional selected channel names.
- Important outputs: `(channel_names, channel_input_dims)`.
- Side effects: reads YAML from disk.
- Notable callers/callees: used by `preprocess.save_dataset_presets.main`.
- Reuse guidance: canonical preprocess-side channel resolver.
- Duplication-risk notes: high.

## `preprocess.save_dataset_presets.main`

- File: `preprocess/save_dataset_presets.py`
- Signature: `main() -> None`
- Purpose and contract: parses CLI args, resolves YAML channels, determines dataset name and metadata variants, and generates one preset pickle per `(metadata_variant, split)` combination using `PSGPretrainDataset`.
- Important inputs: CLI flags such as `--index`, `--config`, `--split`, `--n-tokens`, `--allow-missing-channels`.
- Important outputs: preset pickle files.
- Side effects: reads CSVs and YAML, creates output directories, writes pickles, prints summary lines.
- Notable callers/callees: imports `PSGPretrainDataset` lazily when not in dry-run mode.
- Reuse guidance: canonical preset generator.
- Duplication-risk notes: high.

## `preprocess.merge_dataset_presets.main`

- File: `preprocess/merge_dataset_presets.py`
- Signature: `main() -> None`
- Purpose and contract: loads several preset pickle lists, validates each input payload is a list, concatenates them, and writes a merged pickle.
- Important inputs: `--inputs`, `--output`.
- Important outputs: merged preset pickle.
- Side effects: pickle reads and write.
- Notable callers/callees: standalone script.
- Reuse guidance: canonical preset merge helper.
- Duplication-risk notes: low-medium.

## `preprocess.split_index_by_dataset.assign_splits`

- File: `preprocess/split_index_by_dataset.py`
- Signature: `assign_splits(df: pd.DataFrame, group_key: pd.Series, seed: int, shuffle: bool) -> Tuple[pd.Series, Dict[str, Dict[str, int]]]`
- Purpose and contract: splits each dataset group into train/val/test with capped 10% val/test partitions.
- Important inputs: dataframe, grouping series, RNG seed, shuffle flag.
- Important outputs: split labels and per-group counts.
- Side effects: none.
- Notable callers/callees: used by `split_index_by_dataset.main`.
- Reuse guidance: canonical split assignment logic for grouped CSV preparation.
- Duplication-risk notes: medium.

## `preprocess.mask_missing_stats.main`

- File: `preprocess/mask_missing_stats.py`
- Signature: `main() -> None`
- Purpose and contract: streams a CSV in chunks, interprets `*_mask` columns as presence flags, and writes overall/per-dataset missing statistics plus row-wise missing-channel histograms.
- Important inputs: `--csv`, `--dataset-col`, `--chunksize`, `--out-prefix`.
- Important outputs: four CSV reports.
- Side effects: large CSV reads, output CSV writes, stdout summaries.
- Notable callers/callees: standalone script.
- Reuse guidance: canonical mask-statistics tool.
- Duplication-risk notes: low-medium.

## `preprocess.watchpat_zzp_to_edf.main`

- File: `preprocess/watchpat_zzp_to_edf.py`
- Signature: `main(argv: Optional[Sequence[str]] = None) -> int`
- Purpose and contract: `unknown` in full detail from this initialization, but code inspection confirms it is the CLI entrypoint for converting WatchPAT `.zzp` archives into EDF plus optional JSON summaries.
- Important inputs: input path(s), output path(s), writer/backend flags.
- Important outputs: EDF files and optional summaries.
- Side effects: archive reads and EDF writes.
- Notable callers/callees: standalone script; low coupling to the main train/eval stack.
- Reuse guidance: treat as a standalone utility unless you need WatchPAT preprocessing specifically.
- Duplication-risk notes: low.

## `sleep2vec.metrics.compute_downstream_metrics`

- File: `sleep2vec/metrics.py`
- Signature: `compute_downstream_metrics(gts, preds, *, is_classification: bool, output_dim: int | None = None, stage_names=None)`
- Purpose and contract: computes classification or regression metrics, with special handling for binary ROC-AUC and five-stage sleep staging metrics.
- Important inputs: ground truths, predictions, task kind, optional output dimension and stage names.
- Important outputs: metric dictionary.
- Side effects: none.
- Notable callers/callees: used by `Sleep2vecFinetuning._finalize_epoch`.
- Reuse guidance: canonical downstream metric aggregator.
- Duplication-risk notes: medium-high.

## `sleep2vec.metrics.save_result_csv`

- File: `sleep2vec/metrics.py`
- Signature: `save_result_csv(pretrain_result: Mapping[str, float], csv_path: str, args: Any | None = None)`
- Purpose and contract: appends a metrics row to a CSV, carrying selected runtime metadata like checkpoint path, LR, batch size, few-shot count, label name, and channel list.
- Important inputs: metric mapping, destination CSV path, optional argparse namespace.
- Important outputs: none.
- Side effects: CSV read/merge/write, directory creation.
- Notable callers/callees: used by `sleep2vec.finetune.supervised` and `sleep2vec.infer.run_inference`.
- Reuse guidance: canonical result-export helper.
- Duplication-risk notes: medium.

## `sleep2vec.callbacks.pair_acc_logger.PairAccLoggerCallback`

- File: `sleep2vec/callbacks/pair_acc_logger.py`
- Signature: `PairAccLoggerCallback(modality_names, *, log_prefix="val_pair_acc", matrix_key="val_pair_acc_matrix", train_pair_monitor_enabled=True, train_pair_log_prefix="train_pair_sampling", train_pair_skew_warn_threshold=0.05, train_pair_min_unique_coverage_warn_threshold=0.1)`
- Purpose and contract: logs validation pair-accuracy summaries and train-time pair-sampling distribution diagnostics, including optional unique-sample coverage.
- Important inputs: modality names and monitoring thresholds.
- Important outputs: callback state and W&B logging behavior.
- Side effects: W&B image/table logging, Lightning metric logging, warning logging.
- Notable callers/callees: attached in `sleep2vec/pretrain.py` and `sleep2vec/adapt.py`.
- Reuse guidance: canonical observability callback for pair-based training.
- Duplication-risk notes: medium.
