# Runtime Orchestration

## `sleep2vec.pretrain.sleep2vec_pretrain`

- File: `sleep2vec/pretrain.py`
- Signature: `sleep2vec_pretrain(args) -> None`
- Purpose and contract: canonical pretrain entrypoint after argument parsing; binds YAML into runtime args, builds loaders, configures callbacks, persists run artifacts, and launches `trainer.fit`.
- Important inputs/outputs: CLI namespace in; no direct return value.
- Side effects: creates experiment directories, copies config, writes `cli_args.yaml`, initializes W&B, runs training.
- Key callers/callees: called from `__main__`; calls `load_pretrain_config`, `get_pretrain_dataloader`, `Sleep2vecPretraining`, and `dump_cli_args_yaml`.
- Reuse guidance: reuse this flow for any new pretrain CLI behavior rather than adding separate orchestration scripts.
- Duplication risk notes: artifact-writing and trainer construction overlap conceptually with `finetune.py`, but this remains the canonical pretrain path.

## `sleep2vec.finetune.prepare_dataloader`

- File: `sleep2vec/finetune.py`
- Signature: `prepare_dataloader(args) -> tuple[DataLoader, DataLoader, DataLoader]`
- Purpose and contract: build train/val/test finetune loaders and log their batch counts.
- Important inputs/outputs: normalized finetune `args` in; three dataloaders out.
- Side effects: logging only.
- Key callers/callees: caller is `supervised`; callee is `get_finetune_dataloaders`.
- Reuse guidance: use this wrapper if future finetune flows need the standard three-loader contract.
- Duplication risk notes: loader-building semantics live in `sleep2vec/utils.py`, not here.

## `sleep2vec.finetune.supervised`

- File: `sleep2vec/finetune.py`
- Signature: `supervised(args, config_bundle) -> None`
- Purpose and contract: canonical finetune orchestration routine; persists run artifacts, builds loaders, instantiates `Sleep2vecFinetuning`, trains, evaluates, and writes results. When running distributed built-in `ahi` finetune, it installs the reusable custom progress-bar callback from `sleep2vec.callbacks.progress_bar`.
- Important inputs/outputs: CLI namespace and config bundle in; no direct return value.
- Side effects: creates run directories, writes YAML snapshots, trains/tests models, copies `best.ckpt`, appends results CSV.
- Key callers/callees: called from `__main__`; calls `prepare_dataloader`, `Sleep2vecFinetuning`, `persist_run_config_and_args`, `build_distributed_ahi_progress_bar`, and `save_result_csv`.
- Reuse guidance: extend this routine instead of creating parallel finetune scripts.
- Duplication risk notes: keep callback implementations out of this entrypoint; configuration-copy behavior overlaps with `pretrain.py`.

## `sleep2vec.callbacks.progress_bar.build_distributed_ahi_progress_bar`

- File: `sleep2vec/callbacks/progress_bar.py`
- Signature: `build_distributed_ahi_progress_bar() -> RichProgressBar | TQDMProgressBar`
- Purpose and contract: construct the dedicated distributed-AHI progress bar callback, preserving batch-level updates while skipping the default rank-zero-only train-epoch-end UI refresh/postfix work that would otherwise delay later epoch-end hooks under DDP.
- Important inputs/outputs: no inputs; returns the appropriate Lightning progress-bar callback for the current optional-rich availability.
- Side effects: none.
- Key callers/callees: caller is `sleep2vec.finetune.supervised`; instantiates `DistributedAHIRichProgressBar` or `DistributedAHITQDMProgressBar`.
- Reuse guidance: use this factory for any future distributed built-in AHI finetune path instead of recreating callback subclasses in an entrypoint.
- Duplication risk notes: post-hoc barrier callbacks are not a substitute for this callback because the skew originates inside Lightning's default progress-bar epoch-end hook.

## `sleep2vec.finetune.build_version_name`

- File: `sleep2vec/finetune.py`
- Signature: `build_version_name(args) -> str`
- Purpose and contract: derive a stable experiment name from label, channels, few-shot setting, and pretrained/scratch mode when `--version-name` is absent.
- Important inputs/outputs: namespace in, string out.
- Side effects: none.
- Key callers/callees: caller is finetune `__main__`.
- Reuse guidance: use this naming helper instead of re-encoding run naming rules elsewhere.
- Duplication risk notes: naming is entrypoint-specific; if new run types need matching semantics, extend this function rather than cloning it.

## `sleep2vec.infer._build_inference_loader`

- File: `sleep2vec/infer.py`
- Signature: `_build_inference_loader(args) -> DataLoader`
- Purpose and contract: create a single deterministic loader for evaluation-only runs, choosing dataset names from the requested split or overrides.
- Important inputs/outputs: namespace in, dataloader out.
- Side effects: seeds Python, NumPy, and Torch RNGs.
- Key callers/callees: caller is `run_inference`; callee is `_build_finetune_loader`.
- Reuse guidance: use this for inference-only dataloader creation.
- Duplication risk notes: do not duplicate eval-split/source resolution in new inference code.

## `sleep2vec.infer.run_inference`

- File: `sleep2vec/infer.py`
- Signature: `run_inference(args) -> None`
- Purpose and contract: canonical inference driver; normalizes config, builds trainer and loader, optionally averages checkpoints, runs evaluation, and writes metrics.
- Important inputs/outputs: namespace in; no direct return value.
- Side effects: optional W&B run, trainer evaluation, optional results CSV output.
- Key callers/callees: called from `__main__`; calls `apply_finetune_config`, `_build_inference_loader`, `select_checkpoints`, `average_checkpoints`, and `Sleep2vecFinetuning`.
- Reuse guidance: extend here for inference-only behavior changes.
- Duplication risk notes: checkpoint averaging policy belongs here plus `checkpoints.py`, not in trainer code; built-in `ahi` fine-threshold search for standalone inference is injected here while the fallback-to-saved-threshold path remains inside the trainer-side AHI evaluation contract.

## `sleep2vec.checkpoints.select_checkpoints`

- File: `sleep2vec/checkpoints.py`
- Signature: `select_checkpoints(ckpt_dir: Path, *, end_ckpt: Path | None, num_ckpts: int) -> list[Path]`
- Purpose and contract: choose candidate checkpoints for averaging, preferring epoch ordering and falling back to modification time.
- Important inputs/outputs: checkpoint directory and selection bounds in, ordered file list out.
- Side effects: reads filesystem metadata only.
- Key callers/callees: caller is `infer.run_inference`; callee is `_parse_epoch`.
- Reuse guidance: use for every N-checkpoint averaging flow.
- Duplication risk notes: epoch-first then mtime fallback is a tested contract; do not silently replace it.

## `sleep2vec.checkpoints.average_checkpoints`

- File: `sleep2vec/checkpoints.py`
- Signature: `average_checkpoints(filenames: Sequence[Path], *, device: torch.device | str = cpu) -> dict[str, torch.Tensor]`
- Purpose and contract: load multiple checkpoint state dicts and produce one averaged tensor map.
- Important inputs/outputs: list of checkpoint paths in, averaged state dict out.
- Side effects: loads checkpoint files from disk.
- Key callers/callees: caller is `infer.run_inference`; callee is `_load_state_dict`.
- Reuse guidance: reuse for checkpoint averaging rather than open-coding tensor accumulation.
- Duplication risk notes: supports raw dicts, `state_dict`, and `model` wrappers already.

## `Sleep2vecFinetuning.on_save_checkpoint`, `on_load_checkpoint`, and `on_test_start`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signatures:
  - `on_save_checkpoint(self, checkpoint) -> None`
  - `on_load_checkpoint(self, checkpoint) -> None`
  - `on_test_start(self) -> None`
- Purpose and contract: persist model/finetune config snapshots into checkpoints, save the validation-fitted `ahi_eval_threshold`, reload that threshold on restore, and fail fast before AHI test-time reuse when neither a stored threshold nor an explicit test-search grid is available.
- Important inputs/outputs: checkpoint mapping in/out; no return value.
- Side effects: mutates checkpoint payload and may raise `ValueError`.
- Key callers/callees: called by Lightning; `on_save_checkpoint` also snapshots layer-mix weights when available.
- Reuse guidance: keep `ahi` threshold persistence here instead of patching checkpoint state in entrypoints.
- Duplication risk notes: do not invent a second storage location for the fitted `ahi` threshold.

## `Sleep2vecFinetuning._extract_ahi_event_records`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signature: `_extract_ahi_event_records(self, batch, logits) -> list[dict[str, np.ndarray]]`
- Purpose and contract: convert one batch of raw `ahi` token labels, sigmoid scores, scalar AHI/TST metadata, and the built-in auxiliary `stage5` token stream into per-sample records for final event-based AHI reduction. The record keeps token-level `stage5` plus a per-second `second_valid_mask` so partially masked tokens stay aligned with second-level `truth`/`score`.
- Important inputs/outputs: batch plus logits in; list of `{truth, score, true_ahi, tst_hours, stage5, second_valid_mask}` records out.
- Side effects: none.
- Key callers/callees: caller is `_shared_step`; downstream consumer is `compute_ahi_event_metrics`.
- Reuse guidance: use this helper when the final AHI reduction contract changes.
- Duplication risk notes: sample-boundary preservation and scalar-summary alignment belong here, not in metrics helpers.

## `Sleep2vecFinetuning._finalize_epoch`

- File: `sleep2vec/sleep2vec_finetuning.py`
- Signature: `_finalize_epoch(self, stage: str)`
- Purpose and contract: reduce cached epoch outputs into train/val/test metrics. Validation/test loss is accumulated locally during step execution, reduced once across ranks at epoch end, and then logged with `sync_dist=False` so callback-visible monitor values stay identical on every rank without relying on Lightning's step-level eval-loss sync. The dedicated `ahi` path now keeps train-time pointwise metrics as reduced confusion-count totals (`tp/fp/tn/fn`) instead of concatenating every token-level prediction across the epoch; train accuracy/precision/recall/F1 are computed once from globally reduced counts, logged as normal train metrics on every rank, and train ROC-AUC is intentionally skipped. Full event-eval validation still runs only when `args.monitor == "val_ahi_pearson"` and `args.monitor_mod == "max"`, otherwise lightweight validation logs the manually reduced `val_loss` plus emitted pointwise AHI metrics from gathered arrays, and the existing test-stage event-eval path is reused once an explicit or saved threshold/search grid is available. When rank zero emits the summary scatter plot, all ranks rejoin through a strategy barrier before exiting the validation/test epoch so later train-epoch collectives do not get ahead of the rank-zero-only visualization work.
- Important inputs/outputs: stage name plus cached outputs in; logs metrics and returns reduced arrays/records when present.
- Side effects: emits Lightning metrics, clears epoch caches, and may update `self._ahi_eval_threshold`.
- Key callers/callees: callers are `on_train_epoch_end`, `on_validation_epoch_end`, and `on_test_epoch_end`; callees include `compute_ahi_pointwise_metrics`, `compute_ahi_event_metrics`, `compute_downstream_metrics`, and `_eval_visualizer.log`.
- Reuse guidance: keep epoch-level metric branching here rather than scattering task-specific logic across callbacks or entrypoints.
- Duplication risk notes: `ahi` final evaluation intentionally bypasses `compute_downstream_metrics` and the generic classification visualizer; only the scalar-summary scatter plot is reused from the shared visualization surface, while lightweight validation should keep reusing `compute_ahi_pointwise_metrics` instead of inventing a second token-level reducer. Train-time AHI pointwise metrics intentionally do not reuse the array-based reducer because epoch-wide concatenation and ROC-AUC sorting are too expensive at full token scale; keep the reduced confusion-count path there and do not reintroduce train-time full-array accumulation. Do not reintroduce step-level `sync_dist=True` eval-loss logging alongside this epoch-end reduction path.

## `sleep2vec.metrics.compute_ahi_pointwise_metrics`

- File: `sleep2vec/metrics.py`
- Signature: `compute_ahi_pointwise_metrics(gts, preds) -> dict[str, float]`
- Purpose and contract: wrap binary token-level metrics under `ahi_pointwise_*` names for lightweight validation logging (and any small-array callers that already materialized all predictions).
- Important inputs/outputs: flattened binary labels and sigmoid scores in; namespaced metrics dict out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecFinetuning._finalize_epoch`; callee is `compute_binary_label_metrics`.
- Reuse guidance: use this for lightweight AHI validation/reporting whenever callers already materialized token-level arrays, instead of logging generic binary keys directly.
- Duplication risk notes: final AHI validation/test metrics do not belong here.

## `sleep2vec.metrics.select_best_ahi_threshold`

- File: `sleep2vec/metrics.py`
- Signature: `select_best_ahi_threshold(records, *, search_thresholds=...) -> tuple[float, dict[str, Any]]`
- Purpose and contract: search the configured threshold grid, skipping records without usable scalar `tst_hours`, and choose the threshold that maximizes Pearson, then minimizes MAE, then prefers the higher threshold on exact metric ties. Current runtime callers use a coarse grid during finetune validation and a fine grid during standalone inference, while reusing a prepared-record cache and emitting progress logs during the threshold loop.
- Important inputs/outputs: per-sample `{truth, score, true_ahi, tst_hours}` records in; selected threshold plus cached aggregate out.
- Side effects: none.
- Key callers/callees: caller is `compute_ahi_event_metrics`; callee is `_aggregate_ahi_records`.
- Reuse guidance: reuse this if threshold search policy changes.
- Duplication risk notes: coarse validation search and fine inference search must stay centralized here rather than diverging in trainer or entrypoint code.

## `sleep2vec.metrics.compute_ahi_event_metrics`

- File: `sleep2vec/metrics.py`
- Signature: `compute_ahi_event_metrics(records, *, threshold: float | None = None, search_thresholds=..., severity_thresholds=...) -> tuple[dict[str, float], float]`
- Purpose and contract: convert per-sample 1-second `ahi` predictions into two related outputs. Detection metrics still deduplicate gathered windows by keeping the first record for each `(path, token_start)` before enforcing per-recording contiguity, merge adjacent runs, keep events whose inclusive duration is at least 10 seconds, mask wake-only predicted events using the required `stage5` stream (optionally filtered by per-window `second_valid_mask`), and match events with inclusive IoU arithmetic. Summary AHI metrics (`ahi_mae`, `ahi_pearson`, ICC, and severity summaries) instead count stage-filtered raw predicted positive runs without merge or min-duration filtering so the scalar prediction aligns with NPZ ground-truth `true_ahi`. Threshold search still optimizes those scalar summary metrics against `tst_hours`, and `TST < 2h` exclusion remains a runtime guardrail for final AHI summaries rather than part of the detection definition itself.
- Important inputs/outputs: per-sample `truth` / `score` / `true_ahi` / `tst_hours` records in; metrics dict plus chosen threshold out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecFinetuning._finalize_epoch`; callees include `select_best_ahi_threshold`, `binary_sequence_to_segments`, `merge_intervals`, `filter_segments_by_duration`, and `vectorized_event_stats`.
- Reuse guidance: use this for every final `ahi` validation/test/infer metric path instead of re-deriving event counts in trainer code; callers should change only the supplied threshold grid, not the aggregation logic or the prepared-record reuse path.
- Duplication risk notes: threshold search, scalar-summary semantics, gathered-window deduplication, stage-aware prediction masking, inclusive event-overlap arithmetic, and the split between detection and scalar-summary post-processing must stay centralized here.

## `sleep2vec.metrics.compute_downstream_metrics`

- File: `sleep2vec/metrics.py`
- Signature: `compute_downstream_metrics(gts, preds, *, is_classification: bool, is_multilabel: bool = False, output_dim: int | None = None, stage_names=None)`
- Purpose and contract: reduce non-AHI downstream predictions into multiclass classification or regression metrics, with stage-specific metrics for sleep-staging outputs.
- Important inputs/outputs: ground truth and predictions in, metrics dict out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecFinetuning._finalize_epoch` for non-AHI paths; callees include `multiclass_metrics_fn`, `roc_auc_from_two_logits`, and `macro_specificity`.
- Reuse guidance: use this as the only generic downstream metric reducer.
- Duplication risk notes: `ahi` val/test/infer bypass this function by design.

## `sleep2vec.metrics.save_result_csv`

- File: `sleep2vec/metrics.py`
- Signature: `save_result_csv(pretrain_result: Mapping[str, float], csv_path: str, args: Any | None = None) -> None`
- Purpose and contract: append a metrics row plus selected run metadata to a CSV file, creating the file and parent directory as needed.
- Important inputs/outputs: metrics mapping and destination path in; no return value.
- Side effects: creates directories and writes or appends to CSV.
- Key callers/callees: callers are `finetune.supervised` and `infer.run_inference`.
- Reuse guidance: reuse for any tabular experiment summary output.
- Duplication risk notes: current column policy is encoded here; do not create parallel result-writer variants casually.
