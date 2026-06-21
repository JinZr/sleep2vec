# Runtime Orchestration

## `sleep2vec.pretrain.sleep2vec_pretrain`

- File: `sleep2vec/pretrain.py`
- Signature: `sleep2vec_pretrain(args) -> None`
- Purpose and contract: canonical pretrain entrypoint after argument parsing; binds YAML and data-backend settings into runtime args, builds loaders, configures callbacks, persists run artifacts, and launches `trainer.fit`.
- Important inputs/outputs: CLI namespace in; no direct return value.
- Side effects: creates experiment directories, copies config, writes `cli_args.yaml`, initializes W&B, runs training.
- Key callers/callees: called from `__main__`; calls `load_pretrain_config`, `apply_model_config_args`, `apply_data_backend_args`, `get_pretrain_dataloader`, `Sleep2vecPretraining`, and `persist_run_config_and_args`-adjacent helpers.
- Reuse guidance: reuse this flow for any new pretrain CLI behavior rather than adding separate orchestration scripts.
- Duplication risk notes: pretrain remains the canonical non-adaptation contrastive runtime path.

## `sleep2vec.adapt._resolve_adapt_run_artifacts`

- File: `sleep2vec/adapt.py`
- Signature: `_resolve_adapt_run_artifacts(*, ckpt_path: Path | None, pretrained_backbone_path: Path | None, version_name: str, backbone_arch: str, phase: str, exp_info: str = "") -> AdaptRunArtifacts`
- Purpose and contract: resolve the correct run directory, checkpoint directory, W&B reuse id, and trainer resume checkpoint for staged adaptation.
- Important inputs/outputs: phase, checkpoint, and naming inputs in; `AdaptRunArtifacts` out.
- Side effects: filesystem inspection only.
- Key callers/callees: caller is `sleep2vec_adapt`; callees include `_require_checkpoint_file`, `_resolve_stage1_transition_checkpoint`, `_validate_checkpoint_dir_for_phase`, and `_validate_saved_phase`.
- Reuse guidance: use this helper for any future adaptation resume or stage-transition behavior.
- Duplication risk notes: stage-specific checkpoint layout and strict cross-phase validation belong here, not in callers.

## `sleep2vec.adapt.sleep2vec_adapt`

- File: `sleep2vec/adapt.py`
- Signature: `sleep2vec_adapt(args) -> None`
- Purpose and contract: canonical staged adaptation entrypoint; loads config, applies data-backend settings from YAML, derives initial pair probabilities, builds loaders, resolves run artifacts, persists snapshots, configures callbacks, and launches training.
- Important inputs/outputs: CLI namespace in; no direct return value.
- Side effects: creates or reuses run directories, writes root and phase-scoped snapshots, initializes W&B, runs training.
- Key callers/callees: called from `__main__`; calls `load_pretrain_config`, `apply_model_config_args`, `apply_data_backend_args`, `initial_pair_probs_for_phase`, `get_pretrain_dataloader`, `_resolve_adapt_run_artifacts`, and `Sleep2vecAdaptation`.
- Reuse guidance: extend this routine instead of creating separate adaptation orchestration scripts.
- Duplication risk notes: stage1 vs stage2 semantics, callback selection, and artifact layout are centralized here.

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
- Purpose and contract: canonical finetune orchestration routine; persists run artifacts, builds loaders, instantiates `Sleep2vecFinetuning`, trains, evaluates, and writes results.
- Important inputs/outputs: CLI namespace and config bundle in; no direct return value.
- Side effects: creates run directories, writes YAML snapshots, trains/tests models, copies `best.ckpt`, appends results CSV.
- Key callers/callees: called from `__main__`; calls `prepare_dataloader`, `Sleep2vecFinetuning`, `save_result_csv`, and `persist_run_config_and_args`.
- Reuse guidance: extend this routine instead of creating parallel finetune scripts.
- Duplication risk notes: distributed AHI progress-bar selection is centralized here.

## `sleep2vec.finetune.build_version_name`

- File: `sleep2vec/finetune.py`
- Signature: `build_version_name(args) -> str`
- Purpose and contract: derive a stable experiment name from label, channel selection, few-shot setting, and pretrained-vs-scratch mode when `--version-name` is absent.
- Important inputs/outputs: namespace in, string out.
- Side effects: none.
- Key callers/callees: caller is finetune `__main__`.
- Reuse guidance: use this naming helper instead of re-encoding run naming rules elsewhere.
- Duplication risk notes: if new run types need matching semantics, extend this function rather than cloning it.

## `sleep2vec.infer._build_inference_loader`

- File: `sleep2vec/infer.py`
- Signature: `_build_inference_loader(args) -> DataLoader`
- Purpose and contract: create a single deterministic loader for evaluation-only runs, choosing dataset names from the requested split or overrides.
- Important inputs/outputs: namespace in, dataloader out.
- Side effects: seeds Python, NumPy, and Torch RNGs.
- Key callers/callees: caller is `run_inference`; callee is `_build_finetune_loader`.
- Reuse guidance: use this for inference-only dataloader creation.
- Duplication risk notes: do not duplicate eval-split/source resolution in new inference code.

## `sleep2vec.infer._init_wandb`

- File: `sleep2vec/infer.py`
- Signature: `_init_wandb(args)`
- Purpose and contract: initialize a rank-zero-only W&B run for inference when requested.
- Important inputs/outputs: namespace in, W&B run or `None` out.
- Side effects: external W&B initialization.
- Key callers/callees: caller is `run_inference`; callee is `is_rank_zero_process`.
- Reuse guidance: use this exact gating if new inference artifacts need W&B.
- Duplication risk notes: keep rank-zero gating centralized.

## `sleep2vec.infer._log_inference_outputs_to_wandb`

- File: `sleep2vec/infer.py`
- Signature: `_log_inference_outputs_to_wandb(args, metrics, prediction_row_count)`
- Purpose and contract: log inference metrics plus `prediction_row_count` to W&B, then attach metrics CSV, prediction CSV, manifest JSON, and overview CSV as one inference artifact.
- Important inputs/outputs: prepared inference args, metrics mapping, and row count in; no return value.
- Side effects: external W&B metric and artifact logging.
- Key callers/callees: caller is `run_inference`; callees are `wandb.log`, `wandb.Artifact`, and `wandb.log_artifact`.
- Reuse guidance: extend this helper for inference-only W&B artifact changes instead of putting upload logic in CSV writers.
- Duplication risk notes: artifact contents should remain aligned with `prepare_inference_result_paths` and `save_inference_manifest`.

## `sleep2vec.infer.run_inference`

- File: `sleep2vec/infer.py`
- Signature: `run_inference(args) -> None`
- Purpose and contract: canonical inference driver; normalizes config, applies optional NPZ preset override, builds trainer and loader, optionally averages checkpoints, runs evaluation, writes automatic metrics, prediction CSV, overview CSV, and manifest artifacts under a run-local inference directory, and optionally logs the same outputs to W&B.
- Important inputs/outputs: namespace in; no direct return value.
- Side effects: optional W&B run, trainer evaluation, creation of `results/inference/<namespace>/<label>/<prediction_run_id>/`, metrics CSV writes, prediction CSV writes, shared overview append, `run_manifest.json` write, and optional W&B artifact upload.
- Key callers/callees: called from `__main__`; calls `apply_finetune_config`, `_build_inference_loader`, `select_checkpoints`, `average_checkpoints`, `_init_wandb`, `prepare_inference_result_paths`, `save_result_csv`, `save_prediction_csv`, `save_inference_manifest`, and `_log_inference_outputs_to_wandb`.
- Reuse guidance: extend here for inference-only behavior changes.
- Duplication risk notes: checkpoint averaging policy belongs here plus `checkpoints.py`; inference artifact naming and metadata belongs in `sleep2vec.results`, not trainer code.

## `sleep2vec.extract_embeddings.run_extraction`

- File: `sleep2vec/extract_embeddings.py`
- Signature: `run_extraction(args, *, namespace: str = "sleep2vec") -> Path`
- Purpose and contract: export token-level backbone hidden states from a pretrain or downstream checkpoint for a selected layer, trimming CLS/padding and writing manifest-style NPZ or Kaldi outputs.
- Important inputs/outputs: config path, checkpoint path, output directory/format, layer index, split, data backend, optional NPZ index/preset or Kaldi manifest in; output `manifest.json` path out.
- Side effects: loads checkpoint weights strictly, builds a deterministic extraction dataloader, writes per-split manifest CSV, per-channel embedding files or ark/scp files, and root `manifest.json`.
- Key callers/callees: called from `__main__`; calls `load_pretrain_config` or `load_finetune_config`, `apply_data_backend_args`, `PSGPretrainDataset` or `KaldiPSGDataset`, optional downstream adapter insertion when finetune LoRA is enabled, and `Sleep2vecPretrainModel._token_embeddings_to_hidden`.
- Reuse guidance: use this entrypoint for persistent token embedding exports rather than routing through downstream prediction code.
- Duplication risk notes: package-local mirrors under `sleep2vec2` and `sleep2expert` must keep imports inside their namespaces; `sleep2expert` passes modality names into the MoE-capable backbone but does not collect MoE aux for embedding export.

## `sleep2vec.sleep2vec_inference.extract_prediction_records`

- File: `sleep2vec/sleep2vec_inference.py`
- Signature: `extract_prediction_records(args, batch, logits, targets) -> list[dict[str, object]]`
- Purpose and contract: convert one test/inference batch into intermediate prediction records for scalar classification, sequence classification, scalar regression, sequence regression, and multilabel outputs.
- Important inputs/outputs: runtime args, batch metadata with `path` and optional `token_start`, logits, and targets in; record dictionaries with path, token start, labels, predictions, probabilities or scores out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecFinetuning`; callees are `_extract_multilabel_prediction_records`, `_extract_classification_prediction_records`, and `_extract_regression_prediction_records`.
- Reuse guidance: use this hook when adding non-AHI prediction export behavior rather than saving logits directly from trainer steps.
- Duplication risk notes: task-shape-specific masking of `-1` labels belongs here.

Survival prediction rows are built directly by `Sleep2vecFinetuning._build_survival_prediction_rows` because they use survival sidecar vectors, survival keys, disease names, and raw log-risk lists rather than ordinary `label_name` targets.

## `sleep2vec.sleep2vec_inference.build_prediction_rows`

- File: `sleep2vec/sleep2vec_inference.py`
- Signature: `build_prediction_rows(records: list[dict[str, object]]) -> list[dict[str, object]]`
- Purpose and contract: group intermediate prediction records into one row per path, deduplicating repeated `(path, token_start)` windows before building task-family-specific row fields.
- Important inputs/outputs: per-window records in; per-path CSV-ready row dictionaries out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecFinetuning._finalize_epoch`; callees include `_group_prediction_records`, `_build_classification_prediction_row`, `_build_regression_prediction_row`, and `_build_multilabel_prediction_row`.
- Reuse guidance: use for non-AHI path-level prediction CSV rows.
- Duplication risk notes: DDP duplicate-window defense is implemented here; do not recreate grouping inside CSV writers.

## `sleep2vec.sleep2vec_inference.build_ahi_prediction_rows`

- File: `sleep2vec/sleep2vec_inference.py`
- Signature: `build_ahi_prediction_rows(records: list[dict[str, np.ndarray]], threshold: float) -> list[dict[str, object]]`
- Purpose and contract: merge AHI event-window records by path and emit path-level event predictions, probability scores, threshold, true AHI, predicted AHI, TST hours, and token-start provenance.
- Important inputs/outputs: gathered AHI records plus validation-fitted threshold in; per-path CSV-ready row dictionaries out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecFinetuning._finalize_epoch`; callees include `_merge_ahi_window_records` and `_evaluate_single_ahi_record` from `sleep2vec.metrics`.
- Reuse guidance: use this for AHI prediction CSV output; keep AHI threshold semantics tied to validation-fitted checkpoint state.
- Duplication risk notes: AHI prediction rows must stay aligned with event-metric merge semantics.

## `sleep2vec.checkpoints.load_pretrain_init_weights`

- File: `sleep2vec/checkpoints.py`
- Signature: `load_pretrain_init_weights(model: torch.nn.Module, path: Path | str, *, device: torch.device | str = "cpu", strict: bool = False) -> PretrainInitLoadResult`
- Purpose and contract: load checkpoint weights into a model, preferring averaged-model prefixes when present and returning explicit load diagnostics.
- Important inputs/outputs: model and checkpoint path in, `PretrainInitLoadResult` out.
- Side effects: loads checkpoint files from disk and mutates model weights.
- Key callers/callees: callers are `Sleep2vecAdaptation` and adaptation-related tests; callees include `load_checkpoint`, `extract_pretrain_init_state_dict`, and `get_state_dict_from_checkpoint`.
- Reuse guidance: use for weight initialization from pretrain-style checkpoints instead of open-coding prefix stripping.
- Duplication risk notes: `model.` vs `ema_model.` handling belongs here.

## `sleep2expert.checkpoints.load_pretrain_init_weights`

- File: `sleep2expert/checkpoints.py`
- Signature: `load_pretrain_init_weights(module, ckpt_path, *, device=cpu, strict=False, prefixes=("ema_model.", "model.")) -> PretrainInitLoadResult`
- Purpose and contract: package-local checkpoint initializer for the sleep2expert standalone backbone, including rejection of legacy HF/RoFormer key layouts and compatible dense-FFN to MoE expert expansion.
- Important inputs/outputs: module and checkpoint path in; explicit load result out.
- Side effects: loads checkpoint files and mutates module weights.
- Key callers/callees: callers are sleep2expert adaptation/finetune/pretrain initialization paths; callees include `extract_pretrain_init_state_dict` and `initialize_moe_from_dense_if_possible`.
- Reuse guidance: use this loader for sleep2expert pretrain-to-MoE initialization.
- Duplication risk notes: dense-to-MoE expansion and incomplete MoE checkpoint rejection should not be reimplemented in experiment scripts.

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

## `sleep2vec.results.save_result_csv`

- File: `sleep2vec/results.py`
- Signature: `save_result_csv(pretrain_result: Mapping[str, float], csv_path: str, args: Any | None = None) -> None`
- Purpose and contract: append a metrics row plus selected run metadata to a CSV file, using rank-zero gating and a lockfile to tolerate concurrent local writers.
- Important inputs/outputs: metrics mapping and destination path in; no return value.
- Side effects: creates directories, locks a sibling file, and writes or appends to CSV.
- Key callers/callees: callers are `finetune.supervised` and `infer.run_inference`; callees include `_resolve_result_source`, `_resolve_experiment_version`, `_ordered_result_columns`, `_result_csv_lock`, and `is_rank_zero_process`.
- Reuse guidance: reuse for any tabular experiment summary output.
- Duplication risk notes: current column ordering and schema-expansion policy is encoded here; do not create parallel result-writer variants casually.

## `sleep2vec.results.save_training_run_manifest`

- File: `sleep2vec/results.py`
- Signature: `save_training_run_manifest(args, *, manifest_path, status, monitor=None, monitor_mode=None, best_model_path=None, best_model_score=None, last_checkpoint_path=None, results_csv_path=None, survival_per_disease_metrics_csv_path=None, metrics=None) -> None`
- Purpose and contract: write a lightweight finetune training `run_manifest.json` with status, config/CLI paths, label, monitor, checkpoint identity, test-after-fit policy, result paths, metrics, and git metadata.
- Important inputs/outputs: normalized finetune args plus manifest fields in; JSON manifest out.
- Side effects: rank-zero-gated atomic JSON write.
- Key callers/callees: caller is `sleep2vec.finetune.supervised`; callees include `_write_json_atomic`, `_json_safe`, and `_git_manifest`.
- Reuse guidance: use this writer for future finetune run-manifest updates instead of scattering JSON writes in entrypoints.
- Duplication risk notes: keep aggregate metrics CSV and run manifest separate.

## `sleep2vec.results.prepare_inference_result_paths`

- File: `sleep2vec/results.py`
- Signature: `prepare_inference_result_paths(args: Any, *, namespace: str = PACKAGE_NAMESPACE, root: str | Path = DEFAULT_INFERENCE_RESULTS_ROOT, checkpoint_paths: Sequence[str | Path] | None = None, timestamp: str | None = None) -> None`
- Purpose and contract: attach inference output paths and checkpoint metadata to the runtime args namespace before writing metrics, predictions, survival per-disease metrics, overview rows, or manifest files.
- Important inputs/outputs: args plus namespace/root/checkpoint inputs in; mutates `args.prediction_run_id`, `run_dir`, `inference_*_csv_path`, `inference_survival_per_disease_metrics_csv_path`, `manifest_path`, checkpoint tags, task family, and timestamp fields.
- Side effects: namespace mutation only.
- Key callers/callees: caller is `infer.run_inference`; callees include `make_prediction_run_id` and `_resolve_checkpoint_info`.
- Reuse guidance: call this before every automatic inference export path across root and package-local variants.
- Duplication risk notes: output layout and `prediction_run_id` construction should not be rebuilt in entrypoints.

## `sleep2vec.results.make_prediction_run_id`

- File: `sleep2vec/results.py`
- Signature: `make_prediction_run_id(args: Any, *, timestamp: str | None = None, namespace: str = PACKAGE_NAMESPACE, ckpt_info: Mapping[str, Any] | None = None) -> str`
- Purpose and contract: generate a unique, slug-safe inference run id from timestamp, namespace, experiment version, label, split, checkpoint tag, and a short hash over run inputs plus a launch nonce.
- Important inputs/outputs: args, optional timestamp/namespace/checkpoint info in; prediction run id string out.
- Side effects: none.
- Key callers/callees: caller is `prepare_inference_result_paths`; callees include `_resolve_checkpoint_info`, `_resolve_experiment_version`, `_stringify_optional_path`, and `_slug_piece`.
- Reuse guidance: use this through `prepare_inference_result_paths` for normal inference runs; call directly only in tests or tooling that needs the same id contract without path mutation.
- Duplication risk notes: run id fields are part of metrics/prediction/manifest linkage, so do not create namespace-specific alternatives.

## `sleep2vec.results.save_prediction_csv`

- File: `sleep2vec/results.py`
- Signature: `save_prediction_csv(rows: Sequence[Mapping[str, Any]], csv_path: str, args: Any | None = None) -> None`
- Purpose and contract: write or append path-level prediction rows while preserving inference metadata, deterministic metadata column ordering, JSON serialization for list/dict values, rank-zero gating, and lockfile protection.
- Important inputs/outputs: CSV-ready prediction rows, destination path, and optional args in; CSV file out. Empty rows still create a header-only file with prediction metadata columns.
- Side effects: creates directories, locks a sibling file, and writes or appends to CSV.
- Key callers/callees: caller is `infer.run_inference`; callees include `_add_inference_run_metadata`, `_serialize_prediction_value`, `_ordered_prediction_columns`, `_result_csv_lock`, and `_write_result_csv`.
- Reuse guidance: use this writer for all per-path prediction exports.
- Duplication risk notes: it intentionally mirrors result CSV metadata via `prediction_run_id`; do not add separate prediction metadata schemas.

## `sleep2vec.results.save_inference_manifest`

- File: `sleep2vec/results.py`
- Signature: `save_inference_manifest(args: Any, metrics: Mapping[str, Any] | None = None, *, prediction_row_count: int = 0) -> None`
- Purpose and contract: write `run_manifest.json` for an inference run with paths, checkpoint identity, runtime settings, metrics, and prediction-row count.
- Important inputs/outputs: args namespace, metrics mapping, and row count in; JSON file out.
- Side effects: rank-zero-gated atomic JSON write.
- Key callers/callees: caller is `infer.run_inference`; callees include `_json_safe` and `_write_json_atomic`.
- Reuse guidance: use with `prepare_inference_result_paths` so manifests stay aligned with metrics and prediction CSV paths.

## `sleep2vec.results.save_survival_per_disease_metrics_csv`

- File: `sleep2vec/results.py`
- Signature: `save_survival_per_disease_metrics_csv(rows: Sequence[Mapping[str, Any]], csv_path: str, args: Any | None = None) -> None`
- Purpose and contract: write or append survival per-disease metric rows with inference/finetune metadata, rank-zero gating, schema expansion, and lockfile protection.
- Important inputs/outputs: rows with `stage`, `disease_idx`, `disease`, `n_labeled`, `n_events`, and `c_index`; CSV file out when rows are non-empty.
- Key callers/callees: callers are `infer.run_inference` and `finetune.supervised`; callees include `_add_inference_run_metadata`, `_ordered_survival_per_disease_columns`, `_result_csv_lock`, and `_write_result_csv`.
- Duplication risk notes: manifest schema should stay here rather than in namespace-specific inference entrypoints.

## `sleep2expert.routing_analysis.run_routing_analysis`

- File: `sleep2expert/routing_analysis.py`
- Signature: `run_routing_analysis(args: argparse.Namespace) -> list[dict[str, Any]]`
- Purpose and contract: export MoE routing summaries from a sleep2expert finetune or pretrained-only model to CSV, optionally rendering routing heatmap PNGs and logging them to W&B.
- Important inputs/outputs: config, checkpoint/pretrained options, eval split, output CSV path, optional heatmap directory, and device controls in; routing row dictionaries and written artifacts out.
- Side effects: loads model weights, runs evaluation forwards, writes CSV/PNG files, and may initialize W&B.
- Key callers/callees: caller is `main`; callees include `apply_finetune_config`, `_build_inference_loader`, `Sleep2vecFinetuning`, `_load_analysis_weights`, `build_routing_rows`, `_write_rows`, and `write_routing_heatmaps`.
- Reuse guidance: use this CLI for persistent routing inspection instead of reading `last_moe_aux` manually.
- Duplication risk notes: `last_moe_aux` is transient runtime state; this module is the explicit export path.

## `sleep2vec.metrics.binary_specificity` and `macro_specificity`

- File: `sleep2vec/metrics.py`
- Signatures:
  - `binary_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float`
  - `macro_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float`
- Purpose and contract: compute true-negative-rate specificity for binary tasks and macro one-vs-rest specificity for multiclass tasks.
- Important inputs/outputs: integer labels/predictions in; scalar float out, returning `0.0` when a denominator is empty.
- Side effects: none.
- Key callers/callees: callers are `compute_binary_label_metrics` and `compute_downstream_metrics`.
- Reuse guidance: use these reducers when adding classification metrics instead of recalculating specificity in trainer code.
- Duplication risk notes: binary specificity is class-1-vs-class-0, while stage aliases continue to report macro `spec`.

## `sleep2vec.metrics.compute_downstream_metrics`

- File: `sleep2vec/metrics.py`
- Signature: `compute_downstream_metrics(gts, preds, *, is_classification: bool, is_multilabel: bool = False, output_dim: int | None = None, stage_names=None)`
- Purpose and contract: reduce per-sample predictions into classification, regression, or multilabel metrics, including recall/specificity for classification and stage-aware reporting for remapped staging tasks.
- Important inputs/outputs: ground truth and predictions in, metrics dict out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecFinetuning._finalize_epoch`; callees include `compute_binary_label_metrics`, `compute_ahi_pointwise_metrics`, and `roc_auc_from_two_logits`.
- Reuse guidance: use this as the generic downstream metric reducer.
- Duplication risk notes: do not implement alternative epoch metric logic in trainers unless the contract truly changes.

## `sleep2vec.metrics.compute_survival_c_index`

- File: `sleep2vec/metrics.py`
- Signature: `compute_survival_c_index(pred, event_time, is_event, has_label) -> float`
- Purpose and contract: compute the mean finite disease-wise concordance index for Cox survival predictions from `compute_survival_c_index_by_disease` rows.
- Important inputs/outputs: `[N, L]` raw log-risk predictions plus same-shaped event time, event indicator, and label mask arrays/tensors in; scalar float out or `nan` when no disease is computable.
- Side effects: imports `sksurv.metrics.concordance_index_censored`.
- Key callers/callees: caller is compatibility code; survival epoch finalization calls `compute_survival_c_index_by_disease` directly.
- Reuse guidance: use this metric reducer for survival validation/test monitoring instead of invoking sksurv directly from trainers.
- Duplication risk notes: shape checks and disease-skip policy belong here.

## `sleep2vec.metrics.compute_survival_c_index_by_disease`

- File: `sleep2vec/metrics.py`
- Signature: `compute_survival_c_index_by_disease(pred, event_time, is_event, has_label, disease_names=None) -> list[dict[str, Any]]`
- Purpose and contract: compute one row per survival disease with disease index/name, labeled count, event count, and finite or `nan` c-index.
- Important inputs/outputs: `[N, L]` raw log-risk predictions plus same-shaped sidecar arrays/tensors and optional `L` disease names in; metric row list out.

## `utils.check_configs.check_config_file`

- File: `utils/check_configs.py`
- Signature: `check_config_file(path: Path) -> None`
- Purpose and contract: validate one config file against runtime loader compatibility, tokenizer-parity checks, repo `ppg_*finetune*` policy, and `preset_build` strictness.
- Important inputs/outputs: config path in; raises on failure, returns `None` on success.
- Side effects: reads YAML from disk only.
- Key callers/callees: caller is `utils.check_configs.main`; callees include `_validate_runtime_loader_contract`, `_validate_repo_policy`, and `_validate_preset_build_contract`.
- Reuse guidance: use this tooling path for repo-wide config validation instead of ad hoc shell loops.
- Duplication risk notes: static config policy belongs here, not in entrypoints.
