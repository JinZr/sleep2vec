# Runtime Orchestration

## `sleep2vec.pretrain.sleep2vec_pretrain`

- File: `sleep2vec/pretrain.py`
- Signature: `sleep2vec_pretrain(args) -> None`
- Purpose and contract: canonical pretrain entrypoint after argument parsing; binds YAML into runtime args, resolves NPZ/Kaldi backend settings, builds loaders, configures callbacks, persists run artifacts, and launches `trainer.fit`.
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
- Purpose and contract: canonical staged adaptation entrypoint; loads config, resolves NPZ/Kaldi backend settings, derives initial pair probabilities, builds loaders, resolves run artifacts, persists snapshots, configures callbacks, and launches training.
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
- Purpose and contract: canonical finetune orchestration routine; persists run artifacts, builds loaders, instantiates `Sleep2vecFinetuning`, trains, evaluates, and writes results. For `sleep2expert`, downstream MoE tuning remains inside the Lightning module, not in this orchestration routine.
- Important inputs/outputs: CLI namespace and config bundle in; no direct return value.
- Side effects: creates run directories, writes YAML snapshots and `moe_finetune_status.json`, logs MoE fine-tune status to W&B, trains/tests models, copies `best.ckpt`, appends results CSV.
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

## `sleep2vec.infer.run_inference`

- File: `sleep2vec/infer.py`
- Signature: `run_inference(args) -> None`
- Purpose and contract: canonical inference driver; normalizes config, rejects legacy `--inference-preset-path` overrides for Kaldi backends, builds trainer and loader, optionally averages checkpoints, runs evaluation, and writes metrics.
- Important inputs/outputs: namespace in; no direct return value.
- Side effects: optional W&B run, trainer evaluation, optional results CSV output.
- Key callers/callees: called from `__main__`; calls `apply_finetune_config`, `_build_inference_loader`, `select_checkpoints`, `average_checkpoints`, `_init_wandb`, and `save_result_csv`.
- Reuse guidance: extend here for inference-only behavior changes.
- Duplication risk notes: checkpoint averaging policy belongs here plus `checkpoints.py`, not in trainer code.

## `sleep2expert.routing_analysis.run_routing_analysis`

- File: `sleep2expert/routing_analysis.py`
- Signature: `run_routing_analysis(args) -> list[dict[str, Any]]`
- Purpose and contract: export sparse-MoE routing summaries from downstream evaluation batches to a stable CSV schema, with optional derived per-modality heatmap PNGs and W&B image logging.
- Important inputs/outputs: normalized CLI namespace with finetune config, concrete checkpoint path or `--pretrained-only` backbone path, label name, eval split, optional analysis tag, output path, optional heatmap output directory, and optional W&B settings in; aggregated routing rows out and CSV written to `args.output`.
- Side effects: loads checkpoint weights unless `--pretrained-only` is set, iterates the inference loader, runs `Sleep2vecFinetuning._get_eval_model()(batch)` in eval mode, writes a CSV with appended `analysis_tag` and `split` columns, and may write/log heatmap images.
- Key callers/callees: called from `python -m sleep2expert.routing_analysis`; calls `apply_finetune_config`, `sleep2expert.infer._build_inference_loader`, `Sleep2vecFinetuning`, `_get_eval_model`, `load_checkpoint`, `select_checkpoints`, `average_checkpoints`, and `build_routing_rows`.
- Reuse guidance: use this for `sleep2expert` PHASE-MoE route usage, route-collapse, site/source shortcut exports, and derived routing usage heatmaps.
- Duplication risk notes: this is the package-local route-export surface; do not create a separate token-dump or metrics path for the same `last_moe_aux` contract.

## `sleep2expert.model_stats` helper family

- File: `sleep2expert/model_stats.py`
- Functions:
  - `count_total_parameters(model) -> int`
  - `count_trainable_parameters(model) -> int`
  - `estimate_active_parameters_per_token(model_config) -> int`
  - `estimate_moe_ffn_active_flops(model_config, seq_len) -> int`
  - `estimate_dense_equivalent_ffn_flops(model_config, seq_len) -> int`
  - `summarize_expert_usage(moe_aux) -> dict[int, int]`
- Purpose and contract: centralize lightweight PHASE-MoE parameter, active FFN compute, and expert-usage estimates for experiment logging and audit tables.
- Important inputs/outputs: model or typed config in; scalar counts or expert-id usage counts out.
- Side effects: none.
- Key callers/callees: caller is `sleep2expert.pretrain.sleep2vec_pretrain` for W&B hparam logging; routing/diagnostic callers can reuse `summarize_expert_usage`.
- Reuse guidance: use this module for active-compute accounting instead of adding one-off formulas in entrypoints or notebooks.

## `sleep2expert.checkpoints.initialize_moe_from_dense_if_possible`

- File: `sleep2expert/checkpoints.py`
- Signature: `initialize_moe_from_dense_if_possible(module: torch.nn.Module, filtered_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]`
- Purpose and contract: expand compatible dense standalone RoFormer FFN tensors into MoE expert FFN keys before pretrain initialization; complete MoE expert/layer-norm checkpoint tensors load directly, while missing or shape-incompatible dense-to-MoE init fails fast instead of leaving random experts.
- Important inputs/outputs: target module and prefix-stripped checkpoint state in; expanded state dict out.
- Side effects: none.
- Key callers/callees: caller is `sleep2expert.checkpoints.load_pretrain_init_weights`.
- Reuse guidance: use this helper for sleep2expert dense-to-MoE init instead of adding legacy/HF key conversion.

## `sleep2vec.checkpoints.load_pretrain_init_weights`

- File: `sleep2vec/checkpoints.py`
- Signature: `load_pretrain_init_weights(model: torch.nn.Module, path: Path | str, *, device: torch.device | str = "cpu", strict: bool = False) -> PretrainInitLoadResult`
- Purpose and contract: load checkpoint weights into a model, preferring averaged-model prefixes when present and returning explicit load diagnostics.
- Important inputs/outputs: model and checkpoint path in, `PretrainInitLoadResult` out.
- Side effects: loads checkpoint files from disk and mutates model weights.
- Key callers/callees: callers are `Sleep2vecAdaptation` and adaptation-related tests; callees include `load_checkpoint`, `extract_pretrain_init_state_dict`, and `get_state_dict_from_checkpoint`.
- Reuse guidance: use for weight initialization from pretrain-style checkpoints instead of open-coding prefix stripping.
- Duplication risk notes: `model.` vs `ema_model.` handling belongs here.

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

## `sleep2vec.metrics.compute_downstream_metrics`

- File: `sleep2vec/metrics.py`
- Signature: `compute_downstream_metrics(gts, preds, *, is_classification: bool, is_multilabel: bool = False, output_dim: int | None = None, stage_names=None)`
- Purpose and contract: reduce per-sample predictions into classification, regression, or multilabel metrics, including stage-aware reporting for remapped staging tasks.
- Important inputs/outputs: ground truth and predictions in, metrics dict out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecFinetuning._finalize_epoch`; callees include `compute_binary_label_metrics`, `compute_ahi_pointwise_metrics`, and `roc_auc_from_two_logits`.
- Reuse guidance: use this as the generic downstream metric reducer.
- Duplication risk notes: do not implement alternative epoch metric logic in trainers unless the contract truly changes.

## `utils.check_configs.check_config_file`

- File: `utils/check_configs.py`
- Signature: `check_config_file(path: Path) -> None`
- Purpose and contract: validate one config file against runtime loader compatibility, tokenizer-parity checks, repo `ppg_*finetune*` policy, and `preset_build` strictness.
- Important inputs/outputs: config path in; raises on failure, returns `None` on success.
- Side effects: reads YAML from disk only.
- Key callers/callees: caller is `utils.check_configs.main`; callees include `_validate_runtime_loader_contract`, `_validate_repo_policy`, and `_validate_preset_build_contract`.
- Reuse guidance: use this tooling path for repo-wide config validation instead of ad hoc shell loops.
- Duplication risk notes: static config policy belongs here, not in entrypoints.
