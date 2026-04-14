# Runtime Orchestration

## `sleep2vec.pretrain.sleep2vec_pretrain`

- File: `sleep2vec/pretrain.py`
- Signature: `sleep2vec_pretrain(args) -> None`
- Purpose and contract: canonical pretrain entrypoint after argument parsing; binds YAML into runtime args, builds loaders, configures callbacks, persists run artifacts, and launches `trainer.fit`.
- Important inputs/outputs: CLI namespace in; no direct return value.
- Side effects: creates experiment directories, copies config, writes `cli_args.yaml`, initializes W&B, runs training.
- Key callers/callees: called from `__main__`; calls `load_pretrain_config`, `get_pretrain_dataloader`, `Sleep2vecPretraining`, `dump_cli_args_yaml`.
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
- Purpose and contract: canonical finetune orchestration routine; persists run artifacts, builds loaders, instantiates `Sleep2vecFinetuning`, trains, evaluates, and writes results.
- Important inputs/outputs: CLI namespace and config bundle in; no direct return value.
- Side effects: creates run directories, writes YAML snapshots, trains/tests models, copies `best.ckpt`, appends results CSV.
- Key callers/callees: called from `__main__`; calls `prepare_dataloader`, `Sleep2vecFinetuning`, `save_result_csv`, `dump_cli_args_yaml`.
- Reuse guidance: extend this routine instead of creating parallel finetune scripts.
- Duplication risk notes: configuration-copy behavior overlaps with `pretrain.py`.

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

## `sleep2vec.infer._init_wandb`

- File: `sleep2vec/infer.py`
- Signature: `_init_wandb(args)`
- Purpose and contract: initialize a rank-zero-only W&B run for inference when requested.
- Important inputs/outputs: namespace in, W&B run or `None` out.
- Side effects: external W&B initialization.
- Key callers/callees: caller is `run_inference`; callee is `_is_rank_zero`.
- Reuse guidance: use this exact gating if new inference artifacts need W&B.
- Duplication risk notes: keep rank-zero gating centralized.

## `sleep2vec.infer.run_inference`

- File: `sleep2vec/infer.py`
- Signature: `run_inference(args) -> None`
- Purpose and contract: canonical inference driver; normalizes config, builds trainer and loader, optionally averages checkpoints, runs evaluation, and writes metrics.
- Important inputs/outputs: namespace in; no direct return value.
- Side effects: optional W&B run, trainer evaluation, optional results CSV output.
- Key callers/callees: called from `__main__`; calls `apply_finetune_config`, `_build_inference_loader`, `select_checkpoints`, `average_checkpoints`, `_init_wandb`.
- Reuse guidance: extend here for inference-only behavior changes.
- Duplication risk notes: checkpoint averaging policy belongs here plus `checkpoints.py`, not in trainer code.

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

## `sleep2vec.metrics.compute_downstream_metrics`

- File: `sleep2vec/metrics.py`
- Signature: `compute_downstream_metrics(gts, preds, *, is_classification: bool, is_multilabel: bool = False, output_dim: int | None = None, stage_names=None)`
- Purpose and contract: reduce per-sample predictions into multiclass classification, seq multi-label binary, or regression metrics, with stage-specific metrics for sleep-staging outputs.
- Important inputs/outputs: ground truth and predictions in, metrics dict out.
- Side effects: none.
- Key callers/callees: caller is `Sleep2vecFinetuning._finalize_epoch`; callees are `compute_binary_label_metrics` for `ahi`-style flattened binary outputs and `roc_auc_from_two_logits` for two-logit classification.
- Reuse guidance: use this as the only downstream metric reducer.
- Duplication risk notes: do not implement alternative epoch metric logic in trainers unless the contract truly changes.

## `sleep2vec.metrics.save_result_csv`

- File: `sleep2vec/metrics.py`
- Signature: `save_result_csv(pretrain_result: Mapping[str, float], csv_path: str, args: Any | None = None) -> None`
- Purpose and contract: append a metrics row plus selected run metadata to a CSV file, creating the file and parent directory as needed.
- Important inputs/outputs: metrics mapping and destination path in; no return value.
- Side effects: creates directories and writes or appends to CSV.
- Key callers/callees: callers are `finetune.supervised` and `infer.run_inference`.
- Reuse guidance: reuse for any tabular experiment summary output.
- Duplication risk notes: current column policy is encoded here; do not create parallel result-writer variants casually.
