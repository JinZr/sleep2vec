# Functions: Runtime Entrypoints

## `sleep2vec.pretrain.sleep2vec_pretrain`

- File: `sleep2vec/pretrain.py`
- Signature: `sleep2vec_pretrain(args)`
- Purpose and contract: loads pretrain config, applies YAML-derived runtime fields, builds dataloaders, chooses whether to reuse a checkpoint directory or create a fresh run directory, persists config snapshots, constructs the Lightning module, optionally loads pretrain init weights, and launches training.
- Important inputs: argparse namespace with config path, runtime hyperparameters, checkpoint options, and logging fields.
- Important outputs: none; drives trainer side effects.
- Side effects: creates log directories, copies config and CLI args, may resume from checkpoint, starts W&B logging, writes checkpoints.
- Notable callers/callees: script entrypoint; uses `load_pretrain_config`, `apply_model_config_args`, `get_pretrain_dataloader`, `persist_run_config_and_args`, `Sleep2vecPretraining`.
- Reuse guidance: canonical contrastive pretrain entrypoint.
- Duplication-risk notes: medium-high.

## `sleep2vec.adapt._resolve_adapt_run_artifacts`

- File: `sleep2vec/adapt.py`
- Signature: `_resolve_adapt_run_artifacts(*, ckpt_path, pretrained_backbone_path, version_name, backbone_arch, phase, exp_info="") -> AdaptRunArtifacts`
- Purpose and contract: resolves whether adaptation is resuming the same phase, transitioning from stage 1 to stage 2, or starting a fresh run; returns the correct save path, run name, W&B id behavior, trainer checkpoint path, and whether root config snapshots should be rewritten.
- Important inputs: checkpoint/pretrained paths, version metadata, phase.
- Important outputs: `AdaptRunArtifacts`.
- Side effects: none directly; raises on invalid checkpoint/phase combinations.
- Notable callers/callees: called by `sleep2vec_adapt`; uses `_require_checkpoint_file`, `_validate_checkpoint_dir_for_phase`, `_validate_saved_phase`, `_resolve_stage1_transition_checkpoint`.
- Reuse guidance: canonical run-directory state machine for adaptation.
- Duplication-risk notes: very high.

## `sleep2vec.adapt.sleep2vec_adapt`

- File: `sleep2vec/adapt.py`
- Signature: `sleep2vec_adapt(args)`
- Purpose and contract: branch-specific adaptation entrypoint that loads pretrain-style config with `adapt`, derives initial pair probabilities for the selected phase, builds train/validation loaders, resolves run artifacts, persists stage-aware snapshots, constructs `Sleep2vecAdaptation`, and starts training.
- Important inputs: argparse namespace including `phase`, adaptation config path, pretrained checkpoint path, and runtime settings.
- Important outputs: none; drives trainer side effects.
- Side effects: creates or reuses `log-adapt/<run>/...`, writes stage snapshots, starts W&B logging, writes checkpoints.
- Notable callers/callees: script entrypoint; uses `initial_pair_probs_for_phase`, `get_pretrain_dataloader`, `_resolve_adapt_run_artifacts`, `Sleep2vecAdaptation`.
- Reuse guidance: canonical wearable adaptation runtime.
- Duplication-risk notes: very high.

## `sleep2vec.finetune.prepare_dataloader`

- File: `sleep2vec/finetune.py`
- Signature: `prepare_dataloader(args)`
- Purpose and contract: wraps downstream loader creation and logs train/val/test loader lengths.
- Important inputs: argparse namespace with finetune-ready fields.
- Important outputs: `(train_loader, val_loader, test_loader)`.
- Side effects: logging.
- Notable callers/callees: called by `supervised`; delegates to `get_finetune_dataloaders`.
- Reuse guidance: small wrapper only; reuse the underlying loader factory for behavior.
- Duplication-risk notes: low.

## `sleep2vec.finetune.supervised`

- File: `sleep2vec/finetune.py`
- Signature: `supervised(args, config_bundle)`
- Purpose and contract: persists run snapshots, builds downstream dataloaders, constructs `Sleep2vecFinetuning`, sets callbacks and trainer, runs training if epochs are positive, copies best checkpoint to `best.ckpt`, tests on the final checkpoint choice, and appends results to CSV.
- Important inputs: argparse namespace plus parsed finetune config bundle.
- Important outputs: none.
- Side effects: writes log directories and checkpoints, W&B logging, result CSV updates.
- Notable callers/callees: main downstream runtime path; uses `persist_run_config_and_args`, `prepare_dataloader`, `Sleep2vecFinetuning`, `save_result_csv`.
- Reuse guidance: canonical downstream training path.
- Duplication-risk notes: medium-high.

## `sleep2vec.finetune.build_version_name`

- File: `sleep2vec/finetune.py`
- Signature: `build_version_name(args) -> str`
- Purpose and contract: derives a stable run name from label, channel selection, few-shot setting, and whether a pretrained backbone is used when `--version-name` is omitted.
- Important inputs: argparse namespace.
- Important outputs: version string.
- Side effects: none.
- Notable callers/callees: downstream CLI path.
- Reuse guidance: use this if new downstream modes still map to the same naming convention.
- Duplication-risk notes: medium.

## `sleep2vec.infer.run_inference`

- File: `sleep2vec/infer.py`
- Signature: `run_inference(args)`
- Purpose and contract: applies finetune config, normalizes trainer precision for CPU, builds a single evaluation dataloader, optionally averages several checkpoints, runs Lightning test-only evaluation, optionally logs to W&B, and writes metrics to CSV.
- Important inputs: argparse namespace with config, checkpoint, label, averaging, and optional W&B fields.
- Important outputs: none.
- Side effects: may read and average multiple checkpoints, may initialize W&B, may append results to CSV.
- Notable callers/callees: script entrypoint; uses `apply_finetune_config`, `_build_inference_loader`, `select_checkpoints`, `average_checkpoints`, `save_result_csv`.
- Reuse guidance: canonical inference-only path.
- Duplication-risk notes: medium-high.

## `sleep2vec.checkpoints.select_checkpoints`

- File: `sleep2vec/checkpoints.py`
- Signature: `select_checkpoints(ckpt_dir: Path, *, end_ckpt: Path | None, num_ckpts: int) -> list[Path]`
- Purpose and contract: chooses the trailing checkpoint set for averaging, preferring epoch naming when available and falling back to modification time otherwise.
- Important inputs: checkpoint directory, optional end checkpoint, desired count.
- Important outputs: ordered checkpoint list.
- Side effects: filesystem reads.
- Notable callers/callees: used by `sleep2vec.infer.run_inference`.
- Reuse guidance: canonical selection logic for inference-time averaging.
- Duplication-risk notes: high.

## `sleep2vec.checkpoints.average_checkpoints`

- File: `sleep2vec/checkpoints.py`
- Signature: `average_checkpoints(filenames, *, device=torch.device("cpu")) -> dict[str, torch.Tensor]`
- Purpose and contract: averages matching tensors across several Lightning checkpoints, using integer floor division for non-floating tensors.
- Important inputs: sequence of checkpoint paths, target device.
- Important outputs: averaged state dict.
- Side effects: checkpoint file reads, logging.
- Notable callers/callees: used by `sleep2vec.infer.run_inference`.
- Reuse guidance: canonical checkpoint averaging implementation.
- Duplication-risk notes: high.
