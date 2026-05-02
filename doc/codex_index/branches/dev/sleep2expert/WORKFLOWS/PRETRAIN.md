# Pretrain Workflow

## Purpose

Run contrastive multimodal pretraining from a YAML recipe plus CLI runtime flags.

## Entry Command

Canonical entrypoint: `python -m sleep2vec.pretrain --config ...`

Primary code path:

1. `sleep2vec.pretrain.sleep2vec_pretrain`
2. `sleep2vec.config.load_pretrain_config`
3. `sleep2vec.common.apply_model_config_args`
4. `sleep2vec.utils.get_pretrain_dataloader`
5. `sleep2vec.sleep2vec_modelling.Sleep2vecPretraining`
6. `sleep2vec.pretrain_model.Sleep2vecPretrainModel`
7. `sleep2vec.losses.create_loss`
8. Lightning `trainer.fit(...)`

## Detailed Flow

1. Load YAML.
   - Reads `model`, `loss`, `data`, optional `model_averaging`, and optional `adapt`.
   - Requires `model.backbone`, `model.projection`, and `model.cls`.
2. Copy selected config values into `args`.
   - `mask_rate`
   - `max_tokens`
   - `channel_names`
   - `channel_input_dims`
   - optionally `backbone_arch`
3. Build train and validation loaders.
   - Train loader is a single `PSGPretrainDataset`.
   - Validation is a single loader whose `SequentialPairEvalBatchSampler` iterates supported modality pairs.
   - Missing-channel mode changes worker counts, sampler choice, and batch sharding.
4. Resolve experiment directory.
   - Resume path: infer experiment directory from `--ckpt-path`.
   - Fresh run: create `log-pretrain/<run-name>/checkpoints`.
5. Persist artifacts.
   - Copy YAML to `config.yaml`.
   - Dump the bound CLI namespace to `cli_args.yaml`.
6. Build Lightning module.
   - `Sleep2vecPretraining` creates `Sleep2vecPretrainModel`.
   - Loss is created from the registry.
   - `sleep2expert` MoE configs add local MoE regularization from `model.last_moe_aux` after contrastive loss; it is not a registry loss.
   - `sleep2expert.model_stats` estimates total/trainable parameters and FFN active-compute stats, and `sleep2expert.pretrain` logs them to W&B hparams.
   - Optional model averager is attached.
7. Build trainer.
   - Standard mode: callbacks enabled.
   - Diagnostics mode: progress bar, checkpointing, and validation are disabled.
8. Train.
   - `trainer.fit(model, train_dataloaders=..., val_dataloaders=..., ckpt_path=...)`

## Batch Contract Used By Pretraining

- `tokens`: channel tensors
- `mlm_mask`: channel token masks
- `length`: valid token counts
- `token_start`: token offset used later by AHI event aggregation on downstream paths
- `w`, `h`: only required for `weighted_info_nce`
- `pair`: optional, mainly for sampler-aware logging and pair-eval batches

## Important Runtime Decisions

- Pair-first missing-channel training is selected inside `DefaultDataset.dataloader`, not in the entrypoint.
- Lightning distributed sampler injection is disabled when the batch sampler already shards by rank.
- Validation pair metrics are aggregated inside one callback-aware loader, not by spawning one loader object per pair.
- `sleep2expert` MoE regularization logs `*_moe_*` diagnostics but keeps `val_contrastive_acc` as the checkpoint monitor.
- `sleep2expert` MoE pretraining logs active params per token, MoE active FFN FLOPs, dense-equivalent FFN FLOPs, top-k, expert count, MoE layer ids, and expert hidden size.

## Outputs

- Checkpoints under `log-pretrain/<run>/checkpoints/`
- `config.yaml` and `cli_args.yaml` one directory above checkpoints
- W&B run under project `sleep2expert-pretrain`

## Edit Hotspots

- Change YAML schema: `sleep2vec/config.py`
- Change dataloader construction or missing-channel policy: `sleep2vec/utils.py`, `data/default_dataset.py`, `data/samplers.py`
- Change backbone forward contract: `sleep2vec/pretrain_model.py`
- Change loss semantics: `sleep2vec/losses/`
- Change `sleep2expert` MoE auxiliary loss semantics: `sleep2expert/losses/moe_regularization.py`
- Change trainer/callback/checkpoint behavior: `sleep2vec/pretrain.py`, `sleep2vec/sleep2vec_modelling.py`, `sleep2vec/callbacks/pair_acc_logger.py`
