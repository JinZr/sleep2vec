# Workflow: Pretrain Contrastive

## Runtime Sequence

1. CLI enters `sleep2vec.pretrain.sleep2vec_pretrain`.
2. `sleep2vec.config.load_pretrain_config` parses YAML.
3. `sleep2vec.common.apply_model_config_args` copies channel names and input widths into `args`.
4. `sleep2vec.utils.get_pretrain_dataloader` builds the train loader plus one validation loader per channel pair.
5. `sleep2vec.common.persist_run_config_and_args` snapshots config and CLI args beside the run directory.
6. `Sleep2vecPretraining` constructs `Sleep2vecPretrainModel` and the configured loss.
7. Lightning trainer runs with checkpointing, early stopping, LR monitoring, and pair-accuracy logging unless diagnostics mode disables them.

## Key Contracts

- `args.version_name` is required for fresh pretrain runs.
- `--ckpt-path` means Lightning resume in-place.
- `--pretrained-backbone-path` means initialize model weights before training when not resuming.
- `allow_missing_channels=True` changes dataset and sampler behavior; it is not just a filtering flag.
- validation runs one dataloader per channel pair, not one mixed validation loader.

## Missing-Channel Path

When `--allow-missing-channels` is enabled:

- presets must contain `payload["available_channels"]`
- train batches may use `PairFirstBatchSampler`
- validation samples may be filtered to those that support the scheduled pair
- Lightning distributed sampler injection is disabled if the batch sampler already shards across ranks

## Main Reuse Points

- config parsing: `sleep2vec.config.load_pretrain_config`
- data assembly: `sleep2vec.utils.get_pretrain_dataloader`
- model/loss wrapper: `sleep2vec.sleep2vec_modelling.Sleep2vecPretraining`
- checkpoint init: `sleep2vec.checkpoints.load_pretrain_init_weights`

## Common Change Boundaries

- add or change channels: update YAML and dataset-width plumbing, not the entrypoint first
- change train pair behavior: update samplers and `PairAccLoggerCallback` together
- change run-directory semantics: start with `persist_run_config_and_args` and checkpoint helpers
