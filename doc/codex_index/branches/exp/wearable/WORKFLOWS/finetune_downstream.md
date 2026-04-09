# Workflow: Finetune Downstream

## Runtime Sequence

1. Finetune CLI parses arguments.
2. `sleep2vec.common.apply_finetune_config` loads YAML and applies:
   - `channel_names`
   - `channel_input_dims`
   - `data_channel_names`
   - task semantics
   - LoRA and freeze flags
3. `sleep2vec.finetune.build_version_name` derives a run name when needed.
4. `sleep2vec.finetune.supervised` persists config snapshots and builds train/val/test loaders.
5. `Sleep2vecFinetuning` constructs the shared backbone plus `Sleep2vecDownstreamModel`.
6. Optional backbone init loading happens through `load_pretrained_backbone`.
7. Optional LoRA insertion happens through `freeze_backbone_and_insert_lora`.
8. Lightning trains, copies `best.ckpt`, tests, and appends metrics to the requested CSV.

## Task Contracts

- built-in labels: `stage5`, `sex`, `age`
- custom labels require `finetune.task` in YAML
- token-level prediction is only supported for `stage5` in current runtime code
- `data.data_channel_names` must match `model.channels`

## Main Reuse Points

- task resolution: `sleep2vec.common.apply_task_flags`
- downstream loader build: `sleep2vec.utils.get_finetune_dataloaders`
- backbone wrapper: `sleep2vec.pretrain_model.Sleep2vecPretrainModel`
- downstream wrapper: `sleep2vec.downstream_model.Sleep2vecDownstreamModel`
- metrics: `sleep2vec.metrics.compute_downstream_metrics`

## Common Change Boundaries

- new downstream task semantics: start in YAML parser and `apply_task_flags`
- new head type: register it in `sleep2vec/downstreams/head_registry.py`
- layer-mix behavior: update `Sleep2vecDownstreamModel` and its tests together
