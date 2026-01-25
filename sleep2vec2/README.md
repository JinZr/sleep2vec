# sleep2vec2 recipe

This README documents the sleep2vec2 training recipe and how it differs from the
original sleep2vec package. The recipe is driven by YAML model/loss configs and
CLI training flags, matching the workflow used by sleep2vec.

## Overview
- Uses the sleep2vec2 codepath (`sleep2vec2/`) with a local RoFormer encoder
  implementation and optional MoE routing.
- Tokenizers include `sundial2` variants with extra normalization/scaling
  utilities (see `sleep2vec2/modules/tokenizers.py` and `sleep2vec2/scaling.py`).
- Pretrain validation is performed over all channel pairs, producing a separate
  val loader per pair.
- Missing-channel pretraining can bucket by available-channel signature to keep
  batches consistent.

## Requirements
- Python 3.10+ with CUDA GPUs recommended.
- `sleep2vec2/scaling.py` relies on `k2` (manual install; see k2 docs for
  pre-built wheels).
- Weights & Biases is used by default; set `WANDB_MODE=offline` or `disabled`
  if needed.

## Pretrain (contrastive)
Use the sleep2vec2 entrypoint and the configs in `configs2/`.

```bash
python -m sleep2vec2.pretrain \
  --config configs2/sleep2vec_dense_pretrain_cls.yaml \
  --pretrain-data-index /path/to/index.csv \
  --pretrain-preset-path /path/to/pretrain_cache.pkl \
  --version-name exp001 \
  --epochs 120 --lr 5e-5 --batch-size 320 \
  --devices 0 1 --num-workers 8
```

### Missing-channel pretrain
If your preset includes `payload["available_channels"]`, you can enable
missing-channel support and bucket by available-channel signatures:

```bash
python -m sleep2vec2.pretrain \
  --config configs2/sleep2vec_dense_pretrain_cls.yaml \
  --pretrain-data-index /path/to/index.csv \
  --pretrain-preset-path /path/to/pretrain_cache.pkl \
  --version-name exp001-missing-chn \
  --epochs 120 --lr 5e-5 --batch-size 320 \
  --devices 0 1 --num-workers 8 \
  --allow-missing-channels --min-channels 6 --bucket-by-available-channels
```

### MoE pretrain
The MoE recipe uses `roformer_moe` and logs MoE routing metrics:

```bash
python -m sleep2vec2.pretrain \
  --config configs2/sleep2vec_moe_pretrain_cls.yaml \
  --pretrain-data-index /path/to/index.csv \
  --pretrain-preset-path /path/to/pretrain_cache.pkl \
  --version-name exp002-moe \
  --epochs 120 --lr 5e-5 --batch-size 320 \
  --devices 0 1 --num-workers 8
```

## Finetune
The sleep2vec2 finetune entrypoint uses the same YAML schema as sleep2vec. You
can reuse existing configs in `configs/` or copy them and swap tokenizers/backbone
names as needed.

```bash
python -m sleep2vec2.finetune \
  --config configs/sleep2vec_dense_finetune_cls.yaml \
  --label-name stage5 --results-csv-path outputs.csv \
  --version-name exp001-stage5 \
  --epochs 50 --lr 1e-5 --devices 0 1
```

## Validation behavior
- Pretrain validation builds one DataLoader per channel pair via
  `data.channel_selection.build_all_pairs`.
- Pair-wise validation accuracy is logged via `PairAccLoggerCallback`, including
  an optional heatmap when W&B is enabled.

## Key files
- `configs2/sleep2vec_dense_pretrain_cls.yaml`: main sleep2vec2 pretrain config.
- `configs2/sleep2vec_moe_pretrain_cls.yaml`: MoE pretrain config.
- `sleep2vec2/pretrain.py`: sleep2vec2 pretrain entrypoint.
- `sleep2vec2/utils.py`: pretrain dataloader construction (pair val loaders,
  missing-channel bucketing).
