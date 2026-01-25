# sleep2vec

<div align="center">
  <img src="doc/image/banner.png" width="550"/>
  <p><strong>Modular self-supervised learning for sleep signals</strong></p>
  <p>Models and losses live in YAML; training hyperparameters stay on the CLI.</p>
</div>

---

**Quick links** · `configs/` recipes · `configs2/` sleep2vec2 recipes · `sleep2vec/` source · `sleep2vec2/` recipe README · `data/` datasets & loaders · `preprocess/` caching scripts · `utils/` helpers

---

## Table of Contents
- [sleep2vec](#sleep2vec)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Setup](#setup)
  - [Data Format \& Caches](#data-format--caches)
  - [Quick Start](#quick-start)
    - [Pretrain (contrastive)](#pretrain-contrastive)
    - [Finetune — classification](#finetune--classification)
    - [Finetune — regression](#finetune--regression)
  - [Inference Only](#inference-only)
  - [Configuration Knobs](#configuration-knobs)
  - [Diagnostics Mode](#diagnostics-mode)
  - [Working Tips](#working-tips)
  - [Repository Layout](#repository-layout)

---

## Overview
- Refactored training flow: **YAML defines architecture & loss**, **CLI sets training hyperparameters** (epochs, lr, devices, etc.).
- Supports contrastive pretraining plus downstream classification or regression finetuning.
- Extensible registries for backbones, tokenizers, projection heads, losses, model averaging, LoRA-backed heads, and downstream heads.
- WandB logging is enabled by default; inference-only runner is included for evaluating checkpoints.

---

## Setup
- Python 3.10+ with CUDA GPUs recommended; PyTorch/Lightning versions are pinned in `requirements.txt` (`torch==2.5.1`, `pytorch-lightning==2.5.5`).
- Install: `pip install -r requirements.txt` (choose the correct PyTorch wheel for your CUDA version).
- Pair-accuracy heatmap logging uses `matplotlib` + `seaborn` (already included in `requirements.txt`).
- Authenticate to Weights & Biases before running (`WANDB_API_KEY=...` or `WANDB_MODE=offline`) because entrypoints call `wandb.login()`.
- Default precision is bf16/bf16-mixed; pass `--precision 32` if your GPUs do not support bf16.

---

## Data Format & Caches
- **Index CSV** (used by pretrain/finetune): required columns `path`, `split` (`train|val|test`), `duration` (seconds), `age`, `sex`; optional extra label columns (e.g., disease flags) are consumed when `meta_data_names` is set.
- **NPZ contents per row**: keys `heartbeat`, `breath`, `eeg_original`, `ecg_original`, `eog_original`, `emg_original`, `spo2`, `resp_original`, `resp_nasal_original`, `stage5`. Each NPZ stores contiguous 30 s windows. 128 Hz channels expect 3840 frames/token; 4 Hz channels expect 120 frames/token; `stage5` is one label per token.
- **Preset pickles**: both CLIs expect a precomputed pickle of `SampleIndex` objects (see `preprocess/save_Dataset_preset.py`). Point `--pretrain-preset-path` / YAML `data.finetune_preset_path` to an existing pickle; these scripts do **not** fall back to CSV when a path is provided.
- To build a preset, edit `preprocess/save_Dataset_preset.py` (set `index=[...]`, `save_preset_path`, `split`, `n_tokens`) and run it. Reuse the produced pickle paths in your YAML/CLI flags.
- **Missing-channel pretrain**: if you enable `--allow-missing-channels`, presets must carry `payload["available_channels"]` (auto-populated during preset creation) so the bucketed sampler can group by montage.

---

## Quick Start

> Update the dataset paths in the commands below (`--pretrain-data-index`, `--pretrain-preset-path`, and YAML `data.finetune_*`) to point to your own CSV/pickle files.

### Pretrain (contrastive)
```bash
python -m sleep2vec.pretrain \
  --config configs/sleep2vec_dense_pretrain.yaml \
  --pretrain-data-index /path/to/index.csv \
  --pretrain-preset-path /path/to/pretrain_cache.pkl \
  --version-name exp001 \
  --epochs 120 --lr 5e-5 --batch-size 320 \
  --devices 0 1 --num-workers 8
```
Optional:
- `--warmup-steps N` to override the default LR warmup (3% of total steps).
- `--allow-missing-channels` to accept samples with missing channels; pair with `--min-channels` and (recommended) `--bucket-by-available-channels`.

Example (missing-channel pretrain):
```bash
python -m sleep2vec.pretrain \
  --config configs/sleep2vec_dense_pretrain.yaml \
  --pretrain-data-index /path/to/index.csv \
  --pretrain-preset-path /path/to/pretrain_cache.pkl \
  --version-name exp001-missing-chn \
  --epochs 120 --lr 5e-5 --batch-size 320 \
  --devices 0 1 --num-workers 8 \
  --allow-missing-channels --min-channels 6 --bucket-by-available-channels
```

### Finetune — classification
```bash
python -m sleep2vec.finetune \
  --config configs/sleep2vec_dense_finetune_cls.yaml \
  --label-name stage5 --results-csv-path outputs.csv \
  --version-name exp001-stage5 \
  --epochs 50 --lr 1e-5 --devices 0 1
```
Notes:
- `stage5` is a **per-token sequence labeling** task (`is_seq=True`). Use token-level downstream (`model.cls.downstream: tokens`).
- Do **not** add `stage5` to `data.data_channel_names`; it is loaded as a label into `batch["tokens"]["stage5"]` automatically when `--label-name stage5`.

### Finetune — regression
```bash
python -m sleep2vec.finetune \
  --config configs/sleep2vec_dense_finetune_reg.yaml \
  --label-name age --results-csv-path outputs.csv \
  --version-name exp001-age \
  --epochs 50 --lr 1e-5 --devices 0 1
```

> [!Note]
> `--version-name` is required for pretraining run naming; downstream runs auto-generate a version when omitted. Ensure your YAML `data.*` paths point to real preset pickles.

---

## Inference Only
Evaluate a fine-tuned checkpoint without training:

```bash
python -m sleep2vec.infer \
  --config configs/sleep2vec_dense_finetune_cls.yaml \
  --ckpt-path log-finetune/exp001-stage5/checkpoints/epoch=49.ckpt \
  --label-name stage5 --batch-size 12 --devices 0 \
  --eval-split test --results-csv-path outputs.csv
```
Use `--override-dataset-names` to test on a different dataset list than the YAML specifies.
To average checkpoints before inference, pass `--avg-ckpts N` (and `--avg-ckpt-dir` if `--ckpt-path` is `best/last`).
Use `--wandb` to enable W&B logging during inference (needed for confusion matrix logging).

---

## Configuration Knobs

**Backbone**  
- Register builders in `sleep2vec/backbones/encoder_factory.py` with `@register_backbone`.
- Select via YAML:
  ```yaml
  model:
    backbone:
      name: roformer
      hidden_size: 768
      num_hidden_layers: 12
      num_attention_heads: 16
      vocab_size: 1
      config_overrides: {}   # add custom kwargs (e.g., MoE routing)
  ```

**Tokenizers**  
- Implement and register in `sleep2vec/modules/tokenizers.py` using `@register_tokenizer("my_tokenizer")`.
- Set per-channel (tokenizer block must supply `name` and `out_dim`):
  ```yaml
  model:
    channels:
      - name: eeg_original
        input_dim: 3840
        tokenizer:
          name: my_tokenizer
          out_dim: 768        # must match across channels
          kwargs: {}
  ```

**Projection Head**  
- Register in `sleep2vec/modules/projection.py` via `@register_projection`.
- Toggle or adjust:
  ```yaml
  model:
    projection:
      name: my_proj
      enabled: true
      hidden_dim: 768
      out_dim: 256
      kwargs: {}
  ```

**Pretrain Loss**  
- Add implementations under `sleep2vec/losses/` and register with `@register_loss`.
- Choose in YAML:
  ```yaml
  loss:
    name: my_loss
    temperature: 0.2
    params: {}
  ```

**Downstream Head**  
- Heads live in `sleep2vec/downstreams/heads/` and register via `sleep2vec/downstreams/head_registry.py`.
- YAML separates channel and temporal aggregation:
  ```yaml
  model:
    head:
      name: classification   # regression | temporal_conv | temporal_transformer
      dropout: 0.1
      hidden_dim: null
      channel_agg:
        name: gated_scalar   # mean | concat | gated_scalar
      temporal_agg:
        name: mean           # mean | attn
  ```
- Temporal aggregation modules are in `sleep2vec/downstreams/temporal_aggregation/`.
- Sequence heads like `temporal_transformer` accept a padding mask from the backbone to ignore padded tokens.


**CLS vs Tokens (downstream representation)**  
`model.cls` controls (1) whether a learnable CLS token is added, and (2) what representation downstream heads consume:
```yaml
model:
  cls:
    embedding_type: null    # null/none -> no CLS token; "bert" -> prepend learnable CLS token
    downstream: tokens      # "tokens" (token-level) or "cls" (sequence-level, non-seq tasks only)
```
- `embedding_type: bert` adds a BERT-style CLS token and exposes both `cls_hidden` and `token_hidden`.
- `downstream: tokens` uses token-level features (sequence tasks) or token pooling (non-seq tasks via `model.head.temporal_agg`).
- `downstream: cls` uses the CLS embedding for **non-seq** tasks and requires `embedding_type: bert`.
- For `--label-name stage5` (`is_seq=True`), downstream is always token-level; if you set `downstream: cls` it will be ignored (a warning is logged).
- If `model.cls` is omitted, the default is “no CLS token + token/pooled downstream”.

**Model Averaging**  
- Strategies live in `sleep2vec/averagings/` (`ema.py` and `running_mean.py` included).
- Configure (omit the block entirely to disable):
  ```yaml
  model_averaging:
    name: ema
    params:
      enabled: true
      base_momentum: 0.996
      final_momentum: 1.0
      use_for_eval: true
  ```
- When `use_for_eval: true`, finetune/infer will evaluate with the averaged weights if present.
- Downstream loading can request averaged weights with `use_ema="ema"` when calling `load_pretrained_backbone`.

**Optimization & Checkpointing**  
- Pretrain/finetune use linear warmup + cosine LR decay; override warmup with `--warmup-steps`.
- Finetune saves `best.ckpt` and `last.ckpt` plus periodic checkpoints; set `--ckpt-every-n-epochs` to control frequency.
- `--precision` and `--gradient-clip-val` are supported by both pretrain and finetune CLIs.

**Pair-accuracy heatmap (pretrain)**  
- Validation uses per-pair dataloaders to log contrastive accuracy per modality pair.
- W&B logs a heatmap image (`val_pair_acc_matrix`) plus scalar metrics under `val_pair_acc/<pair>`.

**LoRA fine-tuning**  
- Controlled by YAML `lora` block (parsed by `apply_finetune_config`):
  ```yaml
  lora:
    freeze_backbone_and_insert_lora: true
    insert_lora: true
    separate_adapters: false
  ```
- When enabled, `finetune.py` injects PEFT LoRA adapters into the transformer backbone and freezes base weights.

---

## Diagnostics Mode
- Enable hooks with `--print-diagnostics`; control duration with `--diagnostics-steps` (default 5).
- Behavior: disables the progress bar, skips validation/checkpointing, and stops after the requested steps. Stats print to stdout.
- Example (pretrain):
  ```bash
  python -m sleep2vec.pretrain \
    --config configs/sleep2vec_dense_pretrain.yaml \
    --pretrain-data-index /path/to/index.csv \
    --pretrain-preset-path /path/to/pretrain_cache.pkl \
    --version-name debug-diag \
    --print-diagnostics --diagnostics-steps 5 --precision 32 --devices 0
  ```
  (Use the same flags with `sleep2vec.finetune` for downstream diagnostics.)

> [!Important]
> Prefer `--precision 32` when using diagnostics; mixed precision can distort the printed tensor statistics.

---

## Working Tips
- Maintain separate YAML per stage (`*_pretrain.yaml`, `*_finetune_*.yaml`); only pretrain YAML defines `loss`.
- All channels must share the same `out_dim`; the builder enforces this.
- `data.data_channel_names` in finetune YAML must match `model.channels` (input modalities only); per-token labels like `stage5` are loaded automatically when used as `--label-name`.
- When experimenting, adjust CLI flags for training schedules and keep structural changes in YAML for reproducibility.

---

## Repository Layout
- `configs/` — training recipes for pretrain/finetune.
- `sleep2vec/` — core library: registries, encoders, tokenizers, projection, losses, averaging, downstream heads, Lightning entrypoints.
- `data/` — dataset/index definitions, metadata helpers, NPZ loaders, channel-selection & samplers.
- `preprocess/` — scripts to build index CSVs/presets, split/merge dataset indices, and inspect missing-channel stats.
- `utils/` — misc helpers.
