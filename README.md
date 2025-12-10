# sleep2vec

<div align="center">
  <img src="doc/image/banner.png" width="550"/>
  <p><strong>Modular self-supervised learning for sleep signals</strong></p>
  <p>Models and losses live in YAML; training hyperparameters stay on the CLI.</p>
</div>

---

**Quick links** · `configs/` recipes · `sleep2vec/` source · `data/` datasets & loaders · `preprocess/` caching scripts · `utils/` helpers

---

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Data Format & Caches](#data-format--caches)
- [Quick Start](#quick-start)
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
- Authenticate to Weights & Biases before running (`WANDB_API_KEY=...` or `WANDB_MODE=offline`) because entrypoints call `wandb.login()`.
- Default precision is bf16/bf16-mixed; pass `--precision 32` if your GPUs do not support bf16.

---

## Data Format & Caches
- **Index CSV** (used by pretrain/finetune): required columns `path`, `split` (`train|val|test`), `duration` (seconds), `age`, `sex`; optional extra label columns (e.g., disease flags) are consumed when `meta_data_names` is set.
- **NPZ contents per row**: keys `heartbeat`, `breath`, `eeg_original`, `ecg_original`, `eog_original`, `emg_original`, `spo2`, `resp_original`, `resp_nasal_original`, `stage5`. Each NPZ stores contiguous 30 s windows. 128 Hz channels expect 3840 frames/token; 4 Hz channels expect 120 frames/token; `stage5` is one label per token.
- **Preset pickles**: both CLIs expect a precomputed pickle of `SampleIndex` objects (see `preprocess/save_Dataset_preset.py`). Point `--pretrain-preset-path` / YAML `data.finetune_preset_path` to an existing pickle; these scripts do **not** fall back to CSV when a path is provided.
- To build a preset, edit `preprocess/save_Dataset_preset.py` (set `index=[...]`, `save_preset_path`, `split`, `n_tokens`) and run it. Reuse the produced pickle paths in your YAML/CLI flags.

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

### Finetune — classification
```bash
python -m sleep2vec.finetune \
  --config configs/sleep2vec_dense_finetune_cls.yaml \
  --label-name stage5 --results-csv-path outputs.csv \
  --version-name exp001-stage5 \
  --epochs 50 --lr 1e-5 --devices 0 1
```

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

---

## Configuration Knobs

**Backbone**  
- Register builders in `sleep2vec/backbone/encoder_factory.py` with `@register_backbone`.
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
- Set per-channel:
  ```yaml
  model:
    channels:
      - name: eeg_original
        input_dim: 3840
        out_dim: 768        # must match across channels
        tokenizer: my_tokenizer
        tokenizer_kwargs: {}
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
- Heads live in `sleep2vec/downstream/heads.py` and register via `sleep2vec/downstream/head_registry.py`.
- YAML separates channel and temporal aggregation:
  ```yaml
  model:
    head:
      name: classification   # or regression
      dropout: 0.1
      hidden_dim: null
      channel_agg:
        name: gated_scalar   # mean | concat | gated_scalar
      temporal_agg:
        name: mean           # mean | attn
  ```
- Temporal aggregation modules are in `sleep2vec/downstream/temporal_aggregation/`.

**Model Averaging (EMA / running mean)**  
- Implemented in `sleep2vec/averaging/`; configure at top level in YAML:
  ```yaml
  model_averaging:
    name: ema
    params:
      enabled: true
      base_momentum: 0.996
      final_momentum: 1.0
      use_for_eval: true
  ```
- Downstream loading can request averaged weights with `use_ema="ema"` when calling `load_pretrained_backbone`.

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
- `data.data_channel_names` in finetune YAML must match `model.channels`; mismatch raises early.
- When experimenting, adjust CLI flags for training schedules and keep structural changes in YAML for reproducibility.

---

## Repository Layout
- `configs/` — training recipes for pretrain/finetune.
- `sleep2vec/` — core library: registries, encoders, tokenizers, projection, losses, averaging, downstream heads, Lightning entrypoints.
- `data/` — dataset/index definitions, metadata helpers, NPZ loaders.
- `preprocess/` — scripts/notebooks to build index CSVs and preset pickles.
- `utils/` — misc helpers.
