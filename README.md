# sleep2vec

<div align="center">
  <img src="doc/image/banner.png" width="550"/>
  <p><strong>Modular self-supervised learning for sleep signals</strong></p>
  <p>Models and losses live in YAML; training hyperparameters stay on the CLI.</p>
</div>

---

**Quick links** · `configs/` recipes · `sleep2vec/` source · `preprocess/` utilities · `utils/` helpers

---

## Table of Contents
- [sleep2vec](#sleep2vec)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Quick Start](#quick-start)
  - [Configuration Knobs](#configuration-knobs)
  - [Diagnostics Mode](#diagnostics-mode)
  - [Working Tips](#working-tips)
  - [Repository Layout](#repository-layout)

---

## Overview
- Refactored training flow: **YAML defines architecture & loss**, **CLI sets training hyperparameters** (epochs, lr, devices, etc.).
- Supports contrastive pretraining plus downstream classification or regression finetuning.
- Extensible registries for backbones, tokenizers, projection heads, losses, and downstream heads.

---

## Quick Start

```bash
# Pretrain (contrastive)
python -m sleep2vec.pretrain \
  --config configs/sleep2vec_dense_pretrain.yaml \
  --epochs 120 --lr 5e-5 --devices 0 1

# Finetune — classification
python -m sleep2vec.finetune \
  --config configs/sleep2vec_dense_finetune_cls.yaml \
  --label-name stage5 --results-csv-path outputs.csv \
  --epochs 50 --lr 1e-5

# Finetune — regression
python -m sleep2vec.finetune \
  --config configs/sleep2vec_dense_finetune_reg.yaml \
  --label-name age --results-csv-path outputs.csv \
  --epochs 50 --lr 1e-5

# Diagnostics-only run (no progress bar)
python -m sleep2vec.pretrain \
  --config configs/sleep2vec_dense_pretrain.yaml \
  --print-diagnostics --diagnostics-steps 5 --devices 0
```

> [!Note]
> Keep hyperparameter tweaks on the CLI; change architectures or losses in YAML.

---

## Configuration Knobs

**Backbone**  
- Register builders in `sleep2vec/encoder_factory.py` with `@register_backbone`.
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
- Implement and register in `sleep2vec/pretrain/tokenizers.py` using `@register_tokenizer("my_tokenizer")`.
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
- Register in `sleep2vec/pretrain/projection.py` via `@register_projection`.
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
- For quick tweaks, edit YAML head settings:
  ```yaml
  model:
    head:
      name: classification   # or regression
      agg: gated_scalar      # mean | concat also available
      dropout: 0.1
      hidden_dim: null
  ```
- For new heads, implement in `sleep2vec/downstream/heads.py`, register with `@register_head`, and reference by name.

**Model Averaging**  
- Strategies live in `sleep2vec/model_averaging.py` (EMA and running_mean included).
- Configure (omit the block entirely to disable):
  ```yaml
  model_averaging:
    name: ema               # or running_mean
    params:
      enabled: true
      base_momentum: 0.996
      final_momentum: 1.0
      use_for_eval: true
  ```
- Downstream loading can request averaged weights via `use_ema="ema"` (or `False` for student weights).

---

## Diagnostics Mode
- Enable hooks with `--print-diagnostics`; control duration with `--diagnostics-steps` (default 5).
- Behavior: disables the progress bar, skips validation/checkpointing, and stops after the requested steps. Stats print to stdout.
- Example:
  ```bash
  python -m sleep2vec.finetune \
    --config configs/sleep2vec_dense_finetune_cls.yaml \
    --label-name stage5 --results-csv-path /tmp/out.csv \
    --print-diagnostics --diagnostics-steps 5 --devices 0
  ```

> [!Important]
> Set `--precision 32` when using diagnostics. Mixed precision distorts the collected stats.

---

## Working Tips
- Maintain separate YAML per stage (`*_pretrain.yaml`, `*_finetune_*.yaml`); only pretrain YAML defines `loss`.
- All channels must share the same `out_dim`; the builder enforces this.
- When experimenting, adjust CLI flags for training schedules and keep structural changes in YAML for reproducibility.

---

## Repository Layout
- `configs/` — training recipes for pretrain/finetune.
- `sleep2vec/` — core library: registries, encoders, tokenizers, heads, losses, model averaging.
- `preprocess/` — data preprocessing utilities.
- `utils/` — misc helpers (logging, data loading, etc.).
