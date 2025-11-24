# Sleep2Vec Refactor Quick Guide

This repo now separates **model/loss definition (YAML)** from **training hyperparameters (CLI flags)**. Use the examples in `configs/` as templates and follow the steps below to swap components.

## Running
- Pretrain: `python -m sleep2vec.pretrain_main --config configs/sleep2vec_dense_pretrain.yaml --epochs 120 --lr 5e-5 --devices 0 1`
- Finetune (classification): `python finetune.py --config configs/sleep2vec_dense_finetune_cls.yaml --label-name stage5 --results-csv-path outputs.csv --epochs 50 --lr 1e-5`
- Finetune (regression): `python finetune.py --config configs/sleep2vec_dense_finetune_reg.yaml --label-name age --results-csv-path outputs.csv --epochs 50 --lr 1e-5`

Only change CLI flags for training hyperparameters (epochs, lr, devices, etc.). All model/loss choices belong in YAML.

## 1) Change backbone
1. Pick or create a backbone builder registered via `@register_backbone` in `sleep2vec/encoder_factory.py` (e.g., add `moe_roformer`).
2. Edit your YAML:
   ```yaml
   model:
     backbone:
       name: roformer            # switch to your new registry name
       hidden_size: 768
       num_hidden_layers: 12
       num_attention_heads: 16
       vocab_size: 1
       config_overrides: {}      # put extra kwargs here (e.g., MoE routing)
   ```
3. Point `--config` to this YAML and run. No CLI change needed for backbone selection.

## 2) Change tokenizer
1. Implement and register a tokenizer in `sleep2vec/pretrain/tokenizers.py` using `@register_tokenizer("my_tokenizer")`.
2. For each channel in YAML, set the tokenizer:
   ```yaml
   model:
     channels:
       - name: eeg_original
         input_dim: 3840
         out_dim: 768            # all channels must share out_dim
         tokenizer: my_tokenizer # registry name
         tokenizer_kwargs: {}    # optional extra args
   ```
3. Keep `out_dim` consistent across channels; the builder will enforce this.

## 3) Change projection head
1. Register a projection builder in `sleep2vec/pretrain/projection.py` with `@register_projection("my_proj")`.
2. Update YAML:
   ```yaml
   model:
     projection:
       name: my_proj
       enabled: true
       hidden_dim: 768
       out_dim: 256
       kwargs: {}        # optional extra args for your builder
   ```
3. Set `enabled: false` to disable projection entirely.

## 4) Change loss (pretrain contrastive)
1. Implement a loss in `sleep2vec/losses/` and register with `@register_loss("my_loss")`.
2. Edit pretrain YAML:
   ```yaml
   loss:
     name: my_loss
     temperature: 0.2
     params:
       # custom kwargs passed to your loss
   ```
3. Run `python -m sleep2vec.pretrain_main --config your_pretrain.yaml ...`. Loss is no longer controlled via CLI.

## 5) Change downstream model architecture
Two options:
1. **Adjust head settings in YAML** (no code change):
   - Classification head: set `model.head.name: classification`, `agg: gated_scalar|mean|concat`, `dropout`, `hidden_dim`.
   - Regression head: set `model.head.name: regression`, choose `agg`, `hidden_dim`, `dropout`.
   Example:
   ```yaml
   model:
     head:
       name: classification
       agg: gated_scalar
       dropout: 0.1
       hidden_dim: null
   ```
2. **Add a new head class**:
   - Implement in `sleep2vec/downstream/heads.py` and register with `@register_head("my_head")`.
   - Reference it in finetune YAML: `model.head.name: my_head` plus any kwargs in `model.head.kwargs`.

If you need multi-branch fusion changes, extend `FeatureFusion` or create a new head that performs custom fusion, then register it.

## Tips
- Keep YAML per stage: `*_pretrain.yaml` includes `loss`; finetune YAML omits `loss` and only describes the model.
- All channels must share the same `out_dim`.
- Training hyperparameters stay on CLI; model/loss config stays in YAML.
