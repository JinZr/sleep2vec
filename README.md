# Sleep2Vec Refactor Quick Guide

This repo now separates **model/loss definition (YAML)** from **training hyperparameters (CLI flags)**. Use the examples in `configs/` as templates and follow the steps below to swap components.

## Running
- Pretrain: `python -m sleep2vec.pretrain --config configs/sleep2vec_dense_pretrain.yaml --epochs 120 --lr 5e-5 --devices 0 1`
- Finetune (classification): `python -m sleep2vec.finetune --config configs/sleep2vec_dense_finetune_cls.yaml --label-name stage5 --results-csv-path outputs.csv --epochs 50 --lr 1e-5`
- Finetune (regression): `python -m sleep2vec.finetune --config configs/sleep2vec_dense_finetune_reg.yaml --label-name age --results-csv-path outputs.csv --epochs 50 --lr 1e-5`
- Diagnostics-only run (no progress bar):  
  - Pretrain: `python -m sleep2vec.pretrain --config configs/sleep2vec_dense_pretrain.yaml --print-diagnostics --diagnostics-steps 5 --devices 0`  
  - Finetune: `python -m sleep2vec.finetune --config configs/sleep2vec_dense_finetune_cls.yaml --label-name stage5 --results-csv-path /tmp/out.csv --print-diagnostics --diagnostics-steps 5 --devices 0`

Only change CLI flags for training hyperparameters (epochs, lr, devices, etc.). All model/loss choices belong in YAML.

### Diagnostics mode (icefall-style tensor stats)
- Flags: `--print-diagnostics` enables hooks that capture activations, gradients, and parameter stats; `--diagnostics-steps` controls how many train steps to observe (default 5).
- Behavior: progress bar is disabled, validation and checkpointing are skipped, and training stops after the given steps. Stats print to stdout at the end.

> [!Important]
> ``--precision`` has to be set to `32` when using diagnostics mode, as mixed precision interferes with accurate stats collection and may cause unexpected behavior.


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
3. Run `python -m sleep2vec.pretrain --config your_pretrain.yaml ...`. Loss is no longer controlled via CLI.

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

## 6) Switch model averaging
1. Register a strategy with `@register_model_averager("my_avg")` in `sleep2vec/model_averaging.py` (EMA is registered by default).
2. Toggle it in pretrain YAML:
   ```yaml
   model_averaging:
     name: ema
     params:
       enabled: true
       base_momentum: 0.996
       final_momentum: 1.0
       use_for_eval: true
   ```
3. To load averaged weights downstream, pass the averaging name (e.g., `use_ema="ema"`) or `False` to fall back to student weights.
4. Icefall-style arithmetic running average is available as `name: running_mean`:
   ```yaml
   model_averaging:
     name: running_mean
     params:
       enabled: true
       average_period: 200   # update cadence in steps
       start_step: 200       # first update step (defaults to average_period)
       use_for_eval: true
   ```

## Tips
- Keep YAML per stage: `*_pretrain.yaml` includes `loss`; finetune YAML omits `loss` and only describes the model.
- All channels must share the same `out_dim`.
- Training hyperparameters stay on CLI; model/loss config stays in YAML.
