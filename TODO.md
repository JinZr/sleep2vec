# TODO: Adding a MoE-based Sleep2Vec model (sleep2vec_moe) while reusing pretrain.py/finetune.py

Goal: introduce a MoE Transformer backbone and any bespoke heads/tokenizers under a new recipe directory (e.g., `configs/sleep2vec_moe_*`), without changing `pretrain.py`/`finetune.py` logic.

## 1) Backbone (MoE) – registry hook
- Add MoE implementation and register it:
  - File: `sleep2vec/encoder_factory.py` (or new module imported there)
  - Decorate a builder with `@register_backbone("moe_roformer")` (or your chosen name).
  - The builder should accept `BackboneConfig` and return a `TransformerEncoderFactory`-like object exposing `.build() -> (encoder, hidden_size)`.
  - Support MoE hyperparams via `config_overrides` in YAML (e.g., `num_experts`, `top_k`, router settings).
- Ensure the MoE model is PEFT-compatible if you plan to use LoRA in finetune.

## 2) Tokenizers (optional)
- If MoE requires special input projections, register new tokenizers in `sleep2vec/pretrain/tokenizers.py` via `@register_tokenizer("my_moe_tok")`.
- Update channel entries in the MoE YAML to use the new tokenizer and `out_dim` matching the MoE hidden size.

## 3) Projection head (optional)
- If contrastive projection differs, register a new projection in `sleep2vec/pretrain/projection.py` with `@register_projection("my_moe_proj")`.
- Reference it in MoE pretrain YAML under `model.projection`.

## 4) Downstream head/fusion (optional)
- If MoE outputs need a custom fusion/head, add a new head in `sleep2vec/downstream/heads.py` and register with `@register_head("my_moe_head")`.
- In MoE finetune YAML, set `model.head.name: my_moe_head` and supply kwargs in `model.head.kwargs`.

## 5) YAML recipes
- Create configs (examples):
  - `configs/sleep2vec_moe_pretrain.yaml`:
    ```yaml
    model:
      backbone:
        name: moe_roformer
        hidden_size: 768
        num_hidden_layers: 12
        num_attention_heads: 16
        config_overrides:
          num_experts: 4
          top_k: 2
      projection:
        name: simclr
        enabled: true
        hidden_dim: 768
        out_dim: 128
      channels:
        - name: eeg_original
          input_dim: 3840
          out_dim: 768
          tokenizer: sundial
    loss:
      name: weighted_info_nce
      temperature: 0.2
      params:
        hard_scale: 0.1
    ```
  - `configs/sleep2vec_moe_finetune_cls.yaml` and `_reg.yaml`: same `model` block, adjust `model.head` for classification/regression.
- Keep training hyperparameters on CLI (epochs, lr, devices, etc.).

## 6) Reuse pretrain/finetune entrypoints
- Pretrain: `python pretrain.py --config configs/sleep2vec_moe_pretrain.yaml --epochs ... --lr ...`
- Finetune: `python finetune.py --config configs/sleep2vec_moe_finetune_cls.yaml --label-name ... --results-csv-path ...`
- The entrypoints will auto-copy the YAML into the run directory; no code changes needed if registry names match YAML.

## 7) Validation checklist
- Ensure YAML `channels` share the same `out_dim` matching MoE hidden size.
- Confirm `config_overrides` aligns with your MoE implementation signature.
- Run a small sanity experiment to verify shapes, router behavior, and that LoRA (if used) still targets attention/FFN modules as expected.
