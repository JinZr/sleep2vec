# TODO: Build a MoE-based Sleep2Vec in a sibling package (sleep2vec_moe) while sharing pretrain.py/finetune.py

Goal: add an MoE backbone and any MoE-specific components under a new `sleep2vec_moe/` folder, reuse as much code as possible via soft links/imports, and keep using the existing `pretrain.py` and `finetune.py` entrypoints driven by YAML recipes.

## Plan
1) Create folder & reuse via symlinks
   - Add `sleep2vec_moe/` next to `sleep2vec/`.
   - Symlink modules that stay identical (e.g., `downstream/`, `losses/`, registries, common tokenizers/projection, builders).
   - Keep MoE-specific code (encoder/backbone or tokenizer variants) in dedicated files inside `sleep2vec_moe/`.

2) Register MoE backbone
   - Implement a builder in `sleep2vec_moe/encoder_factory.py` (or import into the shared registry) and register with `@register_backbone("moe_roformer")` (or similar).
   - Accept MoE hyperparams via `config_overrides` in YAML (e.g., `num_experts`, `top_k`, router type).
   - Ensure `.build()` returns `(encoder, hidden_size)` to match the existing factory contract.

3) Tokenizers (if needed)
   - If MoE needs special tokenizers, add and register them in `sleep2vec_moe/pretrain/tokenizers.py`; otherwise soft-link to the shared tokenizers.
   - Point the MoE YAML channels to these tokenizers; keep `out_dim` consistent across channels.

4) Projection/head tweaks (optional)
   - Register any MoE-specific projection in `pretrain/projection.py` (shared or MoE copy).
   - For downstream, add heads/fusion in `downstream/heads.py` and register (or reuse existing ones).

5) YAML recipes
   - Add `configs/sleep2vec_moe_pretrain.yaml` plus finetune YAMLs (cls/reg) pointing to the MoE backbone and tokenizers.
   - Keep training hyperparameters on the CLI; model/loss selections live in YAML.

6) Running
   - Pretrain: `python pretrain.py --config configs/sleep2vec_moe_pretrain.yaml ...`
   - Finetune: `python finetune.py --config configs/sleep2vec_moe_finetune_cls.yaml --label-name ... --results-csv-path ...`

7) Validation checklist
   - Verify MoE config_overrides align with the implementation.
   - Confirm all channels share the same `out_dim` and match the MoE hidden size.
   - Run a small smoke test to check shape/routing and that `pretrain.py`/`finetune.py` still work unchanged.
