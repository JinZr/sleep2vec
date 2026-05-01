# Variant Surfaces

## Branch State

On commit `21bbd67bc7b69dce4b119141cd201779688f016c`, `sleep2vec2/` is an active standalone mirror recipe. It duplicates the base package runtime and keeps its data and preprocessing paths under the `sleep2vec2` namespace:

- `sleep2vec2/`: package-local copy of the base runtime
- `sleep2vec2/data/`: package-local copy of top-level `data/`
- `sleep2vec2/preprocess/`: package-local copy of top-level `preprocess/`
- `sleep2vec2/visualization/assets/`: package-local copy of tracked visualization font assets
- `configs/sleep2vec2/`: duplicated YAML recipe tree
- `sleep2vec2/backbones/roformer/`: copied standalone RoFormer implementation

The following variant roots still contain no tracked source files:

- `sleep2vec_moe/`
- `sleep2vec_hires/`

No file or directory symlinks were used for the `sleep2vec2` mirror.

## Indexed Function Status

- Indexed functions: summarized at surface level only in this branch index update.
- Key active surface: `sleep2vec2.backbones.encoder_factory.build_roformer` now resolves `backbone.name: roformer` to `sleep2vec2.backbones.roformer.RoFormerEncoderModel`.
- Key isolation surface: copied runtime modules import `sleep2vec2.*`, `sleep2vec2.data.*`, and `sleep2vec2.preprocess.*` instead of base `sleep2vec`, top-level `data`, or top-level `preprocess`.
- Key finetune limitation: `sleep2vec2` configs keep LoRA disabled, and `sleep2vec2.config.load_finetune_config` rejects enabled LoRA flags because standalone RoFormer PEFT compatibility is not part of the current contract.

## Reuse Guidance

- For `sleep2vec2`, reuse the mirrored local implementation first:
  - `sleep2vec2.registry`
  - `sleep2vec2.builders`
  - `sleep2vec2.backbones.encoder_factory`
  - `sleep2vec2.config`
  - `sleep2vec2.common`
  - `sleep2vec2.checkpoints`
  - `sleep2vec2.modules.tokenizers`
  - `sleep2vec2.modules.projection`
  - `sleep2vec2.data`
  - `sleep2vec2.preprocess`
- Preserve the standalone RoFormer replacement by keeping `sleep2vec2.backbones.encoder_factory` pointed at `sleep2vec2.backbones.roformer`.
- Preserve the reduced finetune contract by keeping LoRA disabled unless a future change adds explicit PEFT compatibility tests for standalone RoFormer.
- Do not add Hugging Face checkpoint key translation unless checkpoint compatibility becomes an explicit requirement.
- Do not treat `__pycache__` contents as authoritative code.

## Ownership Notes

- Variant-specific `sleep2vec2/` files are assigned to the variant-maintainer ownership boundary in repository policy.
- `sleep2vec_moe/` and `sleep2vec_hires/` have no explicit tracked-source owner on this branch because they have no tracked source files here.

## Unknowns

- Whether `sleep2vec2` should eventually support existing HF-named checkpoints is intentionally unresolved; the current contract covers forward/API parity only.
- Whether `sleep2vec_moe/` or `sleep2vec_hires/` real implementations exist on another branch is unknown.
