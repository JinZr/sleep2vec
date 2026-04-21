# Variant Surfaces

## Branch State

On commit `825a30433e1f3d4cfcf6e4338cde5c29426411f3`, the following directories exist but contain no tracked source files:

- `sleep2vec2/`
- `sleep2vec_moe/`
- `sleep2vec_hires/`

No file or directory symlinks were found under these roots during index generation.

## Indexed Function Status

- Indexed functions: none
- Reason: there are no tracked Python source files to catalog on this branch

## Reuse Guidance

- If future work revives these variant packages, reuse the base package extension surface first:
  - `sleep2vec.registry`
  - `sleep2vec.builders`
  - `sleep2vec.backbones.encoder_factory`
  - `sleep2vec.config`
  - `sleep2vec.common`
  - `sleep2vec.checkpoints`
  - `sleep2vec.modules.tokenizers`
  - `sleep2vec.modules.projection`
- Do not treat `__pycache__` contents as authoritative code.

## Ownership Notes

- `sleep2vec_moe/` and variant-specific `sleep2vec2/` files are conceptually assigned to the variant-maintainer ownership boundary in repository policy.
- `sleep2vec_hires/` has no explicit tracked-source owner on this branch because it has no tracked source files here.

## Unknowns

- Whether real variant implementations exist on another branch is unknown.
- Whether the local bytecode artifacts reflect out-of-branch source is unknown.
