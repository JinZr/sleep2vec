# Variant Surfaces

## Branch State

On commit `21bbd67bc7b69dce4b119141cd201779688f016c` plus the current working-tree delta, `sleep2vec2/` and `sleep2expert/` are active standalone mirror recipes. Each duplicates the base package runtime and keeps its data and preprocessing paths under its own namespace:

- `sleep2vec2/`: package-local copy of the base runtime
- `sleep2vec2/data/`: package-local copy of top-level `data/`
- `sleep2vec2/preprocess/`: package-local copy of top-level `preprocess/`
- `sleep2vec2/data/kaldi_io.py`, `sleep2vec2/data/kaldi_psg_dataset.py`, and `sleep2vec2/preprocess/convert_npz_to_kaldi.py`: package-local Kaldi backend mirror for `kaldi_native_io` storage
- `sleep2vec2/visualization/assets/`: package-local copy of tracked visualization font assets
- `configs/sleep2vec2/`: duplicated YAML recipe tree
- `sleep2vec2/backbones/roformer/`: copied standalone RoFormer implementation
- `sleep2expert/`: package-local copy of the `sleep2vec2` standalone runtime
- `sleep2expert/data/`: package-local data copy
- `sleep2expert/preprocess/`: package-local preprocessing copy
- `sleep2expert/data/kaldi_io.py`, `sleep2expert/data/kaldi_psg_dataset.py`, and `sleep2expert/preprocess/convert_npz_to_kaldi.py`: package-local Kaldi backend mirror for `kaldi_native_io` storage
- `sleep2expert/visualization/assets/`: package-local copy of tracked visualization font assets
- `configs/sleep2expert/`: duplicated YAML recipe tree
- `sleep2expert/backbones/roformer/`: copied standalone RoFormer implementation

The following variant roots still contain no tracked source files:

- `sleep2vec_moe/`
- `sleep2vec_hires/`

No file or directory symlinks were used for the `sleep2vec2` or `sleep2expert` mirrors.

## Indexed Function Status

- Indexed functions: summarized at surface level only in this branch index update.
- Key active surface: `<variant>.backbones.encoder_factory.build_roformer` resolves `backbone.name: roformer` to `<variant>.backbones.roformer.RoFormerEncoderModel`.
- Key isolation surface: copied runtime modules import `<variant>.*`, `<variant>.data.*`, and `<variant>.preprocess.*` instead of base `sleep2vec`, top-level `data`, top-level `preprocess`, or another variant namespace. The `sleep2vec2` and `sleep2expert` Kaldi backends follow this rule.
- Key finetune limitation: standalone variant configs keep LoRA disabled, and each variant `config.load_finetune_config` rejects enabled LoRA flags because standalone RoFormer PEFT compatibility is not part of the current contract.
- Key inference surface: `<variant>.infer` mirrors base `--inference-preset-path` behavior by overriding the effective finetune preset after YAML binding and recording the evaluated preset in `<variant>.results` CSV metadata.

## Reuse Guidance

- For `sleep2vec2` or `sleep2expert`, reuse the mirrored local implementation first:
  - `<variant>.registry`
  - `<variant>.builders`
  - `<variant>.backbones.encoder_factory`
  - `<variant>.config`
  - `<variant>.common`
  - `<variant>.checkpoints`
  - `<variant>.modules.tokenizers`
  - `<variant>.modules.projection`
  - `<variant>.data`
  - `<variant>.preprocess`
- For standalone Kaldi data, reuse the package-local `KaldiPSGDataset` and `convert_npz_to_kaldi` implementation, such as `sleep2vec2.data.kaldi_psg_dataset.KaldiPSGDataset` or `sleep2expert.data.kaldi_psg_dataset.KaldiPSGDataset`, rather than the top-level `data/` or `preprocess/` implementations.
- Preserve the standalone RoFormer replacement by keeping `<variant>.backbones.encoder_factory` pointed at `<variant>.backbones.roformer`.
- Preserve the reduced finetune contract by keeping LoRA disabled unless a future change adds explicit PEFT compatibility tests for standalone RoFormer.
- Do not add Hugging Face checkpoint key translation unless checkpoint compatibility becomes an explicit requirement.
- Do not treat `__pycache__` contents as authoritative code.

## Ownership Notes

- Variant-specific `sleep2vec2/` and `sleep2expert/` files are assigned to the variant-maintainer ownership boundary in repository policy.
- `sleep2vec_moe/` and `sleep2vec_hires/` have no explicit tracked-source owner on this branch because they have no tracked source files here.

## Unknowns

- Whether standalone variants should eventually support existing HF-named checkpoints is intentionally unresolved; the current contract covers forward/API parity only.
- Whether `sleep2vec_moe/` or `sleep2vec_hires/` real implementations exist on another branch is unknown.
