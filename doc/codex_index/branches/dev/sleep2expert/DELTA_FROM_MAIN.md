# Delta From Main

## Branch Baseline

- Branch: `dev/sleep2expert`
- Baseline commit: `21bbd67bc7b69dce4b119141cd201779688f016c`
- `main` currently points at the same commit; the branch delta is the working-tree `sleep2vec2` and `sleep2expert` standalone recipe additions.

## Added Surfaces

- `sleep2vec2/`: standalone mirror of the base `sleep2vec` runtime with imports rewritten to the `sleep2vec2` namespace.
- `sleep2vec2/data/`: local copy of top-level data contracts.
- `sleep2vec2/preprocess/`: local copy of top-level preprocessing scripts and notebook.
- `sleep2vec2/visualization/assets/`: local copy of tracked visualization font assets used by the copied plotting theme.
- `configs/sleep2vec2/`: duplicated YAML recipe tree.
- `sleep2vec2/backbones/roformer/`: copied standalone RoFormer implementation from `/Users/zrjin/git/vendor/roformer_standalone`.
- `tests/test_sleep2vec2_namespace.py` and `tests/test_sleep2vec2_roformer_parity.py`: isolation and forward-parity coverage.
- `sleep2expert/`: standalone mirror of `sleep2vec2` with imports rewritten to the `sleep2expert` namespace.
- `sleep2expert/data/kaldi_io.py`, `sleep2expert/data/kaldi_psg_dataset.py`, and `sleep2expert/preprocess/convert_npz_to_kaldi.py`: package-local Kaldi backend mirror for `kaldi_native_io` storage.
- `configs/sleep2expert/`: duplicated YAML recipe tree.
- `tests/test_sleep2expert_namespace.py` and `tests/test_sleep2expert_roformer_parity.py`: isolation and parity coverage against `sleep2vec2`.
- `tests/test_sleep2expert_kaldi_backend.py`: package-local config, routing, dataset, and converter coverage for the Kaldi backend.

## Behavioral Contract

- Existing base `sleep2vec`, top-level `data`, and top-level `preprocess` behavior is unchanged.
- `sleep2vec2` keeps the same YAML schema, CLI shape, dataloader behavior, preset shape, and model class names as the base recipe.
- `sleep2expert` keeps behavior identical to `sleep2vec2`; namespace, config directory, and error-message recipe name are the only intended differences.
- `backbone.name: roformer` inside `sleep2vec2` resolves to the package-local standalone `RoFormerEncoderModel`.
- `backbone.name: roformer` inside `sleep2expert` resolves to the package-local standalone `RoFormerEncoderModel`.
- LoRA is disabled for standalone variant finetune configs and rejected by variant-local config loaders until standalone RoFormer PEFT compatibility is explicitly added.
- `sleep2expert` supports `data.backend: kaldi` through package-local config binding, dataset routing, reader pooling, and NPZ-to-Kaldi conversion; Kaldi flows use `manifest.csv`/`manifest.json` instead of legacy NPZ preset pickles.
- Existing Hugging Face checkpoint key translation is not added for standalone variants; parity scope is forward/API/numeric parity when weights are copied in tests.

## Validation Notes

- Static checks verified copied imports stay inside `sleep2vec2`.
- `configs/sleep2vec2/` was validated with the existing config checker and with `sleep2vec2.config` loaders.
- Static checks verify copied imports stay inside `sleep2expert`.
- `configs/sleep2expert/` is validated with `sleep2expert.config` loaders.
- Kaldi-specific runtime tests require `kaldi_native_io`; they should skip when that optional dependency is unavailable.
- Runtime parity tests require an environment with `torch`, `transformers`, and `pytest`.
