# Delta From Main

## Branch Baseline

- Branch: `dev/sleep2expert`
- Baseline commit: `21bbd67bc7b69dce4b119141cd201779688f016c`
- `main` currently points at the same commit; the branch delta is the working-tree `sleep2vec2` standalone recipe addition.

## Added Surfaces

- `sleep2vec2/`: standalone mirror of the base `sleep2vec` runtime with imports rewritten to the `sleep2vec2` namespace.
- `sleep2vec2/data/`: local copy of top-level data contracts.
- `sleep2vec2/preprocess/`: local copy of top-level preprocessing scripts and notebook.
- `sleep2vec2/visualization/assets/`: local copy of tracked visualization font assets used by the copied plotting theme.
- `configs/sleep2vec2/`: duplicated YAML recipe tree.
- `sleep2vec2/backbones/roformer/`: copied standalone RoFormer implementation from `/Users/zrjin/git/vendor/roformer_standalone`.
- `tests/test_sleep2vec2_namespace.py` and `tests/test_sleep2vec2_roformer_parity.py`: isolation and forward-parity coverage.

## Behavioral Contract

- Existing base `sleep2vec`, top-level `data`, and top-level `preprocess` behavior is unchanged.
- `sleep2vec2` keeps the same YAML schema, CLI shape, dataloader behavior, preset shape, and model class names as the base recipe.
- `backbone.name: roformer` inside `sleep2vec2` resolves to the package-local standalone `RoFormerEncoderModel`.
- LoRA is disabled for `sleep2vec2` finetune configs and rejected by `sleep2vec2.config` until standalone RoFormer PEFT compatibility is explicitly added.
- Existing Hugging Face checkpoint key translation is not added for `sleep2vec2`; parity scope is forward/API/numeric parity when weights are copied in tests.

## Validation Notes

- Static checks verified copied imports stay inside `sleep2vec2`.
- `configs/sleep2vec2/` was validated with the existing config checker and with `sleep2vec2.config` loaders.
- Runtime parity tests require an environment with `torch`, `transformers`, and `pytest`.
