# Variant Surfaces

## Branch State

On commit `dbe6a5e4cf40811138a35870b011a2a6d1bf8b83`, tracked variant coverage is:

- `sleep2vec2/`: active standalone dense mirror with 105 tracked files
- `sleep2expert/`: active standalone MoE-capable mirror with 109 tracked files

## `sleep2vec2` Standalone Mirror

- Files: `sleep2vec2/`, `configs/sleep2vec2/`, `tests/test_sleep2vec2_namespace.py`, `tests/test_sleep2vec2_roformer_parity.py`, `tests/test_sleep2vec2_kaldi_backend.py`
- Purpose and contract: keep a package-local dense recipe mirror whose runtime, data, preprocessing, visualization, LoRA/DoRA adapter behavior, and config imports stay under `sleep2vec2`.
- Important inputs/outputs: same pretrain/adapt/finetune/infer contracts as root `sleep2vec`, including package-local finetune imbalance loss/sampler schema, LoRA rank/alpha/dropout/target/use_dora settings, distributed-aware weighted metadata sampler, automatic inference prediction export, inference W&B artifacts, downstream specificity metrics, plus package-local Kaldi conversion and dataset routing.
- Side effects: runtime side effects mirror root entrypoints, including inference artifact writes and optional W&B artifact logging, but W&B projects use `sleep2vec2-*` names.
- Reuse guidance: edit the package-local implementation directly when working in `sleep2vec2`; do not shortcut through root `sleep2vec`, `data`, or `preprocess`.
- Duplication-risk notes: behavior parity is intentional duplication. Namespace-crossing imports and silent legacy RoFormer checkpoint compatibility are regressions.

## `sleep2expert` Standalone MoE Mirror

- Files: `sleep2expert/`, `configs/sleep2expert/`, `tests/test_sleep2expert_namespace.py`, `tests/test_sleep2expert_roformer_parity.py`, `tests/test_sleep2expert_moe_*.py`, `tests/test_sleep2expert_routing_analysis.py`
- Purpose and contract: keep a package-local mirror that adds MoE RoFormer layers, routing aux capture, pretrain MoE regularization, finetune MoE tuning, LoRA/DoRA adapter behavior, dense-to-MoE checkpoint expansion, and routing export.
- Important inputs/outputs: same dense runtime inputs as root for non-MoE recipes, including package-local finetune imbalance loss/sampler schema, LoRA rank/alpha/dropout/target/use_dora settings, distributed-aware weighted metadata sampler, automatic inference prediction export, inference W&B artifacts, downstream specificity metrics, and standalone RoFormer `model.backbone.attention_backend`; MoE recipes add `model.backbone.moe`, optional `finetune.moe_tuning` blocks, and optional expert LoRA targets `dense_in` / `dense_out`.
- Side effects: runtime side effects mirror root entrypoints, W&B projects use `sleep2expert-*` names, and routing analysis writes CSV/PNG artifacts.
- Reuse guidance: route schema changes through `sleep2expert.config`, sparse routing through `sleep2expert.backbones.roformer.moe`, MoE loss through `sleep2expert.losses.moe_regularization`, MoE LoRA optimizer grouping through `sleep2expert.sleep2vec_finetuning`, and persistent routing inspection through `sleep2expert.routing_analysis`.
- Duplication-risk notes: `last_moe_aux` is in-memory state only; use routing export for persistent analysis. Do not implement parallel router metrics inside trainers or ad hoc scripts. Router LoRA remains unsupported.

## Package-Local Data And Preprocess Mirrors

- Files: `sleep2vec2/data/`, `sleep2vec2/preprocess/`, `sleep2expert/data/`, `sleep2expert/preprocess/`
- Purpose and contract: preserve standalone recipe operation for data loading, Kaldi backends, preset generation, split assignment, and WatchPAT conversion.
- Important inputs/outputs: same `SampleIndex`, NPZ preset, Kaldi manifest, and batch contracts as root unless a package-local test says otherwise.
- Side effects: mirror root data/preprocessing file writes and optional `kaldi_native_io` usage.
- Reuse guidance: when changing root data/preprocess contracts that variants must share, update package-local mirrors and the variant parity tests in the same pass.
- Duplication-risk notes: root/variant converter and dataset behavior can drift silently if only one namespace is patched.

## Indexed Function Status

This file indexes variant-specific deltas, not every mirrored function. For common dense runtime behavior, use the root function catalogs first, then apply the package-local namespace rule.

Important variant-specific functions and classes:

- `sleep2vec2.backbones.roformer.RoFormerEncoderModel` and `sleep2expert.backbones.roformer.RoFormerEncoderModel`: standalone RoFormer parity surfaces with eager or SDPA attention backends.
- `sleep2vec2.downstream_model.Sleep2vecDownstreamModel` and `sleep2expert.downstream_model.Sleep2vecDownstreamModel`: package-local LoRA/DoRA insertion mirrors, including separate channel adapters.
- `sleep2vec2.infer._log_inference_outputs_to_wandb` and `sleep2expert.infer._log_inference_outputs_to_wandb`: package-local mirrors of root inference W&B artifact logging.
- `sleep2vec2.metrics.binary_specificity` and `sleep2expert.metrics.binary_specificity`: package-local mirrors of root binary specificity reporting.
- `sleep2expert.config.MoeConfig` and `_validate_moe_config`: MoE backbone schema and strict validation.
- `sleep2expert.config._build_finetune_moe_tuning_config`: MoE finetune mode, LR-scale including `lora`, and regularization parser.
- `sleep2vec2.sleep2vec_inference.*` and `sleep2expert.sleep2vec_inference.*`: package-local mirrors of root inference prediction extraction.
- `sleep2expert.backbones.roformer.moe.TopKRouter`: learned/random/hard router implementation.
- `sleep2expert.backbones.roformer.moe.SparseMoEFFN`: sparse expert FFN execution.
- `sleep2expert.losses.moe_regularization.compute_moe_regularization`: pretrain MoE auxiliary loss.
- `sleep2expert.losses.moe_regularization.compute_downstream_moe_regularization`: supported downstream MoE auxiliary loss subset.
- `sleep2expert.checkpoints.initialize_moe_from_dense_if_possible`: dense FFN to MoE expert checkpoint expansion.
- `sleep2expert.routing_analysis.run_routing_analysis`: persistent routing CSV and heatmap export.
- `sleep2expert.model_stats.*`: parameter, active-parameter, FFN FLOP, and expert-usage summaries.

## Ownership Notes

- `sleep2vec2/` is a dense standalone variant boundary. Keep it package-local and parity-tested, including LoRA/DoRA adapter parity.
- `sleep2expert/` is the active MoE standalone variant boundary. Keep MoE schema, routing, regularization, tuning, LoRA grouping, and export changes inside this namespace unless deliberately changing root contracts.

## Unknowns

- Package-local notebooks are not indexed as reusable source beyond workflow-level notes.
