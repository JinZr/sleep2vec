# Variants And Routing Workflow

## Purpose

Work safely in the active standalone variants and export sleep2expert MoE routing summaries without crossing package namespaces.

## Active Variant Namespaces

- `sleep2vec2/`: dense standalone mirror with package-local runtime, data, preprocessing, visualization, survival/LSTM support, LoRA/DoRA adapter support, and standalone RoFormer code.
- `sleep2expert/`: standalone mirror with MoE RoFormer, MoE regularization, MoE finetune tuning, survival/LSTM support, LoRA/DoRA adapter support, checkpoint expansion, runtime route filtering, routing export, and compact subnetwork export.

## Package-Local Rule

When editing `sleep2vec2` or `sleep2expert`:

1. Keep imports inside the package namespace.
2. Keep `data/` and `preprocess/` behavior package-local.
3. Keep inference prediction extraction and CSV/manifest writing mirrored through package-local `sleep2vec_inference.py` and `results.py`.
4. Use root `sleep2vec` docs only for shared dense contracts, then apply variant-specific deltas from `FUNCTIONS/VARIANT_SURFACES.md`.
5. Run or update variant tests when behavior changes.

Primary namespace guards:

- `tests/variants/test_sleep2vec2_namespace.py`
- `tests/variants/test_sleep2expert_namespace.py`
- `tests/variants/test_variant_data_protocol.py`

## sleep2expert MoE Recipe Flow

1. Define MoE backbone fields under `model.backbone.moe`.
2. Validate schema through `sleep2expert.config.load_pretrain_config` or `load_finetune_config`.
3. Build sparse RoFormer layers through `sleep2expert.backbones.roformer.moe`.
4. During pretraining, consume `Sleep2vecPretrainModel.last_moe_aux` through `compute_moe_regularization`.
5. During finetuning, set `finetune.moe_tuning` for trainability groups, LR scales, and the supported downstream MoE regularization subset.
6. If LoRA is enabled, use attention target modules by default or opt into expert targets with `dense_in` / `dense_out`; router LoRA targets are unsupported.
7. Use `sleep2expert.checkpoints.load_pretrain_init_weights` for checkpoint initialization; it handles compatible dense-to-MoE expansion and rejects legacy standalone-incompatible RoFormer keys.

## Compact Subnetwork Export

Canonical entrypoint:

`python -m sleep2expert.export_subnetwork --config <yaml> --ckpt-path <ckpt> --output-dir <dir> --route-expert-groups <group> [<group> ...]`

This command rewrites a MoE config and checkpoint into a compact artifact containing only selected expert groups. It remaps expert ids into a dense `0..N-1` range, slices learned router weights, removes invalid optimizer/scheduler resume state, writes `config.yaml`, `model.ckpt`, `expert_id_map.csv`, and `manifest.json`, and rejects non-empty output directories before loading the checkpoint.

Use route filtering during inference when the model artifact should remain unchanged; use `export_subnetwork` when downstream work needs a smaller standalone config/checkpoint pair.

## Routing Export

Canonical entrypoint:

`python -m sleep2expert.routing_analysis --config <yaml> --ckpt-path <ckpt> --label-name <label> --output <csv>`

Primary code path:

1. `sleep2expert.routing_analysis.parse_args`
2. `sleep2expert.common.apply_finetune_config`
3. `sleep2expert.infer._build_inference_loader`
4. `sleep2expert.sleep2vec_finetuning.Sleep2vecFinetuning`
5. `_load_analysis_weights`
6. evaluation forward pass
7. `build_routing_rows`
8. `_write_rows`
9. optional `write_routing_heatmaps`

Important options:

- `--label-name`: required downstream label name used by the finetune config.
- `--pretrained-only`: export routing from a pretrained backbone without loading a downstream checkpoint.
- `--eval-split`: choose `train`, `val`, or `test`.
- `--override-dataset-names`: replace configured dataset names for the export.
- `--heatmap-dir`: write per-modality routing heatmap PNGs.
- `--wandb`: log routing heatmaps to W&B.
- `--avg-ckpts` / `--avg-ckpt-dir`: average concrete checkpoint files before export.

## Outputs

- Routing CSV with layer, modality, sample context, expert id/group, usage count, mean router probability, and router entropy.
- Optional routing heatmap PNGs under the requested heatmap directory.
- Optional W&B images under `routing_heatmap/<modality>`.

## Edit Hotspots

- Change namespace parity or mirror behavior: package-local files plus namespace/parity tests.
- Change automatic inference prediction export in a variant: package-local `infer.py`, `results.py`, `sleep2vec_inference.py`, and `sleep2vec_finetuning.py`.
- Change LoRA adapter parity in a variant: package-local `config.py`, `common.py`, `downstream_model.py`, `sleep2vec_finetuning.py`, and variant namespace tests.
- Change MoE schema: `sleep2expert/config.py`.
- Change sparse routing execution: `sleep2expert/backbones/roformer/moe.py`.
- Change MoE auxiliary losses or metrics: `sleep2expert/losses/moe_regularization.py`.
- Change MoE finetune trainability or optimizer grouping, including the `lora` group: `sleep2expert/sleep2vec_finetuning.py`.
- Change routing export row schema or heatmaps: `sleep2expert/routing_analysis.py`, `sleep2expert/visualization/routing_heatmap.py`.
- Change compact subnetwork export: `sleep2expert/export_subnetwork.py` and `tests/variants/test_sleep2expert_subnetwork_export.py`.
