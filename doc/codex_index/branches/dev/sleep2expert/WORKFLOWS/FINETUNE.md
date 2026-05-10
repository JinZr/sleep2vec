# Finetune Workflow

## Purpose

Attach a downstream head to the shared backbone, optionally load pretrained weights, optionally insert LoRA, then train and evaluate on train/val/test splits.

## Entry Command

Canonical entrypoint: `python -m sleep2vec.finetune --config ... --label-name ...`

Primary code path:

1. `sleep2vec.common.apply_finetune_config`
2. `sleep2vec.finetune.build_version_name`
3. `sleep2vec.finetune.supervised`
4. `sleep2vec.utils.get_finetune_dataloaders`
5. `sleep2vec.sleep2vec_finetuning.Sleep2vecFinetuning`
6. `sleep2vec.downstream_model.Sleep2vecDownstreamModel`
7. Lightning `trainer.fit(...)` then `trainer.test(...)`

## Detailed Flow

1. Load and bind finetune YAML.
   - Parse typed config bundle with `load_finetune_config`.
   - Copy channels, data paths, backend settings, task semantics, LoRA flags, and eval-visualization config into `args`.
   - Reject mismatched `data.data_channel_names`.
   - Reject Kaldi configs that still point at legacy NPZ preset pickles.
2. Resolve version name.
   - Prefer `--version-name`.
   - Otherwise derive from label, channel selection, few-shot mode, pretrained-vs-scratch, and optional tag.
3. Persist run artifacts.
   - `log-finetune/<version>/config.yaml`
   - `log-finetune/<version>/cli_args.yaml`
   - `log-finetune/<version>/moe_finetune_status.json` for `sleep2expert` MoE fine-tune status.
4. Build train/val/test loaders.
   - Always uses `allow_missing_channels=False`.
   - Selects `PSGPretrainDataset` for NPZ or `KaldiPSGDataset` for Kaldi.
   - Built-in sequence tasks append runtime label channels to `dataset_channel_names`.
   - `stage3` and `stage4` still pull raw labels from `stage5`.
   - `ahi` adds `ahi` as the primary token label source and `stage5` as an auxiliary label source.
5. Instantiate `Sleep2vecFinetuning`.
   - Creates `Sleep2vecPretrainModel` backbone.
   - Wraps it in `Sleep2vecDownstreamModel`.
   - Optionally loads pretrained backbone checkpoint.
   - Optionally freezes backbone and inserts LoRA adapters.
   - Optionally freezes tokenizers.
   - For `sleep2expert` configs with `finetune.moe_tuning`, applies the downstream MoE trainability policy after checkpoint/LoRA/tokenizer setup and before model averaging.
   - Optionally attaches a model averager.
   - Optionally enables downstream evaluation visualizations.
6. Fit.
   - Monitor comes from task semantics in `apply_task_flags`.
   - Best checkpoint is copied to `best.ckpt` when available.
7. Test.
   - Uses best model after training, or `--ckpt-path` when `epochs == 0`.
   - Result metrics append via `sleep2vec.results.save_result_csv`.

## Label Semantics

Built-in labels:

- `stage3`: classification, `output_dim=3`, sequence prediction, source labels from `stage5`
- `stage4`: classification, `output_dim=4`, sequence prediction, source labels from `stage5`
- `stage5`: classification, `output_dim=5`, sequence prediction
- `ahi`: multilabel token prediction with `output_dim=30`, validation/test reduced through event-level AHI metrics and a fitted threshold
- `sex`: classification, `output_dim=2`, non-sequence
- `age`: regression, `output_dim=1`, non-sequence

Built-in `age` and `sex` loaders reject presets/indexes missing valid labels after split/source filtering. This prevents stage/AHI-only presets with omitted or dummy `age`/`sex` metadata from being used for those tasks.

Custom labels require `finetune.task` in YAML.

## Important Runtime Decisions

- Task semantics are enforced before loaders are built.
- Kaldi finetune uses `data.kaldi_data_root` and `data.kaldi_manifest` from YAML; `data.finetune_preset_path` must be null.
- CLS vs token downstream behavior is defined by `model.cls`, not by folder naming in `configs/`.
- Layer mix is applied inside `Sleep2vecDownstreamModel`, not in the trainer.
- `sleep2expert` downstream MoE tuning is opt-in through `finetune.moe_tuning`; absent configs retain the legacy finetune trainability and two-group optimizer behavior.
- Canonical `sleep2expert` MoE downstream recipes live under `configs/sleep2expert/moe/`: conservative router-frozen classification/regression configs, a head-only few-shot probe, and `finetune_ablations/` for router-trainable and top-layer expert-only policies.
- `sleep2expert.finetune` defaults the base `--lr` to `1e-4`; downstream MoE `lr_scales` multiply that base, so conservative recipes use head LR `1e-4` and backbone/expert LR `1e-5` unless overridden on the CLI.
- Downstream train-time MoE aux collection is off by default and turns on only for enabled `finetune.moe_tuning.moe_regularization`; the supported supervised auxiliary loss is router z-loss only.
- `sleep2expert` fine-tune logs a run-start MoE status snapshot to W&B and local JSON, including MoE architecture, tuning mode, LR scales, aux settings, and actual group trainability.
- Validation/test forwards reuse eval-time `last_moe_aux` to log scalar `*_downstream_moe_*` diagnostics; detailed token-level routing stays in `python -m sleep2expert.routing_analysis`.
- AHI validation fits and stores an `ahi_eval_threshold` inside the checkpoint; test and inference require that threshold.
- Confusion matrices, ROC curves, and regression scatter plots are logged from `Sleep2vecFinetuning`, not from entrypoints.

## Outputs

- Checkpoints under `log-finetune/<version>/checkpoints/`
- Stable `best.ckpt` copy when training ran and a best checkpoint exists
- MoE fine-tune status JSON at `log-finetune/<version>/moe_finetune_status.json`
- Optional results CSV row via `sleep2vec.results.save_result_csv`
- W&B run under project `sleep2expert-finetune`

## Edit Hotspots

- Change task semantics: `sleep2vec/common.py`, `sleep2vec/config.py`
- Change head/layer-mix/LoRA behavior: `sleep2vec/downstream_model.py`, `sleep2vec/downstreams/`
- Change per-stage loss/metrics aggregation or AHI threshold behavior: `sleep2vec/sleep2vec_finetuning.py`, `sleep2vec/metrics.py`
- Change finetune data loader or built-in label-channel wiring: `sleep2vec/utils.py`, `data/default_dataset.py`, `data/utils.py`
- Change Kaldi finetune loading: `sleep2vec/common.py`, `sleep2vec/utils.py`, `data/kaldi_psg_dataset.py`
