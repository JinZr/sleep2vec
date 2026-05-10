# sleep2vec

<div align="center">
  <img src="doc/image/banner.png" width="550"/>
  <p><strong>Modular self-supervised learning for sleep signals</strong></p>
  <p>Models and losses live in YAML; training hyperparameters stay on the CLI.</p>
</div>

---

**Quick links** · `configs/` recipes · `sleep2vec/` source · `data/` datasets & loaders · `preprocess/` caching scripts · `utils/` helpers

---

## Table of Contents
- [sleep2vec](#sleep2vec)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Setup](#setup)
  - [Data Format \& Caches](#data-format--caches)
  - [Kaldi Backend Recipes](#kaldi-backend-recipes)
  - [Sleep2Wave Kaldi Backend Recipes](#sleep2wave-kaldi-backend-recipes)
  - [Quick Start](#quick-start)
    - [Pretrain (contrastive)](#pretrain-contrastive)
    - [Adaptation — staged wearable expansion](#adaptation--staged-wearable-expansion)
    - [Finetune — classification](#finetune--classification)
    - [Finetune — regression](#finetune--regression)
  - [Inference Only](#inference-only)
  - [Configuration Knobs](#configuration-knobs)
  - [Diagnostics Mode](#diagnostics-mode)
  - [Working Tips](#working-tips)
  - [Repository Layout](#repository-layout)

---

## Overview
- Refactored training flow: **YAML defines architecture & loss**, **CLI sets training hyperparameters** (epochs, lr, devices, etc.).
- Supports contrastive pretraining, staged modality adaptation for new sensors, plus downstream classification or regression finetuning.
- Extensible registries for backbones, tokenizers, projection heads, losses, model averaging, LoRA-backed heads, and downstream heads.
- Dataset channel names and per-token input widths now come from YAML `model.channels`, so custom modalities such as wearable `ppg` or `actigraphy_vm` can be added without editing the dataset registry.
- WandB logging is enabled by default; inference-only runner is included for evaluating checkpoints.

---

## Setup
- Python 3.10+ with CUDA GPUs recommended; PyTorch/Lightning versions are pinned in `requirements.txt` (`torch==2.7.0`, `pytorch-lightning==2.6.1`).
- Install: `pip install -r requirements.txt` (choose the correct PyTorch wheel for your CUDA version).
- Pair-accuracy heatmap logging uses `matplotlib` + `seaborn` (already included in `requirements.txt`).
- Authenticate to Weights & Biases before running (`WANDB_API_KEY=...` or `WANDB_MODE=offline`) because entrypoints call `wandb.login()`.
- Default precision is bf16/bf16-mixed; pass `--precision 32` if your GPUs do not support bf16.
- Main entrypoints: `python -m sleep2vec.pretrain ...`, `python -m sleep2vec.adapt --phase stage1|stage2 ...`, `python -m sleep2vec.finetune ...`, `python -m sleep2vec.infer ...`.

---

## Data Format & Caches
- **Index CSV** (used by pretrain/finetune): required columns `path`, `split` (`train|val|test`), and `duration` (seconds). `age` and `sex` are optional for stage/AHI-only workflows, but built-in `age`/`sex` tasks require valid labels after split/source filtering; generate those presets from indexes carrying the corresponding real column. Optional extra label columns (e.g., disease flags) are consumed when `meta_data_names` is set. For the built-in `sex` task, the normalized contract is `sex=female|male`, encoded as `0=female`, `1=male`. If your source metadata uses `sexM`, convert it to `sex` with `1 -> male`, `0 -> female` during preprocessing.
- **NPZ contents per row**: every non-label key used at runtime must be declared in YAML `model.channels` with a matching `name` and `input_dim` (frames per token). Built-in examples include `heartbeat`, `breath`, `eeg_original`, `ecg_original`, `eog_original`, `emg_original`, `spo2`, `resp_original`, and `resp_nasal_original`; this branch also ships wearable examples for `ppg` and `actigraphy`. In this repo, `actigraphy` stores vector magnitude (VM). `stage5` remains a special per-token label channel and always uses width `1`. Built-in `ahi` additionally requires a flat 1 Hz `ah_event` array plus scalar NPZ keys `ahi` and `tst`.
- **Preset pickles**: both CLIs expect a precomputed pickle of `SampleIndex` objects (see `preprocess/save_dataset_presets.py`). Point `--pretrain-preset-path` / YAML `data.finetune_preset_path` to an existing pickle; these scripts do **not** fall back to CSV when a path is provided. Preset generation now requires a YAML config so the script can resolve channel names and `input_dim` values from `model.channels`.
- `preprocess/save_dataset_presets.py` also honors an optional top-level `preset_build` block. Use it when preset validation must differ from runtime input modalities, for example token-level PPG staging should validate both `ppg` and `stage5`.
- `preset_build.required_channels` is the YAML source of truth for preset validation channels; do not combine it with CLI `--channels`.
- `preset_build` must define both `required_channels` and `min_channels`; `preset_build.min_channels` overrides CLI `--min-channels` for preset generation only.
- To build presets, run:
  ```bash
  python preprocess/save_dataset_presets.py \
    --config configs/sleep2vec_dense_pretrain.yaml \
    --index /path/to/index.csv \
    --dataset-name shhs \
    --n-tokens 1535 \
    --split train val test
  ```
  Optional flags: `--channels eeg_original ecg_original`, `--meta-data-names hypertension diabetes`, `--include-no-metadata`, `--output-template 'data/{dataset}_{split}_preset_{tokens}{meta_suffix}.pickle'`, `--dry-run`, `--overwrite`.
  Explicit `--meta-data-names` values remain strict: the CSV must contain each requested metadata column except built-in AHI summaries, which come from NPZ.
- `--channels` is now an optional ordered subset of YAML `model.channels`; any requested channel that is missing from the YAML or lacks `input_dim` fails fast.
- **Missing-channel pretrain**: if you enable `--allow-missing-channels`, presets must carry `payload["available_channels"]` (auto-populated during preset creation) so the bucketed sampler can group by montage.
- **WatchPAT `.zzp` conversion**: `preprocess/watchpat_zzp_to_edf.py` converts a WatchPAT archive (`Sleep.dat`, `Patient.dat`, `log.dat`) into EDF for downstream inspection or external preprocessing. Example:
  ```bash
  python preprocess/watchpat_zzp_to_edf.py \
    /path/to/study.zzp \
    /path/to/study.edf \
    --writer auto \
    --json-summary /path/to/study_summary.json
  ```
  Batch conversion example:
  ```bash
  python preprocess/watchpat_zzp_to_edf.py \
    /path/to/zzp_dir \
    /path/to/edf_dir \
    --recursive \
    --skip-existing
  ```
  Batch mode shows a file-level `tqdm` progress bar. Optional flags: `--include-internal-1hz`, `--no-pulse-rate`, `--verbose`, `--json-summary /path/to/summary_dir`. `pyedflib` is used when available; otherwise the script falls back to its built-in manual EDF writer.

---

## Kaldi Backend Recipes
Kaldi storage uses `manifest.csv` as the preset-equivalent artifact. Do not pass legacy NPZ preset pickles with `backend: kaldi`; pre-windowed per-sample/channel matrices are read by `sample_key`, and `token_start` is preserved in the manifest for downstream aggregation.

Pretrain conversion should use model input channels only unless you intentionally want label-like channels in contrastive training:
```bash
python -m preprocess.convert_npz_to_kaldi \
  --index /path/to/index.csv \
  --config configs/sleep2vec_dense_pretrain.yaml \
  --output-dir /data/sleep2vec_kaldi/pretrain_120 \
  --max-tokens 120 \
  --stride-tokens 120 \
  --channels-from-config
```
For heterogeneous datasets, add `--allow-missing-channels --min-channels 2` during conversion and training so pair-first sampling uses `available_channels` from the manifest:
```bash
python -m sleep2vec.pretrain \
  --config configs/sleep2vec_dense_pretrain.yaml \
  --version-name kaldi-pretrain-120 \
  --data-backend kaldi \
  --kaldi-data-root /data/sleep2vec_kaldi/pretrain_120 \
  --kaldi-manifest /data/sleep2vec_kaldi/pretrain_120/manifest.csv \
  --pretrain-preset-path null \
  --allow-missing-channels --min-channels 2
```

Finetune and inference select Kaldi from the YAML `data` block:
```yaml
data:
  backend: kaldi
  kaldi_data_root: /data/sleep2vec_kaldi/ppg_stage5_1535
  kaldi_manifest: /data/sleep2vec_kaldi/ppg_stage5_1535/manifest.csv
  finetune_preset_path: null
```
Convert finetune roots with model channels plus required label channels. For `stage3`, `stage4`, or `stage5`, include `stage5`; for `ahi`, include both `ahi` and `stage5`, and ensure manifest rows contain scalar `ahi` and `tst` metadata. Match the converter windowing to the finetune config; current whole-night PPG configs use `max_tokens: 1535`, so convert with `--max-tokens 1535 --stride-tokens 0`:
```bash
python -m preprocess.convert_npz_to_kaldi \
  --index /path/to/index.csv \
  --config configs/ppg_stage5_finetune.yaml \
  --output-dir /data/sleep2vec_kaldi/ppg_stage5_1535 \
  --max-tokens 1535 \
  --stride-tokens 0 \
  --channels-from-config \
  --extra-channels stage5

python -m preprocess.convert_npz_to_kaldi \
  --index /path/to/index.csv \
  --config configs/ppg_ahi_finetune.yaml \
  --output-dir /data/sleep2vec_kaldi/ppg_ahi_1535 \
  --max-tokens 1535 \
  --stride-tokens 0 \
  --channels-from-config \
  --extra-channels ahi stage5
```
Inference reuses the same Kaldi root and manifest windowing as the checkpoint's finetune configuration. Keep `--avg-ckpts 1` for built-in `ahi`, because its evaluation threshold is checkpoint-specific.

---

## Sleep2Wave Kaldi Backend Recipes
The active `recipe: sleep2wave` autoencoder, diffusion, and generation stack defaults to NPZ presets. Existing tiny/medium YAMLs and `sleep2wave_train.sh` stay on that path unless you explicitly set `data.backend: kaldi`.

Convert NPZ waveform indexes with the package-local converter:
```bash
python -m sleep2wave.preprocess.convert_npz_to_kaldi \
  --index /path/to/sleep2wave_index.csv \
  --config configs/sleep2wave/sleep2wave_autoencoder_medium.yaml \
  --output-dir /data/sleep2wave_kaldi/medium_15e \
  --stride-epochs 15
```

Then opt a sleep2wave YAML into Kaldi:
```yaml
data:
  backend: kaldi
  kaldi_data_root: /data/sleep2wave_kaldi/medium_15e
  kaldi_manifest: /data/sleep2wave_kaldi/medium_15e/manifest.csv
  context_epochs: 15
```

For `backend: kaldi`, do not set `preset_path` or `index`. Generation also uses the YAML data backend unless `--preset-path` or `--index` is passed, which remains an NPZ override.

---

## Quick Start

> Update the dataset paths in the commands below (`--pretrain-data-index`, `--pretrain-preset-path`, and YAML `data.finetune_*`) to point to your own CSV/pickle files.

### Pretrain (contrastive)
```bash
python -m sleep2vec.pretrain \
  --config configs/sleep2vec_dense_pretrain.yaml \
  --pretrain-data-index /path/to/index.csv \
  --pretrain-preset-path /path/to/pretrain_cache.pkl \
  --version-name exp001 \
  --epochs 120 --lr 5e-5 --batch-size 320 \
  --devices 0 1 --num-workers 8
```
Optional:
- `--warmup-steps N` to override the default LR warmup (3% of total steps).
- `--pretrained-backbone-path /path/to/base.ckpt` to initialize the pretrain model from an existing checkpoint. Loader prefers `ema_model.` weights and falls back to `model.`; if `--ckpt-path` is also set, Lightning resume takes precedence.
- `--allow-missing-channels` to accept samples with missing channels; pair with `--min-channels` and (recommended) `--bucket-by-available-channels`.

Example (missing-channel pretrain):
```bash
python -m sleep2vec.pretrain \
  --config configs/sleep2vec_dense_pretrain.yaml \
  --pretrain-data-index /path/to/index.csv \
  --pretrain-preset-path /path/to/pretrain_cache.pkl \
  --version-name exp001-missing-chn \
  --epochs 120 --lr 5e-5 --batch-size 320 \
  --devices 0 1 --num-workers 8 \
  --allow-missing-channels --min-channels 6 --bucket-by-available-channels
```

### Adaptation — staged wearable expansion
`sleep2vec.adapt` runs a two-phase pretrain-style adaptation loop for newly introduced modalities while reusing an existing backbone checkpoint. The YAML must include a top-level `adapt:` block and the `adapt.new_channels` names must also appear in `model.channels`.

Stage 1: train only the new-modality tokenizers (and optionally the shared projection head).
```bash
python -m sleep2vec.adapt \
  --config configs/sleep2vec_dense_adapt_ppg_actigraphy.yaml \
  --phase stage1 \
  --pretrained-backbone-path /path/to/base_pretrain.ckpt \
  --pretrain-data-index /path/to/index.csv \
  --pretrain-preset-path /path/to/wearable_cache.pkl \
  --version-name wearable-v1 \
  --epochs 40 --lr 5e-5 --batch-size 256 \
  --devices 0 1
```

Stage 2: initialize from the stage-1 checkpoint, unfreeze the encoder/legacy tokenizers, and anneal the training pair distribution back toward legacy pairs.
```bash
python -m sleep2vec.adapt \
  --config configs/sleep2vec_dense_adapt_ppg_actigraphy.yaml \
  --phase stage2 \
  --pretrained-backbone-path /path/to/stage1.ckpt \
  --version-name wearable-v1 \
  --epochs 60 --lr 2e-5 --batch-size 256 \
  --devices 0 1
```

Notes:
- Provided example configs: `configs/sleep2vec_dense_adapt_ppg_actigraphy.yaml` and `configs/sleep2vec_dense_adapt_ppg_actigraphy_cls.yaml`.
- Adaptation defaults to missing-channel-aware pair-first sampling (`--allow-missing-channels` on, `--train-pair-monitor-enable` on) because wearable datasets often have heterogeneous sensor availability.
- `adapt.stage2.pair_schedule` is defined as training-progress fractions (`until` in `(0, 1]`) and must end at `1.0`.
- Starting a fresh adapt stage1 run uses `--pretrained-backbone-path` with a base pretrain checkpoint.
- Starting stage2 uses `--pretrained-backbone-path` with a prior adapt stage1 checkpoint; this reuses the existing experiment directory and W&B id but writes new checkpoints under `log-adapt/<run_name>/checkpoints.stage2`, restarting optimizer/scheduler/epoch state from zero.
- Fresh stage2 transition refuses to reuse a non-empty `checkpoints.stage2` directory; resume the old stage2 with `--ckpt-path`, or clear/move the old stage2 checkpoints first.
- Resuming an exact in-progress adapt run uses `--ckpt-path`; this is the only path that forwards a checkpoint into Lightning resume.

### Finetune — classification
```bash
python -m sleep2vec.finetune \
  --config configs/sleep2vec_dense_finetune_cls.yaml \
  --label-name stage5 --results-csv-path outputs.csv \
  --version-name exp001-stage5 \
  --epochs 50 --lr 1e-5 --devices 0 1
```
Notes:
- Built-in sleep-staging labels are `stage3`, `stage4`, and `stage5`. They are all **per-token sequence labeling** tasks (`is_seq=True`) and use token-level downstream (`model.cls.downstream: tokens`).
- `stage3` merges raw `stage5` labels into `W / NREM / REM`; `stage4` merges raw `stage5` labels into `W / N1N2 / N3 / REM`.
- Do **not** add `stage5` or `ahi` to `data.data_channel_names`; built-in sequence labels are loaded automatically into `batch["tokens"][...]` whenever `--label-name` is `stage3`, `stage4`, `stage5`, or `ahi`.
- Built-in `sex` classification is a metadata task with class order `["female", "male"]`, so targets are encoded as `0=female`, `1=male`; presets missing valid `sex` labels are rejected.
- `--pretrained-backbone-path /path/to/pretrain_or_adapt.ckpt` can be used to bootstrap downstream training from a pretrain/adaptation checkpoint; loader prefers `ema_model.` and falls back to `model.`.

### Finetune — regression
```bash
python -m sleep2vec.finetune \
  --config configs/sleep2vec_dense_finetune_reg.yaml \
  --label-name age --results-csv-path outputs.csv \
  --version-name exp001-age \
  --epochs 50 --lr 1e-5 --devices 0 1
```

Custom metadata labels:
- Built-in `age` regression requires valid `age` metadata; stage/AHI-only presets that omit or carry dummy `age`/`sex` labels are not valid for `--label-name age|sex`.
- Set `--label-name` to the CSV column name (e.g., `bmi`) and add a `finetune.task` block in the YAML to define task semantics (type/output_dim/is_seq/monitor/monitor_mod).
- Use the same `--label-name` for `sleep2vec.infer` (required) when evaluating custom tasks.
- Token-level labels (`is_seq: true`) are only supported for built-in sequence labels (`stage3`, `stage4`, `stage5`, `ahi`) unless you extend the dataloader.
- Built-in `ahi` expects a flat 1 Hz NPZ array named `ah_event`; each 30-second token is reshaped into 30 binary labels and trained with sigmoid/BCE.
- Built-in `ahi` also requires scalar NPZ keys `ahi` and `tst`; final validation/test/infer metrics are **not** pointwise second-level metrics.
- The built-in `ahi` evaluator fits the event threshold on validation only, saves it in the checkpoint, reuses it for test/infer, and then uses split post-processing: event detection metrics (`ahi_event_precision/recall/f1`) still operate on merged + duration-filtered events, while scalar summary AHI (`ahi_mae`, `ahi_pearson`, ICC, and severity summaries) counts stage-filtered raw predicted positive runs without merge or min-duration filtering so it aligns with scalar ground-truth `ahi`. The scalar summary denominator remains NPZ `tst`, samples with `TST < 2h` are skipped for final summary metrics, and the monitored key remains `val_ahi_pearson`.
- Example YAMLs: `configs/sleep2vec_dense_finetune_custom_reg.yaml`, `configs/sleep2vec_dense_finetune_custom_cls.yaml`.

> [!Note]
> `--version-name` is required for pretraining/adaptation run naming; downstream runs auto-generate a version when omitted. Ensure your YAML `data.*` paths point to real preset pickles.

---

## Inference Only
Evaluate a fine-tuned checkpoint without training:

```bash
python -m sleep2vec.infer \
  --config configs/sleep2vec_dense_finetune_cls.yaml \
  --ckpt-path log-finetune/exp001-stage5/checkpoints/epoch=49.ckpt \
  --label-name stage5 --batch-size 12 --devices 0 \
  --inference-preset-path /path/to/test_preset_1535.pickle \
  --eval-split test --results-csv-path outputs.csv
```
Use `--override-dataset-names` to test on a different dataset list than the YAML specifies.
Use `--inference-preset-path` to evaluate the same config/checkpoint against a different preset pickle without editing YAML; result CSV rows record the effective preset in `preset_path`.
Use the same `--label-name` that was used for fine-tuning; it is required.
To average checkpoints before inference, pass `--avg-ckpts N` (and `--avg-ckpt-dir` if `--ckpt-path` is `best/last`).
Use `--pretrained-backbone-path` if you want to preload a pretrain/adaptation initialization checkpoint before applying downstream weights.
Use `--wandb` to enable W&B logging during inference (needed for confusion matrix logging).

---

## Configuration Knobs

**Backbone**  
- Register builders in `sleep2vec/backbones/encoder_factory.py` with `@register_backbone`.
- Select via YAML:
  ```yaml
  model:
    backbone:
      name: roformer
      hidden_size: 768
      num_hidden_layers: 12
      num_attention_heads: 16
      vocab_size: 1
      config_overrides: {}   # add custom kwargs (e.g., MoE routing)
  ```

**Tokenizers**  
- Implement and register in `sleep2vec/modules/tokenizers.py` using `@register_tokenizer("my_tokenizer")`.
- Set per-channel (tokenizer block must supply `name` and `out_dim`):
  ```yaml
  model:
    channels:
      - name: eeg_original
        input_dim: 3840
        tokenizer:
          name: my_tokenizer
          out_dim: 768        # must match across channels
          kwargs: {}
  ```

**Projection Head**  
- Register in `sleep2vec/modules/projection.py` via `@register_projection`.
- Toggle or adjust:
  ```yaml
  model:
    projection:
      name: my_proj
      enabled: true
      hidden_dim: 768
      out_dim: 256
      kwargs: {}
  ```

**Pretrain Loss**  
- Add implementations under `sleep2vec/losses/` and register with `@register_loss`.
- Choose in YAML:
  ```yaml
  loss:
    name: my_loss
    temperature: 0.2
    params: {}
  ```

**Downstream Head**  
- Heads live in `sleep2vec/downstreams/heads/` and register via `sleep2vec/downstreams/head_registry.py`.
- YAML separates channel and temporal aggregation:
  ```yaml
  model:
    head:
      name: classification   # regression | temporal_conv | temporal_transformer
      dropout: 0.1
      hidden_dim: null
      channel_agg:
        name: gated_scalar   # mean | concat | gated_scalar
      temporal_agg:
        name: mean           # mean | attn
  ```
- Temporal aggregation modules are in `sleep2vec/downstreams/temporal_aggregation/`.
- Sequence heads like `temporal_transformer` accept a padding mask from the backbone to ignore padded tokens.


**CLS vs Tokens (downstream representation)**  
`model.cls` controls (1) whether a learnable CLS token is added, and (2) what representation downstream heads consume:
```yaml
model:
  cls:
    embedding_type: null    # null/none -> no CLS token; "bert" -> prepend learnable CLS token
    downstream: tokens      # "tokens" (token-level) or "cls" (sequence-level, non-seq tasks only)
```
- `embedding_type: bert` adds a BERT-style CLS token and exposes both `cls_hidden` and `token_hidden`.
- `downstream: tokens` uses token-level features (sequence tasks) or token pooling (non-seq tasks via `model.head.temporal_agg`).
- `downstream: cls` uses the CLS embedding for **non-seq** tasks and requires `embedding_type: bert`.
- For `--label-name stage3`, `stage4`, `stage5`, or `ahi` (`is_seq=True`), downstream is always token-level; if you set `downstream: cls` it will be ignored (a warning is logged).
- `model.cls` is currently required by the config parser. To disable CLS token usage, set `embedding_type: null` with `downstream: tokens`.

**Layer Mix (downstream)**  
Learned scalar mix across transformer blocks (1..L). For sequence tasks, mixing is applied to token-level states; for non-seq tasks, each layer is pooled first and then mixed. Omit the block to disable, or set `enabled: false`.
```yaml
finetune:
  layer_mix:
    enabled: true
    shared_across_modalities: false   # false -> per-modality weights
    layer_indices: [1, 6, 12]         # 1-based block indices; null -> all
```

**Model Averaging**  
- Strategies live in `sleep2vec/averagings/` (`ema.py` and `running_mean.py` included).
- Configure (omit the block entirely to disable):
  ```yaml
  model_averaging:
    name: ema
    params:
      enabled: true
      base_momentum: 0.996
      final_momentum: 1.0
      use_for_eval: true
  ```
- When `use_for_eval: true`, finetune/infer will evaluate with the averaged weights if present.
- Downstream loading can request averaged weights with `use_ema="ema"` when calling `load_pretrained_backbone`.

**Adaptation Config**
- `sleep2vec.adapt` reuses the pretrain model/loss schema and adds a top-level `adapt` block:
  ```yaml
  adapt:
    new_channels: [ppg, actigraphy_vm]
    stage1:
      train_shared_projection: false
    stage2:
      lr_scales:
        encoder: 0.1
        shared_legacy: 0.5
        new_modalities: 1.0
      pair_schedule:
        - until: 0.25
          new_pair_ratio: 1.0
        - until: 0.50
          new_pair_ratio: 0.7
        - until: 1.0
          new_pair_ratio: 0.0
  ```
- `adapt.new_channels` must be a non-empty subset of `model.channels`.
- Stage 1 freezes the encoder, CLS embedding, and legacy tokenizers; only the new modality tokenizers train, plus `proj_head` when `train_shared_projection: true`.
- Stage 2 restores training for encoder/CLS, shared projection, legacy tokenizers, and new tokenizers, with per-group LR scales from `adapt.stage2.lr_scales`.
- `pair_schedule` reallocates pair-first sampling mass toward pairs that include a new modality early in training, then anneals back toward the full pair set.

**Optimization & Checkpointing**  
- Pretrain/finetune use linear warmup + cosine LR decay; override warmup with `--warmup-steps`.
- Finetune saves `best.ckpt` and `last.ckpt` plus periodic checkpoints; set `--ckpt-every-n-epochs` to control frequency.
- Pretrain and adaptation can also warm-start from `--pretrained-backbone-path`; the loader extracts the pretrain-model subtree (`ema_model.` first, then `model.`) and syncs model averaging state from the loaded student weights.
- `--precision` and `--gradient-clip-val` are supported by pretrain, adaptation, and finetune CLIs.

**Pair-accuracy heatmap (pretrain)**  
- Validation uses per-pair dataloaders to log contrastive accuracy per modality pair.
- W&B logs a heatmap image (`val_pair_acc_matrix`) plus scalar metrics under `val_pair_acc/<pair>`.

**LoRA fine-tuning**  
- Controlled by YAML `finetune` block (parsed by `apply_finetune_config`):
  ```yaml
  finetune:
    freeze_tokenizer: true
    lora:
      freeze_backbone_and_insert_lora: true
      insert_lora: true
      separate_adapters: false
  ```
- `freeze_tokenizer: true` freezes tokenizer parameters during downstream finetuning (default).
- When enabled, `finetune.py` injects PEFT LoRA adapters into the transformer backbone and freezes base weights.

---

## Diagnostics Mode
- Enable hooks with `--print-diagnostics`; control duration with `--diagnostics-steps` (default 5).
- Behavior: disables the progress bar, skips validation/checkpointing, and stops after the requested steps. Stats print to stdout.
- Example (pretrain):
  ```bash
  python -m sleep2vec.pretrain \
    --config configs/sleep2vec_dense_pretrain.yaml \
    --pretrain-data-index /path/to/index.csv \
    --pretrain-preset-path /path/to/pretrain_cache.pkl \
    --version-name debug-diag \
    --print-diagnostics --diagnostics-steps 5 --precision 32 --devices 0
  ```
  (Use the same flags with `sleep2vec.adapt` or `sleep2vec.finetune` for adaptation/downstream diagnostics.)

> [!Important]
> Prefer `--precision 32` when using diagnostics; mixed precision can distort the printed tensor statistics.

---

## Working Tips
- Maintain separate YAML per stage (`*_pretrain.yaml`, `*_finetune_*.yaml`); only pretrain YAML defines `loss`.
- When adding a new modality, first declare it in `model.channels` with the correct `input_dim`, regenerate presets with the same `--config`, then pretrain/adapt from a checkpoint as needed.
- All channels must share the same `out_dim`; the builder enforces this.
- `data.data_channel_names` in finetune YAML must match `model.channels` (input modalities only); built-in sequence labels (`stage3`, `stage4`, `stage5`, `ahi`) load their runtime label tokens automatically when used as `--label-name` (`stage3`/`stage4`/`stage5` from raw `stage5`, `ahi` from raw `ah_event` plus scalar NPZ summaries `ahi` and `tst`). For built-in `ahi`, final scalar summary AHI aligns to NPZ ground-truth `ahi`, while event detection metrics continue to use the stricter merged + duration-filtered event path.
- `pretrain.py`, `adapt.py`, and `finetune.py` copy the resolved `config.yaml` plus `cli_args.yaml` into the run directory for reproducibility.
- When experimenting, adjust CLI flags for training schedules and keep structural changes in YAML for reproducibility.

---

## Repository Layout
- `configs/` — training recipes for pretrain, adaptation, and finetune.
- `sleep2vec/` — core library: registries, encoders, tokenizers, projection, losses, averaging, downstream heads, adaptation modules, and Lightning entrypoints.
- `data/` — dataset/index definitions, metadata helpers, NPZ loaders, channel-selection & samplers.
- `preprocess/` — scripts to build index CSVs/presets, split/merge dataset indices, inspect missing-channel stats, and run raw format converters such as WatchPAT `.zzp` to EDF.
- `utils/` — misc helpers.

---

## Join us

We're Five Seasons Medical, building the full stack of AI for human health —
contactless sensors, foundation models for physiology, and LLM agents
that ship to real users every day. Sleep2vec is one piece of it.

The team comes from Tsinghua, Peking University, and top industry labs.
Small, focused, and shipping.

Hiring across ML research, signal processing, LLM agents, and clinical
science. Reach real users, not just benchmarks — chenxuesong@wuji-inc.com
