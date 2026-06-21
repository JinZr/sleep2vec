# Config Validation Workflow

## Purpose

Validate repository YAML configs against the actual runtime loaders, survival/task semantics, temporal-aggregator rules, and the branch-specific preset-build policy.

## Entry Command

Canonical entrypoint: `python utils/check_configs.py [paths...]`

Primary code path:

1. `utils.check_configs.collect_config_paths`
2. `utils.check_configs.check_config_file`
3. `_resolve_config_variant`
4. package-local `load_pretrain_config` or `load_finetune_config`
5. package-local `validate_model_config`
6. package-local preset-build helpers reused from `preprocess/save_dataset_presets.py`, `sleep2vec2.preprocess.save_dataset_presets`, or `sleep2expert.preprocess.save_dataset_presets`

## Detailed Flow

1. Resolve target config paths.
   - No args: validate every YAML under `configs/`.
   - Paths may be files or directories.
   - Tracked example recipes under `configs/examples/**` are part of the checked config surface.
2. Load the raw YAML mapping.
3. Validate runtime loader compatibility.
   - Root configs load through `sleep2vec.config`.
   - `configs/sleep2vec2/**` loads through `sleep2vec2.config`.
   - `configs/sleep2expert/**` loads through `sleep2expert.config`.
   - Configs outside variant directories that contain `model.backbone.moe` or `finetune.moe_tuning` are routed to `sleep2expert`.
   - Finetune configs must load via the selected package's `load_finetune_config`.
   - Pretrain configs must load via the selected package's `load_pretrain_config`.
   - Both paths also pass through `validate_model_config`.
   - `model.head.temporal_agg.name` must be one of `mean`, `attn`, or `lstm`.
   - Survival configs must use `finetune.task.type: survival`, `is_seq: false`, monitor `val_loss/min` or `val_c_index/max`, and provide `finetune.survival`.
4. Validate `preset_build` when present.
   - Both `required_channels` and `min_channels` must be present together.
   - Validation channels are resolved through the same helpers used by `save_dataset_presets.py`.
   - Built-in `ahi` forces `min_channels == len(channel_names)` at validation time.
5. Validate repo-specific policy for `ppg_*finetune*.yaml`.
   - Sequence staging configs must require `[ppg, stage5]` and `min_channels=2`.
   - PPG AHI configs must require `[ppg, ahi, stage5]` and `min_channels=3`.
   - Non-sequence single-channel PPG configs must require `[ppg]` and `min_channels=1`.
   - PPG Cox configs must load through the selected package loader and keep survival sidecars explicit.

## Important Runtime Decisions

- This workflow intentionally reuses runtime config loaders rather than linting YAML shape independently.
- Variant directories are validated with package-local loaders and package-local preset helpers.
- Repo policy enforcement lives here, not inside `sleep2vec/config.py`.
- Example configs are validated both by `check_config_file` and by `apply_finetune_config` for built-in finetune tasks.
- Failures are reported per file so the command can validate the entire tree in one pass.

## Outputs

- Exit code `0` when all checked configs pass
- Exit code `1` plus `[FAIL] <path>: <message>` lines when any config fails

## Edit Hotspots

- Change schema validation: `sleep2vec/config.py`
- Change standalone variant schema validation: `sleep2vec2/config.py` or `sleep2expert/config.py`
- Change built-in task semantics or finetune normalization: `sleep2vec/common.py`
- Change survival sidecar schema: root and package-local `config.py` plus `data/survival.py`
- Change preset-build policy: `preprocess/save_dataset_presets.py`, `utils/check_configs.py`
