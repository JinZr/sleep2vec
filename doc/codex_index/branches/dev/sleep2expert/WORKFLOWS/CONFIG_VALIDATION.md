# Config Validation Workflow

## Purpose

Validate repository YAML configs against the actual runtime loaders and the branch-specific preset-build policy.

## Entry Command

Canonical entrypoint: `python utils/check_configs.py [paths...]`

Primary code path:

1. `utils.check_configs.collect_config_paths`
2. `utils.check_configs.check_config_file`
3. `utils.check_configs._load_config_tools`
4. variant-local `load_pretrain_config` or `load_finetune_config`
5. variant-local `validate_model_config`
6. variant-local preset-build helpers reused from `save_dataset_presets.py`

## Detailed Flow

1. Resolve target config paths.
   - No args: validate every YAML under `configs/`.
   - Paths may be files or directories.
2. Select the config tool family from the path prefix.
   - `configs/sleep2expert/**` uses `sleep2expert.config` and `sleep2expert.preprocess.save_dataset_presets`.
   - `configs/sleep2vec2/**` uses `sleep2vec2.config` and `sleep2vec2.preprocess.save_dataset_presets`.
   - Other configs use `sleep2vec.config` and top-level `preprocess.save_dataset_presets`.
3. Load the raw YAML mapping with the selected preset helper.
4. Validate runtime loader compatibility.
   - Finetune configs must load via `load_finetune_config`.
   - Pretrain configs must load via `load_pretrain_config`.
   - Both paths also pass through `validate_model_config`.
5. Validate `preset_build` when present.
   - Both `required_channels` and `min_channels` must be present together.
   - Validation channels are resolved through the same helpers used by `save_dataset_presets.py`.
   - Built-in `ahi` forces `min_channels == len(channel_names)` at validation time.
6. Validate repo-specific policy for `ppg_*finetune*.yaml`.
   - Sequence staging configs must require `[ppg, stage5]` and `min_channels=2`.
   - PPG AHI configs must require `[ppg, ahi]` and `min_channels=2`.
   - Non-sequence single-channel PPG configs must require `[ppg]` and `min_channels=1`.

## Important Runtime Decisions

- This workflow intentionally reuses runtime config loaders rather than linting YAML shape independently.
- Standalone variant configs must be checked with their package-local config and preprocessing modules.
- Repo policy enforcement lives here, not inside `sleep2vec/config.py`.
- Failures are reported per file so the command can validate the entire tree in one pass.

## Outputs

- Exit code `0` when all checked configs pass
- Exit code `1` plus `[FAIL] <path>: <message>` lines when any config fails

## Edit Hotspots

- Change schema validation: `sleep2vec/config.py`
- Change built-in task semantics or finetune normalization: `sleep2vec/common.py`
- Change standalone variant schema validation: the matching `<variant>/config.py`
- Change preset-build policy: `preprocess/save_dataset_presets.py`, `utils/check_configs.py`
