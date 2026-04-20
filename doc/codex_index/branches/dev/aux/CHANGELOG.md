# Changelog

## 2026-04-20

- Initialized the `dev/aux` branch handbook from the current working tree using the nearest built-in-ahi handbook baseline.
- Added the registered `temporal_unet` downstream head and documented it as the new large-context sequence head for AHI-style token prediction.
- Added a max-reuse metadata auxiliary-task path that reuses the existing temporal aggregator registry plus the existing `classification` / `regression` heads instead of introducing a parallel auxiliary-head registry.
- Added `configs/ppg_ahi_finetune_large_temporal_unet_aux.yaml` as the first large-backbone built-in `ahi` recipe using `temporal_unet` plus a general metadata auxiliary regression target.
- Expanded config, loader, downstream-model, and stage-task tests around auxiliary-task parsing, metadata-target plumbing, and dual-output downstream forward behavior.
- Tightened auxiliary-task fail-fast validation so string metadata targets (`source`, `path`) are rejected early and built-in metadata targets such as `age` must keep their canonical regression semantics.
