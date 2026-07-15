# Changelog

## 2026-07-15 — Close authored recipe contracts

- Added source-aware top-level and section closure inside existing recipe, plan, experiment, decision, execution, hparam, and rendering owners.
- Rejected authored internal metadata and closed user-decision document/name/entry fields before output mutation.
- Shared runtime/preset accepted fields with their renderer mappings and removed inert fields from checked-in recipes and sleep2stat examples.
- Kept hparam base/local provenance intact and made adaptive rounds/suggestions write local overlays rather than flattened effective recipes.
- Resolved task/source roles before closure, kept policy-only decisions out of inert sections, and preflighted mutable adaptive sources and generated suggestions before durable writes.
- Added focused closure/atomicity/consumer tests without adding a schema registry, version, or facade.

## 2026-07-14 — Initialize branch index

- Initialized `agent_tools/refractor` at `40b8b7b811a68369e739e9d2f717651652853ace` using `initialize-branch` mode.
- Added all required branch-index outputs and `DELTA_FROM_MAIN.md`.
- Indexed the 33 tracked `agent_tools` modules by responsibility.
- Cataloged the reusable recipe, consultation, preflight, rendering, workspace, evidence, hparam, experiment, adaptive, progress, and summary entrypoints.
- Documented `run_manifest.tsv` as canonical lifecycle state, `preflight_plan` as the no-write boundary, and existing runtime entrypoints as downstream owners.
- Recorded that local `main` and the branch have no tracked code delta at initialization time.
