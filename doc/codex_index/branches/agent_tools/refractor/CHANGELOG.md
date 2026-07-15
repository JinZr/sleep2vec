# Changelog

## 2026-07-15 — Repair hparam execution identity documentation

- Clarified that planner defaults are limited to the local `REPO_ROOT` manager runtime without a conda wrapper; SSH, separate-workdir, and conda-wrapped targets require explicit Python and full commit identity.
- Indexed reusable `freeze_hparam_execution`: adaptive initialization freezes the round-000 Python/commit pair once, while later rounds may adopt preflight-approved operational execution changes without resolving identity again.
- Replaced stale rendered-CLI snapshot wording with normalized supported-option and validated-argv digests, repository-owned module origin, and same-wrapper pre-`nohup` verification of runtime identity, import safety, and frozen script/config hashes.
- Recorded cross-plan capacity refresh in the locked launcher and explicit queue failure for current or external `missing_pid` blockers.
- Recorded that plans lacking frozen identity, or already-started plans lacking a snapshot, must be recreated rather than upgraded in place.

## 2026-07-15 — Complete static hparam queues and freeze verified targets

- Added public `run_hparam_queue` and `hparam-run-queue`; dry-run previews once, while explicit execute repeatedly reuses the locked launch owner until the current plan is terminal.
- Made generated `run_all.sh` invoke the current control plane rather than importing `agent_tools` from the target runtime checkout.
- Planning now freezes the target Python command and expected runtime commit; execute-time `execution_snapshot.json` verifies the isolated target import root, host, clean worktree, exact commit, normalized supported options, explicit environment digest, and every frozen `argparse` vector.
- Kept `hparam-monitor` observation-only, composed its status transitions into the queue, made `missing_pid` fail instead of loop, skipped probes without an eligible slot, and rejected started plans that lack a snapshot.

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
