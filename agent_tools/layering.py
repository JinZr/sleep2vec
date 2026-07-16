"""Declarative layering map for agent_tools -- the single source of truth for
which modules are the reusable experiment-control kernel, which are sleep2vec
domain specifics, and which are the mixed bridges in between.

This module holds only data (module-name frozensets and an exemption set) and
imports nothing from agent_tools, so the layering guard test can read it without
pulling in torch or any domain package. The human-readable companion is
ARCHITECTURE.md; keep the two in sync (a meta test checks the partition).

Layering (import direction is one-way: L2 -> L1 -> L0, with domain/ a L0-level
domain leaf):
  L0 leaves      -- models, decision_models, transport, ...
  L1 adapters/   -- TaskAdapter protocol + registry (generic) + per-task plugins
  L2 kernel      -- configs, decision_rules, decisions, plan_context, plans
  domain/        -- sleep2vec summary/validator leaves

The guard scans KERNEL_MODULES | MIXED_MODULES for imports that reach into
DOMAIN_MODULES. Pure-kernel modules must stay domain-free (no exemptions);
the mixed bridges carry a fixed set of grandfathered edges
(KNOWN_DOMAIN_IMPORT_EXEMPTIONS) that the guard allows but freezes from growing.
"""

from __future__ import annotations

#: Pure, reusable experiment-control modules (zero domain signal). These must
#: never import a domain module -- the guard allows them no exemptions.
KERNEL_MODULES: frozenset[str] = frozenset(
    {
        "decision_models",
        "transport",
        "manifests",
        "schema_map",
        "gpu_rules",
        "repo",
        "experiment_io",
        "experiment_workspace",
        "experiment_tracking",
        "experiments",
        "run_artifacts",
        "run_evidence",
        "hparam",
        "hparam_runtime",
        "hparam_selection",
        "adaptive_hparam",
        "recipes",
        "progress",
        "markdown",
        "skills",
        "decisions",
        "plans",
        "decision_rules",
    }
)

#: sleep2vec domain modules (summaries, validators, per-task adapters). Named
#: with their package-relative dotted path as the guard normalizes imports.
DOMAIN_MODULES: frozenset[str] = frozenset(
    {
        "domain.sidecar_summaries",
        "domain.finetune_summary",
        "domain.sex_age_summary",
        "domain.presets",
        "domain.index_csv",
        "index_csv",  # top-level re-export shim for domain.index_csv
        "adapters.sleep2stat",
        "adapters.preset_prepare",
        "adapters.finetune",
        "adapters.infer_evaluate",
    }
)

#: Mixed bridges: generic orchestration that still carries domain coupling
#: (hardcoded variant constants, direct domain imports, sleep-specific CLI
#: fields). Tolerated as-is; the guard freezes this set from silently growing.
MIXED_MODULES: frozenset[str] = frozenset(
    {
        "models",
        "configs",
        "plan_rendering",
        "decision_paths",
        "decision_hparam",
        "plan_hparam",
        "plan_context",
        "hparam_postprocess",
        "cli",
    }
)

#: Grandfathered (source, target) import edges from a scanned module into a
#: domain module. The guard permits exactly these and fails on any new one.
#: Each is a deliberate, reviewed tolerance -- removing an edge means deleting
#: both the import and its entry here. See ARCHITECTURE.md "reverse edges".
KNOWN_DOMAIN_IMPORT_EXEMPTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("configs", "domain.finetune_summary"),
        ("configs", "adapters.sleep2stat"),
        ("plan_context", "domain.presets"),
        ("plan_context", "index_csv"),
        ("cli", "domain.presets"),
        ("cli", "index_csv"),
        ("index_csv", "domain.index_csv"),
        ("domain.index_csv", "configs"),  # domain leaf re-entering configs (partial-import break)
    }
)
