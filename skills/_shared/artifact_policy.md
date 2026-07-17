# Artifact Policy

New runnable plans, configs, and scripts belong inside the declared `experiment.root`, grouped under a named step. The workspace holds frozen snapshots, manifests, events, and reports; heavyweight datasets, checkpoints, W&B files, and trainer logs stay in canonical runtime locations and are linked from run artifacts.

Every run has a stable `run-NNN` id and a semantic parameter-derived name. Monitoring must not start pending runs, stopping requires a reason, and historical run trees are not renamed unless the user explicitly asks.

Diagnostic-only context bundles may still use ignored `artifacts/agent_context/` paths.
