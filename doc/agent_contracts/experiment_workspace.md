# Experiment Workspace Contract

An experiment workspace is the durable, human-readable record for related preparation, training, evaluation, and analysis steps. Heavy datasets, checkpoints, W&B files, and trainer logs remain in their runtime locations; the workspace stores frozen snapshots, indexes, events, and reports.

```text
<experiment.root>/
├── experiment.yaml
├── experiment_manifest.tsv  # optional experiment-CLI index
├── README.md
├── events.jsonl
├── run_manifest.tsv
├── run_matrix.csv
├── reports/
│   ├── status.md
│   ├── ranking.csv
│   ├── experiment_ranking.csv
│   └── final.md
├── pipelines/<pipeline-id>/  # optional managed external-evaluation state
│   ├── spec.source.yaml
│   ├── spec.resolved.yaml
│   ├── pipeline.json
│   ├── checkpoints.json
│   ├── preflight.json
│   ├── jobs.tsv
│   ├── execution_snapshot.json
│   ├── results.csv
│   ├── metrics.csv
│   ├── summary.md
│   ├── final.md
│   ├── recipes/<job-id>/attempt-NNN.yaml
│   ├── plans/<job-id>/attempt-NNN/
│   ├── preflight_retries/<job-id>/attempt-NNN.json
│   ├── retry_schedulers/<job-id>/execution_snapshot.json
│   └── results/<job-id>/attempt-NNN/
├── steps/<step.id>/step.yaml
└── <plan directory>/
    ├── recipe.resolved.yaml
    ├── plan.json
    ├── run_all.sh
    ├── execution_snapshot.json  # created by the first verified execute
    └── runs/run-000--<semantic-name>/
        ├── run.json
        ├── config.yaml
        ├── launch.sh
        └── artifacts.json
```

## Ownership and paths

`experiment.yaml` records a stable id, title, objective, canonical root, and explicit baseline. Experiment and step ids use lowercase letters, digits, hyphens, and underscores. Step phase is one of `prepare`, `train`, `evaluate`, or `analyze`. Every step file uses the shared `{step, experiment_id, recipe_path, plans}` envelope. Existing experiment and step metadata are read through the workspace owner and merged through their reducers; missing files may be created only by their designated first producer. Blank, malformed, incomplete, or conflicting metadata is never repaired by overwriting it.

`experiment_manifest.tsv` is optional for plan-created workspaces. When present, it contains exactly one row whose experiment id and root match `experiment.yaml`.

Local recipe roots are based at the repository root; local experiment CLI roots are based at the caller's current working directory. Both are expanded and resolved once. All local repository-owned management locators—including recipe, plan, run, config, script, artifact, report, runtime/checkpoint, adaptive, and event paths—are persisted as absolute paths. SSH roots and locators remain exact remote strings. User-authored semantic data and checkpoint paths are not normalized by this management rule.

A new plan must be contained by its experiment root and registered in its step manifest. A non-empty unmanaged root is rejected rather than adopted, and a completed experiment cannot accept another plan. Historical workspaces are not migrated or renamed.

## Lifecycle entrypoints

- `plan` freezes the effective recipe, configs, commands, hashes, and planned runs.
- `hparam-launch` validates frozen artifacts and explicitly starts one eligible wave; dry-run remains the default.
- `hparam-run-queue` is the explicit long-running action that repeatedly fills available capacity until every current-plan run is terminal; dry-run performs one preview and returns.
- `hparam-monitor` observes registered runs and never schedules pending work.
- `hparam-stop` requires a reason, verifies the canonical PID/process-group/start-token identity, stops the complete process group, and records terminal state only after exit is confirmed.
- `hparam-select` writes step-scoped validation ranking.
- `hparam-adaptive-*` appends rounds and commits replacements through the canonical owner.
- `experiment-run` is the explicit, resumable external-evaluation launcher. Dry-run starts nothing; execute waits for successful source plans, freezes validation-selected checkpoints, and manages the declared job matrix.
- `experiment-rank` writes experiment-wide ranking.
- `experiment-finalize` requires no active runs and a non-empty final report.

`hparam-monitor` and `experiment-monitor` remain non-launching even when a
pipeline has pending jobs. Pipeline locking, frozen state, attempt isolation,
and finalization sequencing belong to
[experiment_pipeline.md](experiment_pipeline.md).

Every mutation other than fresh initialization requires a parseable, root-matching workspace owner. Managed output targets are preflighted before mutation: existing targets must be independent regular files under valid directory ancestry. Local and SSH uncertainty fails closed.

Planning freezes `execution.python` and `execution.runtime_commit`. Only the canonical manager runtime—a local target at `REPO_ROOT` without a conda wrapper—may omit them; planning then freezes the current manager interpreter and manager repository HEAD. SSH targets, separate local workdirs, and conda-wrapped targets must provide both values explicitly.

The first `hparam-launch --execute` or `hparam-run-queue --execute` with an eligible slot probes that exact Python command through the configured target, workdir, conda wrapper, and explicit environment. It requires the planned Git commit, no tracked worktree changes, no untracked or ignored importable Python or extension-module code, target-reported host identity, a runtime module whose resolved origin is inside the verified repository, and successful `argparse` validation of every frozen argument vector from that origin. Untracked experiment artifacts and data remain allowed. The snapshot stores the module origin, normalized supported options, and digests of those options, every validated argv vector, and the explicit execution environment; rendered CLI text is not snapshot evidence. The evidence is atomically written to `execution_snapshot.json`, and every later eligible launch wave re-probes and requires exact equality. Immediately before each process-group start, the same target/env/conda/PYTHONPATH wrapper rechecks Python/version, commit, repository root, hostname, module origin, untracked or ignored importable code, and the selected run's frozen script/config hashes. Target and leaf `PYTHONPATH` contain only `execution.workdir`, so another manager checkout cannot satisfy missing imports.

Dry-run and monitor never probe or create the snapshot. A plan without frozen Python/commit identity must be recreated. A missing snapshot may be established only while every plan run remains `planned` or `pending` and has no committed execution target; once execution identity or later state exists, the plan must be recreated instead of upgraded in place. Removed `trial_*` plans and status files remain unmanaged and read-only. Before calculating capacity, execute-mode launch refreshes observable active blockers from other plans that share the relevant target/host/GPU pool and commits their status transitions. The full queue fails explicitly if either a current-plan run or a relevant cross-plan capacity blocker is `missing_pid`; a queue with no eligible slot does not probe the execution snapshot. External datasets, drivers, and environment outside explicit `execution.env` remain operational dependencies rather than snapshot contents.

Run identity, status vocabulary, reducer precedence, PID/evidence behavior, atomic commit, and projection sequencing belong to [run_manifest.md](run_manifest.md).
