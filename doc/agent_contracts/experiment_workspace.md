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
├── steps/<step.id>/step.yaml
└── <plan directory>/
    ├── recipe.resolved.yaml
    ├── plan.json
    ├── run_all.sh
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
- `hparam-launch` validates frozen artifacts and explicitly starts eligible runs; dry-run remains the default.
- `hparam-monitor` observes registered runs and never schedules pending work.
- `hparam-stop` requires a reason and uses canonical execution identity.
- `hparam-select` writes step-scoped validation ranking.
- `hparam-adaptive-*` appends rounds and commits replacements through the canonical owner.
- `experiment-rank` writes experiment-wide ranking.
- `experiment-finalize` requires no active runs and a non-empty final report.

Every mutation other than fresh initialization requires a parseable, root-matching workspace owner. Managed output targets are preflighted before mutation: existing targets must be independent regular files under valid directory ancestry. Local and SSH uncertainty fails closed.

Run identity, status vocabulary, reducer precedence, PID/evidence behavior, atomic commit, and projection sequencing belong to [run_manifest.md](run_manifest.md).
