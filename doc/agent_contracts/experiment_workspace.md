# Experiment Workspace Contract

An experiment workspace is the durable, human-readable record for a related set of preparation, training, evaluation, and analysis steps. Heavy datasets, checkpoints, W&B files, and trainer logs remain in their canonical runtime locations; the workspace stores frozen snapshots, indexes, links, events, and reports.

```text
<experiment.root>/
├── experiment.yaml
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
└── <contained plan directory>/
    ├── recipe.resolved.yaml
    ├── plan.json
    ├── run_all.sh
    └── runs/run-000--<semantic-name>/
        ├── run.json
        ├── config.yaml
        ├── launch.sh
        └── artifacts.json
```

## Required metadata

`experiment.yaml` records a stable id, title, objective, root, and explicit baseline. Every `steps/<step.id>/step.yaml` uses the same `{step, experiment_id, recipe_path, plans}` envelope whether it is created by `experiment-register-step` or by planning. Both producers read the existing envelope through the workspace owner and merge it through the shared step reducer. A confirmed missing file may be created; an existing zero-byte, null, malformed, incomplete, or conflicting envelope fails and is never repaired by overwriting it. Compatible later writers preserve registered optional step metadata, fill an absent recipe path, and append plan locations without duplication. Experiment and step ids may use lowercase letters, digits, hyphens, and underscores.

Local experiment roots have one durable representation: user-home expansion followed by absolute path resolution. Recipe-relative roots use the repository root as their base, while experiment CLI-relative roots use the caller's current working directory; both persist the resulting canonical absolute path in the recipe, plan, `experiment.yaml`, and `experiment_manifest.tsv`. SSH roots remain exact remote strings and are never resolved against the local filesystem. Existing workspaces with a different stored root spelling are not migrated or silently rewritten.

All repository-owned management locators are persisted as absolute local paths. The rule covers `step.yaml` recipe and plan entries; plan, run, report, config, script, and artifacts locations; frozen runtime/checkpoint locations; adaptive workflow roots, recipe paths, round directories, and registry copies; and those same locators when recorded in events, digests, or suggestions. Relative CLI paths are made absolute at their public entry boundary and remain stable if a later command runs from another working directory. This rule does not rewrite user-authored semantic dataset, external-checkpoint, or other runtime input paths. SSH management locators remain exact remote strings.

New runnable `agent_tools plan` requests fail or ask the user when this metadata is unresolved. When experiment or step metadata itself is unresolved, `plan` returns questions without creating files. Once the binding is complete, other blocked decisions are recorded inside an initialized workspace and may be retried with a new plan directory. The plan directory must be inside the declared experiment root and is indexed in the matching step manifest; placing it below `steps/<step.id>/` is recommended. Existing metadata comparisons include the declared root. A non-empty root without `experiment.yaml` is rejected rather than adopted, and a completed experiment cannot accept another plan.

## Run identity

`run_id` is a stable sequence key within a step and continues across additional plans for that step. The canonical managed key is `(step_id, run_id)`, and both fields are required. `run_name` is derived from the varied parameter values, must remain understandable without opening a config, and does not participate in matching. A supplied complete managed key never falls back to version matching. Runtime `version` is reserved for external evidence lookup only when managed identity is absent; version-only evidence may update a managed row only when it identifies one unique run. Reports and rankings show the step, id, and name together.

## Lifecycle ownership

- `plan` freezes recipes, configs, commands, hashes, the run matrix, and planned manifest rows. Runnable non-hparam finetune, infer, evaluate, preset-prepare, and sleep2stat scripts always enter `REPO_ROOT`, commit `running` through the canonical owner before runtime, and commit `completed` or `failed` on exit without depending on W&B; an already-terminal run is not executed again.
- `hparam-launch` verifies workspace/step registration and frozen hashes, rejects completed experiments, starts eligible planned runs only when explicitly executed, and records launch events.
- `hparam-monitor` is read-only with respect to process scheduling. It updates status snapshots and never fills free slots.
- `hparam-stop` only stops a PID from the canonical run row, requires a reason, rejects an already-terminal canonical run before PID access, and records `stopped` only after the local signal or SSH command succeeds.
- `hparam-adaptive-step` without `--execute` monitors the current round and writes its digest, suggestion, and preview events, but does not stop or supersede runs or create the next round. The execute path supersedes eligible pending runs through the canonical manifest owner before planning or launching the next round.
- `hparam-select` uses the metric and direction frozen in the recipe, requires every runnable registered hparam plan in the same step to use that same selection contract, validates plan/resolved task agreement and ranking topology before reading runtime evidence, skips explicit blocked-plan artifacts, reranks the complete step, and writes `reports/ranking.csv`.
- `experiment-rank` writes the separate experiment-wide `reports/experiment_ranking.csv` report and preserves each `(step_id, run_id)` identity.
- `experiment-finalize` accepts only runs in explicit terminal states and requires a non-empty final report before marking the experiment complete. Missing PID, unknown, blank, planned, pending, launched, and running states remain unresolved.

Every experiment mutation other than initialization requires a parseable `experiment.yaml` whose id and root own the target directory. An optional `experiment_manifest.tsv` must have one unique rectangular header and exactly one row that agrees with that ownership. Initialization alone may accept a genuinely empty directory; copied roots, malformed current tables, legacy tables, and non-empty unmanaged roots fail before any local or remote write.

Managed output targets are validated before the first mutation. Existing targets must be independent regular files, workspace subdirectories leading to them must not be aliases, and duplicate, symlink, hard-link, directory, or out-of-root targets fail closed. The same transport-level rule protects canonical tables, matrices, events, reports, W&B inventories, checkpoint indexes, and step/experiment files locally and over SSH.

The workspace owner also owns canonical reads. Missing, valid empty, and corrupt artifacts are distinct states: only fresh workspace initialization may create a missing run manifest, only a first-step producer may create a confirmed missing step manifest, a header-only current run table is valid and empty, and any existing blank, malformed, unreadable, wrong-format, duplicate-key, recursive-alias, or dangling-symlink canonical artifact fails closed. Authoritative experiment and step YAML share the duplicate-key- and cycle-aware mapping reader. Managed auxiliary tables must retain both identity headers even when they contain zero rows; a header-only removed-format or incomplete table is corrupt. Run-dependent commands never turn missing or corrupt canonical state into an empty experiment. The same lifecycle applies over SSH, where only the explicit missing return code proves absence.

Remote preflight is fail-closed. A confirmed missing file or empty directory is distinct from an SSH, timeout, type, or read failure; uncertain reads never permit directory creation, W&B access, event append, or canonical table writes. Automated tests mock the `experiment_io` SSH boundary and cover missing, rejection, nonzero exit, and timeout without requiring a reachable server; live SSH is an explicit integration check, not part of the local test gate. Remote final-report paths must be absolute and fail before any workspace or SSH access otherwise. An uncertain PID read or process probe is reported as `unknown_remote` by monitoring, while PID-read uncertainty aborts stop before kill or mutation. Locally, empty, non-numeric, non-positive, or invalid-encoding PID content is confirmed corrupt and maps launchable `planned` or `pending` to non-launchable `missing_pid`. A local path/read `OSError` while planned or pending aborts without committing state, allowing a later confirmed absence to launch normally; other states keep shared-reducer precedence, and stop remains fail-closed.

All lifecycle consumers share one status merge rule. The workspace `run_manifest.tsv` is the sole mutable decision input for lifecycle status and execution identity. The plan-local status and launch tables mirror exact post-commit canonical rows and are never read back as evidence. Runtime manifests do not own lifecycle status. Mutation-facing tracking paths submit only managed-keyed observations from their source allowlists; the workspace owner re-reads and conditionally commits canonical state locally or through SSH, and a new managed key must carry the same experiment id as the workspace owner. Terminal states are not overwritten by stale non-terminal evidence; incoming failure may only correct an existing completed or finished state. Active states also cannot be regressed to stale `planned` or `pending` values, and an adaptive `superseded` observation commits only if the freshly read canonical state remains `planned` or `pending`. Plan-owned identity, snapshot, hash, artifact-path, raw `runtime.*` / `yaml:/...` parameters, and canonical execution identity (`target`, `host`, `workdir`, `gpus`, PID/log paths, and command) are immutable once registered. Only the canonical owner performs the first trusted execution-identity fill. Stop and monitoring read execution identity only from canonical state.

Evidence must prove ownership before allowlisting. Global W&B evidence either carries the matching experiment id or resolves by a unique version when that id is absent. Workspace-scoped metrics, checkpoint, ranking, and adaptive-registry rows inherit scope only after their complete table is matched against canonical managed keys and frozen fields; launch/status tables are output projections and are not allowlisted as inputs. Candidate tables likewise validate against the complete workspace before filtering ownership-valid rows from other steps or earlier plans in the same step, after which retained rows also prove current-plan membership. Any explicitly supplied experiment id, version, config, parameter, hash, or frozen locator must match its owner; foreign, unmatched, or drifting rows fail before process access, canonical mutation, report, or ranking writes.

Failed or interrupted run artifacts remain evidence. Agents must not rewrite old run identities or silently reuse their run directories.

Historical workspaces using the removed plan, identity, status, or registry format remain available for manual read-only inspection. Current tooling does not migrate, adopt, mutate, or automatically aggregate them.
