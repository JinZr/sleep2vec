# Agent Tooling

This catalog covers the reusable functions behind `python -m agent_tools`. The tools are agent-facing orchestration around existing runtime and preprocessing entrypoints; they should not replace `sleep2vec`, `sleep2vec2`, `sleep2expert`, or `preprocess` command implementations.

## `agent_tools.cli.main`

- File: `agent_tools/cli.py`
- Signature: `main(argv: list[str] | None = None) -> int`
- Purpose and contract: parse `agent_tools` subcommands and dispatch to command handlers.
- Important inputs/outputs: optional argv in; integer process exit code out.
- Side effects: writes command-specific artifacts through delegated modules and prints JSON or text payloads.
- Key callers/callees: called by `agent_tools/__main__.py`; delegates to `plans`, `decisions`, `hparam`, `experiments`, `adaptive_hparam`, `progress`, and summary modules.
- Reuse guidance: add CLI wiring here only after the underlying reusable function exists in a focused module.
- Duplication-risk notes: avoid encoding workflow logic in `_cmd_*`; keep handlers as adapters.

## `agent_tools.decisions.evaluate_consultation_gates`

- File: `agent_tools/decisions.py`
- Signature: `evaluate_consultation_gates(task: str | None, recipe: dict | None, config_summary: dict | None, cli_args: dict | None, policy: dict, approved_defaults: dict) -> DecisionReport`
- Purpose and contract: enforce stop-and-consult policy for high-impact recipe and runtime decisions, including variantless `sleep2stat` gates, survival sidecar gates, and final-test unlock/checkpoint gates.
- Important inputs/outputs: task, recipe, config summary, CLI/user decisions, policy, and defaults in; `DecisionReport` with resolved decisions, issues, status, and exit-code semantics out.
- Side effects: none.
- Key callers/callees: called by `agent_tools.plans.evaluate_recipe` and `build_context`; delegates path checks to `decision_paths`, task rules to `decision_rules`, and tuning rules to `decision_hparam`, while keeping source resolution and base-finetune recursion in the facade.
- Reuse guidance: use this before generating runnable preset, finetune, inference, evaluation, hparam, or sleep2stat commands.
- Duplication-risk notes: do not reimplement high-impact decision checks in recipe templates or shell generation; edit the owning `decision_*` module, while sleep2stat structural validation remains in `sleep2stat.config.load_config()`.

## `agent_tools.plans.build_context`

- File: `agent_tools/plans.py`
- Signature: `build_context(*, task: str, config: str | Path | None, output_dir: str | Path, label_name: str | None = None, variant: str | None = None, user_decisions_path: str | Path | None = None) -> DecisionReport`
- Purpose and contract: gather repo/config/index/preset context and write a diagnostic bundle without authorizing runnable commands.
- Important inputs/outputs: task, optional config/label/variant/user-decisions, and output directory in; `DecisionReport` out.
- Side effects: writes `context.json`, `context.md`, questions, and `commands.blocked.sh`; runnable commands require `build_plan`.
- Key callers/callees: called by `agent_tools context`; delegates context/index/preset/Markdown work to `plan_context`, command primitives to `plan_rendering`, and task dispatch to `_commands_for_recipe`.
- Reuse guidance: use for diagnostic context bundles rather than assembling context ad hoc.
- Duplication-risk notes: keep generated command routing inside `_commands_for_recipe` so variant and sleep2stat variantless rules stay centralized.

## `agent_tools.plans.build_plan`

- File: `agent_tools/plans.py`
- Signature: `build_plan(*, recipe_path: str | Path, output_dir: str | Path, user_decisions_path: str | Path | None = None, allow_unresolved: bool = False, unlock_final_test: bool = False) -> DecisionReport`
- Purpose and contract: evaluate a task recipe and write a runnable or blocked command plan after the shared read-only preflight succeeds far enough to bind an experiment and step. Hparam ownership must come from the local tuning recipe rather than its base finetune recipe, and a user-selected effective config passes config summary and consultation again before workspace mutation. Runnable non-hparam scripts use `REPO_ROOT` for cwd/PYTHONPATH and commit `running` then `completed/failed` through the canonical owner without W&B. Planned output files may be overwritten only as independent regular files; symlinks and hard links fail the output guard even when overwrite is explicitly allowed. Existing workspace run-matrix and event files remain legal derived/append-only outputs, but they pass the same alias check before a successful plan mutates the workspace.
- Important inputs/outputs: recipe path, output directory, optional decisions, draft allowance, and final-test unlock in; `DecisionReport` out.
- Side effects: unresolved experiment/step metadata produces no files; other blockers initialize the managed workspace and write questions plus `plan.blocked.md` without registering a run. Passing plans write `plan.json`, `plan.md`, `run.sh`, frozen run scripts/configs, and an explicitly unlocked final external-test script when applicable.
- Key callers/callees: called by `agent_tools plan` and adaptive init/step; uses consultation gates and output guards, then delegates final-test/grid/frozen-plan work to `plan_hparam` and rendering to `plan_rendering`.
- Reuse guidance: use for all recipe-backed command generation.
- Duplication-risk notes: do not emit executable training or sleep2stat scripts outside this path unless a recipe has already passed the same gates.

## `agent_tools.plans.preflight_plan`

- File: `agent_tools/plans.py`
- Signature: `preflight_plan(*, recipe_path: str | Path, output_dir: str | Path, user_decisions_path: str | Path | None = None, allow_unresolved: bool = False, unlock_final_test: bool = False) -> tuple[dict, dict | None, DecisionReport]`
- Purpose and contract: run consultation, effective-config validation, local hparam ownership, workspace containment, hparam/final-test, command-support, and output guards without writing files.
- Side effects: reads recipe/config/path state only.
- Key callers/callees: shared by `build_plan` and adaptive initialization/round execution before either path mutates workspace or run state.
- Reuse guidance: use this when a workflow must prove a recipe is safe before creating directories, stopping or superseding runs, or launching processes.

## `agent_tools.plans._commands_for_recipe`

- File: `agent_tools/plans.py`
- Signature: `_commands_for_recipe(recipe: dict, cfg: dict | None = None, decisions: dict | None = None) -> list[str]`
- Purpose and contract: render the approved command list for supported recipe tasks, including variant-aware model commands, preset commands, and variantless sleep2stat commands. Sleep2stat post-run summarize and plot commands are skipped for dry-run recipes; summarize is a read-only run-dir overview and does not inherit runtime worker settings.
- Important inputs/outputs: recipe, optional config summary, and resolved decisions in; shell command strings out.
- Side effects: none.
- Key callers/callees: called by `build_context` and `build_plan`; uses module-qualified helpers from `plan_rendering` for variant routing, runtime args, quoting, environment, and sleep2stat arguments.
- Reuse guidance: add new generated-command behavior here after consultation gates can prove the task is safe.
- Duplication-risk notes: do not render runnable sleep2stat, finetune, infer, hparam, or preset scripts outside this function family.

## `agent_tools.plans.collect_runs`

- File: `agent_tools/plans.py`
- Signature: `collect_runs(root: str | Path, metric: str | None, output: str | Path) -> None`
- Purpose and contract: require an explicit managed workspace root, validate its current `run_manifest.tsv`, then collect only declared rows and each row's exact `runtime_dir` when present. Runtime `run_manifest.json` is resolved through `run_artifacts.find_run_manifest`, so aliases, non-regular files, invalid encoding/JSON, and non-mapping payloads fail rather than being parsed as YAML. A missing, unreadable, or invalid canonical manifest fails instead of producing an empty inventory; a valid header-only manifest still represents zero runs. Before creating a parent directory or opening the output, the command rejects the canonical table itself plus directory, symlink, hard-link, and aliased-ancestor output topologies.
- Important inputs/outputs: run root, optional metric, and output path in; CSV file out.
- Side effects: reads the canonical table and its declared artifact paths, then writes the requested CSV; optional W&B summaries are best-effort evidence, and it does not recursively adopt historical directories.
- Key callers/callees: called by `agent_tools collect-runs`; reads W&B summary files when present.
- Reuse guidance: use for lightweight local run inventory before ranking or manual review.
- Duplication-risk notes: do not treat this as a source of truth when `agent_tools.experiments` manifests are available.

## `agent_tools.decision_paths.validate_input_path`

- File: `agent_tools/decision_paths.py`
- Signature: `validate_input_path(recipe: dict, field: str, raw_path: Any, *, configured: bool) -> DecisionIssue | None`
- Purpose and contract: apply the canonical local, deferred-remote, or timeout-bounded SSH existence rule for an agent-facing input path.
- Side effects: may run one short SSH command when `execution.path_validation=ssh|remote`.
- Key callers/callees: used by decision task rules and `plan_hparam` final checkpoint/config gates.
- Reuse guidance: call this instead of duplicating path-context or SSH checks in plans.

## `agent_tools.plan_rendering.script_lines` and `infer_runtime_cli_args`

- File: `agent_tools/plan_rendering.py`
- Signatures: `script_lines(commands: list[str], *, run_cwd: str | Path | None = None, experiment_root: str | Path | None = None, step_id: str | None = None, run_id: str | None = None) -> list[str]`; `infer_runtime_cli_args(runtime: dict[str, Any]) -> list[Any]`
- Purpose and contract: render the shared shell cwd/PYTHONPATH and optional canonical non-hparam lifecycle wrapper, plus the inference runtime flags used by recipe plans and post-hparam evaluation/export. The lifecycle wrapper refuses terminal replay and preserves runtime or terminal-commit failure.
- Side effects: none.
- Key callers/callees: `script_lines` is used by `plans.build_context` and `plans.build_plan`; `infer_runtime_cli_args` is used by `plans._commands_for_recipe`, `plan_hparam.write_hparam_plan`, and `hparam_postprocess`.
- Reuse guidance: keep generated-script bootstrap/lifecycle and inference CLI propagation here instead of importing or duplicating private plan behavior.

## `agent_tools.plan_hparam.write_hparam_plan`

- File: `agent_tools/plan_hparam.py`
- Signature: `write_hparam_plan(recipe: dict, cfg: dict | None, out: Path, *, unlock_final_test: bool, report: DecisionReport) -> None`
- Purpose and contract: expand the grid, freeze semantic run directories/configs/scripts/hashes, register managed rows, and write validation/final-test launch artifacts.
- Side effects: writes the hparam plan tree and experiment workspace rows/events.
- Key callers/callees: called only by `agent_tools.plans.build_plan`; uses `plan_rendering` and canonical identity/workspace functions.
- Reuse guidance: change hparam plan compilation here while retaining `build_plan` as the public orchestration entrypoint.

## `agent_tools.recipes.load_recipe_with_base`

- File: `agent_tools/recipes.py`
- Signature: `load_recipe_with_base(path: str | Path) -> dict[str, Any]`
- Purpose and contract: load a YAML recipe and merge any `base_recipe`.
- Important inputs/outputs: recipe path in; merged recipe dict out, preserving base recipe metadata under `_base_recipe`.
- Side effects: reads YAML files.
- Key callers/callees: used by plan, doctor, hparam, and adaptive workflows.
- Reuse guidance: use for recipe consumers so checked-in base recipes resolve consistently.
- Duplication-risk notes: avoid separate base-recipe merge behavior in command-specific modules.

## `agent_tools.configs.config_summary`

- File: `agent_tools/configs.py`
- Signature: `config_summary(config_path: str | Path) -> dict[str, Any]`
- Purpose and contract: summarize YAML task, channels, backend, preset/index inputs, survival sidecars, variant guess, or sleep2stat run/data/analyzer/output fields without importing Torch/Lightning.
- Important inputs/outputs: config path in; JSON-ready summary out.
- Side effects: reads the YAML file.
- Key callers/callees: called by `plans`, `doctor`, and CLI summary commands.
- Reuse guidance: use for agent policy checks that need config facts without loading models. For survival configs, this validates local sidecars when path context allows it. For sleep2stat-shaped YAML, this calls `sleep2stat.config.load_config()` and returns blocking issues instead of raising through plan generation.
- Duplication-risk notes: do not infer task semantics from path names when this summary can read YAML fields.

## `agent_tools.configs.sleep2stat_config_summary`

- File: `agent_tools/configs.py`
- Signature: `sleep2stat_config_summary(config_path: str | Path) -> dict[str, Any]`
- Purpose and contract: summarize sleep2stat run, data, analyzer, and output fields for agent planning while delegating structural validation to `sleep2stat.config.load_config`.
- Important inputs/outputs: config path in; JSON-ready summary with `is_sleep2stat`, `data_backend`, `sleep2stat`, `blocking_issues`, and agent risk issues out.
- Side effects: reads the YAML file and imports `sleep2stat.config`.
- Key callers/callees: called by `config_summary`; downstream callers are `decisions`, `plans`, `doctor`, and context generation.
- Reuse guidance: use this for sleep2stat policy checks instead of reading sleep2stat YAML directly in agent tooling.
- Duplication-risk notes: placeholder analyzer checkpoint/config checks are agent risk checks only; canonical schema validation stays in `sleep2stat.config.load_config`.

## `agent_tools.index_csv.index_summary`

- File: `agent_tools/index_csv.py`
- Signature: `index_summary(index_paths: list[str | Path], *, config: str | Path | None = None, label_name: str | None = None, split_values: list[str] | None = None, sample_path_check: int = 0, sample_npz_check: int = 0) -> dict[str, Any]`
- Purpose and contract: summarize index CSV row counts, columns, labels, source/split fields, numeric shifts, split-filtered subsets, survival and multilabel sidecar key coverage, and optional sample path/NPZ checks.
- Important inputs/outputs: one or more CSV paths plus optional config/label/split/check counts in; JSON-ready summary out.
- Side effects: reads CSV files and optionally probes sample paths/NPZ files.
- Key callers/callees: called by `agent_tools index-summary` and context bundles.
- Reuse guidance: use before preset/finetune planning when index contents or subject-level sidecar key coverage are part of the decision surface.
- Duplication-risk notes: keep deeper sample filtering in `data.utils` and preprocessing CLIs, not here; this is a lightweight planning summary.

## `agent_tools.presets.preset_summary`

- File: `agent_tools/presets.py`
- Signature: `preset_summary(preset_path: str | Path) -> dict[str, Any]`
- Purpose and contract: inspect a preset pickle enough for agent planning and diagnostics.
- Important inputs/outputs: preset path in; row/count/channel/source-style summary out.
- Side effects: reads pickle content.
- Key callers/callees: called by `agent_tools preset-summary` and context bundles.
- Reuse guidance: use for lightweight preset checks before command generation.
- Duplication-risk notes: do not validate or rewrite preset payloads here; runtime validation remains in dataset code.

## `agent_tools.progress.write_progress`

- File: `agent_tools/progress.py`
- Signature: `write_progress(run_dir: str | Path, *, status: str, task: str, processed: int = 0, total: int | None = None, success: int = 0, failed: int = 0, start_time: float | None = None, current_item: str | None = None, message: str | None = None) -> Path`
- Purpose and contract: write machine-readable progress for long-running preprocessing and utility jobs.
- Important inputs/outputs: run directory, status, task, counters, optional timing/current item/message in; `status/progress.json` path out.
- Side effects: writes progress JSON under the run directory.
- Key callers/callees: used by preprocessing/utility progress emitters; read by `agent_tools.progress.read_progress`.
- Reuse guidance: use this for agent-visible progress instead of bespoke status files.
- Duplication-risk notes: preserve the `status/progress.json` location so `agent_tools progress` and health checks remain compatible.

## `agent_tools.progress.read_progress`

- File: `agent_tools/progress.py`
- Signature: `read_progress(run_dir: str | Path, *, remote: str | None = None) -> dict[str, Any]`
- Purpose and contract: read progress from a local run directory or through a short SSH `cat`.
- Important inputs/outputs: run directory and optional remote host in; progress dict out.
- Side effects: may execute one SSH command for remote reads.
- Key callers/callees: called by `agent_tools progress` and hparam health checks.
- Reuse guidance: use this instead of custom remote log probing when a process emits progress JSON.
- Duplication-risk notes: remote probes should stay short and read-only.

## `agent_tools.run_artifacts.read_hparam_plan`

- File: `agent_tools/run_artifacts.py`
- Signature: `read_hparam_plan(run_dir: Path) -> dict[str, Any]`
- Purpose and contract: validate the run-only plan shape, complete experiment/step metadata, initialized workspace ownership, step-registered plan path, plan containment, run-to-recipe identity consistency, required canonical run-manifest rows and fields, exact equality between the runtime-consumed plan recipe (including its resolved base recipe) and `recipe.resolved.yaml`, frozen artifact fields, actual config/script hashes, and removed recipe fields before a caller mutates state.
- Side effects: reads the frozen plan and workspace manifests only.
- Key callers/callees: shared by hparam runtime, selection, postprocess, and adaptive workflows.
- Reuse guidance: every operation that consumes a hparam plan should cross this boundary before starting/stopping processes or writing canonical tables.

## `agent_tools.run_artifacts.find_run_manifest`

- File: `agent_tools/run_artifacts.py`
- Signature: `find_run_manifest(run: dict[str, Any]) -> Path | None`
- Purpose and contract: locate runtime evidence only at the managed row's frozen `runtime_dir/run_manifest.json` path. A confirmed missing file means no current evidence; a symlink, dangling symlink, directory, hard link, invalid encoding, invalid JSON, or non-mapping payload is corrupt and fails closed.
- Side effects: reads exact-path filesystem metadata and validates JSON content.
- Key callers/callees: used by hparam selection, run evidence, and adaptive digests together with the checkpoint/metric interpreters in the same module.
- Reuse guidance: use this low-level owner instead of importing an implementation detail from the `hparam` facade.

## `agent_tools.run_evidence.status_row`

- File: `agent_tools/run_evidence.py`
- Signature: `status_row(run_dir: Path, row: dict[str, Any], previous: dict[str, Any] | None = None, *, health: bool = False) -> dict[str, Any]`
- Purpose and contract: own the shared `RUN_EVIDENCE_FIELDS`/`RUN_STATUS_FIELDS` allowlist, produce one PID/process/log/progress/checkpoint observation, and merge it with the previous row through `experiment_workspace.merge_run_row`, preserving terminal-state precedence. Launch mirrors may supply only those execution fields, never stale score, rank, W&B, or other canonical snapshot fields. Only a confirmed absent PID file returns no PID. Empty, non-numeric, non-positive, or invalid-encoding local PID content is confirmed corrupt, so `planned` or `pending` becomes non-launchable `missing_pid`. A local path/read `OSError` while a run is still `planned` or `pending` aborts the observation without committing state. Remote process exit becomes terminal only after a certain log read; timeout, permission, or transport failure remains `unknown_remote`. Remote runtime-manifest and checkpoint evidence is read on the execution host through the same bounded transport and preserves prior fields when that probe is uncertain.
- Side effects: reads local or timeout-bounded remote process/log state.
- Key callers/callees: used by hparam runtime and experiment tracking; delegates checkpoint interpretation to `run_artifacts`.
- Reuse guidance: use for observed run state instead of applying status updates outside the shared reducer.

## `agent_tools.hparam.launch_hparam_runs`

- File: `agent_tools/hparam.py`
- Signature: `launch_hparam_runs(plan_dir: str | Path, *, dry_run: bool = True) -> Path`
- Purpose and contract: validate the current-format registered plan, non-completed experiment, all managed tables, every frozen config/script hash, canonical/mirror/report/event and per-run log/PID targets, plus the single-use frozen runtime/checkpoint roots before creating a run directory, starting any process, or writing launch state. `execution.gpus_per_run` requires a non-empty physical pool from `execution.gpu_pool` or `runtime.devices`; both consultation and runtime reject an unmanaged implicit pool. Canonical status decides launch eligibility and the real active load of each GPU group; local status is mirror-only, and launch rows contribute only the shared execution-evidence allowlist. Without explicit `execution.max_concurrent`, each GPU group admits one run; an explicit higher limit permits user-authorized oversubscription and emits a warning. SSH execute validates the same paths on the execution host; dry-run performs no remote output probe.
- Important inputs/outputs: plan directory and dry-run flag in; `launch_manifest.tsv` path out.
- Side effects: creates log/PID directories, writes `launch_manifest.tsv` and `run_status.tsv`, and starts processes only when `dry_run=False`.
- Key callers/callees: public facade for `hparam_runtime.launch_hparam_runs`, called by `agent_tools hparam-launch` and adaptive step; runtime helpers remain in `hparam_runtime`.
- Reuse guidance: use after `build_plan` has produced a hparam `plan.json`.
- Duplication-risk notes: do not launch runs by scanning scripts directly; this function owns PID/log manifest contracts.

## `agent_tools.hparam.monitor_hparam_runs`

- File: `agent_tools/hparam.py`
- Signature: `monitor_hparam_runs(run_dir: str | Path, *, once: bool = True, health: bool = False) -> Path`
- Purpose and contract: preflight every canonical/mirror/report/event output target, start from canonical `run_manifest.tsv`, ignore the local status mirror, accept only non-status fields from launch evidence, then reduce one current observation into the final row; missing or stale local evidence cannot regress canonical state.
- Important inputs/outputs: run directory plus monitor flags in; `run_status.tsv` path out.
- Side effects: reads process/log/run artifacts, commits canonical status first, then rewrites the local status mirror and reports; it never launches pending runs.
- Key callers/callees: public facade for `hparam_runtime.monitor_hparam_runs`, called by `agent_tools hparam-monitor` and adaptive digest; observed state comes from `run_evidence`.
- Reuse guidance: use before selection/ranking; `run_manifest.tsv` is the durable status owner and `launch_manifest.tsv` is execution evidence only.
- Duplication-risk notes: keep remote probes inside the existing timeout-bounded helpers.

## `agent_tools.hparam.stop_hparam_run`

- File: `agent_tools/hparam.py`
- Signature: `stop_hparam_run(run_dir: str | Path, run_id: str, *, reason: str) -> Path`
- Purpose and contract: preflight every canonical/mirror/report/event output target, validate launch execution evidence against the immutable canonical execution identity, ignore launch values when choosing target/host/PID, fail before PID access or writes when canonical is terminal, otherwise terminate the canonical PID and reduce one `stopped` observation with its non-empty reason.
- Important inputs/outputs: run directory, run id, and reason in; updated `run_status.tsv` path out.
- Side effects: after a successful signal, writes the same reduced final row to canonical status, the local status mirror, and the launch mirror, then appends one experiment event.
- Key callers/callees: public facade for `hparam_runtime.stop_hparam_run`, called by `agent_tools hparam-stop` and adaptive replacement logic.
- Reuse guidance: use this for run termination so PID provenance is checked first.
- Duplication-risk notes: do not kill arbitrary command-line matches outside the manifest.

## `agent_tools.hparam.select_hparam_candidates`

- File: `agent_tools/hparam.py`
- Signature: `select_hparam_candidates(run_dir: str | Path, metric: str | None = None, mode: str | None = None) -> Path`
- Purpose and contract: rank runs by the validation metric and direction frozen in the recipe, require every runnable registered hparam plan in the same step to use that same selection contract, record fixed checkpoint paths, validate every existing experiment-wide ranking row against canonical ownership, and rebuild the complete registered step ranking from finite scores. Unscored runs in the current step have canonical selection fields cleared; an existing invalid-score row from another step fails instead of being silently removed. Explicit blocked-plan artifacts are skipped, while a runnable plan missing `plan.json` or a plan/resolved task mismatch fails. Ranking plus canonical matrix/event targets are preflighted before the first ranking or runtime-evidence read.
- Important inputs/outputs: run directory plus optional matching metric/mode assertions in; experiment `reports/ranking.csv` path out.
- Side effects: reads plan and run manifests; writes ranking CSV and updates experiment manifests/events.
- Key callers/callees: public facade for `hparam_selection.select_hparam_candidates`; uses canonical interpreters from `run_artifacts`.
- Reuse guidance: use for validation-based selection after runs finish.
- Duplication-risk notes: do not replace selected checkpoints with mutable best aliases.

## `agent_tools.hparam.scan_hparam_checkpoints`

- File: `agent_tools/hparam.py`
- Signature: `scan_hparam_checkpoints(run_dir: str | Path, metric: str, mode: str, *, top_k: int | None = None) -> Path`
- Purpose and contract: preflight `checkpoint_ranking.csv`, validate every existing checkpoint-ranking row against canonical workspace ownership and frozen metadata before runtime scanning, then rank fixed epoch checkpoints using W&B history, CSV history, or manifest fallback metrics; repeated empty scans preserve a valid `step_id`/`run_id` header.
- Important inputs/outputs: run directory, metric, mode, and optional top-k in; `checkpoint_ranking.csv` path out.
- Side effects: reads local histories/manifests and writes ranking CSV.
- Key callers/callees: public facade for `hparam_selection.scan_hparam_checkpoints`, called by `agent_tools hparam-checkpoint-scan`.
- Reuse guidance: use when checkpoint-level selection matters more than run-level selection.
- Duplication-risk notes: keep epoch parsing in the shared checkpoint helpers.

## `agent_tools.hparam.generate_external_eval`

- File: `agent_tools/hparam.py`
- Signature: `generate_external_eval(run_dir: str | Path, selected_csv: str | Path, *, unlock_final_test: bool, kaldi_data_root: str | None = None, kaldi_manifest: str | None = None, finetune_data_index: str | None = None, eval_split: str = "test", top_k: int = 1, all_candidates: bool = False) -> Path`
- Purpose and contract: canonicalize the local plan path, preflight the current managed hparam plan, validate the complete candidate table against the workspace before filtering ownership-valid other-step or earlier same-step plan rows, require all registered owner plans in the step to share the caller's frozen selection metric and mode, require every checkpoint path to be a direct child of its frozen checkpoint directory, apply top-k in numeric rank order, then preflight all configs/manifest/script outputs before creating locked final/external inference commands; frozen run fields come from each candidate's owning plan.
- Important inputs/outputs: hparam run directory, selected candidates, explicit unlock, optional replacement data paths, split, base runtime settings, selected-row `runtime.*` overrides, and candidate selection controls in; `external_eval.sh` path out.
- Side effects: writes copied configs, `external_eval_manifest.tsv`, and executable shell script.
- Key callers/callees: public facade for `hparam_postprocess.generate_external_eval`; uses `module_for_variant` and `plan_rendering.infer_runtime_cli_args`.
- Reuse guidance: use for final-test or external-test command generation after an explicit unlock.
- Duplication-risk notes: final-test evaluation must stay locked by this explicit flag.

## `agent_tools.hparam.export_hparam_logits`

- File: `agent_tools/hparam.py`
- Signature: `export_hparam_logits(run_dir: str | Path, selected_csv: str | Path, *, unlock_final_test: bool, val_split: str = "val", test_split: str = "test", skip_test: bool = False, label_name: str | None = None, val_kaldi_data_root: str | None = None, val_kaldi_manifest: str | None = None, val_finetune_data_index: str | None = None, test_kaldi_data_root: str | None = None, test_kaldi_manifest: str | None = None, test_finetune_data_index: str | None = None, batch_size: int = 12, num_workers: int = 8, devices: list[int] | None = None, accelerator: str = "gpu", device: str = "cuda", precision: str = "bf16-mixed", seed: int = 4523, top_k: int = 1, all_candidates: bool = False, execute: bool = False) -> Path`
- Purpose and contract: canonicalize the local plan path, preflight the current managed hparam plan, validate the complete candidate table plus checkpoint ownership before filtering ownership-valid other-step or earlier same-step plan rows, require all registered owner plans in the step to share the caller's frozen selection metric and mode, apply top-k in numeric rank order, then preflight all configs/manifest/logits/script outputs before preparing or executing inference; frozen run fields come from each candidate's owning plan, and dry-run replay scripts persist absolute plan/candidate-table arguments with the repository cwd/PYTHONPATH bootstrap.
- Important inputs/outputs: selected candidates, split/data overrides, runtime controls, unlock/skip-test, and execution flag in; `logits_export_manifest.tsv` path out.
- Side effects: writes copied configs and, for dry runs, a replay script plus manifest; with `execute=True`, runs inference commands and copies produced prediction CSVs to logits paths, then writes `logits_export_manifest.tsv` only after every requested split succeeds.
- Key callers/callees: public facade for `hparam_postprocess.export_hparam_logits`, called by `agent_tools hparam-export-logits`.
- Reuse guidance: use before threshold fitting or ensembling selected models.
- Duplication-risk notes: test logits require `--unlock-final-test` unless explicitly skipped.

## `agent_tools.hparam.threshold_hparam_outputs`

- File: `agent_tools/hparam.py`
- Signature: `threshold_hparam_outputs(run_dir: str | Path, selected_csv: str | Path) -> Path`
- Purpose and contract: canonicalize the local plan path, preflight the current managed hparam plan, validate candidate ownership and same-step owner-plan selection contracts, then require both validation and test predictions/logits, fit a binary F1 threshold on validation data, and apply it to test data.
- Important inputs/outputs: run directory and selected CSV in; `threshold_summary.csv` path out.
- Side effects: reads prediction CSVs and writes threshold summary.
- Key callers/callees: public facade for `hparam_postprocess.threshold_hparam_outputs`, called by `agent_tools hparam-threshold`.
- Reuse guidance: use for post-hparam binary threshold selection.
- Duplication-risk notes: keep exploratory thresholding separate from training metrics.

## `agent_tools.hparam.ensemble_hparam_outputs`

- File: `agent_tools/hparam.py`
- Signature: `ensemble_hparam_outputs(run_dir: str | Path, candidates_csv: str | Path, *, search_combinations: bool = False, max_size: int | None = None, metric: str = "exploratory_test_auroc", mode: str = "max", top_k: int | None = None) -> Path`
- Purpose and contract: canonicalize the local plan path, preflight the current managed hparam plan, validate candidate ownership and same-step owner-plan selection contracts, then average binary prediction probabilities across the retained registered-plan candidates, optionally searching combinations.
- Important inputs/outputs: candidate CSV, combination controls, rank metric/mode, and top-k in; `ensemble_summary.csv` path out.
- Side effects: reads prediction CSVs and writes ensemble summary.
- Key callers/callees: public facade for `hparam_postprocess.ensemble_hparam_outputs`, called by `agent_tools hparam-ensemble`.
- Reuse guidance: use for exploratory ensembling after logits/predictions are exported.
- Duplication-risk notes: treat this as analysis tooling, not a trainer/runtime model averaging feature.

## `agent_tools.experiment_workspace.canonical_local_experiment_root`

- File: `agent_tools/experiment_workspace.py`
- Signature: `canonical_local_experiment_root(raw: str | Path, base_dir: str | Path) -> Path`
- Purpose and contract: produce the sole durable local experiment-root representation by expanding the user home, applying the caller-owned base directory to relative paths, lexically collapsing `.` and `..`, rejecting a direct or dangling symlink at the resulting root, and resolving the remaining absolute path. Symlinked parent components may resolve normally. Recipe roots use the repository root as their base; experiment CLI roots use the caller's current working directory. SSH paths do not call this function and remain exact remote strings.
- Side effects: none beyond filesystem path resolution.
- Reuse guidance: use at public local plan, experiment, and adaptive entry boundaries before persisting or comparing repository-owned management locators; do not add producer-specific normalization or apply it to user semantic input paths.

## `agent_tools.experiment_workspace.read_managed_yaml_mapping`

- File: `agent_tools/experiment_workspace.py`
- Signature: `read_managed_yaml_mapping(text: str, *, source: str | Path) -> dict[str, Any]`
- Purpose and contract: parse authoritative experiment or step YAML as one non-empty mapping while rejecting duplicate keys and recursive aliases at every nesting level instead of accepting YAML last-wins ownership or traversing cyclic node graphs.
- Side effects: none.
- Key callers/callees: used by planning/workspace preflight, experiment mutation, step reads, W&B experiment metadata updates, and `run_artifacts.read_hparam_plan`.
- Reuse guidance: use for authoritative managed YAML; ordinary recipe/config semantics remain owned by their existing loaders.

## `agent_tools.experiment_workspace.read_step_manifest`

- File: `agent_tools/experiment_workspace.py`
- Signature: `read_step_manifest(root: str | Path, step_id: str, *, remote: str | None = None, allow_missing: bool = False) -> dict[str, Any] | None`
- Purpose and contract: distinguish a confirmed absent step from an existing canonical `{step, experiment_id, recipe_path, plans}` envelope, reject blank/null/duplicate-key/incomplete content and invalid phases, and require stored recipe and plan locators to be empty or absolute as defined by the envelope.
- Side effects: reads one local file or performs the shared bounded SSH existence/read operations; it never repairs or writes an existing step.
- Reuse guidance: every step producer must call this reader before `merge_step_manifest`; `allow_missing=True` is limited to first-step creation.

## `agent_tools.experiment_workspace.initialize_run_manifest` and `read_run_manifest`

- File: `agent_tools/experiment_workspace.py`
- Signatures: `initialize_run_manifest(root: str | Path, *, remote: str | None = None) -> Path`; `read_run_manifest(root: str | Path, *, remote: str | None = None) -> list[dict[str, str]]`
- Purpose and contract: initialize a fresh workspace with the canonical `step_id`/`run_id` header, then distinguish that valid header-only state from missing, blank, malformed, non-rectangular, wrong-header, legacy, duplicate-key, symlinked, or hard-linked canonical tables. Non-empty rows require a non-blank experiment owner and exact non-whitespace managed identity. The path itself is part of ownership proof: only initialization may create a missing manifest, and every later canonical read is strict locally and over SSH.
- Side effects: initialization writes one new canonical header; the reader performs local or bounded SSH reads only.
- Key callers/callees: workspace creation calls the initializer; planning, hparam runtime/selection/adaptive paths, experiment tracking, reporting, and `merge_run_manifest` use the reader.
- Reuse guidance: never call generic row/text readers for `run_manifest.tsv`; best-effort readers remain limited to optional evidence.

## `agent_tools.experiment_workspace.ensure_experiment_workspace`

- File: `agent_tools/experiment_workspace.py`
- Signature: `ensure_experiment_workspace(recipe: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]`
- Purpose and contract: validate ownership and metadata before mutating, then initialize the recipe-owned experiment root and named step before runnable artifacts are written.
- Important inputs/outputs: resolved recipe and contained plan directory in; experiment and step paths out.
- Side effects: creates `experiment.yaml`, `README.md`, `events.jsonl`, `reports/`, and `steps/<step.id>/step.yaml` when absent.
- Reuse guidance: use the plan path rather than creating an alternate experiment directory layout.
- Duplication-risk notes: experiment metadata validation and output containment belong in this module.

## `agent_tools.experiment_workspace.merge_step_manifest`

- File: `agent_tools/experiment_workspace.py`
- Signature: `merge_step_manifest(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]`
- Purpose and contract: create or compatibly merge the sole step envelope, `{step, experiment_id, recipe_path, plans}`; step metadata and experiment ownership cannot drift, registered `inputs`/`outputs` survive planner updates, `recipe_path` is filled once, and plan paths are append-only and deduplicated.
- Side effects: none.
- Key callers/callees: used by workspace planning, `experiment-register-step`, and hparam plan preflight.
- Reuse guidance: every producer or validator of `steps/<step.id>/step.yaml` must use this function instead of writing a producer-specific shape.

## `agent_tools.experiment_workspace.run_identity`

- File: `agent_tools/experiment_workspace.py`
- Signature: `run_identity(recipe: dict[str, Any], index: int, parameters: dict[str, Any], *, run_name: str | None = None) -> dict[str, str]`
- Purpose and contract: pair a stable `run-NNN` id with a parameter-derived or explicitly supplied semantic name and globally descriptive runtime version.
- Side effects: none.
- Reuse guidance: use for every new managed run instead of hand-building versions or emitting `run_000` names.

## `agent_tools.experiment_workspace.managed_run_key` and `run_evidence_key`

- File: `agent_tools/experiment_workspace.py`
- Signatures: `managed_run_key(row: dict[str, Any]) -> tuple[str, str] | None`; `run_evidence_key(row: dict[str, Any]) -> tuple[str, ...] | None`
- Purpose and contract: centralize the only managed identity, `(step_id, run_id)`; incomplete rows are not managed, while version grouping is limited to unscoped external evidence.
- Side effects: none.
- Reuse guidance: use these instead of hand-building run keys in experiment manifests, monitoring, or ranking.

## `agent_tools.experiment_workspace.managed_run_parameters`

- File: `agent_tools/experiment_workspace.py`
- Signature: `managed_run_parameters(row: dict[str, Any]) -> dict[str, Any]`
- Purpose and contract: recognize the only current managed parameter spellings, `runtime.*` and `yaml:/...`, while rejecting removed `param.*` fields.
- Side effects: none.
- Key callers/callees: used by managed-table validation, frozen plan preflight, selection, ranking, adaptive digests, and postprocessing.
- Reuse guidance: use this function whenever run parameters are persisted, compared, or projected; never add another prefix transform.

## `agent_tools.experiment_workspace.validate_managed_run_rows`

- File: `agent_tools/experiment_workspace.py`
- Signature: `validate_managed_run_rows(rows: list[dict[str, Any]], *, source: str, cardinality: str) -> None`
- Purpose and contract: reject removed-format rows, blank or whitespace-padded managed identity, and relative repository-owned management locators, while requiring callers to state whether the table is `one_per_run` or `many_per_run`; only the former rejects duplicate `(step_id, run_id)` keys.
- Side effects: none.
- Reuse guidance: call at every external table boundary before writes or process/PID side effects. Use `many_per_run` for metrics/checkpoint evidence and `one_per_run` for plans, run/status/launch manifests, rankings, and adaptive registries.

## `agent_tools.experiment_workspace.resolve_run_row` and `resolve_external_run_row`

- File: `agent_tools/experiment_workspace.py`
- Signatures: `resolve_run_row(rows: list[dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any] | None`; `resolve_external_run_row(rows: list[dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any] | None`
- Purpose and contract: resolve workspace-scoped evidence by complete managed identity with optional experiment-id consistency, falling back to one unique version only when managed identity is absent. The external wrapper requires a supplied experiment id before trusting managed identity; without one it strips unproven step/run claims and permits only the unique-version path. Display names and bare ids never match.
- Side effects: none.
- Reuse guidance: use `resolve_run_row` only after workspace/plan scope is proven, and use `resolve_external_run_row` for global W&B/metric evidence; do not maintain caller-local identity indexes.

## `agent_tools.experiment_workspace.merge_run_row`, `validate_frozen_run_update`, `merge_run_manifest`, and `write_run_matrix`

- File: `agent_tools/experiment_workspace.py`
- Signatures: `merge_run_row(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]`; `validate_frozen_run_update(existing: dict[str, Any], incoming: dict[str, Any], *, require_checkpoint_ownership: bool = False, allow_execution_identity_fill: bool = False) -> None`; `merge_run_manifest(root: str | Path, rows: list[dict[str, Any]], *, remote: str | None = None) -> list[dict[str, Any]]`; `write_run_matrix(root: str | Path, rows: list[dict[str, Any]], *, remote: str | None = None) -> Path`
- Purpose and contract: keep `merge_run_row` as the pure status reducer with terminal precedence, the active-to-scheduled regression guard, and an owner-time rule that permits `superseded` only from canonical `planned/pending`; use `validate_frozen_run_update` at pre-side-effect boundaries; and let `merge_run_manifest` own local and SSH canonical persistence after re-reading current state. Execution evidence cannot initialize canonical execution identity by default; only the canonical owner and its generated launch-row validation opt into the first trusted fill. The validator's explicit checkpoint-ownership mode additionally requires each non-empty evidence `checkpoint_path` to be a direct child of the run's frozen `checkpoint_dir`; empty checkpoint evidence remains valid. The owner verifies every newly introduced key against the experiment id in the workspace's authoritative manifest, rejects changes to registered identity, semantic parameters, snapshots, hashes, artifact paths, and execution identity, preflights canonical/matrix/event targets through the transport owner, returns the complete rows actually committed, and passes those rows directly to `write_run_matrix` without a second manifest read. An empty matrix has exactly the `step_id,run_id` header in both transports.
- Side effects: the reducer returns a new row; the validator is pure and raises on drift; the manifest merger validates and rewrites the canonical manifest and its run matrix through the selected transport.
- Reuse guidance: tracking code must produce ownership-proven narrow observations and submit them to this owner; mirrors, reports, and transition events must consume its returned rows. Do not read or update `run_manifest.tsv` directly or reimplement frozen-field checks.

## `agent_tools.manifests.read_rows` and `agent_tools.experiment_io.path_exists_at`, `read_rows_at`, `validate_managed_output_paths`, and `write_rows_at`

- Files: `agent_tools/manifests.py`, `agent_tools/experiment_io.py`
- Signatures: `read_rows(path: str | Path, *, require_managed_identity: bool = False) -> list[dict[str, str]]`; `path_exists_at(path: str | Path, *, remote: str | None = None) -> bool`; `read_rows_at(path: str | Path, *, remote: str | None = None, require_managed_identity: bool = False, strict: bool = False) -> list[dict[str, str]]`; `validate_managed_output_paths(root: str | Path, paths: list[str | Path], *, remote: str | None = None) -> None`; `write_rows_at(path: str | Path, rows: list[dict[str, Any]], *, remote: str | None = None) -> None`
- Purpose and contract: provide consistent local/SSH delimiter, quoting, timeout, strict-table, and managed-output behavior. `path_exists_at` distinguishes confirmed absence. Optional generic reads may represent genuinely missing input as `[]`; strict mode rejects empty/headerless, duplicate-header, malformed, and non-rectangular tables without imposing run identity, while managed-table mode adds current identity and removed-format checks. `validate_managed_output_paths` performs one fail-closed local or SSH preflight for all targets, rejecting out-of-root paths, aliased subdirectories, duplicate targets, directories, symlinks, and hard links before mutation. Transport, timeout, target-type, and read failures fail closed. Unit and contract tests mock this SSH boundary; no live host is required by the automated gate.
- Side effects: reads or writes local files, or executes one short SSH command.
- Key callers/callees: used by `experiments` lifecycle orchestration and `experiment_tracking` reports/history.
- Reuse guidance: add transport behavior here rather than branching local/remote I/O in each command. Use managed-table mode for launch/status/metric/checkpoint/ranking/registry/candidate evidence; canonical step/run lifecycle semantics remain in the strict workspace readers that compose these primitives.

## `agent_tools.experiment_tracking.experiment_run_rows`

- File: `agent_tools/experiment_tracking.py`
- Signature: `experiment_run_rows(root: Path, *, remote: str | None = None) -> list[dict[str, Any]]`
- Purpose and contract: read the current managed manifest through the workspace owner, prove launch/status evidence belongs to that workspace and managed key, reject any explicit experiment/version/config/frozen-locator drift, and only then project source allowlists for monitoring and ranking. Until canonical execution identity has a trusted target, auxiliary execution fields are ignored rather than used for process access or persistence. The result is an in-memory view only and is never written as a canonical snapshot.
- Side effects: reads experiment tables only.
- Key callers/callees: used by public monitor/rank commands; delegates I/O to `experiment_io` and row reduction to `experiment_workspace`.
- Reuse guidance: use this as the experiment-wide row aggregation boundary.

## `agent_tools.experiment_tracking.wandb_run_observations` and `monitor_run_row`

- File: `agent_tools/experiment_tracking.py`
- Signatures: `wandb_run_observations(run_rows: list[dict[str, Any]], wandb_rows: list[dict[str, Any]]) -> list[dict[str, Any]]`; `monitor_run_row(root: Path, row: dict[str, Any], previous_rows: list[dict[str, str]], *, remote: str | None = None) -> dict[str, Any]`
- Purpose and contract: turn ownership-proven W&B or runtime evidence into narrow managed-keyed observations. W&B evidence uses `resolve_external_run_row`, so a supplied experiment id must match and evidence without one may resolve only by unique version; observations contain only `(step_id, run_id)` plus `WANDB_RUN_FIELDS`, collapse repeated evidence to one row per key, and never persist display `version`. Monitor evidence is workspace-scoped, validates explicit frozen fields before allowlisting, and emits only the managed key plus `RUN_STATUS_FIELDS`; an SSH target/host injected solely for the current transport probe is stripped before canonical commit.
- Side effects: W&B observation projection is pure; monitor projection reads local or remote process/runtime evidence.
- Key callers/callees: used by `experiments.sync_wandb_runs` and `experiments.monitor_experiment`, which submit the observations to `experiment_workspace.merge_run_manifest`.
- Reuse guidance: add evidence interpretation here, but keep canonical persistence in the workspace owner.

## `agent_tools.experiment_tracking.managed_metric_rows` and `checkpoint_rows`

- File: `agent_tools/experiment_tracking.py`
- Signatures: `managed_metric_rows(run_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]`; `checkpoint_rows(root: Path, *, remote: str | None = None) -> list[dict[str, Any]]`
- Purpose and contract: apply provenance-specific ownership before managed metric/checkpoint projection. External W&B metrics use `resolve_external_run_row`; workspace checkpoint rows require an exact canonical key. Checkpoint roots must be real directories rather than symlinks, and local/remote scans include only regular non-symlink files. A real `best-epoch=XX.ckpt` remains valid when its epoch matches an explicit run-manifest epoch even if no plain `epoch=XX.ckpt` exists; an explicit metric epoch still binds only that epoch, while an epoch-less metric may use the validated best fallback. Any supplied ownership, version, config, hash, or frozen-locator drift is validated before fields are replaced, filtered, scanned, or ranked.
- Side effects: metric projection is pure; checkpoint indexing reads declared local/remote runtime and checkpoint locations only after the full prior table passes ownership validation.
- Reuse guidance: use these boundaries instead of joining evidence by display name, bare id, or inferred directory location; unmatched external evidence remains raw, while unmatched workspace-scoped rows fail.

## `agent_tools.experiments.init_experiment`

- File: `agent_tools/experiments.py`
- Signature: `init_experiment(run_dir: str | Path, spec_path: str | Path, *, remote: str | None = None) -> Path`
- Purpose and contract: validate the explicit YAML spec, all five existing ownership fields including root, strict rectangular `experiment_manifest.tsv` lifecycle, current-format managed tables, and every managed output target before creating any local or remote experiment directories; only a genuinely empty target may be initialized, and initialization creates the valid header-only canonical run manifest before any run registration.
- Important inputs/outputs: run directory, experiment spec, optional remote host in; `experiment_manifest.tsv` path out.
- Side effects: creates directories and writes `experiment.yaml`, `README.md`, events, and the experiment manifest.
- Key callers/callees: called by `agent_tools experiment-init`; delegates local/SSH persistence to `experiment_io`.
- Reuse guidance: use before experiment-level monitor/rank operations.
- Duplication-risk notes: keep remote writes routed through the shared remote read/write helpers.

## `agent_tools.experiments.register_experiment_step`

- File: `agent_tools/experiments.py`
- Signature: `register_experiment_step(run_dir: str | Path, spec_path: str | Path, *, remote: str | None = None) -> Path`
- Purpose and contract: preflight initialized experiment ownership and current tables, validate an explicit step spec, and register a new step without adopting an unmanaged directory.
- Side effects: writes `steps/<step.id>/step.yaml` and appends one experiment event.
- Reuse guidance: use the public `experiment-register-step` command instead of creating step directories directly.

## `agent_tools.experiments.finalize_experiment`

- File: `agent_tools/experiments.py`
- Signature: `finalize_experiment(run_dir: str | Path, report_path: str | Path, *, remote: str | None = None) -> Path`
- Purpose and contract: preflight initialized ownership, require at least one managed run with every run terminal, and commit a non-empty final report before marking the experiment completed.
- Side effects: writes `reports/final.md`, updates `experiment.yaml`, and appends one experiment event.
- Reuse guidance: use after monitoring has committed terminal status for every managed run.

## `agent_tools.experiments.sync_wandb_runs`

- File: `agent_tools/experiments.py`
- Signature: `sync_wandb_runs(run_dir: str | Path, *, entity: str, project: str, group: str | None = None, remote: str | None = None) -> Path`
- Purpose and contract: sync W&B metadata, summaries, and history; exact managed identity or identity-absent unique-version evidence may update only the W&B/status field allowlist, while canonical identity and version are reapplied to managed metric rows. Raw W&B and canonical owner targets are preflighted together before the API call or any write.
- Important inputs/outputs: run directory plus W&B entity/project/group and optional remote host in; `wandb/runs.tsv` path out, with recognized W&B states mapped into canonical managed-run status.
- Side effects: calls `wandb.Api()`, writes raw W&B inventories, history, and managed metrics, then submits managed-keyed W&B observations to `experiment_workspace.merge_run_manifest`; unmatched W&B rows remain only in `wandb/runs.tsv`, summaries, and history.
- Key callers/callees: called by `agent_tools experiment-wandb-sync`; delegates W&B payload/history/report shaping to `experiment_tracking` and persistence to `experiment_io`.
- Reuse guidance: use W&B API data rather than browser pages for numeric metrics.
- Duplication-risk notes: avoid scraping W&B UI or mixing browser-rendered values into ranking.

## `agent_tools.experiments.index_checkpoints`

- File: `agent_tools/experiments.py`
- Signature: `index_checkpoints(run_dir: str | Path, *, remote: str | None = None) -> Path`
- Purpose and contract: preflight both the metrics evidence path and checkpoint output path, then index checkpoints only from managed rows whose frozen `runtime_dir` and `checkpoint_dir` are both present. Both paths may be empty for non-checkpoint-producing runs; partial pairs, symlink roots, non-regular checkpoint entries, or prior checkpoint rows outside eligible managed keys fail before replacement. Every non-empty prior `checkpoint_path` must be a direct child of its frozen directory.
- Important inputs/outputs: run directory and optional remote host in; `checkpoint_manifest.tsv` path out.
- Side effects: scans the declared local or remote directories and writes the checkpoint manifest; it does not recursively scan or infer identity from path names.
- Key callers/callees: called by `agent_tools experiment-index-checkpoints`; delegates local/remote checkpoint scanning and metric attachment to `experiment_tracking`.
- Reuse guidance: use before experiment-level candidate ranking.
- Duplication-risk notes: remote checkpoint scans should stay shallow and command-bounded.

## `agent_tools.experiments.monitor_experiment`

- File: `agent_tools/experiments.py`
- Signature: `monitor_experiment(run_dir: str | Path, *, remote: str | None = None) -> dict[str, Any]`
- Purpose and contract: observe launch/status/progress/log/checkpoint evidence for managed runs, submit only the source-owned observation fields to the canonical manifest owner, and build the returned payload and report from the rows actually committed.
- Important inputs/outputs: run directory and optional remote host in; JSON-ready monitor payload out.
- Side effects: commits observations through `experiment_workspace.merge_run_manifest` and writes `reports/monitor.md`; it never writes a complete tracking snapshot directly.
- Key callers/callees: called by `agent_tools experiment-monitor`; delegates row aggregation/reporting to `experiment_tracking` and process observation to `run_evidence.status_row`.
- Reuse guidance: use for experiment-level status snapshots after launch or W&B sync.
- Duplication-risk notes: do not add a separate experiment status schema.

## `agent_tools.experiments.rank_experiment_candidates`

- File: `agent_tools/experiments.py`
- Signature: `rank_experiment_candidates(run_dir: str | Path, *, metric: str, mode: str, remote: str | None = None) -> Path`
- Purpose and contract: preflight metric/checkpoint evidence paths plus ranking outputs before reading or writing, then rank experiment metric rows by step-scoped run identity and attach matching checkpoint paths.
- Important inputs/outputs: run directory, metric, mode, and optional remote host in; `reports/experiment_ranking.csv` path out.
- Side effects: reads metric/checkpoint manifests and writes ranking/report files.
- Key callers/callees: called by `agent_tools experiment-rank`; delegates candidate/checkpoint matching and report rendering to `experiment_tracking`.
- Reuse guidance: use for cross-run ranking from synced W&B/history metrics.
- Duplication-risk notes: keep `experiment_ranking.*` separate from the step-scoped hparam `ranking.csv` selection artifact.

## `agent_tools.adaptive_hparam.init_adaptive_workflow`

- File: `agent_tools/adaptive_hparam.py`
- Signature: `init_adaptive_workflow(recipe_path: str | Path, output_dir: str | Path) -> Path`
- Purpose and contract: preflight the recipe without writes, then initialize an adaptive hparam workflow with round 000 generated from that recipe. Later reads require absolute workflow/recipe locators and validate every registry row against canonical workspace ownership and frozen current-plan fields before mutation.
- Important inputs/outputs: recipe path and workflow root in; workflow root path out.
- Side effects: validates adaptive settings, writes round recipe/plan, `adaptive/workflow.json`, `run_registry.tsv`, README, and experiment-level event rows.
- Key callers/callees: called by `agent_tools hparam-adaptive-init`; uses `preflight_plan` before workspace creation and `build_plan` for the frozen round.
- Reuse guidance: use when a recipe explicitly enables adaptive hparam tuning.
- Duplication-risk notes: adaptive workflows are append-only and marked `external_optimized=true`; do not overwrite round history.

## `agent_tools.adaptive_hparam.digest_hparam_run`

- File: `agent_tools/adaptive_hparam.py`
- Signature: `digest_hparam_run(run_dir: str | Path) -> Path`
- Purpose and contract: preflight digest, incumbent, and event outputs before monitoring can commit canonical state, then summarize one adaptive round's run state, metrics, checkpoints, logs, and params.
- Important inputs/outputs: workflow root or round directory in; digest CSV path out.
- Side effects: monitors run status, writes `adaptive/digests/round_*.csv`, Markdown digest, event rows, and incumbents.
- Key callers/callees: called by `agent_tools hparam-digest` and adaptive step.
- Reuse guidance: use before suggesting the next adaptive round.
- Duplication-risk notes: preserve `external_optimized` fields in digest outputs.

## `agent_tools.adaptive_hparam.suggest_next_round`

- File: `agent_tools/adaptive_hparam.py`
- Signature: `suggest_next_round(workflow_dir: str | Path) -> Path`
- Purpose and contract: generate a deterministic next-round recipe around the current best scored runs.
- Important inputs/outputs: workflow root in; suggestion YAML path out.
- Side effects: reads latest digest and writes `adaptive/suggestions/round_*.yaml` plus rationale Markdown and event rows.
- Key callers/callees: called by `agent_tools hparam-suggest` and adaptive step.
- Reuse guidance: use for controlled adaptive search expansion rather than hand-editing next-round recipes.
- Duplication-risk notes: keep parameter-neighborhood generation in `_suggest_parameters`.

## `agent_tools.adaptive_hparam.adaptive_step`

- File: `agent_tools/adaptive_hparam.py`
- Signature: `adaptive_step(workflow_dir: str | Path, *, execute: bool = False) -> Path`
- Purpose and contract: perform one adaptive iteration: preflight deterministic outputs, monitor/digest, and suggest in preview mode; with `execute=True`, require the complete prospective replacement round to fit `max_rounds` and `max_runs_total`, plan and register it, then first launch against real free capacity. Prospective rounds use the next number above both registered rows and existing `round_NNN` directories, so an uncommitted failed directory remains an untouched single-use artifact and the next step creates a fresh round; `max_rounds` counts committed rounds rather than numeric gaps. If plan registration or a zero-start launch fails, any canonical `planned/pending` rows from that uncommitted plan are superseded before the error returns. If replacements remain pending in a full pool, stop at most one evidence-qualified bad running row before an ordinary launch retry and require a newly confirmed replacement before any further retirement; apart from the single row drained for the current launch attempt, confirmed replacements and retired bad rows remain one-to-one. The first replacement confirmed `launched` or `running` commits the round through the existing `launch_round` event even when another replacement records `launch_failed`, after which current pending runs may be superseded. Launcher failures report confirmed stopped runs and superseded current pending runs separately; an existing commit is never rolled back. A pre-drain failure with no canonically confirmed replacement leaves old runs unchanged and the prior round current. A zero-start failure after a drain keeps the recorded stopped run while terminalizing the uncommitted plan's pending rows; a later failure after commit keeps the new round current and leaves remaining old runs unchanged.
- Important inputs/outputs: workflow root and execute flag in; suggestion path out.
- Side effects: writes digest/suggestion/events; only execute mode may stop or supersede runs, write a next-round plan, and launch runs.
- Key callers/callees: called by `agent_tools hparam-adaptive-step` and adaptive loop.
- Reuse guidance: use as the unit operation for adaptive tuning.
- Duplication-risk notes: run stops must go through `stop_hparam_run` and remain PID-manifest bounded.

## `agent_tools.adaptive_hparam.adaptive_loop`

- File: `agent_tools/adaptive_hparam.py`
- Signature: `adaptive_loop(workflow_dir: str | Path, *, execute: bool = False) -> Path`
- Purpose and contract: repeatedly run adaptive steps until round or total-run budget is exhausted.
- Important inputs/outputs: workflow root and execute flag in; last produced path out.
- Side effects: repeats adaptive step side effects and appends loop completion events.
- Key callers/callees: called by `agent_tools hparam-adaptive-loop`.
- Reuse guidance: use for unattended adaptive workflows only after the recipe budget and external-feedback policy are explicit.
- Duplication-risk notes: do not bypass `adaptive_step`; it owns budget, replacement, suggestion, and launch ordering.

## `agent_tools.skills.validate_skills`

- File: `agent_tools/skills.py`
- Signature: `validate_skills() -> dict[str, Any]`
- Purpose and contract: validate checked-in agent skill manifests and examples.
- Important inputs/outputs: no direct input; validation summary dict out.
- Side effects: reads `skills/manifest.yaml` and skill files.
- Key callers/callees: called by `agent_tools skills --validate`.
- Reuse guidance: run after editing `skills/` or examples.
- Duplication-risk notes: keep skill schema checks here, not in general config validation.

## `agent_tools.repo.repo_summary`

- File: `agent_tools/repo.py`
- Signature: `repo_summary() -> dict[str, Any]`
- Purpose and contract: collect lightweight Git branch/commit/status information for context bundles.
- Important inputs/outputs: no direct input; JSON-ready repo summary out.
- Side effects: runs read-only Git commands.
- Key callers/callees: called by `agent_tools repo-summary` and context bundles.
- Reuse guidance: use for agent artifacts that need branch provenance.
- Duplication-risk notes: do not make this depend on experiment runtime state.
