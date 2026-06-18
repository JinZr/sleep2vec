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
- Purpose and contract: enforce stop-and-consult policy for high-impact recipe and runtime decisions, including variantless `sleep2stat` gates.
- Important inputs/outputs: task, recipe, config summary, CLI/user decisions, policy, and defaults in; `DecisionReport` with resolved decisions, issues, status, and exit-code semantics out.
- Side effects: none.
- Key callers/callees: called by `agent_tools.plans.evaluate_recipe` and `build_context`; delegates task-specific gates to same-file private helpers and uses path checks.
- Reuse guidance: use this before generating runnable preset, finetune, inference, evaluation, hparam, or sleep2stat commands.
- Duplication-risk notes: do not reimplement high-impact decision checks in recipe templates or shell generation; add task-specific gates through the private helpers in `agent_tools.decisions`, while sleep2stat structural validation remains in `sleep2stat.config.load_config()`.

## `agent_tools.plans.build_context`

- File: `agent_tools/plans.py`
- Signature: `build_context(*, task: str, config: str | Path | None, output_dir: str | Path, label_name: str | None = None, variant: str | None = None, user_decisions_path: str | Path | None = None) -> DecisionReport`
- Purpose and contract: gather repo/config/index/preset context, run consultation gates, and write an agent context bundle.
- Important inputs/outputs: task, optional config/label/variant/user-decisions, and output directory in; `DecisionReport` out.
- Side effects: writes `context.json`, `context.md`, and either runnable `commands.sh` / `validation.sh` or blocked questions/scripts.
- Key callers/callees: called by `agent_tools context`; uses `repo_summary`, `config_summary`, `index_summary`, `preset_summary`, `list_skills`, and `_commands_for_recipe`.
- Reuse guidance: use for pre-command context bundles rather than assembling context ad hoc.
- Duplication-risk notes: keep generated command routing inside `_commands_for_recipe` so variant and sleep2stat variantless rules stay centralized.

## `agent_tools.plans.build_plan`

- File: `agent_tools/plans.py`
- Signature: `build_plan(*, recipe_path: str | Path, output_dir: str | Path, user_decisions_path: str | Path | None = None, allow_unresolved: bool = False, unlock_final_test: bool = False) -> DecisionReport`
- Purpose and contract: evaluate a task recipe and write a runnable or blocked command plan.
- Important inputs/outputs: recipe path, output directory, optional decisions, draft allowance, and final-test unlock in; `DecisionReport` out.
- Side effects: writes `plan.json`, `plan.md`, `run.sh`, hparam trial scripts/configs, or blocked plan/questions.
- Key callers/callees: called by `agent_tools plan` and adaptive init/step; uses consultation gates, output overwrite guards, sleep2stat run-dir guards, final-test gates, and hparam plan writers.
- Reuse guidance: use for all recipe-backed command generation.
- Duplication-risk notes: do not emit executable training or sleep2stat scripts outside this path unless a recipe has already passed the same gates.

## `agent_tools.plans._commands_for_recipe`

- File: `agent_tools/plans.py`
- Signature: `_commands_for_recipe(recipe: dict, cfg: dict | None = None, decisions: dict | None = None) -> list[str]`
- Purpose and contract: render the approved command list for supported recipe tasks, including variant-aware model commands, preset commands, and variantless sleep2stat commands. Sleep2stat post-run summarize and plot commands are skipped for dry-run recipes; summarize is a read-only run-dir overview and does not inherit runtime worker settings.
- Important inputs/outputs: recipe, optional config summary, and resolved decisions in; shell command strings out.
- Side effects: none.
- Key callers/callees: called by `build_context` and `build_plan`; for sleep2stat it calls `_sleep2stat_config_run_dir` and `_sleep2stat_runtime_args`.
- Reuse guidance: add new generated-command behavior here after consultation gates can prove the task is safe.
- Duplication-risk notes: do not render runnable sleep2stat, finetune, infer, hparam, or preset scripts outside this function family.

## `agent_tools.plans.collect_runs`

- File: `agent_tools/plans.py`
- Signature: `collect_runs(root: str | Path, metric: str | None, output: str | Path) -> None`
- Purpose and contract: collect run manifests, hparam status tables, threshold summaries, and inference overviews into one CSV.
- Important inputs/outputs: run root, optional metric, and output path in; CSV file out.
- Side effects: scans filesystem under `root` and writes the requested CSV.
- Key callers/callees: called by `agent_tools collect-runs`; reads W&B summary files when present.
- Reuse guidance: use for lightweight local run inventory before ranking or manual review.
- Duplication-risk notes: do not treat this as a source of truth when `agent_tools.experiments` manifests are available.

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
- Purpose and contract: summarize YAML task, channels, backend, preset/index inputs, variant guess, or sleep2stat run/data/analyzer/output fields without importing Torch/Lightning.
- Important inputs/outputs: config path in; JSON-ready summary out.
- Side effects: reads the YAML file.
- Key callers/callees: called by `plans`, `doctor`, and CLI summary commands.
- Reuse guidance: use for agent policy checks that need config facts without loading models. For sleep2stat-shaped YAML, this calls `sleep2stat.config.load_config()` and returns blocking issues instead of raising through plan generation.
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
- Signature: `index_summary(index_paths: list[str | Path], *, config: str | Path | None = None, label_name: str | None = None, sample_path_check: int = 0, sample_npz_check: int = 0) -> dict[str, Any]`
- Purpose and contract: summarize index CSV row counts, columns, labels, source/split fields, numeric shifts, and optional sample path/NPZ checks.
- Important inputs/outputs: one or more CSV paths plus optional config/label/check counts in; JSON-ready summary out.
- Side effects: reads CSV files and optionally probes sample paths/NPZ files.
- Key callers/callees: called by `agent_tools index-summary` and context bundles.
- Reuse guidance: use before preset/finetune planning when index contents are part of the decision surface.
- Duplication-risk notes: keep deeper sample filtering in `data.utils` and preprocessing CLIs, not here.

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

## `agent_tools.hparam.launch_hparam_trials`

- File: `agent_tools/hparam.py`
- Signature: `launch_hparam_trials(plan_dir: str | Path, *, dry_run: bool = True) -> Path`
- Purpose and contract: convert a generated hparam plan into local or SSH trial launch rows.
- Important inputs/outputs: plan directory and dry-run flag in; `launch_manifest.tsv` path out.
- Side effects: creates log/PID directories, writes `launch_manifest.tsv` and `trial_status.tsv`, and starts processes only when `dry_run=False`.
- Key callers/callees: called by `agent_tools hparam-launch` and adaptive step; uses `_launch_command`, `_assigned_gpus`, and `_start_process`.
- Reuse guidance: use after `build_plan` has produced a hparam `plan.json`.
- Duplication-risk notes: do not launch trials by scanning scripts directly; this function owns PID/log manifest contracts.

## `agent_tools.hparam.monitor_hparam_trials`

- File: `agent_tools/hparam.py`
- Signature: `monitor_hparam_trials(run_dir: str | Path, *, once: bool = True, health: bool = False) -> Path`
- Purpose and contract: update hparam trial status from PIDs, logs, manifests, checkpoints, and optional health probes.
- Important inputs/outputs: run directory plus monitor flags in; `trial_status.tsv` path out.
- Side effects: reads process/log/run artifacts, may launch pending trials when capacity exists, and rewrites launch/status TSVs.
- Key callers/callees: called by `agent_tools hparam-monitor`, adaptive digest, and experiment monitoring helpers.
- Reuse guidance: use before selection/ranking when launch manifests are the active source of truth.
- Duplication-risk notes: keep remote probes inside the existing timeout-bounded helpers.

## `agent_tools.hparam.stop_hparam_trial`

- File: `agent_tools/hparam.py`
- Signature: `stop_hparam_trial(run_dir: str | Path, trial_id: str) -> Path`
- Purpose and contract: terminate one PID recorded in a hparam launch manifest.
- Important inputs/outputs: run directory and trial id in; updated `trial_status.tsv` path out.
- Side effects: sends `SIGTERM` locally or `kill -TERM` through SSH and marks the trial stopped.
- Key callers/callees: called by `agent_tools hparam-stop` and adaptive replacement logic.
- Reuse guidance: use this for trial termination so PID provenance is checked first.
- Duplication-risk notes: do not kill arbitrary command-line matches outside the manifest.

## `agent_tools.hparam.select_hparam_candidates`

- File: `agent_tools/hparam.py`
- Signature: `select_hparam_candidates(run_dir: str | Path, metric: str, mode: str) -> Path`
- Purpose and contract: rank trials by a requested metric from run manifests and record fixed checkpoint paths.
- Important inputs/outputs: run directory, metric, and mode in; `candidate_ranking.csv` path out.
- Side effects: reads plan and run manifests; writes candidate ranking CSV.
- Key callers/callees: called by `agent_tools hparam-select`; uses `_find_run_manifest`, `_metric_value`, and `_fixed_checkpoint_path`.
- Reuse guidance: use for validation-based selection after trials finish.
- Duplication-risk notes: do not replace selected checkpoints with mutable best aliases.

## `agent_tools.hparam.scan_hparam_checkpoints`

- File: `agent_tools/hparam.py`
- Signature: `scan_hparam_checkpoints(run_dir: str | Path, metric: str, mode: str, *, top_k: int | None = None) -> Path`
- Purpose and contract: rank fixed epoch checkpoints using W&B history, CSV history, or manifest fallback metrics.
- Important inputs/outputs: run directory, metric, mode, and optional top-k in; `checkpoint_ranking.csv` path out.
- Side effects: reads local histories/manifests and writes ranking CSV.
- Key callers/callees: called by `agent_tools hparam-checkpoint-scan`.
- Reuse guidance: use when checkpoint-level selection matters more than run-level selection.
- Duplication-risk notes: keep epoch parsing in the shared checkpoint helpers.

## `agent_tools.hparam.generate_external_eval`

- File: `agent_tools/hparam.py`
- Signature: `generate_external_eval(run_dir: str | Path, selected_csv: str | Path, *, unlock_final_test: bool, kaldi_data_root: str | None = None, kaldi_manifest: str | None = None, finetune_data_index: str | None = None, eval_split: str = "test", top_k: int = 1, all_candidates: bool = False) -> Path`
- Purpose and contract: create locked final/external inference configs and commands for selected candidates.
- Important inputs/outputs: hparam run directory, selected candidates, explicit unlock, optional replacement data paths, split, base runtime settings, selected-row `runtime.*` overrides, and candidate selection controls in; `external_eval.sh` path out.
- Side effects: writes copied configs, `external_eval_manifest.tsv`, and executable shell script.
- Key callers/callees: called by `agent_tools hparam-external-eval`; uses `_copy_config_with_data_paths`, `module_for_variant`, and the infer runtime CLI renderer.
- Reuse guidance: use for final-test or external-test command generation after an explicit unlock.
- Duplication-risk notes: final-test evaluation must stay locked by this explicit flag.

## `agent_tools.hparam.export_hparam_logits`

- File: `agent_tools/hparam.py`
- Signature: `export_hparam_logits(run_dir: str | Path, selected_csv: str | Path, *, unlock_final_test: bool, val_split: str = "val", test_split: str = "test", skip_test: bool = False, label_name: str | None = None, val_kaldi_data_root: str | None = None, val_kaldi_manifest: str | None = None, val_finetune_data_index: str | None = None, test_kaldi_data_root: str | None = None, test_kaldi_manifest: str | None = None, test_finetune_data_index: str | None = None, batch_size: int = 12, num_workers: int = 8, devices: list[int] | None = None, accelerator: str = "gpu", device: str = "cuda", precision: str = "bf16-mixed", seed: int = 4523, top_k: int = 1, all_candidates: bool = False, execute: bool = False) -> Path`
- Purpose and contract: prepare or execute inference commands that export validation and optional test logits for selected hparam candidates.
- Important inputs/outputs: selected candidates, split/data overrides, runtime controls, unlock/skip-test, and execution flag in; `logits_export_manifest.tsv` path out.
- Side effects: writes copied configs and manifests; with `execute=True`, runs inference commands and copies produced prediction CSVs to logits paths.
- Key callers/callees: called by `agent_tools hparam-export-logits`; uses `_infer_command`, `_execute_logit_exports`, and `_copy_logits_csv`.
- Reuse guidance: use before threshold fitting or ensembling selected models.
- Duplication-risk notes: test logits require `--unlock-final-test` unless explicitly skipped.

## `agent_tools.hparam.threshold_hparam_outputs`

- File: `agent_tools/hparam.py`
- Signature: `threshold_hparam_outputs(run_dir: str | Path, selected_csv: str | Path) -> Path`
- Purpose and contract: fit a binary F1 threshold on validation predictions/logits and apply it to test predictions/logits.
- Important inputs/outputs: run directory and selected CSV in; `threshold_summary.csv` path out.
- Side effects: reads prediction CSVs and writes threshold summary.
- Key callers/callees: called by `agent_tools hparam-threshold`; uses `_read_binary_predictions`, `_best_f1_threshold`, and `_binary_metrics`.
- Reuse guidance: use for post-hparam binary threshold selection.
- Duplication-risk notes: keep exploratory thresholding separate from training metrics.

## `agent_tools.hparam.ensemble_hparam_outputs`

- File: `agent_tools/hparam.py`
- Signature: `ensemble_hparam_outputs(run_dir: str | Path, candidates_csv: str | Path, *, search_combinations: bool = False, max_size: int | None = None, metric: str = "exploratory_test_auroc", mode: str = "max", top_k: int | None = None) -> Path`
- Purpose and contract: average binary prediction probabilities across candidate outputs, optionally searching combinations.
- Important inputs/outputs: candidate CSV, combination controls, rank metric/mode, and top-k in; `ensemble_summary.csv` path out.
- Side effects: reads prediction CSVs and writes ensemble summary.
- Key callers/callees: called by `agent_tools hparam-ensemble`; uses `_average_binary_predictions` and `_ensemble_summary_row`.
- Reuse guidance: use for exploratory ensembling after logits/predictions are exported.
- Duplication-risk notes: treat this as analysis tooling, not a trainer/runtime model averaging feature.

## `agent_tools.experiments.init_experiment`

- File: `agent_tools/experiments.py`
- Signature: `init_experiment(run_dir: str | Path, name: str, *, remote: str | None = None) -> Path`
- Purpose and contract: initialize an experiment manifest and standard report/W&B directories locally or remotely.
- Important inputs/outputs: run directory, experiment id, optional remote host in; `experiment_manifest.tsv` path out.
- Side effects: creates directories and writes/updates the experiment manifest.
- Key callers/callees: called by `agent_tools experiment-init`.
- Reuse guidance: use before experiment-level monitor/rank operations.
- Duplication-risk notes: keep remote writes routed through the shared remote read/write helpers.

## `agent_tools.experiments.sync_wandb_runs`

- File: `agent_tools/experiments.py`
- Signature: `sync_wandb_runs(run_dir: str | Path, *, entity: str, project: str, group: str | None = None, remote: str | None = None) -> Path`
- Purpose and contract: sync W&B run metadata, summaries, and history into local or remote experiment manifests.
- Important inputs/outputs: run directory plus W&B entity/project/group and optional remote host in; `wandb/runs.tsv` path out.
- Side effects: calls `wandb.Api()`, writes `wandb/runs.tsv`, `wandb/summaries.jsonl`, history CSVs, `metrics_manifest.tsv`, `run_manifest.tsv`, and W&B report files.
- Key callers/callees: called by `agent_tools experiment-wandb-sync`.
- Reuse guidance: use W&B API data rather than browser pages for numeric metrics.
- Duplication-risk notes: avoid scraping W&B UI or mixing browser-rendered values into ranking.

## `agent_tools.experiments.index_checkpoints`

- File: `agent_tools/experiments.py`
- Signature: `index_checkpoints(run_dir: str | Path, *, remote: str | None = None) -> Path`
- Purpose and contract: index checkpoint files and attach the best available metric evidence.
- Important inputs/outputs: run directory and optional remote host in; `checkpoint_manifest.tsv` path out.
- Side effects: scans local files or a remote checkpoint tree and writes checkpoint manifest.
- Key callers/callees: called by `agent_tools experiment-index-checkpoints`; uses metric manifests and checkpoint name parsers.
- Reuse guidance: use before experiment-level candidate ranking.
- Duplication-risk notes: remote checkpoint scans should stay shallow and command-bounded.

## `agent_tools.experiments.monitor_experiment`

- File: `agent_tools/experiments.py`
- Signature: `monitor_experiment(run_dir: str | Path, *, remote: str | None = None) -> dict[str, Any]`
- Purpose and contract: merge launch/status/W&B/progress/log/checkpoint evidence into a current experiment run manifest.
- Important inputs/outputs: run directory and optional remote host in; JSON-ready monitor payload out.
- Side effects: rewrites `run_manifest.tsv` and writes `reports/monitor.md`.
- Key callers/callees: called by `agent_tools experiment-monitor`; reuses hparam `_status_row`.
- Reuse guidance: use for experiment-level status snapshots after launch or W&B sync.
- Duplication-risk notes: do not add a separate experiment status schema.

## `agent_tools.experiments.rank_experiment_candidates`

- File: `agent_tools/experiments.py`
- Signature: `rank_experiment_candidates(run_dir: str | Path, *, metric: str, mode: str, remote: str | None = None) -> Path`
- Purpose and contract: rank experiment metric rows and attach matching checkpoint paths.
- Important inputs/outputs: run directory, metric, mode, and optional remote host in; `candidate_ranking.tsv` path out.
- Side effects: reads metric/checkpoint manifests and writes ranking/report files.
- Key callers/callees: called by `agent_tools experiment-rank`.
- Reuse guidance: use for cross-run ranking from synced W&B/history metrics.
- Duplication-risk notes: keep run-level ranking and hparam plan ranking separate unless their manifests are merged intentionally.

## `agent_tools.adaptive_hparam.init_adaptive_workflow`

- File: `agent_tools/adaptive_hparam.py`
- Signature: `init_adaptive_workflow(recipe_path: str | Path, output_dir: str | Path) -> Path`
- Purpose and contract: initialize an adaptive hparam workflow with round 000 generated from a recipe.
- Important inputs/outputs: recipe path and workflow root in; workflow root path out.
- Side effects: validates adaptive settings, writes round recipe/plan, `adaptive/workflow.json`, `events.jsonl`, `trial_registry.tsv`, and README.
- Key callers/callees: called by `agent_tools hparam-adaptive-init`; uses `build_plan`.
- Reuse guidance: use when a recipe explicitly enables adaptive hparam tuning.
- Duplication-risk notes: adaptive workflows are append-only and marked `external_optimized=true`; do not overwrite round history.

## `agent_tools.adaptive_hparam.digest_hparam_run`

- File: `agent_tools/adaptive_hparam.py`
- Signature: `digest_hparam_run(run_dir: str | Path) -> Path`
- Purpose and contract: summarize one adaptive round's trial state, metrics, checkpoints, logs, and params.
- Important inputs/outputs: workflow root or round directory in; digest CSV path out.
- Side effects: monitors trial status, writes `adaptive/digests/round_*.csv`, Markdown digest, event rows, and incumbents.
- Key callers/callees: called by `agent_tools hparam-digest` and adaptive step.
- Reuse guidance: use before suggesting the next adaptive round.
- Duplication-risk notes: preserve `external_optimized` fields in digest outputs.

## `agent_tools.adaptive_hparam.suggest_next_round`

- File: `agent_tools/adaptive_hparam.py`
- Signature: `suggest_next_round(workflow_dir: str | Path) -> Path`
- Purpose and contract: generate a deterministic next-round recipe around the current best scored trials.
- Important inputs/outputs: workflow root in; suggestion YAML path out.
- Side effects: reads latest digest and writes `adaptive/suggestions/round_*.yaml` plus rationale Markdown and event rows.
- Key callers/callees: called by `agent_tools hparam-suggest` and adaptive step.
- Reuse guidance: use for controlled adaptive search expansion rather than hand-editing next-round recipes.
- Duplication-risk notes: keep parameter-neighborhood generation in `_suggest_parameters`.

## `agent_tools.adaptive_hparam.adaptive_step`

- File: `agent_tools/adaptive_hparam.py`
- Signature: `adaptive_step(workflow_dir: str | Path, *, execute: bool = False) -> Path`
- Purpose and contract: perform one adaptive iteration: monitor/digest, suggest, supersede pending trials, and optionally plan/launch the next round.
- Important inputs/outputs: workflow root and execute flag in; suggestion path out.
- Side effects: writes digest/suggestion/events; with execute, may stop bad running trials, write next-round plan, and launch trials.
- Key callers/callees: called by `agent_tools hparam-adaptive-step` and adaptive loop.
- Reuse guidance: use as the unit operation for adaptive tuning.
- Duplication-risk notes: trial stops must go through `stop_hparam_trial` and remain PID-manifest bounded.

## `agent_tools.adaptive_hparam.adaptive_loop`

- File: `agent_tools/adaptive_hparam.py`
- Signature: `adaptive_loop(workflow_dir: str | Path, *, execute: bool = False) -> Path`
- Purpose and contract: repeatedly run adaptive steps until round or total-trial budget is exhausted.
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
