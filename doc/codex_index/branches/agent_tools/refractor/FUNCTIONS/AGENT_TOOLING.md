# Agent Tooling Function Catalog

This catalog covers the reusable contract-bearing entrypoints behind `python -m agent_tools`. Private one-line adapters and formatting helpers are omitted unless they own an important routing or safety boundary.

## CLI and Decision Models

### `agent_tools.cli.main`

- **File/signature:** `agent_tools/cli.py`; `main(argv: list[str] | None = None) -> int`.
- **Purpose/contract:** parse the supported subcommand surface and dispatch to a thin `_cmd_*` adapter.
- **Inputs/outputs:** optional argv; integer process exit code.
- **Side effects:** delegated command I/O and stdout/stderr rendering.
- **Callers/callees:** `agent_tools.__main__`; calls public functions from plans, hparam, experiments, adaptive, progress, and summary modules.
- **Reuse guidance:** wire a CLI command only after its reusable Python operation exists in the responsibility owner.
- **Duplication risk:** workflow logic in `_cmd_*` creates behavior that Python callers cannot reuse or test independently.

### `agent_tools.decision_models.DecisionReport`

- **File/signature:** `agent_tools/decision_models.py`; dataclass `DecisionReport(status: DecisionStatus, issues: list[DecisionIssue], decisions: dict[str, ResolvedDecision])`.
- **Purpose/contract:** carry aggregate consultation status, issues, resolved decisions, and stable exit semantics (`FAIL -> 1`, unresolved input -> `2`, otherwise `0`).
- **Inputs/outputs:** structured issues and decisions; `exit_code` and `blocking_issues()` views.
- **Side effects:** none.
- **Callers/callees:** produced by decision and plan functions; consumed by CLI and Markdown/JSON writers.
- **Reuse guidance:** represent consultation and plan-gate outcomes with this type instead of returning ad hoc tuples or booleans.
- **Duplication risk:** alternate status precedence or exit-code maps can make CLI and Python behavior disagree.

## Authored Inputs and Summaries

### `agent_tools.recipes.load_recipe_with_base`

- **File/signature:** `agent_tools/recipes.py`; `load_recipe_with_base(path: str | Path) -> dict[str, Any]`.
- **Purpose/contract:** read a managed YAML recipe, reject authored `_...` metadata, require a path-like `base_recipe`, deep-merge an optional base, and retain base/local source snapshots plus the recipe path.
- **Inputs/outputs:** recipe path; effective recipe mapping with internal provenance fields.
- **Side effects:** reads one or two YAML files.
- **Callers/callees:** plans and adaptive workflows; uses `load_recipe`, `_resolve_base_recipe_path`, `_deep_merge`, and authoritative YAML parsing.
- **Reuse guidance:** every recipe-backed operation should use this loader so merge precedence and provenance remain consistent.
- **Duplication risk:** command-local merges lose source-layer information and may accept a different contract.

### `agent_tools.recipes.load_user_decisions`

- **File/signature:** `agent_tools/recipes.py`; `load_user_decisions(path: str | Path | None) -> dict[str, Any]`.
- **Purpose/contract:** require an optional user-decision document to contain exactly one top-level `decisions` mapping; task-aware decision-name and entry closure follows in plans/decisions.
- **Inputs/outputs:** optional path; empty mapping or decision mapping.
- **Side effects:** reads YAML when a path is present.
- **Callers/callees:** `plans.evaluate_recipe` and `plans.build_context`; delegates YAML parsing to `load_yaml_file`.
- **Reuse guidance:** keep user-decision input on this single boundary, then materialize it through plan/decision owners.
- **Duplication risk:** direct reads can bypass decision precedence or task applicability checks.

### `agent_tools.configs.config_summary`

- **File/signature:** `agent_tools/configs.py`; `config_summary(config_path: str | Path, *, variant: str | None = None, validate_survival_local_paths: bool = True) -> dict[str, Any]`.
- **Purpose/contract:** summarize task, model channels, backend, finetune semantics, sidecars, averaging, LoRA, preset-build, sex-age, or sleep2stat config facts without constructing training models.
- **Inputs/outputs:** config path and optional variant/path-check switch; JSON-ready summary with warnings and blocking issues.
- **Side effects:** reads YAML and optional local sidecar paths; sleep2stat shapes delegate to `sleep2stat.config.load_config` through `sleep2stat_config_summary`.
- **Callers/callees:** plan context/evaluation and CLI config summary.
- **Reuse guidance:** use for agent policy facts instead of importing Torch/Lightning or inferring semantics from filenames.
- **Duplication risk:** independent config readers drift from runtime schemas and variant behavior.

### `agent_tools.index_csv.index_summary`

- **File/signature:** `agent_tools/index_csv.py`; `index_summary(index_paths: list[str | Path], *, config: str | Path | None = None, label_name: str | None = None, split_values: list[str] | None = None, preset_path: str | Path | None = None, sample_path_check: int = 0, sample_npz_check: int = 0) -> dict[str, Any]`.
- **Purpose/contract:** produce lightweight row, column, split, label, shift, sidecar-key, and optional sample-path/NPZ evidence for planning.
- **Inputs/outputs:** one or more index paths and optional config/split/check settings; JSON-ready summary.
- **Side effects:** reads CSVs, optional preset/config/sidecars, and a bounded sample of paths/NPZ files.
- **Callers/callees:** CLI and `plan_context`.
- **Reuse guidance:** use before preset/finetune planning when index content is part of a high-impact decision.
- **Duplication risk:** shell summaries often ignore split filtering, sex-age metadata, survival, or multilabel sidecars.

### `agent_tools.presets.preset_summary`

- **File/signature:** `agent_tools/presets.py`; `preset_summary(preset_path: str | Path) -> dict[str, Any]`.
- **Purpose/contract:** inspect a preset pickle enough to report row/count/channel/source-style facts without rewriting it.
- **Inputs/outputs:** preset path; JSON-ready summary.
- **Side effects:** reads pickle content.
- **Callers/callees:** CLI and plan context.
- **Reuse guidance:** use for diagnostics only; runtime dataset code remains the preset payload validator.
- **Duplication risk:** making this a second preset schema/runtime would split ownership.

### `agent_tools.skills.validate_skills`

- **File/signature:** `agent_tools/skills.py`; `validate_skills() -> dict[str, Any]`.
- **Purpose/contract:** validate the skill manifest, required skill headings, AGENTS owner names, referenced index files, and example task/variant shape.
- **Inputs/outputs:** repository skill files; result mapping with `ok`, issues, and listed skills.
- **Side effects:** reads repository documentation/YAML.
- **Callers/callees:** `agent_tools skills --validate`.
- **Reuse guidance:** run after recipe/skill/index changes.
- **Duplication risk:** a separate doc checker may disagree on required headings or variant rules.

## Consultation and Recipe Evaluation

### `agent_tools.decisions.evaluate_consultation_gates`

- **File/signature:** `agent_tools/decisions.py`; `evaluate_consultation_gates(task: str | None, recipe: dict | None, config_summary: dict | None, cli_args: dict | None, policy: dict, *, require_experiment: bool = True) -> DecisionReport`.
- **Purpose/contract:** resolve task and high-impact decisions, enforce supported task/variant/runtime boundaries, aggregate task/path/hparam issues, and optionally require experiment metadata.
- **Inputs/outputs:** task, effective recipe, lightweight config facts, CLI/user decisions, and consultation policy; `DecisionReport`.
- **Side effects:** none except bounded path checks delegated to `decision_paths` may use SSH when explicitly configured.
- **Callers/callees:** plans; delegates to `decision_paths.path_issues`, task rules in `decision_rules`, hparam rules in `decision_hparam`, and base-finetune recursion.
- **Reuse guidance:** call before generating commands for supported recipe tasks.
- **Duplication risk:** policy checks in templates or renderers can silently diverge from consultation status.

### `agent_tools.plans.evaluate_recipe`

- **File/signature:** `agent_tools/plans.py`; `evaluate_recipe(recipe_path: str | Path, user_decisions_path: str | Path | None = None) -> tuple[dict, dict | None, DecisionReport]`.
- **Purpose/contract:** load the recipe, resolve task ownership without inheriting a base finetune task into a local hparam overlay, close base/local/effective authored fields through their existing owners before config reads, materialize only task-owned fields, summarize config/index facts, run consultation, and compare key recipe decisions with finetune config semantics.
- **Inputs/outputs:** recipe and optional user-decision paths; effective recipe, optional config summary, and report.
- **Side effects:** read-only local/optional path probes; it does not create a workspace.
- **Callers/callees:** doctor and `preflight_plan`; calls recipe loaders, `_materialize_decisions`, config/context summaries, consultation, and hparam YAML-override checks.
- **Reuse guidance:** use as the shared doctor/evaluation path; put section validation in existing owners rather than a second evaluation facade.
- **Duplication risk:** evaluating only the merged recipe can hide source-layer mistakes; evaluating raw layers with full semantic requirements can falsely reject inherited fields.

### `agent_tools.plans.build_context`

- **File/signature:** `agent_tools/plans.py`; `build_context(*, task: str, config: str | Path | None, output_dir: str | Path, label_name: str | None = None, variant: str | None = None, user_decisions_path: str | Path | None = None) -> DecisionReport`.
- **Purpose/contract:** create a diagnostic context bundle from a synthetic recipe and consultation result; it is not the managed recipe-plan authorization boundary.
- **Inputs/outputs:** explicit task/config/output and optional label/variant/decisions; report plus context files.
- **Side effects:** writes context Markdown/JSON, questions, validation, and command/blocked scripts as applicable.
- **Callers/callees:** CLI; uses summaries, consultation, `_commands_for_recipe`, `plan_context`, and `plan_rendering`.
- **Reuse guidance:** use for diagnostics; use `build_plan` for recipe-backed managed execution plans.
- **Duplication risk:** adding a separate recipe mode to context would create two recipe-plan APIs.

## Planning and Rendering

### `agent_tools.plans.preflight_plan`

- **File/signature:** `agent_tools/plans.py`; `preflight_plan(*, recipe_path: str | Path, output_dir: str | Path, user_decisions_path: str | Path | None = None, allow_unresolved: bool = False, unlock_final_test: bool = False) -> tuple[dict, dict | None, DecisionReport]`.
- **Purpose/contract:** prove recipe, workspace containment, config freeze, hparam/final-test, command support, and output topology before mutation.
- **Inputs/outputs:** recipe/output paths and plan switches; effective recipe, optional config summary, and report.
- **Side effects:** reads and validates only.
- **Callers/callees:** `build_plan` and adaptive initialization/round progression; calls `evaluate_recipe`, workspace/output guards, hparam final-test checks, and command rendering checks.
- **Reuse guidance:** every adjacent workflow must call this before creating workspace artifacts, launching, stopping, or superseding runs on behalf of a new recipe.
- **Duplication risk:** partial preflight implementations cause mutation-before-validation regressions.

### `agent_tools.plans.build_plan`

- **File/signature:** `agent_tools/plans.py`; `build_plan(*, recipe_path: str | Path, output_dir: str | Path, user_decisions_path: str | Path | None = None, allow_unresolved: bool = False, unlock_final_test: bool = False) -> DecisionReport`.
- **Purpose/contract:** run shared preflight, stop before workspace creation for pre-workspace failures, otherwise create blocked diagnostics or a frozen managed plan.
- **Inputs/outputs:** recipe/output paths and switches; decision report and plan artifacts.
- **Side effects:** after safe preflight, may initialize workspace/step manifests, freeze configs/scripts/hashes, register runs, write plan/recipe artifacts, and append events.
- **Callers/callees:** CLI and adaptive; delegates hparam compilation to `plan_hparam.write_hparam_plan`, ordinary command rendering to `_commands_for_recipe`, and state commits to `experiment_workspace`.
- **Reuse guidance:** this is the only public recipe-backed plan compiler.
- **Duplication risk:** direct script writers bypass consultation, ownership, atomicity, hash, and run-registration contracts.

### `agent_tools.plans._commands_for_recipe`

- **File/signature:** `agent_tools/plans.py`; `_commands_for_recipe(recipe: dict, cfg: dict | None = None) -> list[str]`.
- **Purpose/contract:** finite dispatcher for preset, finetune, infer/evaluate, and variantless sleep2stat command sequences.
- **Inputs/outputs:** effective recipe and optional config facts; rendered shell command strings.
- **Side effects:** none.
- **Callers/callees:** context and plan paths; uses `plan_rendering` option builders and existing package entrypoints.
- **Reuse guidance:** extend this dispatcher only after consultation can prove the new behavior safe.
- **Duplication risk:** a second task switch breaks variant routing and accepted-field-to-renderer closure.

### `agent_tools.plan_rendering.runtime_cli_args`

- **File/signature:** `agent_tools/plan_rendering.py`; `runtime_cli_args(runtime: dict[str, Any], *, variant: str | None = None) -> list[Any]`.
- **Purpose/contract:** translate finetune/hparam runtime values into canonical CLI arguments, including variant-specific behavior.
- **Inputs/outputs:** runtime mapping and optional variant; list of command parts.
- **Side effects:** none.
- **Callers/callees:** ordinary plan and hparam plan rendering.
- **Reuse guidance:** renderer and validation should share the same field-to-flag truth where practical.
- **Duplication risk:** copied flag lists accept inert fields or omit supported ones.

### `agent_tools.plan_rendering.infer_runtime_cli_args`

- **File/signature:** `agent_tools/plan_rendering.py`; `infer_runtime_cli_args(runtime: dict[str, Any]) -> list[Any]`.
- **Purpose/contract:** render inference runtime flags consistently for recipe plans and hparam postprocessing.
- **Inputs/outputs:** runtime mapping; command parts.
- **Side effects:** none.
- **Callers/callees:** `_commands_for_recipe`, `plan_hparam`, and `hparam_postprocess`.
- **Reuse guidance:** use for every generated infer command.
- **Duplication risk:** final-evaluation commands can drift from ordinary inference.

### `agent_tools.plan_rendering.script_lines`

- **File/signature:** `agent_tools/plan_rendering.py`; `script_lines(commands: list[str], *, run_cwd: str | Path | None = None, experiment_root: str | Path | None = None, step_id: str | None = None, run_id: str | None = None) -> list[str]`.
- **Purpose/contract:** add the shared shell prelude, repository cwd/PYTHONPATH, and optional canonical non-hparam lifecycle wrapper.
- **Inputs/outputs:** commands and optional managed identity; complete shell lines.
- **Side effects:** none.
- **Callers/callees:** context and ordinary plan writers.
- **Reuse guidance:** use instead of recreating shell bootstrap or terminal-state handling.
- **Duplication risk:** custom wrappers may lose the original runtime failure or replay terminal runs.

### `agent_tools.plan_hparam.write_hparam_plan`

- **File/signature:** `agent_tools/plan_hparam.py`; `write_hparam_plan(recipe: dict, out: Path, *, unlock_final_test: bool) -> None`.
- **Purpose/contract:** expand the finite search grid, apply runtime/YAML overrides, freeze per-run config and launch script hashes, create stable/semantic identities, register rows, and emit validation/final-test artifacts.
- **Inputs/outputs:** preflight-approved hparam recipe, plan directory, unlock state; frozen plan tree.
- **Side effects:** writes configs/scripts/plan/recipe artifacts and commits managed run rows/events.
- **Callers/callees:** only `plans.build_plan`; uses rendering, config loaders, identity/workspace owners, and final-test helpers.
- **Reuse guidance:** keep hparam compilation here while `build_plan` remains the public mutation boundary.
- **Duplication risk:** an alternate grid writer will not satisfy `read_hparam_plan` or canonical manifest identity.

## Workspace and Managed I/O

### `agent_tools.experiment_workspace.experiment_metadata_issues`

- **File/signature:** `agent_tools/experiment_workspace.py`; `experiment_metadata_issues(recipe: dict[str, Any], *, require_values: bool = True, source_layer: str | None = None, allow_step_io: bool = False) -> list[dict[str, Any]]`.
- **Purpose/contract:** validate closed experiment/recipe-step fields, required values, identifier spelling, and phase vocabulary without mutation; experiment step-registration explicitly enables its separate `inputs`/`outputs` contract.
- **Inputs/outputs:** recipe; structured issue mappings.
- **Side effects:** none.
- **Callers/callees:** consultation, preflight, workspace creation, and experiment lifecycle validation.
- **Reuse guidance:** keep experiment/step authored-field rules in this owner.
- **Duplication risk:** separate metadata validators can allow incompatible workspace identities.

### `agent_tools.experiment_workspace.canonical_local_experiment_root`

- **File/signature:** `agent_tools/experiment_workspace.py`; `canonical_local_experiment_root(raw: str | Path, base_dir: str | Path) -> Path`.
- **Purpose/contract:** convert a local management root to one absolute normalized path and reject a symlink root.
- **Inputs/outputs:** raw root and base directory; canonical absolute `Path`.
- **Side effects:** filesystem metadata read only.
- **Callers/callees:** public plan, experiment, hparam-postprocess, and adaptive boundaries.
- **Reuse guidance:** canonicalize local management roots here; do not apply it to semantic user data/checkpoint paths.
- **Duplication risk:** mixed lexical/resolved roots break ownership comparisons.

### `agent_tools.experiment_workspace.read_managed_yaml_mapping`

- **File/signature:** `agent_tools/experiment_workspace.py`; `read_managed_yaml_mapping(text: str, *, source: str | Path) -> dict[str, Any]`.
- **Purpose/contract:** parse authoritative YAML as one mapping while rejecting empty, cyclic, aliased, duplicate-key, or otherwise unsafe managed structures.
- **Inputs/outputs:** YAML text and source label; mapping.
- **Side effects:** none.
- **Callers/callees:** recipe/spec/frozen-manifest readers.
- **Reuse guidance:** use at every YAML boundary that establishes managed ownership or authored recipe input.
- **Duplication risk:** `yaml.safe_load` alone can silently accept duplicate keys.

### `agent_tools.experiment_workspace.ensure_experiment_workspace`

- **File/signature:** `agent_tools/experiment_workspace.py`; `ensure_experiment_workspace(recipe: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]`.
- **Purpose/contract:** bind a preflight-approved recipe to an experiment and step, create the canonical directory/manifests when fresh, and reject conflicting ownership.
- **Inputs/outputs:** effective recipe and plan output; experiment root and step manifest path.
- **Side effects:** creates workspace directories/YAML/TSV/README/events or validates existing state.
- **Callers/callees:** `build_plan` and adaptive init; uses metadata/output checks, step merge, and run-manifest initialization.
- **Reuse guidance:** do not create managed workspace directories manually.
- **Duplication risk:** parallel initialization loses single-use, ownership, and step-plan registration guarantees.

### `agent_tools.experiment_workspace.read_run_manifest`

- **File/signature:** `agent_tools/experiment_workspace.py`; `read_run_manifest(root: str | Path, *, remote: str | None = None) -> list[dict[str, str]]`.
- **Purpose/contract:** strictly read the canonical `run_manifest.tsv`, distinguishing missing, empty/corrupt, header-only, and valid states.
- **Inputs/outputs:** workspace root and optional remote host; canonical rows.
- **Side effects:** local read or SSH read.
- **Callers/callees:** all lifecycle, hparam, adaptive, and experiment owners.
- **Reuse guidance:** use for canonical state; auxiliary tables must not replace it.
- **Duplication risk:** generic TSV readers can interpret corrupt or removed identity headers as zero runs.

### `agent_tools.experiment_workspace.merge_run_manifest`

- **File/signature:** `agent_tools/experiment_workspace.py`; `merge_run_manifest(root: str | Path, rows: list[dict[str, Any]], *, remote: str | None = None, lock_held: bool = False) -> list[dict[str, Any]]`.
- **Purpose/contract:** validate one-row-per-run observations, preserve frozen fields and status monotonicity, atomically commit canonical state, then refresh the run matrix.
- **Inputs/outputs:** workspace, observations, optional remote/lock state; committed canonical rows.
- **Side effects:** locks locally or retries conditional remote replacement; writes `run_manifest.tsv` and derived matrix/report.
- **Callers/callees:** plans, hparam runtime/selection, experiment monitoring/W&B, adaptive workflows.
- **Reuse guidance:** this is the only canonical run-state commit path.
- **Duplication risk:** direct table writes can regress terminal state, overwrite frozen identity, or race.

### `agent_tools.experiment_io.read_rows_at`

- **File/signature:** `agent_tools/experiment_io.py`; `read_rows_at(path: str | Path, *, remote: str | None = None, require_managed_identity: bool = False, strict: bool = False) -> list[dict[str, str]]`.
- **Purpose/contract:** share strict CSV/TSV semantics between local and SSH reads, including rectangularity and managed identity headers.
- **Inputs/outputs:** path, remote and strictness switches; row mappings.
- **Side effects:** local file read or bounded SSH read.
- **Callers/callees:** experiment lifecycle/tracking and remote workspace operations.
- **Reuse guidance:** use for authoritative non-run tables; use `read_run_manifest` for canonical state.
- **Duplication risk:** separate remote parsers create different missing/empty/corrupt semantics.

### `agent_tools.experiment_io.validate_managed_output_paths`

- **File/signature:** `agent_tools/experiment_io.py`; `validate_managed_output_paths(root: str | Path, paths: list[str | Path], *, remote: str | None = None) -> None`.
- **Purpose/contract:** reject targets outside the managed root, duplicate/aliased targets, symlinked ancestors, non-regular files, and hard links before a multi-artifact mutation.
- **Inputs/outputs:** root, target list, optional remote; raises on unsafe topology.
- **Side effects:** local metadata inspection or bounded SSH metadata script.
- **Callers/callees:** every multi-output plan, hparam, adaptive, and experiment operation.
- **Reuse guidance:** validate the whole output set before the first write.
- **Duplication risk:** validating files one at a time can leave partial output after a later alias failure.

## Frozen Plans and Runtime Evidence

### `agent_tools.run_artifacts.read_hparam_plan`

- **File/signature:** `agent_tools/run_artifacts.py`; `read_hparam_plan(run_dir: Path) -> dict[str, Any]`.
- **Purpose/contract:** fail closed unless the plan, frozen recipe/base, experiment/step registration, run rows, file locations, and config/script hashes form one consistent current-format hparam plan.
- **Inputs/outputs:** plan directory; validated plan mapping.
- **Side effects:** reads plan, recipe, workspace manifests, configs, scripts, and hashes.
- **Callers/callees:** every hparam runtime/selection/postprocess and adaptive operation.
- **Reuse guidance:** cross this boundary before any hparam state mutation or artifact interpretation.
- **Duplication risk:** reading `plan.json` alone trusts stale or substituted artifacts.

### `agent_tools.run_artifacts.find_run_manifest`

- **File/signature:** `agent_tools/run_artifacts.py`; `find_run_manifest(run: dict[str, Any]) -> Path | None`.
- **Purpose/contract:** inspect only the frozen `runtime_dir/run_manifest.json`; return `None` for confirmed absence and fail on aliases, hard links, invalid JSON, or non-mapping payloads.
- **Inputs/outputs:** managed run row; validated manifest path or `None`.
- **Side effects:** exact-path filesystem metadata and JSON read.
- **Callers/callees:** selection, runtime evidence, adaptive digest.
- **Reuse guidance:** use instead of recursive runtime discovery.
- **Duplication risk:** fallback searches can adopt another run's evidence.

### `agent_tools.run_evidence.status_row`

- **File/signature:** `agent_tools/run_evidence.py`; `status_row(run_dir: Path, row: dict[str, Any], previous: dict[str, Any] | None = None, *, health: bool = False) -> dict[str, Any]`.
- **Purpose/contract:** produce one observation from PID/process/log/progress/runtime/checkpoint evidence while preserving prior durable state on ambiguous observation failures.
- **Inputs/outputs:** plan directory, managed row, prior row, optional health mode; observation mapping.
- **Side effects:** local or remote process/log/progress/GPU probes.
- **Callers/callees:** hparam and experiment monitors; uses `read_pid`, runtime artifact checks, and health classification.
- **Reuse guidance:** use as the shared observation reducer, then commit through `merge_run_manifest`.
- **Duplication risk:** independent monitors disagree on terminal or unhealthy states.

### `agent_tools.run_evidence.read_pid`

- **File/signature:** `agent_tools/run_evidence.py`; `read_pid(path: Any, row: dict[str, Any] | None = None, *, expected_script: str | Path | None = None) -> int | None`.
- **Purpose/contract:** safely read a local/remote PID file and optionally prove that the live process command owns the frozen script before signaling it.
- **Inputs/outputs:** PID path, optional row and expected script; PID or `None`.
- **Side effects:** local metadata/process inspection or SSH inspection.
- **Callers/callees:** monitoring and `hparam-stop`.
- **Reuse guidance:** stopping must provide the frozen script identity.
- **Duplication risk:** trusting an unverified reused PID can signal an unrelated process.

## Hparam Operations

### `agent_tools.hparam_runtime.launch_hparam_runs`

- **File/signature:** `agent_tools/hparam_runtime.py`; `launch_hparam_runs(plan_dir: str | Path, *, dry_run: bool = True) -> Path`.
- **Purpose/contract:** validate the frozen plan, serialize launch against the canonical manifest, assign declared GPU groups, and either prepare launch evidence or start the planned scripts.
- **Inputs/outputs:** plan directory and dry-run switch; launch manifest path.
- **Side effects:** writes launch/status evidence and events; execute mode starts processes and commits observations.
- **Callers/callees:** hparam facade/CLI and adaptive execution; calls `read_hparam_plan`, process start, and canonical state owners.
- **Reuse guidance:** use the public export rather than invoking scripts directly.
- **Duplication risk:** unmanaged launch skips GPU accounting, atomic launch reconciliation, and durable identity.

### `agent_tools.hparam_runtime.monitor_hparam_runs`

- **File/signature:** `agent_tools/hparam_runtime.py`; `monitor_hparam_runs(run_dir: str | Path, *, once: bool = True, health: bool = False) -> Path`.
- **Purpose/contract:** observe exactly the runs frozen in one hparam plan, write plan-local status, and commit canonical workspace state.
- **Inputs/outputs:** plan directory and observation switches; status table path.
- **Side effects:** process/evidence probes and status/report/manifest writes.
- **Callers/callees:** CLI, adaptive digest, and loop; uses `read_hparam_plan`, `status_row`, and `merge_run_manifest`.
- **Reuse guidance:** use instead of tailing logs or W&B alone.
- **Duplication risk:** alternate monitors may infer completion from partial artifacts.

### `agent_tools.hparam_runtime.stop_hparam_run`

- **File/signature:** `agent_tools/hparam_runtime.py`; `stop_hparam_run(run_dir: str | Path, run_id: str, *, reason: str) -> Path`.
- **Purpose/contract:** require a recorded reason, identify one current-plan managed run, prove process/script ownership, signal it, and commit stopped state/evidence.
- **Inputs/outputs:** plan directory, stable run id, non-empty reason; status table path.
- **Side effects:** may signal a process; writes status, canonical state, report, and event.
- **Callers/callees:** hparam facade/CLI and adaptive replacement logic.
- **Reuse guidance:** all automated stopping must use this owner.
- **Duplication risk:** direct `kill` loses reason, ownership proof, and durable state.

### `agent_tools.hparam_selection.select_hparam_candidates`

- **File/signature:** `agent_tools/hparam_selection.py`; `select_hparam_candidates(run_dir: str | Path, metric: str | None = None, mode: str | None = None) -> Path`.
- **Purpose/contract:** rank current step runs only by the selection metric/mode frozen in their recipes, preserving valid rows for other steps and validating checkpoint ownership.
- **Inputs/outputs:** plan directory and optional confirming metric/mode; workspace `reports/ranking.csv`.
- **Side effects:** reads registered plans/runtime evidence, writes ranking, commits score/rank/checkpoint observations, and appends an event.
- **Callers/callees:** hparam facade/CLI; uses `read_hparam_plan`, step registrations, runtime manifests, and canonical workspace state.
- **Reuse guidance:** use before external evaluation or logits export.
- **Duplication risk:** arbitrary metric overrides can turn test/external evidence into selection.

### `agent_tools.hparam_selection.scan_hparam_checkpoints`

- **File/signature:** `agent_tools/hparam_selection.py`; `scan_hparam_checkpoints(run_dir: str | Path, metric: str, mode: str, *, top_k: int | None = None) -> Path`.
- **Purpose/contract:** rank checkpoint-level history rows for the exact managed runs while preserving many-per-run identity.
- **Inputs/outputs:** plan directory, metric/mode, optional top-k; plan-local checkpoint ranking CSV.
- **Side effects:** reads frozen/runtime evidence and writes one CSV.
- **Callers/callees:** hparam facade/CLI.
- **Reuse guidance:** use when epoch-level checkpoint review is needed; candidate selection remains the final run-level owner.
- **Duplication risk:** filename-only checkpoint scans can associate scores with the wrong epoch/run.

### `agent_tools.hparam_postprocess.generate_external_eval`

- **File/signature:** `agent_tools/hparam_postprocess.py`; `generate_external_eval(run_dir: str | Path, selected_csv: str | Path, *, unlock_final_test: bool, kaldi_data_root: str | None = None, kaldi_manifest: str | None = None, finetune_data_index: str | None = None, eval_split: str = "test", top_k: int = 1, all_candidates: bool = False) -> Path`.
- **Purpose/contract:** require explicit final-test unlock, validate selected managed candidates, freeze external-eval configs, and write replayable inference commands.
- **Inputs/outputs:** plan/selection plus explicit external data overrides; script path.
- **Side effects:** writes configs, manifest, and executable script.
- **Callers/callees:** hparam facade/CLI; uses frozen owner plans, `infer_runtime_cli_args`, variant module routing, and managed-output checks.
- **Reuse guidance:** use for final/external evaluation generation.
- **Duplication risk:** hand-built infer commands can bypass selection ownership or the external-test lock.

### `agent_tools.hparam_postprocess.export_hparam_logits`

- **File/signature:** `agent_tools/hparam_postprocess.py`; `export_hparam_logits(run_dir: str | Path, selected_csv: str | Path, *, unlock_final_test: bool, val_split: str = "val", test_split: str = "test", skip_test: bool = False, label_name: str | None = None, val_kaldi_data_root: str | None = None, val_kaldi_manifest: str | None = None, val_finetune_data_index: str | None = None, test_kaldi_data_root: str | None = None, test_kaldi_manifest: str | None = None, test_finetune_data_index: str | None = None, batch_size: int = 12, num_workers: int = 8, devices: list[int] | None = None, accelerator: str = "gpu", device: str = "cuda", precision: str = "bf16-mixed", seed: int = 4523, top_k: int = 1, all_candidates: bool = False, execute: bool = False) -> Path`.
- **Purpose/contract:** export validation and optionally unlocked test logits for selected candidates, either as a replay script or by executing exact generated commands.
- **Inputs/outputs:** selection, split/data/runtime settings, unlock/execute switches; export manifest path.
- **Side effects:** writes configs/manifests/scripts/outputs and may run inference.
- **Callers/callees:** hparam facade/CLI; validates owner plans and generated output topology.
- **Reuse guidance:** threshold and ensemble workflows should consume these managed exports.
- **Duplication risk:** direct exports can mix candidate configs, labels, or locked test data.

## Experiment and Adaptive Operations

### `agent_tools.experiments.init_experiment`

- **File/signature:** `agent_tools/experiments.py`; `init_experiment(run_dir: str | Path, spec_path: str | Path, *, remote: str | None = None) -> Path`.
- **Purpose/contract:** initialize or idempotently confirm one local/remote experiment root from an explicit spec and reject non-empty unmanaged or conflicting roots.
- **Inputs/outputs:** target root, spec, optional remote; experiment manifest path.
- **Side effects:** creates managed directories/YAML/TSV/README/events locally or over SSH.
- **Callers/callees:** CLI; uses workspace metadata, strict YAML, managed I/O, and run-manifest initialization.
- **Reuse guidance:** use for explicit lifecycle initialization outside recipe planning.
- **Duplication risk:** manual setup omits ownership and strict empty-root checks.

### `agent_tools.experiments.register_experiment_step`

- **File/signature:** `agent_tools/experiments.py`; `register_experiment_step(run_dir: str | Path, spec_path: str | Path, *, remote: str | None = None) -> Path`.
- **Purpose/contract:** attach a complete step spec to an existing managed experiment using the canonical step envelope/merge rules.
- **Inputs/outputs:** workspace, spec, optional remote; step manifest path.
- **Side effects:** writes step YAML and first-registration event.
- **Callers/callees:** CLI; uses `read_step_manifest` and `merge_step_manifest`.
- **Reuse guidance:** do not write `steps/*/step.yaml` directly.
- **Duplication risk:** incompatible envelopes break plan registration and selection traversal.

### `agent_tools.experiments.monitor_experiment`

- **File/signature:** `agent_tools/experiments.py`; `monitor_experiment(run_dir: str | Path, *, remote: str | None = None) -> dict[str, Any]`.
- **Purpose/contract:** observe all canonical experiment runs, commit observations through the state owner, and write a report.
- **Inputs/outputs:** workspace and optional remote; committed runs and report path.
- **Side effects:** local/remote probes and managed state/report writes.
- **Callers/callees:** CLI; uses `experiment_tracking.monitor_run_row` and `merge_run_manifest`.
- **Reuse guidance:** use for experiment-wide monitoring; hparam monitor is plan-scoped.
- **Duplication risk:** reporting from uncommitted observations can disagree with canonical state.

### `agent_tools.experiments.rank_experiment_candidates`

- **File/signature:** `agent_tools/experiments.py`; `rank_experiment_candidates(run_dir: str | Path, *, metric: str, mode: str, remote: str | None = None) -> Path`.
- **Purpose/contract:** validate managed metric/checkpoint evidence and write experiment-wide candidate ranking without changing frozen selection policy.
- **Inputs/outputs:** workspace, metric, mode, optional remote; ranking CSV.
- **Side effects:** reads managed tables; writes CSV and Markdown report.
- **Callers/callees:** CLI; uses tracking candidate/rank functions and workspace ownership validation.
- **Reuse guidance:** use for cross-step experiment reporting, not as a substitute for hparam recipe selection.
- **Duplication risk:** ranking unmanaged evidence can adopt another experiment's rows.

### `agent_tools.experiments.finalize_experiment`

- **File/signature:** `agent_tools/experiments.py`; `finalize_experiment(run_dir: str | Path, report_path: str | Path, *, remote: str | None = None) -> Path`.
- **Purpose/contract:** require at least one managed run, all runs terminal, and a non-empty final report before marking the experiment completed.
- **Inputs/outputs:** workspace, report, optional remote; final report path.
- **Side effects:** copies report, updates experiment YAML, appends event.
- **Callers/callees:** CLI.
- **Reuse guidance:** use as the sole completion transition.
- **Duplication risk:** status-only completion can finalize active or reportless experiments.

### `agent_tools.adaptive_hparam.init_adaptive_workflow`

- **File/signature:** `agent_tools/adaptive_hparam.py`; `init_adaptive_workflow(recipe_path: str | Path, output_dir: str | Path) -> Path`.
- **Purpose/contract:** validate adaptive settings/budget, preflight round zero before mutation, preserve the hparam local overlay when writing round recipes, build its managed plan, and initialize workflow/registry/docs.
- **Inputs/outputs:** recipe and workflow root; canonical root.
- **Side effects:** after successful preflight, writes adaptive workflow and round artifacts plus workspace state/events.
- **Callers/callees:** CLI; composes recipe loading, `preflight_plan`, `build_plan`, and workspace owners.
- **Reuse guidance:** use rather than creating round directories manually.
- **Duplication risk:** mutation before round-zero preflight leaves invalid partial workflows.

### `agent_tools.adaptive_hparam.adaptive_step`

- **File/signature:** `agent_tools/adaptive_hparam.py`; `adaptive_step(workflow_dir: str | Path, *, execute: bool = False) -> Path`.
- **Purpose/contract:** preflight the mutable source before digest mutation, digest current evidence, generate and preflight the suggested overlay, then in execute mode respect total budget and replacement ordering before planning/launching it.
- **Inputs/outputs:** workflow root and execute switch; latest/next round path.
- **Side effects:** writes digests/suggestions/events; execute mode may stop/supersede runs, create a plan, and launch.
- **Callers/callees:** CLI and `adaptive_loop`; composes existing hparam, plan, evidence, and workspace owners.
- **Reuse guidance:** keep ordering and budget logic here, not in launch or selection modules.
- **Duplication risk:** stopping current runs before a full replacement round is validated can destroy available capacity.

### `agent_tools.adaptive_hparam.adaptive_loop`

- **File/signature:** `agent_tools/adaptive_hparam.py`; `adaptive_loop(workflow_dir: str | Path, *, execute: bool = False) -> Path`.
- **Purpose/contract:** repeat `adaptive_step` until budget or non-progress termination; dry-run performs at most one planning step.
- **Inputs/outputs:** workflow root and execute switch; last path.
- **Side effects:** inherited from `adaptive_step`.
- **Callers/callees:** CLI.
- **Reuse guidance:** use for persistent adaptive execution after initialization.
- **Duplication risk:** external loops may ignore unresolved launch attempts or workflow budget.

## Progress

### `agent_tools.progress.write_progress`

- **File/signature:** `agent_tools/progress.py`; `write_progress(run_dir: str | Path, *, status: str, task: str, processed: int = 0, total: int | None = None, success: int = 0, failed: int = 0, start_time: float | None = None, current_item: str | None = None, message: str | None = None) -> Path`.
- **Purpose/contract:** write the stable machine-readable `status/progress.json` contract for long-running tools.
- **Inputs/outputs:** run identity/status/counters/timing/message; progress path.
- **Side effects:** creates the status directory and writes JSON.
- **Callers/callees:** preprocessing and utility emitters; read by `read_progress` and health checks.
- **Reuse guidance:** preserve the location and field meanings.
- **Duplication risk:** custom progress files are invisible to agent monitoring.

### `agent_tools.progress.read_progress`

- **File/signature:** `agent_tools/progress.py`; `read_progress(run_dir: str | Path, *, remote: str | None = None, timeout_seconds: int = DEFAULT_SSH_TIMEOUT_SECONDS) -> dict[str, Any]`.
- **Purpose/contract:** read and validate local progress or retrieve it through a bounded SSH `cat`, returning structured invalid/missing evidence rather than unbounded polling.
- **Inputs/outputs:** run directory, optional host/timeout; progress mapping.
- **Side effects:** local read or SSH subprocess.
- **Callers/callees:** CLI and run-health evidence.
- **Reuse guidance:** use instead of log-tail progress inference when an emitter supports the contract.
- **Duplication risk:** bespoke remote polling can hang or misclassify stale output.
