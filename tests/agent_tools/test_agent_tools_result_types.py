from pathlib import Path
import subprocess
import sys
import textwrap


def test_result_types_reach_callers(tmp_path: Path):
    probe = tmp_path / "result_type_probe.py"
    probe.write_text(
        textwrap.dedent("""\
            from pathlib import Path
            from agent_tools import (
                adaptive_hparam, checkpoint_test_results, experiment_tracking, experiments,
                experiment_io, experiment_sources, experiment_workspace, hparam_runtime, hparam_selection,
                managed_scheduler, models, experiment_pipeline, experiment_pipeline_results,
                experiment_pipeline_cohort_selection,
                plan_contract, plan_hparam, run_artifacts, run_evidence, slurm,
            )

            from agent_tools.adapters.base import TaskAdapter
            from agent_tools.adapters.hparam_tune import HPARAM_TUNE_ADAPTER

            from typing import Any, Literal

            direct_launch: managed_scheduler.DirectLaunchIdentity = {
                "target": "local", "host": None, "workdir": Path("/runtime"),
                "gpus": "", "pid_path": "/run/pid", "log_path": "/run/stdout.log", "command": "",
            }
            direct_log: str = direct_launch["log_path"]
            direct_launch["log_path"] = Path("/log")  # type: ignore[typeddict-item]
            direct_launch["pid"] = "123"  # type: ignore[typeddict-item]
            direct_launch["planned_runtime_commit"] = None  # type: ignore[typeddict-item]
            slurm_launch = managed_scheduler._slurm_execution_identity({}, {})
            submit_command: str = slurm_launch["command"]
            slurm_launch["command"] = None  # type: ignore[typeddict-item]
            slurm_launch["execution_snapshot_sha256"] = 1  # type: ignore[typeddict-item]
            slurm_launch["log_path"] = None
            verification_options: managed_scheduler.LaunchVerificationOptions = {}
            verification_options["checkpoint_path"] = None
            verification_options["checkpoint_sha256"] = None
            verification_options["checkpoint_path"] = "path"  # type: ignore[typeddict-item]
            verification_options["run_id"] = None  # type: ignore[typeddict-item]
            managed_scheduler.build_launch_command({}, Path("/script"), "/log", "/pid", [], **verification_options)
            frozen_artifact: managed_scheduler.FrozenLaunchArtifact = {"path": "/config", "sha256": "a" * 64}
            frozen_artifact["sha256"] = None  # type: ignore[typeddict-item]

            terminal_sidecar: slurm.TerminalSidecar = {
                "schema_version": 1, "scheduler_job_id": "42", "scheduler_cluster": "cluster",
                "scheduler_submit_token": "token", "node": "node", "started_at": "time",
                "ended_at": "time", "exit_code": 0, "runtime_commit": "",
            }
            terminal_sidecar["exit_code"] = "0"  # type: ignore[typeddict-item]
            terminal_sidecar["scheduler_job_id"] = 42  # type: ignore[typeddict-item]
            allocation_sidecar: slurm.AllocationSidecar = {
                "schema_version": 1, "scheduler_job_id": "42", "scheduler_cluster": "cluster",
                "scheduler_submit_token": "token", "node": "node", "started_at": "time",
                "execution_snapshot": {"runtime_commit": "a" * 40, "module": "sleep2vec.infer"},
            }
            allocation_sidecar["execution_snapshot"]["module"] = None  # type: ignore[typeddict-item]
            allocation_sidecar["schema_version"] = 2  # type: ignore[typeddict-item]
            for generated_sidecar in (allocation_sidecar, terminal_sidecar):
                slurm.sidecar_identity(generated_sidecar, "token")
                managed_scheduler._slurm_sidecar_runtime_commit(generated_sidecar)
                slurm._atomic_create_json(Path("/sidecar"), generated_sidecar)
            slurm.terminal_exit_code(terminal_sidecar)

            source_states = experiment_pipeline._inspect_sources(Path("/workspace"), {}, refresh=False)
            source_complete: bool = source_states[0]["complete"]
            source_states[0]["complete"] = "true"  # type: ignore[typeddict-item]
            frozen_candidates = experiment_pipeline._select_checkpoint_sources(Path("/workspace"), {})
            frozen_score: float = frozen_candidates[0]["score"]
            frozen_candidates[0]["score"] = "1"  # type: ignore[typeddict-item]
            checkpoint_key_count: int = frozen_candidates[0]["state_dict_key_count"]

            logical_jobs = experiment_pipeline_results.logical_job_states({}, [])
            attempt_count: int = logical_jobs[0]["attempt_count"]
            logical_jobs[0]["status"] = "ready"  # type: ignore[typeddict-item]
            execution_result = experiment_pipeline._run_attempts(
                Path("/workspace"), Path("/pipeline"), {}, {}, [], poll_seconds=0,
            )
            execution_result["jobs"][0]["attempt_count"] = "1"  # type: ignore[typeddict-item]
            if "missing_pid_blocker" in execution_result:
                execution_result["missing_pid_blocker"]["status"] = "running"  # type: ignore[typeddict-item]
            for pipeline_result in (
                experiment_pipeline.run_experiment_pipeline(Path("/workspace"), Path("/spec")),
                experiments.run_experiment_pipeline(Path("/workspace"), Path("/spec")),
            ):
                pipeline_result["status"] = "unknown"  # type: ignore[arg-type]

            selection_evidence = experiment_pipeline_results.selection_evidence(Path("/phase"), {}, {})
            evidence_hash: str = selection_evidence[0]["result_manifest_sha256"]
            selection_evidence[0]["result_manifest_sha256"] = None  # type: ignore[typeddict-item]
            _, cohort_decision = experiment_pipeline_cohort_selection.rank_candidates({}, {}, selection_evidence)
            cohort_winner = cohort_decision["winner"]
            if cohort_winner is not None:
                winner_rank: int = cohort_winner["source_rank"]
                cohort_winner["source_rank"] = "1"  # type: ignore[typeddict-item]
                gate_value: int | float = cohort_winner["selection_evidence"][0]["value"]
                cohort_winner["selection_evidence"][0]["value"] = "1"  # type: ignore[typeddict-item]
            experiment_pipeline._write_no_winner_report(Path("/pipeline"), cohort_decision)
            _, typed_decision = experiment_pipeline._validate_cohort_decision(
                Path("/pipeline"), {}, {}, selection_evidence,
            )
            typed_decision["winner"] = "none"  # type: ignore[typeddict-item]
            _, pipeline_metrics = experiment_pipeline_results.build_result_rows({}, {}, {})
            scalar_value: int | float | str = pipeline_metrics[0]["value"]
            pipeline_metrics[0]["value"] = None  # type: ignore[typeddict-item]
            experiment_pipeline_results.write_rows_atomic(Path("/metrics.csv"), pipeline_metrics)

            def snapshot_result(should_write: bool) -> tuple[managed_scheduler.ExecutionSnapshot, bool]:
                return {}, should_write

            managed_scheduler.SchedulerHooks(validated_snapshot=lambda *args: snapshot_result(True))
            managed_scheduler.SchedulerHooks(validated_snapshot=lambda *args: (None, False))
            invalid_snapshot: managed_scheduler.ExecutionSnapshotResult = (None, True)  # type: ignore[assignment]

            def missing_snapshot_write() -> tuple[None, Literal[True]]:
                return None, True

            managed_scheduler.SchedulerHooks(validated_snapshot=missing_snapshot_write)  # type: ignore[arg-type]

            for execution_snapshot in (
                managed_scheduler.inspect_execution_target({}, []),
                managed_scheduler.validated_execution_snapshot(Path("/plan"), {}, [], {})[0],
                hparam_runtime._inspect_execution_target({}, []),
                hparam_runtime._validated_execution_snapshot(Path("/plan"), {}, [], {})[0],
                plan_hparam._inspect_hparam_execution_target({}, []),
            ):
                module_name: str = execution_snapshot["module"]
                snapshot_commit: str = execution_snapshot["runtime_commit"]
                options: list[str] = execution_snapshot["required_options"]
                argv_digest: str = execution_snapshot["validated_argv_sha256"]
                execution_snapshot["module"] = 1  # type: ignore[typeddict-item]
                execution_snapshot["required_options"] = [1]  # type: ignore[list-item]
                execution_snapshot["validated_argv_sha256"] = b"hash"  # type: ignore[typeddict-item]
                execution_snapshot["module_name"]  # type: ignore[typeddict-item]
                managed_scheduler.write_execution_snapshot_file(Path("/snapshot"), execution_snapshot)
                managed_scheduler.build_launch_command(
                    {}, Path("script"), "log", "pid", [], execution_snapshot=execution_snapshot,
                )
                hparam_runtime._launch_command(
                    {}, Path("script"), "log", "pid", [], execution_snapshot=execution_snapshot,
                )

            from agent_tools.index_csv import index_summary
            from agent_tools.domain.presets import preset_summary
            from agent_tools import plan_context, markdown
            from agent_tools.decision_models import DecisionReport, DecisionStatus

            index = index_summary([])
            row_count: int = index["rows"]
            label_count: int = index["label_presence"]["age"]["non_null"]
            missing_paths: list[str] = index["sample_path_check"]["missing_examples"]
            missing_paths = index["sample_path_check"]["checked"]  # type: ignore[assignment]
            index["mask_columns"]["ppg"]["true_count"] = "1"  # type: ignore[typeddict-item]
            index["numeric_shift_metrics"]["age"]["test_mean"] = "1"  # type: ignore[typeddict-item]
            index["survival_covariates"]["age"]["missing_rows"] = "1"  # type: ignore[typeddict-item]
            if index["survival_key"] is not None:
                key_count: int | None = index["survival_key"]["sidecar_key_count"]
                index["survival_key"]["missing_from_sidecars_examples"] = [1]  # type: ignore[list-item]
            preset = preset_summary(Path("samples.pkl"))
            sample_count: int = preset["samples"]
            preset["source_counts"] = {"source": "1"}  # type: ignore[dict-item]
            preset["sidecar_manifest"] = ["unvalidated", 1]
            for summary in (
                plan_context.context_index_summary({}, None),
                plan_context.context_preset_summary({}, None),
            ):
                if summary is not None:
                    errors: list[str] = summary["blocking_issues"]
                    errors = summary["blocking_issues"][0]  # type: ignore[assignment]
            questions = markdown.questions_payload(DecisionReport(status=DecisionStatus.FAIL))
            question_text: str | None = questions[0]["question"]
            questions[0]["field"] = 1  # type: ignore[typeddict-item]

            def consume_context(context: plan_context.ContextPayload) -> str:
                allowed: bool = context["can_generate_commands"]
                commands: list[str] = context["recommended_commands"]
                context["questions"][0]["message"] = 1  # type: ignore[typeddict-item]
                context["recommended_commands"] = [1]  # type: ignore[list-item]
                return plan_context.context_markdown(context)

            def diagnostic_contracts() -> None:
                from agent_tools import configs
                from agent_tools.domain.finetune_summary import finetune_summary_body
                from agent_tools.domain.sex_age_summary import sex_age_baseline_config_summary
                from agent_tools.domain.sidecar_summaries import survival_summary, multilabel_summary
                from agent_tools.adapters.sleep2stat import sleep2stat_config_summary
                from agent_tools.adapters.config_providers import CONFIG_SUMMARY_PROVIDERS

                for diagnostic in (
                    configs.config_summary(Path("config.yaml")),
                    TaskAdapter().config_summary(Path("config.yaml")),
                    CONFIG_SUMMARY_PROVIDERS[0].summarize(Path("config.yaml")),
                ):
                    warnings: list[str] = diagnostic["warnings"]
                    config_path: str = diagnostic["config_path"]
                    diagnostic["blocking_issues"] = [1]  # type: ignore[list-item]
                    diagnostic["_source_config_bytes"] = "bytes"  # type: ignore[arg-type]
                from types import MappingProxyType
                from agent_tools import decision_paths
                from agent_tools.domain.finetune_hparam_profile import compile_finetune_balanced_profile
                readonly: MappingProxyType[str, Any] = MappingProxyType({"data": {}, "finetune": {}})
                decision_paths._config_data(readonly)  # type: ignore[arg-type]
                decision_paths._config_finetune(readonly)  # type: ignore[arg-type]
                compile_finetune_balanced_profile({}, readonly)  # type: ignore[arg-type]
                fine = finetune_summary_body(Path("config.yaml"))
                fine["model"]["channels"][0]["unknown"] = 1  # type: ignore[typeddict-unknown-key]
                fine["model"]["layer_mix_present"] = "yes"  # type: ignore[typeddict-item]
                fine["model"]["head_details"]["kwargs"] = []  # type: ignore[typeddict-item]
                fine["finetune"]["tuning_present"] = 1  # type: ignore[typeddict-item]
                fine["data"]["train_dataset_names"] = "train"  # type: ignore[typeddict-item]
                fine["finetune"]["task"]["output_dim"] = {"invalid": "preserved"}
                sex = sex_age_baseline_config_summary(Path("config.yaml"))
                sex["model"]["features"] = [1]  # type: ignore[list-item]
                stats = sleep2stat_config_summary(Path("config.yaml"))
                stats["agent_risk_issues"] = False  # type: ignore[typeddict-item]
                stats["sleep2stat"]["supported_analyzer_types"] = [1]  # type: ignore[list-item]
                stats["sleep2stat"]["analyzers"][0]["enabled"] = "yes"  # type: ignore[typeddict-item]
                stats["sleep2stat"]["reducers"][0]["source"] = 1  # type: ignore[typeddict-item]
                for sidecar in (survival_summary({}, {}), multilabel_summary({}, {})):
                    if sidecar is not None:
                        valid: bool = sidecar["valid"]
                        count: int | None = sidecar["sidecar_key_count"]
                        sidecar["valid"] = "yes"  # type: ignore[arg-type]
                        sidecar["issues"] = [1]  # type: ignore[list-item]
                        sidecar["output_dim"] = ["unvalidated"]

            def discovery_context(context: plan_context.ContextPayload) -> None:
                available: bool = context["repo"]["git"]["available"]
                context["repo"]["python"]["version"] = 1  # type: ignore[typeddict-item]
                context["skill"]["unexpected"] = "owner"  # type: ignore[typeddict-unknown-key]
                if context["config_summary"] is not None:
                    context["config_summary"]["warnings"] = [1]  # type: ignore[list-item]

            minimal_snapshot: managed_scheduler.ExecutionSnapshot = {
                "module": "runtime_cli", "module_origin": "/runtime_cli.py",
            }
            managed_scheduler.build_launch_command(
                {}, Path("script"), "log", "pid", [], execution_snapshot=minimal_snapshot,
            )
            managed_scheduler.build_launch_command(
                {}, Path("script"), "log", "pid", [],
                execution_snapshot={"module": 1},  # type: ignore[arg-type]
            )
            hparam_runtime._launch_command(
                {}, Path("script"), "log", "pid", [],
                execution_snapshot={"module": 1},  # type: ignore[arg-type]
            )

            wandb_payload = experiment_sources.wandb_run_payload(None, entity="entity", project="project")
            history_filename: str = wandb_payload["history_filename"]
            wandb_payload["history_file"]  # type: ignore[typeddict-item]
            wandb_payload["summary_line"] = []  # type: ignore[typeddict-item]
            wandb_row = wandb_payload["run_row"]
            wandb_row["wandb_run_id"] = 1  # type: ignore[typeddict-item]
            wandb_row["status"] = None  # type: ignore[typeddict-item]
            if "experiment_id" in wandb_row:
                wandb_experiment: str = wandb_row["experiment_id"]
            metric_row = wandb_payload["metric_rows"][0]
            metric_row["metric_scope"] = 1  # type: ignore[typeddict-item]
            metric_row["epoch"] = 1  # type: ignore[typeddict-item]
            metric_row["metric_name"]  # type: ignore[typeddict-item]
            metric_row["value"] = "0.750"
            history_metrics = experiment_sources._history_metric_rows("id", "version", wandb_row, [])
            history_metrics[0]["updated_at"] = None  # type: ignore[typeddict-item]
            experiment_tracking.wandb_run_observations([], [wandb_row])
            experiment_tracking.managed_metric_rows([], [metric_row])
            for checkpoint_observations in (
                experiment_sources._local_checkpoint_rows([]),
                experiment_sources._remote_checkpoint_rows([], None),
                experiment_sources._parse_remote_checkpoint_rows("", [], remote="host"),
                experiment_tracking.checkpoint_rows(Path("/workspace")),
            ):
                checkpoint_observation = checkpoint_observations[0]
                observed_checkpoint_path: str = checkpoint_observation["checkpoint_path"]
                checkpoint_observation["mtime"] = 1.0  # type: ignore[typeddict-item]
                checkpoint_observation["is_last"] = True  # type: ignore[typeddict-item]
                checkpoint_observation["checkpoint"]  # type: ignore[typeddict-item]
                experiment_sources.validate_checkpoint_evidence_rows([], checkpoint_observations)
                experiment_workspace.validate_managed_run_rows(
                    checkpoint_observations, source="checkpoint scan", cardinality="many_per_run",
                )
                checkpoint_metric = experiment_tracking.best_metric_for_checkpoint(checkpoint_observation, [])
                checkpoint_metric["value"] = 1.0  # type: ignore[typeddict-item]
                checkpoint_observation.update(checkpoint_metric)

            direct_health = run_evidence.health_fields(Path("/run"), {}, {}, None, None, "running", None)
            health_count: int | Literal[""] = direct_health["checkpoint_count"]
            progress_age: int | None = direct_health["progress_age_seconds"]
            direct_health["checkpoint_count"] = "0"  # type: ignore[typeddict-item]
            direct_health["io_read_bytes"] = 1.5  # type: ignore[typeddict-item]
            direct_health["log_age_seconds"] = None  # type: ignore[typeddict-item]
            direct_health["progress_age_seconds"] = ""  # type: ignore[typeddict-item]
            direct_health["gpu_summary"] = None  # type: ignore[typeddict-item]
            direct_health["checkpoint_counts"]  # type: ignore[typeddict-item]
            direct_health["progress_processed"] = "003"
            slurm_health: managed_scheduler.SlurmHealthFields = {
                "health_status": "scheduler_queued", "scheduler_health_error": "",
                "scheduler_queue_age_seconds": 0, "scheduler_allocation_age_seconds": "", "log_age_seconds": "",
            }
            slurm_health["scheduler_queue_age_seconds"] = "0"  # type: ignore[typeddict-item]
            slurm_health["scheduler_health_error"] = None  # type: ignore[typeddict-item]
            slurm_health["queue_age_seconds"]  # type: ignore[typeddict-item]
            experiment_tracking.monitor_report((direct_health, slurm_health))
            monitor_result = experiments.monitor_experiment("/workspace")
            monitor_report_path: str = monitor_result["report"]
            monitor_result["runs"] = ["run-000"]  # type: ignore[list-item]
            monitor_result["report"] = Path("/report")  # type: ignore[typeddict-item]
            monitor_result["decision"]  # type: ignore[typeddict-item]

            planned: managed_scheduler.PlannedArgv = {"run_id": "run-000", "args": ["--value", "ok"]}
            planned["args"] = [1]  # type: ignore[list-item]
            planned["run_id"] = 1  # type: ignore[typeddict-item]

            from agent_tools import adaptive_proposals
            from types import MappingProxyType

            proposal_document = adaptive_proposals.build_proposal_input({}, expected_proposal_path="/proposal.json")
            proposal_document["request_id"] = None  # type: ignore[typeddict-item]
            proposal_document["input"]["remaining_budget"]["runs"] = "1"  # type: ignore[typeddict-item]
            proposal_document["input"]["objective"]["mode"] = 1  # type: ignore[typeddict-item]
            proposal_document["input"]["digest_rows"][0]["custom"] = None
            proposal_document["input"]["execution_identity"]["host"] = None
            checked_document = adaptive_proposals.validate_proposal_input(proposal_document)
            checked_document["expected_proposal_path"] = Path("/proposal")  # type: ignore[typeddict-item]
            envelope = adaptive_proposals.validate_parameter_envelopes({"runtime.lr": [0.1]})["runtime.lr"]
            envelope["kind"] = "unknown"  # type: ignore[arg-type]
            validated_proposal = adaptive_proposals.validate_proposal({}, proposal_document)
            validated_proposal["target_round"] = "1"  # type: ignore[typeddict-item]
            validated_proposal["evidence_run_ids"] = [1]  # type: ignore[list-item]
            validated_proposal["proposer"] = None
            validated_proposal["proposer"] = {"agent": "codex", "model": 1}  # type: ignore[typeddict-item]

            def adaptive_events(
                initialized: experiment_workspace.AdaptiveInitEvent,
                binding: experiment_workspace.AdaptiveProposalRequestBinding,
                requested: experiment_workspace.AdaptiveProposalRequestedEvent,
                accepted: experiment_workspace.AdaptiveProposalAcceptedEvent,
                workflow: adaptive_hparam.InitialAdaptiveWorkflow,
                accepted_payload: adaptive_hparam.AcceptedProposalPayload,
            ) -> None:
                snapshot = adaptive_hparam._agent_proposal_input_payload(Path("/workflow"), {}, {}, [])
                snapshot["source_config_sha256"] = None  # type: ignore[typeddict-item]
                generated_binding = adaptive_hparam._proposal_request_event_fields(
                    proposal_document, Path("/input"), "a" * 64, Path("/proposal"),
                )
                generated_binding["target_round"] = "1"  # type: ignore[typeddict-item]
                accepted_payload["proposal_sha256"] = None  # type: ignore[typeddict-item]
                accepted_payload["schema_version"] = 2  # type: ignore[typeddict-item]
                initialized["round"] = "0"  # type: ignore[typeddict-item]
                binding["input_sha256"] = None  # type: ignore[typeddict-item]
                requested["digest"] = Path("/digest")  # type: ignore[typeddict-item]
                accepted["suggestion_sha256"] = None  # type: ignore[typeddict-item]
                created = adaptive_hparam._plan_event(Path("/round"), {"recipe": {"step": {"id": None}}})
                created["step_id"] = None
                created["run_count"] = "1"  # type: ignore[typeddict-item]
                for event_payload in (initialized, requested, accepted, created):
                    experiment_workspace.append_event(Path("/workspace"), "event", event_payload)
                    experiment_workspace.event_matches({}, "event", event_payload)
                raw_event: dict[str, Any] = {"round": None, "request_id": None, "custom": [None]}
                experiment_workspace.append_event(Path("/workspace"), "event", raw_event)
                readonly_event = MappingProxyType(raw_event)
                experiment_workspace.event_matches({}, "event", readonly_event)
                experiment_workspace.append_event(Path("/workspace"), "event", readonly_event)  # type: ignore[arg-type]
                typed_workflow = adaptive_hparam._validate_workflow_payload(Path("/workflow"), workflow)
                typed_workflow["external_optimized"] = False  # type: ignore[typeddict-item]
                typed_workflow["recipe_path"] = None  # type: ignore[typeddict-item]
                raw_workflow: dict[str, Any] = {"custom": None, "objective_metric": None}
                retained_workflow: dict[str, Any] = adaptive_hparam._validate_workflow_payload(
                    Path("/workflow"), raw_workflow,
                )
                retained_workflow["unknown"] = {"nested": None}

            executed_step: Path = adaptive_hparam.adaptive_step("/workflow", execute=True)
            checked_proposal: Path = adaptive_hparam.adaptive_step("/workflow", proposal_path="/proposal.json")
            applied_proposal: Path = adaptive_hparam.adaptive_step(
                "/workflow", proposal_path=Path("/proposal.json"), execute=True,
            )
            pending_step: Path | None = adaptive_hparam.adaptive_step("/workflow")
            required_step: Path = adaptive_hparam.adaptive_step("/workflow")  # type: ignore[assignment]

            def check_step_options(execute: bool, proposal: Path | None) -> None:
                optional_step: Path | None = adaptive_hparam.adaptive_step(
                    "/workflow", proposal_path=proposal, execute=execute,
                )
                nonoptional_step: Path = adaptive_hparam.adaptive_step(
                    "/workflow", proposal_path=proposal, execute=execute,
                )  # type: ignore[assignment]
                proposal_step: Path = adaptive_hparam.adaptive_step(
                    "/workflow", proposal_path=Path("/proposal.json"), execute=execute,
                )
                executing_step: Path = adaptive_hparam.adaptive_step(
                    "/workflow", proposal_path=proposal, execute=True,
                )

            for hparam_plan in (
                run_artifacts.read_hparam_plan(Path("/plan")),
                plan_hparam.commit_hparam_plan(Path("/plan")),
                next(run_artifacts.iter_registered_hparam_plans(
                    Path("/workspace"), "step", selection_metric="score", selection_mode="max", selection_split="val",
                ))[1],
                next(iter(hparam_selection.resolve_hparam_candidates(Path("/plan"), [])[1].values())),
            ):
                plan_recipe: dict[str, Any] = hparam_plan["recipe"]
                plan_runs: list[dict[str, Any]] = hparam_plan["runs"]
                resolved_digest: str = hparam_plan["resolved_recipe_sha256"]
                hparam_plan["recipe"] = []  # type: ignore[typeddict-item]
                hparam_plan["runs"] = {}  # type: ignore[typeddict-item]
                hparam_plan["resolved_recipe_sha256"] = 1  # type: ignore[typeddict-item]
                hparam_plan["resolved_recipe_hash"]  # type: ignore[typeddict-item]
                del hparam_plan["recipe"]  # type: ignore[misc]

            def check_commit(value: object) -> None:
                if models.is_full_git_object_id(value):
                    commit: str = value

            strict_key: tuple[str, str] = experiment_workspace.validated_run_key({})
            optional_key: tuple[str, str] | None = experiment_workspace.managed_run_key({})
            required_key: tuple[str, str] = experiment_workspace.managed_run_key({})  # type: ignore[assignment]

            layouts = plan_hparam.hparam_run_layouts({}, Path("/plan"), 7)
            layout_identity: dict[str, str] = layouts[0]["identity"]
            layout_parameters: dict[str, Any] = layouts[0]["parameters"]
            layout_path: Path = layouts[0]["run_dir"]
            layout_path_text: str = layouts[0]["run_dir"]  # type: ignore[assignment]
            layouts[0]["run_path"]  # type: ignore[typeddict-item]
            layouts[0]["run_dir"] = "/plan/run"  # type: ignore[typeddict-item]
            layouts[0]["identity"]["run_id"] = 7  # type: ignore[assignment]
            layouts[0]["parameters"] = []  # type: ignore[typeddict-item]

            compiled_files = plan_hparam.compile_hparam_run_contracts(
                {}, Path("/plan"), 7, source_config_bytes=b"config",
            )
            compiled_bytes: bytes = compiled_files[0]["config_bytes"]
            compiled_script: str = compiled_files[0]["script_text"]
            compiled_scheduler: str | None = compiled_files[0].get("scheduler_script_text")
            compiled_files[0]["config_bytes"] = "text"  # type: ignore[typeddict-item]
            compiled_files[0]["script_text"] = b"bytes"  # type: ignore[typeddict-item]
            compiled_files[0]["config_byte"]  # type: ignore[typeddict-item]
            del compiled_files[0]["config_bytes"]  # type: ignore[misc]
            row_contracts = plan_hparam.compile_hparam_run_contracts({}, Path("/plan"), 7)
            optional_bytes: bytes | None = row_contracts[0].get("config_bytes")
            required_bytes: bytes = row_contracts[0].get("config_bytes")  # type: ignore[assignment]
            row_contracts[0]["script_text"] = 7  # type: ignore[typeddict-item]

            adapter = TaskAdapter()
            for plan_result in (
                adapter.compile_plan_contract({}, Path("/plan"), run_index_offset=7, config_bytes=b"config"),
                HPARAM_TUNE_ADAPTER.compile_plan_contract(
                    {}, Path("/plan"), run_index_offset=7, config_bytes=b"config",
                ),
                run_artifacts._compile_registered_plan_contract(
                    adapter, {}, Path("/plan"), run_index_offset=7, config_bytes=b"config",
                ),
            ):
                launch_text: str = plan_result["launch_script_text"]
                final_command: str | None = plan_result["final_command"]
                plan_result["launch_script_text"] = b"bytes"  # type: ignore[typeddict-item]
                plan_result["final_eval_config_required"] = "yes"  # type: ignore[typeddict-item]
                plan_result["run_file"]  # type: ignore[typeddict-item]
                for run_files in plan_result["run_files"]:
                    file_bytes: bytes = run_files["config_bytes"]
                    file_script: str = run_files["script_text"]
                    invalid_bytes: str = run_files["config_bytes"]  # type: ignore[assignment]
                    invalid_script: bytes = run_files["script_text"]  # type: ignore[assignment]
                plan_contract.validate_final_eval_contract({}, {}, Path("/plan"), plan_result)
            plan_contract.validate_final_eval_contract({}, {}, Path("/plan"), {})

            resources = slurm.normalize_resources({}, 1)
            cpus: int = resources["cpus_per_task"]
            slurm.submit_token({}, resources, "commit")
            resources["cpu_per_task"]  # type: ignore[typeddict-item]
            resources["cpus_per_task"] = "4"  # type: ignore[typeddict-item]
            slurm.submit_token({}, {**resources, "cpus_per_task": "4"}, "commit")  # type: ignore[typeddict-item]

            capacity = slurm.fixed_node_resource_capacity({}, resources, 4)
            planned_runs: int = capacity["planned_runs"]
            capacity["limits"]  # type: ignore[typeddict-item]
            if capacity["status"] == "known":
                cpu_limit: int = capacity["limits"]["cpu"]
                memory_kib: int = capacity["per_run"]["memory_kib"]
                node_gpus: int = capacity["node_capacity"]["gpus"]
                minimum_waves: int | None = capacity["minimum_waves"]
                required_waves: int = capacity["minimum_waves"]  # type: ignore[assignment]
                capacity["reason"]  # type: ignore[typeddict-item]
                capacity["limits"]["cpu"] = "4"  # type: ignore[assignment]
                capacity["per_run"]["memory_kb"]  # type: ignore[typeddict-item]
                capacity["per_run"]["memory_kib"] = "1024"  # type: ignore[typeddict-item]
                capacity["node_capacity"]["gpus"] = "8"  # type: ignore[typeddict-item]
            else:
                reason: str = capacity["reason"]
                capacity["overall_empty_node_limit"]  # type: ignore[typeddict-item]

            parsed_capabilities = slurm.parse_cluster_scheduling_capabilities(
                version_output="", config_output="", partition_output="",
                reservation_output="", partition="gpu",
            )
            version: str = parsed_capabilities["slurm_version"]
            backfill: bool = parsed_capabilities["backfill_enabled"]
            reservations: int = parsed_capabilities["reservation_count"]
            parsed_capabilities["scheduler_typ"]  # type: ignore[typeddict-item]
            parsed_capabilities["priority_type"] = False  # type: ignore[typeddict-item]
            parsed_capabilities["backfill_enabled"] = 1  # type: ignore[typeddict-item]
            parsed_capabilities["reservation_count"] = "1"  # type: ignore[typeddict-item]

            capabilities = slurm.cluster_scheduling_capabilities({}, partition="gpu")
            partition_state: str = capabilities["partition_state"]
            accounting: bool = capabilities["accounting_enabled"]
            visible_reservations: int = capabilities["reservation_count"]
            capabilities["preemption_enable"]  # type: ignore[typeddict-item]
            capabilities["partition_max_time"] = 1  # type: ignore[typeddict-item]
            capabilities["preemption_enabled"] = 1  # type: ignore[typeddict-item]
            capabilities["reservation_count"] = "1"  # type: ignore[typeddict-item]

            checkpoint_rows = checkpoint_test_results.validate_checkpoint_test_results(
                [], "metric", {}, step_id="tune", run_id="run-001",
            )
            checkpoint_path: str = checkpoint_rows[0]["checkpoint_path"]
            checkpoint_epoch: int = checkpoint_rows[0]["epoch"]
            checkpoint_score: float = checkpoint_rows[0]["score"]
            checkpoint_rows[0]["checkpoint_paths"]  # type: ignore[typeddict-item]
            checkpoint_rows[0]["score"] = "0.5"  # type: ignore[typeddict-item]
            checkpoint_rows[0]["epoch"] = 1.5  # type: ignore[typeddict-item]
            objective_result = adaptive_hparam._test_checkpoint_objective({}, {}, "/checkpoints", [])
            objective_result["score"]  # type: ignore[index]
            if objective_result is not None:
                objective_score: float = objective_result["score"]
                objective_result["checkpoint_paths"]  # type: ignore[typeddict-item]

            class RankingRow(checkpoint_test_results.CheckpointTestResult):
                checkpoint_sha256: str

            ranking_rows: list[RankingRow] = []
            winner = checkpoint_test_results.best_checkpoint_test_result(ranking_rows, "max")
            winner_hash: str = winner["checkpoint_sha256"]
            winner["checkpoint_sha256"] = 1  # type: ignore[typeddict-item]
            winner["checkpoint_sha"]  # type: ignore[typeddict-item]

            identity = run_evidence._parse_process_identity("{}", "/identity.json")
            pid: int = identity["pid"]
            process_group: int = identity["process_group_id"]
            start_token: str = identity["process_start_token"]
            identity["process_start_tokens"]  # type: ignore[typeddict-item]
            identity["pid"] = "12"  # type: ignore[typeddict-item]
            identity["runtime_commit"] = 1  # type: ignore[typeddict-item]
            minimal_identity: run_evidence.ProcessIdentity = {
                "pid": 12, "process_group_id": 12, "process_start_token": "token",
            }
            incomplete_identity: run_evidence.ProcessIdentity = {"pid": 12}  # type: ignore[typeddict-item]
            running: bool | None = run_evidence.process_identity_running({}, minimal_identity)
            run_evidence.stop_process_group({}, minimal_identity)
            run_evidence.process_identity_running({}, {**minimal_identity, "pid": "12"})  # type: ignore[typeddict-item]
            run_evidence.stop_process_group({}, {"pid": 12})  # type: ignore[typeddict-item]
            read_identity = run_evidence.read_process_identity("/identity.json")
            read_identity["pid"]  # type: ignore[index]
            if read_identity is not None:
                read_pid: int = read_identity["pid"]
                read_identity["process_start_tokens"]  # type: ignore[typeddict-item]
                if "runtime_commit" in read_identity:
                    runtime_commit: str = read_identity["runtime_commit"]

            default_files = experiment_io.read_managed_files_at("/workspace", ["/workspace/file"])
            default_text: str = default_files["/workspace/file"]["text"]
            default_sha: str = default_files["/workspace/file"]["sha256"]
            default_files["/workspace/file"]["sha265"]  # type: ignore[typeddict-item]
            default_files["/workspace/file"]["text"] = None  # type: ignore[typeddict-item]
            strict_files = experiment_io.read_managed_files_at("/workspace", [], allow_invalid_utf8=False)
            strict_text: str = strict_files["file"]["text"]
            strict_sha: str = strict_files["file"]["sha256"]
            strict_files["file"]["sha256"] = None  # type: ignore[typeddict-item]
            permissive_files = experiment_io.read_managed_files_at("/workspace", [], allow_invalid_utf8=True)
            optional_text: str | None = permissive_files["file"]["text"]
            permissive_sha: str = permissive_files["file"]["sha256"]
            required_text: str = permissive_files["file"]["text"]  # type: ignore[assignment]
            permissive_files["file"]["sha256"] = None  # type: ignore[arg-type]

            def check_dynamic_file_read(allow_invalid: bool) -> None:
                files = experiment_io.read_managed_files_at("/workspace", [], allow_invalid_utf8=allow_invalid)
                dynamic_text: str | None = files["file"]["text"]
                dynamic_sha: str = files["file"]["sha256"]
                dynamic_required: str = files["file"]["text"]  # type: ignore[assignment]
                files["file"]["sha256"] = 1  # type: ignore[arg-type]

            plan = run_artifacts.read_registered_plan(
                "/plan", workspace="/workspace", workspace_experiment={},
                step_manifest={}, workspace_rows=[], expected_recipe_path=None,
            )
            key: tuple[str, str] = plan["run_keys"][0]
            plan["run_key"]  # type: ignore[typeddict-item]
            plan["run_keys"] = ["run-001"]  # type: ignore[list-item]
            plan["selection"]["metric"]  # type: ignore[index]
            selection = plan["selection"]
            if selection is not None:
                metric: str = selection["metric"]
                selection["metric"] = 1  # type: ignore[typeddict-item]

            steps = experiments._registered_plan_steps(
                Path("/workspace"), {}, [], remote=None, require_registered_rows=True,
            )
            steps[0]["plans"][0]["run_key"]  # type: ignore[typeddict-item]
            lifecycle = experiment_tracking.hparam_selection_lifecycle(steps, [], root=Path("/workspace"))
            lifecycle_step = lifecycle["hparam_steps"][0]
            selected_step = lifecycle["selected_steps"][0]
            plan_path: str = lifecycle_step["plan_path"]
            policy_metric: str = lifecycle_step["selection"]["metric"]
            lifecycle_step["selection"]["mode"] = 1  # type: ignore[typeddict-item]
            lifecycle_step["plans"][0]["run_keys"] = ["run-001"]  # type: ignore[list-item]
            lifecycle["pending_steps"][0]["plan_path"] = None  # type: ignore[typeddict-item]
            selected_step["rankings"]  # type: ignore[typeddict-item]
            selected_step["ranked"] = ["run-001"]  # type: ignore[list-item]
            selected_step["legacy_selection"] = "yes"  # type: ignore[typeddict-item]
            if "checkpoint_audit_rows" in selected_step:
                audit_score: str = selected_step["checkpoint_audit_rows"][0]["score"]
                selected_step["checkpoint_audit_rows"][0]["score"] = 0.5  # type: ignore[assignment]
            report_steps = hparam_selection._selection_report_steps([])
            report_steps[0]["step_id"] = 1  # type: ignore[typeddict-item]
            report_steps[0]["ranked"] = ["run-001"]  # type: ignore[list-item]
            ranking_input: experiment_tracking.HparamSelectionReportStep = {
                "step_id": "step", "selection": {"metric": "score", "mode": "max", "split": "val"}, "rows": [],
            }
            experiment_tracking.validated_hparam_ranking(ranking_input)
            experiment_tracking.hparam_selection_report_text(report_steps, root=Path("/workspace"))
            experiment_tracking.hparam_selection_report_text(lifecycle["selected_steps"], root=Path("/workspace"))
            experiment_tracking._hparam_ranking_matches(report_steps, "csv")
            experiments._validate_hparam_checkpoints([], lifecycle["selected_steps"], remote=None)
            invalid_report_step = {"step_id": "step", "selection": {}, "rows": "bad"}
            experiment_tracking.validated_hparam_ranking(invalid_report_step)  # type: ignore[arg-type]
            experiment_tracking.hparam_selection_report_text(
                [invalid_report_step], root=Path("/workspace"),  # type: ignore[list-item]
            )
            experiments._validate_hparam_checkpoints([], [invalid_report_step], remote=None)  # type: ignore[list-item]
            expected_report: str | None = lifecycle["expected_report"]
            report_valid: bool = lifecycle["report_valid"]
            lifecycle["selected_step"]  # type: ignore[typeddict-item]
            lifecycle["report_valid"] = "yes"  # type: ignore[typeddict-item]
            required_report: str = lifecycle["expected_report"]  # type: ignore[assignment]
            status_snapshot = experiment_tracking.experiment_status_snapshot({}, steps, [], root=Path("/workspace"))
            status_experiment = status_snapshot["experiment"]
            status_title: str = status_experiment["title"]
            status_remote: str | None = status_experiment["remote"]
            status_experiment["experiment_id"]  # type: ignore[typeddict-item]
            required_remote: str = status_experiment["remote"]  # type: ignore[assignment]
            status_experiment["title"] = None  # type: ignore[typeddict-item]
            status_step = status_snapshot["steps"][0]
            status_plans: list[str] = status_step["plans"]
            step_status_counts: dict[str, int] = status_step["status_counts"]
            status_step["plan_controller"] = 1  # type: ignore[typeddict-item]
            status_step["plans"] = [1]  # type: ignore[list-item]
            status_step["status_counts"]["completed"] = "1"  # type: ignore[assignment]
            status_run = status_snapshot["runs"][0]
            run_id: str = status_run["run_id"]
            status_run["execution"]["host"] = 1  # type: ignore[typeddict-item]
            status_run["scheduler"]["job_id"] = 1  # type: ignore[typeddict-item]
            status_run["scheduler"]["jobid"]  # type: ignore[typeddict-item]
            status_run["process"]["pid"] = 123  # type: ignore[typeddict-item]
            status_run["evidence"]["checkpoint_count"] = 2  # type: ignore[typeddict-item]
            status_run["evidence"]["log_age_seconds"] = 1.0  # type: ignore[typeddict-item]
            status_run["blockers"] = [1]  # type: ignore[list-item]
            optional_pid: str | None = status_run["process"]["pid"]
            required_pid: str = status_run["process"]["pid"]  # type: ignore[assignment]
            constructed_run = experiment_tracking._status_run_payload({})
            constructed_run["evidence"]["checkpoint_count"] = 2  # type: ignore[typeddict-item]
            missing_execution_host: experiment_tracking.ExperimentStatusExecution = {  # type: ignore[typeddict-item]
                "target": None,
            }
            public_status = experiments.experiment_status("/workspace")
            public_status["runs"][0]["evidence"]["checkpoint_count"] = 2  # type: ignore[typeddict-item]
            status_summary = status_snapshot["summary"]
            status_counts: dict[str, int] = status_summary["status_counts"]
            run_count: int = status_summary["run_count"]
            status_summary["run_counts"]  # type: ignore[typeddict-item]
            status_summary["run_count"] = "1"  # type: ignore[typeddict-item]
            status_summary["status_counts"]["completed"] = "1"  # type: ignore[assignment]
            status_blocker = status_snapshot["blockers"][0]
            blocker_step: str | None = status_blocker["step_id"]
            blocker_runs: list[str] = status_blocker["run_ids"]
            status_blocker["run_id"]  # type: ignore[typeddict-item]
            status_blocker["blocked_actions"] = [1]  # type: ignore[list-item]
            constructed_blocker = experiment_tracking._status_blocker("missing_stop_reason", "Record a reason")
            constructed_blocker["message"] = None  # type: ignore[typeddict-item]
            manual_choice: bool = status_snapshot["decision"]["manual_choice_required"]
            blocked_actions: list[str] = status_snapshot["decision"]["blocked_actions"]
            status_snapshot["decisions"]  # type: ignore[typeddict-item]
            status_snapshot["decision"]["manual_choice_required"] = 1  # type: ignore[typeddict-item]
            status_snapshot["decision"]["recommended_next"]["argv"]  # type: ignore[index]
            recommended = status_snapshot["decision"]["recommended_next"]
            if recommended is not None:
                action_argv: list[str] = recommended["argv"]
                action_host: str | None = recommended["control_host"]
                recommended["command"]  # type: ignore[typeddict-item]
                recommended["argv"] = [1]  # type: ignore[list-item]
                recommended["required_inputs"] = "report_path"  # type: ignore[typeddict-item]
                if "required_inputs" in recommended:
                    required_inputs: list[str] = recommended["required_inputs"]
            alternative = status_snapshot["decision"]["other_legal_actions"][0]
            alternative["reason"] = 1  # type: ignore[typeddict-item]
            constructed_action = experiment_tracking._status_action("monitor", "Refresh evidence", ["python"])
            constructed_action["argv"] = "python"  # type: ignore[typeddict-item]
            missing_action_fields: experiment_tracking.ExperimentStatusAction = {  # type: ignore[typeddict-item]
                "id": "monitor",
            }
            experiment_tracking.hparam_selection_lifecycle(
                [{"manifest": {}, "plans": ["/plan"]}], [], root=Path("/workspace"),  # type: ignore[list-item]
            )

            report = experiments._hparam_selection_report(Path("/workspace"), remote=None)
            report["text"]  # type: ignore[index]
            if report is not None:
                report_path: str = report["path"]
                report_text: str = report["text"]
                report_sha: str = report["sha256"]
                ranking_path: str = report["ranking_path"]
                ranking_text: str | None = report["ranking_text"]
                ranking_sha: str | None = report["ranking_sha256"]
                report["ranking_sha265"]  # type: ignore[typeddict-item]
                report["sha256"] = None  # type: ignore[typeddict-item]
                required_ranking: str = report["ranking_text"]  # type: ignore[assignment]
                required_ranking_sha: str = report["ranking_sha256"]  # type: ignore[assignment]
                experiment_tracking.hparam_selection_lifecycle(steps, [], root=Path("/workspace"), report=report)
                experiment_tracking.experiment_status_snapshot(
                    {}, steps, [], root=Path("/workspace"), hparam_selection_report=report,
                )
                experiments._validate_hparam_selection_files_unchanged(Path("/workspace"), report, {}, remote=None)
                experiment_tracking.hparam_selection_lifecycle(
                    steps, [], root=Path("/workspace"), report={**report, "sha256": None},  # type: ignore[arg-type]
                )
                experiment_tracking.experiment_status_snapshot(
                    {}, steps, [], root=Path("/workspace"),
                    hparam_selection_report={**report, "ranking_text": 1},  # type: ignore[arg-type]
                )
                experiments._validate_hparam_selection_files_unchanged(
                    Path("/workspace"), {**report, "path": None}, {}, remote=None,  # type: ignore[typeddict-item]
                )
            """),
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    # Unused ignores fail if a producer or consumer regresses to Any.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(root / "pyproject.toml"),
            "--follow-imports=silent",
            "--warn-unused-ignores",
            "--no-incremental",
            str(probe),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
