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
                experiment_tracking,
                execution_snapshot, experiments,
                experiment_io, experiment_workspace, hparam_selection,
                managed_scheduler, models,
                plan_contract, plan_hparam, run_artifacts, run_evidence, slurm,
            )

            from agent_tools.adapters.base import TaskAdapter, recipe_inputs
            from agent_tools.decision_models import DecisionIssue

            from typing import Any, Literal

            direct_launch: managed_scheduler.DirectLaunchIdentity = {
                "target": "local", "workdir": Path("/runtime"),
                "gpus": "", "pid_path": "/run/pid", "log_path": "/run/stdout.log", "command": "",
            }
            direct_log: str = direct_launch["log_path"]
            direct_launch["log_path"] = Path("/log")  # type: ignore[typeddict-item]
            direct_launch["pid"] = "123"  # type: ignore[typeddict-item]
            direct_launch["planned_runtime_commit"] = None  # type: ignore[typeddict-item]
            slurm_launch = managed_scheduler._slurm_execution_identity({}, {})
            submit_command: str = slurm_launch["command"]
            slurm_launch["command"] = None  # type: ignore[typeddict-item]
            slurm_launch["execution_snapshot_sha256"] = ""  # type: ignore[typeddict-unknown-key]
            slurm_launch["log_path"] = None
            verification_options: managed_scheduler.LaunchVerificationOptions = {}
            verification_options["checkpoint_path"] = None
            verification_options["checkpoint_sha256"] = None
            verification_options["checkpoint_path"] = "path"  # type: ignore[typeddict-item]
            verification_options["run_id"] = None  # type: ignore[typeddict-item]
            managed_scheduler.build_launch_command({}, Path("/script"), "/log", "/pid", [], **verification_options)
            frozen_artifact: managed_scheduler.FrozenLaunchArtifact = {"path": "/config", "sha256": "a" * 64}
            frozen_artifact["sha256"] = None  # type: ignore[typeddict-item]

            job = slurm.JobObservation("42", "exp/step/run-000", "PENDING")
            job_state: str = job.state
            slurm.JobObservation(42, "exp/step/run-000", "PENDING")  # type: ignore[arg-type]
            listed_jobs: list[slurm.JobObservation] = slurm.user_jobs(timeout=1)
            submitted_job: str = slurm.submit("/plan/job.sbatch", timeout=1)
            slurm_jobs: dict[str, slurm.JobObservation] = managed_scheduler.slurm_job_snapshot([{"status": "queued"}])

            def snapshot_result(should_write: bool) -> tuple[execution_snapshot.ExecutionSnapshot, bool]:
                return {}, should_write

            managed_scheduler.SchedulerHooks(validated_snapshot=lambda *args: snapshot_result(True))
            managed_scheduler.SchedulerHooks(validated_snapshot=lambda *args: (None, False))
            invalid_snapshot: execution_snapshot.ExecutionSnapshotResult = (None, True)  # type: ignore[assignment]

            def missing_snapshot_write() -> tuple[None, Literal[True]]:
                return None, True

            managed_scheduler.SchedulerHooks(validated_snapshot=missing_snapshot_write)  # type: ignore[arg-type]

            for snapshot in (
                execution_snapshot.inspect_execution_target({}, []),
                execution_snapshot.validated_execution_snapshot(Path("/plan"), {}, [], {})[0],
            ):
                module_name: str = snapshot["module"]
                snapshot_commit: str = snapshot["runtime_commit"]
                options: list[str] = snapshot["required_options"]
                argv_digest: str = snapshot["validated_argv_sha256"]
                snapshot["module"] = 1  # type: ignore[typeddict-item]
                snapshot["required_options"] = [1]  # type: ignore[list-item]
                snapshot["validated_argv_sha256"] = b"hash"  # type: ignore[typeddict-item]
                snapshot["module_name"]  # type: ignore[typeddict-item]
                execution_snapshot.write_execution_snapshot_file(Path("/snapshot"), snapshot)
                managed_scheduler.build_launch_command(
                    {}, Path("script"), "log", "pid", [], execution_snapshot=snapshot,
                )

            from agent_tools.domain.index_csv import index_summary
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
            index["covariates"]["age"]["missing_rows"] = "1"  # type: ignore[typeddict-item]
            if index["survival_key"] is not None:
                key_count: int | None = index["survival_key"]["sidecar_key_count"]
                index["survival_key"]["missing_from_sidecars_examples"] = [1]  # type: ignore[list-item]
            preset = preset_summary(Path("samples.pkl"))
            sample_count: int = preset["samples"]
            preset["source_counts"] = {"source": "1"}  # type: ignore[dict-item]
            preset["sidecar_manifest"] = ["unvalidated", 1]
            summary = plan_context.context_index_summary({}, None)
            if summary is not None:
                errors: list[str] = summary["blocking_issues"]
                errors = summary["blocking_issues"][0]  # type: ignore[assignment]
            questions = markdown.questions_payload(DecisionReport(status=DecisionStatus.FAIL))
            question_text: str | None = questions[0]["question"]
            questions[0]["field"] = 1  # type: ignore[typeddict-item]

            def diagnostic_contracts() -> None:
                from agent_tools import configs
                from agent_tools.domain.finetune_summary import finetune_summary_body
                from agent_tools.domain.sex_age_summary import sex_age_baseline_config_summary
                from agent_tools.domain.sidecar_summaries import survival_summary, multilabel_summary
                from agent_tools.adapters.sleep2stat import sleep2stat_config_summary

                for diagnostic in (
                    configs.config_summary(Path("config.yaml")),
                    sleep2stat_config_summary(Path("config.yaml")),
                    sex_age_baseline_config_summary(Path("config.yaml")),
                ):
                    warnings: list[str] = diagnostic["warnings"]
                    config_path: str = diagnostic["config_path"]
                    diagnostic["blocking_issues"] = [1]  # type: ignore[list-item]
                    diagnostic["_source_config_bytes"] = "bytes"  # type: ignore[arg-type]
                fine = finetune_summary_body(Path("config.yaml"))
                fine["model"]["channels"][0]["unknown"] = 1  # type: ignore[typeddict-unknown-key]
                fine["model"]["layer_mix_present"] = "yes"  # type: ignore[typeddict-item]
                fine["model"]["head_details"]["kwargs"] = []  # type: ignore[typeddict-item]
                fine["finetune"]["tuning_present"] = 1  # type: ignore[typeddict-item]
                fine["data"]["train_dataset_names"] = "train"  # type: ignore[typeddict-item]
                fine["finetune"]["task"]["output_dim"] = {"invalid": "preserved"}
                sex = sex_age_baseline_config_summary(Path("config.yaml"))
                sex["model"]["name"] = 1  # type: ignore[typeddict-item]
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

            minimal_snapshot: execution_snapshot.ExecutionSnapshot = {
                "module": "runtime_cli", "module_origin": "/runtime_cli.py",
            }
            managed_scheduler.build_launch_command(
                {}, Path("script"), "log", "pid", [], execution_snapshot=minimal_snapshot,
            )
            managed_scheduler.build_launch_command(
                {}, Path("script"), "log", "pid", [],
                execution_snapshot={"module": 1},  # type: ignore[arg-type]
            )

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
            experiment_tracking.monitor_report((direct_health,))
            monitor_result = experiments.monitor_experiment("/workspace")
            monitor_report_path: str = monitor_result["report"]
            monitor_result["runs"] = ["run-000"]  # type: ignore[list-item]
            monitor_result["report"] = Path("/report")  # type: ignore[typeddict-item]
            monitor_result["decision"]  # type: ignore[typeddict-item]
            monitor_result["runs"][0]["status"] = None  # type: ignore[assignment]
            canonical_rows = experiment_workspace.read_run_manifest(Path("/workspace"))
            canonical_status: str = canonical_rows[0]["status"]
            canonical_rows[0]["status"] = None  # type: ignore[assignment]
            monitored_row = experiment_tracking.monitor_run_row(Path("/workspace"), canonical_rows[0], canonical_rows)
            monitored_status: models.JsonValue = monitored_row["status"]
            monitored_text: str = monitored_row["status"]  # type: ignore[assignment]
            experiment_tracking.monitor_run_row(Path("/workspace"), monitored_row, [])  # type: ignore[arg-type]
            run_evidence.status_row(
                Path("/run"), canonical_rows[0], monitored_row, script_commits_terminal_status=False,
            )
            run_evidence.status_row(
                Path("/run"), monitored_row, canonical_rows[0], script_commits_terminal_status=False,
            )
            run_evidence.status_row(
                Path("/run"), {"pid_path": Path("/pid")},  # type: ignore[dict-item]
                script_commits_terminal_status=False,
            )
            experiment_events = experiment_workspace.read_experiment_events("/workspace")
            event_value: models.JsonValue = experiment_events[0]["event"]
            event_text: str = experiment_events[0]["event"]  # type: ignore[assignment]
            resolved_canonical = experiment_workspace.resolve_run_row(canonical_rows, experiment_events[0])
            if resolved_canonical is not None:
                resolved_status: str = resolved_canonical["status"]
            resolved_observed = experiment_workspace.resolve_run_row([monitored_row], experiment_events[0])
            if resolved_observed is not None:
                resolved_observed_text: str = resolved_observed["status"]  # type: ignore[assignment]

            planned: execution_snapshot.PlannedArgv = {"run_id": "run-000", "args": ["--value", "ok"]}
            planned["args"] = [1]  # type: ignore[list-item]
            planned["run_id"] = 1  # type: ignore[typeddict-item]

            from types import MappingProxyType

            raw_event: dict[str, Any] = {"round": None, "request_id": None, "custom": [None]}
            experiment_workspace.append_event(Path("/workspace"), "event", raw_event)
            readonly_event = MappingProxyType(raw_event)
            experiment_workspace.event_matches({}, "event", readonly_event)
            experiment_workspace.append_event(Path("/workspace"), "event", readonly_event)  # type: ignore[arg-type]

            for hparam_plan in (
                run_artifacts.read_hparam_plan(Path("/plan")),
                run_artifacts.commit_hparam_plan(Path("/plan")),
                next(run_artifacts.iter_registered_hparam_plans(
                    Path("/workspace"), "step", selection_metric="score", selection_mode="max", selection_split="val",
                ))[1],
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

            frozen_context = plan_contract.frozen_plan_context({})
            frozen_home: str = frozen_context["home"]
            frozen_context["python"] = Path("/python")  # type: ignore[typeddict-item]
            frozen_context["repo"]  # type: ignore[typeddict-item]
            del frozen_context["repo_root"]  # type: ignore[misc]
            for frozen_input in (
                plan_contract.frozen_input_snapshots({})[0],
                plan_contract.frozen_input_snapshot({}, "inputs.config"),
            ):
                frozen_path: str = frozen_input["path"]
                frozen_hash: str = frozen_input["sha256"]
                frozen_input["field"] = None  # type: ignore[typeddict-item]
                frozen_input["sha256"] = b"hash"  # type: ignore[typeddict-item]
                frozen_input["source_path"]  # type: ignore[typeddict-item]
                del frozen_input["sha256"]  # type: ignore[misc]

            def post_freeze_recipe_writers() -> None:
                frozen_recipe: dict[str, models.JsonValue] = {"inputs": {"config": "/config.yaml"}}
                plan_contract.bind_plan_context(frozen_recipe)
                plan_contract.bind_frozen_input_snapshot(frozen_recipe, "inputs.config", "/config.yaml", "a" * 64)
                byte_recipe = {"_plan_context": b"context"}
                plan_contract.bind_plan_context(byte_recipe)  # type: ignore[arg-type]
                plan_contract.bind_frozen_input_snapshot(
                    byte_recipe, "inputs.config", "/config.yaml", "a" * 64  # type: ignore[arg-type]
                )

            class FrozenRecipeHookAdapter(TaskAdapter):
                def bind_effective_recipe(
                    self,
                    recipe: dict[str, models.JsonValue],
                    config_summary: models.ConfigSummaryInput | None,
                    *,
                    source_recipe: dict[str, Any] | None = None,
                ) -> list[DecisionIssue]:
                    recipe["runtime"] = {"devices": [0]}
                    recipe["runtime"] = b"runtime"  # type: ignore[assignment]
                    return []

            class ByteRecipeHookAdapter(TaskAdapter):
                def bind_effective_recipe(  # type: ignore[override]
                    self,
                    recipe: dict[str, bytes],
                    config_summary: models.ConfigSummaryInput | None,
                    *,
                    source_recipe: dict[str, Any] | None = None,
                ) -> list[DecisionIssue]:
                    return []

            frozen_inputs: dict[str, models.JsonValue] = recipe_inputs({})
            recipe_inputs({})["config"] = b"config"  # type: ignore[assignment]

            final_descriptor: plan_contract.FinalEvalConfigDescriptor = {
                "path": "/plan/config.final_eval.yaml", "sha256": "a" * 64, "source_path": "relative.yaml",
            }
            final_descriptor["source_path"] = None  # type: ignore[typeddict-item]
            final_descriptor["source_path"] = {"raw": [None, 1]}  # type: ignore[typeddict-item]
            bound_final: models.BoundFinalEvalConfigSnapshot = {
                "source_path": "relative.yaml", "bytes": b"config", "sha256": "a" * 64,
            }
            bound_final["source_path"] = Path("relative.yaml")  # type: ignore[typeddict-item]
            bound_final["bytes"] = "config"  # type: ignore[typeddict-item]
            del bound_final["sha256"]  # type: ignore[misc]
            del final_descriptor["path"]  # type: ignore[misc]

            generic_run = plan_contract.generic_run_contract({}, Path("/plan"), 7, TaskAdapter())
            generic_config: str = generic_run["config"]
            generic_run["config"] = Path("/config")  # type: ignore[typeddict-item]
            generic_run["run_id"] = 7  # type: ignore[typeddict-item]
            generic_run["scheduler_type"] = "unknown"  # type: ignore[typeddict-item]
            generic_run["experiment_id"] = None  # type: ignore[typeddict-item]
            generic_run["step_id"] = {"raw": None}  # type: ignore[typeddict-item]
            del generic_run["script"]  # type: ignore[misc]

            generic_compiled: plan_contract.GenericCompiledPlanContract = {
                "runs": [generic_run], "commands": [], "script_text": "",
            }
            del generic_compiled["commands"]  # type: ignore[misc]
            del generic_compiled["script_text"]  # type: ignore[misc]
            raw_plan, raw_recipe = run_artifacts._read_plan_documents(Path("/plan"))
            raw_plan["custom"] = {"raw": [None, 1]}
            raw_recipe["custom"] = None
            provenance: models.ConfigDiagnostics = {
                "config_path": "config.yaml", "data_backend": "npz", "warnings": [], "blocking_issues": [],
            }
            provenance["_final_eval_config_snapshot"] = bound_final
            provenance["_final_eval_config_snapshot"] = {"source_path": "x"}  # type: ignore[typeddict-item]

            layouts = plan_hparam.hparam_run_layouts({}, Path("/plan"), 7)
            layout_identity: experiment_workspace.RunIdentity = layouts[0]["identity"]
            layout_parameters: dict[str, Any] = layouts[0]["parameters"]
            layout_path: Path = layouts[0]["run_dir"]
            layout_path_text: str = layouts[0]["run_dir"]  # type: ignore[assignment]
            layouts[0]["run_path"]  # type: ignore[typeddict-item]
            layouts[0]["run_dir"] = "/plan/run"  # type: ignore[typeddict-item]
            layouts[0]["identity"]["run_id"] = 7  # type: ignore[typeddict-item]
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

            resources = slurm.normalize_resources({}, 1)
            cpus: int = resources["cpus_per_task"]
            resources["cpu_per_task"]  # type: ignore[typeddict-item]
            resources["cpus_per_task"] = "4"  # type: ignore[typeddict-item]


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
                process_group: int = read_identity["process_group_id"]
                start_token: str = read_identity["process_start_token"]
                read_identity["process_start_tokens"]  # type: ignore[typeddict-item]
                read_identity["pid"] = "12"  # type: ignore[typeddict-item]
                read_identity["runtime_commit"] = 1  # type: ignore[typeddict-item]
                if "runtime_commit" in read_identity:
                    runtime_commit: str = read_identity["runtime_commit"]

            default_files = experiment_io.read_managed_files_at("/workspace", ["/workspace/file"])
            default_text: str = default_files["/workspace/file"]["text"]
            default_sha: str = default_files["/workspace/file"]["sha256"]
            default_files["/workspace/file"]["sha265"]  # type: ignore[typeddict-item]
            default_files["/workspace/file"]["text"] = None  # type: ignore[typeddict-item]
            default_files["/workspace/file"]["sha256"] = None  # type: ignore[typeddict-item]

            plan = run_artifacts.read_registered_plan(
                "/plan", workspace_experiment={}, step_manifest={}, workspace_rows=[],
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
                Path("/workspace"), {}, [], require_registered_rows=True,
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
            selected_row_status: str = selected_step["rows"][0]["status"]
            selected_step["rows"][0]["status"] = None  # type: ignore[assignment]
            report_steps = hparam_selection._selection_report_steps([])
            report_steps[0]["step_id"] = 1  # type: ignore[typeddict-item]
            report_steps[0]["ranked"] = ["run-001"]  # type: ignore[list-item]
            report_steps[0]["rows"][0]["status"] = 1  # type: ignore[assignment]
            ranking_input: experiment_tracking.HparamSelectionReportStep = {
                "step_id": "step", "selection": {"metric": "score", "mode": "max", "split": "val"}, "rows": [],
            }
            experiment_tracking.validated_hparam_ranking(ranking_input)
            experiment_tracking.hparam_selection_report_text(report_steps, root=Path("/workspace"))
            experiment_tracking.hparam_selection_report_text(lifecycle["selected_steps"], root=Path("/workspace"))
            invalid_report_step = {"step_id": "step", "selection": {}, "rows": "bad"}
            experiment_tracking.validated_hparam_ranking(invalid_report_step)  # type: ignore[arg-type]
            experiment_tracking.hparam_selection_report_text(
                [invalid_report_step], root=Path("/workspace"),  # type: ignore[list-item]
            )
            expected_report: str | None = lifecycle["expected_report"]
            report_valid: bool = lifecycle["report_valid"]
            lifecycle["selected_step"]  # type: ignore[typeddict-item]
            lifecycle["report_valid"] = "yes"  # type: ignore[typeddict-item]
            required_report: str = lifecycle["expected_report"]  # type: ignore[assignment]
            status_snapshot = experiment_tracking.experiment_status_snapshot({}, steps, [], root=Path("/workspace"))
            status_experiment = status_snapshot["experiment"]
            status_title: str = status_experiment["title"]
            status_root: str = status_experiment["root"]
            status_experiment["experiment_id"]  # type: ignore[typeddict-item]
            status_experiment["title"] = None  # type: ignore[typeddict-item]
            status_step = status_snapshot["steps"][0]
            status_plans: list[str] = status_step["plans"]
            step_status_counts: dict[str, int] = status_step["status_counts"]
            status_step["plans"] = [1]  # type: ignore[list-item]
            status_step["status_counts"]["completed"] = "1"  # type: ignore[assignment]
            status_run = status_snapshot["runs"][0]
            run_id: str = status_run["run_id"]
            status_run["execution"]["target"] = 1  # type: ignore[typeddict-item]
            status_run["scheduler"]["job_id"] = 1  # type: ignore[typeddict-item]
            status_run["scheduler"]["jobid"]  # type: ignore[typeddict-item]
            status_run["process"]["pid"] = 123  # type: ignore[typeddict-item]
            status_run["evidence"]["checkpoint_count"] = 2  # type: ignore[typeddict-item]
            status_run["evidence"]["log_age_seconds"]  # type: ignore[typeddict-item]
            status_run["blockers"] = [1]  # type: ignore[list-item]
            optional_pid: str | None = status_run["process"]["pid"]
            required_pid: str = status_run["process"]["pid"]  # type: ignore[assignment]
            missing_execution_target: experiment_tracking.ExperimentStatusExecution = {}  # type: ignore[typeddict-item]
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
            status_blocker["blocked_actions"]  # type: ignore[typeddict-item]
            status_blocker["message"] = None  # type: ignore[typeddict-item]
            status_snapshot["decision"]  # type: ignore[typeddict-item]
            experiment_tracking.hparam_selection_lifecycle(
                [{"manifest": {}, "plans": ["/plan"]}], [], root=Path("/workspace"),  # type: ignore[list-item]
            )

            report = experiments._hparam_selection_report(Path("/workspace"))
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
                experiment_tracking.hparam_selection_lifecycle(
                    steps, [], root=Path("/workspace"), report={**report, "sha256": None},  # type: ignore[arg-type]
                )
                experiment_tracking.experiment_status_snapshot(
                    {}, steps, [], root=Path("/workspace"),
                    hparam_selection_report={**report, "ranking_text": 1},  # type: ignore[arg-type]
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
