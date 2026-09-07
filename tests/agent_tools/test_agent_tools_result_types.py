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
                experiment_io, experiment_workspace, hparam_runtime, hparam_selection, managed_scheduler, models,
                plan_contract, plan_hparam, run_artifacts, run_evidence, slurm,
            )

            from agent_tools.adapters.base import TaskAdapter
            from agent_tools.adapters.hparam_tune import HPARAM_TUNE_ADAPTER

            from typing import Any, Literal

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

            planned: managed_scheduler.PlannedArgv = {"run_id": "run-000", "args": ["--value", "ok"]}
            planned["args"] = [1]  # type: ignore[list-item]
            planned["run_id"] = 1  # type: ignore[typeddict-item]

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
            expected_report: str | None = lifecycle["expected_report"]
            report_valid: bool = lifecycle["report_valid"]
            lifecycle["selected_step"]  # type: ignore[typeddict-item]
            lifecycle["report_valid"] = "yes"  # type: ignore[typeddict-item]
            required_report: str = lifecycle["expected_report"]  # type: ignore[assignment]
            status_snapshot = experiment_tracking.experiment_status_snapshot({}, steps, [], root=Path("/workspace"))
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
