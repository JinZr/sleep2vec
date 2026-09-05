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
                experiment_io, run_artifacts, run_evidence, slurm,
            )

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
            experiment_tracking.hparam_selection_lifecycle(steps, [], root=Path("/workspace"))
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
