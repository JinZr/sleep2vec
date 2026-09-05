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
                adaptive_hparam, checkpoint_test_results, experiment_tracking, experiments, run_artifacts, slurm,
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
