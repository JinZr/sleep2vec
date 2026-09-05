from pathlib import Path
import subprocess
import sys
import textwrap


def test_result_types_reach_callers(tmp_path: Path):
    probe = tmp_path / "result_type_probe.py"
    probe.write_text(
        textwrap.dedent("""\
            from pathlib import Path
            from agent_tools import experiment_tracking, experiments, run_artifacts, slurm

            resources = slurm.normalize_resources({}, 1)
            cpus: int = resources["cpus_per_task"]
            slurm.submit_token({}, resources, "commit")
            resources["cpu_per_task"]  # type: ignore[typeddict-item]
            resources["cpus_per_task"] = "4"  # type: ignore[typeddict-item]
            slurm.submit_token({}, {**resources, "cpus_per_task": "4"}, "commit")  # type: ignore[typeddict-item]

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
