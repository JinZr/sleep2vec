from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
from test_agent_tools_infer_slurm import (  # noqa: F401
    _infer_slurm_recipe,
    _no_external_scheduler,
    _runtime_commit,
    _runtime_probe,
)
import yaml

from agent_tools import experiment_workspace, experiments, plans, slurm
from agent_tools.models import REPO_ROOT


@pytest.mark.parametrize(
    ("variant", "task", "mode", "direct_controller", "expected_exit", "scheduler_state", "expected_status"),
    [
        ("sleep2vec", "infer", "success", False, 0, "COMPLETED", "completed"),
        ("sleep2vec2", "evaluate", "nonzero", True, 17, "FAILED", "failed"),
        ("sleep2expert", "infer", "inner-kill", False, 137, "TIMEOUT", "failed"),
        ("sex_age_baseline", "evaluate", "success", True, 0, "COMPLETED", "completed"),
        ("sleep2vec", "infer", "topology", True, 2, "FAILED", "failed"),
        ("sleep2vec2", "evaluate", "runtime-drift", False, 1, "FAILED", "failed"),
        ("sleep2expert", "infer", "outer-kill", True, -9, "TIMEOUT", "unknown_scheduler"),
        ("sleep2vec", "infer", "success", False, 0, "MISSING", "completed"),
        ("sleep2vec2", "evaluate", "nonzero", True, 17, "MISSING", "failed"),
    ],
)
@pytest.mark.usefixtures("_runtime_probe")
def test_generated_infer_worker_commits_only_authenticated_terminal_evidence(
    tmp_path: Path,
    monkeypatch,
    variant,
    task,
    mode,
    direct_controller,
    expected_exit,
    scheduler_state,
    expected_status,
    request,
):
    recipe_path = _infer_slurm_recipe(
        tmp_path, variant=variant, task=task, runtime_commit=request.getfixturevalue("_runtime_commit")
    )
    recipe = yaml.safe_load(recipe_path.read_text())
    workload_marker = tmp_path / "workload.json"
    fixture_python = tmp_path / "fixture-python"
    fixture_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, signal, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1:2] == ['-m']:\n"
        f"    assert sys.argv[2] == {variant + '.infer'!r}\n"
        f"    Path({str(workload_marker)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "    mode = os.environ['INFER_WORKER_TEST_MODE']\n"
        "    if mode == 'inner-kill':\n"
        "        os.kill(os.getpid(), signal.SIGKILL)\n"
        "    if mode == 'outer-kill':\n"
        "        os.kill(int(os.environ['INFER_WORKER_TEST_OUTER_PID']), signal.SIGKILL)\n"
        "    sys.exit(17 if mode == 'nonzero' else 0)\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    fixture_python.chmod(0o755)
    recipe["execution"].update(python=str(fixture_python), workdir=str(REPO_ROOT))
    recipe["execution"]["scheduler"]["direct_controller"] = direct_controller
    recipe_path.write_text(yaml.safe_dump(recipe))
    workspace = recipe_path.parent
    plan_dir = workspace / "plan"
    result = plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)
    assert result.exit_code == 0, [(issue.field, issue.message) for issue in result.blocking_issues()]
    plan = json.loads((plan_dir / "plan.json").read_text())
    (run,) = plan["runs"]
    monkeypatch.setattr(slurm, "controller_cluster", lambda *_a, **_k: "unit-cluster")
    monkeypatch.setattr(slurm, "submit", lambda *_a, **_k: slurm.parse_sbatch_output("3880\n"))
    experiments.launch_infer_run(plan_dir, dry_run=False)
    (queued,) = experiment_workspace.read_run_manifest(workspace)
    assert queued["status"] == "queued"
    assert queued["scheduler_cluster"] == "unit-cluster"
    manifest_before_worker = (workspace / "run_manifest.tsv").read_bytes()

    srun_marker = tmp_path / "srun.json"
    # Replace the test's denied srun executable, never a system executable.
    fixture_srun = tmp_path / "guarded-bin" / "srun"
    fixture_srun.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"Path({str(srun_marker)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "os.execv(sys.argv[-1], [sys.argv[-1]])\n"
    )
    fixture_srun.chmod(0o755)
    if mode == "runtime-drift":
        fixture_git = tmp_path / "guarded-bin" / "git"
        fixture_git.write_text("#!/bin/sh\nprintf '%040d\\n' 0\n")
        fixture_git.chmod(0o755)
    worker_text = Path(run["script"]).read_text()
    assert worker_text.splitlines().count(run["command"]) == 1
    assert "trap _agent_finish_run EXIT" not in worker_text
    batch_command = shlex.split(
        next(line for line in Path(run["scheduler_script"]).read_text().splitlines() if line.startswith("exec "))
    )
    worker_argv = batch_command[batch_command.index("run-frozen-job") :]
    digest_index = worker_argv.index("--execution-snapshot-sha256") + 1
    assert worker_argv[digest_index] == "${1:-}"
    worker_argv[digest_index] = queued["execution_snapshot_sha256"]
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "tests")]),
        "SLURM_JOB_ID": "3880",
        "SLURM_CLUSTER_NAME": "unit-cluster",
        "SLURM_NTASKS": "2" if mode == "topology" else "1",
        "INFER_WORKER_TEST_MODE": mode,
    }
    # Only target identity/argparse probes are fixture-backed. The generated worker,
    # allocation/runtime guards, srun child handling and sidecar writers execute normally.
    bootstrap = (
        "import os, sys; "
        "from agent_tool_test_helpers import run_execution_preflight_fixture; "
        "from agent_tools import managed_scheduler, slurm; "
        "managed_scheduler.run_execution_command = run_execution_preflight_fixture; "
        "os.environ['INFER_WORKER_TEST_OUTER_PID'] = str(os.getpid()); "
        "sys.exit(slurm._main(sys.argv[1:]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", bootstrap, *worker_argv],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    log_text = Path(run["log_path"]).read_text()
    assert completed.returncode == expected_exit, completed.stderr + log_text
    assert (workspace / "run_manifest.tsv").read_bytes() == manifest_before_worker
    if mode in {"topology", "runtime-drift"}:
        assert not workload_marker.exists()
    else:
        assert json.loads(workload_marker.read_text()) == shlex.split(run["command"])[1:]
    assert srun_marker.exists() == (mode != "topology")
    if srun_marker.exists():
        assert json.loads(srun_marker.read_text()) == [
            "--nodes=1",
            "--ntasks=1",
            "--ntasks-per-node=1",
            "--kill-on-bad-exit=1",
            "--quit-on-interrupt",
            "--label",
            run["script"],
        ]
    terminal_path = Path(run["scheduler_result_path"])
    if mode == "outer-kill":
        assert not terminal_path.exists()
    else:
        terminal = json.loads(terminal_path.read_text())
        assert terminal["exit_code"] == expected_exit
        assert slurm.sidecar_identity(terminal, run["scheduler_submit_token"]) == slurm.JobIdentity(
            "3880", "unit-cluster"
        )

    monkeypatch.setattr(slurm, "active_jobs", lambda *_a, **_k: [])
    if scheduler_state == "MISSING":
        monkeypatch.setattr(slurm, "show_job", lambda *_a, **_k: None)

        def accounting_disabled(*_args, **_kwargs):
            raise slurm.SlurmCommandError(
                "accounting lookup",
                subprocess.CompletedProcess([], 1, "", "Slurm accounting storage is disabled"),
            )

        monkeypatch.setattr(slurm, "accounting_job", accounting_disabled)
    else:
        monkeypatch.setattr(
            slurm,
            "show_job",
            lambda *_a, **_k: slurm.JobObservation("3880", scheduler_state, comment=run["scheduler_submit_token"]),
        )
    experiments.monitor_experiment(workspace)
    (canonical,) = experiment_workspace.read_run_manifest(workspace)
    assert canonical["status"] == expected_status
    assert canonical["scheduler_raw_state"] == scheduler_state
    assert experiments.experiment_status(workspace)["runs"][0]["status"] == expected_status
    monkeypatch.setattr(slurm, "submit", lambda *_a, **_k: pytest.fail("Repeated execute must not resubmit"))
    experiments.launch_infer_run(plan_dir, dry_run=False)
    assert experiment_workspace.read_run_manifest(workspace)[0]["status"] == expected_status
