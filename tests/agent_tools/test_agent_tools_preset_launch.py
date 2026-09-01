from __future__ import annotations

import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import pytest
from test_agent_preset_runtime_identity import (
    _PRESET_SCRIPTS,
    _runtime_recipe,
    preset_runtime as preset_runtime_fixture,
)

from agent_tools import cli, experiments, managed_scheduler, plans, run_evidence
from agent_tools.experiment_workspace import PROCESS_IDENTITY_FIELDS, merge_run_manifest, read_run_manifest
from agent_tools.models import REPO_ROOT


def _commit_runtime_python(preset_runtime):
    workdir = Path(preset_runtime["execution"]["workdir"])
    subprocess.run(["git", "add", "--force", "--", "agent_tools", *_PRESET_SCRIPTS.values()], cwd=workdir, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Preset runtime test",
            "-c",
            "user.email=preset-test@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--quiet",
            "-m",
            "Commit test runtime Python",
        ],
        cwd=workdir,
        check=True,
    )
    preset_runtime["execution"]["runtime_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workdir, text=True
    ).strip()


@pytest.fixture
def preset_runtime(tmp_path):
    runtime = preset_runtime_fixture.__wrapped__(tmp_path)
    package = Path(runtime["execution"]["workdir"]) / "agent_tools"
    package.unlink()
    shutil.copytree(REPO_ROOT / "agent_tools", package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _commit_runtime_python(runtime)
    return runtime


def _plan(tmp_path, preset_runtime, monkeypatch, variant="sleep2vec"):
    recipe = _runtime_recipe(tmp_path, preset_runtime, variant)
    plan_dir = preset_runtime["workspace"] / "plan"
    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)
    assert report.status == plans.DecisionStatus.PASS, [issue.message for issue in report.blocking_issues()]
    return plan_dir, json.loads((plan_dir / "plan.json").read_text())


@pytest.mark.parametrize("variant", _PRESET_SCRIPTS)
def test_new_preset_plan_freezes_managed_direct_launch(tmp_path, preset_runtime, monkeypatch, variant):
    plan_dir, plan = _plan(tmp_path, preset_runtime, monkeypatch, variant)

    assert plan["recipe"]["execution"]["scheduler"] == {"type": "direct"}
    assert plan["runs"][0]["terminal_status_owner"] == "script"
    assert plan["runs"][0]["scheduler_type"] == "direct"
    assert "preset-launch" in (plan_dir / "run.sh").read_text()
    assert plan["commands"][0] not in (plan_dir / "run.sh").read_text()
    assert shlex.split(plan["commands"][0])[:2] == [
        preset_runtime["execution"]["python"],
        _PRESET_SCRIPTS[variant],
    ]
    assert plan["commands"][0] in Path(plan["runs"][0]["script"]).read_text()


def test_preset_launch_preview_does_not_bind_or_spawn(tmp_path, preset_runtime, monkeypatch):
    plan_dir, plan = _plan(tmp_path, preset_runtime, monkeypatch)
    before = read_run_manifest(preset_runtime["workspace"])

    result = experiments.launch_preset_run(plan_dir)

    assert not result.started_keys
    assert result.launch_rows[0]["command"]
    assert read_run_manifest(preset_runtime["workspace"]) == before
    assert not preset_runtime["payload"].exists()
    assert not (Path(plan["runs"][0]["run_dir"]) / "pid").exists()
    assert not (Path(plan["runs"][0]["run_dir"]) / "stdout.log").exists()


def test_real_preset_consultation_can_generate_a_registered_launchable_plan(tmp_path, preset_runtime):
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"
    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)
    assert report.status == plans.DecisionStatus.PASS, [(issue.field, issue.message) for issue in report.issues]
    assert experiments.launch_preset_run(plan_dir).launch_rows[0]["command"]


def _wait_status(workspace, statuses):
    deadline = time.monotonic() + 10
    while True:
        row = read_run_manifest(workspace)[0]
        if row["status"] in statuses:
            return row
        assert time.monotonic() < deadline, row
        time.sleep(0.05)


@pytest.mark.parametrize("variant", _PRESET_SCRIPTS)
@pytest.mark.parametrize("exit_code", [0, 7])
def test_preset_detached_worker_commits_terminal_after_launcher_exits(
    tmp_path, preset_runtime, monkeypatch, variant, exit_code
):
    worker = Path(preset_runtime["execution"]["workdir"]) / _PRESET_SCRIPTS[variant]
    worker.write_text(
        "import os, time\n"
        "assert os.read(0, 1) == b''\n"
        "assert os.fstat(1).st_ino == os.fstat(2).st_ino\n"
        "time.sleep(0.2)\n"
        "os.write(1, b'delayed stdout\\n')\n"
        "os.write(2, b'delayed stderr\\n')\n"
        f"raise SystemExit({exit_code})\n"
    )
    _commit_runtime_python(preset_runtime)
    plan_dir, plan = _plan(tmp_path, preset_runtime, monkeypatch, variant)
    before = read_run_manifest(preset_runtime["workspace"])
    preview = subprocess.run(
        ["bash", str(plan_dir / "run.sh")], env=preset_runtime["env"], capture_output=True, text=True, timeout=10
    )
    assert preview.returncode == 0, preview.stderr
    assert "dry-run" in preview.stdout
    assert read_run_manifest(preset_runtime["workspace"]) == before
    launcher = subprocess.Popen(
        ["bash", str(plan_dir / "run.sh"), "--execute"],
        env=preset_runtime["env"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Close the transport's pipes; the worker must own only its persistent log and DEVNULL.
    launcher.stdout.close()
    launcher.stderr.close()
    launcher.wait(timeout=10)
    row = _wait_status(preset_runtime["workspace"], {"completed", "failed"})
    assert row["status"] == ("completed" if exit_code == 0 else "failed")
    assert row["terminal_status_owner"] == "script"
    assert row["pid"] == row["process_group_id"]
    assert all(row.get(field) for field in PROCESS_IDENTITY_FIELDS)
    assert "delayed stdout\ndelayed stderr\n" in Path(row["log_path"]).read_text()
    assert not preset_runtime["poison_marker"].exists()
    status = experiments.experiment_status(preset_runtime["workspace"])
    assert status["runs"][0]["status"] == row["status"]


@pytest.mark.parametrize("failure", ["missing_python", "runtime_preflight", "script_hash", "config_hash"])
def test_preset_launch_guard_failure_does_not_claim_attempt(tmp_path, preset_runtime, monkeypatch, failure):
    if failure == "missing_python":
        preset_runtime["execution"]["python"] = str(tmp_path / "missing-python")
    plan_dir, plan = _plan(tmp_path, preset_runtime, monkeypatch)
    if failure == "runtime_preflight":
        monkeypatch.setattr(
            managed_scheduler,
            "run_execution_command",
            lambda _execution, command: subprocess.CompletedProcess(command, 2, "", "Target runtime preflight failed"),
        )
    if failure.endswith("_hash"):
        Path(plan["runs"][0][failure.removesuffix("_hash")]).write_text("changed bytes\n")
    before = read_run_manifest(preset_runtime["workspace"])
    with pytest.raises((ValueError, RuntimeError)):
        experiments.launch_preset_run(plan_dir, dry_run=False)
    assert read_run_manifest(preset_runtime["workspace"]) == before
    assert not preset_runtime["payload"].exists()


def test_preset_runtime_preflight_claim_and_start_run_in_order(tmp_path, preset_runtime, monkeypatch):
    plan_dir, _plan_data = _plan(tmp_path, preset_runtime, monkeypatch)
    events = []

    def probe(_execution, command):
        events.append("probe")
        return subprocess.CompletedProcess(command, 0, "{}\n", "")

    original_merge = experiments.merge_run_manifest

    def merge(workspace, rows, **kwargs):
        if rows[0]["status"] == "launched" and "claim" not in events:
            events.append("claim")
        return original_merge(workspace, rows, **kwargs)

    def start(_execution, _command, *, retry_pre_spawn_failure):
        assert retry_pre_spawn_failure is False
        events.append("start")
        return "launched"

    monkeypatch.setattr(managed_scheduler, "run_execution_command", probe)
    monkeypatch.setattr(experiments, "merge_run_manifest", merge)
    monkeypatch.setattr(managed_scheduler, "start_process", start)

    result = experiments.launch_preset_run(plan_dir, dry_run=False)

    assert result.started_keys
    assert events == ["probe", "claim", "start"]


def test_preset_reverifies_artifacts_before_claim(tmp_path, preset_runtime, monkeypatch):
    plan_dir, plan = _plan(tmp_path, preset_runtime, monkeypatch)
    run = plan["runs"][0]
    artifact = Path(run["script"])
    before = read_run_manifest(preset_runtime["workspace"])
    original_verify = experiments.verify_run_snapshot
    calls = 0

    def verify(candidate):
        nonlocal calls
        calls += 1
        original_verify(candidate)
        if calls == 1:
            artifact.write_bytes(artifact.read_bytes() + b"\n# changed before claim\n")

    monkeypatch.setattr(experiments, "verify_run_snapshot", verify)

    with pytest.raises(ValueError, match="Run snapshot hash changed after planning"):
        experiments.launch_preset_run(plan_dir, dry_run=False)

    assert calls == 2
    assert read_run_manifest(preset_runtime["workspace"]) == before
    assert not preset_runtime["payload"].exists()


@pytest.mark.parametrize("mutation_phase", ["claim", "start"])
@pytest.mark.parametrize("artifact", ["script", "config"])
def test_preset_launch_rechecks_artifacts_in_the_actual_start_command(
    tmp_path, preset_runtime, monkeypatch, mutation_phase, artifact
):
    plan_dir, plan = _plan(tmp_path, preset_runtime, monkeypatch)
    run = plan["runs"][0]
    artifact_path = Path(run[artifact])
    original_bytes = artifact_path.read_bytes()
    for name, value in preset_runtime["env"].items():
        monkeypatch.setenv(name, value)
    original_merge = experiments.merge_run_manifest
    original_start = managed_scheduler.start_process
    start_commands = []

    def merge(workspace, rows, **kwargs):
        result = original_merge(workspace, rows, **kwargs)
        if mutation_phase == "claim" and rows[0]["status"] == "launched":
            artifact_path.write_bytes(original_bytes + b"\n# changed after launch claim\n")
        return result

    def start(execution, command, **kwargs):
        assert read_run_manifest(preset_runtime["workspace"])[0]["command"] == command
        start_commands.append(command)
        if mutation_phase == "start":
            artifact_path.write_bytes(original_bytes + b"\n# changed immediately before start\n")
        return original_start(execution, command, **kwargs)

    monkeypatch.setattr(experiments, "merge_run_manifest", merge)
    monkeypatch.setattr(managed_scheduler, "start_process", start)
    pid_path = Path(run["run_dir"]) / "pid"
    try:
        result = experiments.launch_preset_run(plan_dir, dry_run=False)

        assert result.launch_rows[0]["status"] == "launch_failed"
        assert not result.started_keys
        assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "launch_failed"
        assert not preset_runtime["payload"].exists()
        assert not pid_path.exists()
        assert not (Path(run["run_dir"]) / "stdout.log").exists()
        artifact_path.write_bytes(original_bytes)
        assert not experiments.launch_preset_run(plan_dir, dry_run=False).started_keys
        assert len(start_commands) == 1
    finally:
        identity = run_evidence.read_process_identity(pid_path)
        if identity and run_evidence.process_identity_running({}, identity):
            run_evidence.stop_process_group({}, identity)


def test_preset_launch_rechecks_clean_runtime_in_the_actual_start_command(tmp_path, preset_runtime, monkeypatch):
    plan_dir, plan = _plan(tmp_path, preset_runtime, monkeypatch)
    for name, value in preset_runtime["env"].items():
        monkeypatch.setenv(name, value)
    shadow = Path(preset_runtime["execution"]["workdir"]) / "shadow_module.py"
    original_merge = experiments.merge_run_manifest

    def merge(workspace, rows, **kwargs):
        committed = original_merge(workspace, rows, **kwargs)
        if rows[0]["status"] == "launched" and not shadow.exists():
            shadow.write_text("VALUE = 1\n")
        return committed

    monkeypatch.setattr(experiments, "merge_run_manifest", merge)

    result = experiments.launch_preset_run(plan_dir, dry_run=False)

    assert result.launch_rows[0]["status"] == "launch_failed"
    assert not result.started_keys
    assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "launch_failed"
    assert not preset_runtime["payload"].exists()
    assert not (Path(plan["runs"][0]["run_dir"]) / "pid").exists()


@pytest.mark.parametrize("execution_snapshot", [None, {"module": "fixture_runtime"}])
def test_verified_launch_rejects_empty_checkpoint_hash(tmp_path, execution_snapshot):
    with pytest.raises(ValueError, match="frozen checkpoint path and hash"):
        managed_scheduler.build_launch_command(
            {"python": "python", "workdir": str(tmp_path)},
            tmp_path / "launch.sh",
            tmp_path / "stdout.log",
            tmp_path / "pid",
            [],
            execution_snapshot=execution_snapshot,
            config_path=tmp_path / "config.yaml",
            script_sha256="a" * 64,
            config_sha256="b" * 64,
            checkpoint_path=tmp_path / "model.ckpt",
            checkpoint_sha256="",
        )


@pytest.mark.parametrize("lost_receipt", [False, True])
def test_preset_interrupted_attempt_is_not_relaunched(tmp_path, preset_runtime, monkeypatch, lost_receipt):
    plan_dir, _plan_data = _plan(tmp_path, preset_runtime, monkeypatch)
    for name, value in preset_runtime["env"].items():
        monkeypatch.setenv(name, value)
    original_start = managed_scheduler.start_process
    attempts = []

    def interrupt(execution, command, **kwargs):
        row = read_run_manifest(preset_runtime["workspace"])[0]
        assert row["status"] == "launched"
        assert row["command"] == command
        assert row["target"] == "local"
        attempts.append(command)
        if lost_receipt:
            original_start(execution, command, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(managed_scheduler, "start_process", interrupt)
    with pytest.raises(KeyboardInterrupt):
        experiments.launch_preset_run(plan_dir, dry_run=False)
    result = experiments.launch_preset_run(plan_dir, dry_run=False)
    assert not result.started_keys
    assert len(attempts) == 1
    if lost_receipt:
        _wait_status(preset_runtime["workspace"], {"completed"})
    else:
        assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "launched"
        row = read_run_manifest(preset_runtime["workspace"])[0]
        Path(row["log_path"]).write_text("success: completed\n")
        for _ in range(2):
            experiments.monitor_experiment(preset_runtime["workspace"])
            assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "missing_pid"
            assert not experiments.launch_preset_run(plan_dir, dry_run=False).started_keys
        assert len(attempts) == 1
        with pytest.raises(ValueError, match="no recorded process identity"):
            experiments.stop_preset_run(plan_dir, reason="no child receipt")


@pytest.mark.parametrize("after_commit", [False, True])
def test_preset_claim_write_failure_never_spawns(tmp_path, preset_runtime, monkeypatch, after_commit):
    plan_dir, _plan_data = _plan(tmp_path, preset_runtime, monkeypatch)
    original_merge = experiments.merge_run_manifest

    def fail_claim(*args, **kwargs):
        if after_commit:
            original_merge(*args, **kwargs)
        raise OSError("claim persistence failed")

    monkeypatch.setattr(experiments, "merge_run_manifest", fail_claim)
    monkeypatch.setattr(
        managed_scheduler, "start_process", lambda *_args, **_kwargs: pytest.fail("claim must finish first")
    )
    with pytest.raises(OSError, match="claim persistence failed"):
        experiments.launch_preset_run(plan_dir, dry_run=False)
    row = read_run_manifest(preset_runtime["workspace"])[0]
    assert row["status"] == ("launched" if after_commit else "planned")
    if after_commit:
        assert not experiments.launch_preset_run(plan_dir, dry_run=False).started_keys


def _launch_waiting_worker(tmp_path, preset_runtime, monkeypatch):
    worker = Path(preset_runtime["execution"]["workdir"]) / _PRESET_SCRIPTS["sleep2vec"]
    worker.write_text("import time\nprint('worker ready', flush=True)\ntime.sleep(30)\n")
    _commit_runtime_python(preset_runtime)
    plan_dir, _plan_data = _plan(tmp_path, preset_runtime, monkeypatch)
    experiments.launch_preset_run(plan_dir, dry_run=False)
    row = _wait_status(preset_runtime["workspace"], {"running"})
    return plan_dir, row


def test_preset_stop_allows_real_exit_trap_to_take_manifest_lock(tmp_path, preset_runtime, monkeypatch):
    plan_dir, row = _launch_waiting_worker(tmp_path, preset_runtime, monkeypatch)
    identity = {
        field: int(row[field]) if field != "process_start_token" else row[field] for field in PROCESS_IDENTITY_FIELDS
    }
    try:
        experiments.stop_preset_run(plan_dir, reason="local lifecycle regression")
        final = read_run_manifest(preset_runtime["workspace"])[0]
        assert final["status"] == "stopped"
        assert final["stop_requested_at"] and final["stopped_at"]
        assert final["stop_reason"] == "local lifecycle regression"
        assert run_evidence.process_identity_running(final, identity) is False
    finally:
        if run_evidence.process_identity_running(row, identity):
            run_evidence.stop_process_group(row, identity)


def test_preset_stop_planned_run_requires_reason_and_does_not_launch(tmp_path, preset_runtime, monkeypatch):
    plan_dir, _plan_data = _plan(tmp_path, preset_runtime, monkeypatch)
    before = read_run_manifest(preset_runtime["workspace"])
    with pytest.raises(ValueError, match="non-empty reason"):
        experiments.stop_preset_run(plan_dir, reason=" ")
    assert read_run_manifest(preset_runtime["workspace"]) == before
    experiments.stop_preset_run(plan_dir, reason="not needed")
    assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "stopped"
    assert not experiments.launch_preset_run(plan_dir, dry_run=False).started_keys
    assert not preset_runtime["payload"].exists()


def _recorded_attempt(tmp_path, preset_runtime, monkeypatch):
    plan_dir, _plan_data = _plan(tmp_path, preset_runtime, monkeypatch)
    monkeypatch.setattr(managed_scheduler, "start_process", lambda *_args, **_kwargs: "launched")
    experiments.launch_preset_run(plan_dir, dry_run=False)
    row = read_run_manifest(preset_runtime["workspace"])[0]
    identity = {"pid": 12345, "process_group_id": 12345, "process_start_token": "test-start-token"}
    Path(row["pid_path"]).write_text(json.dumps(identity))
    merge_run_manifest(preset_runtime["workspace"], [{**row, **identity}])
    return plan_dir


@pytest.mark.parametrize("exit_in_stop", [False, True])
def test_preset_stop_accepts_authenticated_exit_between_intent_and_signal(
    tmp_path, preset_runtime, monkeypatch, exit_in_stop
):
    plan_dir = _recorded_attempt(tmp_path, preset_runtime, monkeypatch)
    probes = iter([True, True, False] if exit_in_stop else [True, False])
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: next(probes))

    def signal(_row, _identity):
        assert exit_in_stop
        assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "stopping"
        raise RuntimeError("group exited before signal")

    monkeypatch.setattr(run_evidence, "stop_process_group", signal)
    experiments.stop_preset_run(plan_dir, reason="short task race")
    assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "stopped"


@pytest.mark.parametrize("failure", ["unknown", "reused"])
def test_preset_stop_unverified_identity_does_not_record_intent_or_signal(
    tmp_path, preset_runtime, monkeypatch, failure
):
    plan_dir = _recorded_attempt(tmp_path, preset_runtime, monkeypatch)
    before = read_run_manifest(preset_runtime["workspace"])

    def probe(*_args):
        if failure == "reused":
            raise run_evidence.ProcessIdentityError("PID was reused")
        return None

    monkeypatch.setattr(run_evidence, "process_identity_running", probe)
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_args: pytest.fail("must not signal"))
    with pytest.raises(RuntimeError):
        experiments.stop_preset_run(plan_dir, reason="unsafe probe")
    assert read_run_manifest(preset_runtime["workspace"]) == before


def test_preset_stop_uncertain_after_intent_can_be_explicitly_reconciled(tmp_path, preset_runtime, monkeypatch):
    plan_dir = _recorded_attempt(tmp_path, preset_runtime, monkeypatch)
    probes = iter([True, None])
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: next(probes))
    with pytest.raises(RuntimeError, match="after stop intent"):
        experiments.stop_preset_run(plan_dir, reason="original authorized reason")
    stopping = read_run_manifest(preset_runtime["workspace"])[0]
    assert stopping["status"] == "stopping"
    assert stopping["stop_requested_at"]
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_args: pytest.fail("already exited"))
    experiments.stop_preset_run(plan_dir, reason="retry observation")
    final = read_run_manifest(preset_runtime["workspace"])[0]
    assert final["status"] == "stopped"
    assert final["stop_reason"] == "original authorized reason"
    assert final["stop_requested_at"] == stopping["stop_requested_at"]


def test_preset_monitor_preserves_inflight_stop_intent(tmp_path, preset_runtime, monkeypatch):
    plan_dir = _recorded_attempt(tmp_path, preset_runtime, monkeypatch)
    probes = iter([True, None])
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: next(probes))
    with pytest.raises(RuntimeError, match="after stop intent"):
        experiments.stop_preset_run(plan_dir, reason="authorized stop")
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: True)
    for _ in range(2):
        experiments.monitor_experiment(preset_runtime["workspace"])
        assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "stopping"
    row = read_run_manifest(preset_runtime["workspace"])[0]
    merge_run_manifest(
        preset_runtime["workspace"],
        [{"step_id": row["step_id"], "run_id": row["run_id"], "status": "completed"}],
    )
    assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "stopping"
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)
    experiments.stop_preset_run(plan_dir, reason="repeat stop observation")
    assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "stopped"


def test_preset_status_recommends_explicit_managed_execute(tmp_path, preset_runtime, monkeypatch):
    plan_dir, _plan_data = _plan(tmp_path, preset_runtime, monkeypatch)
    action = experiments.experiment_status(preset_runtime["workspace"])["decision"]["recommended_next"]
    assert action["id"] == "preset-launch"
    assert action["argv"] == ["python", "-m", "agent_tools", "preset-launch", "--plan-dir", str(plan_dir), "--execute"]


@pytest.mark.parametrize("command", ["preset-launch", "preset-stop"])
def test_preset_cli_required_arguments_and_launch_default(command):
    parser = cli._build_parser()
    args = [command, "--plan-dir", "/tmp/plan"]
    if command == "preset-stop":
        with pytest.raises(SystemExit):
            parser.parse_args(args)
        args += ["--reason", "authorized stop"]
    parsed = parser.parse_args(args)
    assert parsed.plan_dir == "/tmp/plan"
    if command == "preset-launch":
        assert parsed.execute is False
