from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe
import pytest
from test_agent_plan_blocks_on_ambiguity import _write_preset_recipe
from test_agent_tools_experiment_status import _workspace_files
import yaml

from agent_tools import cli, experiments, plans
from agent_tools.adapters import preset_prepare as preset_adapter
from agent_tools.experiment_workspace import file_sha256, read_run_manifest
from agent_tools.manifests import write_rows
from agent_tools.models import REPO_ROOT

_PRESET_SCRIPTS = {
    "sleep2vec": "preprocess/save_dataset_presets.py",
    "sleep2vec2": "sleep2vec2/preprocess/save_dataset_presets.py",
    "sleep2expert": "sleep2expert/preprocess/save_dataset_presets.py",
}


@pytest.fixture
def preset_runtime(tmp_path: Path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "agent_tools").symlink_to(REPO_ROOT / "agent_tools", target_is_directory=True)
    for script in _PRESET_SCRIPTS.values():
        entrypoint = runtime / script
        entrypoint.parent.mkdir(parents=True, exist_ok=True)
        entrypoint.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "from agent_tools.experiment_workspace import read_run_manifest\n"
            "payload = {'python': sys.executable, 'cwd': str(Path.cwd()), 'argv': sys.argv,\n"
            "           'status': read_run_manifest(os.environ['PRESET_TEST_WORKSPACE'])[0]['status']}\n"
            "Path(os.environ['PRESET_TEST_PAYLOAD']).write_text(json.dumps(payload))\n"
        )
    subprocess.run(["git", "init", "--quiet", str(runtime)], check=True)
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
            "--allow-empty",
            "-m",
            "Initialize test runtime",
        ],
        cwd=runtime,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runtime, text=True).strip()
    calls = tmp_path / "python-calls.txt"
    runtime_python = runtime / "python-runtime"
    runtime_python.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$1\" >> {shlex.quote(str(calls))}\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    runtime_python.chmod(0o755)
    poison = tmp_path / "poison"
    poison.mkdir()
    poison_marker = tmp_path / "poison-ran.txt"
    poison_python = poison / "python"
    poison_python.write_text(f"#!/bin/sh\nprintf poison > {shlex.quote(str(poison_marker))}\nexit 97\n")
    poison_python.chmod(0o755)
    workspace = tmp_path / "workspace"
    payload = tmp_path / "payload.json"
    return {
        "execution": {
            "target": "local",
            "workdir": str(runtime),
            "python": str(runtime_python),
            "runtime_commit": commit,
        },
        "workspace": workspace,
        "payload": payload,
        "calls": calls,
        "poison_marker": poison_marker,
        "env": {
            **os.environ,
            "PATH": str(poison) + os.pathsep + os.environ["PATH"],
            "PRESET_TEST_WORKSPACE": str(workspace),
            "PRESET_TEST_PAYLOAD": str(payload),
        },
    }


def _runtime_recipe(tmp_path: Path, preset_runtime: dict, variant: str = "sleep2vec") -> Path:
    source = tmp_path / "source"
    base = write_finetune_recipe(source, variant=variant)
    config = yaml.safe_load(base.read_text())["inputs"]["config"]
    recipe = _write_preset_recipe(
        source,
        config=config,
        index=source / "index.csv",
        variant=variant,
        execution=preset_runtime["execution"],
    )
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(preset_runtime["workspace"])
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    return recipe


@pytest.mark.parametrize("variant", _PRESET_SCRIPTS)
def test_preset_explicit_identity_runs_real_generated_command_despite_path_change(
    tmp_path: Path, preset_runtime, variant
):
    recipe = _runtime_recipe(tmp_path, preset_runtime, variant)
    plan_dir = preset_runtime["workspace"] / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    plan = json.loads((plan_dir / "plan.json").read_text())
    expected_execution = {**preset_runtime["execution"], "scheduler": {"type": "direct"}}
    assert plan["recipe"]["execution"] == expected_execution
    assert yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())["execution"] == expected_execution
    command = plan["commands"][0]
    assert shlex.split(command)[:2] == [preset_runtime["execution"]["python"], _PRESET_SCRIPTS[variant]]
    assert plan["runs"][0]["command"] == command
    assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "planned"
    result = subprocess.run(
        ["bash", plan["runs"][0]["script"]], cwd=tmp_path, env=preset_runtime["env"], text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(preset_runtime["payload"].read_text())
    assert payload["python"] == sys.executable
    assert payload["cwd"] == preset_runtime["execution"]["workdir"]
    assert payload["argv"] == shlex.split(command)[1:]
    assert payload["status"] == "running"
    assert preset_runtime["calls"].read_text().splitlines() == ["-c", _PRESET_SCRIPTS[variant], "-c"]
    assert not preset_runtime["poison_marker"].exists()
    assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "completed"


def test_preset_missing_python_precedes_lifecycle_and_payload(tmp_path: Path, preset_runtime):
    preset_runtime["execution"]["python"] = str(tmp_path / "missing-python")
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"
    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)
    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    plan = json.loads((plan_dir / "plan.json").read_text())
    before = _workspace_files(preset_runtime["workspace"])

    result = subprocess.run(
        ["bash", plan["runs"][0]["script"]], cwd=tmp_path, env=preset_runtime["env"], text=True, capture_output=True
    )

    assert result.returncode != 0
    assert "missing-python" in result.stderr
    assert not preset_runtime["payload"].exists()
    assert not preset_runtime["poison_marker"].exists()
    assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "planned"
    assert _workspace_files(preset_runtime["workspace"]) == before


def test_preset_runtime_commit_mismatch_runs_and_records_both_commits(tmp_path: Path, preset_runtime):
    preset_runtime["execution"]["runtime_commit"] = "0" * 40
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"
    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)
    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    plan = json.loads((plan_dir / "plan.json").read_text())
    actual_commit = subprocess.check_output(
        ["git", "-C", preset_runtime["execution"]["workdir"], "rev-parse", "HEAD"], text=True
    ).strip()

    result = subprocess.run(
        ["bash", plan["runs"][0]["script"]],
        cwd=tmp_path,
        env=preset_runtime["env"],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert preset_runtime["payload"].exists()
    row = read_run_manifest(preset_runtime["workspace"])[0]
    assert row["status"] == "completed"
    assert row["planned_runtime_commit"] == "0" * 40
    assert row["runtime_commit"] == actual_commit


def test_preset_explicit_executable_name_is_preserved(tmp_path: Path, preset_runtime):
    preset_runtime["execution"]["python"] = "python"
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    plan = json.loads((plan_dir / "plan.json").read_text())
    assert plan["recipe"]["execution"]["python"] == "python"
    assert shlex.split(plan["commands"][0])[0] == "python"


def test_preset_authored_runtime_commit_is_frozen_lowercase(tmp_path: Path, preset_runtime):
    runtime_commit = "A" * 40
    preset_runtime["execution"]["runtime_commit"] = runtime_commit
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    expected = runtime_commit.lower()
    assert yaml.safe_load(recipe.read_text())["execution"]["runtime_commit"] == runtime_commit
    assert json.loads((plan_dir / "plan.json").read_text())["recipe"]["execution"]["runtime_commit"] == expected
    assert yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())["execution"]["runtime_commit"] == expected


@pytest.mark.parametrize("missing_field", ["python", "runtime_commit", "workdir"])
def test_preset_partial_identity_fails_before_workspace_creation(tmp_path: Path, preset_runtime, missing_field):
    preset_runtime["execution"].pop(missing_field)
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    issue = next(issue for issue in report.blocking_issues() if issue.field == f"execution.{missing_field}")
    assert issue.evidence["preflight_before_workspace"] is True
    assert not preset_runtime["workspace"].exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python", "ASK_USER"),
        ("python", []),
        ("python", "conda run -n exp python"),
        ("python", "~/bin/python"),
        ("runtime_commit", ""),
        ("runtime_commit", "ASK_USER"),
        ("runtime_commit", []),
        ("runtime_commit", "g" * 40),
        ("runtime_commit", "a" * 39),
        ("runtime_commit", "a" * 41),
        ("runtime_commit", "a" * 64),
        ("workdir", "relative/runtime"),
        ("workdir", []),
        ("target", "ssh"),
    ],
)
def test_preset_invalid_identity_fails_before_workspace_creation(tmp_path: Path, preset_runtime, field, value):
    preset_runtime["execution"][field] = value
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    issue = next(issue for issue in report.blocking_issues() if issue.field == f"execution.{field}")
    assert issue.evidence["preflight_before_workspace"] is True
    assert not preset_runtime["workspace"].exists()


def test_preset_coherent_command_and_script_hash_tamper_is_rejected(tmp_path: Path, preset_runtime, monkeypatch):
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"
    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)
    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    assert experiments.experiment_status(preset_runtime["workspace"])["summary"]["state"] == "ready_to_launch"
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    old_command = plan["commands"][0]
    changed_command = shlex.join(["python", *shlex.split(old_command)[1:]])
    run = plan["runs"][0]
    run["command"] = changed_command
    plan["commands"] = [changed_command]
    for script in (plan_dir / "run.sh", Path(run["script"])):
        script.write_text(script.read_text().replace(old_command, changed_command, 1))
    run["script_sha256"] = file_sha256(Path(run["script"]))
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = read_run_manifest(preset_runtime["workspace"])
    rows[0]["script_sha256"] = run["script_sha256"]
    write_rows(preset_runtime["workspace"] / "run_manifest.tsv", rows)
    before = _workspace_files(preset_runtime["workspace"])

    with pytest.raises(ValueError, match="commands differ from its frozen recipe"):
        experiments.experiment_status(preset_runtime["workspace"])

    assert _workspace_files(preset_runtime["workspace"]) == before


@pytest.mark.parametrize("variant", _PRESET_SCRIPTS)
def test_preset_manager_defaults_freeze_actual_python_commit_and_workdir(tmp_path: Path, preset_runtime, variant):
    preset_runtime["execution"] = {}
    recipe = _runtime_recipe(tmp_path, preset_runtime, variant)
    plan_dir = preset_runtime["workspace"] / "plan"
    expected = {
        "python": sys.executable,
        "runtime_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "workdir": str(REPO_ROOT),
        "scheduler": {"type": "direct"},
    }

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    plan = json.loads((plan_dir / "plan.json").read_text())
    assert plan["recipe"]["execution"] == expected
    assert yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())["execution"] == expected
    assert shlex.split(plan["commands"][0])[:2] == [sys.executable, _PRESET_SCRIPTS[variant]]
    script = Path(plan["runs"][0]["script"]).read_text()
    assert f"cd {shlex.quote(str(REPO_ROOT))}\n" in script
    assert f"{shlex.quote(sys.executable)} -c " in script
    assert expected["runtime_commit"] in script


@pytest.mark.parametrize("variant", _PRESET_SCRIPTS)
def test_preset_default_generated_command_ignores_path_python_in_simulated_manager_repo(
    tmp_path: Path, preset_runtime, monkeypatch, variant
):
    manager_root = Path(preset_runtime["execution"]["workdir"])
    manager_commit = preset_runtime["execution"]["runtime_commit"]
    # Only manager-location discovery is simulated; the generated command and lifecycle run unchanged.
    monkeypatch.setattr(preset_adapter, "REPO_ROOT", manager_root)
    monkeypatch.setattr(preset_adapter, "repo_summary", lambda: {"git": {"available": True, "commit": manager_commit}})
    preset_runtime["execution"] = {}
    recipe = _runtime_recipe(tmp_path, preset_runtime, variant)
    plan_dir = preset_runtime["workspace"] / "plan"
    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)
    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    plan = json.loads((plan_dir / "plan.json").read_text())
    assert plan["recipe"]["execution"] == {
        "python": sys.executable,
        "runtime_commit": manager_commit,
        "workdir": str(manager_root),
        "scheduler": {"type": "direct"},
    }
    command = plan["commands"][0]
    assert shlex.split(command)[:2] == [sys.executable, _PRESET_SCRIPTS[variant]]

    result = subprocess.run(
        ["bash", plan["runs"][0]["script"]], cwd=tmp_path, env=preset_runtime["env"], text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(preset_runtime["payload"].read_text())
    assert payload == {
        "python": sys.executable,
        "cwd": str(manager_root),
        "argv": shlex.split(command)[1:],
        "status": "running",
    }
    assert not preset_runtime["poison_marker"].exists()
    assert not preset_runtime["calls"].exists()
    assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "completed"


def test_preset_default_identity_is_bound_without_preset_build(tmp_path: Path, preset_runtime):
    preset_runtime["execution"] = {}
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    payload = yaml.safe_load(recipe.read_text())
    config_path = Path(payload["inputs"]["config"])
    config = yaml.safe_load(config_path.read_text())
    config.pop("preset_build")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    payload["preset"].update({"channels": ["ppg", "ahi", "stage5"], "min_channels": 3})
    for field in ("required_channels", "min_channels"):
        payload["decisions"][field]["source"] = "explicit_recipe"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = preset_runtime["workspace"] / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    plan = json.loads((plan_dir / "plan.json").read_text())
    assert plan["recipe"]["execution"]["python"] == sys.executable
    assert plan["recipe"]["execution"]["workdir"] == str(REPO_ROOT)
    assert shlex.split(plan["commands"][0])[0] == sys.executable


@pytest.mark.parametrize("target", [None, "", "local"])
def test_preset_manager_defaults_preserve_authored_execution_metadata(tmp_path: Path, preset_runtime, target):
    metadata = {
        "target": target,
        "workdir": str(REPO_ROOT),
        "path_context": "local",
        "path_validation": "local",
        "host": "metadata-only",
    }
    preset_runtime["execution"] = dict(metadata)
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    execution = json.loads((plan_dir / "plan.json").read_text())["recipe"]["execution"]
    assert {field: execution[field] for field in metadata} == metadata
    assert execution["python"] == sys.executable
    assert execution["runtime_commit"]


def test_preset_explicit_identity_preserves_remote_path_context_and_host(tmp_path: Path, preset_runtime):
    preset_runtime["execution"].update({"path_context": "remote", "path_validation": "defer", "host": "metadata-only"})
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    plan = json.loads((plan_dir / "plan.json").read_text())
    assert plan["recipe"]["execution"] == {**preset_runtime["execution"], "scheduler": {"type": "direct"}}


@pytest.mark.parametrize("context", ["other_workdir", "remote_paths", "local_target_remote_paths", "ssh_target"])
def test_preset_nonmanager_context_requires_explicit_identity_before_workspace_creation(
    tmp_path: Path, preset_runtime, context
):
    execution = {
        "other_workdir": {"workdir": preset_runtime["execution"]["workdir"]},
        "remote_paths": {"path_context": "remote", "path_validation": "defer"},
        "local_target_remote_paths": {"target": "local", "path_context": "remote", "path_validation": "defer"},
        "ssh_target": {"target": "ssh", "host": "metadata-only", "path_validation": "defer"},
    }[context]
    preset_runtime["execution"] = execution
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    issue = next(issue for issue in report.blocking_issues() if issue.field == "execution")
    assert issue.evidence["preflight_before_workspace"] is True
    assert issue.status == plans.DecisionStatus.FAIL
    if context == "ssh_target":
        assert "local" in issue.message.lower()
        assert "ssh" in issue.message.lower()
    else:
        assert "execution.python" in issue.message
        assert "execution.runtime_commit" in issue.message
        assert "execution.workdir" in issue.message
    assert not preset_runtime["workspace"].exists()


@pytest.mark.parametrize("git_state", [{"available": False, "commit": "a" * 40}, {"available": True, "commit": ""}])
def test_preset_missing_manager_commit_fails_before_workspace_creation(
    tmp_path: Path, preset_runtime, monkeypatch, git_state
):
    monkeypatch.setattr(preset_adapter, "repo_summary", lambda: {"git": git_state})
    preset_runtime["execution"] = {}
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    issue = next(issue for issue in report.blocking_issues() if issue.field == "execution.runtime_commit")
    assert issue.evidence["preflight_before_workspace"] is True
    assert not preset_runtime["workspace"].exists()


def test_preset_auto_bound_runtime_commit_is_frozen_lowercase(tmp_path: Path, preset_runtime, monkeypatch):
    runtime_commit = "A" * 40
    preset_runtime["execution"] = {}
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    monkeypatch.setattr(
        preset_adapter,
        "repo_summary",
        lambda: {"git": {"available": True, "commit": runtime_commit}},
    )
    plan_dir = preset_runtime["workspace"] / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    expected = runtime_commit.lower()
    assert json.loads((plan_dir / "plan.json").read_text())["recipe"]["execution"]["runtime_commit"] == expected
    assert yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())["execution"]["runtime_commit"] == expected


@pytest.mark.parametrize("entrypoint", ["evaluate_recipe", "build_plan"])
@pytest.mark.parametrize("has_preset_build", [True, False])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python", "/tmp/runtime with space/bin/python"),
        ("python", ""),
        ("python", "python --isolated"),
        ("runtime_commit", "ASK_USER"),
        ("runtime_commit", []),
        ("runtime_commit", "g" * 40),
        ("runtime_commit", "a" * 39),
        ("runtime_commit", "a" * 41),
        ("runtime_commit", "a" * 64),
    ],
)
def test_preset_invalid_auto_bound_identity_fails_before_workspace_creation(
    tmp_path: Path, preset_runtime, monkeypatch, capsys, entrypoint, has_preset_build, field, value
):
    preset_runtime["execution"] = {}
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    if not has_preset_build:
        payload = yaml.safe_load(recipe.read_text())
        config_path = Path(payload["inputs"]["config"])
        config = yaml.safe_load(config_path.read_text())
        config.pop("preset_build")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        payload["preset"].update({"channels": ["ppg", "ahi", "stage5"], "min_channels": 3})
        for decision_field in ("required_channels", "min_channels"):
            payload["decisions"][decision_field]["source"] = "explicit_recipe"
        recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    if field == "python":
        monkeypatch.setattr(sys, "executable", value)
    else:
        monkeypatch.setattr(preset_adapter, "repo_summary", lambda: {"git": {"available": True, "commit": value}})
    plan_dir = preset_runtime["workspace"] / "plan"

    if entrypoint == "evaluate_recipe":
        _recipe, _config, report = plans.evaluate_recipe(recipe)
    else:
        report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.status == plans.DecisionStatus.FAIL
    issue = next(issue for issue in report.blocking_issues() if issue.field == f"execution.{field}")
    assert issue.evidence["preflight_before_workspace"] is True
    assert not preset_runtime["workspace"].exists()
    assert not plan_dir.exists()
    if entrypoint == "evaluate_recipe":
        assert cli.main(["doctor", "--recipe", str(recipe)]) == 1
        output = capsys.readouterr().out
        assert f"execution.{field}" in output
        assert "FAIL" in output
        assert not preset_runtime["workspace"].exists()


@pytest.mark.parametrize("historical", [False, "runtime_identity", "legacy"])
def test_preset_registered_reader_never_rebinds_default_or_historical_identity(
    tmp_path: Path, preset_runtime, monkeypatch, historical
):
    preset_runtime["execution"] = {}
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"
    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)
    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    if historical:
        original_command = plan["commands"][0]
        if historical == "legacy":
            plan["recipe"]["execution"] = {}
        else:
            plan["recipe"]["execution"].pop("scheduler")
        run = plan["runs"][0]
        for field in ("scheduler_type", "terminal_status_owner"):
            run.pop(field)
        contract = preset_adapter.PRESET_PREPARE_ADAPTER.compile_plan_contract(
            plan["recipe"], plan_dir, run_index_offset=0, config_bytes=Path(run["config"]).read_bytes()
        )
        assert shlex.split(contract["commands"][0])[0] == ("python" if historical == "legacy" else sys.executable)
        plan["commands"] = contract["commands"]
        run["command"] = contract["commands"][0]
        plan_markdown = plan_dir / "plan.md"
        plan_markdown.write_text(plan_markdown.read_text().replace(original_command, run["command"]))
        for script in (plan_dir / "run.sh", Path(run["script"])):
            script.write_text(contract["script_text"])
        run["script_sha256"] = file_sha256(Path(run["script"]))
        (Path(run["run_dir"]) / "run.json").write_text(json.dumps({**run, "commands": plan["commands"]}))
        resolved = {key: value for key, value in plan["recipe"].items() if key != "_recipe_path"}
        (plan_dir / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        rows = read_run_manifest(preset_runtime["workspace"])
        rows[0]["script_sha256"] = run["script_sha256"]
        for field in ("scheduler_type", "terminal_status_owner"):
            rows[0].pop(field)
        write_rows(preset_runtime["workspace"] / "run_manifest.tsv", rows)
    before = _workspace_files(preset_runtime["workspace"])

    def unexpected_binding(*_args, **_kwargs):
        raise AssertionError("Registered readers must not bind current runtime defaults.")

    monkeypatch.setattr(preset_adapter.PRESET_PREPARE_ADAPTER, "bind_effective_recipe", unexpected_binding)
    monkeypatch.setattr(preset_adapter, "repo_summary", unexpected_binding)
    monkeypatch.setattr(preset_adapter, "REPO_ROOT", tmp_path / "reader-repo")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "reader-python"))

    status = experiments.experiment_status(preset_runtime["workspace"])

    assert status["summary"]["state"] == "ready_to_launch"
    if historical:
        assert "preset-launch" not in (plan_dir / "run.sh").read_text()
        for operation in (experiments.launch_preset_run, experiments.stop_preset_run):
            kwargs = {"dry_run": False} if operation is experiments.launch_preset_run else {"reason": "old plan"}
            with pytest.raises(ValueError, match="new local managed direct preset"):
                operation(plan_dir, **kwargs)
    assert _workspace_files(preset_runtime["workspace"]) == before
