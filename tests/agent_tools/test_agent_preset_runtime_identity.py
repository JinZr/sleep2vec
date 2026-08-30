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

from agent_tools import decisions, experiments, plans
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
    assert plan["recipe"]["execution"] == preset_runtime["execution"]
    assert yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())["execution"] == preset_runtime["execution"]
    command = plan["commands"][0]
    assert shlex.split(command)[:2] == [preset_runtime["execution"]["python"], _PRESET_SCRIPTS[variant]]
    assert plan["runs"][0]["command"] == command
    assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "planned"
    result = subprocess.run(
        ["bash", str(plan_dir / "run.sh")], cwd=tmp_path, env=preset_runtime["env"], text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(preset_runtime["payload"].read_text())
    assert payload["python"] == sys.executable
    assert payload["cwd"] == preset_runtime["execution"]["workdir"]
    assert payload["argv"] == shlex.split(command)[1:]
    assert payload["status"] == "running"
    assert preset_runtime["calls"].read_text().splitlines() == ["-c", "-c", _PRESET_SCRIPTS[variant], "-c"]
    assert not preset_runtime["poison_marker"].exists()
    assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "completed"


@pytest.mark.parametrize("failure", ["wrong_commit", "missing_python"])
def test_preset_runtime_identity_failure_precedes_lifecycle_and_payload(tmp_path: Path, preset_runtime, failure):
    if failure == "wrong_commit":
        preset_runtime["execution"]["runtime_commit"] = "0" * 40
    else:
        preset_runtime["execution"]["python"] = str(tmp_path / "missing-python")
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"
    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)
    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    before = _workspace_files(preset_runtime["workspace"])

    result = subprocess.run(
        ["bash", str(plan_dir / "run.sh")], cwd=tmp_path, env=preset_runtime["env"], text=True, capture_output=True
    )

    assert result.returncode != 0
    if failure == "wrong_commit":
        assert "Target runtime commit differs from the frozen plan" in result.stderr
    else:
        assert "missing-python" in result.stderr
    assert not preset_runtime["payload"].exists()
    assert not preset_runtime["poison_marker"].exists()
    assert read_run_manifest(preset_runtime["workspace"])[0]["status"] == "planned"
    assert _workspace_files(preset_runtime["workspace"]) == before


def test_preset_explicit_executable_name_is_preserved(tmp_path: Path, preset_runtime):
    preset_runtime["execution"]["python"] = "python"
    recipe = _runtime_recipe(tmp_path, preset_runtime)
    plan_dir = preset_runtime["workspace"] / "plan"

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0, [issue.message for issue in report.blocking_issues()]
    plan = json.loads((plan_dir / "plan.json").read_text())
    assert plan["recipe"]["execution"]["python"] == "python"
    assert shlex.split(plan["commands"][0])[0] == "python"


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
        ("runtime_commit", "A" * 40),
        ("runtime_commit", []),
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
    # Preset rejects artifacts, but the unrelated generic output warning requires them for PASS.
    monkeypatch.setattr(decisions, "_output_paths_missing", lambda _recipe: False)
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
