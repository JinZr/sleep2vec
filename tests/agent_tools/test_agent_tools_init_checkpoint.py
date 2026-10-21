import hashlib
from pathlib import Path
import shlex
import subprocess
import sys

from agent_tool_test_helpers import FakeLauncher, prepare_hparam_plan_fixture, run_cli, write_yaml
import pytest
from test_agent_plan_blocks_on_ambiguity import _RUNTIME_COMMIT, _hparam_recipe
import yaml

from agent_tools import hparam_runtime, plan_contract, run_artifacts, slurm
from agent_tools.adapters.finetune import FINETUNE_ADAPTER
from agent_tools.decision_paths import finetune_init_checkpoint_issue
from agent_tools.experiment_workspace import read_run_manifest, verify_run_snapshot


@pytest.mark.parametrize("scheduler_type", ["direct", "slurm"])
def test_hparam_init_checkpoint_is_hashed_and_not_used_for_resume(tmp_path, monkeypatch, scheduler_type):
    checkpoint = tmp_path / "selected full.ckpt"
    checkpoint.write_bytes(b"selected full model")
    final_checkpoint = tmp_path / "final.ckpt"
    final_checkpoint.write_bytes(b"final evaluation checkpoint")
    recipe = _hparam_recipe(tmp_path, variant="sleep2vec2", ckpt_path=final_checkpoint)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["init_checkpoint_path"] = str(checkpoint)
    payload["execution"] = {
        "python": sys.executable,
        "runtime_commit": _RUNTIME_COMMIT,
        "scheduler": {"type": scheduler_type},
    }
    if scheduler_type == "slurm":
        payload["execution"]["gpus_per_run"] = 1
        payload["execution"]["scheduler"].update(
            {"partition": "gpu", "cpus_per_task": 8, "memory": "64G", "walltime": "01:00:00"}
        )
    write_yaml(recipe, payload)
    out = tmp_path / "plan"
    prepare_hparam_plan_fixture(recipe, out)
    plan = run_artifacts.read_hparam_plan(out)
    run = plan["runs"][0]
    argv = shlex.split(run["command"])
    assert argv[argv.index("--init-checkpoint-path") + 1] == str(checkpoint)
    assert "--ckpt-path" not in argv
    assert "--no-test-after-fit" in argv
    snapshot = {
        "field": "inputs.init_checkpoint_path",
        "path": str(checkpoint),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    assert run["input_snapshots"] == [snapshot]
    assert plan_contract.frozen_input_snapshot(plan["recipe"], snapshot["field"]) == snapshot
    rows = read_run_manifest(Path(plan["recipe"]["experiment"]["root"]))
    assert rows[0]["input_snapshots"] == str([snapshot])
    if scheduler_type == "slurm":
        assert run["scheduler_type"] == "slurm"
        assert run["script"] in Path(run["scheduler_script"]).read_text()
    script = Path(run["script"]).read_text()
    verification = next(line for line in script.splitlines() if "verify_run_snapshot" in line)
    subprocess.run(shlex.split(verification), check=True)
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="input snapshot hash changed"):
        verify_run_snapshot(run)
    with pytest.raises(ValueError, match="input snapshot hash changed"):
        run_artifacts.read_hparam_plan(out)
    result = subprocess.run(shlex.split(verification), capture_output=True, text=True)
    assert result.returncode != 0
    assert "input snapshot hash changed" in result.stderr
    monkeypatch.setattr(slurm, "submit", lambda *_args, **_kwargs: pytest.fail("Changed input reached Slurm submit"))
    launcher = FakeLauncher()
    for dry_run in (True, False):
        with pytest.raises(ValueError, match="input snapshot hash changed"):
            hparam_runtime.launch_hparam_runs(out, dry_run=dry_run, hooks=launcher.hooks())
    launcher.assert_not_started()
    assert not (out / "launch_manifest.tsv").exists()


def test_adaptive_init_inherits_full_model_checkpoint(tmp_path):
    checkpoint = tmp_path / "selected full.ckpt"
    checkpoint.write_bytes(b"selected full model")
    recipe = _hparam_recipe(tmp_path, variant="sleep2vec2")
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["init_checkpoint_path"] = str(checkpoint)
    payload["adaptive"] = {
        "enabled": True,
        "objective_metric": "val_ahi_pearson",
        "objective_mode": "max",
        "test_feedback_for_selection": False,
        "max_rounds": 2,
        "max_runs_total": 2,
        "round_size": 1,
        "suggest": {"strategy": "best_neighborhood"},
        "replacement": {"enabled": False},
    }
    write_yaml(recipe, payload)
    workflow = tmp_path / "workflow"
    result = run_cli("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow))
    assert result.returncode == 0, result.stdout + result.stderr
    round_dir = workflow / "adaptive" / "rounds" / "round_000"
    plan = run_artifacts.read_hparam_plan(round_dir)
    run = plan["runs"][0]
    snapshot = {
        "field": "inputs.init_checkpoint_path",
        "path": str(checkpoint),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    assert plan["recipe"]["inputs"]["init_checkpoint_path"] == str(checkpoint)
    assert plan_contract.frozen_input_snapshot(plan["recipe"], snapshot["field"]) == snapshot
    assert run["input_snapshots"] == [snapshot]
    argv = shlex.split(run["command"])
    assert argv[argv.index("--init-checkpoint-path") + 1] == str(checkpoint)
    assert "--ckpt-path" not in argv
    rows = read_run_manifest(Path(plan["recipe"]["experiment"]["root"]))
    assert rows[0]["input_snapshots"] == str([snapshot])
    assert not (round_dir / "launch_manifest.tsv").exists()
    assert not Path(run["runtime_dir"]).exists()


@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2expert", "sex_age_baseline"])
def test_full_init_rejects_unsupported_variants(variant):
    issue = finetune_init_checkpoint_issue({"variant": variant, "inputs": {"init_checkpoint_path": "/full.ckpt"}})
    assert issue.status.value == "FAIL"
    assert "only for sleep2vec2" in issue.message


@pytest.mark.parametrize(
    "extra",
    [
        {"inputs": {"pretrained_backbone_path": "/backbone.ckpt"}},
        {"execution": {"target": "ssh"}},
        {"search": {"profile": "finetune_balanced"}},
    ],
)
def test_full_init_rejects_conflicting_or_unsupported_workflows(extra):
    recipe = {"variant": "sleep2vec2", **extra}
    recipe["inputs"] = {**extra.get("inputs", {}), "init_checkpoint_path": "/full.ckpt"}
    assert finetune_init_checkpoint_issue(recipe).status.value == "FAIL"


def test_full_init_rejects_direct_finetune_resume_but_allows_hparam_final_checkpoint():
    recipe = {
        "task": "finetune",
        "variant": "sleep2vec2",
        "inputs": {"init_checkpoint_path": "/init.ckpt", "ckpt_path": "/resume.ckpt"},
    }
    issues = FINETUNE_ADAPTER.preflight_issues(recipe, None, unlock_final_test=False)
    assert issues[0].status.value == "FAIL"
    recipe["task"] = "hparam_tune"
    assert finetune_init_checkpoint_issue(recipe) is None


def test_doctor_rejects_missing_init_file(tmp_path):
    recipe = _hparam_recipe(tmp_path, variant="sleep2vec2")
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["init_checkpoint_path"] = str(tmp_path / "missing.ckpt")
    write_yaml(recipe, payload)
    result = run_cli("doctor", "--recipe", str(recipe))
    assert result.returncode == 1
    assert "init_checkpoint_path" in result.stdout
