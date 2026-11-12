import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from agent_tool_test_helpers import (
    FakeLauncher,
    checkout_commit,
    prepare_hparam_plan_fixture,
    run_cli,
    write_variant_hparam_recipe,
    write_yaml,
)
import pytest
import yaml

from agent_tools import hparam_runtime, hparam_selection, plan_contract, run_artifacts
from agent_tools.adapters.finetune import FINETUNE_ADAPTER
from agent_tools.decision_paths import finetune_init_checkpoint_issue
from agent_tools.experiment_workspace import merge_run_manifest, read_run_manifest, verify_run_snapshot


@pytest.mark.parametrize("scheduler_type", ["direct", "slurm"])
def test_hparam_init_checkpoint_is_hashed_and_not_used_for_resume(tmp_path, scheduler_type):
    checkpoint = tmp_path / "selected full.ckpt"
    checkpoint.write_bytes(b"selected full model")
    final_checkpoint = tmp_path / "final.ckpt"
    final_checkpoint.write_bytes(b"final evaluation checkpoint")
    recipe = write_variant_hparam_recipe(tmp_path, variant="sleep2vec2", ckpt_path=final_checkpoint)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["init_checkpoint_path"] = str(checkpoint)
    payload["execution"] = {
        "python": sys.executable,
        "runtime_commit": checkout_commit(),
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
    result = subprocess.run(shlex.split(verification), capture_output=True, text=True)
    assert result.returncode != 0
    assert "input snapshot hash changed" in result.stderr
    # Plan reads and launch passes do not hash inputs: launch.sh rejects the changed checkpoint when the run
    # starts, and job.sbatch (Slurm) or the launch command (direct) first re-hashes launch.sh itself.
    assert run_artifacts.read_hparam_plan(out) == plan
    launcher = FakeLauncher()
    hparam_runtime.launch_hparam_runs(out, dry_run=True, hooks=launcher.hooks())
    launcher.assert_not_started()
    if scheduler_type == "slurm":
        assert f"{run['script_sha256']}  {run['script']}" in Path(run["scheduler_script"]).read_text()
    else:
        hparam_runtime.launch_hparam_runs(out, dry_run=False, hooks=launcher.hooks())
        identity = json.loads(shlex.split(launcher.commands[0])[-1])[0]
        assert {"path": run["script"], "sha256": run["script_sha256"]} in json.loads(identity[5])


def test_hparam_stop_and_select_work_after_the_init_checkpoint_moves(tmp_path):
    checkpoint = tmp_path / "selected full.ckpt"
    checkpoint.write_bytes(b"selected full model")
    recipe = write_variant_hparam_recipe(
        tmp_path, variant="sleep2vec2", parameters={"runtime.lr": [1e-6, 2e-6]}, max_runs=2
    )
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["init_checkpoint_path"] = str(checkpoint)
    payload["execution"] = {"workdir": str(tmp_path), "python": sys.executable, "runtime_commit": checkout_commit()}
    write_yaml(recipe, payload)
    out = tmp_path / "plan"
    prepare_hparam_plan_fixture(recipe, out)
    plan = run_artifacts.read_hparam_plan(out)
    workspace = Path(plan["recipe"]["experiment"]["root"])
    trained, unlaunched = plan["runs"]
    assert trained["input_snapshots"][0]["path"] == str(checkpoint)
    trained_checkpoint = Path(trained["checkpoint_dir"]) / "epoch=1.ckpt"
    trained_checkpoint.parent.mkdir(parents=True)
    trained_checkpoint.write_text("trained")
    Path(trained["runtime_dir"], "run_manifest.json").write_text(
        json.dumps({"metrics": {"val_ahi_pearson": 0.9}, "best_model_path": str(trained_checkpoint), "epoch": 1})
    )
    merge_run_manifest(workspace, [{"step_id": trained["step_id"], "run_id": trained["run_id"], "status": "completed"}])
    # Training is over and the init checkpoint was archived; only launch.sh needs it, when a run starts.
    checkpoint.rename(tmp_path / "archived-init.ckpt")

    hparam_runtime.stop_hparam_run(out, unlaunched["run_id"], reason="drop the unlaunched candidate")
    ranking = hparam_selection.select_hparam_candidates(out)

    rows = {row["run_id"]: row for row in read_run_manifest(workspace)}
    assert rows[unlaunched["run_id"]]["status"] == "stopped"
    assert [row["run_id"] for row in csv.DictReader(ranking.read_text().splitlines())] == [trained["run_id"]]


@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2expert", "sex_age_baseline"])
def test_full_init_rejects_unsupported_variants(variant):
    issue = finetune_init_checkpoint_issue({"variant": variant, "inputs": {"init_checkpoint_path": "/full.ckpt"}})
    assert issue.status.value == "FAIL"
    assert "only for sleep2vec2" in issue.message


def test_full_init_rejects_pretrained_backbone_combination():
    recipe = {
        "variant": "sleep2vec2",
        "inputs": {"pretrained_backbone_path": "/backbone.ckpt", "init_checkpoint_path": "/full.ckpt"},
    }
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
    recipe = write_variant_hparam_recipe(tmp_path, variant="sleep2vec2")
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["init_checkpoint_path"] = str(tmp_path / "missing.ckpt")
    write_yaml(recipe, payload)
    result = run_cli("doctor", "--recipe", str(recipe))
    assert result.returncode == 1
    assert "init_checkpoint_path" in result.stdout
