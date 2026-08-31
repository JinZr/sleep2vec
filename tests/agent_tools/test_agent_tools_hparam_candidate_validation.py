from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

from agent_tool_test_helpers import config_payload
import pytest
import yaml

from agent_tools import (
    hparam_runtime,
    managed_scheduler,
    plan_contract,
    plan_hparam,
    plans,
    run_artifacts,
    run_evidence,
    slurm,
)
from agent_tools.experiment_workspace import managed_run_key, merge_run_row
from tests.agent_tools.test_agent_tools_hparam_preflight import _recipe, _snapshot, _workspace_files


@pytest.mark.parametrize("search_kind", ["configurations", "parameters"])
@pytest.mark.parametrize("include_invalid", [False, True])
def test_candidate_budget_validates_only_reachable_joint_configs(
    tmp_path: Path, monkeypatch, search_kind: str, include_invalid: bool
):
    recipe_path, workspace = _recipe(tmp_path)
    recipe = yaml.safe_load(recipe_path.read_text())
    if search_kind == "configurations":
        search = {
            "configurations": [
                {"yaml:/model/cls/downstream": "tokens", "yaml:/model/cls/embedding_type": "bert"},
                {"yaml:/model/cls/downstream": "cls", "yaml:/model/cls/embedding_type": "none"},
            ],
            "max_runs": 2 if include_invalid else 1,
        }
    else:
        search = {
            "parameters": {
                "yaml:/model/cls/downstream": ["tokens", "cls"],
                "yaml:/model/cls/embedding_type": ["bert", "none"],
            },
            "max_runs": 4 if include_invalid else 3,
        }
    recipe["search"] = {"method": "grid", **search}
    recipe_path.write_text(yaml.safe_dump(recipe))
    target_calls = []

    def inspect(execution, runs, **_kwargs):
        target_calls.append(runs)
        return _snapshot(execution, runs)

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", inspect)
    plan_dir = workspace / "plans" / "candidates"
    workspace_before = _workspace_files(workspace)

    report = plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)

    if include_invalid:
        assert report.exit_code == 1
        issue = next(issue for issue in report.issues if issue.field == "hparam_search_space")
        assert "model.cls.embedding_type must be set" in issue.message
        assert issue.evidence["preflight_before_workspace"] is True
        assert target_calls == []
        assert _workspace_files(workspace) == workspace_before
        assert not plan_dir.exists()
    else:
        assert report.exit_code == 0
        plan = json.loads((plan_dir / "plan.json").read_text())
        assert len(plan["runs"]) == search["max_runs"]
        assert target_calls


def test_consultation_deduplicates_exact_final_bytes_per_call_and_uses_capture(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.yaml"
    payload = config_payload(tmp_path / "index.csv")
    captured = yaml.safe_dump(payload, sort_keys=False).encode()
    source.write_bytes(captured)
    recipe = {
        "variant": "sleep2vec",
        "inputs": {"config": str(source), "label_name": "ahi"},
        "experiment": {"id": "unit-candidates"},
        "step": {"id": "tune"},
        "execution": {"python": sys.executable, "runtime_commit": "a" * 40, "workdir": str(tmp_path)},
        "evaluation_policy": {"test_after_fit": False, "selection_split": "val"},
        "search": {
            "configurations": [
                {"runtime.lr": 1e-6},
                {"runtime.lr": 2e-6},
                {"runtime.lr": 3e-6, "yaml:/model/cls/downstream": "cls"},
                {"runtime.lr": 4e-6},
            ]
        },
    }
    plan_contract.bind_plan_context(recipe)
    plan_contract.bind_frozen_input_snapshot(recipe, "inputs.config", source, hashlib.sha256(captured).hexdigest())
    source.write_text("model: {}\n")
    validate = plan_hparam.validate_finetune_config_bytes
    calls = []

    def tracked_validate(candidate_recipe, config_bytes):
        calls.append(config_bytes)
        validate(candidate_recipe, config_bytes)

    monkeypatch.setattr(plan_hparam, "validate_finetune_config_bytes", tracked_validate)
    expected = yaml.safe_dump(payload).encode()
    changed = copy.deepcopy(payload)
    changed["model"]["cls"]["downstream"] = "cls"
    expected_changed = yaml.safe_dump(changed).encode()

    for _ in range(2):
        assert plan_hparam.hparam_yaml_override_issues(recipe, config_bytes=captured) == []
    assert calls == [expected, expected_changed] * 2

    contracts = plan_hparam.compile_hparam_run_contracts(recipe, tmp_path / "plan", 0, source_config_bytes=captured)
    assert [contract["config_bytes"] for contract in contracts] == [expected, expected, expected_changed, expected]
    assert len({contract["row"]["command"] for contract in contracts}) == 4
    assert all("runtime" not in yaml.safe_load(contract["config_bytes"]) for contract in contracts)
    assert calls == [expected, expected_changed] * 2  # Artifact compilation is an observer-safe operation.


def test_duplicate_candidates_keep_scientific_contract_checks(tmp_path: Path):
    payload = config_payload(tmp_path / "index.csv")
    recipe = {
        "variant": "sleep2vec",
        "inputs": {"config": "captured.yaml", "data_backend": "npz"},
        "evaluation_policy": {"selection_metric": "val_ahi_pearson", "selection_mode": "max"},
        "search": {"parameters": {"runtime.lr": [1e-6, 2e-6]}},
    }

    # The comparison itself must run for each combo even when the YAML is identical.
    class TrackedBackend(str):
        comparisons = 0

        def __ne__(self, other):
            self.comparisons += 1
            return super().__ne__(other)

    backend = TrackedBackend("npz")
    recipe["inputs"]["data_backend"] = backend
    issues = plan_hparam.hparam_yaml_override_issues(recipe, config_bytes=yaml.safe_dump(payload).encode())

    assert backend.comparisons == 2
    assert issues == []


def test_frozen_config_helper_deduplicates_only_within_call_and_names_invalid_run(tmp_path: Path, monkeypatch):
    config_bytes = yaml.safe_dump(config_payload(tmp_path / "index.csv")).encode()
    distinct_bytes = config_bytes + b"# same config with distinct frozen bytes\n"
    recipe = {"variant": "sleep2vec", "inputs": {"config": str(tmp_path / "never-read.yaml")}}
    runs = [
        ({"run_id": f"run-{index:03d}"}, content)
        for index, content in enumerate([config_bytes, config_bytes, distinct_bytes])
    ]
    before = copy.deepcopy(recipe)
    validate = plan_hparam.validate_finetune_config_bytes
    calls = []

    def tracked_validate(candidate_recipe, content):
        calls.append(content)
        validate(candidate_recipe, content)

    monkeypatch.setattr(plan_hparam, "validate_finetune_config_bytes", tracked_validate)
    for _ in range(2):
        plan_hparam.validate_hparam_run_configs(recipe, runs)
    assert calls == [config_bytes, distinct_bytes] * 2
    assert recipe == before

    with pytest.raises(ValueError, match="run-042.*variant=sleep2vec"):
        plan_hparam.validate_hparam_run_configs(recipe, [*runs, ({"run_id": "run-042"}, b"{}\n")])


def _frozen_plan(tmp_path: Path, *, invalid: bool = False) -> tuple[Path, dict]:
    plan_dir = tmp_path / "workspace" / "plan"
    plan_dir.mkdir(parents=True)
    payload = config_payload(tmp_path / "index.csv")
    if invalid:
        payload["model"]["cls"].update({"downstream": "cls", "embedding_type": "none"})
    config = plan_dir / "config.yaml"
    config.write_text(yaml.safe_dump(payload))
    runs = [
        {"step_id": "tune", "run_id": f"run-{index:03d}", "config": str(config), "status": status}
        for index, status in enumerate(["planned", "pending", "running", "finished"])
    ]
    runs[1].update({"target": "local", "command": "frozen pending command"})
    return plan_dir, {
        "recipe": {
            "variant": "sleep2vec",
            "inputs": {"config": str(tmp_path / "authored-must-not-be-read.yaml")},
            "experiment": {"root": str(plan_dir.parent)},
            "execution": {"target": "local"},
        },
        "runs": runs,
        "execution_snapshot": {},
    }


@pytest.mark.parametrize("boundary", ["preflight", "commit"])
def test_invalid_frozen_config_blocks_registration_before_target_or_workspace_writes(
    tmp_path: Path, monkeypatch, boundary: str
):
    plan_dir, plan = _frozen_plan(tmp_path, invalid=True)
    monkeypatch.setattr(run_artifacts, "read_hparam_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(plan_hparam, "_hparam_registration_state", lambda _plan: (plan_dir.parent, []))

    def ensure_workspace(*_args, **kwargs):
        assert kwargs.get("validate_only"), "Invalid config reached workspace mutation"

    monkeypatch.setattr(plan_hparam, "ensure_experiment_workspace", ensure_workspace)
    monkeypatch.setattr(
        plan_hparam, "_inspect_hparam_execution_target", lambda *_args: pytest.fail("Invalid config reached target")
    )
    monkeypatch.setattr(
        plan_hparam, "merge_run_manifest", lambda *_args: pytest.fail("Invalid config reached registration")
    )
    before = _workspace_files(plan_dir.parent)

    with pytest.raises(ValueError, match="run-000.*model.cls.embedding_type must be set"):
        if boundary == "preflight":
            plan_hparam.preflight_hparam_plan(plan_dir, semantic_out=plan_dir)
        else:
            plan_hparam.commit_hparam_plan(plan_dir)

    assert _workspace_files(plan_dir.parent) == before
    assert not (plan_dir / "plan.md").exists()


def test_registration_validates_same_frozen_bytes_as_card_without_commit_reload(tmp_path: Path, monkeypatch):
    recipe_path, workspace = _recipe(tmp_path)
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe["search"] = {"parameters": {"runtime.lr": [1e-6, 2e-6]}, "max_runs": 2, "method": "grid"}
    recipe_path.write_text(yaml.safe_dump(recipe))
    validate = plan_hparam.validate_hparam_run_configs
    render = plan_hparam.render_hparam_preflight_card
    checked_lists = []
    target_calls = []

    def tracked_validate(candidate_recipe, run_configs):
        checked_lists.append(run_configs)
        return validate(candidate_recipe, run_configs)

    def tracked_render(candidate_recipe, snapshot, run_configs):
        assert run_configs is checked_lists[-1]
        return render(candidate_recipe, snapshot, run_configs)

    def inspect(execution, runs, **_kwargs):
        assert checked_lists
        assert len(runs) == 2
        assert len({run["command"] for run in runs}) == 2
        target_calls.append(runs)
        return _snapshot(execution, runs)

    monkeypatch.setattr(plan_hparam, "validate_hparam_run_configs", tracked_validate)
    monkeypatch.setattr(plan_hparam, "render_hparam_preflight_card", tracked_render)
    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", inspect)
    plan_dir = workspace / "plans" / "tune"

    report = plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)

    assert report.exit_code == 0
    assert len(checked_lists) == 1
    assert target_calls
    plan = run_artifacts.read_hparam_plan(plan_dir)
    assert checked_lists[0] == [(run, Path(run["config"]).read_bytes()) for run in plan["runs"]]


@pytest.mark.parametrize("update_leaf_hash", [False, True])
def test_preflight_validated_commit_still_rejects_frozen_config_drift(
    tmp_path: Path, monkeypatch, update_leaf_hash: bool
):
    recipe_path, workspace = _recipe(tmp_path)
    monkeypatch.setattr(
        managed_scheduler, "inspect_execution_target", lambda execution, runs, **_kwargs: _snapshot(execution, runs)
    )
    commit = plan_hparam.commit_hparam_plan

    def tamper(out, **kwargs):
        assert kwargs["preflight_validated"] is True
        plan_path = Path(out) / "plan.json"
        plan = json.loads(plan_path.read_text())
        run = plan["runs"][0]
        config = Path(run["config"])
        config.write_bytes(config.read_bytes() + b"# changed after registration preflight\n")
        if update_leaf_hash:
            run["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
            plan_path.write_text(json.dumps(plan))
        return commit(out, **kwargs)

    monkeypatch.setattr(plan_hparam, "commit_hparam_plan", tamper)
    before = _workspace_files(workspace)
    plan_dir = workspace / "plans" / "tune"

    report = plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)

    assert report.exit_code == 1
    message = report.blocking_issues()[0].message.lower()
    if update_leaf_hash:
        assert "canonical expected runs field config_sha256" in message, message
    else:
        assert "snapshot hash changed" in message, message
    assert _workspace_files(workspace) == before
    assert not plan_dir.exists()


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize(
    "target,scheduler_type", [("local", "direct"), ("ssh", "direct"), ("local", "slurm"), ("ssh", "slurm")]
)
def test_invalid_frozen_config_blocks_every_submission_route_without_mutation(
    tmp_path: Path, monkeypatch, dry_run: bool, target: str, scheduler_type: str
):
    plan_dir, plan = _frozen_plan(tmp_path, invalid=True)
    plan["recipe"]["execution"].update({"target": target, "scheduler": {"type": scheduler_type}})
    if target == "ssh":
        plan["recipe"]["execution"]["host"] = "unit-host"
    monkeypatch.setattr(run_artifacts, "read_hparam_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(hparam_runtime, "read_run_manifest", lambda _root: plan["runs"])
    monkeypatch.setattr(
        managed_scheduler,
        "launch_managed_runs",
        lambda *_args, **_kwargs: pytest.fail("Invalid config reached scheduler"),
    )
    before = _workspace_files(plan_dir.parent)

    with pytest.raises(ValueError, match="run-000.*model.cls.embedding_type must be set"):
        hparam_runtime._launch_hparam_runs(
            plan_dir, dry_run=dry_run, manifest_lock_held=True, fail_on_missing_pid_blocker=False
        )

    assert _workspace_files(plan_dir.parent) == before


@pytest.mark.parametrize("dry_run", [False, True])
def test_launch_validates_prospective_frozen_bytes_including_pending_command(
    tmp_path: Path, monkeypatch, dry_run: bool
):
    plan_dir, plan = _frozen_plan(tmp_path)
    monkeypatch.setattr(run_artifacts, "read_hparam_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(hparam_runtime, "read_run_manifest", lambda _root: plan["runs"])
    validate = plan_hparam.validate_hparam_run_configs
    checked = []
    paths_checked = []

    def tracked_validate(recipe, run_configs):
        checked.append(run_configs)
        validate(recipe, run_configs)

    def launch(*_args, **_kwargs):
        assert checked

    monkeypatch.setattr(plan_hparam, "validate_hparam_run_configs", tracked_validate)
    monkeypatch.setattr(
        plan_hparam, "validate_hparam_output_paths", lambda *_args, **kwargs: paths_checked.append(kwargs["runs"])
    )
    monkeypatch.setattr(managed_scheduler, "launch_managed_runs", launch)
    before = _workspace_files(plan_dir.parent)
    for _ in range(2):
        hparam_runtime._launch_hparam_runs(
            plan_dir, dry_run=dry_run, manifest_lock_held=True, fail_on_missing_pid_blocker=False
        )

    expected = [(run, Path(run["config"]).read_bytes()) for run in plan["runs"][:2]]
    assert checked == [expected, expected]
    assert paths_checked == ([] if dry_run else [plan["runs"][:2], plan["runs"][:2]])
    assert _workspace_files(plan_dir.parent) == before


def test_active_and_terminal_launch_and_monitor_do_not_validate_configs(tmp_path: Path, monkeypatch):
    plan_dir, plan = _frozen_plan(tmp_path, invalid=True)
    plan["runs"] = plan["runs"][2:]
    monkeypatch.setattr(run_artifacts, "read_hparam_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(hparam_runtime, "read_run_manifest", lambda _root: plan["runs"])
    monkeypatch.setattr(
        plan_hparam, "validate_hparam_run_configs", lambda *_args: pytest.fail("Observer path validated configs")
    )
    monkeypatch.setattr(managed_scheduler, "launch_managed_runs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", lambda _root, rows: rows)
    monkeypatch.setattr(hparam_runtime, "write_status_report", lambda *_args: None)
    for dry_run in (False, True):
        hparam_runtime._launch_hparam_runs(
            plan_dir, dry_run=dry_run, manifest_lock_held=True, fail_on_missing_pid_blocker=False
        )
    assert hparam_runtime.monitor_hparam_runs(plan_dir) == plan_dir / "run_status.tsv"


def test_direct_observation_cannot_add_launchable_runs_and_removes_running_pending(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(run_evidence, "read_pid", lambda *_args: 42)
    monkeypatch.setattr(run_evidence, "process_running", lambda *_args: True)
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda *_args: None)
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args: "")
    rows = [
        {"step_id": "tune", "run_id": f"run-{index:03d}", "target": "local", "status": status}
        for index, status in enumerate(["pending", "running", "unknown_remote", "finished"])
    ]
    by_key = {managed_run_key(row): row for row in rows}
    observed = managed_scheduler.observe_runs(tmp_path, by_key, by_key, dry_run=False)

    assert observed.rows_by_key[managed_run_key(rows[0])]["status"] == "running"
    assert all(row["status"] not in managed_scheduler.LAUNCHABLE_STATUSES for row in observed.rows_by_key.values())
    for previous in rows[1:]:
        for status in managed_scheduler.LAUNCHABLE_STATUSES:
            assert merge_run_row(previous, {"status": status})["status"] == previous["status"]


def test_slurm_observation_and_merge_cannot_add_launchable_runs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(managed_scheduler, "_read_slurm_json", lambda *_args: {})
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda *_args: None)
    monkeypatch.setattr(
        slurm,
        "active_jobs",
        lambda *_args, **_kwargs: [slurm.JobObservation("42", "PENDING", comment="unit-token")],
    )
    for status in ("queued", "running", "unknown_scheduler", "completed"):
        prior = {
            "step_id": "tune",
            "run_id": "run-000",
            "target": "local",
            "status": status,
            "scheduler_type": "slurm",
            "scheduler_job_id": "42",
            "scheduler_cluster": "unit",
            "scheduler_submit_token": "unit-token",
            "scheduler_direct_controller": "false",
            "scheduler_result_path": str(tmp_path / "terminal.json"),
            "allocation_identity_path": str(tmp_path / "allocation.json"),
        }
        observed = managed_scheduler.observe_slurm_run(tmp_path, {"target": "local"}, prior)
        assert observed["status"] == "queued"
        assert merge_run_row(prior, observed)["status"] not in managed_scheduler.LAUNCHABLE_STATUSES
        for regression in managed_scheduler.LAUNCHABLE_STATUSES:
            assert merge_run_row(prior, {"status": regression})["status"] == status
