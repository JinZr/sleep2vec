from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from agent_tools import adaptive_hparam, manifests, plans
from agent_tools.experiment_workspace import merge_run_manifest
from tests.agent_tools import adaptive_hparam_test_support as test_support
from tests.agent_tools.adaptive_hparam_test_support import _agent_recipe

_stub_execution_snapshot_preflight = test_support._stub_execution_snapshot_preflight


def _joint_recipe(tmp_path: Path) -> tuple[Path, Path]:
    recipe = _agent_recipe(tmp_path / "source")
    payload = yaml.safe_load(recipe.read_text())
    workspace = tmp_path / "workspace"
    payload["experiment"] = yaml.safe_load(Path(payload["base_recipe"]).read_text())["experiment"]
    payload["experiment"]["root"] = str(workspace)
    payload["search"] = {
        "method": "grid",
        "max_runs": 2,
        "parameters": {
            "runtime.lr": [5e-7, 2e-6],
            "runtime.epochs": [2, 4],
            "runtime.precision": ["32", "16-mixed"],
        },
        "configurations": [
            {"runtime.lr": 5e-7, "runtime.epochs": 2, "runtime.precision": "32"},
            {"runtime.lr": 2e-6, "runtime.epochs": 4, "runtime.precision": "32"},
        ],
    }
    payload["adaptive"]["round_size"] = 2
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    return recipe, workspace


def test_initial_points_preserve_order_domain_and_initialization_recovery(tmp_path: Path):
    recipe, workspace = _joint_recipe(tmp_path)
    authored = yaml.safe_load(recipe.read_text())

    workflow = adaptive_hparam.init_adaptive_workflow(recipe, workspace / "workflow")

    round_dir = workflow / "adaptive" / "rounds" / "round_000"
    plan = json.loads((round_dir / "plan.json").read_text())
    assert plan["recipe"]["search"] == authored["search"]
    actual_points = [{key: run[key] for key in authored["search"]["parameters"]} for run in plan["runs"]]
    assert actual_points == authored["search"]["configurations"]
    frozen_paths = [round_dir / "plan.json", round_dir / "round_recipe.yaml", workspace / "events.jsonl"]
    before = {path: path.read_bytes() for path in frozen_paths}

    assert adaptive_hparam.init_adaptive_workflow(recipe, workflow) == workflow
    assert {path: path.read_bytes() for path in frozen_paths} == before


@pytest.mark.parametrize("proposal_shape", ["parameters", "configurations"])
def test_next_proposal_can_use_unsampled_domain_after_exact_initial_points(
    tmp_path: Path, monkeypatch, proposal_shape: str
):
    recipe, workspace = _joint_recipe(tmp_path)
    source = yaml.safe_load(recipe.read_text())
    source["decisions"]["hparam_search_space"] = {
        "value": source["search"]["parameters"],
        "source": "explicit_recipe",
    }
    source["decisions"]["hparam_budget"] = {"value": 2, "source": "explicit_recipe"}
    recipe.write_text(yaml.safe_dump(source, sort_keys=False))
    source_bytes = recipe.read_bytes()
    workflow = adaptive_hparam.init_adaptive_workflow(recipe, workspace / "workflow")
    initial_plan = json.loads((workflow / "adaptive" / "rounds" / "round_000" / "plan.json").read_text())
    for run in initial_plan["runs"]:
        checkpoint_dir = Path(run["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True)
        checkpoint = checkpoint_dir / "best-epoch=1.ckpt"
        checkpoint.write_text("fixture checkpoint")
        (checkpoint_dir / "epoch=1.ckpt").write_text("fixture checkpoint")
        (Path(run["runtime_dir"]) / "run_manifest.json").write_text(
            json.dumps(
                {
                    "version": run["version"],
                    "monitor": "val_ahi_pearson",
                    "monitor_mode": "max",
                    "best_model_score": 0.5,
                    "best_model_path": str(checkpoint),
                    "epoch": 1,
                    "status": "finished",
                    "metrics": {"test_auroc": 0.73},
                }
            )
        )
    merge_run_manifest(
        workspace,
        [
            {"step_id": run["step_id"], "run_id": run["run_id"], "status": "finished"}
            for run in initial_plan["runs"]
        ],
    )
    monkeypatch.setattr(adaptive_hparam, "monitor_hparam_runs", lambda _run_dir: None)

    input_path = adaptive_hparam.adaptive_step(workflow)

    assert input_path is not None
    document = json.loads(input_path.read_text())
    envelopes = document["input"]["parameter_envelopes"]
    assert envelopes["runtime.precision"]["choices"] == ["32", "16-mixed"]
    assert envelopes["runtime.epochs"] == {"kind": "integer", "min": 2, "max": 4}
    assert document["input"]["remaining_budget"]["runs"] == 2
    proposal_path = Path(document["expected_proposal_path"])
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    point = {"runtime.lr": 1e-6, "runtime.epochs": 3, "runtime.precision": "16-mixed"}
    proposal_search = (
        {"configurations": [point]}
        if proposal_shape == "configurations"
        else {"parameters": {key: [value] for key, value in point.items()}}
    )
    proposal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "request_id": document["request_id"],
                "target_round": 1,
                **proposal_search,
                "evidence_run_ids": [initial_plan["runs"][0]["run_id"]],
                "rationale": "Test an intermediate training length and the untried precision choice.",
            }
        )
    )
    assert adaptive_hparam.adaptive_step(workflow, proposal_path=proposal_path) == proposal_path

    def fake_launch(run_dir, *, dry_run=True):
        runs = json.loads((Path(run_dir) / "plan.json").read_text())["runs"]
        launch_manifest = Path(run_dir) / "launch_manifest.tsv"
        manifests.write_rows(launch_manifest, [{**run, "status": "launched"} for run in runs])
        merge_run_manifest(
            workspace,
            [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "launched"} for run in runs],
        )
        return launch_manifest

    monkeypatch.setattr(adaptive_hparam, "launch_hparam_runs", fake_launch)
    adaptive_hparam.adaptive_step(workflow, proposal_path=proposal_path, execute=True)
    next_plan = json.loads((workflow / "adaptive" / "rounds" / "round_001" / "plan.json").read_text())
    assert len(next_plan["runs"]) == 1
    assert {key: next_plan["runs"][0][key] for key in point} == point
    assert next_plan["recipe"]["search"] == {"method": "grid", "max_runs": 1, **proposal_search}
    assert "hparam_search_space" not in next_plan["recipe"]["decisions"]
    assert "hparam_budget" not in next_plan["recipe"]["decisions"]
    assert recipe.read_bytes() == source_bytes
    assert initial_plan["recipe"]["decisions"]["hparam_search_space"] == source["decisions"]["hparam_search_space"]
    assert initial_plan["recipe"]["decisions"]["hparam_budget"] == source["decisions"]["hparam_budget"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "keys must exactly match"),
        ("unknown", "keys must exactly match"),
        ("outside", "must be within"),
        ("duplicate", "duplicate configuration points"),
        ("search_budget", "exceeds search.max_runs"),
        ("round_budget", "exceeds adaptive.round_size"),
        ("total_budget", "exceeds adaptive.max_runs_total"),
    ],
)
def test_invalid_initial_points_fail_before_workspace_publication(tmp_path: Path, change: str, message: str):
    recipe, workspace = _joint_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    points = payload["search"]["configurations"]
    if change == "missing":
        points[0].pop("runtime.epochs")
    elif change == "unknown":
        points[0]["runtime.batch_size"] = 8
    elif change == "outside":
        points[0]["runtime.lr"] = 3e-6
    elif change == "duplicate":
        points[1] = copy.deepcopy(points[0])
    elif change == "search_budget":
        payload["search"]["max_runs"] = 1
    else:
        payload["adaptive"]["round_size" if change == "round_budget" else "max_runs_total"] = 1
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(adaptive_hparam.AdaptivePreflightError, match=message):
        adaptive_hparam.init_adaptive_workflow(recipe, workspace / "workflow")

    assert not workspace.exists()


@pytest.mark.parametrize("mode", ["static", "disabled", "best_neighborhood"])
def test_combined_initial_search_remains_exclusive_to_agent_proposals(tmp_path: Path, mode: str):
    recipe, workspace = _joint_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    if mode == "static":
        payload.pop("adaptive")
    elif mode == "disabled":
        payload["adaptive"]["enabled"] = False
    else:
        payload["adaptive"]["suggest"] = {"strategy": "best_neighborhood"}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, _config, report = plans.evaluate_recipe(recipe)

    assert report.exit_code == 1
    assert any("mutually exclusive" in issue.message for issue in report.blocking_issues())
    assert not workspace.exists()


@pytest.mark.parametrize("outside", [False, True])
def test_domain_decision_preserves_initial_points_and_revalidates_them(tmp_path: Path, outside: bool):
    recipe, workspace = _joint_recipe(tmp_path)
    source = yaml.safe_load(recipe.read_text())
    parameters = copy.deepcopy(source["search"]["parameters"])
    if outside:
        parameters["runtime.epochs"] = [3, 4]
    decisions = tmp_path / "decisions.yaml"
    decisions.write_text(yaml.safe_dump({"decisions": {"hparam_search_space": {"value": parameters}}}))

    effective, _config, report = plans.evaluate_recipe(recipe, decisions)

    assert effective["search"]["configurations"] == source["search"]["configurations"]
    assert effective["search"]["parameters"] == parameters
    if outside:
        assert report.exit_code == 1
        assert any("must be within [3, 4]" in issue.message for issue in report.blocking_issues())
    else:
        assert report.exit_code == 0, report.blocking_issues()
    assert not workspace.exists()


@pytest.mark.parametrize("change", ["order", "value"])
def test_initial_point_drift_is_rejected_before_next_proposal(tmp_path: Path, change: str):
    recipe, workspace = _joint_recipe(tmp_path)
    workflow = adaptive_hparam.init_adaptive_workflow(recipe, workspace / "workflow")
    payload = yaml.safe_load(recipe.read_text())
    if change == "order":
        payload["search"]["configurations"].reverse()
    else:
        payload["search"]["configurations"][0]["runtime.epochs"] = 3
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = (workspace / "events.jsonl").read_bytes()

    with pytest.raises(ValueError, match=r"frozen round 000: search.configurations"):
        adaptive_hparam.adaptive_step(workflow)

    assert (workspace / "events.jsonl").read_bytes() == before
    assert not (workflow / "adaptive" / "proposal_inputs").exists()


def test_test_selection_checks_checkpoint_cadence_of_initial_points(tmp_path: Path):
    recipe, workspace = _joint_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].update({"selection_split": "test", "selection_metric": "test_auroc"})
    payload["decisions"]["train_val_test_policy"]["value"] = "test"
    payload["search"]["parameters"]["runtime.ckpt_every_n_epochs"] = [1, 2]
    for index, point in enumerate(payload["search"]["configurations"]):
        point["runtime.ckpt_every_n_epochs"] = index + 1
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, _config, report = plans.evaluate_recipe(recipe)

    assert report.exit_code == 1
    issues = [issue for issue in report.blocking_issues() if issue.field == "runtime.ckpt_every_n_epochs"]
    assert any(issue.evidence.get("effective_values") == [1, 2] for issue in issues)
    assert not workspace.exists()


def test_unused_structured_choice_cannot_drift_from_boolean_to_integer(tmp_path: Path):
    recipe, workspace = _joint_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    tuning_key = "yaml:/finetune/tuning"
    full = {"preset": "full", "groups": {"encoder": {"train": True}}}
    frozen = {"preset": "full", "groups": {"encoder": {"train": False}}}
    payload["search"]["parameters"][tuning_key] = [full, frozen]
    for point in payload["search"]["configurations"]:
        point[tuning_key] = copy.deepcopy(full)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    workflow = adaptive_hparam.init_adaptive_workflow(recipe, workspace / "workflow")
    payload["search"]["parameters"][tuning_key][1]["groups"]["encoder"]["train"] = 0
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = (workspace / "events.jsonl").read_bytes()

    with pytest.raises(ValueError, match="Frozen adaptive round recipe differs from the requested initialization"):
        adaptive_hparam.init_adaptive_workflow(recipe, workflow)

    with pytest.raises(ValueError, match=r"frozen round 000: search.parameters"):
        adaptive_hparam.adaptive_step(workflow)

    assert (workspace / "events.jsonl").read_bytes() == before
    assert not (workflow / "adaptive" / "proposal_inputs").exists()
