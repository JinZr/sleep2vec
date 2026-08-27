from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from agent_tool_test_helpers import write_yaml
from agent_tools import adaptive_hparam, plan_hparam, slurm
from tests.agent_tools import adaptive_hparam_test_support as test_support
from tests.agent_tools.adaptive_hparam_test_support import (
    _adaptive_recipe,
    _run,
)

_stub_execution_snapshot_preflight = test_support._stub_execution_snapshot_preflight


def test_adaptive_accepted_starts_are_backend_aware():
    rows = [
        {"step_id": "direct", "run_id": "run-000", "status": "launched"},
        {
            "step_id": "slurm",
            "run_id": "run-001",
            "status": "queued",
            "scheduler_type": "slurm",
            "scheduler_job_id": "3880",
        },
        {
            "step_id": "slurm",
            "run_id": "run-002",
            "status": "submitting",
            "scheduler_type": "slurm",
        },
        {
            "step_id": "slurm",
            "run_id": "run-003",
            "status": "unknown_scheduler",
            "scheduler_type": "slurm",
            "scheduler_job_id": "3881",
        },
    ]

    assert adaptive_hparam._accepted_start_keys(rows) == {("direct", "run-000"), ("slurm", "run-001")}


@pytest.mark.parametrize(("observed_status", "expected_unresolved"), [("queued", False), ("submitting", True)])
@pytest.mark.parametrize("direct_controller", [False, True])
def test_adaptive_interrupted_slurm_launch_reconciles_by_scheduler_identity(
    tmp_path: Path,
    monkeypatch,
    observed_status: str,
    expected_unresolved: bool,
    direct_controller: bool,
):
    row = {
        "step_id": "train-model",
        "run_id": "run-000",
        "status": "submitting",
        "scheduler_type": "slurm",
        "scheduler_direct_controller": str(direct_controller).lower(),
        "scheduler_submit_token": "unit-token",
        "target": "local",
    }
    observed = {
        **row,
        "status": observed_status,
        **({"scheduler_job_id": "3880"} if observed_status == "queued" else {}),
    }
    merged = []
    observed_executions = []
    monkeypatch.setattr(
        adaptive_hparam.artifacts,
        "read_hparam_plan",
        lambda _plan_dir: {
            "recipe": {
                "execution": {
                    "target": "local",
                    "scheduler": {"direct_controller": not direct_controller},
                }
            }
        },
    )
    monkeypatch.setattr(adaptive_hparam, "read_run_manifest", lambda _workspace: [row])
    monkeypatch.setattr(
        adaptive_hparam.managed_scheduler,
        "observe_slurm_run",
        lambda owner_dir, execution, current: observed_executions.append(execution) or observed,
    )
    monkeypatch.setattr(
        adaptive_hparam,
        "merge_run_manifest",
        lambda _workspace, updates: merged.extend(updates) or updates,
    )
    monkeypatch.setattr(
        adaptive_hparam.evidence,
        "read_process_identity",
        lambda *_args, **_kwargs: pytest.fail("Slurm recovery must not read PID identity"),
    )

    rows, unresolved, reconciled = adaptive_hparam._reconcile_interrupted_launch(
        tmp_path, tmp_path / "round", {("train-model", "run-000")}
    )

    assert unresolved == ({("train-model", "run-000")} if expected_unresolved else set())
    assert reconciled == (set() if expected_unresolved else {("train-model", "run-000")})
    assert merged == ([] if expected_unresolved else [observed])
    assert rows == (merged if merged else [row])
    execution = {"target": "local"}
    if direct_controller:
        execution["scheduler"] = {"direct_controller": True}
    assert observed_executions == [execution]


def test_adaptive_slurm_grace_uses_allocation_start_not_submission_time():
    old = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    recent = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    replacement = {"grace_minutes": 1}

    assert (
        adaptive_hparam._grace_satisfied(
            {
                "scheduler_type": "slurm",
                "launched_at": old,
                "scheduler_started_at": recent,
            },
            {},
            replacement,
        )
        is False
    )
    assert adaptive_hparam._grace_satisfied(
        {"scheduler_type": "slurm", "launched_at": recent, "scheduler_started_at": old}, {}, replacement
    )
    assert adaptive_hparam._grace_satisfied({"scheduler_type": "slurm", "launched_at": old}, {}, replacement) is False


def test_adaptive_retirement_skips_slurm_run_with_verified_terminal_sidecar(tmp_path: Path, monkeypatch):
    recipe = {
        "adaptive": {
            "objective_metric": "val_score",
            "objective_mode": "max",
            "replacement": {"enabled": True, "allow_running_stop": True},
        }
    }
    run = {
        "step_id": "train-model",
        "run_id": "run-000",
        "status": "running",
        "scheduler_type": "slurm",
        "scheduler_exit_code": "0",
    }
    monkeypatch.setattr(
        adaptive_hparam.artifacts,
        "read_hparam_plan",
        lambda _round_dir: {"recipe": {"experiment": {"root": str(tmp_path)}}, "runs": [run]},
    )
    monkeypatch.setattr(adaptive_hparam, "read_run_manifest", lambda _workspace: [run])
    monkeypatch.setattr(adaptive_hparam, "_latest_incumbent_score", lambda _root: 1.0)
    monkeypatch.setattr(
        adaptive_hparam.evidence,
        "log_has_failure",
        lambda *_args, **_kwargs: pytest.fail("terminal Slurm work must not be considered for retirement"),
    )

    assert adaptive_hparam._bad_running_run_keys(tmp_path, tmp_path / "round", recipe) == set()


def test_adaptive_minutes_since_accepts_slurm_sidecar_timestamp():
    minutes = adaptive_hparam._minutes_since(slurm._utc_now())

    assert minutes is not None
    assert 0 <= minutes < 1



def test_adaptive_recipe_requires_explicit_test_feedback_flag(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path, test_feedback=False)

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 1
    assert "adaptive.test_feedback_for_selection" in result.stdout


def test_adaptive_default_test_objective_requires_test_after_fit(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"].pop("objective_metric")
    payload["evaluation_policy"]["test_after_fit"] = False
    payload["decisions"]["test_after_fit"] = {"value": False, "source": "explicit_recipe"}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 2
    assert "test-metric adaptive objective requires test_after_fit=true" in result.stdout


def test_adaptive_runtime_requires_literal_true_enabled_flag():
    with pytest.raises(ValueError, match="adaptive.enabled must be true"):
        adaptive_hparam._validate_adaptive_recipe({"adaptive": {"enabled": "true"}})


@pytest.mark.parametrize(
    ("enabled", "allow_running_stop"),
    [("true", True), (True, "true"), ("false", "false")],
)
def test_adaptive_runtime_never_stops_runs_for_non_boolean_replacement_flags(
    tmp_path: Path, enabled, allow_running_stop
):
    recipe = {"adaptive": {"replacement": {"enabled": enabled, "allow_running_stop": allow_running_stop}}}

    assert adaptive_hparam._bad_running_run_keys(tmp_path, tmp_path / "missing-round", recipe) == set()


def test_adaptive_rejects_removed_run_budget_and_gpu_fields(tmp_path: Path):
    recipe = _adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_trials"] = payload["search"].pop("max_runs")
    payload["adaptive"]["max_trials_total"] = payload["adaptive"].pop("max_runs_total")
    payload["execution"]["gpus_per_trial"] = 1
    recipe.write_text(yaml.safe_dump(payload))

    result = _run("doctor", "--recipe", str(recipe))

    assert result.returncode == 1
    assert "search.max_trials is no longer supported" in result.stdout
    assert "adaptive.max_trials_total is no longer supported" in result.stdout
    assert "execution.gpus_per_trial is no longer supported" in result.stdout



def test_hparam_count_does_not_materialize_search_values():
    recipe = {
        "search": {
            "parameters": {
                "runtime.lr": range(1000),
                "runtime.batch_size": range(1000),
            }
        }
    }

    assert adaptive_hparam._hparam_count(recipe) == 1_000_000


def test_hparam_count_uses_configuration_point_count():
    recipe = {
        "search": {
            "configurations": [
                {"runtime.lr": 1e-6, "runtime.batch_size": 8},
                {"runtime.lr": 2e-6, "runtime.batch_size": 16},
            ],
        }
    }

    assert adaptive_hparam._hparam_count(recipe) == 2


def test_hparam_combos_expands_configuration_points_exactly_and_truncates_by_max_runs():
    points = [
        {"runtime.lr": 1e-6, "yaml:/model/head/name": "classification"},
        {"runtime.lr": 2e-6, "yaml:/model/head/name": "regression"},
    ]

    combos = plan_hparam.hparam_combos({"search": {"configurations": points, "max_runs": 2}})
    assert combos == points
    assert combos[0] is not points[0]  # defensive copy

    truncated = plan_hparam.hparam_combos({"search": {"configurations": points, "max_runs": 1}})
    assert truncated == points[:1]


def test_has_yaml_search_overrides_sees_configuration_point_keys():
    assert plan_hparam.has_yaml_search_overrides({"search": {"configurations": [{"yaml:/model/dim": 128}]}})
    assert not plan_hparam.has_yaml_search_overrides({"search": {"configurations": [{"runtime.lr": 1e-6}]}})


def test_has_yaml_search_overrides_ignores_points_truncated_by_max_runs():
    # A yaml:/ key that only appears in a point beyond max_runs never executes
    # and must not force final_eval_config_path requirements.
    points = [{"runtime.lr": 1e-6}, {"yaml:/model/dim": 128}]

    assert not plan_hparam.has_yaml_search_overrides({"search": {"max_runs": 1, "configurations": points}})
    assert plan_hparam.has_yaml_search_overrides({"search": {"max_runs": 2, "configurations": points}})


@pytest.mark.parametrize("strategy", ["best_neighborhood", "agent_proposal"])
def test_adaptive_source_recipe_rejects_search_configurations(tmp_path: Path, strategy: str):
    recipe_path = _adaptive_recipe(tmp_path)
    payload = yaml.safe_load(recipe_path.read_text())
    payload["search"] = {
        "method": "grid",
        "max_runs": 1,
        "configurations": [{"runtime.lr": 1e-6}],
    }
    payload["adaptive"]["suggest"] = {"strategy": strategy}
    if strategy == "agent_proposal":
        payload["adaptive"].pop("replacement", None)
    write_yaml(recipe_path, payload)

    with pytest.raises(ValueError, match="search.parameters, not search.configurations"):
        adaptive_hparam.init_adaptive_workflow(recipe_path, tmp_path / "workflow")
