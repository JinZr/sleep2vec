from __future__ import annotations

import pytest

from agent_tools import decision_hparam
from agent_tools.decision_models import DecisionIssue, DecisionStatus


def _sentinel(field: str) -> DecisionIssue:
    return DecisionIssue(DecisionStatus.FAIL, field, field)


def test_hparam_tune_facade_preserves_collector_order(monkeypatch):
    monkeypatch.setattr(
        decision_hparam,
        "hparam_recipe_contract_issues",
        lambda *_args, **_kwargs: [_sentinel("contract")],
    )
    monkeypatch.setattr(decision_hparam, "_hparam_config_issues", lambda *_args, **_kwargs: [_sentinel("config")])
    monkeypatch.setattr(decision_hparam, "hparam_search_issues", lambda *_args, **_kwargs: [_sentinel("search")])
    monkeypatch.setattr(
        decision_hparam,
        "_hparam_execution_issues",
        lambda *_args, **_kwargs: [_sentinel("execution")],
    )
    monkeypatch.setattr(decision_hparam, "_hparam_adaptive_issues", lambda *_args, **_kwargs: [_sentinel("adaptive")])
    monkeypatch.setattr(
        decision_hparam,
        "_hparam_search_budget_issues",
        lambda *_args, **_kwargs: [_sentinel("budget")],
    )
    monkeypatch.setattr(
        decision_hparam,
        "_hparam_evaluation_issues",
        lambda *_args, **_kwargs: [_sentinel("evaluation")],
    )

    issues = decision_hparam.hparam_tune_issues(
        {"search": {}, "execution": {}, "runtime": {}, "adaptive": {}},
        None,
        {},
        {},
    )

    assert [issue.field for issue in issues] == [
        "contract",
        "config",
        "search",
        "execution",
        "adaptive",
        "budget",
        "evaluation",
    ]


def test_hparam_execution_facade_preserves_scheduler_runtime_issue_order():
    execution = {
        "scheduler": {"type": "direct", "unexpected": True},
        "gpus_per_trial": 1,
        "log_dir": "logs",
        "target": "ssh",
        "workdir": "relative",
        "python": "python --flag",
        "runtime_commit": "short",
        "path_context": "invalid",
        "path_validation": "invalid",
        "max_concurrent": 0,
        "gpu_pool": "0,1",
        "gpus_per_run": 0,
        "env": {"BAD-NAME": [], "PYTHONPATH": "src"},
    }

    issues = decision_hparam._hparam_execution_issues(execution, {})

    assert [issue.field for issue in issues] == [
        "execution.scheduler",
        "execution.gpus_per_trial",
        "execution.log_dir",
        "execution.host",
        "execution.workdir",
        "execution.python",
        "execution.runtime_commit",
        "execution.path_context",
        "execution.path_validation",
        "execution.max_concurrent",
        "execution.gpu_pool",
        "execution.gpus_per_run",
        "execution.env.BAD-NAME",
        "execution.env.BAD-NAME",
        "execution.env.PYTHONPATH",
    ]
    assert [issue.status for issue in issues] == [DecisionStatus.FAIL] * len(issues)
    assert issues[0].evidence == {
        "scheduler": {"type": "direct", "unexpected": True},
        "preflight_before_workspace": True,
    }
    assert issues[-1].message == "PYTHONPATH is not supported in execution.env; use execution.workdir."


_POSITIVE_INTEGER = "must be a positive integer"
_SINGLE_EXECUTABLE = "must be a single executable name or path without whitespace, arguments, or ~ shorthand"
_FULL_COMMIT = "must be a full 40-character Git commit ID"


@pytest.mark.parametrize(
    ("execution", "field", "message"),
    [
        *[
            ({"gpu_pool": [0, 1], "gpus_per_run": gpus_per_run}, "execution.gpus_per_run", _POSITIVE_INTEGER)
            for gpus_per_run in [0, False, 0.5, 1.0, 1.5, "0.5", "1", "1.5"]
        ],
        *[
            ({"max_concurrent": max_concurrent}, "execution.max_concurrent", _POSITIVE_INTEGER)
            for max_concurrent in [True, 1.0, 1.5, "1", 0]
        ],
        ({"python": ""}, "execution.python", _SINGLE_EXECUTABLE),
        ({"python": "conda run -n exp python"}, "execution.python", _SINGLE_EXECUTABLE),
        ({"python": "~/miniconda/bin/python"}, "execution.python", _SINGLE_EXECUTABLE),
        ({"runtime_commit": "abc123"}, "execution.runtime_commit", _FULL_COMMIT),
    ],
)
def test_hparam_execution_reports_one_invalid_field(execution, field, message):
    issues = decision_hparam._hparam_execution_issues(execution, {})

    assert len(issues) == 1
    assert issues[0].field == field
    assert issues[0].status.value == "FAIL"
    assert message in issues[0].message


def test_hparam_execution_warns_when_slurm_request_voluntarily_lowers_priority():
    issues = decision_hparam._hparam_execution_issues(
        {
            "scheduler": {
                "type": "slurm",
                "partition": "gpu",
                "cpus_per_task": 8,
                "memory": "64G",
                "walltime": "01:00:00",
                "nice": 100,
                "nodelist": "h20-bj-96",
            }
        },
        {},
    )

    priority_issue = next(issue for issue in issues if issue.field == "execution.scheduler.priority")
    assert priority_issue.status.value == "WARN"
    assert "nice=100 voluntarily lowers priority" in priority_issue.message
    assert "nodelist narrows eligible nodes" in priority_issue.message


def test_direct_hparam_allows_slurm_named_environment_variable():
    issues = decision_hparam._hparam_execution_issues(
        {"scheduler": {"type": "direct"}, "env": {"SLURM_JOB_ID": "outer-allocation"}},
        {},
    )

    assert not [issue for issue in issues if issue.field == "execution.env.SLURM_JOB_ID"]


_SLURM_SCHEDULER = {
    "type": "slurm",
    "partition": "gpu",
    "cpus_per_task": 8,
    "memory": "64G",
    "walltime": "01:00:00",
}


@pytest.mark.parametrize("gpus_per_run", [1, 2, 4])
@pytest.mark.parametrize("scheduler", ["slurm", "direct"])
def test_sex_age_baseline_accepts_slurm_and_direct_multi_gpu(scheduler, gpus_per_run):
    execution = (
        {"gpus_per_run": gpus_per_run, "scheduler": _SLURM_SCHEDULER}
        if scheduler == "slurm"
        else {"gpu_pool": list(range(gpus_per_run)), "gpus_per_run": gpus_per_run}
    )

    issues = decision_hparam._hparam_execution_issues(execution, {}, variant="sex_age_baseline")

    assert not [issue for issue in issues if issue.status.value == "FAIL"]
