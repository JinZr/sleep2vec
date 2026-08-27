from __future__ import annotations

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
    monkeypatch.setattr(decision_hparam, "_hparam_search_issues", lambda *_args, **_kwargs: [_sentinel("search")])
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
