from __future__ import annotations

import pytest

from agent_tools import decision_hparam
from agent_tools.decision_models import DecisionStatus


def test_hparam_execution_facade_preserves_scheduler_runtime_issue_order():
    execution = {
        "scheduler": {"type": "direct", "unexpected": True},
        "gpus_per_trial": 1,
        "log_dir": "logs",
        "workdir": "relative",
        "python": "python --flag",
        "runtime_commit": "short",
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
        "execution.workdir",
        "execution.python",
        "execution.runtime_commit",
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


def test_hparam_execution_keeps_slurm_priority_a_plain_directive():
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

    # nice and nodelist are frozen sbatch directives; Slurm, not the planner, owns priority.
    assert not [issue for issue in issues if issue.field.startswith("execution.scheduler")]


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
