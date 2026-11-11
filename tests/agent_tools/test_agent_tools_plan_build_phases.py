from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from agent_tool_test_helpers import config_payload
import pytest
import yaml

from agent_tools import plan_contract, plan_hparam, plans, run_artifacts
from agent_tools.adapters.base import TaskAdapter


def test_hparam_precommit_failure_removes_staging_without_publication(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    out = tmp_path / "plan"
    staging = tmp_path / ".plan.staging"
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS)

    def write_plan(recipe, plan_out, *, write_out, **_kwargs) -> None:
        assert (plan_out, write_out) == (out, staging)
        calls.append("write")
        write_out.mkdir(parents=True)
        (write_out / "plan.json").write_text("{}\n")

    def fail_preflight(physical_out, *, semantic_out):
        assert (physical_out, semantic_out) == (staging, out)
        calls.append("precommit")
        raise ValueError("injected precommit failure")

    monkeypatch.setattr(plan_hparam, "write_hparam_plan", write_plan)
    monkeypatch.setattr(run_artifacts, "preflight_hparam_plan", fail_preflight)
    monkeypatch.setattr(
        run_artifacts,
        "commit_hparam_plan",
        lambda *_args, **_kwargs: pytest.fail("failed precommit must not register the plan"),
    )

    result = plans._materialize_hparam_plan(
        recipe={},
        report=report,
        out=out,
        write_out=staging,
        generated_staging=True,
        run_index_offset=None,
        validate_only=False,
        unlock_final_test=False,
        validated_config_bytes=b"model: {}\n",
        validated_config_sha256=hashlib.sha256(b"model: {}\n").hexdigest(),
        final_eval_config=None,
    )

    assert result.exit_code == 1
    assert calls == ["write", "precommit"]
    assert result.blocking_issues()[0].field == "execution.preflight"
    assert result.blocking_issues()[0].message == "injected precommit failure"
    assert not staging.exists()
    assert not out.exists()


def test_frozen_readers_copy_snapshots(tmp_path: Path):
    context = {"home": str(tmp_path), "python": "/python", "repo_root": str(tmp_path)}
    snapshot = {"field": "inputs.config", "path": "relative/../config.yaml", "sha256": "a" * 64}
    recipe = {"_plan_context": context, "input_snapshots": [snapshot]}
    before = copy.deepcopy(recipe)

    copied_context = plan_contract.frozen_plan_context(recipe)
    copied_snapshots = plan_contract.frozen_input_snapshots(recipe)
    copied_snapshot = plan_contract.frozen_input_snapshot(recipe, "inputs.config")

    assert copied_context == context and copied_context is not context
    assert copied_snapshots == [snapshot] and copied_snapshots is not recipe["input_snapshots"]
    assert copied_snapshots[0] is not snapshot
    assert copied_snapshot == snapshot and copied_snapshot is not snapshot
    copied_context["home"] = "/changed"
    copied_snapshots[0]["sha256"] = "b" * 64
    copied_snapshot["path"] = "/changed"
    assert recipe == before


@pytest.mark.parametrize("phase", ["generic", "hparam_rows", "hparam_materialized"])
def test_compiled_plan_phase_fields_and_row_identity(tmp_path: Path, phase: str):
    config_bytes = yaml.safe_dump(config_payload(tmp_path / "index.csv")).encode()
    recipe = {
        "name": "phase-contract",
        "variant": "sleep2vec",
        "inputs": {"config": str(tmp_path / "source.yaml"), "label_name": "ahi"},
        "experiment": {"id": "phase-contract", "root": str(tmp_path / "workspace")},
        "step": {"id": "tune"},
        "execution": {"python": "/python", "runtime_commit": "a" * 40, "workdir": str(tmp_path)},
        "evaluation_policy": {"test_after_fit": False, "selection_split": "val"},
        "search": {"configurations": [{"runtime.lr": 1e-6}, {"runtime.lr": 2e-6}]},
    }
    plan_contract.bind_plan_context(recipe)
    plan_contract.bind_frozen_input_snapshot(
        recipe, "inputs.config", recipe["inputs"]["config"], hashlib.sha256(config_bytes).hexdigest()
    )
    before = copy.deepcopy(recipe)
    out = tmp_path / "unwritten-plan"
    if phase == "generic":
        compiled = TaskAdapter().compile_plan_contract(recipe, out, run_index_offset=7, config_bytes=config_bytes)
    else:
        run_files = plan_hparam.compile_hparam_run_contracts(
            recipe, out, 7, source_config_bytes=config_bytes if phase == "hparam_materialized" else None
        )

    assert recipe == before
    assert not out.exists()
    if phase == "generic":
        assert set(compiled) == {"runs", "commands", "script_text"}
        assert compiled["commands"] == []
        assert isinstance(compiled["script_text"], str)
        assert compiled["runs"][0]["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    else:
        assert [run_file["row"]["run_id"] for run_file in run_files] == ["run-007", "run-008"]
        for run_file in run_files:
            row = run_file["row"]
            if phase == "hparam_rows":
                assert set(run_file) == {"row"}
                assert "config_sha256" not in row and "script_sha256" not in row
            else:
                assert set(run_file) == {"row", "config_bytes", "script_text"}
                assert run_file["config_bytes"] == config_bytes
                assert row["config_sha256"] == hashlib.sha256(run_file["config_bytes"]).hexdigest()
                assert row["script_sha256"] == hashlib.sha256(run_file["script_text"].encode()).hexdigest()
