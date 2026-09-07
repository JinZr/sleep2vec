from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from agent_tool_test_helpers import config_payload, write_finetune_recipe
import pytest
import yaml

from agent_tools import plan_contract, plan_hparam, plans
from agent_tools.adapters.base import TaskAdapter
from agent_tools.adapters.hparam_tune import HPARAM_TUNE_ADAPTER
from agent_tools.experiment_workspace import read_run_manifest


def test_single_run_deferred_plan_preserves_frozen_bytes_and_registration_boundary(tmp_path: Path, monkeypatch):
    recipe_path = write_finetune_recipe(tmp_path / "workspace")
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = Path(recipe["experiment"]["root"])
    source_config = Path(recipe["inputs"]["config"])
    source_config_bytes = source_config.read_bytes()
    source_config_sha256 = hashlib.sha256(source_config_bytes).hexdigest()
    plan_dir = workspace / "plans" / "finetune"
    staging_dir = workspace / "plans" / ".finetune.staging"
    registered_recipe_path = plan_dir / "recipe.resolved.yaml"
    monkeypatch.setattr(
        plans,
        "plan_registration_lock",
        lambda *_args, **_kwargs: pytest.fail("deferred plans must not acquire the registration lock"),
    )

    report = plans.build_plan(
        recipe_path=recipe_path,
        output_dir=plan_dir,
        source_config_sha256=source_config_sha256,
        staging_dir=staging_dir,
        defer_commit=True,
        registered_recipe_path=registered_recipe_path,
        plan_controller="pipeline",
        run_index_offset=7,
    )

    assert report.exit_code == 0
    assert not plan_dir.exists()
    assert staging_dir.is_dir()
    plan_bytes = (staging_dir / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    run = plan["runs"][0]
    physical_run_dir = staging_dir / Path(run["script"]).parent.relative_to(plan_dir)
    assert plan_bytes == (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
    assert plan["recipe"]["_recipe_path"] == str(registered_recipe_path.resolve())
    assert run["run_id"] == "run-007"
    assert (physical_run_dir / "config.yaml").read_bytes() == source_config_bytes
    assert (staging_dir / "run.sh").read_bytes() == (physical_run_dir / "launch.sh").read_bytes()
    assert (staging_dir / "run.sh").stat().st_mode & 0o111 == 0o111
    assert (physical_run_dir / "launch.sh").stat().st_mode & 0o111 == 0o111
    assert read_run_manifest(workspace) == []
    assert not (workspace / "steps" / recipe["step"]["id"] / "step.yaml").exists()


def test_bound_config_hash_failure_precedes_staging_and_materialization(tmp_path: Path, monkeypatch):
    recipe_path = write_finetune_recipe(tmp_path / "workspace")
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = Path(recipe["experiment"]["root"])
    plan_dir = workspace / "plans" / "finetune"
    staging_dir = workspace / "plans" / ".finetune.staging"

    monkeypatch.setattr(
        plans,
        "_materialize_single_run_plan",
        lambda **_kwargs: pytest.fail("materialization must not start before bound config validation"),
    )

    with pytest.raises(ValueError, match="externally bound SHA-256"):
        plans.build_plan(
            recipe_path=recipe_path,
            output_dir=plan_dir,
            source_config_sha256="0" * 64,
            staging_dir=staging_dir,
            defer_commit=True,
        )

    assert not plan_dir.exists()
    assert not staging_dir.exists()
    assert read_run_manifest(workspace) == []


class _PrecommitFailureAdapter(TaskAdapter):
    task = "precommit_failure"
    materializes_plan = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def write_plan(
        self,
        recipe,
        out,
        *,
        write_out=None,
        run_index_offset=None,
        unlock_final_test,
        source_config_bytes,
        source_config_sha256,
    ) -> None:
        self.calls.append("write")
        write_out.mkdir(parents=True)
        (write_out / "plan.json").write_text("{}\n")

    def precommit_plan(self, out, *, write_out):
        self.calls.append("precommit")
        raise ValueError("injected precommit failure")

    def commit_plan(self, out, *, preflight_validated=False) -> None:
        pytest.fail("failed precommit must not register the plan")


def test_adapter_precommit_failure_removes_staging_without_publication(tmp_path: Path):
    adapter = _PrecommitFailureAdapter()
    out = tmp_path / "plan"
    staging = tmp_path / ".plan.staging"
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS)

    result = plans._materialize_adapter_plan(
        plan_adapter=adapter,
        recipe={},
        report=report,
        out=out,
        write_out=staging,
        output_identity=None,
        generated_staging=False,
        staging_dir=staging,
        defer_commit=False,
        plan_controller=None,
        run_index_offset=None,
        validate_only=False,
        unlock_final_test=False,
        validated_config_bytes=b"model: {}\n",
        validated_config_sha256=hashlib.sha256(b"model: {}\n").hexdigest(),
    )

    assert result.exit_code == 1
    assert adapter.calls == ["write", "precommit"]
    assert result.blocking_issues()[0].field == "execution.preflight"
    assert result.blocking_issues()[0].message == "injected precommit failure"
    assert not staging.exists()
    assert not out.exists()


def test_frozen_readers_copy_snapshots_but_preserve_raw_final_config_mapping(tmp_path: Path):
    context = {"home": str(tmp_path), "python": "/python", "repo_root": str(tmp_path)}
    snapshot = {"field": "inputs.config", "path": "relative/../config.yaml", "sha256": "a" * 64}
    raw_final = {"source_path": None, "bytes": b"config", "custom": {"value": [None, 1]}}
    recipe = {"_plan_context": context, "input_snapshots": [snapshot], "_final_eval_config_snapshot": raw_final}
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
    assert plan_hparam.final_eval_config_snapshot(recipe) is raw_final
    assert raw_final["custom"] is recipe["_final_eval_config_snapshot"]["custom"]


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
    adapter = TaskAdapter() if phase == "generic" else HPARAM_TUNE_ADAPTER

    compiled = adapter.compile_plan_contract(
        recipe, out, run_index_offset=7, config_bytes=b"" if phase == "hparam_rows" else config_bytes
    )

    assert recipe == before
    assert not out.exists()
    if phase == "generic":
        assert set(compiled) == {"runs", "commands", "script_text"}
        assert compiled["commands"] == []
        assert isinstance(compiled["script_text"], str)
        assert compiled["runs"][0]["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    else:
        assert set(compiled) == {
            "runs",
            "run_files",
            "launch_script_text",
            "final_command",
            "final_script_text",
            "final_eval_config_required",
            "final_eval_config_sha256",
        }
        assert compiled["final_command"] is None
        assert compiled["final_script_text"] is None
        assert compiled["final_eval_config_required"] is False
        assert compiled["final_eval_config_sha256"] is None
        assert [row["run_id"] for row in compiled["runs"]] == ["run-007", "run-008"]
        for row, run_file in zip(compiled["runs"], compiled["run_files"]):
            assert row is run_file["row"]
            if phase == "hparam_rows":
                assert set(run_file) == {"row"}
                assert "config_sha256" not in row and "script_sha256" not in row
            else:
                assert set(run_file) == {"row", "config_bytes", "script_text"}
                assert run_file["config_bytes"] == config_bytes
                assert row["config_sha256"] == hashlib.sha256(run_file["config_bytes"]).hexdigest()
                assert row["script_sha256"] == hashlib.sha256(run_file["script_text"].encode()).hexdigest()
